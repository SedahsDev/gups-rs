/// UCX communication layer for the GUPS benchmark.
///
/// Uses RMA (Remote Memory Access) with atomic XOR operations for direct
/// remote table updates. Pattern derived from osss-ucx:
/// - Memory registration -> rkey packing -> PMIx Put/Commit/Fence/Get exchange
/// - Direct `atomic_xor64` on remote table addresses (no message passing per update)
///
/// Single-node mode: No UCX/PMIx needed; operates on local memory directly.
/// Multi-process mode: Full UCX stack with RMA atomics, PMIx for bootstrap.

use std::ffi::CString;

use ucx_sys::context;
use ucx_sys::ep;
use ucx_sys::memh;
use ucx_sys::rma::RemoteKey;
use ucx_sys::worker;
use ucx_sys::worker::RemoteWorkerAddress;
use ucx_sys::RequestParamBuilder;

use pmix::{commit, fence, get_value, init, put_value, Context, GLOBAL, PmixValueBuilder, RANK_WILDCARD};

/// Tags for inter-process control traffic (reduction, verification, etc.).
pub const TAG_SYNC: u64 = 0x3000;
pub const TAG_VERIFY: u64 = 0x4000;

// PMIx key names for data exchange (null-terminated C strings)
const PMIX_KEY_UCX_ADDR: &str = "gups.ucx.addr";
const PMIX_KEY_UCX_MEMH: &str = "gups.ucx.memh";
const PMIX_KEY_UCX_TABLE_ADDR: &str = "gups.ucx.table_addr";

/// Single-node communication (no UCX/PMIx needed).
#[allow(dead_code)]
pub struct UpdateComm {
    pub rank: usize,
    pub size: usize,
}

/// Create a single-node communication context.
pub fn create_single_node(rank: usize, size: usize) -> UpdateComm {
    UpdateComm { rank, size }
}

/// Barrier for single-node mode (no-op).
pub fn barrier_single(_comm: &UpdateComm) {}

/// Multi-process communication context using UCX RMA atomics + PMIx bootstrap.
pub struct CommCtx {
    pub rank: usize,
    pub size: usize,
    context: context::Context,
    pub worker: worker::Worker,
    pub endpoints: Vec<ep::Ep>,
    remote_rkeys: Vec<Option<RemoteKey>>,
    remote_table_addrs: Vec<u64>,
    _memh: memh::MemHandle,
    /// PMIx context — kept alive for the lifetime of this CommCtx.
    /// Drop calls PMIx_Finalize automatically.
    _pmix_ctx: Context,
}

/// Create a multi-process communication context using UCX + PMIx.
///
/// Gets rank and size directly from PMIx (PMIx_Init + PMIX_JOB_SIZE query).
/// No env vars needed — just call this function and it returns (rank, size, CommCtx).
///
/// This function:
/// 1. Initializes PMIx for rank/namespace discovery
/// 2. Queries PMIX_JOB_SIZE for process count
/// 3. Initializes UCX context with Tag + RMA + AMO64 features
/// 4. Packs worker address, registers memory, packs rkey
/// 5. Publishes address/rkey/table_addr via PMIx_Put + PMIx_Commit + PMIx_Fence
/// 6. Retrieves peer data via PMIx_Get
/// 7. Creates UCX endpoints and unpacks remote rkeys
/// 8. Returns a CommCtx ready for atomic XOR operations
pub fn create_multiprocess(table_base: *mut u64, table_bytes: usize) -> (usize, usize, CommCtx) {
    // 1. Initialize PMIx — gets our rank
    let pmix_ctx = init(None).expect("PMIx init");
    let rank = pmix_ctx.get_rank() as usize;
    let my_proc = pmix_ctx.get_proc();

     // 2. Query PMIX_JOB_SIZE via wildcard proc for total process count
    // Fall back to PMIX_SIZE env var if PMIx doesn't publish it
    let wc_proc = pmix_ctx
        .proc_with_nspace(RANK_WILDCARD)
        .expect("wildcard_proc");
    let job_size_key_bytes = "PMIX_JOB_SIZE\0";
    let size = get_value(&wc_proc, job_size_key_bytes.as_bytes(), None)
        .ok()
        .map(|v| v.uint32() as usize)
        .or_else(|| std::env::var("PMIX_SIZE").ok().and_then(|s| s.parse().ok()))
        .unwrap_or_else(|| {
            panic!("Cannot determine job size from PMIx or PMIX_SIZE env var");
        });
    eprintln!("[gups-rs] PMIx rank={}, size={}", rank, size);

    // 3. Initialize UCX context
    let features = context::Flags::Tag | context::Flags::Rma | context::Flags::Amo64 | context::Flags::ExportedMemH;
    let ctx_params = context::ParamsBuilder::new()
        .features(features)
        .estimated_num_eps(size - 1)
        .estimated_num_ppn(2)
        .build();
    let config = context::Config::default();
    let uctx = context::Context::new(&config, &ctx_params).expect("UCX context init");
    drop(config);

    // 4. Create worker
    let wparams = worker::ParamsBuilder::new().build();
    let worker = uctx.worker_create(&wparams).expect("UCX worker create");

    // 5. Pack own worker address
    let packed_addr = worker.pack_address().expect("Worker address pack");
    let own_addr_bytes = packed_addr.to_vec();

    // 6. Register table memory with UCX
    let mut mem_params = memh::MemMapParamsBuilder::new();
    mem_params
        .address(table_base as *mut std::os::raw::c_void)
        .length(table_bytes);
    let memh = memh::MemHandle::map(&uctx, &mut mem_params).expect("Memory registration");

    // 7. Pack rkey for our memory using legacy ucp_rkey_pack
    // (works on all memory domains including self/sysv/posix)
    let packed_rkey = memh::pack_rkey(&uctx, &memh).expect("Rkey pack");
    // ucp_rkey_pack returns: [4 bytes LE length][rkey data]
    // For ep_rkey_unpack, pass the raw buffer pointer directly
    let rkey_data = packed_rkey.as_bytes();

    // 8. Get our table address
    let table_addr = memh.query().expect("Memh query").address() as u64;

    // 9. Publish our data via PMIx_Put
    // Worker address as byte object
    let addr_key = CString::new(PMIX_KEY_UCX_ADDR).unwrap();
    let mut addr_val = PmixValueBuilder::new()
        .byte_object(&own_addr_bytes)
        .expect("byte_object addr")
        .build()
        .expect("build addr");
    put_value(GLOBAL, &addr_key, &mut addr_val).expect("PMIx_Put addr");

    // Packed rkey (memory handle) as byte object
    let memh_key = CString::new(PMIX_KEY_UCX_MEMH).unwrap();
    let mut memh_val = PmixValueBuilder::new()
        .byte_object(rkey_data)
        .expect("byte_object memh")
        .build()
        .expect("build memh");
    put_value(GLOBAL, &memh_key, &mut memh_val).expect("PMIx_Put memh");

    // Table base address as uint64
    let table_key = CString::new(PMIX_KEY_UCX_TABLE_ADDR).unwrap();
    let mut table_val = PmixValueBuilder::new()
        .uint64(table_addr)
        .build()
        .expect("build table_addr");
    put_value(GLOBAL, &table_key, &mut table_val).expect("PMIx_Put table_addr");

    // 10. Commit + Fence (barrier + data exchange)
    commit().expect("PMIx_Commit");
    fence(my_proc, None).expect("PMIx_Fence");

    // 11. Retrieve peer data via PMIx_Get
    let mut peer_addrs: Vec<Vec<u8>> = vec![Vec::new(); size];
    peer_addrs[rank] = own_addr_bytes.clone();
    let mut peer_memh_data: Vec<Vec<u8>> = vec![Vec::new(); size];
    let mut remote_table_addrs = vec![0u64; size];
    remote_table_addrs[rank] = table_addr;

    for peer in 0..size {
        if peer == rank {
            continue;
        }

        let remote_proc = pmix_ctx.proc_with_nspace(peer as u32).expect("proc_with_nspace");

        // Get worker address
        let addr_key_bytes = format!("{}{}", PMIX_KEY_UCX_ADDR, '\0');
        let addr_val = get_value(&remote_proc, addr_key_bytes.as_bytes(), None)
            .expect("PMIx_Get addr");
        peer_addrs[peer] = addr_val.bytes_copy();

        // Get packed rkey (memory handle)
        let memh_key_bytes = format!("{}{}", PMIX_KEY_UCX_MEMH, '\0');
        let memh_val = get_value(&remote_proc, memh_key_bytes.as_bytes(), None)
            .expect("PMIx_Get memh");
        peer_memh_data[peer] = memh_val.bytes_copy();

        // Get table address
        let table_key_bytes = format!("{}{}", PMIX_KEY_UCX_TABLE_ADDR, '\0');
        let table_val = get_value(&remote_proc, table_key_bytes.as_bytes(), None)
            .expect("PMIx_Get table_addr");
        remote_table_addrs[peer] = table_val.uint64();
    }

    drop(packed_addr);

    // 12. Create UCX endpoints to each peer
    let mut endpoints = Vec::with_capacity(size);
    for peer in 0..size {
        if peer == rank {
            // Self endpoint: create one to ourselves
            let own_remote_addr = RemoteWorkerAddress::new(own_addr_bytes.clone());
            let ep_params = ep::ParamsBuilder::new().address(&own_remote_addr).build();
            let ep = worker.create_ep(&ep_params).expect("Self EP create");
            endpoints.push(ep);
            continue;
        }
        let remote_addr = RemoteWorkerAddress::new(peer_addrs[peer].clone());
        let ep_params = ep::ParamsBuilder::new().address(&remote_addr).build();
        let ep = worker.create_ep(&ep_params).expect("EP create for peer");
        endpoints.push(ep);
    }

    // Progress endpoint connections
    loop {
        if !worker.progress() {
            break;
        }
    }

    // 13. Unpack remote rkeys from cached memh data
    let mut remote_rkeys: Vec<Option<RemoteKey>> = (0..size).map(|_| None).collect();
    for peer in 0..size {
        if peer == rank {
            continue;
        }
        let rkey = RemoteKey::unpack(&endpoints[peer], &peer_memh_data[peer])
            .expect("rkey unpack");
        remote_rkeys[peer] = Some(rkey);
    }

    // 14. Flush all endpoints
    let flush_param = RequestParamBuilder::new().no_imm_cmpl().build();
    for peer in 0..size {
        flush_ep_blocking(&worker, &endpoints[peer], &flush_param);
    }

    let ctx = CommCtx {
        rank,
        size,
        context: uctx,
        worker,
        endpoints,
        remote_rkeys,
        remote_table_addrs,
        _memh: memh,
        _pmix_ctx: pmix_ctx,
    };
    (rank, size, ctx)
}

/// Perform an atomic XOR on a remote peer's table entry.
pub fn atomic_xor_remote(
    comm: &CommCtx,
    peer: usize,
    offset: usize,
    value: u64,
) {
    let remote_addr =
        comm.remote_table_addrs[peer] + (offset * std::mem::size_of::<u64>()) as u64;
    let rkey = comm.remote_rkeys[peer].as_ref().expect("rkey for peer");

    let param = RequestParamBuilder::new().build();
    let result = comm.endpoints[peer].amo_xor64(value, remote_addr, rkey, &param);
    if let Err(e) = result {
        eprintln!(
            "atomic_xor64 failed on peer {} offset {}: {:?}",
            peer, offset, e
        );
    }
}

/// Progress the UCX worker.
pub fn progress(comm: &CommCtx) {
    loop {
        if !comm.worker.progress() {
            break;
        }
    }
}

/// Barrier across all processes using UCX tag messages.
pub fn barrier(comm: &CommCtx) {
    let rank = comm.rank;
    let size = comm.size;
    let worker = &comm.worker;
    let endpoints = &comm.endpoints;

    let param = RequestParamBuilder::new().no_imm_cmpl().build();
    let recv_param = RequestParamBuilder::new().no_imm_cmpl().build();
    let my_rank_bytes = (rank as u64).to_le_bytes();

    // Send sync to all peers
    for peer in 0..size {
        if peer == rank {
            continue;
        }
        send_blocking(&endpoints[peer], &my_rank_bytes, TAG_SYNC, worker, &param);
    }

    // Wait for sync from all peers
    for peer in 0..size {
        if peer == rank {
            continue;
        }
        let mut buf = [0u8; 8];
        recv_blocking(worker, &mut buf, TAG_SYNC, &recv_param);
    }
}

/// Send a u64 value via tag message.
pub fn send_u64(ep: &ep::Ep, value: u64, tag: u64) {
    let param = RequestParamBuilder::new().no_imm_cmpl().build();
    let bytes = value.to_le_bytes();
    let _ = ep.tag_send(&bytes, tag, &param);
}

/// Receive a u64 value via tag message.
pub fn recv_u64_value(worker: &worker::Worker, tag: u64) -> u64 {
    let param = RequestParamBuilder::new().no_imm_cmpl().build();
    let mut buf = [0u8; 8];
    recv_blocking(worker, &mut buf, tag, &param);
    u64::from_le_bytes(buf)
}

// ── Internal helpers ──

fn send_blocking(
    ep: &ep::Ep,
    data: &[u8],
    tag: u64,
    worker: &worker::Worker,
    param: &ucx_sys::RequestParam,
) {
    let req = ep.tag_send(data, tag, param);
    if let Ok(Some(r)) = req {
        while !r.check_finished().unwrap_or(false) {
            worker.progress();
        }
    }
}

fn recv_blocking(
    worker: &worker::Worker,
    buf: &mut [u8],
    tag: u64,
    param: &ucx_sys::RequestParam,
) {
    let mut req = worker.tag_recv(buf, tag, u64::MAX, param).expect("tag_recv");
    if let Some(r) = req.take() {
        while !r.check_finished().unwrap_or(false) {
            worker.progress();
        }
    }
}

fn flush_ep_blocking(
    worker: &worker::Worker,
    ep: &ep::Ep,
    param: &ucx_sys::RequestParam,
) {
    // Flush via the endpoint by flushing the worker — UCX flush is worker-wide
    // but we call it once per endpoint to be safe in the original pattern.
    // Actually, worker.flush() flushes all AM/RMA on this worker.
    let req = worker.flush(param);
    if let Ok(Some(r)) = req {
        while !r.check_finished().unwrap_or(false) {
            worker.progress();
        }
    }
    let _ = ep; // ep passed for API compatibility
}
