/// UCX communication layer for the GUPS benchmark.
///
/// Uses RMA (Remote Memory Access) with atomic XOR operations for direct
/// remote table updates. Pattern derived from osss-ucx:
/// - Memory registration -> rkey packing -> PMIx Put/Commit/Fence/Get exchange
/// - Direct `atomic_xor64` on remote table addresses (no message passing per update)
///
/// Single-node mode: No UCX/PMIx needed; operates on local memory directly.
/// Multi-process mode: Full UCX stack with RMA atomics, PMIx for bootstrap.
///
/// UCC is used for collective operations (barrier, allreduce) replacing the
/// previous tag-message-based barrier and verification reduction.
use std::ffi::CString;

use ucx_sys::context;
use ucx_sys::ep;
use ucx_sys::memh;
use ucx_sys::rma::RemoteKey;
use ucx_sys::worker;
use ucx_sys::worker::RemoteWorkerAddress;
use ucx_sys::RequestParamBuilder;

use pmix::{
    commit, fence, get_value, info_with_string_key, init, put_value, Context, PmixValueBuilder,
    GLOBAL, RANK_WILDCARD,
};

use ucc::collective::{CollectiveBuilder, UccCollectiveType, UccReductionOp};
use ucc::context::UccContext;
use ucc::lib_init::UccLib;
use ucc::memory::UccMemHandle;
use ucc::team::{UccTeam, UccTeamParams};

/// Tags for inter-process control traffic (reduction, verification, etc.).
/// Kept for backward compatibility; barrier and allreduce now use UCC.
#[allow(dead_code)]
pub const TAG_SYNC: u64 = 0x3000;
#[allow(dead_code)]
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

/// Allreduce for single-node mode (returns the local value unchanged).
#[allow(dead_code)]
pub fn allreduce_u64_single(_comm: &UpdateComm, value: u64) -> u64 {
    value
}

/// Multi-process communication context using UCX RMA atomics + PMIx bootstrap.
/// UCC is used for collective operations (barrier, allreduce).
#[allow(dead_code)]
pub struct CommCtx {
    pub rank: usize,
    pub size: usize,
    /// UCC team for collective operations (barrier, allreduce, etc.).
    ucc_team: UccTeam,
    /// UCC context for collective operations.
    ucc_context: UccContext,
    /// UCC library handle for collective operations.
    ucc_lib: UccLib,
    remote_rkeys: Vec<Option<RemoteKey>>,
    remote_table_addrs: Vec<u64>,
    _memh: memh::MemHandle,
    pub endpoints: Vec<ep::Ep>,
    pub worker: worker::Worker,
    context: context::Context,
    /// PMIx context — kept alive for the lifetime of this CommCtx.
    /// Drop calls PMIx_Finalize automatically.
    _pmix_ctx: Context,
}

/// Resolve the PMIx server URI from environment or local file.
///
/// OpenPMIX 6.1.0 ignores `PMIX_SERVER_URI` set via `std::env::set_var`,
/// so we read the URI ourselves and pass it through `info_with_string_key`.
///
/// Only resolves when running under a PRTE daemon (detected via PMIX_RANK env var).
/// Standalone single-process runs fall back to bare `PMIx_Init`.
///
/// Lookup order (when under prterun):
/// 1. Versioned env vars: PMIX_SERVER_URI61, PMIX_SERVER_URI51, PMIX_SERVER_URI41, etc.
///    (prterun sets these — they point to the correct session daemon)
/// 2. Unversioned PMIX_SERVER_URI env var
/// 3. URI file at `/run/user/{uid}/prte/uri`
/// 4. `None` if none are available (bare `PMIx_Init` as before)
fn resolve_pmix_server_uri() -> Option<String> {
    // Only resolve URI when running under prterun (PMIX_RANK is set by the daemon)
    // Standalone runs should use init(None) to avoid connecting to stale URIs
    if std::env::var("PMIX_RANK").is_err() {
        return None;
    }

    // 1. Check versioned env vars (prterun sets PMIX_SERVER_URI61, PMIX_SERVER_URI51, etc.)
    // These point to the correct session daemon, not the system daemon
    for key in [
        "PMIX_SERVER_URI61",
        "PMIX_SERVER_URI51",
        "PMIX_SERVER_URI41",
        "PMIX_SERVER_URI4",
        "PMIX_SERVER_URI3",
        "PMIX_SERVER_URI21",
        "PMIX_SERVER_URI20",
        "PMIX_SERVER_URI12",
    ] {
        if let Ok(uri) = std::env::var(key) {
            if !uri.is_empty() {
                return Some(uri);
            }
        }
    }

    // 2. Check unversioned env var
    if let Ok(uri) = std::env::var("PMIX_SERVER_URI") {
        if !uri.is_empty() {
            return Some(uri);
        }
    }

    // 3. Read URI file from systemd runtime directory
    // Use getuid() not getpid() — the URI lives under /run/user/{uid}/
    let uid = unsafe { libc::getuid() };
    let uri_path = format!("/run/user/{}/prte/uri", uid);
    if let Ok(content) = std::fs::read_to_string(&uri_path) {
        let uri = content.lines().next()?.trim().to_string();
        if !uri.is_empty() {
            return Some(uri);
        }
    }
    None
}

/// Create a multi-process communication context using UCX + PMIx + UCC.
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
/// 8. Initializes UCC library, context, and team for collective operations
/// 9. Returns a CommCtx ready for atomic XOR operations and collectives
pub fn create_multiprocess(table_base: *mut u64, table_bytes: usize) -> (usize, usize, CommCtx) {
    // 1. Initialize PMIx — gets our rank
    // Pass server URI explicitly (OpenPMIX 6.1.0 ignores env vars for this)
    let pmix_info =
        resolve_pmix_server_uri().map(|uri| info_with_string_key("pmix.srvr.uri", &uri));
    let pmix_ctx = init(pmix_info).expect("PMIx init");
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
    let features = context::Flags::Tag
        | context::Flags::Rma
        | context::Flags::Amo64
        | context::Flags::ExportedMemH;
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
        let remote_proc = pmix_ctx
            .proc_with_nspace(peer as u32)
            .expect("proc_with_nspace");

        // Get worker address
        let addr_key_bytes = format!("{}{}", PMIX_KEY_UCX_ADDR, '\0');
        let addr_val =
            get_value(&remote_proc, addr_key_bytes.as_bytes(), None).expect("PMIx_Get addr");
        peer_addrs[peer] = addr_val.bytes_copy();

        // Get packed rkey (memory handle)
        let memh_key_bytes = format!("{}{}", PMIX_KEY_UCX_MEMH, '\0');
        let memh_val =
            get_value(&remote_proc, memh_key_bytes.as_bytes(), None).expect("PMIx_Get memh");
        peer_memh_data[peer] = memh_val.bytes_copy();

        // Get table address
        let table_key_bytes = format!("{}{}", PMIX_KEY_UCX_TABLE_ADDR, '\0');
        let table_val =
            get_value(&remote_proc, table_key_bytes.as_bytes(), None).expect("PMIx_Get table_addr");
        remote_table_addrs[peer] = table_val.uint64();
    }

    drop(packed_addr);

    // 12. Create UCX endpoints to each peer
    let mut endpoints = Vec::with_capacity(size);
    #[allow(clippy::needless_range_loop)]
    for peer in 0..size {
        if peer == rank {
            // Self endpoint: create one to ourselves
            let own_remote_addr = RemoteWorkerAddress::new(own_addr_bytes.clone());
            let ep_params = ep::ParamsBuilder::new().address(&own_remote_addr).build();
            let ep = worker.create_ep(ep_params).expect("Self EP create");
            endpoints.push(ep);
            continue;
        }
        let remote_addr = RemoteWorkerAddress::new(peer_addrs[peer].clone());
        let ep_params = ep::ParamsBuilder::new().address(&remote_addr).build();
        let ep = worker.create_ep(ep_params).expect("EP create for peer");
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
        let rkey = RemoteKey::unpack(&endpoints[peer], &peer_memh_data[peer]).expect("rkey unpack");
        remote_rkeys[peer] = Some(rkey);
    }

    // 14. Flush all endpoints
    let flush_param = RequestParamBuilder::new().no_imm_cmpl().build();
    for (_peer, ep) in endpoints.iter().enumerate().take(size) {
        flush_ep_blocking(&worker, ep, &flush_param);
    }

    // 15. Initialize UCC for collective operations (barrier, allreduce, etc.)
    let ucc_lib = UccLib::init().expect("UCC library init");
    let ucc_context = UccContext::new(ucc_lib.clone()).expect("UCC context create");

    // Create UCC team with explicit size
    let mut ucc_team_params = UccTeamParams::default();
    ucc_team_params.with_team_size(size as u64);
    let ucc_team =
        UccTeam::with_params(ucc_context.clone(), ucc_team_params).expect("UCC team create");

    eprintln!("[gups-rs] UCC team created (rank={}, size={})", rank, size);

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
        ucc_lib,
        ucc_context,
        ucc_team,
    };
    (rank, size, ctx)
}

/// Perform an atomic XOR on a remote peer's table entry.
pub fn atomic_xor_remote(comm: &CommCtx, peer: usize, offset: usize, value: u64) {
    let remote_addr = comm.remote_table_addrs[peer] + (offset * std::mem::size_of::<u64>()) as u64;
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

/// Barrier across all processes using UCC collective barrier.
///
/// Uses `init_and_post` which is synchronous — blocks until the collective
/// completes. Replaces the previous tag-message-based barrier.
pub fn barrier(comm: &CommCtx) {
    let stub = [0u8];
    let mem = UccMemHandle::map_slice(&comm.ucc_context, &stub).expect("UCC barrier src map");
    let mut req = CollectiveBuilder::new(UccCollectiveType::Barrier)
        .with_src(&mem)
        .with_dst(&mem)
        .with_count(1)
        .init_and_post(&comm.ucc_team)
        .expect("UCC barrier post");
    let _ = req.finalize();
}

/// Allreduce a u64 value across all processes using UCC collective allreduce.
///
/// Uses SUM reduction to aggregate values from all ranks. The result
/// is the same on all processes after completion.
///
/// Uses `init_and_post` which is synchronous — blocks until the collective
/// completes. Replaces the previous tag-message-based reduction pattern.
pub fn allreduce_u64(comm: &CommCtx, value: u64) -> u64 {
    let mut buf = [value];
    let src_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(buf.as_ptr() as *const u8, std::mem::size_of::<u64>())
    };
    let dst_bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, std::mem::size_of::<u64>())
    };
    let src = UccMemHandle::map_slice(&comm.ucc_context, src_bytes).expect("UCC allreduce src map");
    let dst =
        UccMemHandle::map_slice_mut(&comm.ucc_context, dst_bytes).expect("UCC allreduce dst map");
    let mut req = CollectiveBuilder::new(UccCollectiveType::Allreduce)
        .with_src(&src)
        .with_dst(&dst)
        .with_count(1)
        .with_dtype(7) // UCC_DT_UINT64
        .with_reduction_op(UccReductionOp::Sum)
        .init_and_post(&comm.ucc_team)
        .expect("UCC allreduce post");
    let _ = req.finalize();
    buf[0]
}

// ── Internal helpers ──

fn flush_ep_blocking(worker: &worker::Worker, ep: &ep::Ep, param: &ucx_sys::RequestParam) {
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

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    // ── Single-node (UpdateComm) tests ──

    /// UpdateComm can be constructed with arbitrary rank/size values.
    #[test]
    fn test_update_comm_creation() {
        let comm = UpdateComm { rank: 0, size: 1 };
        assert_eq!(comm.rank, 0);
        assert_eq!(comm.size, 1);
    }

    /// UpdateComm with non-zero rank.
    #[test]
    fn test_update_comm_nonzero_rank() {
        let comm = UpdateComm {
            rank: 42,
            size: 128,
        };
        assert_eq!(comm.rank, 42);
        assert_eq!(comm.size, 128);
    }

    // ── Single-node allreduce tests ──

    /// Single-node allreduce returns the input value unchanged.
    #[test]
    fn test_allreduce_u64_single_identity() {
        let comm = UpdateComm { rank: 0, size: 1 };
        assert_eq!(allreduce_u64_single(&comm, 42), 42);
    }

    /// Single-node allreduce with zero.
    #[test]
    fn test_allreduce_u64_single_zero() {
        let comm = UpdateComm { rank: 0, size: 1 };
        assert_eq!(allreduce_u64_single(&comm, 0), 0);
    }

    /// Single-node allreduce with max u64.
    #[test]
    fn test_allreduce_u64_single_max() {
        let comm = UpdateComm { rank: 0, size: 1 };
        assert_eq!(allreduce_u64_single(&comm, u64::MAX), u64::MAX);
    }

    /// Single-node allreduce ignores comm fields (rank/size don't matter).
    #[test]
    fn test_allreduce_u64_single_ignores_comm() {
        let comm = UpdateComm {
            rank: 99,
            size: 256,
        };
        assert_eq!(allreduce_u64_single(&comm, 12345), 12345);
    }

    // ── Tag constant tests ──

    /// TAG_SYNC and TAG_VERIFY have distinct, non-zero values.
    #[test]
    fn test_tag_constants() {
        assert_eq!(TAG_SYNC, 0x3000);
        assert_eq!(TAG_VERIFY, 0x4000);
        assert_ne!(TAG_SYNC, TAG_VERIFY);
        assert!(TAG_SYNC > 0);
        assert!(TAG_VERIFY > 0);
    }

    // ── PMIx key constant tests ──

    /// PMIx key constants are correct and distinct.
    #[test]
    fn test_pmix_key_constants() {
        assert_eq!(PMIX_KEY_UCX_ADDR, "gups.ucx.addr");
        assert_eq!(PMIX_KEY_UCX_MEMH, "gups.ucx.memh");
        assert_eq!(PMIX_KEY_UCX_TABLE_ADDR, "gups.ucx.table_addr");
    }

    // ── Multi-process tests (require DVM — marked #[ignore]) ──

    /// create_multiprocess initializes UCX + PMIx + UCC and returns valid rank/size.
    ///
    /// #[ignore] — requires PMIx daemon (prrte) to be running.
    #[test]
    #[ignore = "requires PMIx daemon (prrte) for multi-process bootstrap"]
    fn test_create_multiprocess() {
        let table: Vec<u64> = vec![0; 1024];
        let table_ptr = table.as_ptr() as *mut u64;
        let table_bytes = 1024 * std::mem::size_of::<u64>();

        let (rank, size, ctx) = create_multiprocess(table_ptr, table_bytes);

        assert!(rank < size, "rank {} must be less than size {}", rank, size);
        assert!(size > 0, "size must be positive");
        assert_eq!(ctx.rank, rank);
        assert_eq!(ctx.size, size);
        assert_eq!(ctx.endpoints.len(), size);
        assert_eq!(ctx.remote_rkeys.len(), size);
        assert_eq!(ctx.remote_table_addrs.len(), size);
    }

    /// atomic_xor_remote performs XOR on a peer's table entry.
    ///
    /// #[ignore] — requires PMIx daemon (prrte) for multi-process setup.
    #[test]
    #[ignore = "requires PMIx daemon (prrte) for multi-process setup"]
    fn test_atomic_xor_remote() {
        let table: Vec<u64> = vec![0; 1024];
        let table_ptr = table.as_ptr() as *mut u64;
        let table_bytes = 1024 * std::mem::size_of::<u64>();

        let (_rank, _size, ctx) = create_multiprocess(table_ptr, table_bytes);

        // XOR with self (peer == rank) — should work since self EP exists
        atomic_xor_remote(&ctx, ctx.rank, 0, 0xDEADBEEF);
        progress(&ctx);
        // In single-process mode, the self-XOR should be visible locally
        // (this verifies the atomic operation path compiles and runs)
    }

    /// barrier() completes without panic using UCC collective.
    ///
    /// #[ignore] — requires PMIx daemon (prrte) for multi-process setup.
    #[test]
    #[ignore = "requires PMIx daemon (prrte) for multi-process setup"]
    fn test_barrier_multiprocess() {
        let table: Vec<u64> = vec![0; 1024];
        let table_ptr = table.as_ptr() as *mut u64;
        let table_bytes = 1024 * std::mem::size_of::<u64>();

        let (_rank, _size, ctx) = create_multiprocess(table_ptr, table_bytes);

        // Barrier should complete without panic
        barrier(&ctx);
    }

    /// allreduce_u64 sums values across all processes using UCC.
    ///
    /// #[ignore] — requires PMIx daemon (prrte) for multi-process setup.
    #[test]
    #[ignore = "requires PMIx daemon (prrte) for multi-process setup"]
    fn test_allreduce_u64_multiprocess() {
        let table: Vec<u64> = vec![0; 1024];
        let table_ptr = table.as_ptr() as *mut u64;
        let table_bytes = 1024 * std::mem::size_of::<u64>();

        let (rank, size, ctx) = create_multiprocess(table_ptr, table_bytes);

        // Each rank contributes its rank number; sum should be 0+1+...+(size-1)
        let value = rank as u64;
        let result = allreduce_u64(&ctx, value);
        let expected: u64 = (0..size as u64).sum();
        assert_eq!(
            result, expected,
            "allreduce SUM should equal sum of all ranks"
        );
    }

    /// CommCtx endpoints and rkeys arrays have correct size.
    ///
    /// #[ignore] — requires PMIx daemon (prrte) for multi-process setup.
    #[test]
    #[ignore = "requires PMIx daemon (prrte) for multi-process setup"]
    fn test_commctx_array_sizes() {
        let table: Vec<u64> = vec![0; 1024];
        let table_ptr = table.as_ptr() as *mut u64;
        let table_bytes = 1024 * std::mem::size_of::<u64>();

        let (_rank, size, ctx) = create_multiprocess(table_ptr, table_bytes);

        assert_eq!(ctx.endpoints.len(), size, "endpoints array size mismatch");
        assert_eq!(
            ctx.remote_rkeys.len(),
            size,
            "remote_rkeys array size mismatch"
        );
        assert_eq!(
            ctx.remote_table_addrs.len(),
            size,
            "remote_table_addrs array size mismatch"
        );
    }
}
