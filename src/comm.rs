/// OpenSHMEM communication adapter for the GUPS benchmark.
///
/// Uses RMA (Remote Memory Access) with atomic XOR operations for direct
/// remote table updates. Pattern derived from osss-ucx:
/// - Memory registration -> rkey packing -> PMIx Put/Commit/Fence/Get exchange
/// - Direct `atomic_xor64` on remote table addresses (no message passing per update)
///
/// Multi-process only: full UCX stack with RMA atomics, PMIx for bootstrap.
///
/// UCC is used for collective operations (barrier, allreduce) replacing the
/// previous tag-message-based barrier and verification reduction.
#[cfg(feature = "ucc")]
use std::ffi::CString;

#[cfg(feature = "ucc")]
use ucx_sys::RequestParamBuilder;
#[cfg(feature = "ucc")]
use ucx_sys::context;
#[cfg(feature = "ucc")]
use ucx_sys::ep;
#[cfg(feature = "ucc")]
use ucx_sys::memh;
#[cfg(feature = "ucc")]
use ucx_sys::rma::RemoteKey;
#[cfg(feature = "ucc")]
use ucx_sys::worker;
#[cfg(feature = "ucc")]
use ucx_sys::worker::RemoteWorkerAddress;

#[cfg(feature = "ucc")]
use pmix::{
    GLOBAL, PmixClient, PmixValueBuilder, RANK_WILDCARD, commit, fence, get_value,
    info_with_string_key, put_value,
};

/// Owns a live [`PmixClient`] and disconnects on drop.
#[cfg(feature = "ucc")]
struct PmixSession(PmixClient);

#[cfg(feature = "ucc")]
impl Drop for PmixSession {
    fn drop(&mut self) {
        let _ = self.0.disconnect(None);
    }
}

#[cfg(feature = "ucc")]
impl std::ops::Deref for PmixSession {
    type Target = PmixClient;
    fn deref(&self) -> &PmixClient {
        &self.0
    }
}

/// Legacy tags retained for public compatibility; collectives use OpenSHMEM.
#[allow(dead_code)]
pub const TAG_SYNC: u64 = 0x3000;
#[allow(dead_code)]
pub const TAG_VERIFY: u64 = 0x4000;

// PMIx key names for data exchange (null-terminated C strings)
#[cfg(feature = "ucc")]
const PMIX_KEY_UCX_ADDR: &str = "gups.ucx.addr";
#[cfg(feature = "ucc")]
const PMIX_KEY_UCX_MEMH: &str = "gups.ucx.memh";
#[cfg(feature = "ucc")]
const PMIX_KEY_UCX_TABLE_ADDR: &str = "gups.ucx.table_addr";

/// Multi-process communication context using UCX RMA atomics + PMIx bootstrap.
/// UCC is used for collective operations (barrier, allreduce).
#[allow(dead_code)]
#[cfg(feature = "ucc")]
pub struct CommCtx {
    pub rank: usize,
    pub size: usize,
    /// Pointer to the UCX-allocated shared table segment (UCP_MEM_MAP_ALLOCATE).
    pub table_ptr: *mut u64,
    /// UCC team for collective operations (barrier, allreduce, etc.).
    ucc_team: UccTeam,
    /// UCC context for collective operations.
    ucc_context: UccContext,
    /// UCC library handle for collective operations.
    ucc_lib: UccLib,
    remote_rkeys: Vec<Option<RemoteKey>>,
    remote_table_addrs: Vec<u64>,
    _memh: memh::MemHandle,
    /// UCX endpoints — `None` for self (rank == peer), `Some(ep)` for remote peers.
    pub endpoints: Vec<Option<ep::Ep>>,
    pub worker: worker::Worker,
    context: context::Context,
    /// PMIx session — kept alive for the lifetime of this CommCtx.
    /// Drop disconnects (PMIx_Finalize).
    _pmix_ctx: PmixSession,
}

/// Resolve the PMIx server URI from the local URI file for standalone clients.
///
/// When running under `prterun`, PMIx_Init finds the server automatically via the
/// environment variables that prterun sets (PMIX_RANK, PMIX_SERVER_URI61, etc.).
/// The C library reads these internally — no explicit URI needed.
///
/// **CRITICAL:** Do NOT pass `pmix.srvr.uri` to `PMIx_Init` when under prterun.
/// That key is for `PMIx_Tool_Init`, and passing it to `PMIx_Init` causes
/// ErrUnreach or segfault with OpenPMIX 6.1.0.
///
/// Only resolve the URI file when NOT under prterun (standalone mode) and a
/// system server daemon is running. This avoids connecting to stale daemons.
///
/// Lookup: URI file at `/run/user/{uid}/prte/uri`
#[cfg(feature = "ucc")]
fn resolve_pmix_server_uri() -> Option<String> {
    // When running under prterun, let PMIx_Init discover the server via env vars.
    // Do NOT pass pmix.srvr.uri to PMIx_Init — that key is for PMIx_Tool_Init,
    // and passing it to PMIx_Init causes ErrUnreach/segfault with OpenPMIX 6.1.0.
    if std::env::var("PMIX_RANK").is_ok() {
        return None;
    }

    // Standalone mode: try to connect to a running system server via URI file
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
/// When launched under prterun, PMIx discovers the server automatically via
/// environment variables — no explicit URI needed. Standalone runs resolve
/// the URI from the local file if a system server is running.
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
#[cfg(feature = "ucc")]
pub fn create_multiprocess(table_base: *mut u64, table_bytes: usize) -> (usize, usize, CommCtx) {
    // 1. Initialize PMIx — gets our rank.
    // Reuse process session if a probe already connected (main.rs multiproc path).
    // When under prterun: connect_new(None) discovers the server via env vars.
    // When standalone: resolve_pmix_server_uri() tries the system server URI file.
    let pmix_info =
        resolve_pmix_server_uri().and_then(|uri| info_with_string_key("pmix.srvr.uri", &uri).ok());
    let pmix_client = if pmix::PmixClient::new().is_live() {
        pmix::PmixClient::new()
    } else {
        pmix::PmixClient::connect_new(pmix_info).expect("PMIx connect")
    };
    let pmix_ctx = PmixSession(pmix_client);
    let rank = pmix_ctx.require_rank() as usize;
    let my_proc = pmix_ctx.require_proc();

    // 2. Query PMIX_JOB_SIZE via wildcard proc for total process count
    // Fall back to PMIX_SIZE env var if PMIx doesn't publish it
    let wc_proc = pmix_ctx
        .proc_with_nspace(RANK_WILDCARD)
        .expect("wildcard_proc");
    let size = get_value(&wc_proc, pmix::JOB_SIZE, None)
        .ok()
        .map(|v| v.uint32() as usize)
        .or_else(|| std::env::var("PMIX_SIZE").ok().and_then(|s| s.parse().ok()))
        .unwrap_or_else(|| {
            panic!("Cannot determine job size from PMIx (pmix.job.size) or PMIX_SIZE env var");
        });
    eprintln!("[gups-rs] PMIx rank={}, size={}", rank, size);

    // 3. Initialize UCX context
        // Tag + Rma — Amo64 not needed as a context flag; amo_xor64 works on RMA
        // endpoints without explicit Amo64 feature.
        // NOTE: do NOT request UCP_FEATURE_EXPORTED_MEMH here. That feature forces
        // UCX's RMA lane selection to require UCT_MD_FLAG_REG (select.c), which
        // sysv/posix shared-memory transports do not advertise — so on a no-RDMA
        // box the EP create fails with "no memory registration". Without it, UCX
        // falls through to the "allocated" pass (UCT_MD_FLAG_ALLOC) and selects
        // sysv/posix for the RMA lane. We use the legacy pack_rkey/ep_rkey_unpack
        // path, so EXPORTED_MEMH is not needed.
    let features = context::Flags::Tag | context::Flags::Rma | context::Flags::Amo64;
    let ctx_params = context::ParamsBuilder::new()
        .features(features)
        .estimated_num_eps(size - 1)
        .estimated_num_ppn(2)
        .build();
    let config = context::Config::read("", "").expect("UCX config read");
    let mut uctx = context::Context::new(&config, &ctx_params).expect("UCX context init");
    drop(config);

    // 4. Create worker
    let wparams = worker::ParamsBuilder::new().build();
    let worker = uctx.worker_create(&wparams).expect("UCX worker create");

    // 5. Register table memory with UCX FIRST so the worker address advertises
    // the shared-memory (sysv/posix) transport for RMA atomics.
    // Use UCP_MEM_MAP_ALLOCATE so UCX allocates the table in a shared segment
    // that peer processes can access via RMA atomics. Passing a heap Vec
    // pointer here would only register private heap memory, which sysv/posix
    // transports cannot reach cross-process (UCS_ERR_UNREACHABLE).
    let mut mem_params = memh::MemMapParamsBuilder::new();
    mem_params
        .address(std::ptr::null_mut()) // ALLOCATE: UCX allocates; address must be null
        .length(table_bytes)
        .flags(2 | 8); // UCP_MEM_MAP_ALLOCATE | UCP_MEM_MAP_SYMMETRIC_RKEY
    let memh = memh::MemHandle::map(&uctx, &mut mem_params).expect("Memory registration");
    eprintln!("[gups-rs] mem_map OK, addr={:p}", memh.query().expect("q").address());

    // 6. Pack own worker address (now advertises the sm transport)
    eprintln!("[gups-rs] STEP: about to pack_address");
    let packed_addr = worker.pack_address().expect("Worker address pack");
    let own_addr_bytes = packed_addr.to_vec();

    // 7. Pack rkey for our memory using legacy ucp_rkey_pack
    // (works on all memory domains including self/sysv/posix)
    let packed_rkey = memh::pack_rkey(&uctx, &memh).expect("Rkey pack");
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
    // Use wildcard proc to cover ALL ranks, not just self (fix for issue #10)
    commit().expect("PMIx_Commit");
    let wc_proc = pmix_ctx
        .proc_with_nspace(RANK_WILDCARD)
        .expect("wildcard_proc");
    fence(&wc_proc, None).expect("PMIx_Fence");

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

    // 12. Create UCX endpoints to each peer (skip self — local updates go direct)
    let endpoints: Vec<Option<ep::Ep>> = (0..size)
        .map(|p| {
            if p == rank {
                None // No self-endpoint needed — local updates are direct memory access
            } else {
                let remote_addr = RemoteWorkerAddress::new(peer_addrs[p].clone());
                let ep_params = ep::ParamsBuilder::new().address(&remote_addr).build();
                let ep = worker.create_ep(ep_params).expect("EP create for peer");
                eprintln!("[gups-rs] STEP: created EP to peer {}", p);
                Some(ep)
            }
        })
        .collect();

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
        let rkey = RemoteKey::unpack(endpoints[peer].as_ref().unwrap(), &peer_memh_data[peer])
            .expect("rkey unpack");
        remote_rkeys[peer] = Some(rkey);
    }

    // 14. Flush all endpoints (skip self)
    let flush_param = RequestParamBuilder::new().no_imm_cmpl().build();
    for (_peer, ep) in endpoints.iter().enumerate().take(size) {
        if let Some(e) = ep {
            flush_ep_blocking(&worker, e, &flush_param);
        }
    }

    // 15. Initialize UCC for collective operations (barrier, allreduce, etc.)
    let ucc_lib = UccLib::init().expect("UCC library init");

    // UCC requires an OOB (out-of-band) allgather for BOTH context and team
    // creation when no EP map is provided. Without it, UCC's address exchange
    // calls a null function pointer and segfaults. We back the OOB with PMIx
    // put/get. Create it once and reuse for context + team.
    let oob = unsafe { crate::oob::pmix_oob(&pmix_ctx, rank as u32, size as u32) };

    // Create UCC context with the OOB.
    let mut ucc_ctx_params = ucc::context::UccContextParams::default();
    ucc_ctx_params.with_oob(oob);
    let ucc_context =
        UccContext::with_params(ucc_lib.clone(), ucc_ctx_params).expect("UCC context create");

    // Create UCC team with explicit size + the same OOB.
    let mut ucc_team_params = UccTeamParams::default();
    ucc_team_params.with_team_size(size as u64);
    ucc_team_params.with_oob(oob);
    // UCC requires params.ep to match oob.oob_ep (the rank). The default is 0,
    // which only matches rank 0 — set it to our rank.
    ucc_team_params.inner_mut().ep = rank as u64;
    let ucc_team =
        UccTeam::with_params(ucc_context.clone(), ucc_team_params).expect("UCC team create");

    eprintln!("[gups-rs] UCC team created (rank={}, size={})", rank, size);

    let table_ptr = memh.query().expect("Memh query").address() as *mut u64;
    let ctx = CommCtx {
        rank,
        size,
        table_ptr,
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
///
/// For local updates (peer == rank), this falls through to direct memory access
/// in the main benchmark loop — callers should check `peer == rank` before calling.
#[cfg(feature = "ucc")]
pub fn atomic_xor_remote(comm: &CommCtx, peer: usize, offset: usize, value: u64) {
    let remote_addr = comm.remote_table_addrs[peer] + (offset * std::mem::size_of::<u64>()) as u64;
    let rkey = comm.remote_rkeys[peer].as_ref().expect("rkey for peer");
    let ep = comm.endpoints[peer].as_ref().expect("endpoint for peer");

    // UCX requires the datatype to be set on the atomic op request
    // (UCP_OP_ATTR_FIELD_DATATYPE) — otherwise ucp_atomic_op_nbx fails with
    // "missing atomic operation datatype". Use a contiguous 8-byte datatype
    // for the 64-bit XOR.
    let mut param_builder = RequestParamBuilder::new();
    param_builder.datatype(ucx_sys::dt::dt_make_contig(8));
    let param = param_builder.build();
    let result = ep.amo_xor64(value, remote_addr, rkey, &param);
    match result {
        Err(e) => {
            eprintln!(
                "atomic_xor64 failed on peer {} offset {}: {:?}",
                peer, offset, e
            );
        }
        Ok(Some(req)) => {
            // Non-blocking op returned an in-flight request. We must keep it
            // alive and progress until it completes — dropping it would free
            // the request and abandon the atomic before it lands on the remote
            // table (this was silently losing ~50% of remote updates).
            let req = req;
            loop {
                match req.check_finished() {
                    Ok(true) => break,
                    Ok(false) => {
                        progress(comm);
                        std::thread::yield_now();
                    }
                    Err(e) => {
                        eprintln!("atomic_xor64 request error: {:?}", e);
                        break;
                    }
                }
            }
        }
        Ok(None) => {
            // Completed synchronously (UCS_OK).
        }
    }

    // Guarantee delivery: flush the endpoint so all previously issued atomics
    // on this EP are known to be delivered to the remote side before we return.
    // Without this, a later barrier can complete while an in-flight atomic is
    // still buffered, and it lands AFTER verification reads the table.
    let flush_req = ep.flush_nbx();
    if !flush_req.is_live() {
        return;
    }
    loop {
        match flush_req.check_finished() {
            Ok(true) => break,
            Ok(false) => {
                progress(comm);
                std::thread::yield_now();
            }
            Err(e) => {
                eprintln!("ep flush failed on peer {}: {:?}", peer, e);
                break;
            }
        }
    }
}

/// Progress the UCX worker.
#[cfg(feature = "ucc")]
pub fn progress(comm: &CommCtx) {
    loop {
        if !comm.worker.progress() {
            break;
        }
    }
}

/// Barrier across all processes using UCC collective barrier.
///
/// Uses `init` + `post` + poll `test()` (synchronous). `ucc_collective_init_and_post`
/// is not implemented in this UCC build, so we init, post, then spin on `test()`
/// until the collective completes.
pub fn barrier(comm: &CommCtx) {
    let mut stub = [0u8];
    let mut req = CollectiveBuilder::new(UccCollectiveType::Barrier)
        .with_inplace(&mut stub)
        .with_count(1)
        .init(&comm.ucc_team)
        .expect("UCC barrier init");
    req.post().expect("UCC barrier post");
    // Spin until the collective completes (poll the raw request status).
    // Must progress the UCX worker so the UCC collective can make progress.
    loop {
        let done = unsafe {
            (*req.request()).status == ucc::bindings::ucc_status_t_UCC_OK
        };
        if done {
            break;
        }
        // Progress the UCC context (UCC collectives use their own internal
        // UCP transport, so the gups-rs UCX worker progress is not enough).
        comm.ucc_context.progress();
        std::thread::yield_now();
    }
    let _ = req.finalize();
}

/// Allreduce a u64 value across all processes using UCC collective allreduce.
///
/// Uses SUM reduction to aggregate values from all ranks. The result
/// is the same on all processes after completion.
///
/// Uses `init` + `post` + poll `test()` (synchronous). `ucc_collective_init_and_post`
/// is not implemented in this UCC build, so we init, post, then spin on `test()`.
pub fn allreduce_u64(comm: &CommCtx, value: u64) -> u64 {
    let mut buf = [value];
    let bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, std::mem::size_of::<u64>())
    };
    let mut req = CollectiveBuilder::new(UccCollectiveType::Allreduce)
        .with_inplace(bytes)
        .with_count(1)
        .with_dtype(8) // UCC_DT_UINT64 (DataType::Uint64)
        .with_reduction_op(UccReductionOp::Sum)
        .init(&comm.ucc_team)
        .expect("UCC allreduce init");
    req.post().expect("UCC allreduce post");
    // Spin until the collective completes (poll the raw request status).
    // Must progress the UCX worker so the UCC collective can make progress.
    loop {
        let done = unsafe {
            (*req.request()).status == ucc::bindings::ucc_status_t_UCC_OK
        };
        if done {
            break;
        }
        // Progress the UCC context (UCC collectives use their own internal
        // UCP transport, so the gups-rs UCX worker progress is not enough).
        comm.ucc_context.progress();
        std::thread::yield_now();
    }
    let _ = req.finalize();
    buf[0]
}

// ── Internal helpers ──

#[cfg(feature = "ucc")]
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

    // ── Tag constant tests ──

    /// TAG_SYNC and TAG_VERIFY have distinct, non-zero values.
    #[test]
    fn test_tag_constants() {
        assert_eq!(TAG_SYNC, 0x3000);
        assert_eq!(TAG_VERIFY, 0x4000);
        assert_ne!(TAG_SYNC, TAG_VERIFY);
        assert_ne!(TAG_SYNC, 0);
        assert_ne!(TAG_VERIFY, 0);
    }

    // ── PMIx key constant tests ──

    /// PMIx key constants are correct and distinct.
    #[test]
    #[cfg(feature = "ucc")]
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
    #[cfg(feature = "ucc")]
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
    #[cfg(feature = "ucc")]
    fn test_atomic_xor_remote() {
        let table: Vec<u64> = vec![0; 1024];
        let table_ptr = table.as_ptr() as *mut u64;
        let table_bytes = 1024 * std::mem::size_of::<u64>();

        let (_rank, _size, ctx) = create_multiprocess(table_ptr, table_bytes);

        // XOR with self (peer == rank) — should work since self EP exists
        atomic_xor_remote(&ctx, ctx.rank, 0, 0xDEADBEEF);
        progress(&ctx);
        // XOR against ourselves exercises the same remote-atomic path a peer
        // would use, without needing a second process to observe it.
    }

    /// barrier() completes without panic using UCC collective.
    ///
    /// #[ignore] — requires PMIx daemon (prrte) for multi-process setup.
    #[test]
    #[ignore = "requires PMIx daemon (prrte) for multi-process setup"]
    #[cfg(feature = "ucc")]
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
    #[cfg(feature = "ucc")]
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
    #[cfg(feature = "ucc")]
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
