//! UCC OOB (out-of-band) collective implementation backed by PMIx put/get.
//!
//! UCC requires an OOB allgather for team/context creation when no EP map is
//! provided. gups-rs uses PMIx for bootstrap, so we implement the OOB allgather
//! on top of PMIx_Put/PMIx_Commit/PMIx_Fence/PMIx_Get — the same primitives
//! already used for worker-address/rkey exchange.
//!
//! The OOB allgather is synchronous (blocking): each rank puts its `size`-byte
//! chunk under a unique key, commits, fences, then reads every peer's chunk into
//! the contiguous `recv_buf` at offset `oob_ep * size`.
//!
//! UCC calls the OOB allgather MULTIPLE times during context/team creation
//! (e.g. once for `ctx_addr_len` with size=8, then again for the addresses with
//! size=max_addrlen). Each call must use a FRESH set of keys, otherwise a later
//! call overwrites the keys and an earlier call reads stale data of the wrong
//! length. We use a per-process atomic counter in the key so every invocation
//! is unique. UCC invokes the allgather in lockstep across ranks, so the counter
//! stays synchronized.
//!
//! # Safety
//! These callbacks are invoked from C (UCC) across the FFI boundary, so they
//! must never panic (a panic cannot unwind through C). All fallible work is
//! wrapped in `catch_unwind` and returns a UCC error status on failure.

use std::ffi::CString;
use std::os::raw::c_void;
use std::sync::atomic::{AtomicU64, Ordering};

use pmix::{commit, fence, get_value, put_value, GLOBAL, PmixClient, PmixValueBuilder};
use ucc::bindings::{
    ucc_oob_coll, ucc_status_t, ucc_status_t_UCC_ERR_NO_MESSAGE, ucc_status_t_UCC_OK,
};

/// Key prefix for OOB allgather chunks. Each rank publishes under
/// `gups.ucc.oob.<call_id>.<rank>`.
const OOB_KEY_PREFIX: &str = "gups.ucc.oob.";

/// Monotonic counter so each allgather invocation uses fresh keys.
static OOB_CALL_ID: AtomicU64 = AtomicU64::new(0);

/// Context passed to the OOB callbacks via `coll_info`.
struct OobCtx {
    client: *const PmixClient,
    n_eps: u32,
}

/// Build a UCC OOB collective struct backed by PMIx.
///
/// `client` must be a connected PMIx client. The returned struct holds a raw
/// pointer to `client` (and the team size) in `coll_info`, so the caller must
/// keep `client` alive for as long as the OOB is used.
pub unsafe fn pmix_oob(client: &PmixClient, rank: u32, n_eps: u32) -> ucc_oob_coll {
    let ctx = Box::into_raw(Box::new(OobCtx {
        client: client as *const PmixClient,
        n_eps,
    }));
    let mut oob: ucc_oob_coll = std::mem::zeroed();
    oob.allgather = Some(pmix_oob_allgather);
    oob.req_test = Some(pmix_oob_req_test);
    oob.req_free = Some(pmix_oob_req_free);
    oob.coll_info = ctx as *mut c_void;
    oob.n_oob_eps = n_eps;
    oob.oob_ep = rank;
    oob
}

/// Synchronous allgather: each rank's `size`-byte `src_buf` is gathered into
/// `recv_buf` at offset `oob_ep * size`. Blocks until all ranks have published.
unsafe extern "C" fn pmix_oob_allgather(
    src_buf: *mut c_void,
    recv_buf: *mut c_void,
    size: usize,
    allgather_info: *mut c_void,
    request: *mut *mut c_void,
) -> ucc_status_t {
    // Never panic across the FFI boundary — catch everything and return an error.
    let result = std::panic::catch_unwind(|| {
        let ctx = &*(allgather_info as *const OobCtx);
        let client = &*ctx.client;
        let rank = client.require_rank();
        let n_eps = ctx.n_eps;
        let call_id = OOB_CALL_ID.fetch_add(1, Ordering::SeqCst);

        // Publish our chunk under a unique key for THIS call.
        let key = CString::new(format!("{}{}.{}", OOB_KEY_PREFIX, call_id, rank)).unwrap();
        let chunk = std::slice::from_raw_parts(src_buf as *const u8, size);
        let mut val = PmixValueBuilder::new()
            .byte_object(chunk)
            .expect("oob byte_object")
            .build()
            .expect("oob build");
        put_value(GLOBAL, &key, &mut val).expect("oob put");
        commit().expect("oob commit");

        // Fence to make all puts visible to all ranks.
        let my_proc = client.require_proc();
        fence(&my_proc, None).expect("oob fence");

        // Read every peer's chunk into recv_buf at offset oob_ep * size.
        let recv = std::slice::from_raw_parts_mut(recv_buf as *mut u8, size * n_eps as usize);
        for peer in 0..n_eps {
            let peer_key_bytes = format!("{}{}.{}\0", OOB_KEY_PREFIX, call_id, peer);
            let peer_proc = client.proc_with_nspace(peer).expect("oob proc");
            let val = get_value(&peer_proc, peer_key_bytes.as_bytes(), None).expect("oob get");
            let bytes = val.bytes_copy();
            // Guard against a peer publishing a different-length chunk.
            if bytes.len() != size {
                return Err(());
            }
            recv[peer as usize * size..(peer as usize + 1) * size].copy_from_slice(&bytes);
        }

        // Synchronous: mark the request complete immediately.
        if !request.is_null() {
            *request = std::ptr::null_mut();
        }
        Ok(())
    });

    match result {
        Ok(Ok(())) => ucc_status_t_UCC_OK,
        _ => ucc_status_t_UCC_ERR_NO_MESSAGE,
    }
}

/// Synchronous OOB: request is always complete.
unsafe extern "C" fn pmix_oob_req_test(_request: *mut c_void) -> ucc_status_t {
    ucc_status_t_UCC_OK
}

/// Free an OOB request (no-op for synchronous implementation).
unsafe extern "C" fn pmix_oob_req_free(_request: *mut c_void) -> ucc_status_t {
    ucc_status_t_UCC_OK
}
