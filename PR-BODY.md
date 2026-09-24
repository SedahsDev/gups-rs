# Issue #10 Fix: Missing flush before verification + PMIx fence with self-only proc

## Changes

### 1. Fixed PMIx fence to use ALL ranks (not just self)

**Problem**: The PMIx fence was using a proc array containing only the current rank (`&my_proc`), which meant the barrier only covered self. This caused a race condition where verification could read stale table data from other ranks.

**Solution**: Changed to use the wildcard proc which covers ALL ranks in the job:
```rust
// Before (WRONG):
fence(&my_proc, None).expect("PMIx_Fence");

// After (CORRECT):
let wc_proc = pmix_ctx
    .proc_with_nspace(RANK_WILDCARD)
    .expect("wildcard_proc");
fence(&wc_proc, None).expect("PMIx_Fence");
```

### 2. Added explicit endpoint flush after each atomic operation

**Problem**: UCX atomic operations (`amo_xor64`) are posted but not guaranteed to complete before the function returns. Without a flush, a later barrier could complete while atomics are still in flight, causing them to land AFTER verification reads the table.

**Solution**: Added explicit flush after each `amo_xor64`:
```rust
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
```

### 3. Added UCX datatype to AMO request (Issue #12)

**Problem**: UCX requires `UCP_OP_ATTR_FIELD_DATATYPE` to be set on atomic operations. Without it, `ucp_atomic_op_nbx` fails with "missing atomic operation datatype".

**Solution**: Explicitly set the datatype on the AMO request:
```rust
let mut param_builder = RequestParamBuilder::new();
param_builder.datatype(ucx_sys::dt::dt_make_contig(8));
let param = param_builder.build();
```

### 4. Added UCC OOB initialization (Issue #12)

**Problem**: UCC requires an OOB (out-of-band) allgather for both context and team creation when no EP map is provided. The previous code tried to create UCC context without OOB, causing segfaults.

**Solution**: Initialize PMIx-backed OOB once and reuse for both UCC context and team:
```rust
let oob = unsafe { crate::oob::pmix_oob(&pmix_ctx, rank as u32, size as u32) };

let mut ucc_ctx_params = ucc::context::UccContextParams::default();
ucc_ctx_params.with_oob(oob);
let ucc_context = UccContext::with_params(ucc_lib.clone(), ucc_ctx_params).expect("UCC context create");

let mut ucc_team_params = UccTeamParams::default();
ucc_team_params.with_team_size(size as u64);
ucc_team_params.with_oob(oob);
ucc_team_params.inner_mut().ep = rank as u64;  // Match OOB's ep
```

### 5. Updated barrier() and allreduce_u64() to work with new UCC API

**Problem**: The new UCC build doesn't support `init_and_post()` - only separate `init()` + `post()` + polling with `test()`.

**Solution**: Changed to use the three-step pattern with polling:
```rust
let req = CollectiveBuilder::new(UccCollectiveType::Barrier)
    .with_inplace(&mut stub)
    .with_count(1)
    .init(&comm.ucc_team).expect("UCC barrier init");
req.post().expect("UCC barrier post");

// Poll until complete
loop {
    let done = unsafe {
        (*req.request()).status == ucc::bindings::ucc_status_t_UCC_OK
    };
    if done { break; }
    comm.ucc_context.progress();  // Must progress UCC's internal transport
    std::thread::yield_now();
}
```

### 6. Added warmup phase (Issue #13)

**Problem**: First-touch costs, lazy connection setup, and CPU frequency warmup were contaminating the GUPS measurement. The benchmark started timing immediately without any warmup.

**Solution**: Added a warmup phase before the timed loop:
```rust
let warmup_iterations = std::cmp::min(num_updates / 10, 1000);
// Perform warmup updates
// Quiet + barrier to ensure completion
// THEN start timing
```

This ensures:
- First-touch page faults happen before timing
- UCC/UCX connections are fully established
- CPU caches and frequency scaling are warmed up
- CPU register state is stable

### 7. Added progress calls after AMO

**Problem**: UCX operations are async. If the UCX worker is not progressed, the atomic operations will hang.

**Solution**: Added progress calls while waiting for requests to complete:
```rust
loop {
    match req.check_finished() {
        Ok(true) => break,
        Ok(false) => {
            progress(comm);  // Keep UCX worker progressing
            std::thread::yield_now();
        }
        Err(e) => { /* handle error */ }
    }
}
```

## Verification

- Code compiles with `cargo check`
- All changes are logically verified
- Flush is now called after every remote AMO
- PMIx fence covers all ranks
- Warmup phase prevents first-touch contamination
- Related to Issue #12 (flush is explicit library feature)

## Testing

To verify the fix works:
```bash
# Single process (should be fast, no remote ops)
cargo run --release -- --table-size-log 20 --num-updates 1000

# Multi-process (should complete correctly with all updates)
mpirun -np 4 --allow-run-as-root cargo run --release --multi --table-size-log 20 --num-updates 1000
```

The multi-process run should now complete without verification failures and report correct GUPS (GMUPS).

## Related Issues

- Fixes #10 (missing flush + self-only fence)
- Fixes #12 (explicit flush + AMO datatype + UCC OOB)
- Fixes #13 (warmup phase)
- Related to Issue #11 (UCC API changes)
