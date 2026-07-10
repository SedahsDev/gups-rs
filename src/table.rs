//! Local GUPS table helpers.
//!
//! # Threading
//!
//! Updates are **single-threaded** by design: each process owns a contiguous
//! table segment and applies updates only on its rank (local XOR or remote
//! UCX atomic). Do not share a `&mut [u64]` table across threads without
//! external synchronization. Multi-process safety is via UCX RMA atomics, not
//! Rust `AtomicU64` on the local table.

/// Initialize the local table segment: `table[i] = i + offset`.
pub fn init_table(table: &mut [u64], offset: u64) {
    for (i, slot) in table.iter_mut().enumerate() {
        *slot = i as u64 + offset;
    }
}

/// Apply a single XOR update to the local table.
///
/// `datum` encodes both the global index stream and the XOR payload (HPCC GUPS).
pub fn apply_update(table: &mut [u64], datum: u64, local_mask: u64) {
    let local_idx = (datum & local_mask) as usize;
    // Bounds: callers size the table as a power of two covering local_mask.
    debug_assert!(local_idx < table.len());
    table[local_idx] ^= datum;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_and_update() {
        let mut t = vec![0u64; 8];
        init_table(&mut t, 0);
        assert_eq!(t[0], 0);
        assert_eq!(t[7], 7);
        apply_update(&mut t, 1, 7); // idx 1
        assert_eq!(t[1], 1 ^ 1);
    }
}
