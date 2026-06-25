//! HPCC table management.
//!
//! Each process owns a contiguous segment of the global table.
//! Table size is a power of 2, distributed evenly across P processes (P must be power of 2).

/// Initialize the local table segment.
///
/// `global_start` is the global index of the first element owned by this process.
/// Each entry is initialized to its global index.
pub fn init_table(table: &mut [u64], global_start: u64) {
    for (i, entry) in table.iter_mut().enumerate() {
        *entry = global_start + i as u64;
    }
}

/// Apply a single XOR update to the local table.
#[inline]
pub fn apply_update(table: &mut [u64], datum: u64, local_mask: u64) {
    let index = (datum & local_mask) as usize;
    table[index] ^= datum;
}
