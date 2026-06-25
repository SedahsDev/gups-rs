/// Verification logic for the GUPS benchmark.
///
/// After the update phase, replay the RNG to determine the expected value of each
/// table entry and compare against actual values.
use crate::rng::{lfsr_step, starts};

/// Verify the local table segment.
///
/// Returns the number of errors found.
///
/// # Arguments
/// * `table` - The local table to verify
/// * `local_table_size` - Size of the local table (power of 2)
/// * `global_start` - Global index of the first element owned by this process
/// * `num_procs` - Total number of processes (power of 2)
/// * `log_num_procs` - log2(num_procs)
/// * `log_table_size` - log2(global table size)
/// * `proc_num_updates` - Number of updates performed by each process
/// * `my_proc` - This process's rank
#[allow(clippy::too_many_arguments)]
pub fn verify_table(
    table: &[u64],
    local_table_size: u64,
    global_start: u64,
    num_procs: u64,
    log_num_procs: u64,
    log_table_size: u64,
    proc_num_updates: i64,
    my_proc: u64,
) -> u64 {
    let local_mask = local_table_size - 1;
    let proc_mask = num_procs - 1;
    let log_table_local = log_table_size - log_num_procs;

    // For each process, replay its RNG stream and track which updates hit our local table
    // We need to replay all processes' streams to find updates targeting our table segment

    // Create a copy of the initial table values for comparison
    let mut expected: Vec<u64> = Vec::with_capacity(local_table_size as usize);
    for i in 0..local_table_size {
        expected.push(global_start + i);
    }

    // Replay all processes' update streams
    for proc_rank in 0..num_procs {
        // Initialize RNG for this process
        let mut ran = starts(4 * global_start + proc_rank * 4);

        for _ in 0..proc_num_updates {
            ran = lfsr_step(ran);
            let remote_proc = ((ran >> log_table_local) & (proc_mask as i64)) as u64;

            // If this update targets our process, apply it to expected
            if remote_proc == my_proc {
                let index = (ran as u64 & local_mask) as usize;
                expected[index] ^= ran as u64;
            }
        }
    }

    // Compare expected vs actual
    let mut errors: u64 = 0;
    for (&actual, &exp) in table.iter().zip(expected.iter()) {
        if actual != exp {
            errors += 1;
        }
    }

    errors
}
