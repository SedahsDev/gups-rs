//! GUPS (Giga UPdates per Second) Benchmark in Rust
//!
//! Replicates the HPC Challenge GUPS benchmark using UCX for communication.
//!
//! ## Modes
//!
//! **Single-process** (`--single`): No UCX needed, validates the algorithm locally.
//!
//! **Multi-process** (RMA atomics): Uses UCX Remote Memory Access with atomic XOR
//! for direct remote table updates. Pattern derived from osss-ucx SHMEM implementation.
//!
//! ## Usage
//! ```text
//! # Single-process mode (no UCX needed, validates algorithm):
//! gups-rs --single
//!
//! # Multi-process mode (requires UCX, run via prterun or similar):
//! prterun -np 2 ./gups-rs
//! ```

mod comm;
mod rng;
mod table;
mod verify;

use std::env;
use std::process;
use std::time::Instant;

use comm::{allreduce_u64, atomic_xor_remote, barrier, create_multiprocess, progress};
use rng::{lfsr_step, starts};
use table::{apply_update, init_table};

fn print_usage() {
    eprintln!("Usage: gups-rs [options]");
    eprintln!("  -t, --table-size SIZE    Table size as power of 2 (default: auto, half of RAM)");
    eprintln!("  -u, --updates COUNT      Number of updates (default: 4x table size)");
    eprintln!("  --single                 Run in single-process mode (no UCX)");
    eprintln!("  -h, --help               Show help");
}

fn is_power_of_two(n: u64) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

fn log2_floor(n: u64) -> u64 {
    n.trailing_zeros() as u64
}

fn get_total_ram_pages() -> u64 {
    let content = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    for line in content.lines() {
        if line.starts_with("MemTotal:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(kb) = parts[1].parse::<u64>() {
                    return (kb * 1024) / 8;
                }
            }
        }
    }
    (1_000_000_000u64) / 8
}

fn auto_table_size(num_procs: u64) -> u64 {
    let total_elements = get_total_ram_pages();
    let half = total_elements / 2 / num_procs * num_procs;
    let mut size = 1u64;
    while size * 2 <= half {
        size *= 2;
    }
    size
}

/// Single-process mode: no UCX, all updates are local
fn run_single(table_size_log: Option<u64>, num_updates_arg: Option<u64>) {
    let num_procs: u64 = 1;
    let log_num_procs: u64 = 0;

    let table_size = match table_size_log {
        Some(log) => 1u64 << log,
        None => auto_table_size(num_procs),
    };

    let local_table_size = table_size / num_procs;
    let log_table_size = log2_floor(table_size);
    let log_table_local = log_table_size - log_num_procs;

    let default_updates = 4 * table_size;
    let num_updates: u64 = num_updates_arg.unwrap_or(default_updates);
    let proc_num_updates: i64 = (num_updates / num_procs) as i64;

    let mut table: Vec<u64> = vec![0; local_table_size as usize];
    init_table(&mut table, 0);

    println!("Running on {} processors (PowerofTwo)", num_procs);
    println!(
        "Total Main table size = 2^{} = {} words",
        log_table_size, table_size
    );
    println!(
        "PE Main table size = 2^{} = {} words/PE",
        log_table_local, local_table_size
    );
    println!(
        "Default number of updates (RECOMMENDED) = {}",
        default_updates
    );

    #[allow(clippy::erasing_op)]
    let mut ran = starts(4 * 0); // single-process: rank=0, so 4*0 is correct
    let local_mask = local_table_size - 1;

    // Warm-up (discarded): stabilize caches / branch predictors
    let warmup = (proc_num_updates / 100).clamp(1, 10_000);
    for _ in 0..warmup {
        ran = lfsr_step(ran);
        apply_update(&mut table, ran as u64, local_mask);
    }
    // Re-init table after warm-up so verification baseline stays correct
    init_table(&mut table, 0);
    // single-process: rank=0, so 4*0 is correct (matches C ref: starts(4*GlobalStartMyProc))
    #[allow(clippy::erasing_op)]
    let start_val: u64 = 4 * 0;
    ran = starts(start_val);

    let start = Instant::now();

    for _ in 0..proc_num_updates {
        ran = lfsr_step(ran);
        let datum = ran as u64;
        apply_update(&mut table, datum, local_mask);
    }

    let elapsed = start.elapsed();
    let real_time = elapsed.as_secs_f64();

    let gups = (num_updates as f64 * 1e-9) / real_time;
    let gups_per_pe = gups / num_procs as f64;

    println!("Real time used = {:.6} seconds", real_time);
    println!("{:.9} Billion(10^9) Updates    per second [GUP/s]", gups);
    println!(
        "{:.9} Billion(10^9) Updates/PE per second [GUP/s]",
        gups_per_pe
    );

    let verify_start = Instant::now();
    let errors = verify::verify_table(
        &table,
        local_table_size,
        0,
        num_procs,
        log_num_procs,
        log_table_size,
        proc_num_updates,
        0,
    );
    let verify_elapsed = Instant::now().duration_since(verify_start).as_secs_f64();

    let status = if errors as f64 <= 0.01 * table_size as f64 {
        "passed"
    } else {
        "failed"
    };

    println!(
        "Verification:  Real time used = {:.6} seconds",
        verify_elapsed
    );
    println!(
        "Found {} errors in {} locations ({}).",
        errors, table_size, status
    );
}

/// Multi-process mode: uses UCX RMA atomics for direct remote table updates.
/// Rank and size are obtained from PMIx internally by create_multiprocess().
fn run_multi(table_size_log: Option<u64>, num_updates_arg: Option<u64>) {
    // create_multiprocess() handles PMIx init, rank/size discovery, UCX setup,
    // and rkey exchange. It returns (rank, size, CommCtx).
    //
    // We need to know size before allocating the table, but create_multiprocess
    // also needs the table pointer. We allocate a temporary 1-element table
    // just to satisfy the API, then replace it with the real table.
    //
    // Actually, the simpler approach: allocate a large-enough table upfront
    // (max reasonable size), then let create_multiprocess use it.
    // Even simpler: just allocate a dummy page and pass it,
    // then re-register after we know the real size.
    //
    // Phase 1: Connect once to learn rank/size, then keep the process session live.
    // create_multiprocess reuses a Live PmixClient (no second PMIx_Init — re-init
    // after disconnect is not supported by the session state machine).
    let pmix_probe = pmix::PmixClient::connect_new(None).expect("PMIx connect (probe)");
    let rank = pmix_probe.require_rank() as usize;

    // Query pmix.job.size via wildcard proc, fall back to PMIX_SIZE env var
    let wc_proc = pmix_probe
        .proc_with_nspace(pmix::RANK_WILDCARD)
        .expect("wildcard_proc");
    let size = pmix::get_value(&wc_proc, pmix::JOB_SIZE, None)
        .ok()
        .map(|v| v.uint32() as usize)
        .or_else(|| env::var("PMIX_SIZE").ok().and_then(|s| s.parse().ok()))
        .unwrap_or_else(|| {
            eprintln!("Cannot determine job size from PMIx (pmix.job.size) or PMIX_SIZE env var.");
            process::exit(1);
        });

    // Drop the probe *handle* only — do not disconnect; session stays Live for CommCtx.
    drop(pmix_probe);

    if size == 1 {
        eprintln!("PMIX_JOB_SIZE is 1 — nothing to do in multi-process mode.");
        process::exit(1);
    }

    if !is_power_of_two(size as u64) {
        if rank == 0 {
            eprintln!("Number of processes must be a power of 2");
        }
        process::exit(1);
    }

    let log_num_procs = log2_floor(size as u64);

    let table_size = match table_size_log {
        Some(log) => 1u64 << log,
        None => auto_table_size(size as u64),
    };

    let local_table_size = table_size / size as u64;
    let log_table_size = log2_floor(table_size);
    let log_table_local = log_table_size - log_num_procs;
    let global_start = local_table_size * rank as u64;

    let default_updates = 4 * table_size;
    let num_updates: u64 = num_updates_arg.unwrap_or(default_updates);
    let proc_num_updates: i64 = (num_updates / size as u64) as i64;

    // Allocate and initialize table
    let mut table: Vec<u64> = vec![0; local_table_size as usize];
    init_table(&mut table, global_start);

    if rank == 0 {
        println!("Running on {} processors (PowerofTwo)", size);
        println!(
            "Total Main table size = 2^{} = {} words",
            log_table_size, table_size
        );
        println!(
            "PE Main table size = 2^{} = {} words/PE",
            log_table_local, local_table_size
        );
        println!(
            "Default number of updates (RECOMMENDED) = {}",
            default_updates
        );
    }

    // Phase 2: Full UCX + PMIx communication setup (re-init PMIx)
    let table_bytes = local_table_size as usize * std::mem::size_of::<u64>();
    let (_rank, _size, comm_ctx) = create_multiprocess(table.as_mut_ptr(), table_bytes);

    // Barrier to ensure all connections and rkey exchanges are ready
    barrier(&comm_ctx);

    // Initialize RNG
    let mut ran = starts(4 * global_start) as i64;
    let local_mask = local_table_size - 1;
    let proc_mask = size as i64 - 1;

    // Timed update phase
    let start = Instant::now();

    for _iteration in 0..proc_num_updates {
        ran = lfsr_step(ran);
        let remote_proc = ((ran >> log_table_local) & proc_mask) as usize;
        let datum = ran as u64;

        if remote_proc == rank {
            // Local update — direct XOR
            apply_update(&mut table, datum, local_mask);
        } else {
            // Remote update — RMA atomic XOR on peer's table
            let target_offset = (datum & local_mask) as usize;
            let xor_value = datum;
            atomic_xor_remote(&comm_ctx, remote_proc, target_offset, xor_value);
        }

        // Periodically progress to ensure remote atomics complete
        if (_iteration as usize).is_multiple_of(1024) {
            progress(&comm_ctx);
        }
    }

    // Final progress to ensure all pending atomics are flushed
    progress(&comm_ctx);

    let elapsed = start.elapsed();
    let real_time = elapsed.as_secs_f64();

    // Barrier before reporting
    barrier(&comm_ctx);

    if rank == 0 {
        let gups = (num_updates as f64 * 1e-9) / real_time;
        let gups_per_pe = gups / size as f64;

        println!("Real time used = {:.6} seconds", real_time);
        println!("{:.9} Billion(10^9) Updates    per second [GUP/s]", gups);
        println!(
            "{:.9} Billion(10^9) Updates/PE per second [GUP/s]",
            gups_per_pe
        );
    }

    // Verification phase
    barrier(&comm_ctx);

    let verify_start = Instant::now();
    let errors = verify::verify_table(
        &table,
        local_table_size,
        global_start,
        size as u64,
        log_num_procs,
        log_table_size,
        proc_num_updates,
        rank as u64,
    );
    let verify_elapsed = Instant::now().duration_since(verify_start).as_secs_f64();

    // Collect total errors via UCC allreduce (SUM reduction)
    let total_errors: u64 = allreduce_u64(&comm_ctx, errors);

    barrier(&comm_ctx);

    if rank == 0 {
        let status = if total_errors as f64 <= 0.01 * table_size as f64 {
            "passed"
        } else {
            "failed"
        };

        println!(
            "Verification:  Real time used = {:.6} seconds",
            verify_elapsed
        );
        println!(
            "Found {} errors in {} locations ({}).",
            total_errors, table_size, status
        );
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut table_size_log: Option<u64> = None;
    let mut num_updates_arg: Option<u64> = None;
    let mut single_mode = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-t" | "--table-size" => {
                i += 1;
                if i < args.len() {
                    table_size_log = Some(args[i].parse().unwrap_or_else(|_| {
                        eprintln!("Invalid table size: {}", args[i]);
                        process::exit(1);
                    }));
                }
            }
            "-u" | "--updates" => {
                i += 1;
                if i < args.len() {
                    num_updates_arg = Some(args[i].parse().unwrap_or_else(|_| {
                        eprintln!("Invalid update count: {}", args[i]);
                        process::exit(1);
                    }));
                }
            }
            "--single" => {
                single_mode = true;
            }
            "-h" | "--help" => {
                print_usage();
                process::exit(0);
            }
            other => {
                eprintln!("Unknown option: {}", other);
                print_usage();
                process::exit(1);
            }
        }
        i += 1;
    }

    if single_mode {
        run_single(table_size_log, num_updates_arg);
    } else {
        // If PMIX_RANK env var is set, we're under prterun — use multi-process mode.
        // create_multiprocess() gets rank/size from PMIx directly.
        let under_pmix = env::var("PMIX_RANK").is_ok();
        if under_pmix {
            run_multi(table_size_log, num_updates_arg);
        } else {
            // No PMIx env vars detected — fall back to single-node mode
            run_single(table_size_log, num_updates_arg);
        }
    }
}
