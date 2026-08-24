# gups-rs

Rust port of the **GUPS** (Giga Updates Per Second / Global Update Performance) benchmark, using **UCX** for multi-process RMA atomics and **PMIx** for process management.

Depends on path crates: `../ucx-rs`, `../pmix-rs`, `../ucc-rs`.

## Modes

| Mode | Command | Notes |
|------|---------|--------|
| Single-process | `gups-rs --single` | No UCX traffic; validates LFSR + table updates locally |
| Multi-process | `prterun -np N ./gups-rs` | UCX RMA atomic XOR; needs RDMA-capable TLS for RMA |

**Caveat:** `UCX_TLS=tcp` typically has **no RMA**. Use a fabric TLS (rc/dc/ud as appropriate) for multi-process RMA.

## Build

```bash
export PMIX_PREFIX=/path/to/pmix-or-prrte
export UCX_PREFIX=/path/to/ucx
export UCC_PREFIX=/path/to/ucc
export LD_LIBRARY_PATH=$PMIX_PREFIX/lib:$UCX_PREFIX/lib:$UCC_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

cargo build --release
```

UCC is an optional feature enabled by default. For single-process mode without
 a UCC installation, build with `cargo build --no-default-features`.

See the [Build](#build) section above for build prerequisites and commands.

## Usage

```text
gups-rs [options]
  -t, --table-size SIZE    Table size as power of 2 (default: auto, half of RAM)
  -u, --updates COUNT      Number of updates (default: 4x table size)
  --single                 Single-process mode (no UCX)
  -h, --help               Help
```

```bash
# Algorithm smoke
./target/release/gups-rs --single -t 20 -u 1000000

# Multi-process (example)
prterun -np 2 ./target/release/gups-rs -t 24
```

## Layout

- `src/main.rs` — CLI + orchestration
- `src/rng.rs` — LFSR
- `src/table.rs` — table + updates
- `src/comm.rs` — multi-process UCX path
- `src/verify.rs` — verification

## License

BSD-style (see `LICENSE`). See [`REVIEW.md`](./REVIEW.md).
