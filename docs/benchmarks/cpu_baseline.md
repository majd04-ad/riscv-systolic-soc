# CPU-only baseline: GEMM and im2col convolution on Ibex

First benchmark of the project: the reference kernels from `sw/lib` run on
the Ibex core alone, with no accelerator. These are the numbers the systolic
array has to beat.

- **Program:** [`sw/apps/bench_cpu`](../../sw/apps/bench_cpu), INT8 inputs,
  INT32 accumulation, random data from a fixed seed (418).
- **Platform:** Ibex v2.1.0 [Simple System](../../third_party/ibex/examples/simple_system)
  in Verilator 5.020. Single-cycle SRAM for code and data (1 MB), no caches.
- **Compiler:** GCC 13.2 (`riscv64-unknown-elf-gcc`), `-march=rv32imc_zicsr
  -mabi=ilp32 -O2 -fno-tree-loop-distribute-patterns`.
- **Measurement:** `mcycle` and `minstret` read around each phase, minus the
  cost of an empty measurement (18 cycles, 8 instructions on `small`).
- **Check:** every output (im2col patch matrix and conv/GEMM result) is
  hashed and must match the same program built and run on the host.
  On target, plain GEMM must also equal the tiled GEMM, and im2col + GEMM
  must equal direct convolution. All 28 outputs matched in both runs.

Two Ibex configurations from `ibex_configs.yaml`:

| Config | Multiplier | Other |
| --- | --- | --- |
| `small` | `RV32MFast` (multi-cycle) | 2-stage pipeline, no branch target ALU |
| `maxperf` | `RV32MSingleCycle` | 3-stage pipeline (writeback stage), branch target ALU |

`small` is the default Ibex configuration and the likely starting point on
the FPGA.

## GEMM: C (M x N) = A (M x K) * B (K x N)

"plain" is `gemm_i8_i32`, the triple loop. "tiled" is `gemm_i8_i32_tiled`
with `gemm_tile_sw`, the same 4x4 weight-tile order the array will use,
done in software. Its extra cost is the tile copy and loop overhead.

**small**

| GEMM (M x N x K) | MACs | plain cycles | cycles/MAC | CPI | tiled (4x4 SW) cycles | tiled / plain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 x 8 x 8 | 512 | 7,345 | 14.35 | 1.75 | 10,076 | 1.37 |
| 16 x 16 x 16 | 4,096 | 55,737 | 13.61 | 1.80 | 72,726 | 1.30 |
| 32 x 32 x 32 | 32,768 | 435,529 | 13.29 | 1.83 | 553,418 | 1.27 |
| 64 x 64 x 64 | 262,144 | 3,445,353 | 13.14 | 1.84 | 4,318,002 | 1.25 |

**maxperf**

| GEMM (M x N x K) | MACs | plain cycles | cycles/MAC | CPI | tiled (4x4 SW) cycles | tiled / plain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 x 8 x 8 | 512 | 4,712 | 9.20 | 1.12 | 6,851 | 1.45 |
| 16 x 16 x 16 | 4,096 | 34,992 | 8.54 | 1.13 | 48,701 | 1.39 |
| 32 x 32 x 32 | 32,768 | 270,656 | 8.26 | 1.14 | 367,385 | 1.36 |
| 64 x 64 x 64 | 262,144 | 2,130,528 | 8.13 | 1.14 | 2,853,233 | 1.34 |

## Convolution as im2col + GEMM

All layers are 3x3, zero padding 1, HWC input, HWIO weights, no bias. Names
are `<in_h>x<in_w>x<in_c>_<kernel>_c<out_c>[_s<stride>]`. im2col and GEMM are
timed separately; total is their sum. "direct" is `conv2d_direct_i8_i32` on
the same data, for comparison.

**small**

| Layer | MACs | im2col cycles | GEMM cycles | total cycles | im2col share | cycles/MAC | direct conv cycles |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16x16x3_3x3_c8 | 55,296 | 189,389 | 739,630 | 929,019 | 20.4% | 16.80 | 1,370,690 |
| 16x16x8_3x3_c16 | 294,912 | 351,609 | 3,873,070 | 4,224,679 | 8.3% | 14.33 | 5,100,178 |
| 32x32x3_3x3_c16 | 442,368 | 761,085 | 5,907,502 | 6,668,587 | 11.4% | 15.07 | 11,325,890 |
| 8x8x16_3x3_c32 | 294,912 | 152,993 | 3,852,910 | 4,005,903 | 3.8% | 13.58 | 4,082,778 |
| 16x16x16_3x3_c32_s2 | 294,912 | 152,848 | 3,852,910 | 4,005,758 | 3.8% | 13.58 | 4,444,058 |

**maxperf**

| Layer | MACs | im2col cycles | GEMM cycles | total cycles | im2col share | cycles/MAC | direct conv cycles |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16x16x3_3x3_c8 | 55,296 | 148,362 | 461,090 | 609,452 | 24.3% | 11.02 | 947,948 |
| 16x16x8_3x3_c16 | 294,912 | 286,602 | 2,394,402 | 2,681,004 | 10.7% | 9.09 | 3,411,564 |
| 32x32x3_3x3_c16 | 442,368 | 596,106 | 3,679,266 | 4,275,372 | 13.9% | 9.66 | 7,826,236 |
| 8x8x16_3x3_c32 | 294,912 | 126,666 | 2,376,290 | 2,502,956 | 5.1% | 8.49 | 2,687,524 |
| 16x16x16_3x3_c32_s2 | 294,912 | 126,989 | 2,376,290 | 2,503,279 | 5.1% | 8.49 | 2,922,884 |

The last two layers lower to the same 64 x 144 x 32 GEMM, so their GEMM
cycles are identical.

## What the numbers say

- **The GEMM inner loop costs about 13 cycles per MAC on `small`** and about 8
  on `maxperf`. It is 7 instructions (two `lb`, `mul`, three adds, a taken
  branch); load-use stalls, the multi-cycle multiplier and the taken-branch
  penalty bring it to ~1.8 CPI on `small`. At an assumed 50 MHz, the 64^3
  GEMM takes ~69 ms on `small`.
- **GEMM dominates the convolutions:** 76 to 96% of the cycles. im2col is
  4 to 24%, and is largest for few input channels (`in_c = 3`), where each
  output pixel copies 27 bytes one at a time with bounds checks but does
  only 27 x out_c MACs. Once the array takes over the GEMM, im2col will be
  most of what is left on the CPU, so it is the next CPU-side target
  (or a hardware im2col / address generator).
- **im2col + GEMM already beats direct convolution on the CPU** (1.0x to
  1.8x faster), because direct convolution recomputes the padding checks
  and addresses inside the MAC loop.
- **Software tiling costs 25 to 45% over the plain loop.** That is the price
  of running the array's 4x4 tile order on the CPU; it disappears once the
  tile callback drives the hardware.
- **Headroom for the accelerator:** a 4x4 array at one MAC per PE per cycle
  peaks at 16 MACs/cycle, about 200x the `small` core's ~0.076 MACs/cycle,
  before counting data movement. In practice the speedup will be set by how
  fast A rows and tiles reach the array (bus, buffers, DMA).

Caveats: the Simple System's memory is single-cycle with no wait states, so
an FPGA build with slower memory or a bus in the path will be somewhat
slower. Code is `-O2` with plain C; no hand-tuned assembly or loop
unrolling.

## Reproducing

```sh
# Tools: Verilator >= 4.210, a RISC-V GCC with rv32imc/ilp32, libelf-dev,
# srecord, and FuseSoC/edalize from third_party/ibex/python-requirements.txt
# (the lowRISC forks; plus packaging and hjson).

# 1. Build the Simple System simulator (from third_party/ibex).
cd third_party/ibex
fusesoc --cores-root=. run --target=sim --setup --build \
    --build-root=../../build/ibex-small lowrisc:ibex:ibex_simple_system \
    $(python3 util/ibex_config.py small fusesoc_opts)
cd ../..

# 2. Build the benchmark, run it on the host and in the simulator, compare.
make -C sw/apps/bench_cpu run CONFIG=small \
    SIM=$PWD/build/ibex-small/sim-verilator/Vibex_simple_system
```

Use `maxperf` in place of `small` for the other configuration. A run takes
about 3.5 minutes (~60 M simulated cycles at ~280 kHz). Raw logs from the
runs above: [`cpu_baseline_small.txt`](cpu_baseline_small.txt),
[`cpu_baseline_maxperf.txt`](cpu_baseline_maxperf.txt), and the host
reference [`cpu_baseline_host.txt`](cpu_baseline_host.txt).