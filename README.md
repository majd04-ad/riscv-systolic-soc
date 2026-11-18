# Project 418: Ibex RISC-V + Systolic Array Accelerator

An FPGA platform that pairs the 32-bit [lowRISC Ibex](https://github.com/lowRISC/ibex) RISC-V core with a custom 2D systolic array accelerator over a memory-mapped bus. The CPU offloads dense matrix multiplication (GEMM) and im2col-transformed convolutions to the array.

Final engineering project by Majd and Jobran.

## Status

Early setup. First target is a 4x4 INT8 weight-stationary array (INT32 accumulation), parameterized so it can grow to 8x8. The FPGA board is not chosen yet, so RTL stays vendor-neutral and board files live only under `fpga/`.

## Repository layout

| Path                | Contents                                                                           |
| ------------------- | ---------------------------------------------------------------------------------- |
| `rtl/accel/`        | Systolic array: PE, NxN array, input skew, control FSM, tiling                     |
| `rtl/soc/`          | SoC top: Ibex wrapper, bus interconnect, memories, accelerator register block, DMA |
| `rtl/common/`       | Shared packages and small utility modules                                          |
| `sw/lib/`           | Bare-metal drivers and runtime (accelerator driver, startup, linker script)        |
| `sw/apps/`          | Test programs and benchmarks (GEMM, im2col convolution)                            |
| `sim/tb/`           | Verilator testbenches                                                              |
| `sim/golden/`       | Python golden models used to check RTL results                                     |
| `fpga/`             | Board-specific constraints and build scripts (added once a board is picked)        |
| `docs/`             | Design notes, register map, reports                                                |
| `third_party/ibex/` | Ibex core, git submodule pinned to tag `v2.1.0`                                    |

## Getting started

Clone with the Ibex submodule:

```sh
git clone --recurse-submodules <repo-url>
# or, in an existing clone:
git submodule update --init --recursive
```

Tools we expect to use:

- [Verilator](https://www.veripool.org/verilator/) 5.x for simulation
- [FuseSoC](https://github.com/olofk/fusesoc) (Ibex's build system), installed with `pip install -r third_party/ibex/python-requirements.txt`
- A RISC-V GCC toolchain (`riscv32-unknown-elf-gcc`, RV32IMC) for software
- The FPGA vendor toolchain for the chosen board

## Updating Ibex

Ibex is pinned to `v2.1.0`. To move to another tag:

```sh
cd third_party/ibex
git fetch --tags
git checkout <tag>
cd ../..
git add third_party/ibex
git commit -m "Bump Ibex to <tag>"
```

## License

Ibex is licensed under Apache 2.0 (see `third_party/ibex/LICENSE`). The license for this project's own code is MIT.