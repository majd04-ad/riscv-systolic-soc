# Software

Reference C implementations of the workloads the accelerator runs, and
tests for them. Everything is plain C99 with no heap and no libc in the
library itself, so the same code builds for the host now and for Ibex
bare-metal later.

| Path | Contents |
| --- | --- |
| `lib/include/accel_config.h` | `ACCEL_DIM`, the array size (4, or 8 later) |
| `lib/include/gemm.h`, `lib/src/gemm.c` | INT8 GEMM with INT32 accumulation: plain baseline and a version tiled for the array |
| `lib/include/conv.h`, `lib/src/conv.c` | im2col, direct convolution, and convolution as im2col + GEMM |
| `lib/include/requant.h`, `lib/src/requant.c` | INT32 to INT8 requantization between layers |
| `tests/` | Self-checking tests and a minimal harness with no printf or malloc |

## Running the tests

```sh
make -C sw test                 # 4x4 array
make -C sw test ACCEL_DIM=8     # 8x8 array
make -C sw test-sanitize        # with AddressSanitizer and UBSan
```

The runner prints `PASSED`/`FAILED` and exits non-zero on failure. CI runs
the same targets for `ACCEL_DIM` 4 and 8.

## Conventions

- **GEMM:** `C (M x N) = A (M x K) * B (K x N)`, row-major with explicit row
  strides. B holds the weights, which stay in the array (weight-stationary).
- **Tensors:** HWC for activations, HWIO for weights. With these layouts the
  weights are already the `K x out_c` B matrix (`K = k_h * k_w * in_c`), and
  im2col turns the input into a `(out_h * out_w) x K` patch matrix, one row
  per output pixel. The GEMM output is HWC again, so layers chain directly.
- **Padding** is zero padding, which matches symmetric INT8 quantization
  (zero point 0).
- **Requantization:** `clamp(round(x * multiplier / 2^shift) + zero_point)`,
  rounding to nearest with ties away from zero, optional ReLU. The hardware
  requantization stage should match this bit for bit.

## Tiling and the hardware hook

`gemm_i8_i32_tiled` runs the matrix in the order the array will: for each
`ACCEL_DIM`-wide column block of B, for each `ACCEL_DIM`-tall row block,
load one weight tile and stream every row of A through it, accumulating
partial sums in C. Edge tiles are zero-padded.

Each pass goes through a `gemm_tile_fn` callback. Today that is
`gemm_tile_sw`, a software model of one array pass. The accelerator driver
will provide a callback with the same signature that writes the tile and
the rows of A to the accelerator and reads back C, so the tiling code and
these tests carry over unchanged. `tests/test_gemm.c` checks the contract
such a callback can rely on (tile sizes, zero padding, call count).

## Running on Ibex later

Only `tests/platform_host.c` touches stdio; it defines `test_putc`. For a
bare-metal build, add a platform file that writes `test_putc` to the UART
or the simulator's output device, and link it with startup code and a
linker script in place of the host one.