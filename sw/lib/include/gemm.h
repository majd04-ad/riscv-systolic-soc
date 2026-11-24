/*
INT8 GEMM with INT32 accumulation.
All matrices are row-major.
so a sub-matrix of a larger buffer can be passed directly.
C (M x N) = A (M x K) * B (K x N)
In the accelerator's weight-stationary dataflow B holds the weights: a
DIM x DIM tile of B is loaded into the array once, and the rows of A
stream through it.
*/
#ifndef GEMM_H
#define GEMM_H

#include <stddef.h>
#include <stdint.h>

#include "accel_config.h"

/* This is the CPU-only baseline and the reference result. */
void gemm_i8_i32(size_t m, size_t n, size_t k,
                 const int8_t *a, size_t lda,
                 const int8_t *b, size_t ldb,
                 int32_t *c, size_t ldc);

/*
stationary pass of a DIM x DIM array.
w is a full ACCEL_DIM x ACCEL_DIM weight tile, zero-padded when the
matrix edge leaves fewer than ACCEL_DIM valid rows or columns.
For each of the m rows of A the kernel reads kv (<= ACCEL_DIM) inputs
and adds the products into nv (<= ACCEL_DIM) outputs of C:
c[i][j] += sum_{p < kv} a[i][p] * w[p][j]     for i < m, j < nv
 */
typedef void (*gemm_tile_fn)(const int8_t w[ACCEL_DIM][ACCEL_DIM],
                             const int8_t *a, size_t lda,
                             int32_t *c, size_t ldc,
                             size_t m, size_t kv, size_t nv);

/* Software model of one array pass; the default tile kernel. */
void gemm_tile_sw(const int8_t w[ACCEL_DIM][ACCEL_DIM],
                  const int8_t *a, size_t lda,
                  int32_t *c, size_t ldc,
                  size_t m, size_t kv, size_t nv);

/*
GEMM split into ACCEL_DIM x ACCEL_DIM weight tiles, in the order the
hardware will run them: for each column block of B, for each row block
of B, load the weight tile and stream all M rows of A through it.
Partial sums for one output column block accumulate across the K blocks.
tile is the kernel that runs one pass. Pass NULL to use gemm_tile_sw;
the accelerator driver will later pass a function that runs the pass
on the hardware instead.
 */
void gemm_i8_i32_tiled(size_t m, size_t n, size_t k,
                       const int8_t *a, size_t lda,
                       const int8_t *b, size_t ldb,
                       int32_t *c, size_t ldc,
                       gemm_tile_fn tile);

#endif