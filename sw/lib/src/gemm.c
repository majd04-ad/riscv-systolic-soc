#include "gemm.h"
void gemm_i8_i32(size_t m, size_t n, size_t k,
                 const int8_t *a, size_t lda,
                 const int8_t *b, size_t ldb,
                 int32_t *c, size_t ldc)
{
    size_t i, j, p;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            int32_t acc = 0;
            for (p = 0; p < k; p++)
                acc += (int32_t)a[i * lda + p] * (int32_t)b[p * ldb + j];
            c[i * ldc + j] = acc;
        }
    }
}

void gemm_tile_sw(const int8_t w[ACCEL_DIM][ACCEL_DIM],
                  const int8_t *a, size_t lda,
                  int32_t *c, size_t ldc,
                  size_t m, size_t kv, size_t nv)
{
    size_t i, j, p;

    for (i = 0; i < m; i++) {
        for (j = 0; j < nv; j++) {
            int32_t acc = c[i * ldc + j];
            for (p = 0; p < kv; p++)
                acc += (int32_t)a[i * lda + p] * (int32_t)w[p][j];
            c[i * ldc + j] = acc;
        }
    }
}

void gemm_i8_i32_tiled(size_t m, size_t n, size_t k,
                       const int8_t *a, size_t lda,
                       const int8_t *b, size_t ldb,
                       int32_t *c, size_t ldc,
                       gemm_tile_fn tile)
{
    int8_t w[ACCEL_DIM][ACCEL_DIM];
    size_t n0, k0, i, j, p, nv, kv;

    if (tile == NULL)
        tile = gemm_tile_sw;

    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            c[i * ldc + j] = 0;

    for (n0 = 0; n0 < n; n0 += ACCEL_DIM) {
        nv = (n - n0 < ACCEL_DIM) ? n - n0 : ACCEL_DIM;
        for (k0 = 0; k0 < k; k0 += ACCEL_DIM) {
            kv = (k - k0 < ACCEL_DIM) ? k - k0 : ACCEL_DIM;

            /* Load the weight tile zeropadding past the matrix edge*/
            for (p = 0; p < ACCEL_DIM; p++)
                for (j = 0; j < ACCEL_DIM; j++)
                    w[p][j] = (p < kv && j < nv) ? b[(k0 + p) * ldb + n0 + j] : 0;

            tile((const int8_t (*)[ACCEL_DIM])w, a + k0, lda, c + n0, ldc,
                 m, kv, nv);
        }
    }
}