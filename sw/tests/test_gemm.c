#include "gemm.h"
#include "test_util.h"
#include "tests.h"

#define MAX_DIM 24

static int8_t a_buf[MAX_DIM * MAX_DIM];
static int8_t b_buf[MAX_DIM * MAX_DIM];
static int32_t c_ref[MAX_DIM * MAX_DIM];
static int32_t c_out[MAX_DIM * MAX_DIM];

static void fill_rand(int8_t *dst, size_t count)
{
    size_t i;

    for (i = 0; i < count; i++)
        dst[i] = test_rand_i8();
}

static void test_hand_example(void)
{
    static const int8_t a[2 * 3] = { 1, -2, 3,
                                    -4, 5, -6 };
    static const int8_t b[3 * 2] = { 7, -8,
                                     9, 10,
                                   -11, 12 };
    static const int32_t want[2 * 2] = { -44, 8,
                                          83, 10 };
    int32_t c[2 * 2];

    gemm_i8_i32(2, 2, 3, a, 3, b, 2, c, 2);
    test_check_i32("gemm hand example", c, want, 4);

    gemm_i8_i32_tiled(2, 2, 3, a, 3, b, 2, c, 2, NULL);
    test_check_i32("tiled gemm hand example", c, want, 4);
}

/* largest magnitude products must not wrap in the INT32 accumulator*/
static void test_extremes(void)
{
    enum { K = 256 };
    static int8_t a[K], b[K];
    int32_t c;
    size_t i;

    for (i = 0; i < K; i++) {
        a[i] = -128;
        b[i] = -128;
    }
    gemm_i8_i32(1, 1, K, a, K, b, 1, &c, 1);
    CHECK(c == (int32_t)K * 16384);
    gemm_i8_i32_tiled(1, 1, K, a, K, b, 1, &c, 1, NULL);
    CHECK(c == (int32_t)K * 16384);

    for (i = 0; i < K; i++)
        b[i] = 127;
    gemm_i8_i32(1, 1, K, a, K, b, 1, &c, 1);
    CHECK(c == (int32_t)K * -16256);
    gemm_i8_i32_tiled(1, 1, K, a, K, b, 1, &c, 1, NULL);
    CHECK(c == (int32_t)K * -16256);
}

/* weights: C must equal A */
static void test_identity(void)
{
    enum { M = 7, N = 10 };
    size_t i, j;

    test_seed(7);
    fill_rand(a_buf, M * N);
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            b_buf[i * N + j] = (i == j);
    for (i = 0; i < M * N; i++)
        c_ref[i] = a_buf[i];

    gemm_i8_i32_tiled(M, N, N, a_buf, N, b_buf, N, c_out, N, NULL);
    test_check_i32("tiled gemm with identity", c_out, c_ref, M * N);
}

/* Tiled GEMM must equal the plain one for every shape including edges */
static void test_tiled_matches_plain(void)
{
    size_t m, n, k;

    test_seed(1);
    for (m = 1; m <= 13; m++) {
        for (n = 1; n <= 13; n++) {
            for (k = 1; k <= 13; k++) {
                fill_rand(a_buf, m * k);
                fill_rand(b_buf, k * n);
                gemm_i8_i32(m, n, k, a_buf, k, b_buf, n, c_ref, n);
                gemm_i8_i32_tiled(m, n, k, a_buf, k, b_buf, n, c_out, n, NULL);
                if (!test_check_i32("tiled vs plain gemm", c_out, c_ref, m * n))
                    return;
            }
        }
    }
}

/* Sub matrices of larger buffers strides larger than the logical width */
static void test_strides(void)
{
    enum { M = 5, N = 6, K = 9, LDA = 17, LDB = 11, LDC = 13 };
    static int32_t c1[M * LDC], c2[M * LDC];
    size_t i, j;

    test_seed(3);
    fill_rand(a_buf, M * LDA);
    fill_rand(b_buf, K * LDB);
    for (i = 0; i < M * LDC; i++)
        c1[i] = c2[i] = 0x5a5a5a5a;

    gemm_i8_i32(M, N, K, a_buf, LDA, b_buf, LDB, c1, LDC);
    gemm_i8_i32_tiled(M, N, K, a_buf, LDA, b_buf, LDB, c2, LDC, NULL);
    test_check_i32("strided gemm", c2, c1, M * LDC);

    /* Elements between rows of C must be left alone. */
    for (i = 0; i < M; i++)
        for (j = N; j < LDC; j++)
            CHECK(c1[i * LDC + j] == 0x5a5a5a5a);
}

/*
 * A tile kernel that checks the contract a hardware kernel relies on:
 * kv and nv never exceed ACCEL_DIM and the weight tile is zero outside
 * the valid region It counts calls then runs the software kernel.
 */
static unsigned tile_calls;
static unsigned tile_bad;

static void checking_tile(const int8_t w[ACCEL_DIM][ACCEL_DIM],
                          const int8_t *a, size_t lda,
                          int32_t *c, size_t ldc,
                          size_t m, size_t kv, size_t nv)
{
    size_t p, j;

    tile_calls++;
    if (kv == 0 || kv > ACCEL_DIM || nv == 0 || nv > ACCEL_DIM)
        tile_bad++;
    for (p = 0; p < ACCEL_DIM; p++)
        for (j = 0; j < ACCEL_DIM; j++)
            if ((p >= kv || j >= nv) && w[p][j] != 0)
                tile_bad++;
    gemm_tile_sw(w, a, lda, c, ldc, m, kv, nv);
}

static void test_tile_contract(void)
{
    enum { M = 9, N = 2 * ACCEL_DIM + 1, K = 3 * ACCEL_DIM - 1 };
    unsigned want_calls = ((N + ACCEL_DIM - 1) / ACCEL_DIM) *
                          ((K + ACCEL_DIM - 1) / ACCEL_DIM);

    test_seed(11);
    fill_rand(a_buf, M * K);
    fill_rand(b_buf, K * N);
    tile_calls = 0;
    tile_bad = 0;

    gemm_i8_i32(M, N, K, a_buf, K, b_buf, N, c_ref, N);
    gemm_i8_i32_tiled(M, N, K, a_buf, K, b_buf, N, c_out, N, checking_tile);
    test_check_i32("gemm with checking tile kernel", c_out, c_ref, M * N);
    CHECK(tile_calls == want_calls);
    CHECK(tile_bad == 0);
}

void test_gemm(void)
{
    test_hand_example();
    test_extremes();
    test_identity();
    test_tiled_matches_plain();
    test_strides();
    test_tile_contract();
}