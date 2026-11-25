/*
 * CPU-only baseline: GEMM and im2col + GEMM convolution on the RISC-V core,
 * no accelerator. Each kernel is timed with the cycle and retired
 * instruction counters, per phase, and its output is reduced to a hash.
 *
 * The same program builds for the host (platform_host.c), where the
 * counters read 0 and only the hashes matter: they are the reference the
 * Ibex run is checked against (compare.py). On the target it also checks
 * itself: plain GEMM against the tiled GEMM, and im2col + GEMM against
 * direct convolution.
 *
 * Output, one line per measurement:
 *   BENCH <kind> <name> <phase> cycles=<n> instret=<n> macs=<n> hash=<hex>
 */
#include <stddef.h>
#include <stdint.h>

#include "bench_platform.h"
#include "conv.h"
#include "gemm.h"
#include "test_util.h"

typedef struct {
    const char *name;
    size_t m, n, k;
} gemm_case;

typedef struct {
    const char *name;
    conv2d_params p;
} conv_case;

static const gemm_case gemm_cases[] = {
    {"8x8x8", 8, 8, 8},
    {"16x16x16", 16, 16, 16},
    {"32x32x32", 32, 32, 32},
    {"64x64x64", 64, 64, 64},
};

/* in_h, in_w, in_c, out_c, k_h, k_w, stride_h, stride_w, pad_h, pad_w */
static const conv_case conv_cases[] = {
    {"16x16x3_3x3_c8", {16, 16, 3, 8, 3, 3, 1, 1, 1, 1}},
    {"16x16x8_3x3_c16", {16, 16, 8, 16, 3, 3, 1, 1, 1, 1}},
    {"32x32x3_3x3_c16", {32, 32, 3, 16, 3, 3, 1, 1, 1, 1}},
    {"8x8x16_3x3_c32", {8, 8, 16, 32, 3, 3, 1, 1, 1, 1}},
    {"16x16x16_3x3_c32_s2", {16, 16, 16, 32, 3, 3, 2, 2, 1, 1}},
};

#define NUM_ELEMS(a) (sizeof(a) / sizeof((a)[0]))

/* Sized for the largest case above (checked at run time). */
#define MAX_A    (64 * 64)
#define MAX_B    (64 * 64)
#define MAX_C    (64 * 64)
#define MAX_IN   (16 * 16 * 16)
#define MAX_W    (3 * 3 * 16 * 32)
#define MAX_COL  (32 * 32 * 27)
#define MAX_OUT  (32 * 32 * 16)

static int8_t buf_a[MAX_A];
static int8_t buf_b[MAX_B];
static int32_t buf_c[MAX_C];
static int8_t buf_in[MAX_IN];
static int8_t buf_w[MAX_W];
static int8_t buf_col[MAX_COL];
static int32_t buf_out[MAX_OUT];

static unsigned failures;

/* Counter readings to subtract: the cost of an empty measurement. */
static uint32_t overhead_cycles, overhead_instret;

typedef struct {
    uint32_t cycles, instret;
} sample;

static sample sample_now(void)
{
    sample s;

    s.instret = bench_instret();
    s.cycles = bench_cycles();
    return s;
}

static sample sample_since(sample start)
{
    sample end = sample_now(), d;

    d.cycles = end.cycles - start.cycles - overhead_cycles;
    d.instret = end.instret - start.instret - overhead_instret;
    return d;
}

static void calibrate(void)
{
    sample s;

    overhead_cycles = overhead_instret = 0;
    s = sample_since(sample_now());
    overhead_cycles = s.cycles;
    overhead_instret = s.instret;
}

/* FNV-1a over the bytes of the buffer (both platforms are little-endian). */
static uint32_t hash_bytes(const void *data, size_t len)
{
    const uint8_t *p = (const uint8_t *)data;
    uint32_t h = 2166136261u;
    size_t i;

    for (i = 0; i < len; i++) {
        h ^= p[i];
        h *= 16777619u;
    }
    return h;
}

static void put_u32(uint32_t v)
{
    char buf[10];
    int i = 0;

    do {
        buf[i++] = (char)('0' + v % 10);
        v /= 10;
    } while (v != 0);
    while (i > 0)
        test_putc(buf[--i]);
}

static void put_hex(uint32_t v)
{
    int i;

    for (i = 28; i >= 0; i -= 4)
        test_putc("0123456789abcdef"[(v >> i) & 0xf]);
}

static void report(const char *kind, const char *name, const char *phase,
                   sample s, uint32_t macs, uint32_t hash)
{
    test_puts("BENCH ");
    test_puts(kind);
    test_putc(' ');
    test_puts(name);
    test_putc(' ');
    test_puts(phase);
    test_puts(" cycles=");
    put_u32(s.cycles);
    test_puts(" instret=");
    put_u32(s.instret);
    test_puts(" macs=");
    put_u32(macs);
    test_puts(" hash=");
    put_hex(hash);
    test_putc('\n');
}

static void mismatch(const char *name, const char *what)
{
    failures++;
    test_puts("MISMATCH ");
    test_puts(name);
    test_puts(": ");
    test_puts(what);
    test_putc('\n');
}

static void fill_i8(int8_t *dst, size_t count)
{
    size_t i;

    for (i = 0; i < count; i++)
        dst[i] = test_rand_i8();
}

static void bench_gemm(const gemm_case *gc)
{
    size_t m = gc->m, n = gc->n, k = gc->k;
    uint32_t macs = (uint32_t)(m * n * k);
    uint32_t h_plain, h_tiled;
    sample s;

    if (m * k > MAX_A || k * n > MAX_B || m * n > MAX_C) {
        mismatch(gc->name, "case too large for buffers");
        return;
    }

    fill_i8(buf_a, m * k);
    fill_i8(buf_b, k * n);

    s = sample_now();
    gemm_i8_i32(m, n, k, buf_a, k, buf_b, n, buf_c, n);
    s = sample_since(s);
    h_plain = hash_bytes(buf_c, m * n * sizeof(int32_t));
    report("gemm", gc->name, "plain", s, macs, h_plain);

    s = sample_now();
    gemm_i8_i32_tiled(m, n, k, buf_a, k, buf_b, n, buf_c, n, gemm_tile_sw);
    s = sample_since(s);
    h_tiled = hash_bytes(buf_c, m * n * sizeof(int32_t));
    report("gemm", gc->name, "tiled", s, macs, h_tiled);

    if (h_plain != h_tiled)
        mismatch(gc->name, "tiled GEMM differs from plain GEMM");
}

static void bench_conv(const conv_case *cc)
{
    const conv2d_params *p = &cc->p;
    size_t in_n = p->in_h * p->in_w * p->in_c;
    size_t w_n = p->k_h * p->k_w * p->in_c * p->out_c;
    size_t rows = im2col_rows(p), k = im2col_cols(p);
    size_t out_n = rows * p->out_c;
    uint32_t macs = (uint32_t)(rows * k * p->out_c);
    uint32_t h_col, h_out, h_direct;
    sample s_col, s_gemm, s_total, s;

    if (in_n > MAX_IN || w_n > MAX_W || rows * k > MAX_COL || out_n > MAX_OUT) {
        mismatch(cc->name, "case too large for buffers");
        return;
    }

    fill_i8(buf_in, in_n);
    fill_i8(buf_w, w_n);

    /* The two phases of conv2d_im2col_i8_i32 (no bias), timed separately. */
    s = sample_now();
    im2col_hwc_i8(p, buf_in, buf_col);
    s_col = sample_since(s);

    s = sample_now();
    gemm_i8_i32(rows, p->out_c, k, buf_col, k, buf_w, p->out_c, buf_out, p->out_c);
    s_gemm = sample_since(s);

    s_total.cycles = s_col.cycles + s_gemm.cycles;
    s_total.instret = s_col.instret + s_gemm.instret;
    h_col = hash_bytes(buf_col, rows * k);
    h_out = hash_bytes(buf_out, out_n * sizeof(int32_t));
    report("conv", cc->name, "im2col", s_col, 0, h_col);
    report("conv", cc->name, "gemm", s_gemm, macs, h_out);
    report("conv", cc->name, "total", s_total, macs, h_out);

    s = sample_now();
    conv2d_direct_i8_i32(p, buf_in, buf_w, NULL, buf_out);
    s = sample_since(s);
    h_direct = hash_bytes(buf_out, out_n * sizeof(int32_t));
    report("conv", cc->name, "direct", s, macs, h_direct);

    if (h_out != h_direct)
        mismatch(cc->name, "im2col + GEMM differs from direct convolution");
}

int main(void)
{
    size_t i;

    bench_init();
    calibrate();
    test_seed(418);

    test_puts("bench_cpu ACCEL_DIM=");
    test_put_int(ACCEL_DIM);
    test_puts(" overhead_cycles=");
    put_u32(overhead_cycles);
    test_puts(" overhead_instret=");
    put_u32(overhead_instret);
    test_putc('\n');

    for (i = 0; i < NUM_ELEMS(gemm_cases); i++)
        bench_gemm(&gemm_cases[i]);
    for (i = 0; i < NUM_ELEMS(conv_cases); i++)
        bench_conv(&conv_cases[i]);

    test_puts(failures ? "FAILED: " : "PASSED: ");
    put_u32(failures);
    test_puts(" mismatches\n");
    return failures ? 1 : 0;
}