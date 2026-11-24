#include "conv.h"
#include "gemm.h"
#include "requant.h"
#include "test_util.h"
#include "tests.h"

/* Big enough random sweep and  16x16x3 layer below. */
#define MAX_IN   (16 * 16 * 8)
#define MAX_W    (4 * 4 * 8 * 16)
#define MAX_COL  (16 * 16 * 3 * 3 * 8)
#define MAX_OUT  (16 * 16 * 16)

static int8_t in_buf[MAX_IN];
static int8_t w_buf[MAX_W];
static int32_t bias_buf[16];
static int8_t col_buf[MAX_COL];
static int32_t out_ref[MAX_OUT];
static int32_t out_gemm[MAX_OUT];
static int32_t out_tiled[MAX_OUT];

static conv2d_params make_params(size_t in_h, size_t in_w, size_t in_c,
                                 size_t out_c, size_t k_h, size_t k_w,
                                 size_t stride, size_t pad)
{
    conv2d_params p;

    p.in_h = in_h;
    p.in_w = in_w;
    p.in_c = in_c;
    p.out_c = out_c;
    p.k_h = k_h;
    p.k_w = k_w;
    p.stride_h = stride;
    p.stride_w = stride;
    p.pad_h = pad;
    p.pad_w = pad;
    return p;
}

static void fill_rand(int8_t *dst, size_t count)
{
    size_t i;

    for (i = 0; i < count; i++)
        dst[i] = test_rand_i8();
}

static void test_out_dims(void)
{
    CHECK(conv2d_out_dim(5, 3, 1, 0) == 3);
    CHECK(conv2d_out_dim(5, 3, 1, 1) == 5);
    CHECK(conv2d_out_dim(5, 3, 2, 1) == 3);
    CHECK(conv2d_out_dim(6, 3, 2, 0) == 2);
    CHECK(conv2d_out_dim(3, 3, 1, 0) == 1);
    CHECK(conv2d_out_dim(2, 3, 1, 0) == 0);
    CHECK(conv2d_out_dim(2, 3, 1, 1) == 2);
    CHECK(conv2d_out_dim(5, 3, 0, 0) == 0);
}

/* 3x3 single image  2x2 window no padding. */
static void test_im2col_basic(void)
{
    static const int8_t in[9] = { 1, 2, 3,
                                  4, 5, 6,
                                  7, 8, 9 };
    static const int8_t want[4 * 4] = { 1, 2, 4, 5,
                                        2, 3, 5, 6,
                                        4, 5, 7, 8,
                                        5, 6, 8, 9 };
    conv2d_params p = make_params(3, 3, 1, 1, 2, 2, 1, 0);

    CHECK(im2col_rows(&p) == 4);
    CHECK(im2col_cols(&p) == 4);
    im2col_hwc_i8(&p, in, col_buf);
    test_check_i8("im2col basic", col_buf, want, 16);
}

/* 3x3 window padding 1 stride 2: out is 2x2 and the borders are zeros. */
static void test_im2col_pad_stride(void)
{
    static const int8_t in[9] = { 1, 2, 3,
                                  4, 5, 6,
                                  7, 8, 9 };
    static const int8_t want[4 * 9] = { 0, 0, 0,  0, 1, 2,  0, 4, 5,
                                        0, 0, 0,  2, 3, 0,  5, 6, 0,
                                        0, 4, 5,  0, 7, 8,  0, 0, 0,
                                        5, 6, 0,  8, 9, 0,  0, 0, 0 };
    conv2d_params p = make_params(3, 3, 1, 1, 3, 3, 2, 1);

    CHECK(im2col_rows(&p) == 4);
    CHECK(im2col_cols(&p) == 9);
    im2col_hwc_i8(&p, in, col_buf);
    test_check_i8("im2col pad+stride", col_buf, want, 36);
}

/*
 * A window covering the whole image yields one row, and because both use
 * (y, x, c) order that row is the HWC input itself.
 */
static void test_im2col_channels(void)
{
    conv2d_params p = make_params(2, 3, 4, 1, 2, 3, 1, 0);

    test_seed(5);
    fill_rand(in_buf, 2 * 3 * 4);
    CHECK(im2col_rows(&p) == 1);
    CHECK(im2col_cols(&p) == 24);
    im2col_hwc_i8(&p, in_buf, col_buf);
    test_check_i8("im2col channel order", col_buf, in_buf, 24);
}

/* Diagonal diff out = in[y][x] - in[y+1][x+1] + 10. */
static void test_conv_hand(void)
{
    static const int8_t in[9] = { 1, 2, 3,
                                  4, 5, 6,
                                  7, 8, 9 };
    static const int8_t w[4] = { 1, 0,
                                 0, -1 };
    static const int32_t bias[1] = { 10 };
    static const int32_t want[4] = { 6, 6, 6, 6 };
    conv2d_params p = make_params(3, 3, 1, 1, 2, 2, 1, 0);
    int32_t out[4];

    conv2d_direct_i8_i32(&p, in, w, bias, out);
    test_check_i32("direct conv hand example", out, want, 4);
    conv2d_im2col_i8_i32(&p, in, w, bias, col_buf, out, NULL);
    test_check_i32("im2col conv hand example", out, want, 4);
    conv2d_im2col_i8_i32(&p, in, w, bias, col_buf, out, gemm_tile_sw);
    test_check_i32("im2col tiled conv hand example", out, want, 4);
}

/* Runs one layer all three ways and compares. Returns 1 if all is correct. */
static int check_layer(const conv2d_params *p, int with_bias)
{
    size_t out_count = im2col_rows(p) * p->out_c;
    const int32_t *bias = with_bias ? bias_buf : NULL;
    size_t i;

    fill_rand(in_buf, p->in_h * p->in_w * p->in_c);
    fill_rand(w_buf, im2col_cols(p) * p->out_c);
    for (i = 0; i < p->out_c; i++)
        bias_buf[i] = (int32_t)(test_rand() % 20001) - 10000;

    conv2d_direct_i8_i32(p, in_buf, w_buf, bias, out_ref);
    conv2d_im2col_i8_i32(p, in_buf, w_buf, bias, col_buf, out_gemm, NULL);
    conv2d_im2col_i8_i32(p, in_buf, w_buf, bias, col_buf, out_tiled, gemm_tile_sw);

    return test_check_i32("im2col+gemm vs direct conv", out_gemm, out_ref, out_count) &&
           test_check_i32("im2col+tiled gemm vs direct conv", out_tiled, out_ref, out_count);
}

static void report_layer(const conv2d_params *p)
{
    test_puts("  layer: in ");
    test_put_int((int32_t)p->in_h);
    test_putc('x');
    test_put_int((int32_t)p->in_w);
    test_putc('x');
    test_put_int((int32_t)p->in_c);
    test_puts(" k ");
    test_put_int((int32_t)p->k_h);
    test_putc('x');
    test_put_int((int32_t)p->k_w);
    test_puts(" out_c ");
    test_put_int((int32_t)p->out_c);
    test_puts(" stride ");
    test_put_int((int32_t)p->stride_h);
    test_putc(',');
    test_put_int((int32_t)p->stride_w);
    test_puts(" pad ");
    test_put_int((int32_t)p->pad_h);
    test_putc(',');
    test_put_int((int32_t)p->pad_w);
    test_putc('\n');
}

/* Random layer shapes, including ones where the window barely fits. */
static void test_conv_random(void)
{
    unsigned iter, run = 0;

    test_seed(2026);
    for (iter = 0; iter < 400; iter++) {
        conv2d_params p;

        p.in_h = test_rand_range(1, 9);
        p.in_w = test_rand_range(1, 9);
        p.in_c = test_rand_range(1, 5);
        p.out_c = test_rand_range(1, 9);
        p.k_h = test_rand_range(1, 4);
        p.k_w = test_rand_range(1, 4);
        p.stride_h = test_rand_range(1, 3);
        p.stride_w = test_rand_range(1, 3);
        p.pad_h = test_rand_range(0, 2);
        p.pad_w = test_rand_range(0, 2);
        if (im2col_rows(&p) == 0)
            continue;
        run++;
        if (!check_layer(&p, (int)(iter & 1))) {
            report_layer(&p);
            return;
        }
    }
    CHECK(run > 200);
}

/* A small but realistic first layer: 16x16 RGB 3x3 16 filters same padding. */
static void test_conv_layer(void)
{
    conv2d_params p = make_params(16, 16, 3, 16, 3, 3, 1, 1);

    test_seed(418);
    CHECK(im2col_rows(&p) == 256);
    CHECK(im2col_cols(&p) == 27);
    if (!check_layer(&p, 1))
        report_layer(&p);
}

/* Two chained layers with requantization between them. */
static void test_conv_chain(void)
{
    static int8_t act[8 * 8 * 4];
    conv2d_params l1 = make_params(8, 8, 3, 4, 3, 3, 1, 1);
    conv2d_params l2 = make_params(8, 8, 4, 6, 3, 3, 2, 1);
    requant_params rq;
    size_t n2;

    rq.multiplier = 1;
    rq.shift = 9;
    rq.zero_point = 0;
    rq.relu = 1;

    test_seed(99);
    fill_rand(in_buf, 8 * 8 * 3);
    fill_rand(w_buf, im2col_cols(&l1) * l1.out_c);
    conv2d_im2col_i8_i32(&l1, in_buf, w_buf, NULL, col_buf, out_tiled, gemm_tile_sw);
    requant_i32_to_i8_array(out_tiled, act, 8 * 8 * 4, &rq);

    /* The requantized activations are the HWC input of the next layer as is. */
    fill_rand(w_buf, im2col_cols(&l2) * l2.out_c);
    n2 = im2col_rows(&l2) * l2.out_c;
    conv2d_direct_i8_i32(&l2, act, w_buf, NULL, out_ref);
    conv2d_im2col_i8_i32(&l2, act, w_buf, NULL, col_buf, out_tiled, gemm_tile_sw);
    CHECK(im2col_rows(&l2) == 16);
    test_check_i32("chained layer 2", out_tiled, out_ref, n2);
}

void test_conv(void)
{
    test_out_dims();
    test_im2col_basic();
    test_im2col_pad_stride();
    test_im2col_channels();
    test_conv_hand();
    test_conv_random();
    test_conv_layer();
    test_conv_chain();
}