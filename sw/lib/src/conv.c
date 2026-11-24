#include "conv.h"
size_t conv2d_out_dim(size_t in, size_t k, size_t stride, size_t pad)
{
    size_t padded = in + 2 * pad;

    if (stride == 0 || k == 0 || k > padded)
        return 0;
    return (padded - k) / stride + 1;
}

size_t conv2d_out_h(const conv2d_params *p)
{
    return conv2d_out_dim(p->in_h, p->k_h, p->stride_h, p->pad_h);
}

size_t conv2d_out_w(const conv2d_params *p)
{
    return conv2d_out_dim(p->in_w, p->k_w, p->stride_w, p->pad_w);
}

size_t im2col_rows(const conv2d_params *p)
{
    return conv2d_out_h(p) * conv2d_out_w(p);
}

size_t im2col_cols(const conv2d_params *p)
{
    return p->k_h * p->k_w * p->in_c;
}

/*
 * Input coordinates are computed in the padded frame and shifted back so
 * the "is this inside the image" test stays in unsigned arithmetic:
 * iy = oy * stride + ky - pad is inside when pad <= oy * stride + ky < pad + in_h.
 */
void im2col_hwc_i8(const conv2d_params *p, const int8_t *in, int8_t *col)
{
    size_t out_h = conv2d_out_h(p), out_w = conv2d_out_w(p);
    size_t oy, ox, ky, kx, c;

    for (oy = 0; oy < out_h; oy++) {
        for (ox = 0; ox < out_w; ox++) {
            for (ky = 0; ky < p->k_h; ky++) {
                size_t py = oy * p->stride_h + ky;
                int row_ok = py >= p->pad_h && py < p->pad_h + p->in_h;
                for (kx = 0; kx < p->k_w; kx++) {
                    size_t px = ox * p->stride_w + kx;
                    int ok = row_ok && px >= p->pad_w && px < p->pad_w + p->in_w;
                    const int8_t *src = ok
                        ? in + ((py - p->pad_h) * p->in_w + (px - p->pad_w)) * p->in_c
                        : NULL;
                    for (c = 0; c < p->in_c; c++)
                        *col++ = ok ? src[c] : 0;
                }
            }
        }
    }
}

void conv2d_direct_i8_i32(const conv2d_params *p, const int8_t *in,
                          const int8_t *w, const int32_t *bias, int32_t *out)
{
    size_t out_h = conv2d_out_h(p), out_w = conv2d_out_w(p);
    size_t oy, ox, oc, ky, kx, c;

    for (oy = 0; oy < out_h; oy++) {
        for (ox = 0; ox < out_w; ox++) {
            for (oc = 0; oc < p->out_c; oc++) {
                int32_t acc = bias ? bias[oc] : 0;
                for (ky = 0; ky < p->k_h; ky++) {
                    size_t py = oy * p->stride_h + ky;
                    if (py < p->pad_h || py >= p->pad_h + p->in_h)
                        continue;
                    for (kx = 0; kx < p->k_w; kx++) {
                        size_t px = ox * p->stride_w + kx;
                        if (px < p->pad_w || px >= p->pad_w + p->in_w)
                            continue;
                        for (c = 0; c < p->in_c; c++) {
                            int8_t x = in[((py - p->pad_h) * p->in_w + (px - p->pad_w)) * p->in_c + c];
                            int8_t k = w[((ky * p->k_w + kx) * p->in_c + c) * p->out_c + oc];
                            acc += (int32_t)x * (int32_t)k;
                        }
                    }
                }
                out[(oy * out_w + ox) * p->out_c + oc] = acc;
            }
        }
    }
}

void conv2d_im2col_i8_i32(const conv2d_params *p, const int8_t *in,
                          const int8_t *w, const int32_t *bias,
                          int8_t *col, int32_t *out, gemm_tile_fn tile)
{
    size_t rows = im2col_rows(p), k = im2col_cols(p);
    size_t r, oc;

    im2col_hwc_i8(p, in, col);

    if (tile == NULL)
        gemm_i8_i32(rows, p->out_c, k, col, k, w, p->out_c, out, p->out_c);
    else
        gemm_i8_i32_tiled(rows, p->out_c, k, col, k, w, p->out_c, out, p->out_c, tile);

    if (bias != NULL)
        for (r = 0; r < rows; r++)
            for (oc = 0; oc < p->out_c; oc++)
                out[r * p->out_c + oc] += bias[oc];
}