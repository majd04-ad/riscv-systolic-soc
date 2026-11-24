/*
2D convolution on INT8 tensors directly and lowered to GEMM by im2col
matrix whose rows are the receptive fields of each output pixel in the
same (ky, kx, c) order. The convolution is then one GEMM

out (P x out_c) = patches (P x K) * weights (K x out_c),  P = out_h * out_w

with the weights stationary in the array, and its output is again HWC so layers chain without transposes.
Padding is zero padding.
With symmetric INT8 quantisation this is the same as padding with the quantised value of zero.
 */
#ifndef CONV_H
#define CONV_H

#include <stddef.h>
#include <stdint.h>

#include "gemm.h"

typedef struct {
    size_t in_h, in_w, in_c;
    size_t out_c;
    size_t k_h, k_w;
    size_t stride_h, stride_w;  /* >= 1 */
    size_t pad_h, pad_w;        /* zero rows/cols added on each side */
} conv2d_params;

/* Output */
size_t conv2d_out_dim(size_t in, size_t k, size_t stride, size_t pad);

size_t conv2d_out_h(const conv2d_params *p);
size_t conv2d_out_w(const conv2d_params *p);

/* Rows of the patch matrix */
size_t im2col_rows(const conv2d_params *p);
/* Columns of the patch matrix  */
size_t im2col_cols(const conv2d_params *p);

/* Fill col  from an HWC input. */
void im2col_hwc_i8(const conv2d_params *p, const int8_t *in, int8_t *col);

/*Reference convolution, straight from the definition.*/
void conv2d_direct_i8_i32(const conv2d_params *p, const int8_t *in,
                          const int8_t *w, const int32_t *bias, int32_t *out);

/*
Convolution as im2col followed by GEMM. col is scratch space of
im2col_rows(p) * im2col_cols(p) bytes. tile selects the GEMM:
NULL runs the plain gemm_i8_i32, anything else runs gemm_i8_i32_tiled.
 */
void conv2d_im2col_i8_i32(const conv2d_params *p, const int8_t *in,
                          const int8_t *w, const int32_t *bias,
                          int8_t *col, int32_t *out, gemm_tile_fn tile);

#endif 