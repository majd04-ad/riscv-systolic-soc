#include "requant.h"

int8_t requant_i32_to_i8(int32_t x, const requant_params *rq)
{
    int64_t prod = (int64_t)x * rq->multiplier;
    int64_t mag = prod < 0 ? -prod : prod;
    int64_t lo = -128;
    int64_t y;

    if (rq->shift > 0)
        mag = (mag + ((int64_t)1 << (rq->shift - 1))) >> rq->shift;
    y = (prod < 0 ? -mag : mag) + rq->zero_point;

    if (rq->relu && lo < rq->zero_point)
        lo = rq->zero_point;
    if (y < lo)
        y = lo;
    if (y > 127)
        y = 127;
    return (int8_t)y;
}

void requant_i32_to_i8_array(const int32_t *src, int8_t *dst, size_t count,
                             const requant_params *rq)
{
    size_t i;

    for (i = 0; i < count; i++)
        dst[i] = requant_i32_to_i8(src[i], rq);
}