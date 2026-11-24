/*from INT32 accumulators back to INT8, as done between
y = clamp(round(x * multiplier / 2^shift) + zero_point, -128, 127)
Rounding is to nearest so results do notdepend on the sign of x
 */
#ifndef REQUANT_H
#define REQUANT_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int32_t multiplier;  /* fixedmpoint  */
    uint32_t shift;      /* right shift */
    int32_t zero_point;  /* added after scaling */
    int relu;            /* nonzero: clamp the low end at zero point */
} requant_params;

int8_t requant_i32_to_i8(int32_t x, const requant_params *rq);

void requant_i32_to_i8_array(const int32_t *src, int8_t *dst, size_t count,
                             const requant_params *rq);

#endif