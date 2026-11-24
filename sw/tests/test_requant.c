#include "requant.h"
#include "test_util.h"
#include "tests.h"

static int8_t rq(int32_t x, int32_t mult, uint32_t shift, int32_t zp, int relu)
{
    requant_params p;

    p.multiplier = mult;
    p.shift = shift;
    p.zero_point = zp;
    p.relu = relu;
    return requant_i32_to_i8(x, &p);
}

void test_requant(void)
{
    static const int32_t src[6] = { -1000, -3, 0, 3, 5, 1000 };
    static const int8_t want[6] = { -128, -1, 0, 1, 1, 127 };
    int8_t dst[6];
    requant_params p;

    /* Identity scale with saturation */
    CHECK(rq(0, 1, 0, 0, 0) == 0);
    CHECK(rq(100, 1, 0, 0, 0) == 100);
    CHECK(rq(127, 1, 0, 0, 0) == 127);
    CHECK(rq(128, 1, 0, 0, 0) == 127);
    CHECK(rq(-128, 1, 0, 0, 0) == -128);
    CHECK(rq(-129, 1, 0, 0, 0) == -128);

    /* Round to nearest ties away from zero symmetric in sign. */
    CHECK(rq(1, 1, 1, 0, 0) == 1);    /* 0.5  -> 1  */
    CHECK(rq(-1, 1, 1, 0, 0) == -1);  /* -0.5 -> -1 */
    CHECK(rq(2, 1, 1, 0, 0) == 1);
    CHECK(rq(3, 1, 1, 0, 0) == 2);    /* 1.5  -> 2  */
    CHECK(rq(-3, 1, 1, 0, 0) == -2);
    CHECK(rq(5, 1, 2, 0, 0) == 1);    /* 1.25 -> 1  */
    CHECK(rq(6, 1, 2, 0, 0) == 2);    /* 1.5  -> 2  */
    CHECK(rq(-6, 1, 2, 0, 0) == -2);
    CHECK(rq(-5, 1, 2, 0, 0) == -1);

    /* Multiplier then shift: 1000 * 3 / 64 = 46.875 -> 47. */
    CHECK(rq(1000, 3, 6, 0, 0) == 47);
    CHECK(rq(-1000, 3, 6, 0, 0) == -47);

    /* No overflow in the 64-bit product: (2^31 - 1)^2 / 2^62 is just under 1. */
    CHECK(rq(INT32_MAX, INT32_MAX, 62, 0, 0) == 1);
    CHECK(rq(INT32_MIN, INT32_MIN, 62, 0, 0) == 1);
    CHECK(rq(INT32_MIN, INT32_MAX, 31, 0, 0) == -128);

    /* Zero point and ReLU clamping at the zero point. */
    CHECK(rq(0, 1, 0, 10, 0) == 10);
    CHECK(rq(120, 1, 0, 10, 0) == 127);
    CHECK(rq(-5, 1, 0, 0, 1) == 0);
    CHECK(rq(5, 1, 0, 0, 1) == 5);
    CHECK(rq(-50, 1, 0, -20, 1) == -20);
    CHECK(rq(-50, 1, 0, -20, 0) == -70);

    /* Array form. */
    p.multiplier = 1;
    p.shift = 2;
    p.zero_point = 0;
    p.relu = 0;
    requant_i32_to_i8_array(src, dst, 6, &p);
    test_check_i8("requant array", dst, want, 6);
}