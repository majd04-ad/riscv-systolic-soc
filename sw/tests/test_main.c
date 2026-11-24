/*
 * Test for the reference GEMM / im2col / requant software.
 * Returns 0 when every check passes The same file is meant to run as a
 * bare metal program on Ibex with a different platform file.
 */
#include "test_util.h"
#include "tests.h"

#include "accel_config.h"

int main(void)
{
    test_puts("sw/lib tests, ACCEL_DIM=");
    test_put_int(ACCEL_DIM);
    test_putc('\n');

    test_gemm();
    test_conv();
    test_requant();

    test_puts(test_failures ? "FAILED: " : "PASSED: ");
    test_put_int((int32_t)test_failures);
    test_puts(" failures in ");
    test_put_int((int32_t)test_checks);
    test_puts(" checks\n");
    return test_failures ? 1 : 0;
}