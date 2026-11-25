/*
 * Host platform for the benchmark: output on stdout, counters read 0.
 * The host run only produces the reference hashes.
 */
#include <stdio.h>

#include "bench_platform.h"
#include "test_util.h"

void test_putc(char ch)
{
    putchar(ch);
}

void bench_init(void)
{
}

uint32_t bench_cycles(void)
{
    return 0;
}

uint32_t bench_instret(void)
{
    return 0;
}