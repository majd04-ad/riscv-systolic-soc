/*
 * Ibex Simple System platform for the benchmark: characters go to the
 * simulator's output device (ibex_simple_system.log), counters are the
 * mcycle and minstret CSRs.
 *
 * The program is linked with -nostdlib, so this file also provides the
 * memset and memcpy that GCC may emit for loops and struct copies.
 */
#include <stddef.h>
#include <stdint.h>

#include "bench_platform.h"
#include "simple_system_common.h"
#include "test_util.h"

void test_putc(char ch)
{
    putchar(ch);
}

void bench_init(void)
{
    pcount_enable(1);
}

uint32_t bench_cycles(void)
{
    uint32_t v;

    PCOUNT_READ(mcycle, v);
    return v;
}

uint32_t bench_instret(void)
{
    uint32_t v;

    PCOUNT_READ(minstret, v);
    return v;
}

void *memset(void *dst, int c, size_t n)
{
    unsigned char *d = dst;

    while (n--)
        *d++ = (unsigned char)c;
    return dst;
}

void *memcpy(void *dst, const void *src, size_t n)
{
    unsigned char *d = dst;
    const unsigned char *s = src;

    while (n--)
        *d++ = *s++;
    return dst;
}