/*
 * What the benchmark needs from the platform, besides test_putc
 * (test_util.h): free-running cycle and retired-instruction counters.
 * Only differences between two readings are used, so 32 bits that wrap
 * are enough as long as one measurement stays under 2^32 cycles.
 */
#ifndef BENCH_PLATFORM_H
#define BENCH_PLATFORM_H

#include <stdint.h>

void bench_init(void);
uint32_t bench_cycles(void);
uint32_t bench_instret(void);

#endif /* BENCH_PLATFORM_H */