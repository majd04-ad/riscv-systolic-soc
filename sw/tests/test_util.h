#ifndef TEST_UTIL_H
#define TEST_UTIL_H

#include <stddef.h>
#include <stdint.h>

/* Provided by the platform file  */
void test_putc(char ch);

void test_puts(const char *s);
void test_put_int(int32_t v);

/* Deterministic PRNG (xorshift32) so every platform sees the same data. */
void test_seed(uint32_t seed);
uint32_t test_rand(void);
int8_t test_rand_i8(void);
/* Uniform in [lo, hi]. */
uint32_t test_rand_range(uint32_t lo, uint32_t hi);

extern unsigned test_checks;
extern unsigned test_failures;

void test_fail_at(const char *file, int line, const char *msg);

#define CHECK(cond)                                         \
    do {                                                    \
        test_checks++;                                      \
        if (!(cond))                                        \
            test_fail_at(__FILE__, __LINE__, #cond);        \
    } while (0)

/* Compare two INT32 buffers report the first mismatch and Returns 1 if equal. */
int test_check_i32(const char *what, const int32_t *got, const int32_t *want,
                   size_t count);
int test_check_i8(const char *what, const int8_t *got, const int8_t *want,
                  size_t count);

#endif