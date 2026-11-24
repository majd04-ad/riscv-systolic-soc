#include "test_util.h"

unsigned test_checks;
unsigned test_failures;

static uint32_t rng_state = 1;

void test_puts(const char *s)
{
    while (*s)
        test_putc(*s++);
}

void test_put_int(int32_t v)
{
    char buf[12];
    int i = 0;
    uint32_t u = (uint32_t)v;

    if (v < 0) {
        test_putc('-');
        u = 0u - u;
    }
    do {
        buf[i++] = (char)('0' + u % 10);
        u /= 10;
    } while (u != 0);
    while (i > 0)
        test_putc(buf[--i]);
}

void test_seed(uint32_t seed)
{
    rng_state = seed ? seed : 1;
}

uint32_t test_rand(void)
{
    uint32_t x = rng_state;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

int8_t test_rand_i8(void)
{
    return (int8_t)(uint8_t)(test_rand() >> 24);
}

uint32_t test_rand_range(uint32_t lo, uint32_t hi)
{
    return lo + test_rand() % (hi - lo + 1);
}

void test_fail_at(const char *file, int line, const char *msg)
{
    test_failures++;
    test_puts("FAIL ");
    test_puts(file);
    test_putc(':');
    test_put_int(line);
    test_puts(": ");
    test_puts(msg);
    test_putc('\n');
}

int test_check_i32(const char *what, const int32_t *got, const int32_t *want,
                   size_t count)
{
    size_t i;

    test_checks++;
    for (i = 0; i < count; i++) {
        if (got[i] != want[i]) {
            test_failures++;
            test_puts("FAIL ");
            test_puts(what);
            test_puts(": index ");
            test_put_int((int32_t)i);
            test_puts(" got ");
            test_put_int(got[i]);
            test_puts(" want ");
            test_put_int(want[i]);
            test_putc('\n');
            return 0;
        }
    }
    return 1;
}

int test_check_i8(const char *what, const int8_t *got, const int8_t *want,
                  size_t count)
{
    size_t i;

    test_checks++;
    for (i = 0; i < count; i++) {
        if (got[i] != want[i]) {
            test_failures++;
            test_puts("FAIL ");
            test_puts(what);
            test_puts(": index ");
            test_put_int((int32_t)i);
            test_puts(" got ");
            test_put_int(got[i]);
            test_puts(" want ");
            test_put_int(want[i]);
            test_putc('\n');
            return 0;
        }
    }
    return 1;
}