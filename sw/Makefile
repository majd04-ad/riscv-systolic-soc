# Host build of the reference software and its tests.
#
#   make test            build and run the tests
#   make test-sanitize   same, with AddressSanitizer and UBSan
#   make test ACCEL_DIM=8
#
# The library code is freestanding C99 apart from <stdint.h>/<stddef.h>;
# only tests/platform_host.c uses stdio.

CC        ?= cc
ACCEL_DIM ?= 4
BUILD     ?= build/dim$(ACCEL_DIM)

CFLAGS  ?= -O2 -g
CFLAGS  += -std=c99 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror
CPPFLAGS += -Ilib/include -DACCEL_DIM=$(ACCEL_DIM)

LIB_SRCS  := lib/src/gemm.c lib/src/conv.c lib/src/requant.c
TEST_SRCS := tests/test_main.c tests/test_util.c tests/test_gemm.c \
             tests/test_conv.c tests/test_requant.c tests/platform_host.c

.PHONY: all test test-sanitize clean

all: $(BUILD)/run_tests

$(BUILD)/run_tests: $(LIB_SRCS) $(TEST_SRCS) $(wildcard lib/include/*.h tests/*.h)
	@mkdir -p $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LIB_SRCS) $(TEST_SRCS) -o $@

test: $(BUILD)/run_tests
	./$(BUILD)/run_testsExpand

test-sanitize:
	$(MAKE) BUILD=$(BUILD)/sanitize CFLAGS="-O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all" test

clean:
	rm -rf $(BUILD)