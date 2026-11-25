# CPU-only GEMM / im2col benchmark: host reference build and Ibex Simple
# System build. Run from this directory, or with make -C sw/apps/bench_cpu.
#
#   make host        build and run on the host (reference hashes)
#   make ibex        build bench_cpu.elf / .vmem for Ibex (RV32IMC)
#   make run SIM=<path to Vibex_simple_system>  [CONFIG=small]
#                    run on Ibex in Verilator and check against the host
#
# See docs/benchmarks/cpu_baseline.md for how to build the simulator.

ACCEL_DIM ?= 4
OPT       ?= -O2
CONFIG    ?= small
BUILD     ?= build

ROOT       := $(abspath ../../..)
SW         := $(ROOT)/sw
SS_COMMON  := $(ROOT)/third_party/ibex/examples/sw/simple_system/common

LIB_SRCS   := $(SW)/lib/src/gemm.c $(SW)/lib/src/conv.c $(SW)/tests/test_util.c
BENCH_SRCS := bench_cpu.c $(LIB_SRCS)
HDRS       := $(wildcard *.h $(SW)/lib/include/*.h $(SW)/tests/*.h)
CPPFLAGS   := -I. -I$(SW)/lib/include -I$(SW)/tests -DACCEL_DIM=$(ACCEL_DIM)
WARN       := -Wall -Wextra -Wshadow -Werror

# Host
HOST_CC    ?= cc
# Outputs go in directories named after the settings, so changing
# ACCEL_DIM, OPT or ARCH rebuilds instead of reusing a stale binary.
HOST_DIR   := $(BUILD)/host/dim$(ACCEL_DIM)
HOST_BIN   := $(HOST_DIR)/bench_cpu
HOST_LOG   := $(HOST_DIR)/bench.log

# Ibex: any RISC-V GCC that can target rv32imc/ilp32 (riscv32- or riscv64-).
# GCC 12+ needs _zicsr spelled out for the CSR instructions.
RV_CC      ?= $(shell command -v riscv32-unknown-elf-gcc >/dev/null 2>&1 \
                && echo riscv32-unknown-elf-gcc || echo riscv64-unknown-elf-gcc)
RV_PREFIX  := $(patsubst %gcc,%,$(RV_CC))
ARCH       ?= rv32imc_zicsr
# gnu99 for the inline asm in the Simple System helpers.
# -fno-tree-loop-distribute-patterns keeps GCC from turning the kernels'
# loops (and our own memset) into library calls.
RV_CFLAGS  := -march=$(ARCH) -mabi=ilp32 -mcmodel=medany -static $(OPT) -g \
              -ffreestanding -nostdlib -nostartfiles \
              -fno-tree-loop-distribute-patterns -std=gnu99 $(WARN)
empty      :=
space      := $(empty) $(empty)
IBEX_DIR   := $(BUILD)/ibex/dim$(ACCEL_DIM)$(subst $(space),,$(OPT))_$(ARCH)
IBEX_ELF   := $(IBEX_DIR)/bench_cpu.elf
IBEX_SRCS  := $(BENCH_SRCS) platform_ibex.c $(SS_COMMON)/simple_system_common.c \
              $(SS_COMMON)/crt0.S
IBEX_LOG   := $(IBEX_DIR)/bench_$(CONFIG).log

.PHONY: all host ibex run clean

all: host ibex

$(HOST_BIN): $(BENCH_SRCS) platform_host.c $(HDRS)
	@mkdir -p $(dir $@)
	$(HOST_CC) $(CPPFLAGS) -std=c99 -O2 $(WARN) $(BENCH_SRCS) platform_host.c -o $@

$(HOST_LOG): $(HOST_BIN)
	./$(HOST_BIN) > $@.tmp
	mv $@.tmp $@
	@tail -n 1 $@

host: $(HOST_LOG)

$(IBEX_ELF): $(IBEX_SRCS) $(HDRS) $(SS_COMMON)/link.ld
	@mkdir -p $(dir $@)
	$(RV_CC) $(CPPFLAGS) -I$(SS_COMMON) $(RV_CFLAGS) -T $(SS_COMMON)/link.ld \
	    $(IBEX_SRCS) -o $@
	$(RV_PREFIX)size $@

$(IBEX_DIR)/bench_cpu.dis: $(IBEX_ELF)
	$(RV_PREFIX)objdump -d -S $< > $@

ibex: $(IBEX_ELF) $(IBEX_DIR)/bench_cpu.dis

run: $(IBEX_ELF) $(HOST_LOG)
	@test -n "$(SIM)" || { echo "set SIM=<path to Vibex_simple_system>"; exit 1; }
	@mkdir -p $(IBEX_DIR)/sim_$(CONFIG)
	cd $(IBEX_DIR)/sim_$(CONFIG) && $(abspath $(SIM)) --meminit=ram,$(abspath $(IBEX_ELF)) > sim_stdout.txt
	cp $(IBEX_DIR)/sim_$(CONFIG)/ibex_simple_system.log $(IBEX_LOG)
	python3 compare.py $(HOST_LOG) $(IBEX_LOG)

clean:
	rm -rf $(BUILD)