# NOREV-G14 Paired Action Emit -- reproducible build of the frozen submission artifact.
#
# Target artifact: bin/player_norev_g14_pairemit_c74bea0c.so
#   SHA-256 c74bea0c7152d7723e8f1f6937ca074a3a2b01c07eab9371ddfb2a0ae267bb5a
#   14,504 bytes, moveDecision 1,862 B / 338 instructions, entry 0x320, DT_NEEDED=0
#
# `make verify` rebuilds from src/ and must reproduce that SHA byte for byte.

# ---------------------------------------------------------------------------
# Toolchain.  MUST be clang++.
#
# Do NOT write `CXX ?= clang++`.  GNU Make PRE-DEFINES CXX = g++ with origin
# `default`, so `?=` does NOT fire and you silently get g++.  This project has
# measured the same source built with g++ running 267 ns slower online (10 paired
# self-play games, t = 11.5).  The override below only replaces the value when it
# is still Make's built-in default, so an explicit `make CXX=...` still works.
# ---------------------------------------------------------------------------
ifeq ($(origin CXX),default)
CXX := clang++
endif

COMMON := -std=c++17 -O3 -DNDEBUG -fPIC -fno-exceptions -fno-rtti -fno-plt \
          -fno-stack-protector -fomit-frame-pointer

# Frozen semantic switch set inherited from V40.  Changing any of these changes
# the strategy, not just the codegen.
SEMANTIC := -DV7_VEC_WINDOW=1 -DV7_AVX512=1 -DV7_INLINE_SOLVE=1 \
            -DV7_FAST_SELECT=0 -DV7_QBASE_LUT=0 -DV7_PREFETCH=0 \
            -DV7_ARITH_EMIT=0 -DV7_ARITH_PARENT=0 -DV7_ARITH_QBASE=0 \
            -DV7_TRAMPLE=0 -DV7_SNAPSHOT=0 -DV7_PACK_TABLES=1 \
            -DV7_LUT16=0 -DV7_RICH=0 -DV7_OPENNESS=0 -DV7_ZONE=0 \
            -DV7_SPLIT_TARGET=6 -DV26_ZMM_PAIR_WINDOW=1 \
            -DV30_VBMI_BLOCKERS=1 -DV33_WINNER_HLUT=1 -DV37_QBASE_PAIR=1 \
            -DV35_PAIR_MASKS=1 -DV34_TRUST_OFFICIAL_INPUT=1 \
            -DV34_TRUST_GOLD_RANGE=1 -DV40_ASSUME_POSITION_RANGE=1

# Pure-latency switches, all proven bit-exact against their predecessor.
LATENCY := -DV44_ROWOFF=1 -DV44_MASKVEC=1 -DV44_TAIL=2 \
           -DV46_COL5=1 -DV46_IDXFUSE=1 -DV46_DEADINIT=1 -DV49_QBASE16=1

# NOREV-G1 global ordering (gold > not_reversed > distance), baked into the
# QBASE16 table at compile time so it costs no runtime instruction.
STRATEGY := -DV50_NOREV_GLOBAL=1

# The NOREV-G latency chain that produced G14.
#   NOREV_DIRECT_EMIT / NOREV_RAW_GOLD : G6 / G11
#   V47_KEYFUSE / V47_KMASK / V47_GDFUSE : G12 (neutral alone; required for G14's shape)
#   NOREV_PAIR_EMIT : G14 itself -- both units' routes packed, one vpmovzxbd + 32B store
GCHAIN := -DNOREV_DIRECT_EMIT=1 -DNOREV_RAW_GOLD=1 \
          -DV47_KEYFUSE=1 -DV47_KMASK=1 -DV47_GDFUSE=1 \
          -DNOREV_PAIR_EMIT=1

# Behaviour-changing switches in this shared source that stay OFF for G14.
#   NOREV_MASKED_PREV_PAD : G26. Bit-exact with G14 but server WARM is 7/7 SLOWER
#     (+0.30 ns) while COLD is 6/7 faster (-0.60 ns), and 70 paired online games
#     put the difference at p90 -1.36 +- 2.21 ns with first-move rate 0.5073 --
#     indistinguishable.  It also widens PersistentState to prev_quadrant[4] and
#     writes two padding bytes on purpose, which is a maintenance hazard.
#   NOREV_PAIR_PACK_PERMB : G27. Bit-exact, server COLD 7/7 better, but 14 online
#     games read p50 +2.14 / p90 +3.57 -- not adopted.
#   NOREV_PAIR_COMPACT : G25. Online p90 130/130, neutral.
OFF := -DNOREV_MASKED_PREV_PAD=0 -DNOREV_PAIR_PACK_PERMB=0 -DNOREV_PAIR_COMPACT=0

# Explicit ISA subset, deliberately NOT -march=native / -march=znver5.
# znver5 was built and taken online: smaller (-33 B / -18 instructions) and best
# offline COLD, but +12.5 ns on paired online p90.  Pinning the subset also keeps
# the artifact reproducible across build-server kernel/compiler updates.
ISA := -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mbmi2 -mavx512vbmi

LINK := -flto -shared -Wl,-z,defs -Wl,-s -Wl,--as-needed -nostdlib++ -static-libgcc

DEFINES := $(SEMANTIC) $(LATENCY) $(STRATEGY) $(GCHAIN) $(OFF)
CXXFLAGS := $(COMMON) $(DEFINES) $(ISA)

FROZEN_SHA := c74bea0c7152d7723e8f1f6937ca074a3a2b01c07eab9371ddfb2a0ae267bb5a
FROZEN     := bin/player_norev_g14_pairemit_c74bea0c.so
OUT        := bin/player_g14.so

.PHONY: all toolchain verify elf equiv equiv26 bench clean

all: toolchain $(OUT)

# Hard gate: refuse to build with anything but clang.  __builtin_assume is a clang
# builtin (g++ fails outright) but a future edit could remove it, and then a g++
# build would succeed and silently produce a much slower artifact.
toolchain:
	@$(CXX) --version | head -1
	@$(CXX) --version | head -1 | grep -qi clang || \
	  { echo "REFUSING: CXX=$(CXX) is not clang++."; exit 1; }

$(OUT): src/player_v7.cpp src/game_api.h Makefile
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -Isrc -o $@ src/player_v7.cpp $(LINK)
	@stat -c '%n %s bytes' $@
	@sha256sum $@

# Byte-for-byte reproduction of the frozen submission artifact.
verify: all elf
	@echo "--- SHA vs frozen ---"
	@got=$$(sha256sum $(OUT) | cut -d' ' -f1); \
	 echo "built  $$got"; echo "frozen $(FROZEN_SHA)"; \
	 [ "$$got" = "$(FROZEN_SHA)" ] && echo "MATCH: byte-identical to the frozen artifact" \
	   || { echo "MISMATCH"; exit 1; }

elf: $(OUT)
	@echo "--- ELF / ABI ---"
	@echo -n "exports (must be 1): "; nm -D --defined-only $(OUT) | grep -c ' T '
	@echo -n "moveDecision export (must be 1): "; nm -D --defined-only $(OUT) | grep -c moveDecision
	@echo -n "undefined symbols: "; nm -D -u $(OUT) | wc -l
	@echo -n "DT_NEEDED (must be 0): "; readelf -d $(OUT) | grep -c NEEDED || true
	@echo -n "ldd: "; ldd $(OUT) 2>&1 | head -1
	@echo -n "max glibc: "; \
	  g=$$(objdump -T $(OUT) | grep -o 'GLIBC_[0-9.]*' | sort -V | tail -1); \
	  echo "$${g:-none (no versioned glibc symbols)}"
	@echo -n "moveDecision bytes (expect 1862): "; \
	  nm -S -D $(OUT) | awk '/ T moveDecision$$/{print strtonum("0x"$$2); exit}'
	@echo -n "zmm (whole file): "; objdump -d $(OUT) | grep -c '%zmm' || true
	@echo -n "ymm (whole file): "; objdump -d $(OUT) | grep -c '%ymm' || true
# Clamp the disassembly to [addr, addr+size).  `--disassemble=moveDecision` is NOT safe
# here: `-Wl,-s` strips the local helper names, so objdump both starts early (it picks up
# the preceding unnamed function -- there is a `mov %rsp,%rbp` at 0x2e6, before
# moveDecision's 0x320, which made a naive grep report a false stack reference) and runs
# past the end, mis-decoding bytes until .fini.
	@echo -n "%rsp refs inside moveDecision (must be 0): "; \
	  a=$$(nm -S -D $(OUT) | awk '/ T moveDecision$$/{print $$1; exit}'); \
	  s=$$(nm -S -D $(OUT) | awk '/ T moveDecision$$/{print $$2; exit}'); \
	  st=$$((0x$$a)); en=$$((st + 0x$$s)); \
	  objdump -d --start-address=$$st --stop-address=$$en $(OUT) | grep -c '%rsp' || true
# Deliberately NOT counting instructions here.  GNU objdump and llvm-objdump disagree on
# how they line-break long EVEX encodings (GNU reports 378 lines where llvm-objdump counts
# 338 instructions), so any hardcoded expectation would be tool-version noise.  The SHA
# match above is the authoritative identity check; the 338 figure quoted in README.md is
# llvm-objdump's, produced by tools/ll_gate.py.

# Bit-exact decision equivalence against the predecessor G12 that G14 was promoted
# over.  Ordered replay: the kernel carries cross-round state, so a divergence
# surfaces on the NEXT round -- order matters, a shuffled set would hide it.
equiv: $(OUT) validation/so_equiv
	./validation/so_equiv validation/player_norev_g12_raw_fused_b984d312.so $(OUT) $(ROUNDS)

# G26 must also be bit-identical (it is a pure codegen variant of G14).
equiv26: $(OUT) validation/so_equiv
	./validation/so_equiv validation/player_norev_g26_pair_maskedprev_dc18889f.so $(OUT) $(ROUNDS)

validation/so_equiv: validation/so_equiv.cpp validation/rank1_rounds.h src/game_api.h
	$(CXX) -std=c++17 -O2 -Ivalidation -Isrc -o $@ validation/so_equiv.cpp -ldl

# Pinned WARM/COLD. EXCLUSION GATE ONLY -- this project has measured the offline
# ORDER reversing online more than once (znver5, and G27 whose COLD was 7/7 better
# yet read +3.57 ns on paired online p90).  Never promote on these numbers.
bench: $(OUT) validation/bench_so
	taskset -c 5 ./validation/bench_so $(FROZEN) $(OUT)

validation/bench_so: validation/bench_so.cpp validation/rank1_rounds.h src/game_api.h
	$(CXX) -std=c++17 -O2 -Ivalidation -Isrc -o $@ validation/bench_so.cpp -ldl

clean:
	rm -f $(OUT) validation/so_equiv validation/bench_so
