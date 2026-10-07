#!/usr/bin/env bash
# NOREV-G14 -- one-shot build + full self-check on the contest build server.
#
# Standalone equivalent of `make verify elf equiv bench`, for when you would rather
# paste one script than rely on GNU Make being present.  The flags here are kept
# identical to the Makefile on purpose; if you edit one, edit both.
#
# Run this ON the build server (AMD EPYC 9T25, same model as the contest machine):
#     bash build.sh
#
# It never uploads a model and never starts an official game.

set -euo pipefail
cd "$(dirname "$0")"

FROZEN_SHA=c74bea0c7152d7723e8f1f6937ca074a3a2b01c07eab9371ddfb2a0ae267bb5a
FROZEN=bin/player_norev_g14_pairemit_c74bea0c.so
OUT=bin/player_g14.so

# --- toolchain gate -------------------------------------------------------
# Must be clang.  Same source under g++ measured 267 ns slower online (t = 11.5).
CXX=${CXX:-clang++}
if ! "$CXX" --version | head -1 | grep -qi clang; then
  echo "REFUSING: CXX=$CXX is not clang++"; exit 1
fi
"$CXX" --version | head -1

COMMON="-std=c++17 -O3 -DNDEBUG -fPIC -fno-exceptions -fno-rtti -fno-plt
        -fno-stack-protector -fomit-frame-pointer"

SEMANTIC="-DV7_VEC_WINDOW=1 -DV7_AVX512=1 -DV7_INLINE_SOLVE=1
          -DV7_FAST_SELECT=0 -DV7_QBASE_LUT=0 -DV7_PREFETCH=0
          -DV7_ARITH_EMIT=0 -DV7_ARITH_PARENT=0 -DV7_ARITH_QBASE=0
          -DV7_TRAMPLE=0 -DV7_SNAPSHOT=0 -DV7_PACK_TABLES=1
          -DV7_LUT16=0 -DV7_RICH=0 -DV7_OPENNESS=0 -DV7_ZONE=0
          -DV7_SPLIT_TARGET=6 -DV26_ZMM_PAIR_WINDOW=1
          -DV30_VBMI_BLOCKERS=1 -DV33_WINNER_HLUT=1 -DV37_QBASE_PAIR=1
          -DV35_PAIR_MASKS=1 -DV34_TRUST_OFFICIAL_INPUT=1
          -DV34_TRUST_GOLD_RANGE=1 -DV40_ASSUME_POSITION_RANGE=1"

LATENCY="-DV44_ROWOFF=1 -DV44_MASKVEC=1 -DV44_TAIL=2
         -DV46_COL5=1 -DV46_IDXFUSE=1 -DV46_DEADINIT=1 -DV49_QBASE16=1"

STRATEGY="-DV50_NOREV_GLOBAL=1"

GCHAIN="-DNOREV_DIRECT_EMIT=1 -DNOREV_RAW_GOLD=1
        -DV47_KEYFUSE=1 -DV47_KMASK=1 -DV47_GDFUSE=1
        -DNOREV_PAIR_EMIT=1"

# G26 / G27 / G25 stay off: all three are bit-exact with G14 but none reduced
# online latency (see README).
OFF="-DNOREV_MASKED_PREV_PAD=0 -DNOREV_PAIR_PACK_PERMB=0 -DNOREV_PAIR_COMPACT=0"

# Explicit ISA subset, NOT -march=native/znver5 (znver5 measured +12.5 ns online).
ISA="-mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mbmi2 -mavx512vbmi"

LINK="-flto -shared -Wl,-z,defs -Wl,-s -Wl,--as-needed -nostdlib++ -static-libgcc"

ROUNDS=${ROUNDS:-200000}

mkdir -p bin
echo "=== build ==="
# shellcheck disable=SC2086
"$CXX" $COMMON $SEMANTIC $LATENCY $STRATEGY $GCHAIN $OFF $ISA \
       -Isrc -o "$OUT" src/player_v7.cpp $LINK
stat -c '%n %s bytes' "$OUT"
sha256sum "$OUT"

echo "=== SHA vs frozen ==="
got=$(sha256sum "$OUT" | cut -d' ' -f1)
echo "built  $got"
echo "frozen $FROZEN_SHA"
if [ "$got" = "$FROZEN_SHA" ]; then
  echo "MATCH: byte-identical to the frozen submission artifact"
else
  echo "MISMATCH -- do not submit this build"; exit 1
fi

echo "=== ELF / ABI ==="
echo -n "exports (must be 1): ";            nm -D --defined-only "$OUT" | grep -c ' T '
echo -n "moveDecision export (must be 1): ";nm -D --defined-only "$OUT" | grep -c moveDecision
echo -n "undefined symbols: ";              nm -D -u "$OUT" | wc -l
echo -n "DT_NEEDED (must be 0): ";          readelf -d "$OUT" | grep -c NEEDED || true
echo -n "ldd: ";                            ldd "$OUT" 2>&1 | head -1
g=$(objdump -T "$OUT" | grep -o 'GLIBC_[0-9.]*' | sort -V | tail -1 || true)
echo "max glibc: ${g:-none (no versioned glibc symbols)}"
echo -n "zmm (whole file): ";               objdump -d "$OUT" | grep -c '%zmm' || true
echo -n "ymm (whole file): ";               objdump -d "$OUT" | grep -c '%ymm' || true

# Clamp the disassembly to the symbol's own [addr, addr+size).
# `--disassemble=moveDecision` is NOT safe: `-Wl,-s` strips local helper names, so objdump
# starts on the preceding unnamed function (there is a `mov %rsp,%rbp` at 0x2e6, before
# moveDecision's 0x320 -- a naive grep reports a false stack reference) and also runs past
# the end, mis-decoding bytes until .fini.
addr=$(nm -S -D "$OUT" | awk '/ T moveDecision$/{print $1; exit}')
size=$(nm -S -D "$OUT" | awk '/ T moveDecision$/{print $2; exit}')
start=$((0x$addr)); end=$((start + 0x$size))
echo "moveDecision: addr 0x$addr size $((0x$size)) bytes (expect 1862)"
echo -n "%rsp refs inside moveDecision (must be 0): "
objdump -d --start-address=$start --stop-address=$end "$OUT" | grep -c '%rsp' || true
# No instruction count here: GNU objdump and llvm-objdump line-break long EVEX encodings
# differently (378 GNU lines vs llvm-objdump's 338 instructions), so a hardcoded number
# would be tool noise.  The SHA match above is the identity check.

echo "=== bit-exact equivalence (ordered replay, dlopen real .so) ==="
"$CXX" -std=c++17 -O2 -Ivalidation -Isrc -o validation/so_equiv validation/so_equiv.cpp -ldl
echo "-- vs G12 (the predecessor G14 was promoted over) --"
./validation/so_equiv validation/player_norev_g12_raw_fused_b984d312.so "$OUT" "$ROUNDS"
echo "-- vs G26 (pure codegen variant, must also be identical) --"
./validation/so_equiv validation/player_norev_g26_pair_maskedprev_dc18889f.so "$OUT" "$ROUNDS"

echo "=== pinned WARM/COLD -- EXCLUSION GATE ONLY, never promote on this ==="
"$CXX" -std=c++17 -O2 -Ivalidation -Isrc -o validation/bench_so validation/bench_so.cpp -ldl
taskset -c 5 ./validation/bench_so "$FROZEN" "$OUT"

echo "ALL CHECKS PASSED"
