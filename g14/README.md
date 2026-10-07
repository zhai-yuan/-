# NOREV-G14 Paired Action Emit — 自包含发布目录

NOREV-G 低延迟续线的**当前冠军**。本目录可独立上传到构建服务器编译，产物与冻结提交物**逐字节相同**（已实测）。

## 提交文件

```text
bin/player_norev_g14_pairemit_c74bea0c.so
```

| 项 | 值 |
|---|---|
| 大小 | 14,504 字节 |
| SHA-256 | `c74bea0c7152d7723e8f1f6937ca074a3a2b01c07eab9371ddfb2a0ae267bb5a` |
| 语言 | C++（`language_id=2`） |
| `moveDecision` | **1,862 字节 / 338 条指令**（llvm-objdump 口径），入口 `0x320` |
| 导出符号 | 仅 `moveDecision` |
| `DT_NEEDED` | 0（`ldd` = `statically linked`） |
| 栈帧 / `%rsp` 引用 | 0 / 0 |
| branches / push-pop | 4 / 6 |
| zmm / ymm | 91 / 63 |

## 编译工具

| 项 | 值 |
|---|---|
| CPU | AMD EPYC 9T25 128-Core（**与比赛机同型号**） |
| OS | Linux 6.12 x86_64 / EL10 |
| **编译器** | **clang++ 21.1.8（RESF 21.1.8-1.el10）** |
| 链接器 | GNU ld 2.41 |
| 需要的 ISA | AVX2 + AVX-512 F/VL/BW/DQ/**VBMI** + BMI2 |
| 运行期依赖 | 无 |

**必须 clang++。** 同源码 g++ 产物线上 p50 慢 **267 ns**（10 局配对自对战，t = 11.5）。
`Makefile` 与 `build.sh` 都有 `toolchain` 门禁拦住非 clang。

**Makefile 里不能写 `CXX ?= clang++`**：GNU Make 预置 `CXX = g++`（origin `default`），
`?=` 不生效，会**静默用 g++**。本 Makefile 用 `ifeq ($(origin CXX),default)`。

## 编译指令

### 方式一：在构建服务器上直接编译（推荐）

把本目录整个拷到服务器，然后：

```bash
cd release/g14
make verify                       # 编译 + ELF/ABI 体检 + 与冻结 SHA 比对
make equiv   ROUNDS=200000        # 对 G12（G14 晋级时的前身）逐位等价
make equiv26 ROUNDS=200000        # 对 G26（纯代码生成变体）逐位等价
make bench                        # 绑核 WARM/COLD（仅排除门禁）
```

或者一条脚本跑完全部检查（不依赖 GNU Make）：

```bash
cd release/g14
bash build.sh
```

### 方式二：从 Windows 一条命令完成上传 + 远端编译 + 门禁

```powershell
python release/g14/validation/remote_make.py --targets verify equiv equiv26 --rounds 200000
```

### 最小编译命令（只要 `.so`，不做检查）

```bash
clang++ -std=c++17 -O3 -DNDEBUG -fPIC -fno-exceptions -fno-rtti -fno-plt \
        -fno-stack-protector -fomit-frame-pointer \
        -DV7_VEC_WINDOW=1 -DV7_AVX512=1 -DV7_INLINE_SOLVE=1 \
        -DV7_FAST_SELECT=0 -DV7_QBASE_LUT=0 -DV7_PREFETCH=0 \
        -DV7_ARITH_EMIT=0 -DV7_ARITH_PARENT=0 -DV7_ARITH_QBASE=0 \
        -DV7_TRAMPLE=0 -DV7_SNAPSHOT=0 -DV7_PACK_TABLES=1 \
        -DV7_LUT16=0 -DV7_RICH=0 -DV7_OPENNESS=0 -DV7_ZONE=0 \
        -DV7_SPLIT_TARGET=6 -DV26_ZMM_PAIR_WINDOW=1 \
        -DV30_VBMI_BLOCKERS=1 -DV33_WINNER_HLUT=1 -DV37_QBASE_PAIR=1 \
        -DV35_PAIR_MASKS=1 -DV34_TRUST_OFFICIAL_INPUT=1 \
        -DV34_TRUST_GOLD_RANGE=1 -DV40_ASSUME_POSITION_RANGE=1 \
        -DV44_ROWOFF=1 -DV44_MASKVEC=1 -DV44_TAIL=2 \
        -DV46_COL5=1 -DV46_IDXFUSE=1 -DV46_DEADINIT=1 -DV49_QBASE16=1 \
        -DV50_NOREV_GLOBAL=1 \
        -DNOREV_DIRECT_EMIT=1 -DNOREV_RAW_GOLD=1 \
        -DV47_KEYFUSE=1 -DV47_KMASK=1 -DV47_GDFUSE=1 \
        -DNOREV_PAIR_EMIT=1 \
        -DNOREV_MASKED_PREV_PAD=0 -DNOREV_PAIR_PACK_PERMB=0 -DNOREV_PAIR_COMPACT=0 \
        -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mbmi2 -mavx512vbmi \
        -Isrc -o bin/player_g14.so src/player_v7.cpp \
        -flto -shared -Wl,-z,defs -Wl,-s -Wl,--as-needed -nostdlib++ -static-libgcc
```

**`-march=native` / `-march=znver5` 是故意不用的。** znver5 更小（−33 B / −18 条）且离线
COLD 最好，但线上配对 p90 是 **+12.5 ns**。显式钉住 ISA 子集也让产物跨服务器内核/编译器
更新仍可复现。

## 服务器实测（本目录已验证）

```text
built  c74bea0c7152d7723e8f1f6937ca074a3a2b01c07eab9371ddfb2a0ae267bb5a
frozen c74bea0c7152d7723e8f1f6937ca074a3a2b01c07eab9371ddfb2a0ae267bb5a
MATCH: byte-identical to the frozen artifact

exports 1 / moveDecision export 1 / undefined 4 / DT_NEEDED 0
ldd: statically linked        max glibc: none
moveDecision 1862 bytes       %rsp refs inside moveDecision: 0
zmm 91  ymm 63

so_equiv vs G12: fixture 1024 + random 200000 = 201,024 compared, 0 differ  IDENTICAL
so_equiv vs G26: fixture 1024 + random 200000 = 201,024 compared, 0 differ  IDENTICAL
```

## 目录内容

```text
Makefile                                              带 toolchain 门禁的可复现构建
build.sh                                              等价的一体脚本（不依赖 GNU Make）
src/player_v7.cpp                                     共享源码（全部变体的编译期开关都在里面）
src/game_api.h                                        官方 ABI 头
bin/player_norev_g14_pairemit_c74bea0c.so             冻结提交物
validation/so_equiv.cpp                               逐位等价（dlopen 真 .so，有序回放）
validation/bench_so.cpp                               绑核 WARM/COLD（仅排除门禁）
validation/rank1_rounds.h                              1,024 个真实正式 fixture 回合
validation/player_norev_g12_raw_fused_b984d312.so      G14 晋级时的对照前身
validation/player_norev_g26_pair_maskedprev_dc18889f.so G26，必须也逐位相同
validation/remote_make.py                              一键上传 + 远端构建 + 门禁
```

## 开关说明

| 组 | 开关 | 含义 |
|---|---|---|
| 语义（继承 V40） | `V7_*` / `V26_*` / `V30_*` / `V33_*` / `V34_*` / `V35_*` / `V37_*` / `V40_*` | 改这些就是改策略，不只是改代码生成 |
| 纯延迟 | `V44_ROWOFF` `V44_MASKVEC` `V44_TAIL=2` `V46_COL5` `V46_IDXFUSE` `V46_DEADINIT` `V49_QBASE16` | 各自对前身逐位等价 |
| 策略 | `V50_NOREV_GLOBAL=1` | NOREV-G1 全局 `gold > not_reversed > distance`，编译期烤进 QBASE16 表，**运行期零指令** |
| G 链 | `NOREV_DIRECT_EMIT`（G6）`NOREV_RAW_GOLD`（G11）`V47_KEYFUSE/KMASK/GDFUSE`（G12）`NOREV_PAIR_EMIT`（**G14**） | G14 本体：两角色路线拼成 8 字节，一次 `vpmovzxbd + 32B store` |

### 三个刻意关掉的开关

它们都与 G14 **决策逐位相同**，但线上都没有降低延迟：

| 开关 | 版本 | 静态 | 服务器 | 线上 |
|---|---|---|---|---|
| `NOREV_MASKED_PREV_PAD` | G26 | branches 4→2，但 +27 B、`rip_loads` 39→43 | **WARM +0.30 ns（7/7 更慢）** / COLD −0.60 ns（6/7 更快） | **70 局**：p90 `−1.36 ± 2.21`、p50 `+0.45 ± 1.58`、先手率 `0.5073`、净金币 `−0.24` ⇒ 不可区分 |
| `NOREV_PAIR_PACK_PERMB` | G27 | `bzhi`/`shlx` 消失、`vpermb` 1→2，指令数不变 | COLD 31.4 vs 32.6（**7/7 更快**） | 14 局：p50 **+2.14 ± 2.81**、p90 **+3.57 ± 3.57** ⇒ 不采纳 |
| `NOREV_PAIR_COMPACT` | G25 | `.rodata` −2.3 KiB | 打平 | 14 局：p90 `130/130` ⇒ 中性 |

**G26 还有一条独立的维护风险**：它把 `PersistentState` 的 `prev_quadrant[2]` 扩成 `[4]`，
**每回合故意往两个填充字节写入**（源码注释：*"deliberate masked-store padding"*）。当前正确，
但以后往该结构体加字段可能被这次压缩写回静默覆盖；它还有 `#error` 钉住的耦合前提
（要求 `V47_KEYFUSE && V44_TAIL>=2`）。G14 没有这些约束。

## 两个容易踩的工具坑（本目录已修）

**① 不要用 `objdump -d --disassemble=moveDecision` 做静态统计。** `-Wl,-s` 剥掉了局部符号名，
所以 objdump 两端都会越界：它从**前一个匿名函数**开始（`0x2e6` 处有一条 `mov %rsp,%rbp`，
在 moveDecision 的 `0x320` 之前，会让朴素的 `grep -c '%rsp'` 误报一次栈引用），
尾部又一直误解码到 `.fini`。正确做法是用 `nm -S` 取符号地址与大小，再
`--start-address` / `--stop-address` 精确夹取 —— 本目录的 `elf` 目标就是这么做的。

**② 不要在发布检查里硬编码指令条数。** GNU objdump 与 llvm-objdump 对长 EVEX 编码的换行
方式不同（GNU 数出 378 行，llvm-objdump 数出 338 条指令）。**SHA 一致才是权威身份检查**，
README 里的 338 是 llvm-objdump 口径（由 `tools/ll_gate.py` 产出）。

## 延迟现状

G14 相对前身 G12：两个独立 map1-7 双座位区块共 28 局，按预注册规则剔除 1 局平台事件后，
跨局中位 p50 `100/110 ns`、p90 `130/140 ns`，逐局平均改善约 `9.63 / 9.74 ns`。

**目标 `p90 ≤ 120 ns` 尚未达成。** 当前可重复的跨局中位是 **130 ns**，
不要把个别 `120 ns` 的对局写成整体达标。

`bench` 目标只用于排除明显退步。本项目已多次实测**离线排序在线上反转**
（znver5 离线最好、线上 +12.5 ns；G27 的 COLD 是 7/7 一致改善、线上仍 +3.57 ns），
所以**任何晋级都必须由正式对局的 `cost` 决定**。
