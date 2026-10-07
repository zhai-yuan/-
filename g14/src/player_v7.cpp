// player_v7.cpp —— GoldRush V7 内核
//
// 基线语义 = V6 B13 Clipped（见 V6_STRATEGY_DESIGN.md）。所有新特性和延迟优化
// 都在编译期开关后面；把 V7_TRAMPLE / V7_SNAPSHOT 关掉时输出与 V6 逐位一致，
// 由 harness cmp 卡住。
//
// 新增语义（相对 V6）
//   V7_TRAMPLE   NPC 踩踏：只对「路径经过且该格 NPC 数 >= 3」的格子扣
//                ceil(0.05 * 该角色当前持币)。NPC <= 2 的格子完全忽略。
//                用 num_visible_npcs >= 3 做近乎免费的守门谓词。
//   V7_SNAPSHOT  区域快照：每 5 轮到达时在冷分支里挑出金币密度最高的区域，
//                把象限偏好的目标点从固定的地图中心 (8,8) 换成该区域质心。
//                热路径成本与 V6 完全相同（两次比较），因为只是把常量 8
//                换成一个持久化的 byte。
//
// 可见敌方角色：与 V6 一致，直接当作阻挡写入窗口（不预测其行为）。
//
// 延迟优化（保持逐位等价）
//   V7_VEC_WINDOW  向量化无分支窗口构造：钳位宽读 + 寄存器内分类 + vpermd
//                  归一化 + 掩码擦除，取代 50 次标量 load + 50 次 LUT load
//                  + 50 次 store，并消掉「内部/边缘」这个 58% 命中率的坏分支
//                  和它守着的冷函数调用。
//   V7_FAST_SELECT 打包比较键 + 掩码选择，取代 choose_k / result_key /
//                  状态更新里的 6~8 个数据相关分支。
//   V7_QBASE_LUT   qbase 只有 20 种取值，预计算成 20 x __m128i 表。
//   V7_AVX512      用 AVX-512VL 掩码搬移取代 AVX2 的 and+cmpeq+blendv
//                  三件套（1 条 vs 4 条），并获得 32 个向量寄存器。
//                  只用 EVEX 编码的 256 位操作，不引入 512 位宽度惩罚。

#include "game_api.h"

#include <array>
#include <cstdint>
#include <immintrin.h>

// ---------------------------------------------------------------- 开关

#ifndef V7_VEC_WINDOW
#define V7_VEC_WINDOW 1
#endif

// V26: construct the two unit windows as one register-resident pair.  This is
// deliberately a separate switch from the historical
// GOLDRUSH_PAIR_WINDOW_BUILD experiment: that experiment only interleaved two
// scalar 25-cell loops and still materialised both arrays in memory.  Here a
// ZMM lane group classifies the matching row for both units, and the resulting
// DP inputs are already packed as [unit0 Q0..Q3, unit1 Q0..Q3].
#ifndef V26_ZMM_PAIR_WINDOW
#define V26_ZMM_PAIR_WINDOW 1
#endif

// V30 second independent factor: the exact V25 blocker projection, now
// combined with V26's pair-window kernel.  This merges both units' six
// occupant projections into one AVX-512VBMI pipeline.
#ifndef V30_VBMI_BLOCKERS
#define V30_VBMI_BLOCKERS 1
#endif

#if V30_VBMI_BLOCKERS && !defined(__AVX512VBMI__)
#error "V30_VBMI_BLOCKERS requires -mavx512vbmi"
#endif

// 实测为负，默认关闭：GCC 本来就把三路 joint 比较编成 cmov，
// 掩码选择只是净增指令（本机 bench2 中位 66.4 -> 74.5 ns）。
#ifndef V7_FAST_SELECT
#define V7_FAST_SELECT 0
#endif

// 实测为负，默认关闭：20 项 x 16B 的表用数据相关索引加载，
// 比十几条 ALU 位运算更贵（本机 bench2 中位 74.5 -> 89.6 ns）。
#ifndef V7_QBASE_LUT
#define V7_QBASE_LUT 0
#endif

// 实测为正：AVX-512VL 的掩码搬移省掉 and+cmpeq+blendv 三件套，
// 并把可用向量寄存器从 16 提到 32，消掉行向量溢出。
#ifndef V7_AVX512
#define V7_AVX512 1
#endif

#if V26_ZMM_PAIR_WINDOW && (!V7_VEC_WINDOW || !V7_AVX512)
#error "V26_ZMM_PAIR_WINDOW requires V7_VEC_WINDOW=1 and V7_AVX512=1"
#endif

// V6 的窗口数据天然必须落在内存里（solve 收 int32_t* 指针）；V7 的窗口是在
// 寄存器里算出来的，如果 solve 是 noinline，160 字节的 UnitWindow 就被迫
// 存栈再读回 —— 纯亏。内联掉它让整条链留在寄存器里。
#ifndef V7_INLINE_SOLVE
#define V7_INLINE_SOLVE 1
#endif

// NPC 踩踏。只关心「同一格 NPC >= 3」的格子，<= 2 完全忽略。
//
// 0 = 关
// 1 = 罚分模式（模板特化版）：给踩踏格减 ceil(0.05*持币)。语义最准，但为了
//     「无踩踏时零成本」用了 WithTrample 模板，代码体积 19136->23232，
//     线上实测 p50 +50ns —— 体积和守门一起把收益吃光了。
// 2 = 硬阻挡模式：持币超阈值时把踩踏格并进已有 block_bits 通道。热路径零
//     新增指令、零体积增长，但线上实测阈值 260 下踩踏事件 5.40->5.50 根本
//     没触发，白付约 20ns（成本全在读 visible_npcs 那 2 条冷 cache line）。
// 3 = 罚分模式（无模板）：始终做一次掩码减，subtrahend 是运行期 penalty
//     （无踩踏时为 0，等于空操作）。没有模板就没有体积膨胀，代价固定
//     约 10 条掩码减指令。语义与 1 完全相同。
// 4 = key 位域模式：把「路径上非踩踏格数」编码进 key 的空闲 bit，靠 DP
//     已有的 add 自然累加，subtrahend 是编译期常量。代价与 3 相同，
//     但排序语义变成「金币 > 少踩踏 > 距离」而不是按金币精确定价。
#ifndef V7_TRAMPLE
#define V7_TRAMPLE 3
#endif

#if V26_ZMM_PAIR_WINDOW && V7_TRAMPLE
#error "V26 pair-window candidate is an exact V22/TRAMPLE=0 kernel"
#endif

// 硬阻挡/罚分模式的守门阈值：持币达到这个数才去读 NPC 数组。
// 依据：踩踏损失 = ceil(0.05*持币)，格子收益 = ceil(0.65*金币)。
// 阈值抬高能直接减少触及 visible_npcs 冷 cache line 的回合数，而踩踏损失
// 本身与持币成正比 —— 也就是说抬阈值砍掉的是「代价最小」的那部分回合。
#ifndef V7_TRAMPLE_BLOCK_GOLD
#define V7_TRAMPLE_BLOCK_GOLD 400
#endif

// 0 = 关（等价 V6 的朝中心）; 1 = 全局最富区域; 2 = 双角色分头（角色0 最富、角色1 次富）
#ifndef V7_SNAPSHOT
#define V7_SNAPSHOT 1
#endif

// 中心区域的加权分子/分母，用来调「多容易离开中心」。1/1 = 纯密度比较。
#ifndef V7_SNAP_CENTER_NUM
#define V7_SNAP_CENTER_NUM 1
#endif
#ifndef V7_SNAP_CENTER_DEN
#define V7_SNAP_CENTER_DEN 1
#endif

#ifndef V7_BOMB_PHASE_ROUND
#define V7_BOMB_PHASE_ROUND 13
#endif

// 实测为负，默认关闭。
// 本机 bench2 显示预取有约 4ns 收益，但线上 A/B（第三轮，10 对配对）反向：
// p50 200->210、p90 270->330、净金币比值 1.110->0.860。
// 解释：引擎在调用前刚写过整个 GameInput，grid 本来就在 L1/L2，10 条
// prefetchT0 纯属额外 uop 并占用 load 端口。这条只有线上能判，本机会误导。
#ifndef V7_PREFETCH
#define V7_PREFETCH 0
#endif

// 用算式取代 emit 的两张表。
//   0 = 全查表（V7 原样）：HORIZONTAL_LUT 320B + ACTION_LUT 1280B
//   1 = 全算式：两张表都不要，零内存访问，但多约 22 条 ALU 指令
//   2 = 折中：保留小的 HORIZONTAL_LUT(320B)，只把大的 ACTION_LUT(1280B)
//       换成 pdep 算式。去掉 80% 的表字节，只付一半的算术代价。
//
// 这两张表合计占 .rodata 的 62%，而且每次调用都用数据相关索引跨表随机跳。
// 线上两次调用之间引擎会重写整个 GameInput 并推进世界，D-cache 大概率被冲掉，
// 所以这些 load 很可能次次 miss L1；而本机 benchmark 里它们一直是热的。
// 本机实测算式版慢约 4ns —— 这个方向的取舍**只能靠线上自对战判定**。
#ifndef V7_ARITH_EMIT
#define V7_ARITH_EMIT 1
#endif

// 用 BMI2 的 pdep 取代 SPREAD_PARENT_LUT（把 4 个 bit 散布到 0/4/8/12 位）。
// pdep 在 Zen3 起是 1 uop / 3 周期，等价且省掉一张表。
#ifndef V7_ARITH_PARENT
#define V7_ARITH_PARENT 1
#endif

// V33: keep the four DP parent decisions in plane-major form and extract only
// the two winning quadrants after choose_k.  The control value 0 preserves the
// frozen V30 representation byte-for-byte; the candidate removes eight spread
// table loads and replaces them with two winner-only BMI2 pext instructions.
#ifndef V33_WINNER_HLUT
#define V33_WINNER_HLUT 1
#endif

// V37: expand both packed TAIL_LUT words directly into the single YMM qbase
// consumed by paired DP.  The control value 0 preserves frozen V33 bytewise.
#ifndef V37_QBASE_PAIR
#define V37_QBASE_PAIR 1
#endif

// V38 stacks the already isolated V35 packed two-unit edge/occupant masks.
#ifndef V35_PAIR_MASKS
#define V35_PAIR_MASKS 1
#endif

// V38 also enables the empirically audited official-input contract fast path.
// 1,438 official games contained no null/out-of-range own positions and no
// visible cell above GOLD_EXACT_MAX (observed maximum 610).
#ifndef V34_TRUST_OFFICIAL_INPUT
#define V34_TRUST_OFFICIAL_INPUT 1
#endif
#ifndef V34_TRUST_GOLD_RANGE
#define V34_TRUST_GOLD_RANGE 1
#endif

// V40: make the already-audited own-position contract visible to Clang's
// range analysis.  This emits no checks; it only removes impossible halves of
// the five row/column clamp chains.
#ifndef V40_ASSUME_POSITION_RANGE
#define V40_ASSUME_POSITION_RANGE 1
#endif

// ---------------------------------------------------------------- V44 标量轴
//
// V44 只动 V40 里那半数的**标量整数**工作，三个簇各一个开关，默认 0 = 逐字节复现 V40。
//
// A. V44_ROWOFF —— 行地址。V40 对五个窗口行 x 两个角色各算一次
//    `rr = clamp(r-2+k,0,16)` 再算 `&grid[rr][cb]`；`__builtin_assume` 之后仍剩 8 条
//    `cmovbl`，加上每行 `shl/lea/add/add` 的 68 倍乘。改成查预钳位好的行字节偏移表，
//    于是每行只剩一条 `movz` + 折进寻址模式的 `(base,idx,scale)`。
//      1 = uint16 绝对偏移表 `17*clamp(r-2+k,0,16)`，base = &grid[0][cb]，
//          最少指令（连 r*68 都不用算），代价 170B .rodata。
//      2 = uint8 相对偏移表 `17*(clamp(r-2+k,0,16)-r)+34`（取值 0..68），
//          base = &grid[r][cb]，偏移用 `-136(base,idx,4)` 的位移折回来，
//          .rodata 只 85B，但每角色要多算一次 r*68。
//    **不采用「一个 base 读五条连续行」的方案**，理由见 README：把 slot 与行的错位
//    折进任何已有量都不成立，唯一无分支的形式要求 base = grid + 68*(r-2)，
//    在 r<2 时落到 GameInput 对象前最多 132 字节 —— 越界读，不可接受。
#ifndef V44_ROWOFF
#define V44_ROWOFF 0
#endif

// B. V44_MASKVEC —— 掩码搬运。
//    classify 里 `negative & odd & ~is_bomb` 用了 vpmovd2m x2 + vpcmpneqd + kandd x2
//    + knotw，每行 5 个 k 操作、共 25 个。raw 已被钳到 >= -6，所以负数只可能是
//    -1..-6，于是
//        bombpat = -3 时 blocked == {-1,-5} == sign & bit0 & bit1
//        bombpat =  0 时 blocked == {-1,-3,-5} == sign & bit0
//    两种情况都是「(raw & D) == D」，D 只随相位在 0x80000003 / 0x80000001 之间切换。
//    于是整条掩码链塌成一条 `vpandd` + 一条 `vpcmpeqd`。
//    `~negative` 那次 maskz 也不需要了：把 raw 先 `max(raw,0)`，raw<0 的格子
//    score = (0*MUL+ADD)>>8 = 1945，被 `& ~2047` 抹成 0，与 V40 的 maskz 结果相同。
//      1 = 只改 classify
//      2 = 1 + 擦除掩码：把 colmask 预先 or 进那条 xmm，再用一次 vmovq + kmovq +
//          三次 kshiftrq 搬前四行，第五行单独走 vpextrw
#ifndef V44_MASKVEC
#define V44_MASKVEC 0
#endif

// C. V44_TAIL —— choose_k / emit 尾。
//    cap2/exact3/exact4 三次水平约简各 2 vpshufd + 2 vpmaxsd = 12 条。
//    但 a3 = hmax(max(m2,e3))、a4 = hmax(max(m3,e4))，所以可以**先**把跑动最大值
//    做出来，再把两个 8 lane 量塞进一条 ZMM，一对 permute+max 同时约简两个。
//      1 = 跑动最大值 + m2|m3 打包进一条 ZMM 约简（m4 单独 ymm 约简），
//          并把「非法退化成 STAY」的两次比较折成向量 max(.,0)
//      2 = 1 再把整个 choose_k 搬进向量：一条 vpermt2d 把六个约简值打包成
//          AB = [a2,a3,a4,a2 | b4,b3,b2,b4]，于是
//            * 三个 joint 和 = (AB&GD) 的两半相加，一条 vpaddd 全出；
//            * 边际比较不用算两个差：marginal1 > marginal0 等价于
//              (a2&GM)+(b4&GM) > (a4&GM)+(b2&GM)，即同一条 T 向量的 lane0 与
//              lane2 比大小，一次 vpcmpgtd；
//            * k 标签用一条 vpternlogd 装出来（同时把 argmax 用的 lane 号
//              折进同一个常量）；
//            * key0/key1 由一条 vpermd 取出，只剩两次 vector->GPR 搬移
//              （V40 要六次）。
//          lane3 刻意复制 lane0 且带同一个 k 标签，所以它就算并列胜出也给出
//          相同的 k 与 key，不需要额外的 zero-mask。
#ifndef V44_TAIL
#define V44_TAIL 0
#endif

#if (V44_ROWOFF || V44_MASKVEC || V44_TAIL) && !V26_ZMM_PAIR_WINDOW
#error "V44 scalar axes target the V26/V40 pair-window kernel"
#endif
#if V44_MASKVEC && !V34_TRUST_GOLD_RANGE
#error "V44_MASKVEC drops the raw_max tracking path; it needs V34_TRUST_GOLD_RANGE=1"
#endif
#if V44_TAIL && V7_TRAMPLE
#error "V44_TAIL folds STAY_KEY into a vector max; that assumes STAY_KEY == 0"
#endif
#if V44_MASKVEC >= 2 && !V35_PAIR_MASKS
#error "V44_MASKVEC=2 rewrites the V35 packed two-unit erase mask"
#endif

// ---------------------------------------------------------------- V46 合并轴
//
// V46 把另外两条支线上已经**单独验证过**的三个瘦身合并到 V44STK
// （`V44_ROWOFF=1 V44_MASKVEC=1 V44_TAIL=2`，2157B / 396 条）之上。
// 三个开关默认 0；全 0 时产物必须逐字节复现 V44STK（`adedbcce…`），
// 也就等价于冻结 V40（V44STK 已被 201,024 回合证明与 V40 逐位同决策）。
//
// 每个开关都是从别处**原样搬过来**的，不做二次发明；它们各自的编译期证明
// 一并搬来，所以「表值等价」这一层不依赖任何新论证。
//
// D. V46_COL5 —— 列越界掩码改成「每个 off 一条 5 字节窗口行模式 {q,q,h,q,q}」。
//    源自 zy_v43_shuffle 的 V43_COL5。关键观察：COLMASK_Q/COLMASK_H 只在
//    c0 ∈ {0,1,15,16} 非零，而这四个 c0 与 off = (c0-2)-cb0 ∈ {-2,-1,4,5}
//    一一对应；c0 ∈ [2,14] 全部落到 off ∈ [0,3] 且掩码恒为 0。所以列掩码是
//    off 的函数，可以用**已经算好**的 (off+2) 直接索引，不必再读两张以 c0 为
//    索引的表。热路径里 4 条 movzbl + 两组 shl/or + 每行一条 or 变成
//    2 条 vmovq + 1 条 vpunpcklbw + 1 条 vpor，并且少两个长活标量值
//    （colq_pair / colh_pair 原本活到最后一行）。
//    在 V43 的栈上实测 −19B / −11 条且 push/pop 10→8；但 V43 也实测**单独**打开
//    时 push/pop 反而 10→12 —— 同一个改动在不同寄存器压力下方向相反。
//    V44STK 的 push/pop 已经是 8，所以这里必须在 V44STK 上实测，不能外推。
#ifndef V46_COL5
#define V46_COL5 0
#endif
#if V46_COL5 && !V35_PAIR_MASKS
#error "V46_COL5 rewrites the V35 packed two-unit erase mask assembly"
#endif
// V46_COL5 与 V44_MASKVEC=2 都改写擦除掩码的装配，但**方向互补**，所以这里刻意
// 允许两者共存（V44 轴上不可能测到这个组合，因为那时还没有 COL5）：
//   V44_MASKVEC=2 在 V44 轴上被判 NO-GO，理由是「把 colmask 折进 XMM」要花
//   vpbroadcastw + vpinsrw + vpor 三条，比它省下的五条 `orl` 更贵（净 +18B）。
//   COL5 把同一个折叠变成**一次表读**，构造成本归零。所以叠在 COL5 上时，
//   =2 剩下的净差只是「一次 kmovq + 三次 kshiftrq + 一次 vpextrw」对
//   「五次 vpextrw + 五次 kmovd」—— 这是一个全新的组合，必须实测。
#if V46_COL5 && V44_MASKVEC >= 2 && !V35_PAIR_MASKS
#error "the COL5 + MASKVEC=2 combination rewrites the V35 packed erase mask"
#endif

// E. V46_IDXFUSE —— IDX_Q / IDX_H 交错进一张表。
//    源自 zy_v43_shuffle 的 V43_IDXFUSE。同一个 off 的两个索引向量放进一个
//    `alignas(64)` 结构体（q 在 +0、h 在 +0x20），于是热函数只需要**一个**表
//    基址寄存器（H 用 disp8 = +0x20 寻址），少一条 `leaq (%rip)`；
//    运行时每个角色首次触达的表线也从两条降到一条。
//    注意：本轮线上实测已经证明「压 .rodata cache line」这条**不值钱**
//    （18→11 条 line、文件 −4096B，配对 p50 差 0，se 3.4）。所以这里唯一值得
//    计价的是**指令与字节**，footprint 只当副产品记录。
#ifndef V46_IDXFUSE
#define V46_IDXFUSE 0
#endif

// F. V46_DEADINIT —— 删掉 36 字节的聚合初始化。
//    源自 zy_v41_storefuse 的 V41_STORE_FUSE 变体 5（**只有变体 5**；变体 1~4
//    把整个 GameOutput 融成一次 store，在体积、指令数和两个离线口径上同时退化，
//    已判 NO-GO，不搬）。
//    `GameOutput out = {{STAY x6}, 3, 0, 0}` 的 36 个字节**全部**是死写：
//      actions[0..3] 被角色 0 的 emit 覆盖，actions[k..5] 被角色 1 的 emit 覆盖，
//      而 k ∈ {2,3,4} ⇒ k+4 >= 6，所以六个 STAY 一个都活不下来；
//      k / order 被 emit 之后的修补覆盖，vp 由下面显式写 0。
//    删掉它不引入任何对 k 的新依赖，所以不制造串行链 —— 这正是变体 1~4 的病根。
//    V41 实测 −18B / −3 条 / GameOutput store 6→4 / −1 次 RIP 载入 / −64B .rodata。
#ifndef V46_DEADINIT
#define V46_DEADINIT 0
#endif
#if V46_DEADINIT && !V34_TRUST_OFFICIAL_INPUT
#error "V46_DEADINIT removes the aggregate initialiser that the null-input early return would have returned"
#endif
#if V46_DEADINIT && !V33_WINNER_HLUT
#error "V46_DEADINIT is written against the V33 winner-only parent plane emit"
#endif

// ---------------------------------------------------------------- V49 pacing 轴
//
// 两个开关，默认全 0；全 0 时产物必须逐字节复现 V46STK（`692fc6c4…`），
// 也就等价于冻结 V40 与冻结 V22（V46STK 已被 201,024 有序回合双向证明）。
//
// 这一轴要解决的行为问题（离线已量，30 局正式日志）：够不到金币时角色会
// UURR / LLDD / UURR 来回打转，在两个对角象限之间振荡而出不去。
// 「≥6 连续回合困在 ≤3 个不同格」的**零产出**空转段每局 3.6 个、占 4.04% 的
// 角色回合、约值 66 金币/局。病根已定位在键序：
//     gold > dist(bit 10..7) > not_reversed(bit 6) > qrank > 象限 > endpoint
// `dist` 偏好**最远**的落点，所以一条 4 步的反向路径压过一条 3 步的非反向路径，
// bit 6 那份 not_reversed 永远轮不到说话。
//
// **显而易见的修法是被禁止的**：无条件把 not_reversed 抬到 `dist` 之上就是 V23。
// V23 打了 24 局正式对局并且输了：`5-0-7`、均差 `−132.6 ± 115.4`。
// 离线它命中了自己的靶子（精确 180° 掉头 20.88% → 14.33%，空转段 2.3 → 0.5/局，
// −85%），但它**同时把蹲守打掉了**：蹲守段 19.8 → 12.8/局（−35%），
// 而蹲守回合的拾金是 3.37 金币/回合、基线只有 1.74。信号本身分不清
// 「被困住、该走」和「有产出、该回来」，因为**蹲守的定义就是反复回到同一片富格，
// 而「回去」在净位移上正是一次精确反向**。
// 另外记死一条：`gold_contract_check.py` 实测 796 个分歧回合里 V23 的即时金币
// 与 V22 **完全相同 796/796、全程 25944 : 25944**。所以
// **「即时金币中性」不是安全性论证，本轴不使用它。**
//
// 本轴只负责「让这个改动的延迟代价 ≤ 0」，**不负责证明它更强**。
// 行为安全性由另一条支线的离线段落前置门禁（空转段下降 **且** 蹲守段不变）判定，
// 在那条门禁通过之前本轴的产物不得打正式对局。
//
// G. V49_QBASE16 —— 纯延迟，行为逐位不变。
//    今天 V37 的成对 qbase 要 9 条指令加两个 32 字节向量常量：
//      2 x movzwl TAIL_LUT + 2 x vpbroadcastd + vinserti128
//      + vpsrlvd(shifts) + vpandd(7) + vpslld(4) + vpord(quadrants)
//    一条 qbase lane 是 `(tail << 4) | (q << 2)`，而 tail <= 7，所以 lane <= 0x7C；
//    带上 bit-10 的副本也只到 0x47C —— **完整落在 uint16 里**。
//    于是把每个 tail_index 的四条 lane 预计算成 8 字节，整条链塌成 4 条：
//      vmovq &QB16[idx0] + vmovq &QB16[idx1] + vpunpcklqdq + vpmovzxwd
//    合计 6 条（含两次 movzwl 形态的索引计算）而不是 9 条，并且 `shifts` 与
//    `quadrants` 这两个常量直接消失。表 = 40 项 x 4 lane x uint16 = 320 字节
//    （前 20 项门关、后 20 项门开，见 H）。
//    **注意本仓已实测 `V7_QBASE_LUT=1` 是负的**（bench2 74.5 → 89.6 ns），但那是
//    20 x 16 字节、两次独立 128 位载入、g++/AVX2 时代的形状。这里是每项 8 字节、
//    每个角色一条 `vmovq`，形状不同 —— **必须重测，两个方向都不得预设**。
#ifndef V49_QBASE16
#define V49_QBASE16 0
#endif
#if V49_QBASE16 && !V37_QBASE_PAIR
#error "V49_QBASE16 replaces the V37 paired qbase expansion chain"
#endif
#if V49_QBASE16 && (V7_ARITH_QBASE || V7_QBASE_LUT)
#error "V49_QBASE16 is mutually exclusive with the other qbase formulations"
#endif

// H. V49_NOREV_GATE —— 行为改动本体，**并且刻意做成零指令**。
//    把 not_reversed 的第二份副本放到 **bit 10**，也就是 `dist` 字段那个恒为 0 的
//    最高位（dist 取值 1..4 只用到 bits 7..9）。这是 V23b 的落位，选它的理由是
//    `GOLD_SHIFT` 留在 11，于是 `NEG` / `VALID_FLOOR` / `GOLD_EXACT_MAX` /
//    魔数 / 三个哨兵**一个都不动**（V23a 放 bit 11 要把 GOLD_SHIFT 抬到 15，
//    一整批嵌在指令里的立即数跟着变）。
//
//    门是**干旱计数器**：只有这个角色连续 V49_DRY 个回合的规划收益为 0 时才置位。
//      * 持久 `uint8 dry_streak[2]`，在 V49_DRY 处**饱和**，在 moveDecision
//        的最末尾用 `choice.keyN >> GOLD_SHIFT` 更新 —— 那个量已经算好了
//        （`V7_ZONE` / `V7_RICH` 用的是同一个），读它零成本。
//        饱和的两个作用：门的判据退化成一次相等比较，且 500 回合的长干旱
//        不可能把 uint8 绕回去把门误关。
//      * **门本身折进表索引，而不是在运行期拼一个向量**：持久
//        `uint8 tail_bias[2]` 在计数器更新时被写成 20 或 0，热路径索引
//        `QB16[(prev_quadrant+1)*4 + preferred_quadrant + tail_bias[u]]`。
//        表加倍到 40 项，上面 20 项已经把 bit-10 副本烤进去了。
//        `tail_bias` 与 `prev_quadrant` / `target_*` 同在一条已经被载入的
//        cache line 上，而那次索引加法折进本来就有的寻址里 ⇒ 热路径 ~0 条新指令。
//      * **刻意只用滞后的干旱信号**。不读「窗口里有没有金币」：`raw_max` 是在
//        窗口构造**内部**产出的，而 qbase 在窗口之前就算完了，所以那一项会加一条
//        **100% 回合都在**的依赖边。台账给合取版的定价是 +35.5 对滞后版 +32.7
//        净金币/局 —— 2.8 金币买不下这条边。
//
//    两个连带处理（两处都必须是零额外指令，并在反汇编里核对）：
//      * **`choose_k` 不能看见 bit 10。** V44_TAIL>=2 路径里
//        `gd = AB & ~((1 << DIST_SHIFT) - 1)`，把立即数改成同时挖掉 bit 10 ——
//        同一条 `vpandd`，只是广播常量不同。两个 dist 相加最多 4+4=8，
//        正好占到 bit 10 而不再往上，所以进位停在 `GOLD_SHIFT` 之下。
//      * **`key_dist` 只读 3 位**，`(key >> DIST_SHIFT) & 7` 而不是 `& 15`，
//        否则 bit 10 置位时 `emit_parent` 会把 `moved` 读错。同一条指令，
//        不同立即数；dist 的 1..4 在 3 位里完整保留。
//
//    **有案在册的 UB 陷阱**：这个位提取的早期版本用了一次融合移位，从 bit 11 挪到
//    bit 10 之后 q=3 那条的移位量变成 **−1，负移位是 UB**。它编译通过、5,988 回合
//    零非法动作，却把掉头率悄悄压到 10.69%，低于 14.13% 的**被迫下限**。
//    所以这里的值全部在编译期烤进表里并穷举断言，**不在运行期算任何移位量**。
#ifndef V49_NOREV_GATE
#define V49_NOREV_GATE 0
#endif

// V50: unconditional global ordering
//
//   gold > not_reversed > dist > original tail/q/endpoint
//
// The high-priority copy is baked into the same 20-entry QB16 table used by the
// behavior-neutral V49_QBASE16 path.  It therefore adds no runtime instruction,
// state byte, branch, load, or cache line.  V50 deliberately does not compile the
// V49 drought gate/state machine.
#ifndef V50_NOREV_GLOBAL
#define V50_NOREV_GLOBAL 0
#endif
#if V50_NOREV_GLOBAL && !V49_QBASE16
#error "V50_NOREV_GLOBAL requires the 20-entry V49_QBASE16 representation"
#endif
#if V50_NOREV_GLOBAL && V49_NOREV_GATE
#error "V50 global and V49 gated not-reverse are mutually exclusive"
#endif
#if V50_NOREV_GLOBAL && V7_NOREV_HIGH
#error "V50 and V7_NOREV_HIGH are two implementations of the same key field"
#endif
#if V50_NOREV_GLOBAL && (V7_OPENNESS || V7_TRAMPLE == 4)
#error "V50 borrows dist bit 10; OPENNESS/TRAMPLE=4 use a different key layout"
#endif

// NOREV-G2 latency axis: update both previous-quadrant bytes with one packed,
// branch-free read/modify/write.  The frozen NOREV-G1 tail has two data-dependent
// `je` instructions here.  They do not affect this round's output, but a miss still
// lands inside the measured moveDecision call and disproportionately raises p90.
// Mode 0 keeps the frozen source shape; mode 1 is required to be bit-exact.
#ifndef NOREV_PACKED_PREV
#define NOREV_PACKED_PREV 0
#endif
#if NOREV_PACKED_PREV < 0 || NOREV_PACKED_PREV > 1
#error "NOREV_PACKED_PREV must be 0 or 1"
#endif

// NOREV-G3 latency axis, ported from the proven-equivalent V47 experiment.
// Build valid keys as zero-masked score|qbase|metadata ternary-logic values.
// This removes the negative-sentinel background copies and the later max(.,0)
// clamps without changing any selected key.
#ifndef V47_KEYFUSE
#define V47_KEYFUSE 0
#endif
#if V47_KEYFUSE && !V44_TAIL
#error "V47_KEYFUSE relies on the V44_TAIL running-max/argmax tail"
#endif
#if V47_KEYFUSE && V7_TRAMPLE
#error "V47_KEYFUSE needs STAY_KEY == 0 and non-negative valid scores"
#endif

// NOREV-G4 latency axis: keep the four parent comparison planes in AVX-512
// mask registers until their final 32-bit concatenation.
#ifndef V47_KMASK
#define V47_KMASK 0
#endif
#if V47_KMASK && !V33_WINNER_HLUT
#error "V47_KMASK writes the V33 winner-only parent plane word"
#endif

// NOREV-G5 latency axis: derive the gold-only joint sums from the already
// accumulated gold+distance sums instead of summing the same lanes twice.
#ifndef V47_GDFUSE
#define V47_GDFUSE 0
#endif
#if V47_GDFUSE && V44_TAIL < 2
#error "V47_GDFUSE rewrites the V44_TAIL=2 vector choose_k"
#endif

// 干旱门的阈值：连续多少个「规划收益为 0」的回合之后才允许惩罚反向。
// 2 是台账上的取值（滞后版净 +32.7/局）；调高更保守。
#ifndef V49_DRY
#define V49_DRY 2
#endif
#if V49_NOREV_GATE && (V49_DRY < 1 || V49_DRY > 254)
#error "V49_DRY 必须落在 1..254：计数器是 uint8 且在阈值处饱和"
#endif

// 调试用：把门永久钉死在关闭状态，用来把「管线接对了」从「行为变了」里隔离出来。
// 打开它之后产物必须逐字节复现基座。
#ifndef V49_FORCE_GATE_OFF
#define V49_FORCE_GATE_OFF 0
#endif

// I. V49_ROWFOLD —— 把 `(prev_quadrant + 1) * 4 + tail_bias` 整体预存成**一个字节**。
//    实测出来的修正项，不是设计时想到的：H 的原始写法（持久 `tail_bias` + 热路径
//    一次加法）在热路径上并不是零成本，反汇编里它实打实多了
//    `movzbl tail_bias` + 一次加法（每角色），因为 `tail_index` 那条链
//      movsbl prev_quadrant / leal 4(,r,4) / or preferred / lea (r,bit1,2)
//    已经把可用的寻址槽位占满了，第三个加数塞不进去。
//
//    修法：既然 `prev_quadrant` 与 `tail_bias` **都只在回合末尾被写**，就把
//    `(prev_quadrant + 1) * 4 + tail_bias` 在那里一次算好存成一个字节。于是热路径
//    从「movsbl + leal + or」变成「movzbl + or」——
//    **不但把门的加法降到零，还比基座少一条指令**。
//    `(prev_quadrant + 1) * 4 <= 16` 与 `tail_bias ∈ {0, 20}` 的低两位都是 0，
//    而 `preferred_quadrant ∈ 0..3` 只占低两位，所以 `or` 与 `+` 等价（下面断言）。
#ifndef V49_ROWFOLD
#define V49_ROWFOLD 0
#endif
#if V49_ROWFOLD && !V49_NOREV_GATE
#error "V49_ROWFOLD folds the V49_NOREV_GATE bias into the stored row index"
#endif

#if V49_NOREV_GATE && !V49_QBASE16
#error "V49_NOREV_GATE bakes the bit-10 copies into the V49_QBASE16 table"
#endif
#if V49_NOREV_GATE && V7_NOREV_HIGH
#error "V49_NOREV_GATE 与 V7_NOREV_HIGH 是同一个字段的两套实现，不能同时开"
#endif
#if V49_NOREV_GATE && (V7_OPENNESS || V7_TRAMPLE == 4)
#error "V49_NOREV_GATE 借的是 dist 的最高位；OPENNESS / TRAMPLE=4 会重排整个 key"
#endif

// 直接向量化算出 qbase，取代 TAIL_LUT 查表 + 4 次位域拆解 + set_epi32。
#ifndef V7_ARITH_QBASE
#define V7_ARITH_QBASE 1
#endif

// 把「每次调用都要碰」的小表打包进一块 64 字节对齐的连续内存。
//
// 动机（本轮实测的地板测量）：线上地板 40ns、v7 总 p50 210ns，所以策略部分
// 是 170ns，而本机同样的工作只要 54ns —— 3.1 倍。已排除的解释：输入不真实、
// 指令数、代码体积、代码布局（PGO 打平）。剩下最可能的是**每次调用的冷缓存
// 首次触达**：引擎在两次调用之间跑了大量代码，把我们的代码和表全冲掉了。
//
// 这一项只改数据布局，不改指令数量：
//   打包前：ROWMASK / COLMASK_Q / COLMASK_H / LANEBITS / TAIL_LUT /
//           SPREAD_PARENT_LUT 是 6 个独立对象，GCC 可能散布在 .rodata 各处，
//           单次调用要碰其中 4~5 条不同的 cache line。
//   打包后：合计 255 字节 = 4 条连续 line，硬件预取器能顺序拿到。
// 线上实测：自对战 15-5 胜未打包版，是本仓第一个赢下来的延迟改动。
#ifndef V7_PACK_TABLES
#define V7_PACK_TABLES 1
#endif

// 把 ACTION_LUT 从 uint32 压成 uint16（每个动作码只要 3 bit，用一条 pdep 展开），
// 1280B -> 640B，20 条 cache line -> 10 条。与打包表同一个靶子（line 数），
// 但针对的是 emit 那两次冷访问，所以两者互不重叠、可以叠加。
// 详见 make_action_lut16 处的注释。
#ifndef V7_LUT16
#define V7_LUT16 0
#endif

// 把 parent 回溯的两级数据相关查表：
//   HORIZONTAL_LUT -> ACTION_LUT
// 合并成一个最终动作表。表项按 (moved, quadrant+endpoint, parents) 排列，
// 热路径只保留 winner parent 的 pext 和一次 uint32 load。默认关闭，供独立
// 延迟归因；只改变表示，不改变任何路径选择语义。
#ifndef NOREV_DIRECT_EMIT
#define NOREV_DIRECT_EMIT 0
#endif

#if NOREV_DIRECT_EMIT < 0 || NOREV_DIRECT_EMIT > 2
#error "NOREV_DIRECT_EMIT must be 0, 1 (full), or 2 (compact)"
#endif

#if NOREV_DIRECT_EMIT && V7_ARITH_EMIT
#error "NOREV_DIRECT_EMIT requires V7_ARITH_EMIT=0"
#endif

#if NOREV_DIRECT_EMIT && V7_LUT16
#error "NOREV_DIRECT_EMIT replaces V7_LUT16 and requires V7_LUT16=0"
#endif

// Expand the two winning four-byte routes together, then write one 32-byte
// action vector.  The store intentionally covers actions[0..5], k and order;
// k/order are written with their final values immediately afterwards.
#ifndef NOREV_PAIR_EMIT
#define NOREV_PAIR_EMIT 0
#endif
#ifndef NOREV_PAIR_PACK_PERMB
#define NOREV_PAIR_PACK_PERMB 0
#endif
#ifndef NOREV_PAIR_COMPACT
#define NOREV_PAIR_COMPACT 0
#endif
#if NOREV_PAIR_PACK_PERMB && !NOREV_PAIR_EMIT
#error "NOREV_PAIR_PACK_PERMB requires NOREV_PAIR_EMIT"
#endif
#if NOREV_PAIR_COMPACT && (!NOREV_PAIR_EMIT || NOREV_DIRECT_EMIT != 2)
#error "NOREV_PAIR_COMPACT requires pair emit with the compact direct-action LUT"
#endif
#if NOREV_PAIR_EMIT && (!V33_WINNER_HLUT || \
                        (NOREV_DIRECT_EMIT != 1 && !NOREV_PAIR_COMPACT))
#error "NOREV_PAIR_EMIT requires winner planes and a matching direct-action LUT"
#endif

// Update the two previous-quadrant bytes while the winning key vector is still
// live in solve_pair.  Two extra bytes are deliberate masked-store padding.
#ifndef NOREV_MASKED_PREV_PAD
#define NOREV_MASKED_PREV_PAD 0
#endif
#if NOREV_MASKED_PREV_PAD && (!V47_KEYFUSE || V44_TAIL < 2)
#error "NOREV_MASKED_PREV_PAD relies on winner key == 0 iff the unit stays"
#endif

// 0: teammate + both visible enemies (frozen behaviour)
// 1: teammate only; enemy current positions are not static walls in a simultaneous turn
// 2: no dynamic occupants (diagnostic upper bound only)
#ifndef NOREV_BLOCKER_SCOPE
#define NOREV_BLOCKER_SCOPE 0
#endif
#if NOREV_BLOCKER_SCOPE < 0 || NOREV_BLOCKER_SCOPE > 2
#error "NOREV_BLOCKER_SCOPE must be 0, 1, or 2"
#endif

// Form the six qword validity bits entirely in vector registers.  This keeps
// collect_blockers_pair's occupant construction, LANEBITS lookup and merge
// unchanged; it only replaces the valid32 -> GPR -> PEXT compression chain.
#ifndef NOREV_BLOCKER_PAIRMAX_VALID
#define NOREV_BLOCKER_PAIRMAX_VALID 0
#endif
#if NOREV_BLOCKER_PAIRMAX_VALID && (!V30_VBMI_BLOCKERS || !V34_TRUST_OFFICIAL_INPUT)
#error "NOREV_BLOCKER_PAIRMAX_VALID requires V30 blockers and official valid self positions"
#endif

// Replace each unit's two-bound column clamp with one 17-byte packed lookup.
#ifndef NOREV_COLGEOM_LUT
#define NOREV_COLGEOM_LUT 0
#endif

// Official cells are exactly {-5,-3,-1,0,positive gold}; max(raw,-6) is an
// identity on that domain and costs one vpmaxsd per window row.
#ifndef NOREV_TRUST_CELL_DOMAIN
#define NOREV_TRUST_CELL_DOMAIN 0
#endif
#if NOREV_TRUST_CELL_DOMAIN && !V34_TRUST_OFFICIAL_INPUT
#error "NOREV_TRUST_CELL_DOMAIN requires the audited official GameInput contract"
#endif

// Keep the complete phase-dependent blocker mask observable as one object.
// Without this guard Clang may encode only a one-byte phase and rebuild the
// 32-bit value in every normal round.  The candidate is strictly semantic-
// equivalent; final assembly decides whether the volatile load is worthwhile.
#ifndef NOREV_DIRECT_BLOCKBITS
#define NOREV_DIRECT_BLOCKBITS 0
#endif

// NOREV-G7: 当前冻结策略的两个目标点是常量 (8,3)/(8,8)。把四次标量
// compare+setcc 合成一次 128-bit 比较和 movemask，同时产出两个角色的
// preferred quadrant。只重写索引形成，不改变 qrank 或 key 的任何 bit。
#ifndef NOREV_SIMD_PREF
#define NOREV_SIMD_PREF 0
#endif

#if NOREV_SIMD_PREF && (V7_SPLIT_TARGET != 6 || V7_SNAPSHOT || V7_RICH || V7_ZONE)
#error "NOREV_SIMD_PREF requires frozen constant targets (8,3)/(8,8)"
#endif

// Semantic latency probe: score a visible gold cell by its raw amount instead of
// ceil(0.65*g). This deletes the vector multiply/add/arithmetic-shift chain from
// all five classified rows. GOLD_SHIFT and every higher-priority key field stay
// unchanged. Unlike the G2-G10 axes this can reorder close-gold paths, so it must
// pass a separate strategy-strength gate before any promotion.
#ifndef NOREV_RAW_GOLD
#define NOREV_RAW_GOLD 0
#endif

// Raw-gold-only representation experiment.  Pack two window rows to signed
// int16, classify there, and use VPMADDWD both to select the interleaved row
// and restore the existing `cell << GOLD_SHIFT` int32 representation.  Unlike
// the rejected V47 int16 shape this contains no 0.65 pricing multiply chain.
#ifndef NOREV_RAW_INT16
#define NOREV_RAW_INT16 0
#endif
#if NOREV_RAW_INT16 && !NOREV_RAW_GOLD
#error "NOREV_RAW_INT16 is only exact for the raw-gold scoring path"
#endif
#if NOREV_RAW_INT16 && !(V26_ZMM_PAIR_WINDOW && V44_MASKVEC && \
                         V44_ROWOFF == 1 && V35_PAIR_MASKS && V46_COL5 && \
                         V34_TRUST_OFFICIAL_INPUT && V34_TRUST_GOLD_RANGE)
#error "NOREV_RAW_INT16 requires the audited V46 pair-window shape"
#endif

// ---------------------------------------------------------------- 富点记忆
//
// 这是本仓第一个**故意改变决策**的改动（此前所有版本都与 V6 逐位一致）。
// 依据是 60 局真实日志里挖出来的三组数字：
//
//   1. 外围金币不是均匀撒的。日志第 1 行那张地形图里标 "2" 的格子只占外围
//      格子的 9.6%，却承接了 47.6% 的外围金币出现次数，单堆中位 7 金币
//      （map 4 上是 32），而普通外围格只有 3.5。这些格子就是「锚点」。
//   2. 锚点在 grid 里空着时显示 0，和普通空地一模一样 —— 策略没法从地形
//      识别它，只能在对局中根据「哪里出过大堆」反推。
//   3. 「两个角色 5x5 窗口内都没有金币」的回合占 14.2%（map 4 是 20.4%）。
//      这种回合里角色离中心中位 7 步、离最近锚点中位只有 4 步，
//      60.6% 的情况一个回合（4 步）就能走到锚点。而当前策略把 target 固定在
//      中心 (8,8)，等于让角色花两个回合往回走，路过的还是刚被自己吃空的地方。
//
// 所以做三件事，全部在函数末尾的冷区，热路径一条指令都不加：
//   观测：本回合持币比上回合多出一大截 => 上回合的终点踩到了大堆。拾取后
//         角色就站在那一格上，所以「本回合的 my_units[k]」正是堆的位置。
//         my_units_gold 紧跟 my_units，同一条 cache line，读它不额外付 miss。
//   使用：只在「本回合这个角色拿不到金币」时把 target 从中心改成富点。
//         有金币的回合 key 的金币字段主导排序，方向偏好完全不参与，
//         所以不需要把 target 改回去 —— 少一半 store。
//   遗忘：已经走到富点跟前（曼哈顿 <= 2，即富点已进视野）却还是空的，
//         说明那个堆没了，清掉记忆回到中心，避免角色守着一个空格发呆。
//
// 规则依据（为什么这些回合可以不在乎先手权）：赛制写的是「每回合按决策耗时
// 从小到大依次执行移动。同一网格的金币/炸弹由先执行者获取」——
// 先手权每回合独立判定，且只在双方争夺同一格金币时才有价值。
// 本回合窗口里没有金币可拿，这一回合的先手权对我方就没有直接收益。
#ifndef V7_RICH
#define V7_RICH 0
#endif

// 一回合净增持币达到多少才算「踩到了大堆」。
// 定标：拾取率 65%，所以锚点中位堆 7 金币到手 4.6、map 4 的 32 金币到手 21；
// 而普通回合 4 步内可达金币中位 11.95、到手约 7.8。阈值要压在普通回合之上，
// 否则每回合都在改写富点，记住的就不是锚点而是最近走过的任意一格。
#ifndef V7_RICH_GAIN
#define V7_RICH_GAIN 12
#endif

// 两者都会改写 state.target_*，同开语义不清（谁最后写谁赢）。
// 冠军配置本来就是 V7_SNAPSHOT=0，所以这里直接禁掉这种组合。
#if V7_RICH && V7_SNAPSHOT
#error "V7_RICH 与 V7_SNAPSHOT 都改写 state.target_*，不能同时开启（用 -DV7_SNAPSHOT=0）"
#endif

// 两个角色的初始方向目标。V6 到 v11 一直是 (8,8) / (8,8) —— 两个角色朝同一点。
//
// 怀疑它是个浪费，依据来自 map 4 的日志：那张图上我方 500 回合两个角色合计
// 只走过 92.9 个不同格（五图平均 117.6），而它的外围可通行格只有 121 个、
// 中心可通行只有 49 个。两个角色的 5x5 视野大量重叠时，等于只有一个在探索。
//
// 让它们各守中心区的一个对角是**零指令成本**的分离 —— 只改这两个初值常量，
// 一条指令都不加，`.text` 一字节不变。方向偏好仍然落在 key 的 bit 4..6、
// 被金币字段压着，所以只在候选路径金币相同时才起作用。
// 档位 3/4 是地图六的数据逼出来的。那张图中心 9x9 墙密度 69.1%、可通行仅 25 格，
// 站在中心十字通道上 5x5 视野只有 9~11 格不是墙（四角开阔区有 24 格）。
// 按「角色真的站在该区时窗口内的金币总量」统计 50 局：
//     右侧 c13-16 r6-10  6.96      左带 c0-2 r6-10  6.52
//     四角开阔区         4.64      中心十字         3.90
//     上下中段           2.93      中心其余         2.84
// 而我方 48.4% 的时间待在最后两档、只有 16.2% 在前两档。
// 「中心十字驻留」在输的局里还高 2.07pp —— 与产出表同向。
// 所以让两个角色各守一侧的中部横带，正好也对上它们的出生角（(0,0) / (16,16)）。
//   0 = 关（both (8,8)，即 v11 语义）
//   1 = 保守分离 (6,6) / (10,10)，各偏中心 2 格，仍在最热区内
//   2 = 激进分离 (4,4) / (12,12)，正好是中心 9x9 的两个对角
//   3 = 左右分守 (8,1) / (8,15)，直接指向地图六产出最高的两条带
//   4 = 左右分守但略靠内 (8,3) / (8,13)，折中
//   5 = 更靠内 (8,5) / (8,11)，偏移只有 3
//   6 = 一守一游 (8,3) / (8,8)：角色 1 留在中心保底，角色 0 去外围拿上限。
//       动机是把「最坏情况」压住 —— 偏移量这一维已经调到峰值（档位 4），
//       但它在 map 3/4 仍然输给 v11 8~12%。让一个角色守中心，
//       等于在任何地图上都保留 v11 的那一半行为。
//
// 已测出的「偏移量」响应曲线（都是零延迟成本，只改这一对常量）：
//   偏移 7 (8,1)/(8,15)  地图六 6-2 胜，但 map 1~5 是 3-7 输（map 4 比值 1.973）
//   偏移 5 (8,3)/(8,13)  地图六 6-2 胜，map 1~5 是 7-3 胜  <= 目前最优
//   对角偏移 4 (4,4)/(12,12)  map 4 专项 4-8 输
// 只差两格就从 9-9 变成 13-5，所以这个量是敏感的，档位 5 用来补曲线的内侧。
#ifndef V7_SPLIT_TARGET
#define V7_SPLIT_TARGET 0
#endif

#if V7_SPLIT_TARGET && (V7_RICH || V7_SNAPSHOT)
#error "V7_SPLIT_TARGET 只改 target 初值，与会动态改写 target 的开关同开没有意义"
#endif

// ---------------------------------------------------------------- 象限开阔度
//
// 为什么需要它：V7_SPLIT_TARGET=4 在地图六把胜率从 54% 抬到 68%，但那两个
// 坐标 (8,3)/(8,13) 是**在地图六的分区产出表上调出来的常量** —— 换一张新地图
// 就没有依据了。要做成地图无关的，得回到那次分析真正得出的规律：
//
//     每回合产出 ≈ 金币密度 x 有效视野
//
// 「金币密度」已经由 key 的金币字段（最高位）主导排序。
// 「有效视野」此前被完全忽略 —— 而它是**局部可观测**的：5x5 窗口里有多少格
// 不是墙/雾。地图六的中心十字通道每格只有 9~11 格可见，四角开阔区有 24 格，
// 我们却把 42.4% 的时间花在前者，这才是那 54% 的根因。
//
// 实现上有个恰好的巧合：UnitWindow 的 c11/c12/c21/c22 就是**按象限归一化后**的
// 对角 2x2 块，每个 __m128i 的 4 个 lane 正好对应 4 个象限，和 qbase 的布局
// 完全对齐。所以「这个方向的腹地开不开」只要 4 次 cmpgt + 4 次加减就能算出，
// 而且结果可以直接 or 进 qbase，不需要动窗口构造，也不需要重排代码。
//
// 字段位置：用 key 里腾出来的 bit 11..14（见下面 key 位布局一节）。
// 它排在**金币之下、距离之上**，语义因此是：
//     先抢金币；金币一样时朝腹地更开阔的方向走；再一样才比走多远。
// 压过距离是刻意的 —— 原来的「尽量走满 4 步」会把角色送进死通道深处。
//
// 腾出这 4 bit 的代价是把单格金币的精确计价上限从 3000 降到 1000。
// 实测 12000 个真实回合、29823 个金币格里单格最大值只有 205，
// 所以这个代价是零。三条不变量由下面的 static_assert 钉死。
// V23：把 not_reversed 再放一份到 dist **之上**的 bit 11。
//
// 这一节的病因和 V7_OPENNESS 那节写的是同一个 —— 「尽量走满 4 步压过一切非
// 金币因素」会把角色送进死通道深处 —— 但信号换成已有的 not_reversed，而不是
// 开阔度（开阔度线上 4-6，原因是信号本身错了：最开阔的地图四角产出反而最低）。
//
// 为什么是 not_reversed：2026-08-19 在 30 局正式日志上量过（tools/loop_cost_audit.py、
// tools/notreversed_order_gate.py、tools/centre_pull_gate.py）：
//   * 「≥6 回合困在 ≤3 格里」的段落每局 18.9 个，其中 3.6 个是**零产出**的空转
//     （拾金 0.109/回合，对基线 1.744），占 4.04% 的角色回合，折损 66 金币/局；
//     另外 15.3 个是**蹲守富格**（拾金 3.370/回合），必须完好保留。
//   * 精确 180 度掉头占已移动相邻回合对的 20.15%，分解为 14.13% 被迫（金币在身后，
//     正确行为）+ 6.01% 可消除（最远候选全是反向，而存在更近的非反向候选）。
//     6.01% + 14.13% = 20.14%，与独立测得的 20.15% 吻合到 0.01pp。
//   * game 643167 的实例：unit0 在 (3,8)/(1,6)/(1,10) 三格间空转 28 回合。round 82
//     的候选集里出口 (0,12) 是 d3 且未被罚，被选中的 (3,8) 是 d4 **且被罚**，
//     `dist` 直接压过惩罚。把 not_reversed 抬到 dist 之上，该回合改选 (0,8)，环断开。
//
// 安全性：空转段拾金 0.109/回合 ⇒ 金币字段几乎恒为 0 ⇒ 比较落到本字段，惩罚生效；
// 蹲守段拾金 3.370/回合 ⇒ 金币字段非零且能区分 ⇒ 比较在金币就结束，本字段不参与。
// **金币字段自动保护了蹲守行为**，这也是为什么本字段必须在金币之下。
//
// 实现上不动 TAIL_LUT（uint16、12 bit、已打包进 256B 热表，不能变大），
// not_reversed 的第二份副本由 prev_quadrant 直接向量化算出，见 lift_norev()。
// tail 组里 bit 6 的旧副本保留不动：它只在 bit 11 打平时才参与，而 bit 11 打平
// 就意味着 not_reversed 相等，所以旧副本恒无作用，留着是零成本。
// 两种放置方式，语义完全相同（同一个排序），只有布局不同 —— 所以对比这两者可以把
// 布局效应从语义效应里单独隔离出来：
//
//   1 = 放 bit 11。需要把 GOLD_SHIFT 抬到 15，于是 MAGIC_SHIFT、AND 掩码、NEG、
//       VALID_FLOOR、GOLD_EXACT_MAX 一整批**嵌在指令里的立即数**跟着变，编码长度
//       随之变化。实测 .text +136 / .rodata +64，线上配对 p50 +13.8ns（t=2.1）。
//   2 = 放 bit 10。dist 占 bits 7..10 共 4 位，但取值只有 1..4，**bit 10 在 key 里
//       恒为 0**，它存在只是给 choose_k 的两个 dist 相加（最多 4+4=8，正好占到
//       bit 10 且不再往上）留进位空间。借用它 => GOLD_SHIFT 留在 11，上面那一批
//       立即数一个都不动。代价只有 key_dist 的掩码 15->7 与一条额外的 or。
//       此外 not_reversed 的四个位本来就在已加载的 tail 字里（bits 2/5/8/11），
//       所以每条 lane 用编译期立即数直接取，**不引入任何新的 .rodata 常量**。
//
//   3 = bit 10 的布局（同 =2），但**加一道干涸门**：只有这个角色连续
//       V7_NOREV_DRY 个回合的规划收益为 0 时才置位。=1/=2 之所以线上失败，
//       原因已经量出来了，不是延迟（配对 p50 −13.8ns，反而更快），而是它
//       无条件生效，把**蹲守**也一起打断了：空转段 2.3→0.5/局（−85%，目标
//       达成），但蹲守段 19.8→12.8（−35%）。而蹲守段的拾金是 3.37/回合、
//       基线只有 1.74，所以每打断一段亏约 12 金币，7 段就是 −84/局，
//       与实测的 −132.6 ± 115.4 相容。
//
//       用户当时指出「金币优先，有金币应该还是能回去」——实测证明这话是对的：
//       796 个分歧回合里 V23 与 V22 收的金币**完全相同**（全程 25944 : 25944）。
//       所以不是「回不去」，而是**同金币时落点被改写**：富区里多格各剩 35% 余额、
//       数值相近极易并列，每次并列都往外推一格，连续 6 回合就漂出富区。
//       段是被提前打断的，不是效率变低（蹲守拾金率 3.313→3.416，没降）。
//
//       门的选取是量出来的，不是猜的（tools/drought_nogold_conjunction.py，
//       tools/drought_gate_specificity.py）：
//         * 「视野无金币」单独用：触及 65~71% 的蹲守段。**先前据此判死是错的** ——
//           那是用频率给价值定价。实测每个被触及的段尾部只值 7.4 金币，因为
//           蹲守段本来就是「窗口空了所以离开」才结束的，门开的回合平均落在段的
//           最后 1~2 回合，打断的是本来就要结束的段。台账净 +35.5/局。
//         * 「距上次拿到金币的回合数」单独用：**按价值反选**。门开的蹲守回合拾金
//           是基线的 1.1~2.6 倍，且倍数随阈值上升 —— 蹲守就是「取走 65%、等回补、
//           再取一大笔」，长干旱恰好标记兑现前夜。单独用必然重演 V23。
//         * 两者合取：`drought>=4` 的回合里蹲守仍看得见金币的占 58.4%、空转只有
//           28.6%，所以无金币项正好否掉干旱项错选的那些回合。蹲守段触及降到
//           13.4%（d>=4）/ 8.6%（d>=6），门开时的即时金币恒为 0。
//
//       这里取的是**滞后一回合的无金币项**，即「连续 V7_NOREV_DRY 个回合规划收益
//       为 0」。理由是依赖边：直接读窗口要在「窗口构造 → key 构造」之间加一条数据
//       依赖，而 choice.key >> GOLD_SHIFT 是上一回合末尾就算好的持久状态，
//       **入口即就绪，零新增依赖边**。台账代价是 32.7 对 35.5 净金币/局。
//       它还额外覆盖「看得见但被封死」——game 643167 里 U1 被三面墙加一个炸弹
//       围住 28 回合，窗口里有金币但一步都走不到，规划收益同样是 0。
//
//       环的规模按「连续 ≥6 回合没走出一个 5x5 包围盒且拾金低于基线 25%」计，
//       上限是 162 金币/局（紧定义「≤3 个不同格」只有 68，那是下界；放松到
//       「≤6 个不同格」会把正常长途移动也算进来，因为一回合能走 4 步）。
#ifndef V7_NOREV_HIGH
#define V7_NOREV_HIGH 0
#endif

// 干涸门的阈值：连续多少个规划收益为 0 的回合之后才允许惩罚反向。
// 2 是台账上的取值（净 +32.7/局）；调高更保守（d>=6 时 +24.9）。
#ifndef V7_NOREV_DRY
#define V7_NOREV_DRY 2
#endif

#if (V7_NOREV_HIGH == 2 || V7_NOREV_HIGH == 3) && V7_ARITH_QBASE
#error "V7_NOREV_HIGH=2/3 走 TAIL_LUT 的标量提取路径，与 ARITH_QBASE 不兼容"
#endif

#if V7_NOREV_HIGH == 3 && (V7_NOREV_DRY < 1 || V7_NOREV_DRY > 254)
#error "V7_NOREV_DRY 必须落在 1..254：计数器是 uint8 且在阈值处饱和"
#endif

#ifndef V7_OPENNESS
#define V7_OPENNESS 0
#endif

#if V7_NOREV_HIGH && (V7_OPENNESS || V7_TRAMPLE == 4)
#error "V7_NOREV_HIGH 与 V7_OPENNESS / V7_TRAMPLE=4 都要占 bit 11..14"
#endif

#if V7_NOREV_HIGH && V7_QBASE_LUT
#error "V7_NOREV_HIGH 需要在 qbase 上追加 bit 11；QBASE_LUT 是预算好的常量表，二者不兼容"
#endif

#if V7_OPENNESS && V7_TRAMPLE == 4
#error "V7_OPENNESS 与 V7_TRAMPLE=4 都要占 bit 11..14"
#endif

// ---------------------------------------------------------------- 区域产出记忆
//
// 这一版要解决的是 V7_SPLIT_TARGET=4 的过拟合问题：那对坐标 (8,3)/(8,13)
// 是在地图六的分区产出表上调出来的，换张新地图就没有依据。
//
// 先试过 V7_OPENNESS（象限开阔度进 key），线上被否 —— 原因写在那一节：
// 地图六上最开阔的四角（有效视野 17.5）产出只有 4.64，而左右两条带
// （视野 16.8/17.1）产出 6.52/6.96。**最开阔不等于产出最高**，
// 因为外围的金币刷新本来就稀疏。追开阔度会把角色引到地图边角去。
//
// 真正要最大化的是产出本身（金币密度 x 有效视野）。它是个长期统计量，
// 5x5 窗口给不了 —— 但可以在对局中自己累积出来：
//   * 把 17x17 切成 3x3 = 9 个粗区；
//   * 每回合把「这个角色本回合实际拿到的金币」记到它所在的区，用指数移动
//     平均（EMA）滑动更新。EMA 自带归一化，不需要另外记访问次数，
//     也自动淡忘早期被吃空的区；
//   * 每 32 回合挑一次 EMA 最高和次高的区，把两个角色的方向目标分别设过去。
//     次高给第二个角色是为了让它们分头，别挤在一处。
//
// 关键的一点：3x3 粗区的中心正好是 (3/8/13, 3/8/13) 这 9 个点，
// 而 V7_SPLIT_TARGET=4 那对手调的坐标 (8,3) 和 (8,13) 恰好就是其中两个。
// 也就是说这个机制**有能力自己找出地图六的那个解**，同时在别的地图上会
// 收敛到别的答案 —— 这就是「地图无关」的含义。
//
// 成本：更新是每角色 4 条指令（都在函数末尾冷区）；挑区是 9 次比较，
// 但每 32 回合才做一次，摊销下来不到 0.3 次迭代/回合。
#ifndef V7_ZONE
#define V7_ZONE 0
#endif

// EMA 的衰减档：新值权重 1/2^V7_ZONE_DECAY。4 => 半衰期约 11 回合，
// 32 回合的挑区周期内能反映最近 2~3 个周期的产出。
#ifndef V7_ZONE_DECAY
#define V7_ZONE_DECAY 4
#endif

// 乐观初始化：所有区的 EMA 从一个高于真实稳态的值起步。
//
// 这是第一版 V7_ZONE 失败的直接原因。初值全 0 时，开局挑区所有区并列，
// tie-break 固定选中区 0 和区 1；这两个区一旦有了样本就永远是最高的，
// 于是整局锁死在开局那两个区 —— 线上实测 v19 在地图六的中心驻留 38.6%，
// 比手调的 v17（26.3%）还高，就是这个症状。
//
// 乐观初始化是多臂赌博机里的标准解法：没去过的区看起来最好，所以会被
// 依次尝试；去过之后 EMA 会朝真实产出收敛（`ema -= ema>>DECAY` 让它下降），
// 于是自然完成「探索 -> 利用」的过渡，不需要随机数也不需要访问计数。
//
// 取值必须和真实稳态同量级，否则整局都衰减不完 —— 这是前两版失败的直接原因。
//
// 标定：喂进 EMA 的是「单回合实际收益」。60 局真实日志里我方每回合拾取
// 3.887 金币（两个角色合计），所以单角色 g ≈ 2。稳态 = g << DECAY，
// DECAY=3 时是 16。第一版把初值设成 640（高出 40 倍），从 640 衰减到 16
// 需要约 45 个周期 = 1400+ 回合，而一局只有 500 —— 于是整局停在「探索」
// 状态，挑区实际由初值和 tie-break 决定，和真实产出完全无关。
// 线上表现也印证了：v19 只打平 v17，v19a（更灵敏）反而 3-7。
//
// 取 32 = 稳态的 2 倍：未访问的区看起来比任何已访问的区都好（保证轮着试），
// 而访问后 6 个周期左右（约 96 回合）就能落回真实水平，开始真正的「利用」。
#ifndef V7_ZONE_OPTIMISTIC
#define V7_ZONE_OPTIMISTIC 32
#endif

// 先验区的起步值。区索引 3 和 5 对应 (8,3) 和 (8,13) —— 也就是
// V7_SPLIT_TARGET=4 手调出来的那对坐标，它在六张结构差异极大的图上
// 13-5 胜 v11（中心墙密度从 14.8% 到 69.1% 都覆盖了）。
//
// 把它作为**先验**而不是硬编码，是这一版的核心：
//   * 开局挑区必然选中区 3/5，所以起手行为和 v17 完全一样；
//   * 如果这两个区在当前地图上产出确实高，EMA 保持领先，整局维持 v17 的解；
//   * 如果产出低（v17 在 map 3/4 就是这样），EMA 衰减到真实水平，
//     其他区的乐观初值会反超，策略自己切换过去。
// 这样最坏情况有 v17 兜底，同时保留在新地图上自我修正的能力。
#ifndef V7_ZONE_PRIOR
#define V7_ZONE_PRIOR 48
#endif
static_assert(V7_ZONE_PRIOR >= V7_ZONE_OPTIMISTIC,
              "先验区必须起步不低于其他区，否则开局不会选它");

// 每多少回合重挑一次目标区。太短会跟着噪声乱跳，太长跟不上金币刷新。
// 必须是 2 的幂（用 & 判断）。
#ifndef V7_ZONE_PERIOD
#define V7_ZONE_PERIOD 32
#endif
static_assert((V7_ZONE_PERIOD & (V7_ZONE_PERIOD - 1)) == 0,
              "V7_ZONE_PERIOD 必须是 2 的幂");

#if V7_ZONE && (V7_SPLIT_TARGET || V7_RICH || V7_SNAPSHOT)
#error "V7_ZONE 会动态改写 state.target_*，不能和其他改写 target 的开关同开"
#endif

// 把 apply_snapshot 挪到函数末尾（输出已经算完之后），并在入口对 Snapshot
// 那 3 条 cache line 做定向预取。
// 依据：第二轮实测快照让 p50 只 +10ns 但 p90 +60ns —— 成本集中在 20% 的快照
// 回合，是读 148 字节冷结构体的 miss，不是计算量。而快照本身描述的是「上几轮
// 的情况」，它只影响下一回合的目标点，所以完全不必待在关键路径上。
// 语义代价：快照晚一回合生效（对一个回顾性统计量来说可以忽略）。
#ifndef V7_SNAP_DEFERRED
#define V7_SNAP_DEFERRED 1
#endif

namespace {

// ---------------------------------------------------------------- 常量

constexpr int A_UP = 0;
constexpr int A_DOWN = 1;
constexpr int A_LEFT = 2;
constexpr int A_RIGHT = 3;
constexpr int A_STAY = 4;

constexpr int FOG = -5;
constexpr int BOMB = -3;
constexpr int WALL = -1;

// ---------------- key 位布局 ----------------
//
// V6/V7 默认（GOLD_SHIFT=11）：
//   bit 31..11 金币 | 10..7 距离 | 6..4 tailRank | 3..2 象限 | 1..0 endpoint
//
// **关于「高位金币字段是否过剩」的重要结论**（本轮实测踩过的坑）：
// 一条路径最多 4 格、每格 ceil(0.65g)，取值只需约 10 bit，却占了 20 bit，
// 看起来能白拿 10 位元数据。但这 10 位其实在做实事 —— 它们是 NEG 硬排除
// 机制的**饱和余量**。把 GOLD_SHIFT 从 11 抬到 18 时出现了穿墙 bug：
//   一个 NEG(-2^26) + 两个 250 金币格(各 163<<18 = 4.27e7) = +1.8e7 > 0
// 于是一条经过墙的路径被判成合法。V6 文档只说「NEG 足够小，四次加法后仍为
// 负」，没写这个不变量对 GOLD_SHIFT 的依赖。
//
// 正确做法是同时抬高 NEG，并重新解不等式组：
//   (1) |NEG| >> 4*R<<S            一个 NEG 必须压过其余格的金币
//   (2) 约 6*|NEG| + 4*R<<S < 2^31 最坏链上约 6 个 NEG 不能溢出 int32
//   (3) 8*R < 2^(31-S)             choose_k 要把两个 key 的金币相加
// 其中 R = ceil(0.65 * GOLD_EXACT_MAX)。
// S=15 / NEG=-(1<<28) / G_max=1000 (R=650) 的一组可行解：
//   4R<<15 = 8.5e7 << 2.68e8 ✓   6*2.68e8 = 1.6e9 < 2.1e9 ✓   8*650=5200 < 65536 ✓
// 于是 bit 11..14 这 4 位可以真正拿来用。
constexpr int DIST_SHIFT = 7;

#if V7_TRAMPLE == 4
// TRAMPLE=4 的布局：
//   bit 31..15 金币 | 14..11 clean | 10..7 距离 | 6..4 tailRank | 3..2 象限 | 1..0 endpoint
// clean = CLEAN_MAX - 路径上的踩踏格数。每个踩踏格贡献 -CLEAN_ONE，靠 DP
// 已有的 add 自然累加；构造 key 时统一 +CLEAN_BIAS 抬成非负，避免向金币借位。
// 排在金币之下、距离之上 => 语义是「金币相同则少踩踏，再相同才比距离」。
constexpr int GOLD_SHIFT = 15;
constexpr int CLEAN_SHIFT = 11;
constexpr int CLEAN_MAX = 4;
constexpr int32_t CLEAN_ONE = 1 << CLEAN_SHIFT;
constexpr int32_t CLEAN_BIAS = CLEAN_MAX << CLEAN_SHIFT;
constexpr int32_t NEG = -(1 << 28);
constexpr int32_t VALID_FLOOR = -(1 << 20);
constexpr int32_t GOLD_EXACT_MAX_CFG = 1000;
#elif V7_OPENNESS
// OPENNESS 的布局（和 TRAMPLE=4 用的是同一组参数，那次已经把可行性验证过）：
//   bit 31..15 金币 | 14..11 象限开阔度 | 10..7 距离 | 6..4 tailRank
//   | 3..2 象限 | 1..0 endpoint
//
// 开阔度放在金币字段**之下**是正确的，因为它是纯序数偏好 —— 不需要和金币
// 互相换算。这一点和踩踏正好相反：踩踏损失必须能和金币比大小（少踩一格
// vs 少拿 14 金币），塞在金币之下就只能破平局，那正是 V7_TRAMPLE=4 失败的
// 原因（见 6.7 节）。方向偏好没有这个问题。
//
// 放在距离**之上**是刻意的：原排序里「尽量走满 4 步」压过一切非金币因素，
// 会把角色一路送进死通道深处。现在先看那个方向的腹地开不开。
constexpr int GOLD_SHIFT = 15;
constexpr int OPEN_SHIFT = 11;
constexpr int OPEN_MAX = 4;              // 象限的 2x2 对角块最多 4 格
constexpr int32_t NEG = -(1 << 28);
constexpr int32_t VALID_FLOOR = -(1 << 20);
constexpr int32_t GOLD_EXACT_MAX_CFG = 1000;
static_assert(OPEN_SHIFT + 3 <= GOLD_SHIFT, "开阔度字段必须完整落在金币字段之下");
static_assert(OPEN_SHIFT >= DIST_SHIFT + 4, "开阔度必须排在距离字段之上");
#elif V7_NOREV_HIGH == 1
// NOREV_HIGH=1 的布局（参数与 OPENNESS/TRAMPLE=4 同一组，那两次已把可行性验证过）：
//   bit 31..15 金币 | 11 不走完全反向 | 10..7 距离 | 6..4 tailRank
//   | 3..2 象限 | 1..0 endpoint
// bit 12..14 空着，留给以后的候选量。
constexpr int GOLD_SHIFT = 15;
constexpr int NOREV_SHIFT = 11;
constexpr int32_t NEG = -(1 << 28);
constexpr int32_t VALID_FLOOR = -(1 << 20);
constexpr int32_t GOLD_EXACT_MAX_CFG = 1000;
static_assert(NOREV_SHIFT + 1 <= GOLD_SHIFT, "反向位必须完整落在金币字段之下");
static_assert(NOREV_SHIFT >= DIST_SHIFT + 4, "反向位必须排在距离字段之上");
#else
constexpr int GOLD_SHIFT = 11;
constexpr int32_t NEG = -(1 << 26);
// 踩踏罚分让「合法但很差」的路径也可能为负，所以不能再用 score >= 0 当合法。
// 单格罚分上限 PEN_CLAMP=255，四步最多 4*255<<11 = 2088960 < 2^22。
constexpr int32_t VALID_FLOOR = -(1 << 22);
constexpr int32_t GOLD_EXACT_MAX_CFG = 3000;
#if V7_NOREV_HIGH >= 2
// 借 dist 字段那个恒为 0 的最高位。GOLD_SHIFT 保持 11，所以整套金币/哨兵/魔数常量
// 一个都不动，这正是这一版相对 =1 的全部意义。=3 复用同一套布局，只多一道门。
constexpr int NOREV_SHIFT = 10;
static_assert(NOREV_SHIFT == DIST_SHIFT + 3,
              "反向位必须正好落在距离字段那个恒为 0 的最高位上");
static_assert(NOREV_SHIFT < GOLD_SHIFT, "反向位必须落在金币字段之下");
// 距离取值 1..4 只占 bits 7..9，所以 bit 10 在 key 里恒为 0，可以借用；
// choose_k 求和时两个距离最多 4+4=8，仍在 bits 7..10 内，不会进位到金币。
static_assert(4 < (1 << (NOREV_SHIFT - DIST_SHIFT)),
              "距离的最大取值必须容得下，且不占用被借走的那一位");
static_assert(8 < (1 << (NOREV_SHIFT - DIST_SHIFT + 1)),
              "两个距离相加的进位必须仍落在金币字段之下");
#endif
#if V49_NOREV_GATE || V50_NOREV_GLOBAL
// V49（=V23b 的落位，加一道干旱门）：借 dist 字段那个恒为 0 的最高位。
//   bit 31..11 金币 | 10 不走完全反向 | 9..7 距离 | 6..4 tailRank
//   | 3..2 象限 | 1..0 endpoint
// GOLD_SHIFT 留在 11，所以 NEG / VALID_FLOOR / GOLD_EXACT_MAX / MAGIC_SHIFT /
// 三个哨兵一个都不动 —— 这正是 bit 10 相对 bit 11 的全部意义。
constexpr int NOREV_SHIFT = 10;
static_assert(NOREV_SHIFT == DIST_SHIFT + 3,
              "反向位必须正好落在距离字段那个恒为 0 的最高位上");
static_assert(NOREV_SHIFT < GOLD_SHIFT, "反向位必须落在金币字段之下");
// 距离取值 1..4 只占 bits 7..9，所以 bit 10 在 key 里恒为 0，可以借用；
// choose_k 求和时两个距离最多 4+4=8，仍在 bits 7..10 内，不会进位到金币。
static_assert(4 < (1 << (NOREV_SHIFT - DIST_SHIFT)),
              "距离的最大取值必须容得下，且不占用被借走的那一位");
static_assert(8 < (1 << (NOREV_SHIFT - DIST_SHIFT + 1)),
              "两个距离相加的进位必须仍落在金币字段之下");
// 一条 qbase lane 加上 bit-10 副本之后的上界，uint16 表的存在前提。
static_assert(((7u << 4) | (3u << 2) | (1u << NOREV_SHIFT)) <= 0xFFFFu,
              "带 bit-10 副本的 qbase lane 必须完整落在 uint16 里");
#if V50_NOREV_GLOBAL
constexpr int V50_ALL_LOWER_FIELDS_MAX =
    (4 << DIST_SHIFT) | (7 << 4) | (3 << 2) | 3;
static_assert((1 << NOREV_SHIFT) > V50_ALL_LOWER_FIELDS_MAX,
              "同金币时 not_reversed 必须压过最大 distance/tail/Q/endpoint");
static_assert((1 << GOLD_SHIFT) >
              ((1 << NOREV_SHIFT) | V50_ALL_LOWER_FIELDS_MAX),
              "一个 gold score 单位必须压过全部平局字段");
#endif
#endif
#endif

// 编译期把上面那三条不等式钉死，以后改 GOLD_SHIFT / GOLD_EXACT_MAX 会直接
// 编译失败，而不是产出一个偶发穿墙的二进制。
constexpr int64_t MAX_REWARD = (int64_t(GOLD_EXACT_MAX_CFG) * 13 + 19) / 20;
constexpr int64_t MAX_PATH_GOLD = 4 * MAX_REWARD;

// (1) 硬排除不变量：只要路径上有一个 NEG，即使其余格全是最大金币，
//     分数也必须仍低于合法下界。这就是「穿墙 bug」的确切判据。
static_assert(int64_t(NEG) + (MAX_PATH_GOLD << GOLD_SHIFT) < int64_t(VALID_FLOOR),
              "NEG 不足以压过累计金币：硬排除会失效（会出现穿墙路径）");
// (2) 最坏依赖链上约 6 个 NEG 累加不能溢出 int32
static_assert(6 * -int64_t(NEG) + (MAX_PATH_GOLD << GOLD_SHIFT) < (int64_t(1) << 31),
              "最坏 NEG 链会溢出 int32");
// (3) choose_k 会把两个 key 的金币字段相加，不能溢出该字段
static_assert((2 * MAX_PATH_GOLD) < (int64_t(1) << (31 - GOLD_SHIFT)),
              "choose_k 把两个 key 的金币相加会溢出金币字段");

// 无效候选哨兵，必须低于任何合法 key、且高于 NEG 累加的量级。
constexpr int32_t KEY_INVALID = NEG / 2;
constexpr int32_t PEN_CLAMP = 255;
static_assert(KEY_INVALID < VALID_FLOOR, "无效哨兵必须低于合法下界");

// 免除法的 ceil(0.65*g)<<GOLD_SHIFT。
// 基式：floor((13g+19)/20) = ((13g+19) * 26215) >> 19，由 tools/magic_check.cpp
// 穷举验证 g <= 3359 精确、g <= 6299 不溢出。把两次乘法折成一次：
//   ceil(0.65g) = (g*340795 + 498085) >> 19
// 再把 <<GOLD_SHIFT 折进移位量（GOLD_SHIFT <= 19 时是右移，否则要左移）。
constexpr int32_t MAGIC_MUL = 340795;
constexpr int32_t MAGIC_ADD = 498085;
constexpr int32_t MAGIC_BASE_SHIFT = 19;
constexpr int32_t MAGIC_SHIFT = MAGIC_BASE_SHIFT - GOLD_SHIFT;   // >= 0
constexpr int32_t GOLD_EXACT_MAX = GOLD_EXACT_MAX_CFG;
static_assert(MAGIC_SHIFT >= 0, "GOLD_SHIFT 不能超过 19，否则折叠移位为负");

constexpr int CELL_LUT_MAX = 63;
// bits 7..9 hold the only reachable single-unit distances (1..4); bit 10 is spare.
// Give the behavior-neutral QB16 control the same extraction/masking instructions as V50.
constexpr int QB16_SPARE_SHIFT = DIST_SHIFT + 3;
static_assert(QB16_SPARE_SHIFT == 10, "QB16 spare bit must remain key bit 10");

#if V7_NOREV_HIGH >= 2 || V49_NOREV_GATE || V50_NOREV_GLOBAL || V49_QBASE16
// bit 10 被反向位借走了，所以距离只读 3 位。取值 1..4 完整保留，同一条指令
// （同一个 shr + and，只是 and 的立即数 15 -> 7）。不改这里的话，门开的那些回合
// emit_parent 会把 moved 读成 8..12，动作串直接错。
// QBASE16 对照也固定使用 &7：其 bit 10 恒为 0，故与 &15 严格等价；这样 V50 开关
// 只改预计算表值，不再改变任何 .text 字节。
inline int key_dist(int32_t key) { return (key >> DIST_SHIFT) & 7; }
static_assert(((4 | (1 << (QB16_SPARE_SHIFT - DIST_SHIFT))) & 7) == 4,
              "3 位掩码必须完整保留 dist 的 1..4 并且挖掉反向位");
#else
inline int key_dist(int32_t key) { return (key >> DIST_SHIFT) & 15; }
#endif
inline int key_quadrant(int32_t key) { return (key >> 2) & 3; }

#if V7_NOREV_HIGH == 1
// not_reversed 的第二份副本，直接从 prev_quadrant 算出，不经过 TAIL_LUT
// （热表已压在 256 字节里，不能变大）。被罚的象限恰好是 previous ^ 3；
// previous < 0 时用 q 取不到的 -1 让比较恒不相等，于是四条 lane 全部置位。
// 四条 lane 一次算完：set1 + cmpeq + andnot，加上外面的 or 共 4 条向量指令/角色。
inline __m128i lift_norev(__m128i qbase, int previous) {
    const __m128i qv = _mm_setr_epi32(0, 1, 2, 3);
    const __m128i opp = _mm_set1_epi32(previous < 0 ? -1 : (previous ^ 3));
    return _mm_or_si128(
        qbase,
        _mm_andnot_si128(_mm_cmpeq_epi32(qv, opp),
                         _mm_set1_epi32(1 << NOREV_SHIFT)));
}
#define V7_LIFT_NOREV(v, unit) lift_norev((v), state.prev_quadrant[unit])
#else
#define V7_LIFT_NOREV(v, unit) (v)
#endif

#if V7_NOREV_HIGH >= 2
// not_reversed 的四个位在 tail 字里是 bits 2/5/8/11（每象限 3 位的最高位）。
// 每条 lane 的搬移量都是编译期常量，所以是 shr-立即数 + and-立即数 + shl-立即数，
// **不引入任何新的向量常量或 .rodata**。
//
// 先前写成一次融合移位 `((t) & (4u << 3q)) << (NOREV_SHIFT - 2 - 3q)`，对 bit 11 成立
// （q=3 时移位量为 0），但换到 bit 10 之后 q=3 的移位量是 -1 —— **负移位是 UB**。
// clang 把它编成了别的东西，结果反向率被压到 10.69%，低于 14.13% 的被迫下限。
// 这个错误是 validation/layout_isolation.py 的「两版语义必须逐位相同」这条对照抓出来的。
#if V7_NOREV_HIGH == 3
// 干涸门：把 not_reversed 位与门相与，直接写在四条 lane 表达式里。
// 门是 0/1，所以每条 lane 只多一条 and-寄存器；门本身只依赖上一回合末尾写下的
// 持久状态 —— 入口即就绪，**不在「窗口构造 -> key 构造」之间新增任何依赖边**。
// 计数器在阈值处饱和（见回合末尾的更新），所以判据是一次相等比较而不是 >=，
// 也因此 500 回合的长干涸不可能把 uint8 绕回去把门误关。
//
// tail 里 bit 6 的旧副本不受影响：门只作用在被抬到 bit NOREV_SHIFT 的那一份上，
// 所以门关时逐位复现 V22（V22 的 key 在 bit 6 有一份低优先级的 not_reversed）。
//
// 另外两种「更省指令」的写法都建过，服务器 clang 实测更差
// （控制组 .text 5484 / .rodata 2816 / jumps 56 / cmov 36 / stackrefs 156）：
//   b) 先把四个位从 tail 字副本里一次掩掉（每角色 1 条 and）：
//      .text 5832、**.rodata +64**、**jumps 57**、cmov 40、stackrefs 164
//   c) 装配后对 qbase 一条向量 and：
//      .text 5902、.rodata +64、jumps 56、cmov 38、**stackrefs 183**
// 本版（逐 lane）：.text 5809、**.rodata 不变**、jumps 56、cmov 38、stackrefs 166。
// 按 AGENTS.md 的优先级「先无分支 > 再减体积 > 最后才是减指令数」取这一版：
// 少的那几条指令换来新分支或新常量并不值得。
#define V7_NOREV_BIT(t, q) \
    (static_cast<int>(((((t) >> (3 * (q) + 2)) & 1u) & norev_gate) << NOREV_SHIFT))
#define V7_NOREV_GATE(unit) \
    const uint32_t norev_gate = \
        (state.dry_streak[unit] == V7_NOREV_DRY) ? 1u : 0u;
#else
#define V7_NOREV_BIT(t, q) \
    (static_cast<int>((((t) >> (3 * (q) + 2)) & 1u) << NOREV_SHIFT))
// =2 无条件生效，门恒开，所以什么都不用声明。
#define V7_NOREV_GATE(unit)
#endif
#else
#define V7_NOREV_BIT(t, q) 0
#define V7_NOREV_GATE(unit)
#endif

// ---------------------------------------------------------------- 持久状态

struct PersistentState {
#if NOREV_MASKED_PREV_PAD
    signed char prev_quadrant[4];
#else
    signed char prev_quadrant[2];
#endif
    signed char target_row[2];
    signed char target_col[2];
#if V7_ZONE
    // 9 个粗区的产出 EMA。uint16 的量程足够：喂进去的是「单回合平均产出」
    // （见 settle_zones 里的 >> ZONE_PERIOD_LOG），稳态约等于它 << DECAY，
    // 单回合产出上限 532，所以稳态不超过 8512，远不到 65535。
    uint16_t zone_ema[9];
    // 自上次结算以来累积的收益。热路径只往这里加，区索引和 EMA 更新
    // 全部推迟到 V7_ZONE_PERIOD 回合一次的冷函数里 ——
    // 第一版每回合都做区索引 + EMA 读改写，那个数据相关的
    // store-to-load 依赖让本机慢了 5ns。
    uint16_t acc_gold[2];
#endif
#if V7_RICH
    // 上一回合结束时各角色的持币，用来算本回合的净增量。
    // int16 足够：实测单个角色终局持币最多约 2000，远小于 32767，
    // 而用 16 位让这两个字段和上面三对 byte 一起挤在同一条 cache line 里。
    int16_t prev_gold[2];
    // 刻意**不**另设 rich_row/rich_col：target_* 本来就只有「中心」和
    // 「富点」两种取值，让它兼任记忆即可 —— 少两个字段、少两次写。
    // 代价是富点恰好落在 (8,8) 时无法与「没有富点」区分，
    // 但那种情况下的行为（朝中心）正好也是我们要的，所以无害。
#endif
#if V7_NOREV_HIGH == 3
    // 连续多少个回合的规划收益是 0。在 V7_NOREV_DRY 处饱和，所以：
    //   * 门的判据是相等比较，一条 cmp；
    //   * 500 回合的长干涸不会把 uint8 绕回 0 从而误关门。
    // 刻意放在结构体**最后**：上面那七组按位置初始化的 state 初值（V7_SPLIT_TARGET
    // 的各分支）都只写前三个数组，追加的字段会被零初始化，正是要的初值。
    uint8_t dry_streak[2];
#endif
#if V49_NOREV_GATE
    // H：干旱计数器 + 折进表索引的门。
    //
    // dry_streak = 连续多少个回合的规划收益是 0，在 V49_DRY 处**饱和**：
    //   * 门的判据因此是一次相等比较，一条 cmp；
    //   * 500 回合的长干旱不会把 uint8 绕回 0 从而把门误关。
    // tail_bias = 门的物化形式，取值 20（门开，索引落进 QB16 上半区）或 0。
    //   把门做成**索引偏置**而不是运行期向量，是这一轴「零延迟」的全部机制：
    //   热路径那次加法折进本来就存在的寻址计算里。
    //
    // 刻意放在结构体**最后**：上面那七组按位置初始化的 state 初值（V7_SPLIT_TARGET
    // 的各分支）都只写前三个数组，追加的字段会被零初始化 ——
    // dry_streak = 0（无干旱）、tail_bias = 0（门关），正是要的初值。
    // 两个字段都是 uint8[2]，与 prev_quadrant / target_* 一起挤在同一条
    // cache line 上（sizeof(PersistentState) 见下面的 static_assert），
    // 所以读 tail_bias 不引入任何新的 cache line 触达。
    uint8_t dry_streak[2];
    uint8_t tail_bias[2];
#if V49_ROWFOLD
    // I：`(prev_quadrant + 1) * 4 + tail_bias`，在回合末尾一次算好。
    // 初值 0 = prev_quadrant −1（无上一象限）且门关，正是要的。
    uint8_t qb16_row[2];
#endif
#endif
};

#if V49_NOREV_GATE
// 门要免费的前提：tail_bias 必须与 prev_quadrant / target_* 同在一条 cache line 上。
// 当前配置下整个结构体只有 10 字节，所以这条是构造性成立的，钉住它防止以后
// 有人往前面塞字段把 tail_bias 推出去。
static_assert(sizeof(PersistentState) <= 64,
              "tail_bias 必须与 prev_quadrant / target_* 同在一条 cache line 上");
static_assert(V49_DRY >= 1 && V49_DRY <= 254,
              "计数器是 uint8 且在阈值处饱和，阈值必须落在 1..254");
#endif

// 初值即 V6 语义：无上一象限，目标是地图中心 (8,8)。
#if V7_SPLIT_TARGET == 6
PersistentState state = {{-1, -1}, {8, 8}, {3, 8}};      // (8,3) / (8,8) 一守一游
#elif V7_SPLIT_TARGET == 5
PersistentState state = {{-1, -1}, {8, 8}, {5, 11}};     // (8,5) / (8,11)
#elif V7_SPLIT_TARGET == 4
PersistentState state = {{-1, -1}, {8, 8}, {3, 13}};     // (8,3) / (8,13)
#elif V7_SPLIT_TARGET == 3
PersistentState state = {{-1, -1}, {8, 8}, {1, 15}};     // (8,1) / (8,15)
#elif V7_SPLIT_TARGET == 2
PersistentState state = {{-1, -1}, {4, 12}, {4, 12}};    // (4,4) / (12,12)
#elif V7_SPLIT_TARGET == 1
PersistentState state = {{-1, -1}, {6, 10}, {6, 10}};    // (6,6) / (10,10)
#else
PersistentState state = {{-1, -1}, {8, 8}, {8, 8}
#if V7_ZONE
    // 乐观初始化：见文件头 V7_ZONE_OPTIMISTIC 处的说明。
    // 没去过的区必须看起来比去过的更有吸引力，否则整局会锁死在开局那两个区。
    // 区索引 = zr*3+zc。区 3 = (8,3) 所在区、区 5 = (8,13) 所在区，
    // 也就是 v17 手调出来的那对坐标 —— 用更高的起步值把它做成先验。
    , {V7_ZONE_OPTIMISTIC, V7_ZONE_OPTIMISTIC, V7_ZONE_OPTIMISTIC,
       V7_ZONE_PRIOR,      V7_ZONE_OPTIMISTIC, V7_ZONE_PRIOR,
       V7_ZONE_OPTIMISTIC, V7_ZONE_OPTIMISTIC, V7_ZONE_OPTIMISTIC}
    , {0, 0}
#endif
#if V7_RICH
    , {0, 0}
#endif
};
#endif

// Declared here (rather than next to choose_k) so V44_TAIL=2 can hand the joint
// choice straight out of solve_pair without a round trip through UnitResult.
struct KChoice {
    int k;
    int32_t key0;
    int32_t key1;
};

struct UnitResult {
    int32_t k2;
    int32_t k3;
    int32_t k4;
#if !V33_WINNER_HLUT
    uint32_t parent_mask;
#endif
};

// ---------------------------------------------------------------- 标量分类（冷回退）

inline int pick65(int value) { return value >= 1 ? (value * 13 + 19) / 20 : 0; }

inline bool is_blocked_terrain(int value) {
    return value == WALL || value == BOMB || value == FOG;
}

constexpr int32_t small_cell_score(int value) {
    if (value == WALL || value == BOMB || value == FOG) return NEG;
    return value >= 1 ? ((value * 13 + 19) / 20) << GOLD_SHIFT : 0;
}

constexpr std::array<int32_t, CELL_LUT_MAX + 6> make_cell_lut() {
    std::array<int32_t, CELL_LUT_MAX + 6> result = {};
    for (int value = -5; value <= CELL_LUT_MAX; ++value) {
        result[static_cast<unsigned>(value + 5)] = small_cell_score(value);
    }
    return result;
}

constexpr std::array<int32_t, CELL_LUT_MAX + 6> make_phase_cell_lut() {
    auto result = make_cell_lut();
    result[static_cast<unsigned>(BOMB + 5)] = 0;   // round 0-12 炸弹可穿过
    return result;
}

constexpr std::array<uint8_t, GRID_SIZE> make_colgeom_lut() {
    std::array<uint8_t, GRID_SIZE> result = {};
    for (int c = 0; c < GRID_SIZE; ++c) {
        const int x = c - 2;
        const int cb = x < 0 ? 0 : (x > GRID_SIZE - 8 ? GRID_SIZE - 8 : x);
        const int off_index = x - cb + 2;
        result[static_cast<unsigned>(c)] = static_cast<uint8_t>(
            static_cast<unsigned>(cb) | (static_cast<unsigned>(off_index) << 4));
    }
    return result;
}

constexpr auto COLGEOM_LUT = make_colgeom_lut();
constexpr bool colgeom_lut_is_exact() {
    for (int c = 0; c < GRID_SIZE; ++c) {
        const int x = c - 2;
        const int cb = x < 0 ? 0 : (x > GRID_SIZE - 8 ? GRID_SIZE - 8 : x);
        const int off_index = x - cb + 2;
        const unsigned packed = COLGEOM_LUT[static_cast<unsigned>(c)];
        if (static_cast<int>(packed & 15u) != cb ||
            static_cast<int>(packed >> 4) != off_index) return false;
    }
    return true;
}
static_assert(colgeom_lut_is_exact(), "packed column geometry must equal both clamps");

// 唯一可写的一格：round 13 时一次性改成 NEG。
auto CELL_LUT = make_phase_cell_lut();

// 向量路径做同一件事：相位前 bombpat = -3（把 -3 从 blocked 里挖掉），
// 相位后换成 0（0 不可能 < 0，于是 blocked 退化成纯 raw<0）。
int32_t g_bomb_pattern = BOMB;

#if V44_MASKVEC
// V44 axis B carries the same phase in one AND-mask instead of a compare value:
// bit31 | bit1 | bit0 before the phase (blocked == {-1,-5}), bit31 | bit0 after it
// (blocked == {-1,-3,-5}).  See classify_row_pair_vec for the enumeration.
#if NOREV_DIRECT_BLOCKBITS
volatile int32_t g_block_bits = static_cast<int32_t>(0x80000003u);
#else
int32_t g_block_bits = static_cast<int32_t>(0x80000003u);
#endif
constexpr int32_t V44_BLOCK_BITS_PHASE2 = static_cast<int32_t>(0x80000001u);
#endif

inline int32_t classify_cell(int value) {
    const uint32_t index = static_cast<uint32_t>(value) + 5u;
    if (__builtin_expect(index < CELL_LUT.size(), 1)) return CELL_LUT[index];
    if (is_blocked_terrain(value)) return NEG;
    return pick65(value) << GOLD_SHIFT;
}

// ---------------------------------------------------------------- 小表

#if !V33_WINNER_HLUT
constexpr std::array<uint16_t, 16> make_spread_parent_lut() {
    std::array<uint16_t, 16> result = {};
    for (int bits = 0; bits < 16; ++bits) {
        uint16_t spread = 0;
        for (int q = 0; q < 4; ++q) {
            spread |= static_cast<uint16_t>(((bits >> q) & 1) << (4 * q));
        }
        result[bits] = spread;
    }
    return result;
}
[[maybe_unused]] constexpr auto SPREAD_PARENT_LUT = make_spread_parent_lut();
#endif

constexpr std::array<uint16_t, 20> make_tail_lut() {
    std::array<uint16_t, 20> result = {};
    for (int previous_index = 0; previous_index < 5; ++previous_index) {
        const int previous = previous_index - 1;
        for (int preferred = 0; preferred < 4; ++preferred) {
            uint16_t packed = 0;
            for (int q = 0; q < 4; ++q) {
                const uint32_t not_reversed = previous < 0 || q != (previous ^ 3);
                const uint32_t qrank = static_cast<uint32_t>((~(q ^ preferred)) & 3);
                packed |= static_cast<uint16_t>((qrank | (not_reversed << 2)) << (3 * q));
            }
            result[previous_index * 4 + preferred] = packed;
        }
    }
    return result;
}
constexpr auto TAIL_LUT = make_tail_lut();

#if V7_NOREV_HIGH >= 2
// 编译期穷举：20 个 TAIL_LUT 条目 x 4 个象限，提取出的位必须恰好等于该象限的
// not_reversed 且不污染任何其它位。第一版用一次融合移位，对 q=3 得到 -1 的负移位
// （UB），线上表现为反向率被压到被迫下限之下；这条断言就是为了不再出现那种错误。
// =3 的门在这里恒开，所以验的是同一个提取；门本身的正确性由回合末尾的饱和更新
// 与 V7_NOREV_DRY 的范围检查保证。
constexpr bool norev_bit_extraction_is_exact() {
    for (int i = 0; i < 20; ++i) {
        const uint32_t t = TAIL_LUT[static_cast<unsigned>(i)];
        // =3 的提取多一个门；这里验的是门开那一档，=2 下这个名字用不到。
        const uint32_t norev_gate = 1u;
        (void)norev_gate;
        for (int q = 0; q < 4; ++q) {
            const uint32_t want = (t >> (3 * q + 2)) & 1u;
            if (V7_NOREV_BIT(t, q) != static_cast<int>(want << NOREV_SHIFT)) return false;
        }
    }
    return true;
}
static_assert(norev_bit_extraction_is_exact(),
              "反向位的提取必须与 tail 字里的 not_reversed 逐位一致");
#endif

// qbase 只依赖 packed_tails，后者只有 20 种取值 —— 直接把 __m128i 预计算掉。
struct alignas(16) QBase {
    int32_t lane[4];
};

constexpr std::array<QBase, 20> make_qbase_lut() {
    std::array<QBase, 20> result = {};
    const auto tails = make_tail_lut();
    for (int i = 0; i < 20; ++i) {
        const uint32_t packed = tails[static_cast<unsigned>(i)];
        for (int q = 0; q < 4; ++q) {
            const uint32_t tail = (packed >> (3 * q)) & 7u;
            result[static_cast<unsigned>(i)].lane[q] =
                static_cast<int32_t>((tail << 4) | (static_cast<uint32_t>(q) << 2));
        }
    }
    return result;
}
constexpr auto QBASE_LUT = make_qbase_lut();

#if V49_QBASE16
// ------------------------------------------------------------------ G/H：QB16
//
// 一条 qbase lane 是 `(tail << 4) | (q << 2)`，tail = (packed >> 3q) & 7 <= 7，
// 所以 lane <= (7 << 4) | (3 << 2) = 0x7C；带上 bit-10 的 not_reversed 副本也只到
// 0x47C。**完整落在 uint16 里**，于是每个 tail_index 的四条 lane 只占 8 字节，
// 热路径一条 `vmovq` 就取完一个角色的整组 qbase。
//
// 40 项，两个半区：
//   [0, 20)  门关：与 V37 那条展开链逐位相同（下面 constexpr 穷举核对）。
//   [20, 40) 门开：同上再 OR 上 bit NOREV_SHIFT 的 not_reversed 副本。
// 门因此**折进索引**（+0 或 +20），热路径不需要任何运行期的门计算。
//
// 所有值都在编译期烤好。**刻意不在运行期算移位量** —— 早期版本用一次融合移位
// `((t) & (4u << 3q)) << (NOREV_SHIFT - 2 - 3q)`，从 bit 11 挪到 bit 10 之后
// q=3 那条的移位量是 −1（**负移位 = UB**）。它编译通过、5,988 回合零非法动作，
// 却把掉头率压到 10.69%，低于 14.13% 的被迫下限。常规门禁抓不到那种错误，
// 只有「烤进表 + 穷举断言」能。
struct QB16Entry {
    uint16_t lane[4];
};
static_assert(sizeof(QB16Entry) == 8, "一项必须正好 8 字节，热路径靠一条 vmovq 取它");

constexpr int QB16_GATE_STRIDE = 20;
// 门没编译进来时上半区不可达，所以表就只有 20 项 / 160 字节 ——
// 纯延迟那个开关不为它还没用上的门付一个字节的 .rodata。
constexpr int QB16_ENTRIES = V49_NOREV_GATE ? 40 : 20;
#if V49_NOREV_GATE || V50_NOREV_GLOBAL
constexpr int QB16_NOREV_SHIFT = NOREV_SHIFT;
#else
constexpr int QB16_NOREV_SHIFT = 0;   // 门没编译进来，这个名字只是让表构造式成立
#endif

constexpr std::array<QB16Entry, QB16_ENTRIES> make_qb16_lut() {
    std::array<QB16Entry, QB16_ENTRIES> result = {};
    const auto tails = make_tail_lut();
    for (int i = 0; i < 20; ++i) {
        const uint32_t packed = tails[static_cast<unsigned>(i)];
        for (int q = 0; q < 4; ++q) {
            const uint32_t tail = (packed >> (3 * q)) & 7u;
            const uint32_t base = (tail << 4) | (static_cast<uint32_t>(q) << 2);
            const uint32_t not_reversed = (packed >> (3 * q + 2)) & 1u;
            const uint32_t global_norev = V50_NOREV_GLOBAL
                ? (not_reversed << QB16_NOREV_SHIFT) : 0u;
            result[static_cast<unsigned>(i)].lane[static_cast<unsigned>(q)] =
                static_cast<uint16_t>(base | global_norev);
            if (QB16_ENTRIES > 20) {
                // not_reversed 是每象限 3 位里的最高位，即 packed 的 bit (3q + 2)。
                // 移位量 3q+2 是**循环常量**，恒为 2/5/8/11，永不为负 ——
                // 这正是那个 UB 陷阱被绕开的地方，而且这里在编译期求值。
                result[static_cast<unsigned>(i + QB16_GATE_STRIDE)]
                    .lane[static_cast<unsigned>(q)] = static_cast<uint16_t>(
                        base | (not_reversed << QB16_NOREV_SHIFT));
            }
        }
    }
    return result;
}
constexpr auto QB16 = make_qb16_lut();
static_assert(sizeof(QB16) == static_cast<size_t>(QB16_ENTRIES) * 8,
              "每项 4 lane x uint16 = 8 字节，不允许有填充");
static_assert(sizeof(QB16) == (V49_NOREV_GATE ? 320u : 160u),
              "门关 20 项 = 160 字节；门开 40 项 = 320 字节");

// (1) 每一项都必须真的落在 uint16 里 —— 这是整张表能存在的前提。
//     constexpr 求值里 uint16_t 会截断而不报错，所以从 int 侧重算一遍再比。
constexpr bool qb16_fits_uint16() {
    const auto tails = make_tail_lut();
    for (int i = 0; i < 20; ++i) {
        const uint32_t packed = tails[static_cast<unsigned>(i)];
        for (int q = 0; q < 4; ++q) {
            const uint32_t tail = (packed >> (3 * q)) & 7u;
            if (tail > 7u) return false;
            const uint32_t base = (tail << 4) | (static_cast<uint32_t>(q) << 2);
            if (base > 0xFFFFu) return false;
            if (base > 0x7Cu) return false;          // (7<<4)|(3<<2)
#if V49_NOREV_GATE || V50_NOREV_GLOBAL
            const uint32_t gated = base | (1u << NOREV_SHIFT);
            if (gated > 0xFFFFu) return false;
            if (gated > 0x47Cu) return false;        // 0x7C | 0x400
#endif
        }
    }
    return true;
}
static_assert(qb16_fits_uint16(), "每条 QB16 lane 都必须完整落在 uint16 里");

// (2) 门关的那半区必须**逐位复现 V37 那条展开链算出来的 qbase**。
//     这里把旧表达式原样重写一遍再逐项比对，20 个 tail_index x 4 lane 全覆盖。
//     旧链（player_v7.cpp 的 V37_QBASE_PAIR 分支）：
//        tails = (broadcast(packed) >> {0,3,6,9}) & 7
//        qbase = (tails << 4) | {0,4,8,12}
//     注意 {0,4,8,12} 就是 q << 2，所以下面的 `q << 2` 与那个常量向量同义。
constexpr bool qb16_base_matches_selected_ordering() {
    const auto tails = make_tail_lut();
    for (int i = 0; i < 20; ++i) {
        const uint32_t packed = tails[static_cast<unsigned>(i)];
        for (int q = 0; q < 4; ++q) {
            const uint32_t shift = static_cast<uint32_t>(3 * q);       // {0,3,6,9}
            const uint32_t quadrant_const = static_cast<uint32_t>(q) << 2;  // {0,4,8,12}
            uint32_t old_lane = (((packed >> shift) & 7u) << 4) | quadrant_const;
#if V50_NOREV_GLOBAL
            const uint32_t not_reversed = (packed >> (3 * q + 2)) & 1u;
            old_lane |= not_reversed << NOREV_SHIFT;
#endif
            if (QB16[static_cast<unsigned>(i)].lane[static_cast<unsigned>(q)] !=
                static_cast<uint16_t>(old_lane)) return false;
        }
    }
    return true;
}
static_assert(qb16_base_matches_selected_ordering(),
              "QB16 base half must exactly match the selected global ordering");

#if V49_NOREV_GATE
// (3) 门开半区与门关半区**只差 bit NOREV_SHIFT，且只在 not_reversed 置位处相差**。
//     这条同时否掉两类错误：污染了别的位，以及在该置位的地方漏置 / 在不该置位的
//     地方乱置。`want` 是从 (previous, q) **独立重算**的，不复用 tail 字里那份，
//     所以它同时核对了「表里的 not_reversed 就是 make_tail_lut 的那个定义」。
constexpr bool qb16_gated_half_differs_only_in_norev_bit() {
    const auto tails = make_tail_lut();
    for (int i = 0; i < 20; ++i) {
        const uint32_t packed = tails[static_cast<unsigned>(i)];
        // tail_index = (previous + 1) * 4 + preferred，所以 previous = i / 4 - 1。
        const int previous = i / 4 - 1;
        for (int q = 0; q < 4; ++q) {
            const uint32_t lo = QB16[static_cast<unsigned>(i)]
                                    .lane[static_cast<unsigned>(q)];
            const uint32_t hi = QB16[static_cast<unsigned>(i + QB16_GATE_STRIDE)]
                                    .lane[static_cast<unsigned>(q)];
            const uint32_t want = (previous < 0 || q != (previous ^ 3)) ? 1u : 0u;
            if (((packed >> (3 * q + 2)) & 1u) != want) return false;
            if ((hi ^ lo) != (want << NOREV_SHIFT)) return false;
        }
    }
    return true;
}
static_assert(qb16_gated_half_differs_only_in_norev_bit(),
              "门开半区只能在 bit NOREV_SHIFT 上、且只在 not_reversed 处与门关半区不同");

// (4) 表里 dist 字段的三个有效位必须恒为 0：qbase 只带 tail / 象限 / endpoint，
//     dist 是在 solve_pair 里 OR 进来的。若 QB16 污染了 bits 7..9，dist 会被改写。
constexpr bool qb16_leaves_dist_field_alone() {
    for (int i = 0; i < QB16_ENTRIES; ++i)
        for (int q = 0; q < 4; ++q)
            if ((QB16[static_cast<unsigned>(i)].lane[static_cast<unsigned>(q)] &
                 (7u << DIST_SHIFT)) != 0u) return false;
    return true;
}
static_assert(qb16_leaves_dist_field_alone(),
              "QB16 不得碰 dist 的三个有效位，那三位由 solve_pair 写");
#endif
#endif  // V49_QBASE16

#if V7_ARITH_EMIT != 1
constexpr std::array<uint8_t, 5 * 4 * 16> make_horizontal_lut() {
    std::array<uint8_t, 5 * 4 * 16> result = {};
    for (int parents = 0; parents < 16; ++parents) {
        const uint32_t p11 = parents & 1u;
        const uint32_t p12 = (parents >> 1) & 1u;
        const uint32_t p21 = (parents >> 2) & 1u;
        const uint32_t p22 = (parents >> 3) & 1u;
        const uint32_t pat11 = p11 ? 0x2u : 0x1u;
        const uint32_t pat12 = p12 ? (pat11 | 0x4u) : 0x3u;
        const uint32_t pat21 = p21 ? 0x4u : pat11;
        const uint32_t pat22 = p22 ? (pat21 | 0x8u) : pat12;
        for (int endpoint = 0; endpoint < 4; ++endpoint) {
            for (int moved = 0; moved <= 4; ++moved) {
                uint32_t horizontal = 0;
                if (moved == 4) horizontal = pat22;
                else if (moved == 3) horizontal = endpoint == 2 ? pat12 : pat21;
                else if (moved == 2) horizontal = endpoint == 0 ? 0u : endpoint == 1 ? 0x3u : pat11;
                else if (moved == 1) horizontal = endpoint == 0 ? 0u : 0x1u;
                result[(parents * 4 + endpoint) * 5 + moved] = static_cast<uint8_t>(horizontal);
            }
        }
    }
    return result;
}
constexpr auto HORIZONTAL_LUT = make_horizontal_lut();
#endif

#if !V7_ARITH_EMIT

constexpr std::array<uint32_t, 4 * 5 * 16> make_action_lut() {
    std::array<uint32_t, 4 * 5 * 16> result = {};
    for (int quadrant = 0; quadrant < 4; ++quadrant) {
        const int vertical = (quadrant & 1) ? A_DOWN : A_UP;
        const int horizontal = (quadrant & 2) ? A_RIGHT : A_LEFT;
        for (int moved = 0; moved <= 4; ++moved) {
            for (int mask = 0; mask < 16; ++mask) {
                uint32_t packed = 0;
                for (int step = 0; step < 4; ++step) {
                    const int action = step >= moved ? A_STAY
                        : ((mask >> step) & 1) ? horizontal : vertical;
                    packed |= static_cast<uint32_t>(action) << (8 * step);
                }
                result[(quadrant * 5 + moved) * 16 + mask] = packed;
            }
        }
    }
    return result;
}
#if !V7_LUT16
constexpr auto ACTION_LUT = make_action_lut();
#endif

#if NOREV_DIRECT_EMIT == 1
// 5 moved × 16 (quadrant:2 + endpoint:2) × 16 parent patterns.
// 1,280 个 uint32 = 5,120B。虽然表比原来的两张表大，但一次 emit 只碰
// 一个最终 cache line，并删除 HORIZONTAL_LUT load 到 ACTION_LUT load 之间的
// 串行依赖。每个 metadata group 的 16 个 parent pattern 恰好占一条 64B line。
constexpr std::array<uint32_t, 5 * 16 * 16> make_final_action_lut() {
    const auto horizontal = make_horizontal_lut();
    const auto actions = make_action_lut();
    std::array<uint32_t, 5 * 16 * 16> result = {};
    for (unsigned moved = 0; moved <= 4; ++moved) {
        for (unsigned qep = 0; qep < 16; ++qep) {
            const unsigned quadrant = qep >> 2;
            const unsigned endpoint = qep & 3u;
            for (unsigned parents = 0; parents < 16; ++parents) {
                const unsigned mask = horizontal[(parents * 4 + endpoint) * 5 + moved];
                const unsigned dst = parents | (qep << 4) | (moved << 8);
                result[dst] = actions[(quadrant * 5 + moved) * 16 + mask];
            }
        }
    }
    return result;
}

alignas(64) constexpr auto FINAL_ACTION_LUT = make_final_action_lut();
static_assert(sizeof(FINAL_ACTION_LUT) == 5120,
              "direct emit table must be exactly 80 cache lines");

constexpr bool final_action_lut_is_lossless() {
    const auto horizontal = make_horizontal_lut();
    const auto actions = make_action_lut();
    const auto final = make_final_action_lut();
    for (unsigned moved = 0; moved <= 4; ++moved) {
        for (unsigned qep = 0; qep < 16; ++qep) {
            const unsigned quadrant = qep >> 2;
            const unsigned endpoint = qep & 3u;
            for (unsigned parents = 0; parents < 16; ++parents) {
                const unsigned mask = horizontal[(parents * 4 + endpoint) * 5 + moved];
                const uint32_t old_value = actions[(quadrant * 5 + moved) * 16 + mask];
                const uint32_t new_value = final[parents | (qep << 4) | (moved << 8)];
                if (old_value != new_value) return false;
            }
        }
    }
    return true;
}
static_assert(final_action_lut_is_lossless(),
              "direct emit must equal HORIZONTAL_LUT -> ACTION_LUT for all 1280 inputs");
#elif NOREV_DIRECT_EMIT == 2
// Only ten (moved, endpoint) pairs can ever be emitted:
//   0:{0}, 1:{0,1}, 2:{0,1,2}, 3:{2,3}, 4:{2}.
// The injective code 2*moved+endpoint maps those pairs to
// {0,2,3,4,5,6,8,9,10}; with quadrant and parents the maximum index is 703.
constexpr bool direct_emit_endpoint_is_reachable(unsigned moved, unsigned endpoint) {
    return (moved == 0 && endpoint == 0) ||
           (moved == 1 && endpoint <= 1) ||
           (moved == 2 && endpoint <= 2) ||
           (moved == 3 && endpoint >= 2) ||
           (moved == 4 && endpoint == 2);
}

constexpr std::array<uint32_t, 704> make_compact_final_action_lut() {
    const auto horizontal = make_horizontal_lut();
    const auto actions = make_action_lut();
    std::array<uint32_t, 704> result = {};
    for (unsigned moved = 0; moved <= 4; ++moved) {
        for (unsigned endpoint = 0; endpoint < 4; ++endpoint) {
            if (!direct_emit_endpoint_is_reachable(moved, endpoint)) continue;
            const unsigned meta = 2 * moved + endpoint;
            for (unsigned quadrant = 0; quadrant < 4; ++quadrant) {
                for (unsigned parents = 0; parents < 16; ++parents) {
                    const unsigned mask = horizontal[(parents * 4 + endpoint) * 5 + moved];
                    const unsigned dst = parents | (quadrant << 4) | (meta << 6);
                    result[dst] = actions[(quadrant * 5 + moved) * 16 + mask];
                }
            }
        }
    }
    return result;
}

alignas(64) constexpr auto COMPACT_FINAL_ACTION_LUT =
    make_compact_final_action_lut();
static_assert(sizeof(COMPACT_FINAL_ACTION_LUT) == 2816,
              "compact direct emit table must be exactly 44 cache lines");

constexpr bool compact_final_action_lut_is_lossless() {
    const auto horizontal = make_horizontal_lut();
    const auto actions = make_action_lut();
    const auto compact = make_compact_final_action_lut();
    for (unsigned moved = 0; moved <= 4; ++moved) {
        for (unsigned endpoint = 0; endpoint < 4; ++endpoint) {
            if (!direct_emit_endpoint_is_reachable(moved, endpoint)) continue;
            const unsigned meta = 2 * moved + endpoint;
            for (unsigned quadrant = 0; quadrant < 4; ++quadrant) {
                for (unsigned parents = 0; parents < 16; ++parents) {
                    const unsigned mask = horizontal[(parents * 4 + endpoint) * 5 + moved];
                    const uint32_t old_value = actions[(quadrant * 5 + moved) * 16 + mask];
                    const uint32_t new_value = compact[
                        parents | (quadrant << 4) | (meta << 6)];
                    if (old_value != new_value) return false;
                }
            }
        }
    }
    return true;
}
static_assert(compact_final_action_lut_is_lossless(),
              "compact direct emit must equal the old two-level LUT on every reachable key");

#if NOREV_PAIR_COMPACT
// Prove the exact index shape used by emit_parent_pair below.  In particular,
// endpoint comes from key bits 0..1 and quadrant from bits 2..3; every reachable
// (moved, endpoint) pair must remain in the 704-entry compact table and return
// the same four packed actions as the original two-level reconstruction.
constexpr bool pair_compact_index_is_lossless() {
    const auto horizontal = make_horizontal_lut();
    const auto actions = make_action_lut();
    const auto compact = make_compact_final_action_lut();
    for (unsigned moved = 0; moved <= 4; ++moved) {
        for (unsigned endpoint = 0; endpoint < 4; ++endpoint) {
            if (!direct_emit_endpoint_is_reachable(moved, endpoint)) continue;
            for (unsigned quadrant = 0; quadrant < 4; ++quadrant) {
                const unsigned key_low = (quadrant << 2) | endpoint;
                const unsigned q = (key_low >> 2) & 3u;
                const unsigned ep = key_low & 3u;
                const unsigned meta = 2 * moved + ep;
                for (unsigned parents = 0; parents < 16; ++parents) {
                    const unsigned index = parents | (q << 4) | (meta << 6);
                    if (index >= compact.size()) return false;
                    const unsigned mask = horizontal[(parents * 4 + ep) * 5 + moved];
                    const uint32_t expected =
                        actions[(q * 5 + moved) * 16 + mask];
                    if (compact[index] != expected) return false;
                }
            }
        }
    }
    return true;
}
static_assert(pair_compact_index_is_lossless(),
              "paired compact emit must cover every reachable moved/endpoint key");
#endif
#endif

#if V7_LUT16
// ACTION_LUT 的半宽版本：每项从 uint32 压成 uint16。
//
// 为什么可以压：表里每个字节存一个动作码，取值只有 0..4（A_UP/DOWN/LEFT/RIGHT
// 各 0..3，A_STAY=4），所以 3 bit 就够，4 个动作 = 12 bit ≤ 16 bit。
//
// 为什么值得压：`emit_parent` 每次 `moveDecision` 被调用两次（两个单位各一次），
// 两次的 (quadrant, moved, mask) 通常不同。uint32 版 1280 字节横跨 20 条
// cache line，两次访问几乎必然落在两条不同的线上；压到 640 字节 = 10 条线后，
// 撞进同一条线的概率翻倍。表本身是冷的（每回合只碰 2 项），所以这里省的是
// **cache line 数**，不是字节数 —— 与第 6.8 节热表打包同一个靶子。
//
// 代价：展开时多一条 `pdep`。mask 0x07070707 恰好有 12 个置位，与 12 bit 源
// 一一对应，所以一条指令就能还原成「每字节一个动作码」的原始布局，
// 展开结果与 uint32 版逐位相同（下面的 static_assert 直接验证这一点）。
constexpr std::array<uint16_t, 4 * 5 * 16> make_action_lut16() {
    const auto full = make_action_lut();
    std::array<uint16_t, 4 * 5 * 16> result = {};
    for (unsigned i = 0; i < full.size(); ++i) {
        uint16_t packed = 0;
        for (int step = 0; step < 4; ++step) {
            packed |= static_cast<uint16_t>(((full[i] >> (8 * step)) & 7u) << (3 * step));
        }
        result[i] = packed;
    }
    return result;
}
constexpr auto ACTION_LUT16 = make_action_lut16();
static_assert(sizeof(ACTION_LUT16) == 640, "半宽表应当是 640 字节 = 10 条 cache line");

// 编译期证明「压缩再展开 == 原表」：每个字节都必须 <8，否则 3 bit 装不下。
constexpr bool action_lut16_is_lossless() {
    const auto full = make_action_lut();
    const auto half = make_action_lut16();
    for (unsigned i = 0; i < full.size(); ++i) {
        uint32_t restored = 0;
        for (int step = 0; step < 4; ++step) {
            restored |= static_cast<uint32_t>((half[i] >> (3 * step)) & 7u) << (8 * step);
        }
        if (restored != full[i]) return false;
    }
    return true;
}
static_assert(action_lut16_is_lossless(), "半宽表展开后必须与 uint32 表逐位相同");
#endif  // V7_LUT16

#endif  // !V7_ARITH_EMIT

// ---------------------------------------------------------------- 向量窗口的表
//
// 每行宽读 8 个 int（列基址钳位到 [0,9]，保证 cb+7 <= 16 永不越界），
// 然后一次 vpermd 把窗口列搬到固定 lane：
//   行 0/1/3/4 (IDX_Q): lane0=wc0 lane1=wc1 lane2=wc3 lane3=wc4 lane4..7=wc2
//   行 2       (IDX_H): lane0=wc1 lane1=wc3 lane2=wc0 lane3=wc4 lane4..7=wc2(未用)
// wc2 必须占满 lane4..7：取轴向值时用 extracti128 拿上半 128 位，
// 紧接的 unpacklo 会同时读 lane4 和 lane5，两者必须相等。

struct alignas(32) Idx8 {
    int32_t lane[8];
};

constexpr std::array<Idx8, 8> make_idx_q_lut() {
    std::array<Idx8, 8> table = {};
    const int order[5] = {0, 1, 3, 4, 2};
    for (int o = 0; o < 8; ++o) {
        const int off = o - 2;
        for (int i = 0; i < 4; ++i) {
            table[static_cast<unsigned>(o)].lane[i] = (off + order[i]) & 7;
        }
        for (int i = 4; i < 8; ++i) {
            table[static_cast<unsigned>(o)].lane[i] = (off + 2) & 7;
        }
    }
    return table;
}

constexpr std::array<Idx8, 8> make_idx_h_lut() {
    std::array<Idx8, 8> table = {};
    const int order[4] = {1, 3, 0, 4};
    for (int o = 0; o < 8; ++o) {
        const int off = o - 2;
        for (int i = 0; i < 4; ++i) {
            table[static_cast<unsigned>(o)].lane[i] = (off + order[i]) & 7;
        }
        for (int i = 4; i < 8; ++i) {
            table[static_cast<unsigned>(o)].lane[i] = (off + 2) & 7;
        }
    }
    return table;
}
constexpr auto IDX_Q = make_idx_q_lut();
constexpr auto IDX_H = make_idx_h_lut();

#if V46_IDXFUSE
// E（搬自 V43_IDXFUSE）：交错版。同一个 off 的 Q/H 索引向量共享一条 cache line，
// 且热函数只需要一个表基址寄存器（H 用 disp8 = +0x20 寻址）。
struct alignas(64) IdxQH {
    Idx8 q;
    Idx8 h;
};
static_assert(sizeof(IdxQH) == 64, "IdxQH 必须正好一条 cache line");

constexpr std::array<IdxQH, 8> make_idx_qh_lut() {
    std::array<IdxQH, 8> table = {};
    const auto q = make_idx_q_lut();
    const auto h = make_idx_h_lut();
    for (unsigned o = 0; o < 8; ++o) {
        table[o].q = q[o];
        table[o].h = h[o];
    }
    return table;
}
constexpr auto IDX_QH = make_idx_qh_lut();
#define V46_IDXQ(i) (IDX_QH[(i)].q)
#define V46_IDXH(i) (IDX_QH[(i)].h)
#else
#define V46_IDXQ(i) (IDX_Q[(i)])
#define V46_IDXH(i) (IDX_H[(i)])
#endif

#if NOREV_RAW_INT16
// VPACKSSDW places source-A dword i at (i/4)*8+i%4 and source B at +4.
// Interleave those positions so VPMADDWD can select either packed row while
// restoring the existing `cell << GOLD_SHIFT` int32 representation.
constexpr int norev_raw16_pack_pos_a(int i) {
    return (i / 4) * 8 + (i % 4);
}

struct alignas(128) NorevRaw16Idx {
    uint16_t q0[16];
    uint16_t q1[16];
    uint16_t h0[8];
    uint16_t h1[8];
    uint16_t pad[16];
};
static_assert(sizeof(NorevRaw16Idx) == 128,
              "raw-int16 index entry must be exactly two cache lines");

constexpr std::array<NorevRaw16Idx, 8> make_norev_raw16_idx() {
    std::array<NorevRaw16Idx, 8> table = {};
    const auto q = make_idx_q_lut();
    const auto h = make_idx_h_lut();
    for (unsigned off = 0; off < 8; ++off) {
        for (int lane = 0; lane < 8; ++lane) {
            const int a0 = norev_raw16_pack_pos_a(q[off].lane[lane]);
            const int a1 = norev_raw16_pack_pos_a(q[off].lane[lane] + 8);
            table[off].q0[2 * lane] = static_cast<uint16_t>(a0);
            table[off].q0[2 * lane + 1] = static_cast<uint16_t>(a0 + 4);
            table[off].q1[2 * lane] = static_cast<uint16_t>(a1);
            table[off].q1[2 * lane + 1] = static_cast<uint16_t>(a1 + 4);
            table[off].h0[lane] = static_cast<uint16_t>(h[off].lane[lane]);
            table[off].h1[lane] = static_cast<uint16_t>(h[off].lane[lane] + 8);
        }
    }
    return table;
}
constexpr auto NOREV_RAW16_IDX = make_norev_raw16_idx();

alignas(64) constexpr int16_t NOREV_RAW16_MW_EVEN[32] = {
    1 << GOLD_SHIFT, 0, 1 << GOLD_SHIFT, 0,
    1 << GOLD_SHIFT, 0, 1 << GOLD_SHIFT, 0,
    1 << GOLD_SHIFT, 0, 1 << GOLD_SHIFT, 0,
    1 << GOLD_SHIFT, 0, 1 << GOLD_SHIFT, 0,
    1 << GOLD_SHIFT, 0, 1 << GOLD_SHIFT, 0,
    1 << GOLD_SHIFT, 0, 1 << GOLD_SHIFT, 0,
    1 << GOLD_SHIFT, 0, 1 << GOLD_SHIFT, 0,
    1 << GOLD_SHIFT, 0, 1 << GOLD_SHIFT, 0};
alignas(64) constexpr int16_t NOREV_RAW16_MW_ODD[32] = {
    0, 1 << GOLD_SHIFT, 0, 1 << GOLD_SHIFT,
    0, 1 << GOLD_SHIFT, 0, 1 << GOLD_SHIFT,
    0, 1 << GOLD_SHIFT, 0, 1 << GOLD_SHIFT,
    0, 1 << GOLD_SHIFT, 0, 1 << GOLD_SHIFT,
    0, 1 << GOLD_SHIFT, 0, 1 << GOLD_SHIFT,
    0, 1 << GOLD_SHIFT, 0, 1 << GOLD_SHIFT,
    0, 1 << GOLD_SHIFT, 0, 1 << GOLD_SHIFT,
    0, 1 << GOLD_SHIFT, 0, 1 << GOLD_SHIFT};

constexpr int16_t NOREV_RAW16_NEG = static_cast<int16_t>(-32768);
constexpr uint16_t norev_raw16_block_bits(int32_t bits) {
    return static_cast<uint16_t>(0x8000u | (static_cast<uint32_t>(bits) & 3u));
}
constexpr bool norev_raw16_blocking_is_exact(int32_t bits) {
    for (int raw = -5; raw <= GOLD_EXACT_MAX; ++raw) {
        const bool b32 = (raw & bits) == bits;
        const uint16_t raw16 = static_cast<uint16_t>(static_cast<int16_t>(raw));
        const uint16_t b16 = norev_raw16_block_bits(bits);
        if (b32 != ((raw16 & b16) == b16)) return false;
    }
    return true;
}
static_assert(GOLD_EXACT_MAX <= 32767,
              "official raw gold must pack to signed int16 without saturation");
static_assert((1 << GOLD_SHIFT) <= 32767,
              "VPMADDWD multiplier must fit signed int16");
static_assert(int64_t(NOREV_RAW16_NEG) * (int64_t(1) << GOLD_SHIFT) == int64_t(NEG),
              "INT16_MIN widened through VPMADDWD must reproduce NEG exactly");
static_assert(norev_raw16_blocking_is_exact(static_cast<int32_t>(0x80000003u)),
              "pre-phase raw-int16 blocker classification must match int32");
static_assert(norev_raw16_blocking_is_exact(static_cast<int32_t>(0x80000001u)),
              "post-phase raw-int16 blocker classification must match int32");
#endif

// 窗口列 -> lane 位掩码。wc2 覆盖 lane4..7，所以是 0xF0。
constexpr uint8_t LANEBITS_Q[5] = {0x01, 0x02, 0xF0, 0x04, 0x08};
constexpr uint8_t LANEBITS_H[5] = {0x04, 0x01, 0x00, 0x02, 0x08};

// 8 行 x 8 列 = 64 字节，正好一条 cache line。
// 之前用 8x32 步长时，不同窗口行会落在不同 cache line 上：三个占位者分处不同行
// 就要碰 3 条线。线上 D-cache 是冷的，这个差别是实打实的。
// 行/列索引都 & 7 后仍落在表内且越界处为 0，所以不需要任何判断。
constexpr std::array<uint8_t, 8 * 8> make_lanebits_lut() {
    std::array<uint8_t, 8 * 8> table = {};
    for (int wr = 0; wr < 5; ++wr) {
        for (int wc = 0; wc < 5; ++wc) {
            table[static_cast<unsigned>(wr * 8 + wc)] =
                (wr == 2) ? LANEBITS_H[wc] : LANEBITS_Q[wc];
        }
    }
    return table;
}
constexpr auto LANEBITS = make_lanebits_lut();

// 列越界掩码：窗口列 j 对应网格列 c0-2+j，越界则该 lane 要擦成 NEG。
constexpr std::array<uint8_t, GRID_SIZE> make_colmask(bool horizontal_row) {
    std::array<uint8_t, GRID_SIZE> table = {};
    for (int c0 = 0; c0 < GRID_SIZE; ++c0) {
        uint8_t m = 0;
        for (int j = 0; j < 5; ++j) {
            const int c = c0 - 2 + j;
            if (c < 0 || c >= GRID_SIZE) {
                m = static_cast<uint8_t>(m | (horizontal_row ? LANEBITS_H[j] : LANEBITS_Q[j]));
            }
        }
        table[static_cast<unsigned>(c0)] = m;
    }
    return table;
}
constexpr auto COLMASK_Q = make_colmask(false);
constexpr auto COLMASK_H = make_colmask(true);

#if V46_COL5
// D（搬自 V43_COL5）：off -> 该 off 唯一对应的 c0。off != 0 时映射是双射；
// off == 0 覆盖 c0 ∈ [2,11]，这一整段的列掩码都是 0，取任一代表即可。
constexpr int colmask5_repr_c0(int off) {
    return off < 0 ? off + 2 : (off == 0 ? 2 : off + 11);
}

// 8 个 off × 8 字节 = 64 字节。每项前 5 字节按窗口行 k 排布：
// k=0,1,3,4 用 COLMASK_Q，k=2 用 COLMASK_H。
constexpr std::array<uint8_t, 8 * 8> make_colmask5_lut() {
    std::array<uint8_t, 8 * 8> t = {};
    const auto cq = make_colmask(false);
    const auto ch = make_colmask(true);
    for (int o = 0; o < 8; ++o) {
        const int c0 = colmask5_repr_c0(o - 2);
        const uint8_t q = cq[static_cast<unsigned>(c0)];
        const uint8_t h = ch[static_cast<unsigned>(c0)];
        const unsigned b = static_cast<unsigned>(o) * 8u;
        t[b + 0] = q;
        t[b + 1] = q;
        t[b + 2] = h;
        t[b + 3] = q;
        t[b + 4] = q;
    }
    return t;
}

// 编译期证明：对每个合法 c0，COL5 表给出的五个窗口行掩码与原来的
// COLMASK_Q/COLMASK_H 完全一致。它自己重算 cb 与 off，所以不依赖热路径。
constexpr bool colmask5_is_equivalent() {
    const auto cq = make_colmask(false);
    const auto ch = make_colmask(true);
    const auto t5 = make_colmask5_lut();
    for (int c0 = 0; c0 < GRID_SIZE; ++c0) {
        int cb = c0 - 2;
        cb = cb < 0 ? 0 : cb;
        cb = cb > GRID_SIZE - 8 ? GRID_SIZE - 8 : cb;
        const unsigned o = static_cast<unsigned>((c0 - 2) - cb + 2);
        for (int k = 0; k < 5; ++k) {
            const uint8_t want = (k == 2) ? ch[static_cast<unsigned>(c0)]
                                          : cq[static_cast<unsigned>(c0)];
            if (t5[o * 8u + static_cast<unsigned>(k)] != want) return false;
        }
    }
    return true;
}
static_assert(colmask5_is_equivalent(),
              "COL5 表必须与 COLMASK_Q/COLMASK_H 逐项等价");
#endif

// 行越界掩码：窗口行 k 对应网格行 r0-2+k，整行越界则 8 个 lane 全擦。
constexpr std::array<uint8_t, GRID_SIZE * 5> make_rowmask() {
    std::array<uint8_t, GRID_SIZE * 5> table = {};
    for (int r0 = 0; r0 < GRID_SIZE; ++r0) {
        for (int k = 0; k < 5; ++k) {
            const int row = r0 - 2 + k;
            table[static_cast<unsigned>(r0 * 5 + k)] =
                (row < 0 || row >= GRID_SIZE) ? 0xFF : 0x00;
        }
    }
    return table;
}
constexpr auto ROWMASK_RAW = make_rowmask();

#if V44_ROWOFF == 1
// 预钳位的**绝对**行偏移，单位是 int32 元素：ROWOFF16[r*5+k] = 17*clamp(r-2+k,0,16)。
// 于是窗口行 k 的地址就是 `&grid[0][cb] + 4*ROWOFF16[r*5+k]`，
// 一条 `movzwl` 加一条折进 `(base,idx,4)` 的 load，clamp 与 68 倍乘全部消失。
constexpr std::array<uint16_t, GRID_SIZE * 5> make_rowoff16() {
    std::array<uint16_t, GRID_SIZE * 5> t = {};
    for (int r = 0; r < GRID_SIZE; ++r) {
        for (int k = 0; k < 5; ++k) {
            int rr = r - 2 + k;
            rr = rr < 0 ? 0 : rr;
            rr = rr > GRID_SIZE - 1 ? GRID_SIZE - 1 : rr;
            t[static_cast<unsigned>(r * 5 + k)] =
                static_cast<uint16_t>(GRID_SIZE * rr);
        }
    }
    return t;
}
constexpr auto ROWOFF16 = make_rowoff16();
static_assert(GRID_SIZE * (GRID_SIZE - 1) <= 0xFFFF, "绝对行偏移必须装进 uint16");
#elif V44_ROWOFF == 2
// **相对** r 的行偏移，取值 17*(-2..2)+34 = {0,17,34,51,68}，装得进 uint8。
// 地址 = &grid[r][cb] + 4*ROWOFF8[r*5+k] - 4*2*17，后面那个常量折进 load 的位移，
// 所以指令数与 =1 相同、只多一次每角色的 r*68；换来 .rodata 只要一半。
constexpr std::array<uint8_t, GRID_SIZE * 5> make_rowoff8() {
    std::array<uint8_t, GRID_SIZE * 5> t = {};
    for (int r = 0; r < GRID_SIZE; ++r) {
        for (int k = 0; k < 5; ++k) {
            int rr = r - 2 + k;
            rr = rr < 0 ? 0 : rr;
            rr = rr > GRID_SIZE - 1 ? GRID_SIZE - 1 : rr;
            t[static_cast<unsigned>(r * 5 + k)] =
                static_cast<uint8_t>(GRID_SIZE * (rr - r) + 2 * GRID_SIZE);
        }
    }
    return t;
}
constexpr auto ROWOFF8 = make_rowoff8();
static_assert(4 * GRID_SIZE <= 0xFF, "相对行偏移必须装进 uint8");
#endif

#if V7_PACK_TABLES
// 单次调用必然要碰的小表打包成一块 64 字节对齐的连续内存。
// 合计 85+17+17+64+40+32 = 255 字节 = 4 条连续 cache line，
// 硬件预取器可以顺序取回；散放时这些访问会落在 4~5 条互不相邻的 line 上。
struct alignas(64) HotTables {
#if V30_VBMI_BLOCKERS
    // One aligned ZMM load feeds VPERMB.  This is the locked V25 table order;
    // total HotTables size remains 255 bytes / four cache lines.
    uint8_t lanebits[8 * 8];             // 64
    uint8_t rowmask[GRID_SIZE * 5 + (V35_PAIR_MASKS ? 3 : 0)];
#if V46_COL5
    uint8_t colmask5[8 * 8];             // 64，取代 colmask_q+colmask_h(34)
#else
    uint8_t colmask_q[GRID_SIZE];        // 17
    uint8_t colmask_h[GRID_SIZE];        // 17
#endif
#else
    uint8_t rowmask[GRID_SIZE * 5 + (V35_PAIR_MASKS ? 3 : 0)];
    uint8_t colmask_q[GRID_SIZE];        // 17
    uint8_t colmask_h[GRID_SIZE];        // 17
    uint8_t lanebits[8 * 8];             // 64
#endif
    uint16_t tail[20];                   // 40
#if !V33_WINNER_HLUT
    uint16_t spread[16];                 // 32
#endif
};

constexpr HotTables make_hot_tables() {
    HotTables t = {};
    const auto rm = make_rowmask();
    const auto cq = make_colmask(false);
    const auto ch = make_colmask(true);
    const auto lb = make_lanebits_lut();
    const auto tl = make_tail_lut();
#if !V33_WINNER_HLUT
    const auto sp = make_spread_parent_lut();
#endif
    for (unsigned i = 0; i < rm.size(); ++i) t.rowmask[i] = rm[i];
#if V46_COL5 && V30_VBMI_BLOCKERS
    (void)cq;
    (void)ch;
    const auto c5 = make_colmask5_lut();
    for (unsigned i = 0; i < c5.size(); ++i) t.colmask5[i] = c5[i];
#else
    for (unsigned i = 0; i < cq.size(); ++i) t.colmask_q[i] = cq[i];
    for (unsigned i = 0; i < ch.size(); ++i) t.colmask_h[i] = ch[i];
#endif
    for (unsigned i = 0; i < lb.size(); ++i) t.lanebits[i] = lb[i];
    for (unsigned i = 0; i < tl.size(); ++i) t.tail[i] = tl[i];
#if !V33_WINNER_HLUT
    for (unsigned i = 0; i < sp.size(); ++i) t.spread[i] = sp[i];
#endif
    return t;
}

constexpr HotTables HOT = make_hot_tables();
static_assert(sizeof(HotTables) <= 256, "热表应当压在 4 条 cache line 内");

#define V7_ROWMASK   HOT.rowmask
#if V46_COL5 && V30_VBMI_BLOCKERS
// 热路径只用 COL5；冷/未实例化路径继续引用原始表（未被引用则不进 .rodata）。
#define V7_COLMASK5  HOT.colmask5
#define V7_COLMASK_Q COLMASK_Q
#define V7_COLMASK_H COLMASK_H
#else
#define V7_COLMASK_Q HOT.colmask_q
#define V7_COLMASK_H HOT.colmask_h
#endif
#define V7_LANEBITS  HOT.lanebits
#define V7_TAIL_LUT  HOT.tail
#if !V33_WINNER_HLUT
#define V7_SPREAD    HOT.spread
#endif
#else
#define V7_ROWMASK   ROWMASK_RAW
#define V7_COLMASK_Q COLMASK_Q
#define V7_COLMASK_H COLMASK_H
#define V7_LANEBITS  LANEBITS
#define V7_TAIL_LUT  TAIL_LUT
#if !V33_WINNER_HLUT
#define V7_SPREAD    SPREAD_PARENT_LUT
#endif
#endif

// ---------------------------------------------------------------- 向量小工具

inline __m256i combine_lanes(__m128i low, __m128i high) {
    return _mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 1);
}

inline __m256i select_max_tie_left_256(__m256i left, __m256i right,
                                       __m256i* right_greater) {
    const __m256i take_right = _mm256_cmpgt_epi32(right, left);
    *right_greater = take_right;
    return _mm256_blendv_epi8(left, right, take_right);
}

inline __m256i valid_key_vector_256(__m256i score, __m256i metadata) {
    const __m256i valid = _mm256_cmpgt_epi32(score, _mm256_set1_epi32(VALID_FLOOR));
#if V7_TRAMPLE == 4
    // clean 字段靠 DP 的 add 累加了 -踩踏格数，这里统一 +CLEAN_MAX 把它抬成
    // 非负的 (4 - 踩踏格数)，从而不会向金币字段借位。合法性判据用的是加偏置
    // 之前的 score，所以偏置不影响阻挡判定。
    const __m256i biased = _mm256_add_epi32(score, _mm256_set1_epi32(CLEAN_BIAS));
    return _mm256_blendv_epi8(_mm256_set1_epi32(KEY_INVALID),
                              _mm256_or_si256(biased, metadata), valid);
#else
    return _mm256_blendv_epi8(_mm256_set1_epi32(KEY_INVALID),
                              _mm256_or_si256(score, metadata), valid);
#endif
}

#if V47_KEYFUSE
// A valid path contains no NEG cell, hence score >= 0; its metadata always has
// dist >= 1, hence the completed key is strictly positive.  Invalid lanes can
// therefore become the existing STAY fallback (0) before the running max.
static_assert(VALID_FLOOR < 0, "V47_KEYFUSE needs a negative validity floor");
static_assert((1 << DIST_SHIFT) > 0, "V47_KEYFUSE needs a positive dist unit");
static_assert(int64_t(NEG) + (MAX_PATH_GOLD << GOLD_SHIFT) < int64_t(VALID_FLOOR),
              "V47_KEYFUSE reuses the hard-exclusion invariant to prove score >= 0");
static_assert(KEY_INVALID < 0, "the sentinel being replaced must have been negative");

__attribute__((always_inline)) inline
__m256i valid_key_zero_256(__m256i score, __m256i qbase, __m256i metadata) {
    const __mmask8 valid =
        _mm256_cmpgt_epi32_mask(score, _mm256_set1_epi32(VALID_FLOOR));
    return _mm256_maskz_ternarylogic_epi32(
        valid, score, qbase, metadata, 0xFE);
}
#endif

#if V47_KMASK
__attribute__((always_inline)) inline
__m256i select_max_tie_left_k256(__m256i left, __m256i right,
                                  __mmask8* right_greater) {
    *right_greater = _mm256_cmpgt_epi32_mask(right, left);
    return _mm256_max_epi32(left, right);
}
#endif

inline __m256i horizontal_max_halves(__m256i value) {
    value = _mm256_max_epi32(value, _mm256_shuffle_epi32(value, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm256_max_epi32(value, _mm256_shuffle_epi32(value, _MM_SHUFFLE(1, 0, 3, 2)));
}

// 按 8 位 lane 掩码把若干 lane 换成 replacement。
inline __m256i lane_replace(__m256i value, unsigned mask8, __m256i replacement) {
#if V7_AVX512
    return _mm256_mask_mov_epi32(value, static_cast<__mmask8>(mask8), replacement);
#else
    const __m256i bit = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    const __m256i sel = _mm256_and_si256(_mm256_set1_epi32(static_cast<int>(mask8)), bit);
    return _mm256_blendv_epi8(value, replacement, _mm256_cmpeq_epi32(sel, bit));
#endif
}

// 按 8 位 lane 掩码给若干 lane 减去 penalty。
inline __m256i lane_subtract(__m256i value, unsigned mask8, __m256i penalty) {
#if V7_AVX512
    return _mm256_mask_sub_epi32(value, static_cast<__mmask8>(mask8), value, penalty);
#else
    const __m256i bit = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    const __m256i sel = _mm256_and_si256(_mm256_set1_epi32(static_cast<int>(mask8)), bit);
    const __m256i live = _mm256_cmpeq_epi32(sel, bit);
    return _mm256_sub_epi32(value, _mm256_and_si256(penalty, live));
#endif
}

// ---------------------------------------------------------------- 占位 / 踩踏位图
//
// 把「哪些 lane 要擦 / 要扣」压成一个 uint64：每个窗口行占 8 位。
// 全程不碰内存，也没有数据相关分支。

inline uint64_t lane_bit_of(int dr, int dc) {
    const unsigned wr = static_cast<unsigned>(dr + 2);
    const unsigned wc = static_cast<unsigned>(dc + 2);
    const unsigned ok = static_cast<unsigned>(wr < 5u) & static_cast<unsigned>(wc < 5u);
    const unsigned index = ((wr & 7u) << 3) | (wc & 7u);
    const uint64_t bits = static_cast<uint64_t>(V7_LANEBITS[index]) * ok;
    return bits << ((wr & 7u) * 8u);
}

// 三个占位者：队友 + 两个可见敌方角色（与 V6 完全一致）
inline uint64_t collect_blockers(const Position& self, const Position& mate,
                                 const Position& enemy0, const Position& enemy1) {
    const Position occ[3] = {mate, enemy0, enemy1};
    uint64_t bits = 0;
    for (int t = 0; t < 3; ++t) {
        const uint64_t valid = static_cast<uint64_t>(
            static_cast<unsigned>(occ[t].row >= 0) & static_cast<unsigned>(occ[t].col >= 0));
        bits |= lane_bit_of(occ[t].row - self.row, occ[t].col - self.col) * valid;
    }
    return bits;
}

#if V30_VBMI_BLOCKERS
struct BlockerPair {
    uint64_t unit0;
    uint64_t unit1;
};

// Verbatim data flow from the locked V25/fcd5 candidate.  Each qword contains
// one Position as [col:int32 | row:int32].  Lanes 0..2 project
// mate/enemy0/enemy1 around unit0; lanes 4..6 do the same around unit1.
// Lanes 3 and 7 are padding and are forcibly masked off.
__attribute__((always_inline)) inline BlockerPair collect_blockers_pair(
        const Position& self0, const Position& self1,
        const Position& enemy0, const Position& enemy1) {
    const auto pack = [](const Position& p) -> uint64_t {
        return static_cast<uint64_t>(static_cast<uint32_t>(p.row)) |
               (static_cast<uint64_t>(static_cast<uint32_t>(p.col)) << 32);
    };

    const uint64_t p0 = pack(self0);
    const uint64_t p1 = pack(self1);
    const uint64_t e0 = pack(enemy0);
    const uint64_t e1 = pack(enemy1);

    const __m512i occ = _mm512_setr_epi64(
        static_cast<long long>(p1), static_cast<long long>(e0),
        static_cast<long long>(e1), 0,
        static_cast<long long>(p0), static_cast<long long>(e0),
        static_cast<long long>(e1), 0);
    const __m512i self = _mm512_setr_epi64(
        static_cast<long long>(p0), static_cast<long long>(p0),
        static_cast<long long>(p0), static_cast<long long>(p0),
        static_cast<long long>(p1), static_cast<long long>(p1),
        static_cast<long long>(p1), static_cast<long long>(p1));

    // Dword layout per qword is [wr, wc].  Unsigned <5 reproduces lane_bit_of's
    // relative-coordinate range check.  The absolute >=0 check is separate so
    // invalid (-1,-1) near self=(0,0) cannot turn into a valid relative lane.
    const __m512i rel = _mm512_add_epi32(
        _mm512_sub_epi32(occ, self), _mm512_set1_epi32(2));
#if NOREV_BLOCKER_PAIRMAX_VALID
    // For each coordinate, the original predicate is
    //
    //     occ >= 0 && unsigned(rel) < 5.
    //
    // OR-ing rel with occ's arithmetic sign mask maps every negative absolute
    // coordinate to UINT32_MAX and leaves every non-negative one unchanged.
    // Swapping adjacent dwords and taking unsigned max then duplicates
    // max(row, col) into both halves of each qword.  Because
    //
    //     m | (uint64_t(m) << 32) < 5 | (uint64_t(5) << 32)  iff  m < 5,
    //
    // one unsigned qword compare produces exactly the old pairwise-AND mask.
    // Lanes 3 and 7 are padding and remain suppressed by the input writemask.
    const __m512i checked = _mm512_or_si512(
        rel, _mm512_srai_epi32(occ, 31));
    const __m512i pair_max = _mm512_max_epu32(
        checked,
        _mm512_shuffle_epi32(checked, _MM_SHUFFLE(2, 3, 0, 1)));
    const __mmask8 valid64 = _mm512_mask_cmplt_epu64_mask(
        static_cast<__mmask8>(0x77u), pair_max,
        _mm512_set1_epi64(static_cast<long long>(0x0000000500000005ULL)));
#else
    __mmask16 valid32 = _mm512_cmplt_epu32_mask(rel, _mm512_set1_epi32(5));
    valid32 &= _mm512_cmpge_epi32_mask(occ, _mm512_setzero_si512());
    const uint32_t adjacent = static_cast<uint32_t>(valid32) &
                              (static_cast<uint32_t>(valid32) >> 1);
    const __mmask8 valid64 = static_cast<__mmask8>(
        _pext_u32(adjacent, 0x5555u) & 0x77u);
#endif

    // idx = ((wr & 7) << 3) | (wc & 7).  VPERMB performs all six lookups
    // against the same aligned 64-byte table.
    const __m512i wr8 = _mm512_and_si512(
        _mm512_slli_epi64(rel, 3), _mm512_set1_epi64(0x38));
    const __m512i wc = _mm512_and_si512(
        _mm512_srli_epi64(rel, 32), _mm512_set1_epi64(7));
    const __m512i index = _mm512_or_si512(wr8, wc);
    const __m512i table = _mm512_load_si512(
        reinterpret_cast<const void*>(V7_LANEBITS));
    const __m512i looked = _mm512_permutexvar_epi8(index, table);

    __m512i bits = _mm512_and_si512(looked, _mm512_set1_epi64(0xFF));
    bits = _mm512_sllv_epi64(bits, wr8);
    bits = _mm512_maskz_mov_epi64(valid64, bits);

    const __m512i plus1 = _mm512_permutexvar_epi64(
        _mm512_set_epi64(0, 0, 0, 5, 0, 0, 0, 1), bits);
    const __m512i plus2 = _mm512_permutexvar_epi64(
        _mm512_set_epi64(0, 0, 0, 6, 0, 0, 0, 2), bits);
    const __m512i merged = _mm512_ternarylogic_epi64(bits, plus1, plus2, 0xFE);
    return {
        static_cast<uint64_t>(_mm_cvtsi128_si64(_mm512_castsi512_si128(merged))),
        static_cast<uint64_t>(_mm_cvtsi128_si64(
            _mm512_extracti64x2_epi64(merged, 2)))
    };
}
#endif

#if V7_TRAMPLE
// NPC 踩踏：只关心「同一格 >= 3 个 NPC」的格子，<= 2 的完全忽略。
//
// 先在绝对坐标里一次性判出哪些 NPC 处在一个 >=3 的堆里（7x7 定长比较，
// 无内存、无数据相关分支），绝大多数回合结果为空、直接整段跳过。
// 只有非空时才把那几个格子映射进两个角色的窗口 lane 位图。
// 这比「对每个角色都把 7 个 NPC 映射一遍」便宜约 3 倍。
struct TrampleSet {
    uint32_t slots;                 // 哪些 NPC 槽位属于 >=3 的堆
    int32_t id[MAX_NPCS];           // row*GRID_SIZE + col，无效槽为互不相同的负数
};

inline TrampleSet find_npc_stacks(const NpcInfo* npcs) {
    TrampleSet out;
    for (int i = 0; i < MAX_NPCS; ++i) {
        const Position p = npcs[i].pos;
        const int valid = static_cast<int>(
            static_cast<unsigned>(p.row >= 0) & static_cast<unsigned>(p.col >= 0));
        // 无效槽用互不相同的负数，保证永远匹配不上任何格子（包括彼此）
        out.id[i] = valid ? (p.row * GRID_SIZE + p.col) : (-1 - i);
    }
    uint32_t slots = 0;
    for (int i = 0; i < MAX_NPCS; ++i) {
        int count = 0;
        for (int j = 0; j < MAX_NPCS; ++j) {
            count += static_cast<int>(out.id[j] == out.id[i]);
        }
        slots |= static_cast<uint32_t>(count >= 3) << i;
    }
    out.slots = slots;
    return out;
}

__attribute__((noinline))
uint64_t trample_bits_for(const TrampleSet& set, const Position& self) {
    uint64_t bits = 0;
    uint32_t remaining = set.slots;
    while (remaining != 0) {
        const int i = __builtin_ctz(remaining);
        remaining &= remaining - 1;
        const int id = set.id[i];
        bits |= lane_bit_of(id / GRID_SIZE - self.row, id % GRID_SIZE - self.col);
    }
    return bits;
}
#endif

// ---------------------------------------------------------------- 窗口 -> DP 输入

struct UnitWindow {
    __m128i c11, c12, c21, c22;
    __m128i d10, d20, d01, d02;
    __m128i axis1, axis2;
};

#if V26_ZMM_PAIR_WINDOW
// Every lane is already in solve_pair order:
//   lanes 0..3 = unit 0 quadrants, lanes 4..7 = unit 1 quadrants.
// Keeping this representation across the window/DP boundary avoids the two
// UnitWindow objects and the ten inserti128 operations formerly needed by
// solve_pair().  The functions using it are always_inline so this aggregate
// is expected to be scalar-replaced into vector registers, not materialised.
struct PairWindow {
    __m256i c11, c12, c21, c22;
    __m256i d10, d20, d01, d02;
    __m256i axis1, axis2;
};
#endif

#if V7_OPENNESS
// 象限开阔度：该象限的 2x2 对角块里有多少格不是阻挡（0..4）。
//
// 这里占了一个便宜：c11/c12/c21/c22 已经是**按象限归一化过**的对角 2x2 块，
// 每个 __m128i 的 4 个 lane 正好就是 4 个象限 —— 和 qbase 的 lane 布局完全
// 对齐。所以整件事就是 4 次比较加 4 次累加，不用碰窗口构造，也不用重排代码。
//
// 为什么用 2x2 对角块而不是整个象限的 8 格：轴向臂（d10/d20/d01/d02）被
// 相邻两个象限共享，把它们算进来会让相邻象限的开阔度互相污染。2x2 对角块是
// 这个方向**独占的腹地**，「走过去之后还有没有路」正是它回答的问题。
//
// 阻挡的格子分数是 NEG，远低于 VALID_FLOOR；空地是 0，金币是正数。
// 所以 cmpgt(cell, VALID_FLOOR) 就是「非阻挡」，返回 -1，累加后取负即计数。
__attribute__((always_inline)) inline __m128i quadrant_openness(const UnitWindow& w) {
    const __m128i floor_v = _mm_set1_epi32(VALID_FLOOR);
    __m128i acc = _mm_cmpgt_epi32(w.c11, floor_v);
    acc = _mm_add_epi32(acc, _mm_cmpgt_epi32(w.c12, floor_v));
    acc = _mm_add_epi32(acc, _mm_cmpgt_epi32(w.c21, floor_v));
    acc = _mm_add_epi32(acc, _mm_cmpgt_epi32(w.c22, floor_v));
    return _mm_sub_epi32(_mm_setzero_si128(), acc);   // 0..OPEN_MAX
}
#endif

// cell[25]（行主序）-> DP 需要的向量布局。标量路径共用。
inline void finish_from_cells(const int32_t* cell, UnitWindow* out) {
    const int32_t up1 = cell[7], up2 = cell[2];
    const int32_t dn1 = cell[17], dn2 = cell[22];
    const int32_t lf1 = cell[11], lf2 = cell[10];
    const int32_t rt1 = cell[13], rt2 = cell[14];
    out->c11 = _mm_set_epi32(cell[18], cell[8], cell[16], cell[6]);
    out->c12 = _mm_set_epi32(cell[19], cell[9], cell[15], cell[5]);
    out->c21 = _mm_set_epi32(cell[23], cell[3], cell[21], cell[1]);
    out->c22 = _mm_set_epi32(cell[24], cell[4], cell[20], cell[0]);
    out->d10 = _mm_set_epi32(dn1, up1, dn1, up1);
    out->d20 = _mm_set_epi32(dn1 + dn2, up1 + up2, dn1 + dn2, up1 + up2);
    out->d01 = _mm_set_epi32(rt1, rt1, lf1, lf1);
    out->d02 = _mm_set_epi32(rt1 + rt2, rt1 + rt2, lf1 + lf2, lf1 + lf2);
    out->axis1 = _mm_set_epi32(dn1, rt1, lf1, up1);
    out->axis2 = _mm_set_epi32(dn1 + dn2, rt1 + rt2, lf1 + lf2, up1 + up2);
}

// 把 lane 位图翻译回行主序，写入阻挡与踩踏罚分。
inline void apply_lane_bits(int32_t* cell, uint64_t block_bits,
                            uint64_t trample_bits, int32_t penalty) {
    for (int wr = 0; wr < 5; ++wr) {
        const unsigned bm = static_cast<unsigned>((block_bits >> (wr * 8)) & 0xFFu);
        const unsigned tm = static_cast<unsigned>((trample_bits >> (wr * 8)) & 0xFFu);
        if ((bm | tm) == 0) continue;
        for (int wc = 0; wc < 5; ++wc) {
            const uint8_t lb = (wr == 2) ? LANEBITS_H[wc] : LANEBITS_Q[wc];
            if (lb == 0) continue;
            if (bm & lb) cell[wr * 5 + wc] = NEG;
            else if (tm & lb) cell[wr * 5 + wc] -= penalty;
        }
    }
}

// 完全通用的标量回退：角色位置越界，或金币超出魔数精确范围时使用（极冷）。
__attribute__((noinline, cold))
void build_unit_window_general(const int grid[GRID_SIZE][GRID_SIZE],
                               const Position& pos,
                               uint64_t block_bits,
                               uint64_t trample_bits,
                               int32_t penalty,
                               UnitWindow* out) {
    int32_t cell[25];
    const int r0 = pos.row;
    const int c0 = pos.col;
    for (int wr = 0; wr < 5; ++wr) {
        for (int wc = 0; wc < 5; ++wc) {
            const int r = r0 - 2 + wr;
            const int c = c0 - 2 + wc;
            cell[wr * 5 + wc] =
                (static_cast<unsigned>(r) >= GRID_SIZE || static_cast<unsigned>(c) >= GRID_SIZE)
                    ? NEG
                    : classify_cell(grid[r][c]);
        }
    }
    apply_lane_bits(cell, block_bits, trample_bits, penalty);
    finish_from_cells(cell, out);
}

#if !V7_VEC_WINDOW
// V6 风格的标量窗口：内部走无边界判断的直筒 5x5 循环，贴边才进冷函数。
// 保留它是为了能公平对比「V6 速度的窗口 + V7 新特性」这一组合。
__attribute__((noinline, cold))
void build_unit_window_clipped(const int grid[GRID_SIZE][GRID_SIZE],
                               const Position& pos,
                               uint64_t block_bits,
                               uint64_t trample_bits,
                               int32_t penalty,
                               UnitWindow* out) {
    int32_t cell[25];
    for (int i = 0; i < 25; ++i) cell[i] = NEG;
    const int r0 = pos.row;
    const int c0 = pos.col;
    const int wr0 = r0 < 2 ? 2 - r0 : 0;
    const int wr1 = r0 > GRID_SIZE - 3 ? GRID_SIZE + 2 - r0 : 5;
    const int wc0 = c0 < 2 ? 2 - c0 : 0;
    const int wc1 = c0 > GRID_SIZE - 3 ? GRID_SIZE + 2 - c0 : 5;
    for (int wr = wr0; wr < wr1; ++wr) {
        const int* src = &grid[r0 - 2 + wr][c0 - 2 + wc0];
        int32_t* dst = cell + wr * 5 + wc0;
        for (int wc = wc0; wc < wc1; ++wc) *dst++ = classify_cell(*src++);
    }
    apply_lane_bits(cell, block_bits, trample_bits, penalty);
    finish_from_cells(cell, out);
}

inline void build_unit_window_scalar_fast(const int grid[GRID_SIZE][GRID_SIZE],
                                          const Position& pos,
                                          uint64_t block_bits,
                                          uint64_t trample_bits,
                                          int32_t penalty,
                                          UnitWindow* out) {
    const int r0 = pos.row;
    const int c0 = pos.col;
    if (__builtin_expect(static_cast<unsigned>(r0) >= GRID_SIZE ||
                         static_cast<unsigned>(c0) >= GRID_SIZE, 0)) {
        build_unit_window_general(grid, pos, block_bits, trample_bits, penalty, out);
        return;
    }
    if (r0 >= 2 && r0 <= GRID_SIZE - 3 && c0 >= 2 && c0 <= GRID_SIZE - 3) {
        int32_t cell[25];
        for (int wr = 0; wr < 5; ++wr) {
            const int* src = &grid[r0 - 2 + wr][c0 - 2];
            int32_t* dst = cell + wr * 5;
            for (int wc = 0; wc < 5; ++wc) *dst++ = classify_cell(*src++);
        }
        apply_lane_bits(cell, block_bits, trample_bits, penalty);
        finish_from_cells(cell, out);
    } else {
        build_unit_window_clipped(grid, pos, block_bits, trample_bits, penalty, out);
    }
}
#endif  // !V7_VEC_WINDOW

#if V7_VEC_WINDOW
// 一行的分类：blocked -> NEG，其余 -> ceil(0.65*g)<<11。全程寄存器内。
//
// 必须逐位复刻 V6 的 CELL_LUT 语义，包括这几个容易漏的分支：
//   * 阻挡只有 -1 墙 / -5 雾 / -3 炸弹（相位后）。这三个值都是奇数，
//     而 -2/-4 在 V6 里既不阻挡也无收益（走 pick65 分支返回 0），
//     所以用「负 且 奇」判阻挡，再把炸弹按相位挖掉。
//   * 所有 raw < 0 且不阻挡的格子，V6 都给 0（不是负分）。魔数公式对负数
//     会产出负值，必须用 andnot(lt0) 强制归零 —— 相位前的炸弹就靠这一步。
//   * raw <= -6 超出规则定义的取值集，V6 的 LUT 会落到 pick65 分支返回 0；
//     先把 raw 钳到 >= -6（偶数、不阻挡、随后归零）即可对齐。
// 用移位代替比较常量，少占两个向量寄存器：
//   srai(raw,31)            == cmpgt(0, raw)      （raw<0 的全 1 掩码）
//   srai(slli(raw,31),31)   == 低位广播成全 1     （奇数掩码）
inline __m256i classify_row_vec(__m256i raw_in, __m256i bombpat
#if !V34_TRUST_GOLD_RANGE
                                , __m256i* raw_max
#endif
                                ) {
    const __m256i raw = _mm256_max_epi32(raw_in, _mm256_set1_epi32(-6));
#if !V34_TRUST_GOLD_RANGE
    *raw_max = _mm256_max_epi32(*raw_max, raw);
#endif
    const __m256i lt0 = _mm256_srai_epi32(raw, 31);
    const __m256i odd = _mm256_srai_epi32(_mm256_slli_epi32(raw, 31), 31);
    const __m256i isb = _mm256_cmpeq_epi32(raw, bombpat);
    const __m256i blocked = _mm256_andnot_si256(isb, _mm256_and_si256(lt0, odd));
    __m256i score = _mm256_mullo_epi32(raw, _mm256_set1_epi32(MAGIC_MUL));
    score = _mm256_add_epi32(score, _mm256_set1_epi32(MAGIC_ADD));
    score = _mm256_srai_epi32(score, MAGIC_SHIFT);
    score = _mm256_and_si256(score, _mm256_set1_epi32(~((1 << GOLD_SHIFT) - 1)));
    score = _mm256_andnot_si256(lt0, score);        // raw<0 一律 0
    // 注意：MAGIC_SHIFT 已经把 <<GOLD_SHIFT 折进去了，所以这里的 and 掩码
    // 用的是 ~((1<<GOLD_SHIFT)-1)，正好把折叠后残留的低位清零。
    return _mm256_blendv_epi8(score, _mm256_set1_epi32(NEG), blocked);
}

// 返回 false 表示金币超出魔数精确范围，调用方应改走标量回退。
//
// WithTrample 做成模板参数而不是运行期判断：踩踏在绝大多数回合不发生，
// 而 lane_subtract 在 make_row 里每角色要调 5 次。做成运行期分支的话
// 这 5 次（两角色共 10 次）无论如何都要付；模板特化让不发生时是真正的零成本。
template <bool WithTrample>
__attribute__((always_inline)) inline
bool build_unit_window_vec(const int grid[GRID_SIZE][GRID_SIZE],
                                  const Position& pos,
                                  uint64_t block_bits,
                                 uint64_t trample_bits,
                                 int32_t penalty,
                                 UnitWindow* out) {
    const int r0 = pos.row;
    const int c0 = pos.col;
    // 列基址钳位：8 个 int 的窗口右端最远 9+7=16，正好是最后一列。
    int cb = c0 - 2;
    cb = cb < 0 ? 0 : cb;
    cb = cb > GRID_SIZE - 8 ? GRID_SIZE - 8 : cb;
    const int off = (c0 - 2) - cb;              // [-2, 5]

    const __m256i bombpat = _mm256_set1_epi32(g_bomb_pattern);
    const __m256i negv = _mm256_set1_epi32(NEG);
    const uint8_t* rowm = &V7_ROWMASK[static_cast<unsigned>(r0) * 5u];
    const unsigned colm_q = V7_COLMASK_Q[static_cast<unsigned>(c0)];
    const unsigned colm_h = V7_COLMASK_H[static_cast<unsigned>(c0)];
    const __m256i idx_q = _mm256_load_si256(
        reinterpret_cast<const __m256i*>(&IDX_Q[static_cast<unsigned>(off + 2)]));
    const __m256i idx_h = _mm256_load_si256(
        reinterpret_cast<const __m256i*>(&IDX_H[static_cast<unsigned>(off + 2)]));
    const __m256i penv = _mm256_set1_epi32(penalty);
    if constexpr (!WithTrample) {
        (void)trample_bits;
        (void)penalty;
    }

#if !V34_TRUST_GOLD_RANGE
    __m256i raw_max = _mm256_set1_epi32(-6);
#endif

    // 一行搞定：钳位宽读 -> 分类 -> vpermd 归一化 -> 踩踏扣分 -> 掩码擦除。
    // 刻意按 (1,3) (0,4) (2) 的顺序取用并立刻消化，让同时存活的行向量不超过 2 个，
    // 避免 16 个 ymm 寄存器不够而溢出到栈 —— 那正是要消掉的内存往返。
    const auto make_row = [&](int k) __attribute__((always_inline)) -> __m256i {
        int rr = r0 - 2 + k;
        rr = rr < 0 ? 0 : rr;                   // 钳位读，重复行随后被掩码擦掉
        rr = rr > GRID_SIZE - 1 ? GRID_SIZE - 1 : rr;
        const __m256i raw =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&grid[rr][cb]));
        __m256i v = classify_row_vec(raw, bombpat
#if !V34_TRUST_GOLD_RANGE
                                     , &raw_max
#endif
                                     );
        v = _mm256_permutevar8x32_epi32(v, k == 2 ? idx_h : idx_q);
        const unsigned mask = rowm[k] | (k == 2 ? colm_h : colm_q) |
            static_cast<unsigned>((block_bits >> (k * 8)) & 0xFFu);
        if constexpr (WithTrample) {
            // 阻挡 lane 随后会被 lane_replace 覆盖，所以这里不必再排除
            v = lane_subtract(v, static_cast<unsigned>((trample_bits >> (k * 8)) & 0xFFu), penv);
        }
        return lane_replace(v, mask, negv);
    };

    // rows{1,3} -> c11 / c12 / d10
    {
        const __m256i r1 = make_row(1);
        const __m256i r3 = make_row(3);
        const __m128i a = _mm256_castsi256_si128(r1);
        const __m128i b = _mm256_castsi256_si128(r3);
        const __m128i lo = _mm_unpacklo_epi32(a, b);   // [a0,b0,a1,b1]
        const __m128i hi = _mm_unpackhi_epi32(a, b);   // [a2,b2,a3,b3]
        out->c11 = _mm_alignr_epi8(hi, lo, 8);         // [a1,b1,a2,b2]
        out->c12 = _mm_blend_epi32(lo, hi, 0xC);       // [a0,b0,a3,b3]
        // 上半 128 位四个 lane 全是该行的中心列
        out->d10 = _mm_unpacklo_epi32(_mm256_extracti128_si256(r1, 1),
                                      _mm256_extracti128_si256(r3, 1));
    }
    // rows{0,4} -> c21 / c22 / d20
    {
        const __m256i r0v = make_row(0);
        const __m256i r4 = make_row(4);
        const __m128i a = _mm256_castsi256_si128(r0v);
        const __m128i b = _mm256_castsi256_si128(r4);
        const __m128i lo = _mm_unpacklo_epi32(a, b);
        const __m128i hi = _mm_unpackhi_epi32(a, b);
        out->c21 = _mm_alignr_epi8(hi, lo, 8);
        out->c22 = _mm_blend_epi32(lo, hi, 0xC);
        out->d20 = _mm_add_epi32(out->d10,
            _mm_unpacklo_epi32(_mm256_extracti128_si256(r0v, 1),
                               _mm256_extracti128_si256(r4, 1)));
    }
    // row 2 -> d01 / d02（该行 permute 后低 128 位是 [lf1, rt1, lf2, rt2]）
    {
        const __m128i h = _mm256_castsi256_si128(make_row(2));
        out->d01 = _mm_shuffle_epi32(h, _MM_SHUFFLE(1, 1, 0, 0));     // [l1,l1,r1,r1]
        out->d02 = _mm_add_epi32(out->d01,
                                 _mm_shuffle_epi32(h, _MM_SHUFFLE(3, 3, 2, 2)));
    }
    out->axis1 = _mm_blend_epi32(out->d10, out->d01, 0x6);            // [u1,l1,r1,d1]
    out->axis2 = _mm_blend_epi32(out->d20, out->d02, 0x6);

    // 金币超出魔数精确范围时让调用方改走标量精确路径（极冷）
#if V34_TRUST_GOLD_RANGE
    return true;
#else
    const __m256i over = _mm256_cmpgt_epi32(raw_max, _mm256_set1_epi32(GOLD_EXACT_MAX));
    return _mm256_testz_si256(over, over) != 0;
#endif
}

#if V26_ZMM_PAIR_WINDOW
// Classify the matching row for both units in one 16-lane operation.  The low
// eight lanes belong to unit 0 and the high eight lanes to unit 1.  Unlike a
// naive pair wrapper this halves the classification chain itself; it does not
// merely interleave two scalar loops.
#if V44_MASKVEC
// B: the whole mask chain collapses to one AND plus one compare.
//
// `raw` is clamped to >= -6, so the only negative cell values reaching this point are
// -1..-6.  V40 computes `blocked = negative & odd & ~is_bomb`; enumerating the six
// possible negatives against the two runtime bomb patterns:
//
//   raw   bits(1,0)  V40 blocked, bombpat=-3   V40 blocked, bombpat=0
//   -1      1 1        yes                       yes
//   -2      1 0        no  (even)                no  (even)
//   -3      0 1        no  (is_bomb)             yes
//   -4      0 0        no  (even)                no  (even)
//   -5      1 1        yes                       yes
//   -6      1 0        no  (even)                no  (even)
//
// so bombpat=-3 selects exactly {sign & bit1 & bit0} and bombpat=0 exactly
// {sign & bit0}.  Both are `(raw & D) == D` with D = 0x80000003 / 0x80000001,
// i.e. g_block_bits.  Two instructions, one k write, no kandd/knotw at all.
//
// The `~negative` maskz on the score also disappears: feeding max(raw,0) into the
// magic multiply gives (0*MAGIC_MUL + MAGIC_ADD) >> MAGIC_SHIFT = 1945 for every
// negative cell, and 1945 & ~((1<<GOLD_SHIFT)-1) == 0 — the same value V40's
// zero-masking produced.
static_assert(GOLD_SHIFT == 11 && MAGIC_SHIFT == 8,
              "V44_MASKVEC relies on (MAGIC_ADD >> MAGIC_SHIFT) & GOLD_MASK == 0");
static_assert(((MAGIC_ADD >> MAGIC_SHIFT) & ~((1 << GOLD_SHIFT) - 1)) == 0,
              "raw<0 must still score exactly 0 without the maskz");

__attribute__((always_inline)) inline
__m512i classify_row_pair_vec(__m512i raw_in, __m512i blockbits) {
#if NOREV_TRUST_CELL_DOMAIN
    const __m512i raw = raw_in;
#else
    const __m512i raw = _mm512_max_epi32(raw_in, _mm512_set1_epi32(-6));
#endif
    const __mmask16 blocked = _mm512_cmpeq_epi32_mask(
        _mm512_and_si512(raw, blockbits), blockbits);

    __m512i score;
#if NOREV_RAW_GOLD
    score = _mm512_slli_epi32(
        _mm512_max_epi32(raw_in, _mm512_setzero_si512()), GOLD_SHIFT);
#else
    score = _mm512_mullo_epi32(
        _mm512_max_epi32(raw_in, _mm512_setzero_si512()),
        _mm512_set1_epi32(MAGIC_MUL));
    score = _mm512_add_epi32(score, _mm512_set1_epi32(MAGIC_ADD));
    score = _mm512_srai_epi32(score, MAGIC_SHIFT);
    score = _mm512_and_si512(score, _mm512_set1_epi32(~((1 << GOLD_SHIFT) - 1)));
#endif
    return _mm512_mask_mov_epi32(score, blocked, _mm512_set1_epi32(NEG));
}
#else
__attribute__((always_inline)) inline
__m512i classify_row_pair_vec(__m512i raw_in, __m512i bombpat
#if !V34_TRUST_GOLD_RANGE
                              , __m512i* raw_max
#endif
                              ) {
    const __m512i raw = _mm512_max_epi32(raw_in, _mm512_set1_epi32(-6));
#if !V34_TRUST_GOLD_RANGE
    *raw_max = _mm512_max_epi32(*raw_max, raw);
#endif

    const __mmask16 negative = _mm512_cmp_epi32_mask(
        raw, _mm512_setzero_si512(), _MM_CMPINT_LT);
    const __mmask16 odd = _mm512_test_epi32_mask(raw, _mm512_set1_epi32(1));
    const __mmask16 is_bomb = _mm512_cmpeq_epi32_mask(raw, bombpat);
    const __mmask16 blocked = static_cast<__mmask16>(negative & odd & ~is_bomb);

    __m512i score = _mm512_mullo_epi32(raw, _mm512_set1_epi32(MAGIC_MUL));
    score = _mm512_add_epi32(score, _mm512_set1_epi32(MAGIC_ADD));
    score = _mm512_srai_epi32(score, MAGIC_SHIFT);
    score = _mm512_and_si512(score, _mm512_set1_epi32(~((1 << GOLD_SHIFT) - 1)));
    score = _mm512_maskz_mov_epi32(static_cast<__mmask16>(~negative), score);
    return _mm512_mask_mov_epi32(score, blocked, _mm512_set1_epi32(NEG));
}
#endif  // V44_MASKVEC

__attribute__((always_inline)) inline
bool build_pair_window_vec(const int grid[GRID_SIZE][GRID_SIZE],
                           const Position& pos0, uint64_t block0,
                           const Position& pos1, uint64_t block1,
                           PairWindow* out) {
    const int r0 = pos0.row;
    const int r1 = pos1.row;
    const int c0 = pos0.col;
    const int c1 = pos1.col;

#if V40_ASSUME_POSITION_RANGE
    static_assert(V34_TRUST_OFFICIAL_INPUT,
                  "V40 range assumptions require the audited official contract");
    __builtin_assume(static_cast<unsigned>(r0) < GRID_SIZE);
    __builtin_assume(static_cast<unsigned>(r1) < GRID_SIZE);
    __builtin_assume(static_cast<unsigned>(c0) < GRID_SIZE);
    __builtin_assume(static_cast<unsigned>(c1) < GRID_SIZE);
#endif

#if NOREV_COLGEOM_LUT
    const unsigned cg0 = COLGEOM_LUT[static_cast<unsigned>(c0)];
    const unsigned cg1 = COLGEOM_LUT[static_cast<unsigned>(c1)];
    const int cb0 = static_cast<int>(cg0 & 15u);
    const int cb1 = static_cast<int>(cg1 & 15u);
    // Keep the old `off + 2` consumers unchanged; Clang folds the -2/+2 pair.
    const int off0 = static_cast<int>(cg0 >> 4) - 2;
    const int off1 = static_cast<int>(cg1 >> 4) - 2;
#else
    int cb0 = c0 - 2;
    cb0 = cb0 < 0 ? 0 : cb0;
    cb0 = cb0 > GRID_SIZE - 8 ? GRID_SIZE - 8 : cb0;
    int cb1 = c1 - 2;
    cb1 = cb1 < 0 ? 0 : cb1;
    cb1 = cb1 > GRID_SIZE - 8 ? GRID_SIZE - 8 : cb1;
    const int off0 = (c0 - 2) - cb0;
    const int off1 = (c1 - 2) - cb1;
#endif

    const uint8_t* rowm0 = &V7_ROWMASK[static_cast<unsigned>(r0) * 5u];
    const uint8_t* rowm1 = &V7_ROWMASK[static_cast<unsigned>(r1) * 5u];
#if !V46_COL5
    const unsigned colq0 = V7_COLMASK_Q[static_cast<unsigned>(c0)];
    const unsigned colh0 = V7_COLMASK_H[static_cast<unsigned>(c0)];
    const unsigned colq1 = V7_COLMASK_Q[static_cast<unsigned>(c1)];
    const unsigned colh1 = V7_COLMASK_H[static_cast<unsigned>(c1)];
#endif

#if V35_PAIR_MASKS
    const __m128i row_pair = _mm_unpacklo_epi8(
        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(rowm0)),
        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(rowm1)));
    const __m128i block_pair = _mm_unpacklo_epi8(
        _mm_cvtsi64_si128(static_cast<long long>(block0)),
        _mm_cvtsi64_si128(static_cast<long long>(block1)));
#if V46_COL5
    // D: 两个角色各一条 8 字节读，直接得到 {q,q,h,q,q}；unpacklo 与行/占位掩码
    // 用同一个成对布局，于是三者一次 OR 就位，逐行的 `orl` 常量退化成 0。
    const __m128i col_pair = _mm_unpacklo_epi8(
        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(
            &V7_COLMASK5[static_cast<unsigned>(off0 + 2) * 8u])),
        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(
            &V7_COLMASK5[static_cast<unsigned>(off1 + 2) * 8u])));
    const __m128i row_block_pair = _mm_or_si128(
        _mm_or_si128(row_pair, block_pair), col_pair);
    const unsigned v46_colq = 0u;
    const unsigned v46_colh = 0u;
#if V44_MASKVEC >= 2
    // 16-bit 字 k 已经是窗口行 k 的完整擦除掩码（行 | 占位者 | 列，两个角色），
    // 与 V44_MASKVEC=2 期望的布局完全一致，所以那条路径原样可用 ——
    // 而它在 V44 轴上的构造代价（3 条）在这里已经由表读吸收掉了。
    const __mmask64 erase_lo = static_cast<__mmask64>(
        static_cast<unsigned long long>(_mm_cvtsi128_si64(row_block_pair)));
    const unsigned erase_hi =
        static_cast<unsigned>(_mm_extract_epi16(row_block_pair, 4));
#define V44_ERASE_ROW(sh) \
    static_cast<__mmask16>(_kshiftri_mask64(erase_lo, (sh)))
#endif
#else
    const unsigned colq_pair = colq0 | (colq1 << 8);
    const unsigned colh_pair = colh0 | (colh1 << 8);
    const unsigned v46_colq = colq_pair;
    const unsigned v46_colh = colh_pair;
#if V44_MASKVEC >= 2
    // B, second half: fold the two column masks into the same XMM the row/occupant
    // masks already live in, so the five per-row `orl` disappear; then move rows
    // 0..3 with ONE kmovq plus three kshiftrq instead of four extract+kmov pairs.
    const __m128i row_block_pair = _mm_or_si128(
        _mm_or_si128(row_pair, block_pair),
        _mm_insert_epi16(_mm_set1_epi16(static_cast<short>(colq_pair)),
                         static_cast<int>(colh_pair), 2));
    const __mmask64 erase_lo = static_cast<__mmask64>(
        static_cast<unsigned long long>(_mm_cvtsi128_si64(row_block_pair)));
    const unsigned erase_hi =
        static_cast<unsigned>(_mm_extract_epi16(row_block_pair, 4));
#define V44_ERASE_ROW(sh) \
    static_cast<__mmask16>(_kshiftri_mask64(erase_lo, (sh)))
#else
    const __m128i row_block_pair = _mm_or_si128(row_pair, block_pair);
#endif  // V44_MASKVEC >= 2
#endif  // V46_COL5
#endif  // V35_PAIR_MASKS

#if NOREV_RAW_INT16
    const __m512i idxw_pair = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm256_load_si256(
            reinterpret_cast<const __m256i*>(
                NOREV_RAW16_IDX[static_cast<unsigned>(off0 + 2)].q0))),
        _mm256_load_si256(reinterpret_cast<const __m256i*>(
            NOREV_RAW16_IDX[static_cast<unsigned>(off1 + 2)].q1)), 1);
    const __m256i idxw_h = _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_load_si128(reinterpret_cast<const __m128i*>(
            NOREV_RAW16_IDX[static_cast<unsigned>(off0 + 2)].h0))),
        _mm_load_si128(reinterpret_cast<const __m128i*>(
            NOREV_RAW16_IDX[static_cast<unsigned>(off1 + 2)].h1)), 1);
#else
    __m512i idx_q = _mm512_castsi256_si512(_mm256_load_si256(
        reinterpret_cast<const __m256i*>(&V46_IDXQ(static_cast<unsigned>(off0 + 2)))));
    const __m256i idx_q1 = _mm256_add_epi32(_mm256_load_si256(
        reinterpret_cast<const __m256i*>(&V46_IDXQ(static_cast<unsigned>(off1 + 2)))),
        _mm256_set1_epi32(8));
    idx_q = _mm512_inserti64x4(idx_q, idx_q1, 1);
    __m512i idx_h = _mm512_castsi256_si512(_mm256_load_si256(
        reinterpret_cast<const __m256i*>(&V46_IDXH(static_cast<unsigned>(off0 + 2)))));
    const __m256i idx_h1 = _mm256_add_epi32(_mm256_load_si256(
        reinterpret_cast<const __m256i*>(&V46_IDXH(static_cast<unsigned>(off1 + 2)))),
        _mm256_set1_epi32(8));
    idx_h = _mm512_inserti64x4(idx_h, idx_h1, 1);
#endif

#if V44_MASKVEC
    const __m512i blockbits = _mm512_set1_epi32(g_block_bits);
#else
    const __m512i bombpat = _mm512_set1_epi32(g_bomb_pattern);
#endif
    const __m512i negv = _mm512_set1_epi32(NEG);
#if !V34_TRUST_GOLD_RANGE
    __m512i raw_max = _mm512_set1_epi32(-6);
#endif

#if V44_ROWOFF == 1
    // A(=1): one base per unit, five pre-clamped absolute row offsets from .rodata.
    const char* const rowbase0 =
        reinterpret_cast<const char*>(&grid[0][cb0]);
    const char* const rowbase1 =
        reinterpret_cast<const char*>(&grid[0][cb1]);
    const uint16_t* const rowoff0 = &ROWOFF16[static_cast<unsigned>(r0) * 5u];
    const uint16_t* const rowoff1 = &ROWOFF16[static_cast<unsigned>(r1) * 5u];
#elif V44_ROWOFF == 2
    // A(=2): same idea with an 85-byte table; the -4*2*GRID_SIZE bias folds into the
    // load displacement, so the instruction count matches =1 apart from r*68.
    const char* const rowbase0 =
        reinterpret_cast<const char*>(&grid[r0][cb0]);
    const char* const rowbase1 =
        reinterpret_cast<const char*>(&grid[r1][cb1]);
    const uint8_t* const rowoff0 = &ROWOFF8[static_cast<unsigned>(r0) * 5u];
    const uint8_t* const rowoff1 = &ROWOFF8[static_cast<unsigned>(r1) * 5u];
#endif

#if NOREV_RAW_INT16
    const __m512i neg16v = _mm512_set1_epi16(NOREV_RAW16_NEG);
    const __m512i bb16 = _mm512_set1_epi16(static_cast<int16_t>(
        norev_raw16_block_bits(g_block_bits)));

    const auto load_row32 = [&](int k) __attribute__((always_inline)) -> __m512i {
        __m512i raw = _mm512_castsi256_si512(_mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(
                rowbase0 + 4 * static_cast<int>(rowoff0[k]))));
        return _mm512_inserti64x4(raw, _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(
                rowbase1 + 4 * static_cast<int>(rowoff1[k]))), 1);
    };
    const auto classify16 = [&](__m512i raw) __attribute__((always_inline)) -> __m512i {
        const __mmask32 blocked = _mm512_cmpeq_epi16_mask(
            _mm512_and_si512(raw, bb16), bb16);
        const __m512i score = _mm512_max_epi16(raw, _mm512_setzero_si512());
        return _mm512_mask_mov_epi16(score, blocked, neg16v);
    };
    const auto classify16y = [&](__m256i raw) __attribute__((always_inline)) -> __m256i {
        const __m256i bb = _mm512_castsi512_si256(bb16);
        const __mmask16 blocked = _mm256_cmpeq_epi16_mask(
            _mm256_and_si256(raw, bb), bb);
        const __m256i score = _mm256_max_epi16(raw, _mm256_setzero_si256());
        return _mm256_mask_mov_epi16(
            score, blocked, _mm512_castsi512_si256(neg16v));
    };
    const auto pair_rows16 = [&](int row_a, int row_b,
                                 __mmask16 erase_a, __mmask16 erase_b,
                                 __m512i* out_a, __m512i* out_b)
        __attribute__((always_inline)) {
        const __m512i packed = _mm512_packs_epi32(
            load_row32(row_a), load_row32(row_b));
        const __m512i score = classify16(
            _mm512_permutexvar_epi16(idxw_pair, packed));
        *out_a = _mm512_mask_mov_epi32(
            _mm512_madd_epi16(score, _mm512_load_si512(
                reinterpret_cast<const void*>(NOREV_RAW16_MW_EVEN))),
            erase_a, negv);
        *out_b = _mm512_mask_mov_epi32(
            _mm512_madd_epi16(score, _mm512_load_si512(
                reinterpret_cast<const void*>(NOREV_RAW16_MW_ODD))),
            erase_b, negv);
    };
    const __mmask16 erase_row[5] = {
        static_cast<__mmask16>(_mm_extract_epi16(row_block_pair, 0)),
        static_cast<__mmask16>(_mm_extract_epi16(row_block_pair, 1)),
        static_cast<__mmask16>(_mm_extract_epi16(row_block_pair, 2)),
        static_cast<__mmask16>(_mm_extract_epi16(row_block_pair, 3)),
        static_cast<__mmask16>(_mm_extract_epi16(row_block_pair, 4))};
#else
#if V44_MASKVEC >= 2
    const auto make_pair_row = [&](int k, __mmask16 mask)
        __attribute__((always_inline)) -> __m512i {
#elif V35_PAIR_MASKS
    const auto make_pair_row = [&](int k, unsigned pair_mask)
        __attribute__((always_inline)) -> __m512i {
#else
    const auto make_pair_row = [&](int k) __attribute__((always_inline)) -> __m512i {
#endif
#if V44_ROWOFF == 1
        __m512i raw = _mm512_castsi256_si512(_mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(
                rowbase0 + 4 * static_cast<int>(rowoff0[k]))));
        raw = _mm512_inserti64x4(raw, _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(
                rowbase1 + 4 * static_cast<int>(rowoff1[k]))), 1);
#elif V44_ROWOFF == 2
        constexpr int kRowBias = 4 * 2 * GRID_SIZE;
        __m512i raw = _mm512_castsi256_si512(_mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(
                rowbase0 + (4 * static_cast<int>(rowoff0[k]) - kRowBias))));
        raw = _mm512_inserti64x4(raw, _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(
                rowbase1 + (4 * static_cast<int>(rowoff1[k]) - kRowBias))), 1);
#else
        int rr0 = r0 - 2 + k;
        rr0 = rr0 < 0 ? 0 : rr0;
        rr0 = rr0 > GRID_SIZE - 1 ? GRID_SIZE - 1 : rr0;
        int rr1 = r1 - 2 + k;
        rr1 = rr1 < 0 ? 0 : rr1;
        rr1 = rr1 > GRID_SIZE - 1 ? GRID_SIZE - 1 : rr1;

        __m512i raw = _mm512_castsi256_si512(_mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(&grid[rr0][cb0])));
        raw = _mm512_inserti64x4(raw, _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(&grid[rr1][cb1])), 1);
#endif
#if V44_MASKVEC
        __m512i v = classify_row_pair_vec(raw, blockbits);
#else
        __m512i v = classify_row_pair_vec(raw, bombpat
#if !V34_TRUST_GOLD_RANGE
                                          , &raw_max
#endif
                                          );
#endif
        v = _mm512_permutexvar_epi32(k == 2 ? idx_h : idx_q, v);

#if V44_MASKVEC >= 2
        // mask already assembled by the caller
#elif V35_PAIR_MASKS
        const __mmask16 mask = static_cast<__mmask16>(pair_mask);
#else
        const unsigned mask0 = rowm0[k] | (k == 2 ? colh0 : colq0) |
            static_cast<unsigned>((block0 >> (k * 8)) & 0xFFu);
        const unsigned mask1 = rowm1[k] | (k == 2 ? colh1 : colq1) |
            static_cast<unsigned>((block1 >> (k * 8)) & 0xFFu);
        const __mmask16 mask = static_cast<__mmask16>(mask0 | (mask1 << 8));
#endif
        return _mm512_mask_mov_epi32(v, mask, negv);
    };
#endif

    // rows {1,3}: c11 / c12 / d10 for both units.
#if NOREV_RAW_INT16
    __m512i pr1, pr3;
    pair_rows16(1, 3, erase_row[1], erase_row[3], &pr1, &pr3);
#elif V44_MASKVEC >= 2
    const __m512i pr1 = make_pair_row(1, V44_ERASE_ROW(16));
    const __m512i pr3 = make_pair_row(3, V44_ERASE_ROW(48));
#elif V35_PAIR_MASKS
    const __m512i pr1 = make_pair_row(
        1, static_cast<unsigned>(_mm_extract_epi16(row_block_pair, 1)) | v46_colq);
    const __m512i pr3 = make_pair_row(
        3, static_cast<unsigned>(_mm_extract_epi16(row_block_pair, 3)) | v46_colq);
#else
    const __m512i pr1 = make_pair_row(1);
    const __m512i pr3 = make_pair_row(3);
#endif
    const __m128i r1u0lo = _mm512_extracti32x4_epi32(pr1, 0);
    const __m128i r3u0lo = _mm512_extracti32x4_epi32(pr3, 0);
    const __m128i r1u1lo = _mm512_extracti32x4_epi32(pr1, 2);
    const __m128i r3u1lo = _mm512_extracti32x4_epi32(pr3, 2);
    const __m128i p13u0lo = _mm_unpacklo_epi32(r1u0lo, r3u0lo);
    const __m128i p13u0hi = _mm_unpackhi_epi32(r1u0lo, r3u0lo);
    const __m128i p13u1lo = _mm_unpacklo_epi32(r1u1lo, r3u1lo);
    const __m128i p13u1hi = _mm_unpackhi_epi32(r1u1lo, r3u1lo);
    out->c11 = combine_lanes(_mm_alignr_epi8(p13u0hi, p13u0lo, 8),
                             _mm_alignr_epi8(p13u1hi, p13u1lo, 8));
    out->c12 = combine_lanes(_mm_blend_epi32(p13u0lo, p13u0hi, 0xC),
                             _mm_blend_epi32(p13u1lo, p13u1hi, 0xC));
    out->d10 = combine_lanes(
        _mm_unpacklo_epi32(_mm512_extracti32x4_epi32(pr1, 1),
                           _mm512_extracti32x4_epi32(pr3, 1)),
        _mm_unpacklo_epi32(_mm512_extracti32x4_epi32(pr1, 3),
                           _mm512_extracti32x4_epi32(pr3, 3)));

    // rows {0,4}: c21 / c22 and the second vertical step.
#if NOREV_RAW_INT16
    __m512i pr0, pr4;
    pair_rows16(0, 4, erase_row[0], erase_row[4], &pr0, &pr4);
#elif V44_MASKVEC >= 2
    const __m512i pr0 = make_pair_row(0, static_cast<__mmask16>(erase_lo));
    const __m512i pr4 = make_pair_row(4, static_cast<__mmask16>(erase_hi));
#elif V35_PAIR_MASKS
    const __m512i pr0 = make_pair_row(
        0, static_cast<unsigned>(_mm_extract_epi16(row_block_pair, 0)) | v46_colq);
    const __m512i pr4 = make_pair_row(
        4, static_cast<unsigned>(_mm_extract_epi16(row_block_pair, 4)) | v46_colq);
#else
    const __m512i pr0 = make_pair_row(0);
    const __m512i pr4 = make_pair_row(4);
#endif
    const __m128i r0u0lo = _mm512_extracti32x4_epi32(pr0, 0);
    const __m128i r4u0lo = _mm512_extracti32x4_epi32(pr4, 0);
    const __m128i r0u1lo = _mm512_extracti32x4_epi32(pr0, 2);
    const __m128i r4u1lo = _mm512_extracti32x4_epi32(pr4, 2);
    const __m128i p04u0lo = _mm_unpacklo_epi32(r0u0lo, r4u0lo);
    const __m128i p04u0hi = _mm_unpackhi_epi32(r0u0lo, r4u0lo);
    const __m128i p04u1lo = _mm_unpacklo_epi32(r0u1lo, r4u1lo);
    const __m128i p04u1hi = _mm_unpackhi_epi32(r0u1lo, r4u1lo);
    out->c21 = combine_lanes(_mm_alignr_epi8(p04u0hi, p04u0lo, 8),
                             _mm_alignr_epi8(p04u1hi, p04u1lo, 8));
    out->c22 = combine_lanes(_mm_blend_epi32(p04u0lo, p04u0hi, 0xC),
                             _mm_blend_epi32(p04u1lo, p04u1hi, 0xC));
    const __m256i v20 = combine_lanes(
        _mm_unpacklo_epi32(_mm512_extracti32x4_epi32(pr0, 1),
                           _mm512_extracti32x4_epi32(pr4, 1)),
        _mm_unpacklo_epi32(_mm512_extracti32x4_epi32(pr0, 3),
                           _mm512_extracti32x4_epi32(pr4, 3)));
    out->d20 = _mm256_add_epi32(out->d10, v20);

    // row 2: horizontal arms.  Only chunks 0 and 2 are live; chunks 1 and 3
    // are the replicated centre column used by the vertical extraction.
#if NOREV_RAW_INT16
    __m256i h16 = _mm512_cvtsepi32_epi16(load_row32(2));
    h16 = classify16y(_mm256_permutexvar_epi16(idxw_h, h16));
    h16 = _mm256_mask_mov_epi16(
        h16, erase_row[2], _mm256_set1_epi16(NOREV_RAW16_NEG));
    const __m512i ph = _mm512_slli_epi32(
        _mm512_cvtepi16_epi32(h16), GOLD_SHIFT);
#elif V44_MASKVEC >= 2
    const __m512i ph = make_pair_row(2, V44_ERASE_ROW(32));
#elif V35_PAIR_MASKS
    const __m512i ph = make_pair_row(
        2, static_cast<unsigned>(_mm_extract_epi16(row_block_pair, 2)) | v46_colh);
#else
    const __m512i ph = make_pair_row(2);
#endif
    const __m128i hu0 = _mm512_extracti32x4_epi32(ph, 0);
    const __m128i hu1 = _mm512_extracti32x4_epi32(ph, 2);
    out->d01 = combine_lanes(
        _mm_shuffle_epi32(hu0, _MM_SHUFFLE(1, 1, 0, 0)),
        _mm_shuffle_epi32(hu1, _MM_SHUFFLE(1, 1, 0, 0)));
    const __m256i h2 = combine_lanes(
        _mm_shuffle_epi32(hu0, _MM_SHUFFLE(3, 3, 2, 2)),
        _mm_shuffle_epi32(hu1, _MM_SHUFFLE(3, 3, 2, 2)));
    out->d02 = _mm256_add_epi32(out->d01, h2);
    out->axis1 = _mm256_blend_epi32(out->d10, out->d01, 0x66);
    out->axis2 = _mm256_blend_epi32(out->d20, out->d02, 0x66);

#if V34_TRUST_GOLD_RANGE
    return true;
#else
    return _mm512_cmpgt_epi32_mask(raw_max, _mm512_set1_epi32(GOLD_EXACT_MAX)) == 0;
#endif
}

// Invalid positions and out-of-contract cell values are extremely cold.  Keep
// the scalar reconstruction out of moveDecision, then convert once to the same
// packed representation consumed by the common DP body.
__attribute__((noinline, cold))
void build_pair_window_general(const int grid[GRID_SIZE][GRID_SIZE],
                               const Position& pos0, uint64_t block0,
                               const Position& pos1, uint64_t block1,
                               PairWindow* out) {
    UnitWindow w0, w1;
    build_unit_window_general(grid, pos0, block0, 0, 0, &w0);
    build_unit_window_general(grid, pos1, block1, 0, 0, &w1);
    out->c11 = combine_lanes(w0.c11, w1.c11);
    out->c12 = combine_lanes(w0.c12, w1.c12);
    out->c21 = combine_lanes(w0.c21, w1.c21);
    out->c22 = combine_lanes(w0.c22, w1.c22);
    out->d10 = combine_lanes(w0.d10, w1.d10);
    out->d20 = combine_lanes(w0.d20, w1.d20);
    out->d01 = combine_lanes(w0.d01, w1.d01);
    out->d02 = combine_lanes(w0.d02, w1.d02);
    out->axis1 = combine_lanes(w0.axis1, w1.axis1);
    out->axis2 = combine_lanes(w0.axis2, w1.axis2);
}
#endif  // V26_ZMM_PAIR_WINDOW
#endif  // V7_VEC_WINDOW

// ---------------------------------------------------------------- DP

// 把 4 个 bit 散布到 bit 0/4/8/12。SPREAD_PARENT_LUT[b] 就是这个函数，
// 而 pdep 一条指令做完（Zen3 起 1 uop / 3 周期），还省掉一张 32 字节的表。
#if !V33_WINNER_HLUT
inline uint32_t spread4(uint32_t bits) {
#if V7_ARITH_PARENT
    return _pdep_u32(bits, 0x1111u);
#else
    return V7_SPREAD[bits & 15u];
#endif
}

inline uint32_t assemble_parent_mask(uint32_t b11, uint32_t b12,
                                     uint32_t b21, uint32_t b22) {
    return spread4((~b11) & 15u) |
        (spread4((~b12) & 15u) << 1) |
        (spread4((~b21) & 15u) << 2) |
        (spread4((~b22) & 15u) << 3);
}
#endif

#if V7_INLINE_SOLVE
__attribute__((always_inline)) inline
#else
__attribute__((noinline))
#endif
#if V26_ZMM_PAIR_WINDOW
void solve_pair(const PairWindow& pair,
#if V37_QBASE_PAIR
                __m256i qbase,
#else
                __m128i qbase0, __m128i qbase1,
#endif
#if V33_WINNER_HLUT
                uint32_t* parent_planes,
#endif
#if V44_TAIL >= 2
                KChoice* choice,
#endif
                UnitResult* out0, UnitResult* out1) {
    const __m256i d10 = pair.d10;
    const __m256i d20 = pair.d20;
    const __m256i d01 = pair.d01;
    const __m256i d02 = pair.d02;
    const __m256i c11 = pair.c11;
    const __m256i c12 = pair.c12;
    const __m256i c21 = pair.c21;
    const __m256i c22 = pair.c22;
#else
void solve_pair(const UnitWindow& w0, const UnitWindow& w1,
                __m128i qbase0, __m128i qbase1,
#if V33_WINNER_HLUT
                uint32_t* parent_planes,
#endif
                UnitResult* out0, UnitResult* out1) {
    const __m256i d10 = combine_lanes(w0.d10, w1.d10);
    const __m256i d20 = combine_lanes(w0.d20, w1.d20);
    const __m256i d01 = combine_lanes(w0.d01, w1.d01);
    const __m256i d02 = combine_lanes(w0.d02, w1.d02);
    const __m256i c11 = combine_lanes(w0.c11, w1.c11);
    const __m256i c12 = combine_lanes(w0.c12, w1.c12);
    const __m256i c21 = combine_lanes(w0.c21, w1.c21);
    const __m256i c22 = combine_lanes(w0.c22, w1.c22);
#endif

#if V47_KMASK
    __mmask8 b11, b12, b21, b22;
    const __m256i d11 = _mm256_add_epi32(
        select_max_tie_left_k256(d10, d01, &b11), c11);
    const __m256i d12 = _mm256_add_epi32(
        select_max_tie_left_k256(d11, d02, &b12), c12);
    const __m256i d21 = _mm256_add_epi32(
        select_max_tie_left_k256(d20, d11, &b21), c21);
    const __m256i d22 = _mm256_add_epi32(
        select_max_tie_left_k256(d21, d12, &b22), c22);
#else
    __m256i b11, b12, b21, b22;
    const __m256i d11 = _mm256_add_epi32(select_max_tie_left_256(d10, d01, &b11), c11);
    const __m256i d12 = _mm256_add_epi32(select_max_tie_left_256(d11, d02, &b12), c12);
    const __m256i d21 = _mm256_add_epi32(select_max_tie_left_256(d20, d11, &b21), c21);
    const __m256i d22 = _mm256_add_epi32(select_max_tie_left_256(d21, d12, &b22), c22);
#endif

#if !V37_QBASE_PAIR || !V26_ZMM_PAIR_WINDOW
    const __m256i qbase = combine_lanes(qbase0, qbase1);
#endif
#if V26_ZMM_PAIR_WINDOW
    const __m256i axis1 = pair.axis1;
    const __m256i axis2 = pair.axis2;
#else
    const __m256i axis1 = combine_lanes(w0.axis1, w1.axis1);
    const __m256i axis2 = combine_lanes(w0.axis2, w1.axis2);
#endif
    const __m256i endpoint = _mm256_setr_epi32(0, 1, 1, 0, 0, 1, 1, 0);
#define V47_META_AXIS(d) \
    _mm256_or_si256(_mm256_set1_epi32((d) << DIST_SHIFT), endpoint)
#define V47_META_DIAG(d, e) _mm256_set1_epi32(((d) << DIST_SHIFT) | (e))
#if V47_KEYFUSE
    const __m256i key_axis1 = valid_key_zero_256(
        axis1, qbase, V47_META_AXIS(1));
    const __m256i key_axis2 = valid_key_zero_256(
        axis2, qbase, V47_META_AXIS(2));
    const __m256i key_d11 = valid_key_zero_256(
        d11, qbase, V47_META_DIAG(2, 2));
    const __m256i key_d12 = valid_key_zero_256(
        d12, qbase, V47_META_DIAG(3, 2));
    const __m256i key_d21 = valid_key_zero_256(
        d21, qbase, V47_META_DIAG(3, 3));
    const __m256i key_d22 = valid_key_zero_256(
        d22, qbase, V47_META_DIAG(4, 2));
#else
    const __m256i key_axis1 = valid_key_vector_256(
        axis1, _mm256_or_si256(qbase, V47_META_AXIS(1)));
    const __m256i key_axis2 = valid_key_vector_256(
        axis2, _mm256_or_si256(qbase, V47_META_AXIS(2)));
    const __m256i key_d11 = valid_key_vector_256(
        d11, _mm256_or_si256(qbase, V47_META_DIAG(2, 2)));
    const __m256i key_d12 = valid_key_vector_256(
        d12, _mm256_or_si256(qbase, V47_META_DIAG(3, 2)));
    const __m256i key_d21 = valid_key_vector_256(
        d21, _mm256_or_si256(qbase, V47_META_DIAG(3, 3)));
    const __m256i key_d22 = valid_key_vector_256(
        d22, _mm256_or_si256(qbase, V47_META_DIAG(4, 2)));
#endif

#if V44_TAIL
    // C: build the running maxima BEFORE reducing.  V40 reduces three 8-lane
    // quantities separately (3 x 2 vpshufd + 2 vpmaxsd = 12) and then re-maxes the
    // scalars.  Because hmax commutes with max,
    //     a2 = max(0, hmax(m2)),  m2 = max(axis1, axis2, d11)
    //     a3 = max(0, hmax(m3)),  m3 = max(m2, d12, d21)
    //     a4 = max(0, hmax(m4)),  m4 = max(m3, d22)
    // so m2 and m3 can share one ZMM and one permute+max pair reduces both, and the
    // "all candidates invalid -> STAY" clamp becomes a vector max against zero
    // instead of two scalar compares.
    const __m256i m2 = _mm256_max_epi32(
        _mm256_max_epi32(key_axis1, key_axis2), key_d11);
    const __m256i m3 = _mm256_max_epi32(
        m2, _mm256_max_epi32(key_d12, key_d21));
    const __m256i m4 = _mm256_max_epi32(m3, key_d22);

    // The four 128-bit lanes of m23 are [m2 unit0, m2 unit1, m3 unit0, m3 unit1];
    // vpshufd works inside each 128-bit lane, so one pair reduces all four.
    __m512i m23 = _mm512_inserti64x4(_mm512_castsi256_si512(m2), m3, 1);
    m23 = _mm512_max_epi32(m23, _mm512_shuffle_epi32(m23, _MM_PERM_BADC));
    m23 = _mm512_max_epi32(m23, _mm512_shuffle_epi32(m23, _MM_PERM_CDAB));
    __m256i r4 = horizontal_max_halves(m4);
#if !V47_KEYFUSE
    m23 = _mm512_max_epi32(m23, _mm512_setzero_si512());
    r4 = _mm256_max_epi32(r4, _mm256_setzero_si256());
#endif
    const __m256i cap2 = _mm512_castsi512_si256(m23);
    const __m256i exact3 = _mm512_extracti64x4_epi64(m23, 1);
    const __m256i exact4 = r4;
#else
    __m256i cap2 = _mm256_max_epi32(key_axis1, key_axis2);
    cap2 = horizontal_max_halves(_mm256_max_epi32(cap2, key_d11));
    const __m256i exact3 = horizontal_max_halves(_mm256_max_epi32(key_d12, key_d21));
    const __m256i exact4 = horizontal_max_halves(key_d22);
#endif

    // 全部候选都非法时退化成「原地不动」。这个兜底 key 必须和其他 key 带
    // 同样的常量偏置，否则 choose_k 里三个 joint 的偏移量会不一致 —— 
    // GOLD_SHIFT=18 那版最初就是在这里出了 1/5000 的差异。
    // 语义上也对：零长度路径的踩踏格数是 0，所以 clean = CLEAN_MAX。
#if V7_TRAMPLE == 4
    constexpr int32_t STAY_KEY = CLEAN_BIAS;
#else
    constexpr int32_t STAY_KEY = 0;
#endif
#if V44_TAIL >= 2
    static_assert(STAY_KEY == 0, "V44_TAIL folds the STAY fallback into max(.,0)");
    // AB = [a2, a3, a4, a2 | b4, b3, b2, b4].  horizontal_max_halves broadcast each
    // unit's max over its whole 128-bit lane, so any lane of a group can be picked.
    // Lane 3 deliberately duplicates lane 0 AND carries lane 0's k tag, so a tie
    // there yields the same k and the same keys; no zero-masking is needed.
    // Because each unit's max was broadcast over its whole 128-bit lane, EVERY lane
    // of the low half of y2/y3/y4 holds a2/a3/a4 and every lane of the high half
    // holds b2/b3/b4.  So AB needs no cross-lane permute at all — two immediate
    // blends do it, and no new .rodata constant is required.
    const __m256i y2 = _mm512_castsi512_si256(m23);
    const __m256i y3 = _mm512_extracti64x4_epi64(m23, 1);
    const __m256i y4 = r4;
    const __m256i AB = _mm256_blend_epi32(
        _mm256_blend_epi32(y2, y3, 0x22), y4, 0x94);

    // Three joint sums at once: lane0 = k=2 (u0.k2 + u1.k4), lane1 = k=3, lane2 = k=4.
    //
    // H：门编译进来之后这个掩码必须**同时挖掉 bit NOREV_SHIFT**，否则「不走完全
    // 反向」会开始影响 6 步怎么在两个角色之间分配 —— 那是另一件事，不在本轴的
    // 语义改动范围内。推理与下面标量 choose_k 的 GOLD_DISTANCE_MASK 一段相同：
    //   * 两个 dist 相加最多 4+4=8，正好占到 bit NOREV_SHIFT 且不再往上；
    //   * 两个输入在该位上都已被掩成 0，所以那一位正是留给进位的；
    //   * 进位停在 NOREV_SHIFT，仍在 GOLD_SHIFT 之下 ⇒ 不污染金币字段。
    // 指令数不变：同一条 vpandd，只是广播的立即数不同（在反汇编里核对过）。
#if V49_NOREV_GATE || V50_NOREV_GLOBAL || V49_QBASE16
    constexpr int32_t V49_GD_MASK =
        ~((1 << DIST_SHIFT) - 1) & ~(1 << QB16_SPARE_SHIFT);
    static_assert((V49_GD_MASK & (1 << QB16_SPARE_SHIFT)) == 0,
                  "k 分配必须对反向位不可见");
    static_assert(((V49_GD_MASK >> DIST_SHIFT) & 7) == 7,
                  "距离的三个有效位必须完整参与 k 分配");
    static_assert((V49_GD_MASK & ((1 << DIST_SHIFT) - 1)) == 0,
                  "DIST_SHIFT 以下的位（tailRank / 象限 / endpoint）不参与 k 分配");
    static_assert((V49_GD_MASK & ~((1 << GOLD_SHIFT) - 1)) ==
                  ~((1 << GOLD_SHIFT) - 1),
                  "金币字段必须完整参与 k 分配");
    // 4 + 4 = 8 => 进位落在 NOREV_SHIFT，且不超过它。
    static_assert(4 < (1 << (QB16_SPARE_SHIFT - DIST_SHIFT)),
                  "单个 dist 必须落在被借走那一位之下");
    static_assert(8 < (1 << (QB16_SPARE_SHIFT - DIST_SHIFT + 1)),
                  "两个 dist 相加的进位必须仍落在金币字段之下");
#else
    constexpr int32_t V49_GD_MASK = ~((1 << DIST_SHIFT) - 1);
#endif
    const __m256i gd = _mm256_and_si256(AB, _mm256_set1_epi32(V49_GD_MASK));
    const __m128i sums = _mm_add_epi32(_mm256_castsi256_si128(gd),
                                       _mm256_extracti128_si256(gd, 1));

    // marginal1 > marginal0  <=>  (a2&GM) + (b4&GM) > (a4&GM) + (b2&GM),
    // i.e. lane0 > lane2 of the gold-only sum.  One compare instead of two
    // differences plus two compares.
#if V47_GDFUSE
    // Each distance is in [0,4].  Their sum can carry into the borrowed bit 10,
    // but never into gold at bit 11.  Masking the existing sums therefore gives
    // exactly the separately accumulated gold-only sum, including NOREV's bit-10
    // layout (V49_GD_MASK already cleared that bit in each operand).
    static_assert(2 * 4 < (1 << (GOLD_SHIFT - DIST_SHIFT)),
                  "V47_GDFUSE needs distance sums not to carry into gold");
    const __m128i gsum = _mm_and_si128(
        sums, _mm_set1_epi32(~((1 << GOLD_SHIFT) - 1)));
#else
    const __m256i gm = _mm256_and_si256(
        AB, _mm256_set1_epi32(~((1 << GOLD_SHIFT) - 1)));
    const __m128i gsum = _mm_add_epi32(_mm256_castsi256_si128(gm),
                                       _mm256_extracti128_si256(gm, 1));
#endif
    const __m128i pref = _mm_cmpgt_epi32(_mm_shuffle_epi32(gsum, 0x00),
                                         _mm_shuffle_epi32(gsum, 0xAA));

    // Tags.  V40 forms joint | (tiebit << 1) and then compares; here the tie bit and
    // the argmax lane number are folded into ONE ternary-logic select, because the
    // joint sums are multiples of 1<<DIST_SHIFT and therefore disjoint from both.
    //   lane0 (k=2): tie bit 2 if marginal1 > marginal0   -> ((2<<2)|0) = 8   else 0
    //   lane1 (k=3): constant bit 4                       -> ((4<<2)|1) = 17 both
    //   lane2 (k=4): tie bit 2 if marginal0 >= marginal1   -> 2 : ((2<<2)|2) = 10
    static_assert(DIST_SHIFT >= 5, "the k tag plus lane number must fit below DIST_SHIFT");
    const __m128i tags = _mm_ternarylogic_epi32(
        pref, _mm_setr_epi32(8, 17, 2, 8), _mm_setr_epi32(0, 17, 10, 0), 0xCA);
    __m128i cand = _mm_or_si128(_mm_slli_epi32(sums, 2), tags);
    cand = _mm_max_epi32(cand, _mm_shuffle_epi32(cand, 0x4E));
    cand = _mm_max_epi32(cand, _mm_shuffle_epi32(cand, 0xB1));

    const int tag = _mm_cvtsi128_si32(cand) & 3;
    // key0 = AB[tag], key1 = AB[tag+4]; only index lanes 0/1 are read, so the index
    // is built in a GPR pair instead of loading a constant vector.
    const __m256i sel = _mm256_castsi128_si256(
        _mm_insert_epi32(_mm_cvtsi32_si128(tag), tag | 4, 1));
    const __m256i won = _mm256_permutevar8x32_epi32(AB, sel);
    choice->k = 2 + tag;
#if NOREV_MASKED_PREV_PAD
    // `sel` has low indices {tag, tag|4, 0, 0}; lanes 2/3 therefore contain
    // harmless duplicate keys and are allowed to write the two padding bytes.
    // qbase has already been materialised for this round, so updating state here
    // is exactly equivalent to the two conditional stores at the call-site.
    const __m128i winner_keys = _mm256_castsi256_si128(won);
    const __mmask8 moved = _mm_test_epi32_mask(winner_keys, winner_keys);
    const __m128i winner_q = _mm_and_si128(
        _mm_srli_epi32(winner_keys, 2), _mm_set1_epi32(3));
    _mm_mask_cvtepi32_storeu_epi8(state.prev_quadrant, moved, winner_q);
#endif
    choice->key0 = _mm_cvtsi128_si32(_mm256_castsi256_si128(won));
    choice->key1 = _mm_extract_epi32(_mm256_castsi256_si128(won), 1);
#elif V44_TAIL
    static_assert(STAY_KEY == 0, "V44_TAIL folds the STAY fallback into max(.,0)");
    const int32_t a2 = _mm_cvtsi128_si32(_mm256_castsi256_si128(cap2));
    const int32_t b2 = _mm_cvtsi128_si32(_mm256_extracti128_si256(cap2, 1));
    const int32_t a3 = _mm_cvtsi128_si32(_mm256_castsi256_si128(exact3));
    const int32_t b3 = _mm_cvtsi128_si32(_mm256_extracti128_si256(exact3, 1));
    const int32_t a4 = _mm_cvtsi128_si32(_mm256_castsi256_si128(exact4));
    const int32_t b4 = _mm_cvtsi128_si32(_mm256_extracti128_si256(exact4, 1));
#else
    int32_t a2 = _mm_cvtsi128_si32(_mm256_castsi256_si128(cap2));
    int32_t b2 = _mm_cvtsi128_si32(_mm256_extracti128_si256(cap2, 1));
    if (a2 < 0) a2 = STAY_KEY;
    if (b2 < 0) b2 = STAY_KEY;
    const int32_t ae3 = _mm_cvtsi128_si32(_mm256_castsi256_si128(exact3));
    const int32_t be3 = _mm_cvtsi128_si32(_mm256_extracti128_si256(exact3, 1));
    const int32_t a3 = ae3 > a2 ? ae3 : a2;
    const int32_t b3 = be3 > b2 ? be3 : b2;
    const int32_t ae4 = _mm_cvtsi128_si32(_mm256_castsi256_si128(exact4));
    const int32_t be4 = _mm_cvtsi128_si32(_mm256_extracti128_si256(exact4, 1));
    const int32_t a4 = ae4 > a3 ? ae4 : a3;
    const int32_t b4 = be4 > b3 ? be4 : b3;
#endif

#if V47_KMASK
    const __mmask16 planes_lo = _mm512_kunpackb(
        static_cast<__mmask16>(b12), static_cast<__mmask16>(b11));
    const __mmask16 planes_hi = _mm512_kunpackb(
        static_cast<__mmask16>(b22), static_cast<__mmask16>(b21));
    const __mmask32 planes = _mm512_kunpackw(
        static_cast<__mmask32>(planes_hi), static_cast<__mmask32>(planes_lo));
    *parent_planes = ~static_cast<uint32_t>(planes);
#if V44_TAIL >= 2
    (void)out0;
    (void)out1;
#else
    *out0 = {a2, a3, a4};
    *out1 = {b2, b3, b4};
#endif
#else
    const uint32_t m11 = static_cast<uint32_t>(_mm256_movemask_ps(_mm256_castsi256_ps(b11)));
    const uint32_t m12 = static_cast<uint32_t>(_mm256_movemask_ps(_mm256_castsi256_ps(b12)));
    const uint32_t m21 = static_cast<uint32_t>(_mm256_movemask_ps(_mm256_castsi256_ps(b21)));
    const uint32_t m22 = static_cast<uint32_t>(_mm256_movemask_ps(_mm256_castsi256_ps(b22)));
#if V33_WINNER_HLUT
    *parent_planes = ((~m11) & 0xFFu) |
        (((~m12) & 0xFFu) << 8) |
        (((~m21) & 0xFFu) << 16) |
        (((~m22) & 0xFFu) << 24);
#if V44_TAIL >= 2
    (void)out0;
    (void)out1;
#else
    *out0 = {a2, a3, a4};
    *out1 = {b2, b3, b4};
#endif
#else
    *out0 = {a2, a3, a4, assemble_parent_mask(m11, m12, m21, m22)};
    *out1 = {b2, b3, b4, assemble_parent_mask(m11 >> 4, m12 >> 4, m21 >> 4, m22 >> 4)};
#endif
#endif  // V47_KMASK
}

// ---------------------------------------------------------------- 输出

#if V7_ARITH_EMIT == 1
// HORIZONTAL_LUT 的算式版：bit t 为 1 表示第 t 步是横向。
// 这就是原表构造函数的逐字翻译，只是不再落表。
inline uint32_t horizontal_mask_arith(uint32_t parents, int moved, int endpoint) {
    const uint32_t p11 = parents & 1u;
    const uint32_t p12 = (parents >> 1) & 1u;
    const uint32_t p21 = (parents >> 2) & 1u;
    const uint32_t p22 = (parents >> 3) & 1u;
    const uint32_t pat11 = p11 ? 0x2u : 0x1u;
    const uint32_t pat12 = p12 ? (pat11 | 0x4u) : 0x3u;
    const uint32_t pat21 = p21 ? 0x4u : pat11;
    const uint32_t pat22 = p22 ? (pat21 | 0x8u) : pat12;
    const uint32_t m3 = (endpoint == 2) ? pat12 : pat21;
    const uint32_t m2 = (endpoint == 0) ? 0u : (endpoint == 1) ? 0x3u : pat11;
    const uint32_t m1 = (endpoint == 0) ? 0u : 0x1u;
    return moved == 4 ? pat22 : moved == 3 ? m3 : moved == 2 ? m2 : moved == 1 ? m1 : 0u;
}
#endif

#if V7_ARITH_EMIT
// ACTION_LUT 的算式版：把 4 个动作码打进一个 uint32 的 4 个字节。
// 动作编码恰好让这件事变成纯算术：
//   vertical   = q & 1          (UP=0 / DOWN=1)
//   horizontal = 2 | (q >> 1)   (LEFT=2 / RIGHT=3)
// 于是「先全填 vertical，再给横向步加上 (horizontal - vertical)」即可，
// 而 pdep 把 4 位掩码一次性散布成 4 个字节的 0/1。
inline uint32_t pack_actions_arith(int moved, int quadrant, uint32_t hmask) {
    const uint32_t v = static_cast<uint32_t>(quadrant) & 1u;
    const uint32_t delta = 2u + ((static_cast<uint32_t>(quadrant) >> 1) & 1u) - v;
    uint32_t packed = 0x01010101u * v;
    packed += _pdep_u32(hmask, 0x01010101u) * delta;
    // 第 moved 步之后一律 STAY(4)
    const uint32_t keep = moved >= 4 ? 0xFFFFFFFFu
                                     : ((1u << (static_cast<unsigned>(moved) * 8u)) - 1u);
    return (packed & keep) | (0x04040404u & ~keep);
}
#endif  // V7_ARITH_EMIT

#if V33_WINNER_HLUT
inline void emit_parent(int32_t key, uint32_t parent_planes,
                        unsigned unit_lane, int* out) {
#else
inline void emit_parent(int32_t key, uint32_t parent_mask, int* out) {
#endif
    const int moved = key_dist(key);
    const int quadrant = key_quadrant(key);
#if !NOREV_DIRECT_EMIT || NOREV_DIRECT_EMIT == 2
    const int endpoint = key & 3;
#endif
#if V33_WINNER_HLUT
    const uint32_t parents = _pext_u32(
        parent_planes, 0x01010101u << (unit_lane + quadrant));
#else
    const uint32_t parents = (parent_mask >> (quadrant * 4)) & 15u;
#endif
#if V7_ARITH_EMIT
#if V7_ARITH_EMIT == 1
    const uint32_t mask = horizontal_mask_arith(parents, moved, endpoint);
#else
    const uint32_t mask = HORIZONTAL_LUT[(parents * 4 + endpoint) * 5 + moved];
#endif
    const uint32_t packed = pack_actions_arith(moved, quadrant, mask);
    // VPMOVZXBD 一条指令把低 4 字节零扩展成 4 个 int32
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out),
                     _mm_cvtepu8_epi32(_mm_cvtsi32_si128(static_cast<int>(packed))));
#else
#if NOREV_DIRECT_EMIT == 1
    const uint32_t packed = FINAL_ACTION_LUT[
        parents | ((static_cast<uint32_t>(key) & 15u) << 4) |
        (static_cast<uint32_t>(moved) << 8)];
#elif NOREV_DIRECT_EMIT == 2
    const uint32_t compact_meta =
        2u * static_cast<uint32_t>(moved) + static_cast<uint32_t>(endpoint);
    const uint32_t packed = COMPACT_FINAL_ACTION_LUT[
        parents | (static_cast<uint32_t>(quadrant) << 4) | (compact_meta << 6)];
#else
    const uint32_t mask = HORIZONTAL_LUT[(parents * 4 + endpoint) * 5 + moved];
#if V7_LUT16
    // 半宽表：一条 pdep 把 4×3bit 还原成 4×8bit，结果与 uint32 表逐位相同。
    const uint32_t packed = _pdep_u32(ACTION_LUT16[(quadrant * 5 + moved) * 16 + mask],
                                      0x07070707u);
#else
    const uint32_t packed = ACTION_LUT[(quadrant * 5 + moved) * 16 + mask];
#endif
#endif  // NOREV_DIRECT_EMIT
    const __m128i bytes = _mm_cvtsi32_si128(static_cast<int>(packed));
    const __m128i words = _mm_unpacklo_epi8(bytes, _mm_setzero_si128());
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out),
                     _mm_unpacklo_epi16(words, _mm_setzero_si128()));
#endif
}

#if NOREV_PAIR_EMIT
#if NOREV_PAIR_PACK_PERMB
// The source byte vector is [packed0 byte 0..3, packed1 byte 0..3, zero...].
// For k=2/3/4 select the surviving prefix of unit 0 followed by all four bytes
// of unit 1.  Index 8 names a byte that is provably zero, reproducing the two
// zero high bytes of the old uint64_t merge when k is 2 or 3.  The whole 24-byte
// table is cache-line aligned, so every lookup touches exactly one cache line.
alignas(64) constexpr uint64_t NOREV_PAIR_PACK_PERMB_INDEX[3] = {
    UINT64_C(0x0808070605040100), // k=2: 0,1,4,5,6,7,zero,zero
    UINT64_C(0x0807060504020100), // k=3: 0,1,2,4,5,6,7,zero
    UINT64_C(0x0706050403020100), // k=4: 0,1,2,3,4,5,6,7
};
static_assert(sizeof(NOREV_PAIR_PACK_PERMB_INDEX) == 24,
              "paired route selector must remain a 24-byte single-line table");

constexpr uint64_t pair_pack_old(uint32_t packed0, uint32_t packed1,
                                 unsigned k) {
    const uint64_t keep = (UINT64_C(1) << (8u * k)) - 1u;
    return (static_cast<uint64_t>(packed0) & keep) |
           (static_cast<uint64_t>(packed1) << (8u * k));
}

constexpr uint64_t pair_pack_permb_scalar(uint32_t packed0, uint32_t packed1,
                                          unsigned k) {
    const uint64_t src = static_cast<uint64_t>(packed0) |
                         (static_cast<uint64_t>(packed1) << 32);
    const uint64_t selectors = NOREV_PAIR_PACK_PERMB_INDEX[k - 2u];
    uint64_t out = 0;
    for (unsigned byte = 0; byte < 8; ++byte) {
        const unsigned index = static_cast<unsigned>(selectors >> (8u * byte)) & 15u;
        const unsigned value = index < 8u
            ? static_cast<unsigned>(src >> (8u * index)) & 255u : 0u;
        out |= static_cast<uint64_t>(value) << (8u * byte);
    }
    return out;
}

constexpr bool pair_pack_permb_is_lossless() {
    // Byte-basis enumeration proves the permutation for arbitrary packed0/1,
    // not merely for one example route.
    for (unsigned k = 2; k <= 4; ++k) {
        for (unsigned source_byte = 0; source_byte < 8; ++source_byte) {
            const uint64_t basis = UINT64_C(0xA5) << (8u * source_byte);
            const uint32_t packed0 = static_cast<uint32_t>(basis);
            const uint32_t packed1 = static_cast<uint32_t>(basis >> 32);
            if (pair_pack_permb_scalar(packed0, packed1, k) !=
                pair_pack_old(packed0, packed1, k)) return false;
        }
    }
    return true;
}
static_assert(pair_pack_permb_is_lossless(),
              "VPERMB paired route merge must equal BZHI/SHLX for k=2..4");
#endif

__attribute__((always_inline)) inline
void emit_parent_pair(int32_t key0, int32_t key1, uint32_t parent_planes,
                      int k, int* out) {
    const unsigned moved0 = static_cast<unsigned>(key_dist(key0));
    const unsigned moved1 = static_cast<unsigned>(key_dist(key1));
    const unsigned q0 = static_cast<unsigned>(key_quadrant(key0));
    const unsigned q1 = static_cast<unsigned>(key_quadrant(key1));
    const uint32_t parents0 = _pext_u32(parent_planes, 0x01010101u << q0);
    const uint32_t parents1 = _pext_u32(parent_planes, 0x10101010u << q1);
#if NOREV_PAIR_COMPACT
    const unsigned endpoint0 = static_cast<unsigned>(key0) & 3u;
    const unsigned endpoint1 = static_cast<unsigned>(key1) & 3u;
    const unsigned meta0 = 2u * moved0 + endpoint0;
    const unsigned meta1 = 2u * moved1 + endpoint1;
    const uint32_t packed0 = COMPACT_FINAL_ACTION_LUT[
        parents0 | (q0 << 4) | (meta0 << 6)];
    const uint32_t packed1 = COMPACT_FINAL_ACTION_LUT[
        parents1 | (q1 << 4) | (meta1 << 6)];
#else
    const uint32_t packed0 = FINAL_ACTION_LUT[
        parents0 | ((static_cast<uint32_t>(key0) & 15u) << 4) | (moved0 << 8)];
    const uint32_t packed1 = FINAL_ACTION_LUT[
        parents1 | ((static_cast<uint32_t>(key1) & 15u) << 4) | (moved1 << 8)];
#endif

#if NOREV_PAIR_PACK_PERMB
    const __m128i routes = _mm_insert_epi32(
        _mm_cvtsi32_si128(static_cast<int>(packed0)),
        static_cast<int>(packed1), 1);
    const __m128i selectors = _mm_loadl_epi64(
        reinterpret_cast<const __m128i*>(
            &NOREV_PAIR_PACK_PERMB_INDEX[static_cast<unsigned>(k) - 2u]));
    const __m128i packed = _mm_permutexvar_epi8(selectors, routes);
    const __m256i actions = _mm256_cvtepu8_epi32(
        packed);
#else
    const unsigned shift = static_cast<unsigned>(k) * 8u;
    const uint64_t keep0 = _bzhi_u64(static_cast<uint64_t>(packed0), shift);
    const uint64_t packed = keep0 | (static_cast<uint64_t>(packed1) << shift);
    const __m256i actions = _mm256_cvtepu8_epi32(
        _mm_cvtsi64_si128(static_cast<long long>(packed)));
#endif
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), actions);
}

static_assert(__builtin_offsetof(GameOutput, k) == 6 * sizeof(int),
              "paired action store relies on k immediately following actions[5]");
static_assert(__builtin_offsetof(GameOutput, order) == 7 * sizeof(int),
              "paired action store covers exactly through order");
#endif

#if V46_DEADINIT
// F（搬自 V41 变体 5）：删掉聚合初始化的前提是 GameOutput 的布局与写序。
// 布局钉死，任何字段位置变化都会在编译期爆掉。
static_assert(sizeof(GameOutput) == 36, "GameOutput must be 9 x int32");
static_assert(sizeof(int) == 4, "action words are int32");
static_assert(S == 6, "the emit pair covers exactly six actions");
static_assert(__builtin_offsetof(GameOutput, actions) == 0, "actions first");
static_assert(__builtin_offsetof(GameOutput, k) == 24, "k right after actions[5]");
static_assert(__builtin_offsetof(GameOutput, order) == 28, "order right after k");
static_assert(__builtin_offsetof(GameOutput, vp) == 32, "vp is the 9th dword");

// 编译期证明：聚合初始化写的 9 个 dword **全部**被后面的写覆盖，所以删掉它
// 不改变 moveDecision 返回的任何一个字节。
// 后续写序（与 V40 一字不改）：
//   emit_parent(unit0) 写 actions[0..3]
//   emit_parent(unit1) 写 actions[k..k+3]，k ∈ {2,3,4}
//   out.k / out.order / out.vp 各写一次
// 用「每个 dword 一个互不相同的标号」而不是统一值，写错位置也会被抓到。
constexpr bool deadinit_is_fully_overwritten() {
    for (int k = 2; k <= 4; ++k) {
        // 0 = 尚未被覆盖（即聚合初始化的值仍然可见）
        int written[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < 4; ++i) written[i] = 1;              // unit 0
        for (int i = 0; i < 4; ++i) {
            const int idx = k + i;
            if (idx < 9) written[idx] = 2;                       // unit 1
        }
        written[6] = 3;                                          // out.k
        written[7] = 4;                                          // out.order
        written[8] = 5;                                          // out.vp
        for (int i = 0; i < 9; ++i) if (written[i] == 0) return false;
        // 六个 action dword 必须来自两次 emit：0..k-1 来自 unit 0，k..5 来自 unit 1。
        for (int i = 0; i < 6; ++i) {
            const int want = (i < k) ? 1 : 2;
            if (written[i] != want) return false;
        }
    }
    return true;
}
static_assert(deadinit_is_fully_overwritten(),
              "每一个 GameOutput dword 都必须被后续写覆盖，聚合初始化才是死写");
// k+4 >= 6 是「六个 STAY 一个都活不下来」的全部理由，单独钉一遍。
static_assert(2 + 4 >= S && 4 + 4 >= S,
              "unit 1 的 emit 必须覆盖到 actions[5]，否则 STAY 不是死写");
#endif  // V46_DEADINIT

// ---------------------------------------------------------------- k 选择

inline KChoice choose_k(const UnitResult& u0, const UnitResult& u1) {
    constexpr int32_t GOLD_MASK = ~((1 << GOLD_SHIFT) - 1);
#if V7_NOREV_HIGH
    // 步数分配只看金币和距离，刻意不看方向偏好：bit 11 落在 DIST_SHIFT 之上的
    // 保留区里，所以必须显式挖掉，否则「不走完全反向」会开始影响 6 步怎么分。
    // 两个距离相加最多 4+4=8，仍落在 bit 7..10（可容 15），不会进位到 bit 11；
    // 金币在 bit 15 以上独立相加，不受影响。
    constexpr int32_t GOLD_DISTANCE_MASK =
        ~((1 << DIST_SHIFT) - 1) & ~(1 << NOREV_SHIFT);
    static_assert((GOLD_DISTANCE_MASK & (1 << NOREV_SHIFT)) == 0,
                  "k 分配必须对反向位不可见");
    // 距离字段在输入里只用到 bits DIST_SHIFT..(NOREV_SHIFT-1) 之外的部分要完整保留。
    // =1 时反向位在距离之上，距离 4 位全留；=2 时反向位借走了距离的最高位，
    // 留 3 位（取值 1..4 完整），求和进位落进被借走那一位，而它在输入里恒为 0。
    static_assert((GOLD_DISTANCE_MASK & ~(1 << NOREV_SHIFT) &
                   ~((1 << DIST_SHIFT) - 1)) == (GOLD_DISTANCE_MASK),
                  "k 分配必须保留 DIST_SHIFT 以上除反向位之外的全部位");
    static_assert(((GOLD_DISTANCE_MASK >> DIST_SHIFT) & 7) == 7,
                  "距离的三个有效位必须完整参与 k 分配");
#elif V49_NOREV_GATE || V50_NOREV_GLOBAL || V49_QBASE16
    // 与 V44_TAIL>=2 向量路径里的 V49_GD_MASK 同一个掩码、同一套理由，
    // 搬过来是为了「不走 V44_TAIL=2 的配置也正确」，两条路径不能有一条漏挖。
    // 步数分配只看金币和距离，刻意不看方向偏好：bit NOREV_SHIFT 落在 DIST_SHIFT
    // 之上的保留区里，所以必须显式挖掉，否则「不走完全反向」会开始影响 6 步怎么分。
    // 两个距离相加最多 4+4=8，正好占到 bit NOREV_SHIFT 且不再往上；
    // 金币在 GOLD_SHIFT 以上独立相加，不受影响。
    constexpr int32_t GOLD_DISTANCE_MASK =
        ~((1 << DIST_SHIFT) - 1) & ~(1 << QB16_SPARE_SHIFT);
    static_assert((GOLD_DISTANCE_MASK & (1 << QB16_SPARE_SHIFT)) == 0,
                  "k 分配必须对反向位不可见");
    static_assert(((GOLD_DISTANCE_MASK >> DIST_SHIFT) & 7) == 7,
                  "距离的三个有效位必须完整参与 k 分配");
    static_assert(4 < (1 << (QB16_SPARE_SHIFT - DIST_SHIFT)),
                  "单个 dist 必须落在被借走那一位之下");
    static_assert(8 < (1 << (QB16_SPARE_SHIFT - DIST_SHIFT + 1)),
                  "两个 dist 相加的进位必须仍落在金币字段之下");
#else
    constexpr int32_t GOLD_DISTANCE_MASK = ~((1 << DIST_SHIFT) - 1);
#endif
    const int32_t marginal0 = (u0.k4 & GOLD_MASK) - (u0.k2 & GOLD_MASK);
    const int32_t marginal1 = (u1.k4 & GOLD_MASK) - (u1.k2 & GOLD_MASK);
    const int32_t joint2 = ((u0.k2 & GOLD_DISTANCE_MASK) + (u1.k4 & GOLD_DISTANCE_MASK)) |
                           ((marginal1 > marginal0) << 1);
    const int32_t joint3 = ((u0.k3 & GOLD_DISTANCE_MASK) + (u1.k3 & GOLD_DISTANCE_MASK)) |
                           (1 << 2);
    const int32_t joint4 = ((u0.k4 & GOLD_DISTANCE_MASK) + (u1.k2 & GOLD_DISTANCE_MASK)) |
                           ((marginal0 >= marginal1) << 1);

#if V7_FAST_SELECT
    // 三个 joint key 恒不相等：main 是 128 的倍数，joint2/joint4 的低 7 位只能是
    // 0 或 2，joint3 恒为 4，所以 max 无歧义。于是可以在低 2 位挂一个 k 标签，
    // 用两次无分支 max 同时选出 k，再用掩码选出两个 key。
    const int32_t e2 = (joint2 << 2);
    const int32_t e3 = (joint3 << 2) | 1;
    const int32_t e4 = (joint4 << 2) | 2;
    int32_t best = e2 > e3 ? e2 : e3;
    best = best > e4 ? best : e4;
    const int32_t tag = best & 3;
    const int32_t is3 = -static_cast<int32_t>(tag == 1);
    const int32_t is4 = -static_cast<int32_t>(tag == 2);
    int32_t key0 = u0.k2;
    key0 = (key0 & ~is3) | (u0.k3 & is3);
    key0 = (key0 & ~is4) | (u0.k4 & is4);
    int32_t key1 = u1.k4;
    key1 = (key1 & ~is3) | (u1.k3 & is3);
    key1 = (key1 & ~is4) | (u1.k2 & is4);
    return {2 + static_cast<int>(tag), key0, key1};
#else
    int best_k = 2;
    int32_t best = joint2;
    if (joint3 > best) { best = joint3; best_k = 3; }
    if (joint4 > best) best_k = 4;
    const int32_t key0 = best_k == 2 ? u0.k2 : best_k == 3 ? u0.k3 : u0.k4;
    const int32_t key1 = best_k == 2 ? u1.k4 : best_k == 3 ? u1.k3 : u1.k2;
    return {best_k, key0, key1};
#endif
}

// ---------------------------------------------------------------- 快照

#if V7_SNAPSHOT
// 赛事组固定的五区划分（QA 图）：
//   1 中央 9x9   rows 4-12,  cols 4-12   面积 81  质心 (8,8)
//   2 上 13x4    rows 0-3,   cols 0-12   面积 52  质心 (2,6)
//   3 左 4x13    rows 4-16,  cols 0-3    面积 52  质心 (10,2)
//   4 下 13x4    rows 13-16, cols 4-16   面积 52  质心 (14,10)
//   5 右 4x13    rows 0-12,  cols 13-16  面积 52  质心 (6,14)
// 81 + 52*4 = 289 = 17*17，自洽。
// weight = 4212/面积，把 gold_remaining 换算成可比的密度（4212 = 81*52）。
struct RegionGeom {
    signed char row;
    signed char col;
    int32_t weight;
};

constexpr RegionGeom REGION[6] = {
    {8, 8, 52},      // 索引 0 不用（区域编号从 1 开始）
    {8, 8, 52},      // 1 中央 9x9
    {2, 6, 81},      // 2 上
    {10, 2, 81},     // 3 左
    {14, 10, 81},    // 4 下
    {6, 14, 81},     // 5 右
};

__attribute__((noinline, cold))
void apply_snapshot(const Snapshot& snap) {
    int best_id = 1, second_id = 1;
    int64_t best_score = -1, second_score = -1;
    for (int i = 0; i < REGION_COUNT; ++i) {
        const int id = snap.regions[i].id;
        if (id < 1 || id > 5) continue;          // 不信任越界编号
        int64_t score =
            static_cast<int64_t>(snap.regions[i].gold_remaining) * REGION[id].weight;
        if (id == 1) score = score * V7_SNAP_CENTER_NUM / V7_SNAP_CENTER_DEN;
        if (score > best_score) {
            second_score = best_score;
            second_id = best_id;
            best_score = score;
            best_id = id;
        } else if (score > second_score) {
            second_score = score;
            second_id = id;
        }
    }
    state.target_row[0] = REGION[best_id].row;
    state.target_col[0] = REGION[best_id].col;
#if V7_SNAPSHOT >= 2
    // 双角色分头：角色 1 去次富区域，天然把两个角色岔开
    state.target_row[1] = REGION[second_id].row;
    state.target_col[1] = REGION[second_id].col;
#else
    state.target_row[1] = REGION[best_id].row;
    state.target_col[1] = REGION[best_id].col;
    (void)second_id;
#endif
}
#endif  // V7_SNAPSHOT

// 目标点 -> 象限偏好。target=(8,8) 时与 V6 的 centre_quadrant 完全等价：
//   V6:   ((row > 8) ? 0 : 1) | ((col > 8) ? 0 : 2)
//   本式: (8 >= row ? 1 : 0) | (8 >= col ? 2 : 0)
// 对 row<=8 都给 1、row>8 都给 0，列同理。
inline int preferred_quadrant(int unit, const Position& pos) {
    const int tr = state.target_row[unit];
    const int tc = state.target_col[unit];
    return static_cast<int>(tr >= pos.row) | (static_cast<int>(tc >= pos.col) << 1);
}

inline int tail_index(int unit, const Position& pos) {
    return (state.prev_quadrant[unit] + 1) * 4 + preferred_quadrant(unit, pos);
}

#if V49_QBASE16
// G/H：QB16 的索引。门关时恒等于 tail_index（0..19）；门开时加 20 落进上半区。
// tail_bias 是持久的 uint8，只在上一回合末尾被写，所以**入口即就绪**，
// 而这次加法折进 QB16 那条 `vmovq disp(base, idx, 8)` 的寻址里。
inline int qb16_index(int unit, const Position& pos) {
#if V49_ROWFOLD
    // I：行号（含门偏置）已经预存成一个字节，热路径只剩 movzbl + or。
    // `or` 与 `+` 在这里等价，因为 qb16_row 的低两位恒为 0 而 preferred 只占低两位。
    return static_cast<int>(state.qb16_row[unit]) | preferred_quadrant(unit, pos);
#elif V49_NOREV_GATE && !V49_FORCE_GATE_OFF
    return tail_index(unit, pos) + static_cast<int>(state.tail_bias[unit]);
#else
    // V49_FORCE_GATE_OFF：索引永远落在门关半区，所以决策必须逐位复现基座；
    // 但计数器与 tail_bias 的更新照常执行，于是这条门禁检验的正是「管线接对了」，
    // 而不是「把改动整段编译掉了」。
    return tail_index(unit, pos);
#endif
}
#endif

#if V7_ZONE
// 3x3 粗区的代表点。刻意取 3 / 8 / 13 而不是各段的几何中心 ——
// 一是让 (8,3) 和 (8,13) 落在候选里（那是地图六手调出来的最优解，
// 这个机制应当有能力自己找到它），二是 target 只参与「在我上/下、左/右」
// 这四种方向比较，精确落点本来就不重要。
constexpr signed char ZONE_CENTER[3] = {3, 8, 13};

// 17 行/列切三段：0..5 / 6..11 / 12..16。两次比较，无除法无查表。
inline int zone_index(int r, int c) {
    const int zr = r >= 12 ? 2 : (r >= 6 ? 1 : 0);
    const int zc = c >= 12 ? 2 : (c >= 6 ? 1 : 0);
    return zr * 3 + zc;
}

// V7_ZONE_PERIOD 的 log2，用来把累积收益还原成单回合均值。
constexpr int zone_period_log(int p) {
    int n = 0;
    while ((1 << n) < p) ++n;
    return n;
}
constexpr int ZONE_PERIOD_LOG = zone_period_log(V7_ZONE_PERIOD);

// 结算一次：把这一周期累积的收益记进两个角色**当前**所在的区，
// 然后挑出产出最高和次高的区分别作为它们的新目标。
// 次高给第二个角色是为了让它们分头 —— 挤在一处会让两个 5x5 视野大面积重叠，
// 等于只有一个角色在找金币。
//
// noinline + cold 是必须的：它每 V7_ZONE_PERIOD 回合才执行一次，
// 内联进 moveDecision 会让这些比较白占热路径的 i-cache。
// （实测内联那版 `.text` +661 字节、本机 +4.2ns；标 cold 后降到 +451。）
//
// 用「结算时刻的位置」而不是逐回合的位置，是刻意的近似：一个周期内角色
// 移动范围有限，而这样热路径就只剩两次加法，不必每回合算区索引。
__attribute__((noinline, cold))
void settle_zones(const Position& p0, const Position& p1) {
    uint16_t* ema = state.zone_ema;
    const int z0 = zone_index(p0.row, p0.col);
    const int z1 = zone_index(p1.row, p1.col);
    // >> ZONE_PERIOD_LOG 把周期累积还原成单回合均值，
    // 这样 EMA 的量级与 V7_ZONE_PERIOD 无关，衰减档才有一致含义。
    const uint16_t g0 = static_cast<uint16_t>(state.acc_gold[0] >> ZONE_PERIOD_LOG);
    const uint16_t g1 = static_cast<uint16_t>(state.acc_gold[1] >> ZONE_PERIOD_LOG);
    ema[z0] = static_cast<uint16_t>(ema[z0] - (ema[z0] >> V7_ZONE_DECAY) + g0);
    ema[z1] = static_cast<uint16_t>(ema[z1] - (ema[z1] >> V7_ZONE_DECAY) + g1);
    state.acc_gold[0] = 0;
    state.acc_gold[1] = 0;

    int best = 0;
    int second = 1;
    if (state.zone_ema[1] > state.zone_ema[0]) {
        best = 1;
        second = 0;
    }
    for (int i = 2; i < 9; ++i) {
        const uint16_t v = state.zone_ema[i];
        if (v > state.zone_ema[best]) {
            second = best;
            best = i;
        } else if (v > state.zone_ema[second]) {
            second = i;
        }
    }
    state.target_row[0] = ZONE_CENTER[best / 3];
    state.target_col[0] = ZONE_CENTER[best % 3];
    state.target_row[1] = ZONE_CENTER[second / 3];
    state.target_col[1] = ZONE_CENTER[second % 3];
}
#endif

}  // namespace

// ---------------------------------------------------------------- 入口

// V22_MOVE_ALIGN64: 把热入口按 64B 对齐。默认关闭，所以既有产物逐字节不变。
//
// 这条是从 V17 借过来的。V17 的归因结论是：服务器每回合只调用 moveDecision 一次，
// 指令 cache 和分支预测器都保持不了热，所以入口地址、入口对齐和 I-cache 组的分布
// 会直接体现在线上 cost 上——V16 曾经因为冷代码被链接器放到热入口之前、把入口从
// 0x970 推到 0xdd0，普通回合 P50 就涨了 70ns，而 C++ 热路径一条语句都没变。
// 对齐只改属性，不改任何一条语句，因此决策必然逐位不变；是否有收益只能看线上 cost。
#ifndef V22_MOVE_ALIGN64
#define V22_MOVE_ALIGN64 0
#endif

extern "C" GameOutput moveDecision(const GameInput* input)
#if V22_MOVE_ALIGN64
    __attribute__((visibility("default"), aligned(64)));
#else
    __attribute__((visibility("default")));
#endif

extern "C" GameOutput moveDecision(const GameInput* input) {
#if V46_DEADINIT
    // F: 没有聚合初始化 —— 36 个字节全部被后面的写覆盖，见
    // deadinit_is_fully_overwritten() 的编译期证明。
    GameOutput out;
#else
    GameOutput out = {{A_STAY, A_STAY, A_STAY, A_STAY, A_STAY, A_STAY}, 3, 0, 0};
#endif
#if !V34_TRUST_OFFICIAL_INPUT
    if (__builtin_expect(input == nullptr, 0)) return out;
#endif

    // B13 相位：一次性把炸弹从「可穿过」翻成「阻挡」。
    // 官方一进程一局，所以不做跨局复位，也不每轮写 last_round。
    if (__builtin_expect(input->round == V7_BOMB_PHASE_ROUND, 0)) {
        CELL_LUT[static_cast<unsigned>(BOMB + 5)] = NEG;
        g_bomb_pattern = 0;   // 0 不可能 < 0，于是 blocked 退化成纯 raw<0
#if V44_MASKVEC
        g_block_bits = V44_BLOCK_BITS_PHASE2;
#endif
    }

    const Position pos0 = input->my_units[0];
    const Position pos1 = input->my_units[1];
    const Position enemy0 = input->visible_enemies[0];
    const Position enemy1 = input->visible_enemies[1];

#if V7_PREFETCH
    // 尽早发出预取，让后面 tails/占位者/守门那几十个周期去掩盖内存延迟。
    // 行索引钳位到 [0,16]，保证不会预取到结构体之外。
    {
        const int* base = &input->grid[0][0];
        for (int k = -2; k <= 2; ++k) {
            int a = pos0.row + k;
            int b = pos1.row + k;
            a = a < 0 ? 0 : (a > GRID_SIZE - 1 ? GRID_SIZE - 1 : a);
            b = b < 0 ? 0 : (b > GRID_SIZE - 1 ? GRID_SIZE - 1 : b);
            __builtin_prefetch(base + a * GRID_SIZE, 0, 3);
            __builtin_prefetch(base + b * GRID_SIZE, 0, 3);
        }
#if V7_SNAPSHOT
        // Snapshot 有 148 字节、约 3 条 cache line，线上是冷的。第二轮实测
        // 快照让 p50 只 +10ns 但 p90 +60ns —— 典型的「成本集中在 20% 回合」，
        // 也就是这几条线的缺失。所以在真正读它之前就把它拉进来。
        if (input->snapshot_valid != 0) {
            const char* snap = reinterpret_cast<const char*>(&input->snapshot);
            __builtin_prefetch(snap, 0, 3);
            __builtin_prefetch(snap + 64, 0, 3);
            __builtin_prefetch(snap + 128, 0, 3);
        }
#endif
#if V7_TRAMPLE
        if (input->num_visible_npcs >= 3) {
            const char* npc = reinterpret_cast<const char*>(input->visible_npcs);
            __builtin_prefetch(npc, 0, 3);
            __builtin_prefetch(npc + 64, 0, 3);
        }
#endif
    }
#endif

#if V7_SNAPSHOT
#if V7_SNAP_DEFERRED
    // 入口只发预取，真正的读放到函数末尾（见结尾的 apply_snapshot）。
    // Snapshot 148 字节横跨 3 条 cache line，线上是冷的；把读操作推到所有
    // 真实工作之后，这三次 miss 就完全被计算掩盖掉。
    if (input->snapshot_valid != 0) {
        const char* snap = reinterpret_cast<const char*>(&input->snapshot);
        __builtin_prefetch(snap, 0, 1);
        __builtin_prefetch(snap + 64, 0, 1);
        __builtin_prefetch(snap + 128, 0, 1);
    }
#else
    if (__builtin_expect(input->snapshot_valid != 0, 0)) apply_snapshot(input->snapshot);
#endif
#endif

#if V37_QBASE_PAIR && V49_QBASE16
    // G/H：整条展开链塌成一次成对表读。
    //
    // 旧链 9 条：2 x movzwl TAIL_LUT + 2 x vpbroadcastd + vinserti128
    //            + vpsrlvd(shifts) + vpandd(7) + vpslld(4) + vpord(quadrants)
    // 新链 4 条向量：vmovq + vmovq + vpunpcklqdq + vpmovzxwd
    //            （加两次索引计算，合计 6 条），并且 `shifts` 与 `quadrants`
    //            这两个 32 字节常量直接从 .rodata 消失。
    //
    // lane 顺序与旧链**逐位相同**：
    //   旧链 repeated 的低 128 位是角色 0、高 128 位是角色 1，
    //   每半区 lane j 用移位量 3j，也就是象限 j。
    //   新链 vpunpcklqdq(lo, hi) 把角色 0 的四个 uint16 放进 word 0..3、
    //   角色 1 的放进 word 4..7，vpmovzxwd 再零扩展成 dword 0..7 ——
    //   低半区 = 角色 0 的 q=0..3，高半区 = 角色 1 的 q=0..3。同一个布局。
    static_assert(V7_NOREV_HIGH == 0,
                  "V49_QBASE16 replaces the V7_NOREV_HIGH qbase formulations");
#if NOREV_SIMD_PREF
    // input->my_units is laid out as {u0.row,u0.col,u1.row,u1.col}. For the
    // frozen targets, preferred bits are exactly:
    //   {row0 < 9, col0 < 4, row1 < 9, col1 < 9}.
    // The movemask therefore packs both preferred quadrants into one nibble.
    const __m128i unit_positions = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(input->my_units));
    const __m128i pref_thresholds = _mm_setr_epi32(9, 4, 9, 9);
    const unsigned preferred_pair = static_cast<unsigned>(_mm_movemask_ps(
        _mm_castsi128_ps(_mm_cmpgt_epi32(pref_thresholds, unit_positions))));
    const unsigned qb_index0 = static_cast<unsigned>(
        (state.prev_quadrant[0] + 1) * 4) | (preferred_pair & 3u);
    const unsigned qb_index1 = static_cast<unsigned>(
        (state.prev_quadrant[1] + 1) * 4) | ((preferred_pair >> 2) & 3u);
    const __m128i qb0 = _mm_loadl_epi64(
        reinterpret_cast<const __m128i*>(&QB16[qb_index0]));
    const __m128i qb1 = _mm_loadl_epi64(
        reinterpret_cast<const __m128i*>(&QB16[qb_index1]));
#else
    const __m128i qb0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(
        &QB16[static_cast<unsigned>(qb16_index(0, pos0))]));
    const __m128i qb1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(
        &QB16[static_cast<unsigned>(qb16_index(1, pos1))]));
#endif
    const __m256i qbase = _mm256_cvtepu16_epi32(_mm_unpacklo_epi64(qb0, qb1));
#elif V37_QBASE_PAIR
    // Frozen V33 uses the ordinary 3-bit tail layout.  Higher no-reverse modes
    // borrow bits outside it and need a separate paired formulation.
    static_assert(V7_NOREV_HIGH == 0,
                  "V37 paired qbase is for the frozen V33 tail layout");
    const uint32_t packed_tail0 =
        TAIL_LUT[static_cast<unsigned>(tail_index(0, pos0))];
    const uint32_t packed_tail1 =
        TAIL_LUT[static_cast<unsigned>(tail_index(1, pos1))];
    const __m128i repeated0 = _mm_set1_epi32(static_cast<int>(packed_tail0));
    const __m128i repeated1 = _mm_set1_epi32(static_cast<int>(packed_tail1));
    const __m256i repeated = _mm256_inserti128_si256(
        _mm256_castsi128_si256(repeated0), repeated1, 1);
    const __m256i shifts = _mm256_setr_epi32(0, 3, 6, 9, 0, 3, 6, 9);
    const __m256i quadrants = _mm256_setr_epi32(0, 4, 8, 12, 0, 4, 8, 12);
    const __m256i tails = _mm256_and_si256(
        _mm256_srlv_epi32(repeated, shifts), _mm256_set1_epi32(7));
    const __m256i qbase =
        _mm256_or_si256(_mm256_slli_epi32(tails, 4), quadrants);
#elif V7_ARITH_QBASE
    // 直接算出 qbase = [(tail<<4)|(q<<2)]，q=0..3，不查 TAIL_LUT。
    //   qrank        = (~(q ^ preferred)) & 3          朝目标的四级排序
    //   not_reversed = previous < 0 || q != (previous ^ 3)   不走完全反向
    //   tail         = qrank | (not_reversed << 2)
    const __m128i qv = _mm_setr_epi32(0, 1, 2, 3);
    const auto make_qbase_arith = [&](int unit, const Position& pos) -> __m128i {
        const int preferred = preferred_quadrant(unit, pos);
        const int previous = state.prev_quadrant[unit];
        const __m128i qrank = _mm_and_si128(
            _mm_xor_si128(_mm_xor_si128(qv, _mm_set1_epi32(preferred)),
                          _mm_set1_epi32(-1)),
            _mm_set1_epi32(3));
        // previous < 0 时没有反向可言，用一个 q 取不到的值(-1)让比较恒不相等
        const __m128i opp = _mm_set1_epi32(previous < 0 ? -1 : (previous ^ 3));
        const __m128i not_rev = _mm_andnot_si128(_mm_cmpeq_epi32(qv, opp),
                                                _mm_set1_epi32(1));
        const __m128i tail = _mm_or_si128(qrank, _mm_slli_epi32(not_rev, 2));
        const __m128i base =
            _mm_or_si128(_mm_slli_epi32(tail, 4), _mm_slli_epi32(qv, 2));
#if V7_NOREV_HIGH
        // 这条路径上 not_rev 已经是向量了，直接再移一次即可，不必重算。
        return _mm_or_si128(base, _mm_slli_epi32(not_rev, NOREV_SHIFT));
#else
        return base;
#endif
    };
    const __m128i qbase0 = make_qbase_arith(0, pos0);
    const __m128i qbase1 = make_qbase_arith(1, pos1);
#elif V7_QBASE_LUT
    const __m128i qbase0 = _mm_load_si128(
        reinterpret_cast<const __m128i*>(&QBASE_LUT[static_cast<unsigned>(tail_index(0, pos0))]));
    const __m128i qbase1 = _mm_load_si128(
        reinterpret_cast<const __m128i*>(&QBASE_LUT[static_cast<unsigned>(tail_index(1, pos1))]));
#else
    const __m128i qbase0 = [&] {
        const uint32_t t = TAIL_LUT[static_cast<unsigned>(tail_index(0, pos0))];
        V7_NOREV_GATE(0)
        return V7_LIFT_NOREV(_mm_set_epi32(
            static_cast<int>((((t >> 9) & 7u) << 4) | (3u << 2)) | V7_NOREV_BIT(t, 3),
            static_cast<int>((((t >> 6) & 7u) << 4) | (2u << 2)) | V7_NOREV_BIT(t, 2),
            static_cast<int>((((t >> 3) & 7u) << 4) | (1u << 2)) | V7_NOREV_BIT(t, 1),
            static_cast<int>((t & 7u) << 4) | V7_NOREV_BIT(t, 0)), 0);
    }();
    const __m128i qbase1 = [&] {
        const uint32_t t = TAIL_LUT[static_cast<unsigned>(tail_index(1, pos1))];
        V7_NOREV_GATE(1)
        return V7_LIFT_NOREV(_mm_set_epi32(
            static_cast<int>((((t >> 9) & 7u) << 4) | (3u << 2)) | V7_NOREV_BIT(t, 3),
            static_cast<int>((((t >> 6) & 7u) << 4) | (2u << 2)) | V7_NOREV_BIT(t, 2),
            static_cast<int>((((t >> 3) & 7u) << 4) | (1u << 2)) | V7_NOREV_BIT(t, 1),
            static_cast<int>((t & 7u) << 4) | V7_NOREV_BIT(t, 0)), 1);
    }();
#endif

#if NOREV_BLOCKER_SCOPE == 2
    uint64_t block0 = 0;
    uint64_t block1 = 0;
#elif NOREV_BLOCKER_SCOPE == 1
    // Official my-unit positions are always valid.  Keep only the teammate as
    // a hard occupant; the two directions are exact opposites.
    const int mate_dr = pos1.row - pos0.row;
    const int mate_dc = pos1.col - pos0.col;
    uint64_t block0 = lane_bit_of(mate_dr, mate_dc);
    uint64_t block1 = lane_bit_of(-mate_dr, -mate_dc);
#elif V30_VBMI_BLOCKERS
    const BlockerPair blockers = collect_blockers_pair(pos0, pos1, enemy0, enemy1);
    uint64_t block0 = blockers.unit0;
    uint64_t block1 = blockers.unit1;
#else
    uint64_t block0 = collect_blockers(pos0, pos1, enemy0, enemy1);
    uint64_t block1 = collect_blockers(pos1, pos0, enemy0, enemy1);
#endif

    uint64_t trample0 = 0, trample1 = 0;
#if V7_TRAMPLE == 4
    // clean 模式的减数是编译期常量：每个踩踏格让 clean 字段 -1
    constexpr int32_t penalty0 = CLEAN_ONE;
    constexpr int32_t penalty1 = CLEAN_ONE;
#else
    int32_t penalty0 = 0, penalty1 = 0;
#endif
#if V7_TRAMPLE
    // 三级守门，按「代价从低到高」排序：
    //   1) 持币是否够到需要在意踩踏。my_units_gold 紧跟 my_units，几乎必然
    //      与刚读过的位置同一条 cache line，所以这一级基本零成本；而开局
    //      很多回合持币都不够，直接整段跳过（硬阻挡模式下这一级还是精确的：
    //      低于阈值本来就不会阻挡任何格子）。
    //   2) 可见 NPC >= 3。同格 >=3 必然要求这条成立。
    //   3) 在绝对坐标里判堆，为空时（绝大多数情况）连窗口映射都不做。
    const int32_t gold0 = input->my_units_gold[0];
    const int32_t gold1 = input->my_units_gold[1];
    // 所有模式共用同一个阈值，这样阈值就是一个纯粹的「守门成本 vs 覆盖率」旋钮：
    // 抬高它 = 更少回合去读 visible_npcs 那 2 条冷 cache line，而被砍掉的
    // 恰好是踩踏损失最小（持币最少）的那些回合。
    const bool gold_matters = (gold0 >= V7_TRAMPLE_BLOCK_GOLD) ||
                              (gold1 >= V7_TRAMPLE_BLOCK_GOLD);
    if (__builtin_expect(gold_matters && input->num_visible_npcs >= 3, 0)) {
        const TrampleSet stacks = find_npc_stacks(input->visible_npcs);
        if (__builtin_expect(stacks.slots != 0, 0)) {
            const uint64_t t0 = trample_bits_for(stacks, pos0);
            const uint64_t t1 = trample_bits_for(stacks, pos1);
#if V7_TRAMPLE == 2
            // 硬阻挡：并进 block_bits，热路径零新增成本
            if (gold0 >= V7_TRAMPLE_BLOCK_GOLD) block0 |= t0;
            if (gold1 >= V7_TRAMPLE_BLOCK_GOLD) block1 |= t1;
#elif V7_TRAMPLE == 4
            // clean 字段模式：subtrahend 是编译期常量，不依赖持币
            trample0 = t0;
            trample1 = t1;
#else
            // 罚分模式：按金币精确定价 ceil(0.05 * 持币)
            trample0 = t0;
            trample1 = t1;
            if (trample0 | trample1) {
                int32_t p0 = (gold0 + 19) / 20;
                int32_t p1 = (gold1 + 19) / 20;
                p0 = p0 < 0 ? 0 : (p0 > PEN_CLAMP ? PEN_CLAMP : p0);
                p1 = p1 < 0 ? 0 : (p1 > PEN_CLAMP ? PEN_CLAMP : p1);
                penalty0 = p0 << GOLD_SHIFT;
                penalty1 = p1 << GOLD_SHIFT;
            }
#endif
        }
    }
#endif

#if V26_ZMM_PAIR_WINDOW
    PairWindow pair;
#if V34_TRUST_GOLD_RANGE
#if V34_TRUST_OFFICIAL_INPUT
    (void)build_pair_window_vec(input->grid, pos0, block0,
                                pos1, block1, &pair);
#else
    const bool positions_valid =
        static_cast<unsigned>(pos0.row) < GRID_SIZE &&
        static_cast<unsigned>(pos0.col) < GRID_SIZE &&
        static_cast<unsigned>(pos1.row) < GRID_SIZE &&
        static_cast<unsigned>(pos1.col) < GRID_SIZE;
    if (__builtin_expect(!positions_valid, 0)) {
        build_pair_window_general(input->grid, pos0, block0,
                                  pos1, block1, &pair);
    } else {
        (void)build_pair_window_vec(input->grid, pos0, block0,
                                    pos1, block1, &pair);
    }
#endif
#elif V34_TRUST_OFFICIAL_INPUT
    if (__builtin_expect(!build_pair_window_vec(input->grid, pos0, block0,
                                                pos1, block1, &pair), 0)) {
        build_pair_window_general(input->grid, pos0, block0,
                                  pos1, block1, &pair);
    }
#else
    const bool positions_valid =
        static_cast<unsigned>(pos0.row) < GRID_SIZE &&
        static_cast<unsigned>(pos0.col) < GRID_SIZE &&
        static_cast<unsigned>(pos1.row) < GRID_SIZE &&
        static_cast<unsigned>(pos1.col) < GRID_SIZE;
    if (__builtin_expect(!positions_valid ||
                         !build_pair_window_vec(input->grid, pos0, block0,
                                                pos1, block1, &pair), 0)) {
        build_pair_window_general(input->grid, pos0, block0,
                                  pos1, block1, &pair);
    }
#endif
#else
    UnitWindow w0, w1;
#if V7_VEC_WINDOW
    if (__builtin_expect(static_cast<unsigned>(pos0.row) < GRID_SIZE &&
                         static_cast<unsigned>(pos0.col) < GRID_SIZE &&
                         static_cast<unsigned>(pos1.row) < GRID_SIZE &&
                         static_cast<unsigned>(pos1.col) < GRID_SIZE, 1)) {
        bool ok;
#if V7_TRAMPLE == 1
        // 模式 1 保留两份特化：无踩踏时零成本，但代价是代码体积几乎翻倍
        // （19136 -> 23232），线上实测反而更慢。保留只为对照。
        if (__builtin_expect((trample0 | trample1) == 0, 1)) {
            ok = build_unit_window_vec<false>(input->grid, pos0, block0, 0, 0, &w0);
            ok &= build_unit_window_vec<false>(input->grid, pos1, block1, 0, 0, &w1);
        } else {
            ok = build_unit_window_vec<true>(input->grid, pos0, block0, trample0, penalty0, &w0);
            ok &= build_unit_window_vec<true>(input->grid, pos1, block1, trample1, penalty1, &w1);
        }
#elif V7_TRAMPLE >= 3
        // 模式 3/4：只实例化 <true> 这一份，始终做那一次掩码减。
        // 无踩踏时 tmask=0，掩码减是空操作，但指令照发 —— 换来的是
        // **只有一份代码**，没有模板膨胀，也不需要任何运行期分支。
        ok = build_unit_window_vec<true>(input->grid, pos0, block0, trample0, penalty0, &w0);
        ok &= build_unit_window_vec<true>(input->grid, pos1, block1, trample1, penalty1, &w1);
#else
        ok = build_unit_window_vec<false>(input->grid, pos0, block0, 0, 0, &w0);
        ok &= build_unit_window_vec<false>(input->grid, pos1, block1, 0, 0, &w1);
#endif
        if (__builtin_expect(!ok, 0)) {
            build_unit_window_general(input->grid, pos0, block0, trample0, penalty0, &w0);
            build_unit_window_general(input->grid, pos1, block1, trample1, penalty1, &w1);
        }
    } else {
        build_unit_window_general(input->grid, pos0, block0, trample0, penalty0, &w0);
        build_unit_window_general(input->grid, pos1, block1, trample1, penalty1, &w1);
    }
#else
    build_unit_window_scalar_fast(input->grid, pos0, block0, trample0, penalty0, &w0);
    build_unit_window_scalar_fast(input->grid, pos1, block1, trample1, penalty1, &w1);
#endif
#endif  // V26_ZMM_PAIR_WINDOW

    UnitResult u0, u1;
#if V33_WINNER_HLUT
    uint32_t parent_planes;
#endif
#if V44_TAIL >= 2
    KChoice v44choice;
#endif
#if V26_ZMM_PAIR_WINDOW
#if V37_QBASE_PAIR
    solve_pair(pair, qbase,
#else
    solve_pair(pair, qbase0, qbase1,
#endif
#if V33_WINNER_HLUT
               &parent_planes,
#endif
#if V44_TAIL >= 2
               &v44choice,
#endif
               &u0, &u1);
#else
#if V7_OPENNESS
    // 把象限开阔度并进 qbase 的 bit 11..14。放在这里而不是 qbase 的计算处，
    // 是为了不必把 qbase 移到窗口构造之后 —— 那会打乱现有的指令调度。
    // 每个角色多 8 条 SSE 指令 + 1 次 or + 1 次移位。
    solve_pair(w0, w1,
               _mm_or_si128(qbase0, _mm_slli_epi32(quadrant_openness(w0), OPEN_SHIFT)),
               _mm_or_si128(qbase1, _mm_slli_epi32(quadrant_openness(w1), OPEN_SHIFT)),
#if V33_WINNER_HLUT
               &parent_planes,
#endif
               &u0, &u1);
#else
    solve_pair(w0, w1, qbase0, qbase1,
#if V33_WINNER_HLUT
               &parent_planes,
#endif
               &u0, &u1);
#endif
#endif  // V26_ZMM_PAIR_WINDOW

#if V44_TAIL >= 2
    const KChoice choice = v44choice;
#else
    const KChoice choice = choose_k(u0, u1);
#endif
#if NOREV_PAIR_EMIT
    emit_parent_pair(choice.key0, choice.key1, parent_planes,
                     choice.k, out.actions);
#else
#if V33_WINNER_HLUT
    emit_parent(choice.key0, parent_planes, 0, out.actions);
    emit_parent(choice.key1, parent_planes, 4, out.actions + choice.k);
#else
    emit_parent(choice.key0, u0.parent_mask, out.actions);
    emit_parent(choice.key1, u1.parent_mask, out.actions + choice.k);
#endif
#endif
    // LUT 发射固定写 4 个 int，角色 1 那次可能盖到 k/order 上，
    // 所以这两个字段必须在发射之后写。
    out.k = choice.k;
    out.order = 0;
#if V46_DEADINIT
    // 原来由聚合初始化写的 vp，现在显式写一次。emit 最远只写到 actions[7]
    // （k=4），所以 vp 从来不会被覆盖，位置无关紧要，这里跟着 order 走。
    out.vp = 0;
#endif

    const int dist0 = key_dist(choice.key0);
    const int dist1 = key_dist(choice.key1);
#if NOREV_MASKED_PREV_PAD
    // Already updated from the live winner vector inside solve_pair.
    (void)dist0;
    (void)dist1;
#elif NOREV_PACKED_PREV
    // Semantics are exactly the two conditional stores below, but both bytes share
    // one mask and one store.  __builtin_memcpy keeps the operation alias-safe while
    // Clang lowers the fixed two-byte copies to ordinary scalar loads/stores.
    uint16_t old_prev;
    __builtin_memcpy(&old_prev, state.prev_quadrant, sizeof(old_prev));
    const uint32_t qpair =
        static_cast<uint32_t>(key_quadrant(choice.key0)) |
        (static_cast<uint32_t>(key_quadrant(choice.key1)) << 8);
    const uint32_t moved_mask =
        ((0u - static_cast<uint32_t>(dist0 != 0)) & 0x00FFu) |
        ((0u - static_cast<uint32_t>(dist1 != 0)) & 0xFF00u);
    const uint16_t next_prev = static_cast<uint16_t>(
        (static_cast<uint32_t>(old_prev) & ~moved_mask) |
        (qpair & moved_mask));
    __builtin_memcpy(state.prev_quadrant, &next_prev, sizeof(next_prev));
#elif V7_FAST_SELECT
    // 无条件写 + cmov，去掉两个数据相关分支
    state.prev_quadrant[0] = dist0 > 0
        ? static_cast<signed char>(key_quadrant(choice.key0)) : state.prev_quadrant[0];
    state.prev_quadrant[1] = dist1 > 0
        ? static_cast<signed char>(key_quadrant(choice.key1)) : state.prev_quadrant[1];
#else
    if (dist0 > 0) state.prev_quadrant[0] = static_cast<signed char>(key_quadrant(choice.key0));
    if (dist1 > 0) state.prev_quadrant[1] = static_cast<signed char>(key_quadrant(choice.key1));
#endif

#if V7_NOREV_HIGH == 3
    // ---- 干涸计数器（依据见文件头 V7_NOREV_HIGH=3 那一段）----
    //
    // 全在 out 写完之后，所以本回合走哪已经定了，这里只改下一回合的门。
    // 取的量是 choice.key 的金币字段 —— 这条路径本回合规划到的收益，已经算好了，
    // 读它是零成本（V7_RICH 用的是同一个量）。为 0 就是「4 步内没有可达金币」，
    // 它覆盖 5x5 窗口全空，也覆盖「看得见但被墙/炸弹封死」。
    // 全部候选非法时的兜底 key 是大负数，同样归入干涸，正确。
    //
    // 更新写成饱和加：`s = (g > 0) ? 0 : s + (s < V7_NOREV_DRY)`，两条 cmov，
    // 到阈值就停住。既避免 uint8 绕回，又让门的判据退化成一次相等比较。
    {
        const int32_t dry_g0 = choice.key0 >> GOLD_SHIFT;
        const int32_t dry_g1 = choice.key1 >> GOLD_SHIFT;
        state.dry_streak[0] = dry_g0 > 0 ? 0 : static_cast<uint8_t>(
            state.dry_streak[0] + (state.dry_streak[0] < V7_NOREV_DRY));
        state.dry_streak[1] = dry_g1 > 0 ? 0 : static_cast<uint8_t>(
            state.dry_streak[1] + (state.dry_streak[1] < V7_NOREV_DRY));
    }
#endif

#if V49_NOREV_GATE
    // ---- H：干旱计数器 + 表索引偏置（依据见文件头 V49_NOREV_GATE 那一段）----
    //
    // 全在 out 写完之后，所以本回合走哪已经定了；这里只改**下一回合**的门。
    // 取的量是 choice.key 的金币字段 —— 这条路径本回合规划到的收益，已经算好了，
    // 读它零成本（V7_ZONE / V7_RICH 用的是同一个量）。为 0 就是「4 步内没有可达
    // 金币」，它同时覆盖 5x5 窗口全空与「看得见但被墙/炸弹封死」。
    // 全部候选非法时的兜底 key 是大负数，同样归入干旱，正确。
    //
    // 更新写成饱和加：`s = (g > 0) ? 0 : s + (s < V49_DRY)`，两条 cmov，到阈值停住。
    // 既避免 uint8 绕回，又让门的判据退化成一次相等比较。
    // 紧接着把门**物化成索引偏置**存进 tail_bias，于是下一回合的热路径不需要
    // 任何门计算：那次加法折进 QB16 的寻址里。
    {
        const int32_t dry_g0 = choice.key0 >> GOLD_SHIFT;
        const int32_t dry_g1 = choice.key1 >> GOLD_SHIFT;
        const uint8_t s0 = dry_g0 > 0 ? 0 : static_cast<uint8_t>(
            state.dry_streak[0] + (state.dry_streak[0] < V49_DRY));
        const uint8_t s1 = dry_g1 > 0 ? 0 : static_cast<uint8_t>(
            state.dry_streak[1] + (state.dry_streak[1] < V49_DRY));
        state.dry_streak[0] = s0;
        state.dry_streak[1] = s1;
#if V49_FORCE_GATE_OFF
        const uint8_t bias0 = 0, bias1 = 0;
#else
        const uint8_t bias0 = (s0 == V49_DRY) ? uint8_t(QB16_GATE_STRIDE) : uint8_t(0);
        const uint8_t bias1 = (s1 == V49_DRY) ? uint8_t(QB16_GATE_STRIDE) : uint8_t(0);
#endif
        state.tail_bias[0] = bias0;
        state.tail_bias[1] = bias1;
#if V49_ROWFOLD
        // I：把整个行号预存掉。prev_quadrant 在上面几行刚更新完，所以这里读到的
        // 就是**下一回合**要用的那个值 —— 与不折叠时热路径读它的语义完全一致。
        // 低两位恒为 0（(pq+1)*4 与 20 都是 4 的倍数），所以下一回合的
        // `| preferred_quadrant` 与 `+` 等价。
        static_assert(QB16_GATE_STRIDE % 4 == 0,
                      "门偏置必须是 4 的倍数，否则它会撞进 preferred_quadrant 的两位");
        static_assert((3 + 1) * 4 + QB16_GATE_STRIDE + 3 < QB16_ENTRIES,
                      "折叠后的行号加上 preferred 必须仍落在表内");
        state.qb16_row[0] = static_cast<uint8_t>(
            (state.prev_quadrant[0] + 1) * 4 + bias0);
        state.qb16_row[1] = static_cast<uint8_t>(
            (state.prev_quadrant[1] + 1) * 4 + bias1);
#endif
    }
#endif

#if V7_SNAPSHOT && V7_SNAP_DEFERRED
    // 本回合的输出已经算完，这里读快照只影响**下一回合**的目标点。
    // 快照本身描述的就是过去几轮的统计，晚一回合生效可以忽略；
    // 换来的是它那 3 次冷 cache line miss 完全不在关键路径上。
    if (__builtin_expect(input->snapshot_valid != 0, 0)) apply_snapshot(input->snapshot);
#endif

#if V7_ZONE
    // ---- 区域产出记忆（设计与依据见文件头 V7_ZONE 处的注释）----
    //
    // 全在 out 写完之后，所以不影响本回合走哪，只影响后续的方向偏好。
    //
    // 热路径这里只做两次累加：choice.key 的金币字段就是这条路径本回合的收益，
    // 已经算好了，取它是零成本。区索引、EMA 读改写、挑区全部推迟到
    // settle_zones（每 V7_ZONE_PERIOD 回合一次的 cold 函数）。
    // 累积上限：单回合最多约 532，32 回合 = 17024，不越 uint16。
    {
        const int32_t g0 = choice.key0 >> GOLD_SHIFT;
        const int32_t g1 = choice.key1 >> GOLD_SHIFT;
        // 全部候选非法时的兜底 key 是大负数，按 0 收益处理
        state.acc_gold[0] = static_cast<uint16_t>(
            state.acc_gold[0] + (g0 > 0 ? g0 : 0));
        state.acc_gold[1] = static_cast<uint16_t>(
            state.acc_gold[1] + (g1 > 0 ? g1 : 0));
    }
    if (__builtin_expect((input->round & (V7_ZONE_PERIOD - 1)) == 0, 0)) {
        settle_zones(pos0, pos1);
    }
#endif

#if V7_RICH
    // ---- 富点记忆（依据与设计见文件头 V7_RICH 处的注释）----
    //
    // 位置在 out 全部写完之后：本回合走哪已经定了，这里只改持久状态，
    // 也就是下一回合的方向偏好。所以哪怕这几条指令花掉几纳秒，
    // 影响的也只是本回合的 cost 排序，不会改变本回合的动作。
    //
    // 观测：本回合持币 - 上回合持币 = 上一回合的净收益。拾取之后角色就站在
    // 被拾取的那一格上，所以此刻的 pos_k 正是那个堆的坐标 —— 不需要任何
    // 额外的坐标推导。my_units_gold 紧跟 my_units，同一条 cache line。
    // 踩踏扣款只会让增量变小甚至为负，天然不会被误记成富点。
    const int32_t hold0 = input->my_units_gold[0];
    const int32_t hold1 = input->my_units_gold[1];
    const int32_t gain0 = hold0 - state.prev_gold[0];
    const int32_t gain1 = hold1 - state.prev_gold[1];
    state.prev_gold[0] = static_cast<int16_t>(hold0);
    state.prev_gold[1] = static_cast<int16_t>(hold1);

    // 记住：直接把方向目标钉在这一格。下一回合的 qbase 立刻朝它排序，
    // 不用多等一个回合 —— 空回合平均只有 1.88 个回合长，等一回合就等于
    // 放弃一半机会。
    // target 只在「候选路径金币完全相同」时才参与排序（它落在 key 的
    // bit 4..6，被金币字段压着），所以把它从中心改成富点，对能拿到金币的
    // 回合几乎没有影响。
    if (__builtin_expect(gain0 >= V7_RICH_GAIN, 0)) {
        state.target_row[0] = static_cast<signed char>(pos0.row);
        state.target_col[0] = static_cast<signed char>(pos0.col);
    }
    if (__builtin_expect(gain1 >= V7_RICH_GAIN, 0)) {
        state.target_row[1] = static_cast<signed char>(pos1.row);
        state.target_col[1] = static_cast<signed char>(pos1.col);
    }

    // 遗忘：这个角色本回合一枚金币都拿不到（选中 key 的金币字段为 0；
    // 全部候选非法时的兜底 key 是大负数，也归到这一类 —— 那种回合走哪都
    // 无所谓），而且目标已经进了它的 5x5 视野。既然看得见还是空的，
    // 那个堆确实没了：退回中心，免得角色守着一个空格来回打转。
    //
    // 两处省指令的写法：
    //   * `(unsigned)(d + 2) <= 4` 就是 `|d| <= 2`，不需要 abs；
    //     用切比雪夫距离而不是曼哈顿，语义上也更准 —— 判据本来就是
    //     「它进没进 5x5 视野」。
    //   * 目标已经是中心时这段是幂等的（写回同样的值），所以不必先判断
    //     「到底有没有富点」，省掉一次比较和一个分支。
    if (__builtin_expect(((choice.key0 >> GOLD_SHIFT) <= 0) &
            (static_cast<unsigned>(pos0.row - state.target_row[0] + 2) <= 4u) &
            (static_cast<unsigned>(pos0.col - state.target_col[0] + 2) <= 4u), 0)) {
        state.target_row[0] = 8;
        state.target_col[0] = 8;
    }
    if (__builtin_expect(((choice.key1 >> GOLD_SHIFT) <= 0) &
            (static_cast<unsigned>(pos1.row - state.target_row[1] + 2) <= 4u) &
            (static_cast<unsigned>(pos1.col - state.target_col[1] + 2) <= 4u), 0)) {
        state.target_row[1] = 8;
        state.target_col[1] = 8;
    }
#endif
    return out;
}
