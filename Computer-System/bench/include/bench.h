#ifndef BENCH_H
#define BENCH_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * ROI (Region of Interest) 标记
 * ---------------------------------------------------------------------------
 * 一个静态链接的 C 程序，启动时 libc 要跑掉几十万条指令；如果直接统计整个
 * 程序，这些指令会稀释掉内核的测量结果。gem5 提供了"伪指令"(m5ops)让程序
 * 自己告诉模拟器"从这里开始统计""到这里为止"。
 *
 * 通常的做法是链接 gem5 的 libm5.a，但那需要 RISC-V 的 C++ 交叉编译器。
 * 这里改成直接用内联汇编发射那条伪指令，好处有两个：
 *   1. 只需要 C 交叉编译器，实验环境更容易搭
 *   2. 你在第 5 次实验里要自己造一条指令，用的正是这个技术
 *
 * 编码规则（见 gem5/util/m5/src/abi/riscv/m5op.S）：
 *   低 7 位 opcode = 0x7b（RISC-V 的 custom-3 编码空间）
 *   第 31:25 位 = 功能号，reset_stats=0x40, dump_reset_stats=0x42
 *   参数通过 a0、a1 传入，返回值写回 a0
 *
 *   0x40 << 25 | 0x7b = 0x8000007b
 *   0x42 << 25 | 0x7b = 0x8400007b
 *
 * 两个标记都用 dump_reset(0x42) 而不是 reset(0x40)，于是 stats.txt 会分成
 * 三段，启动开销自己就显形了：
 *
 *   第 0 段 = 程序开始 -> ROI_BEGIN   启动段（libc 初始化等）
 *   第 1 段 = ROI_BEGIN -> ROI_END    ★ 内核，这才是要测的
 *   第 2 段 = ROI_END -> 程序退出     收尾段
 *
 * 这样量化"启动占了多少"只需要跑一次、用同一个二进制，不必去跟另一个
 * 编译出来的版本相减——两个二进制的代码布局不同，相减会引入伪差异。
 */
#if defined(__riscv) && !defined(NO_M5OPS)

#define M5OP_RAW(encoding)                                                     \
  do {                                                                         \
    register uint64_t _a0 __asm__("a0") = 0;                                   \
    register uint64_t _a1 __asm__("a1") = 0;                                   \
    __asm__ __volatile__(".long " #encoding                                    \
                         : "+r"(_a0)                                           \
                         : "r"(_a1)                                            \
                         : "memory");                                          \
  } while (0)

/* 转储启动段的统计并清零：测量区间从此刻开始 */
#define ROI_BEGIN() M5OP_RAW(0x8400007b)
/* 转储 ROI 的统计并清零：第 1 段就是纯粹的内核 */
#define ROI_END() M5OP_RAW(0x8400007b)

#else /* 本机(x86)编译时标记退化为空操作，同一份源码可以用 perf 测真机 */

#define ROI_BEGIN()                                                            \
  do {                                                                         \
  } while (0)
#define ROI_END()                                                              \
  do {                                                                         \
  } while (0)

#endif

/* 确定性随机数：每个学生一个种子，代码路径相同但数据不同 */
typedef struct {
  uint32_t s;
} rng_t;

static inline void rng_init(rng_t *r, uint32_t seed) { r->s = seed ? seed : 1u; }

static inline uint32_t rng_u32(rng_t *r) {
  r->s ^= r->s << 13;
  r->s ^= r->s >> 17;
  r->s ^= r->s << 5;
  return r->s;
}

static inline int8_t rng_i8(rng_t *r) {
  return (int8_t)((rng_u32(r) & 0xff) - 128);
}

static inline uint32_t bench_seed(int argc, char **argv, int pos) {
  if (argc > pos)
    return (uint32_t)strtoul(argv[pos], NULL, 0);
  return 20260101u;
}

#endif /* BENCH_H */
