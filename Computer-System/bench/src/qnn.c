/*
 * qnn - quantized MLP inference, the workload that runs through all 7 labs.
 *
 * 784 -> 128 -> 10, int8 weights, int32 accumulation. ~101.6K MACs per sample.
 * Weight matrix W1 is 98 KB: bigger than a typical L1, smaller than a typical
 * L2, which is what makes the lab-2 cache sweep produce a visible knee.
 *
 * The whole kernel funnels through dot_i8(), which is the single hot loop that
 * lab 5 replaces with a custom instruction and lab 6 offloads to an accelerator.
 */
#include "bench.h"

#define IN_DIM 784
#define HID_DIM 128
#define OUT_DIM 10

static int8_t w1[HID_DIM][IN_DIM];
static int8_t w2[OUT_DIM][HID_DIM];
static int32_t b1[HID_DIM];
static int32_t b2[OUT_DIM];
#ifndef QNN_ACCEL /* 卸载给加速器之后，这三个缓冲被 xbat/hbat/acc2 取代 */
static int8_t xbuf[IN_DIM];
static int8_t hbuf[HID_DIM];
static int32_t ybuf[OUT_DIM];
#endif

/*
 * DOTP8_ACC selects which implementation of dot_i8() gets compiled:
 *   0  baseline scalar loop (default -- byte-identical to labs 1-4)
 *   1  one dotp8 per 8 MACs, single accumulator
 *   2  two independent accumulators
 *   4  four independent accumulators
 *   9  YOUR TURN -- the version you write in lab 5 task 4
 * DOTP8_UNROLL4 additionally forces a 4x unrolled body that still funnels
 * everything into ONE accumulator. It is the control group for the 4-accumulator
 * version: same instruction count, same memory traffic, chain still intact.
 * Every variant must print the same checksum. That is the honesty check.
 */
#ifndef DOTP8_ACC
#define DOTP8_ACC 0
#endif

#if DOTP8_ACC > 0
/* may_alias makes "read an int8_t array through an int64_t lvalue" legal under
   C11 6.5p7. It does NOT change the code GCC emits here -- the loop only reads,
   so strict aliasing gives the optimizer nothing to exploit. Verified: with the
   attribute, without it, and with -fno-strict-aliasing all produce byte-identical
   loops. Lab 5 task 4c has students reproduce that diff themselves. */
typedef int64_t i64_a __attribute__((may_alias));

/* .insn r <opcode>, <funct3>, <funct7>, rd, rs1, rs2
   0x0b = custom-0. "+r" makes acc both source and destination. */
#define DOTP8(acc, va, vb)                                                     \
  asm volatile(".insn r 0x0b, 0, 0, %0, %1, %2" : "+r"(acc) : "r"(va), "r"(vb))
#endif

/*
 * THE hot loop. Kept in its own non-inlined function on purpose: students need
 * a single, obvious place to swap in the custom instruction in lab 5.
 */
#if DOTP8_ACC == 0
__attribute__((noinline)) int32_t dot_i8(const int8_t *a, const int8_t *b,
                                         int n) {
  int32_t acc = 0;
  for (int i = 0; i < n; i++)
    acc += (int32_t)a[i] * (int32_t)b[i];
  return acc;
}

#elif DOTP8_ACC == 9
/* ===================== 实验五 任务 4：你来写这一段 =====================
   目标：用 dotp8 把下面这个循环换掉，一条指令干 8 次乘加。
   可用工具：上面已经给好的 i64_a 类型和 DOTP8(acc, va, vb) 宏。
   注意 n 一定是 8 的倍数（IN_DIM=784，HID_DIM=128），不用写尾巴。
   写完先跑 checksum 对照，对不上就别往下测。
   ====================================================================== */
__attribute__((noinline)) int32_t dot_i8(const int8_t *a, const int8_t *b,
                                         int n) {
  int32_t acc = 0;
  for (int i = 0; i < n; i++)
    acc += (int32_t)a[i] * (int32_t)b[i];
  return acc;
}

#elif defined(DOTP8_UNROLL4)
__attribute__((noinline)) int32_t dot_i8(const int8_t *a, const int8_t *b,
                                         int n) {
  const i64_a *pa = (const i64_a *)a;
  const i64_a *pb = (const i64_a *)b;
  int64_t a0 = 0;
  int m = n / 8, i = 0;
  for (; i + 4 <= m; i += 4) {
    DOTP8(a0, pa[i + 0], pb[i + 0]);
    DOTP8(a0, pa[i + 1], pb[i + 1]);
    DOTP8(a0, pa[i + 2], pb[i + 2]);
    DOTP8(a0, pa[i + 3], pb[i + 3]);
  }
  for (; i < m; i++)
    DOTP8(a0, pa[i], pb[i]);
  return (int32_t)a0;
}

#elif DOTP8_ACC == 1
__attribute__((noinline)) int32_t dot_i8(const int8_t *a, const int8_t *b,
                                         int n) {
  const i64_a *pa = (const i64_a *)a;
  const i64_a *pb = (const i64_a *)b;
  int64_t acc = 0;
  for (int i = 0; i < n / 8; i++)
    DOTP8(acc, pa[i], pb[i]);
  return (int32_t)acc;
}

#elif DOTP8_ACC == 2
__attribute__((noinline)) int32_t dot_i8(const int8_t *a, const int8_t *b,
                                         int n) {
  const i64_a *pa = (const i64_a *)a;
  const i64_a *pb = (const i64_a *)b;
  int64_t a0 = 0, a1 = 0;
  int m = n / 8, i = 0;
  for (; i + 2 <= m; i += 2) {
    DOTP8(a0, pa[i + 0], pb[i + 0]);
    DOTP8(a1, pa[i + 1], pb[i + 1]);
  }
  for (; i < m; i++)
    DOTP8(a0, pa[i], pb[i]);
  return (int32_t)(a0 + a1);
}

#elif DOTP8_ACC == 4 && defined(DOTP8_COMPACT)
/* Same 4 accumulators, same unroll, same memory. The only difference is that
   the loop is driven by pointer compares instead of an index, which drops 3
   overhead instructions (6 bytes) -- just enough for the body to fit inside
   one 64-byte fetch block. */
__attribute__((noinline)) int32_t dot_i8(const int8_t *a, const int8_t *b,
                                         int n) {
  const i64_a *pa = (const i64_a *)a;
  const i64_a *pb = (const i64_a *)b;
  const i64_a *end = pa + n / 8;
  const i64_a *end4 = pa + (n / 8 & ~3);
  int64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  while (pa < end4) {
    DOTP8(a0, pa[0], pb[0]);
    DOTP8(a1, pa[1], pb[1]);
    DOTP8(a2, pa[2], pb[2]);
    DOTP8(a3, pa[3], pb[3]);
    pa += 4;
    pb += 4;
  }
  while (pa < end) {
    DOTP8(a0, pa[0], pb[0]);
    pa++;
    pb++;
  }
  return (int32_t)(a0 + a1 + a2 + a3);
}

#elif DOTP8_ACC == 4
__attribute__((noinline)) int32_t dot_i8(const int8_t *a, const int8_t *b,
                                         int n) {
  const i64_a *pa = (const i64_a *)a;
  const i64_a *pb = (const i64_a *)b;
  int64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  int m = n / 8, i = 0;
  for (; i + 4 <= m; i += 4) {
    DOTP8(a0, pa[i + 0], pb[i + 0]);
    DOTP8(a1, pa[i + 1], pb[i + 1]);
    DOTP8(a2, pa[i + 2], pb[i + 2]);
    DOTP8(a3, pa[i + 3], pb[i + 3]);
  }
  for (; i < m; i++)
    DOTP8(a0, pa[i], pb[i]);
  return (int32_t)(a0 + a1 + a2 + a3);
}

#else
#error "DOTP8_ACC must be 0, 1, 2 or 4"
#endif

/* ===========================================================================
 * 实验六：整层卸载给加速器
 *
 * 打开 QNN_ACCEL 之后，dot_i8 那个循环在 CPU 上彻底消失了：CPU 只写 5 个
 * MMIO 寄存器下命令，然后读 STATUS 等结果。和实验五的对比是本次实验的主线——
 *   实验五：CPU 用 load 把 98 KB 权重搬进寄存器，再用 dotp8 算
 *   实验六：CPU 一个字节都不搬，加速器自己去内存里取
 *
 * QNN_BATCH 一次交给加速器几个样本。它是本次实验最重要的软件旋钮：
 * batch=1 时每算 1 次乘加就要搬 1 个字节权重（算术强度 1），
 * batch=B 时同一份权重被 B 个样本共用，算术强度变成 B。
 * ========================================================================= */
#ifdef QNN_ACCEL

/* 加速器 MMIO 寄存器在进程地址空间里的位置。必须和 cssys_se.py 里
   --accel-addr 的取值一致，否则一访问就是段错误。 */
#ifndef ACCEL_BASE
#define ACCEL_BASE 0x40000000UL
#endif

#ifndef QNN_BATCH
#define QNN_BATCH 1
#endif

#define ACC_REG(i) (((volatile uint64_t *)(uintptr_t)ACCEL_BASE)[i])
enum {
  ACC_A = 0,      /* 矩阵 A 的地址 */
  ACC_B = 1,      /* 输入向量（batch 个）的地址 */
  ACC_OUT = 2,    /* int32 结果数组的地址 */
  ACC_DIMS = 3,   /* (rows << 32) | n */
  ACC_GO = 4,     /* 写它就开始算 */
  ACC_STATUS = 5, /* 读它会一直挂到算完 */
  ACC_BATCH = 6
};

static int8_t xbat[QNN_BATCH][IN_DIM];
static int8_t hbat[QNN_BATCH][HID_DIM];
static int32_t acc1[QNN_BATCH][HID_DIM];
static int32_t acc2[QNN_BATCH][OUT_DIM];

/* 下一条命令：out[s][r] = sum_i A[r][i] * B[s][i] */
__attribute__((noinline)) static void accel_layer(const int8_t *A,
                                                  const int8_t *B, int32_t *out,
                                                  int rows, int n, int batch) {
  ACC_REG(ACC_A) = (uint64_t)(uintptr_t)A;
  ACC_REG(ACC_B) = (uint64_t)(uintptr_t)B;
  ACC_REG(ACC_OUT) = (uint64_t)(uintptr_t)out;
  ACC_REG(ACC_DIMS) = ((uint64_t)rows << 32) | (uint32_t)n;
  ACC_REG(ACC_BATCH) = (uint64_t)batch;
  ACC_REG(ACC_GO) = 1;

  /* ---- 一条不能省的 fence ------------------------------------------------
     volatile 只管住编译器，管不住硬件。在乱序核上，读 STATUS 和写 GO 是
     两个不同的地址(0x28 和 0x20)，LSQ 的访存序违例检查只看地址是否重叠，
     所以它完全允许这条 load 投机到那条 store 前面去执行。一旦发生，读到的
     就是"加速器还没开始，不忙"，同步被静默跳过，下一条命令直接撞上还在算
     的加速器——gem5 里表现为 dotp8_accel.cc 的 "上一个作业没做完" panic。

     真实驱动里这一句就是 wmb()/mb()。RISC-V 的 fence 在 gem5 的 O3 模型里
     带 IsReadBarrier|IsWriteBarrier，会让访存相关性单元把后面所有 load 都
     挂到它后面，正好是我们要的语义。
     （另一半保险在硬件侧：TLB 必须把 MMIO 页标成 UNCACHEABLE|STRICT_ORDER。
       两边都要做——这是实验六任务 1 的核心结论。）

     用 -DQNN_NO_FENCE 编译可以把这一句去掉，专门用来复现"少了它会怎样"。 */
#ifndef QNN_NO_FENCE
  __asm__ __volatile__("fence ow, ir" ::: "memory");
#endif

  /* 这一条 load 是全部的同步机制：加速器不算完就不回复，
     这条 uncacheable load 就一直卡在 CPU 的访存队列头上出不去。
     换句话说，卸载期间 CPU 是完全空转的——这正是任务 5 要量化的代价。 */
  volatile uint64_t st = ACC_REG(ACC_STATUS);
  (void)st;
}
#endif /* QNN_ACCEL */

static int8_t requantize(int32_t acc) {
  acc >>= 10; /* fixed-point rescale */
  if (acc < 0)
    acc = 0; /* ReLU */
  if (acc > 127)
    acc = 127;
  return (int8_t)acc;
}

static void init_weights(uint32_t seed) {
  rng_t r;
  rng_init(&r, seed);
  for (int j = 0; j < HID_DIM; j++) {
    for (int i = 0; i < IN_DIM; i++)
      w1[j][i] = rng_i8(&r);
    b1[j] = (int32_t)(rng_u32(&r) % 2048) - 1024;
  }
  for (int j = 0; j < OUT_DIM; j++) {
    for (int i = 0; i < HID_DIM; i++)
      w2[j][i] = rng_i8(&r);
    b2[j] = (int32_t)(rng_u32(&r) % 2048) - 1024;
  }
}

#ifndef QNN_ACCEL
static int infer(void) {
  for (int j = 0; j < HID_DIM; j++)
    hbuf[j] = requantize(dot_i8(w1[j], xbuf, IN_DIM) + b1[j]);
  for (int j = 0; j < OUT_DIM; j++)
    ybuf[j] = dot_i8(w2[j], hbuf, HID_DIM) + b2[j];

  int best = 0;
  for (int j = 1; j < OUT_DIM; j++)
    if (ybuf[j] > ybuf[best])
      best = j;
  return best;
}
#endif

#ifdef QNN_ACCEL
/* 一次推理 nb 个样本，两层都交给加速器。
   CPU 这边只剩下重量化(requantize)和取最大值——它们会在任务 5 里
   摇身一变成为新的瓶颈，这就是 Amdahl 定律最直观的一次现身。 */
static void infer_batch(int nb, int *cls_out) {
  accel_layer(&w1[0][0], &xbat[0][0], &acc1[0][0], HID_DIM, IN_DIM, nb);
  for (int b = 0; b < nb; b++)
    for (int j = 0; j < HID_DIM; j++)
      hbat[b][j] = requantize(acc1[b][j] + b1[j]);

  accel_layer(&w2[0][0], &hbat[0][0], &acc2[0][0], OUT_DIM, HID_DIM, nb);
  for (int b = 0; b < nb; b++) {
    int best = 0;
    for (int j = 0; j < OUT_DIM; j++)
      acc2[b][j] += b2[j];
    for (int j = 1; j < OUT_DIM; j++)
      if (acc2[b][j] > acc2[b][best])
        best = j;
    cls_out[b] = best;
  }
}
#endif

int main(int argc, char **argv) {
  int samples = (argc > 1) ? atoi(argv[1]) : 16;
  uint32_t seed = bench_seed(argc, argv, 2);

  init_weights(seed);

  rng_t r;
  rng_init(&r, seed ^ 0x5a5a5a5au);
  uint32_t checksum = 0;
  int hist[OUT_DIM] = {0};

#ifdef QNN_ACCEL
  /* 攒够 QNN_BATCH 个样本再下一条命令。样本的产生顺序、每个样本得到的
     随机数、最后的 checksum 都和基线一模一样——只是权重少搬了 batch-1 遍。 */
  int cls_buf[QNN_BATCH];
  ROI_BEGIN();
  for (int s = 0; s < samples; s += QNN_BATCH) {
    int nb = (samples - s < QNN_BATCH) ? (samples - s) : QNN_BATCH;
    for (int b = 0; b < nb; b++)
      for (int i = 0; i < IN_DIM; i++)
        xbat[b][i] = rng_i8(&r);
    infer_batch(nb, cls_buf);
    for (int b = 0; b < nb; b++) {
      hist[cls_buf[b]]++;
      checksum = checksum * 31u + (uint32_t)cls_buf[b];
    }
  }
  ROI_END();
#else
  ROI_BEGIN();
  for (int s = 0; s < samples; s++) {
    for (int i = 0; i < IN_DIM; i++)
      xbuf[i] = rng_i8(&r);
    int cls = infer();
    hist[cls]++;
    checksum = checksum * 31u + (uint32_t)cls;
  }
  ROI_END();
#endif

  printf("qnn samples=%d seed=0x%08x checksum=0x%08x\n", samples, seed,
         checksum);
  printf("class histogram:");
  for (int j = 0; j < OUT_DIM; j++)
    printf(" %d", hist[j]);
  printf("\n");
  return 0;
}
