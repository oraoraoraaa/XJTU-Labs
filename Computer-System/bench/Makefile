# 计算机系统综合设计实验 —— 基准程序构建
#
#   make          交叉编译出 RISC-V 静态可执行文件到 bin/riscv/（自带 ROI 标记）
#   make noroi    同一份源码但去掉 ROI 标记，第 1 次实验用来量化"启动开销占多少"
#   make noalign  故意关掉对齐参数的版本，第 1 次实验用来演示对齐悬崖
#   make native   用同一份源码编译本机(x86)版本，供第 1 次实验和真机 perf 对比
#   make clean
#
# ROI 标记通过内联汇编发射 gem5 伪指令，只在 __riscv 目标下生效，
# 所以本机版本不需要任何改动就能直接跑。

CROSS ?= riscv64-linux-gnu-

BENCHES := qnn gemm pchase branchy stream

# 对齐参数不是可选项，是测量前提。
#
# 实测：qnn 的热函数 dot_i8 只要在 .text 里挪动 16 字节，落到取指块边界的
# 另一侧，整个程序就会慢 3 倍（937M tick -> 2866M tick）——指令数、访存流量、
# 数据地址全都一模一样。O3 的取指单元一次只能取一个 64B 块，循环体一旦跨界
# 就要多取一次。
#
# 本课程要测的是"改了某个设计参数带来多少收益"，绝不能让这种和设计决策无关
# 的运气因素混进来。强制把函数和循环对齐到 64B，热点就永远从取指块开头开始。
ALIGN := -falign-functions=64 -falign-loops=64 -falign-jumps=64

CFLAGS_COMMON := -O2 -Wall -Iinclude $(ALIGN)

RV_CC     := $(CROSS)gcc
RV_FLAGS  := $(CFLAGS_COMMON) -static
RV_NOALIGN_FLAGS := -O2 -Wall -Iinclude -static
NAT_CC    := gcc
NAT_FLAGS := $(CFLAGS_COMMON)

RV_BIN  := bin/riscv
NAT_BIN := bin/native

RV_TARGETS    := $(addprefix $(RV_BIN)/,$(BENCHES))
NOROI_TARGETS := $(addsuffix _noroi,$(RV_TARGETS))
# 对齐悬崖的演示只用 qnn 一个程序就够了
NOALIGN_TARGETS := $(RV_BIN)/qnn_noalign $(RV_BIN)/qnn_noalign_noroi
NAT_TARGETS   := $(addprefix $(NAT_BIN)/,$(BENCHES))

# 第 5 次实验：同一份 qnn.c，靠 -DDOTP8_ACC=N 换掉 dot_i8 的实现。
#   N=1 一条 dotp8 抵 8 次乘加，但只有一个累加器（会踩到循环携带依赖）
#   N=2/4 拆成多个独立累加器，用软件手段把那条依赖链断开
# qnn_dotp8u4 是 qnn_dotp8x4 的对照组：同样展开 4 次、指令数几乎相同，
# 但仍然只有一个累加器。有它才能把"展开省下的指令"和"断开的依赖链"分开。
# 基线仍然是 bin/riscv/qnn，一个字节都不许变。
# qnn_dotp8x4c 和 qnn_dotp8x4 的指令数、访存量、累加器个数全都一样，
# 只是循环体从 66 字节缩到了 64 字节以内 —— 用来演示"取指块装不下"的代价。
DOTP8_TARGETS := $(RV_BIN)/qnn_dotp8 $(RV_BIN)/qnn_dotp8u4 \
                 $(RV_BIN)/qnn_dotp8x2 $(RV_BIN)/qnn_dotp8x4 \
                 $(RV_BIN)/qnn_dotp8x4c

# 第 6 次实验：同一份 qnn.c，靠 -DQNN_ACCEL 把整层运算交给加速器。
# CPU 只写 MMIO 寄存器下命令，一个权重字节都不用自己搬。
# -DQNN_BATCH=N 表示一次交给加速器 N 个样本：N 个样本共用同一份权重，
# 算术强度(乘加/字节)就从 1 涨到 N —— 这是本次实验唯一能突破带宽墙的旋钮。
# 四个版本的 checksum 必须互相一致，也必须和基线 qnn 一致。
# qnn_accel_nofence 是任务 1 的"反面教材"：只去掉写 GO 和读 STATUS 之间的
# 那一条 fence，别的一个字节不改。在乱序核上它会把加速器打崩——
# 用来证明"volatile 管不住硬件"。
ACCEL_TARGETS := $(RV_BIN)/qnn_accel $(RV_BIN)/qnn_accel_b2 \
                 $(RV_BIN)/qnn_accel_b4 $(RV_BIN)/qnn_accel_b8 \
                 $(RV_BIN)/qnn_accel_nofence

.PHONY: all noroi noalign native dotp8 dotp8my accel clean
all: $(RV_TARGETS)
noroi: $(NOROI_TARGETS)
noalign: $(NOALIGN_TARGETS)
native: $(NAT_TARGETS)
dotp8: $(DOTP8_TARGETS)
dotp8my: $(RV_BIN)/qnn_dotp8my
accel: $(ACCEL_TARGETS)

$(RV_BIN)/qnn_accel: src/qnn.c include/bench.h | $(RV_BIN)
	$(RV_CC) $(RV_FLAGS) -DQNN_ACCEL -o $@ $<

$(RV_BIN)/qnn_accel_b2: src/qnn.c include/bench.h | $(RV_BIN)
	$(RV_CC) $(RV_FLAGS) -DQNN_ACCEL -DQNN_BATCH=2 -o $@ $<

$(RV_BIN)/qnn_accel_b4: src/qnn.c include/bench.h | $(RV_BIN)
	$(RV_CC) $(RV_FLAGS) -DQNN_ACCEL -DQNN_BATCH=4 -o $@ $<

$(RV_BIN)/qnn_accel_b8: src/qnn.c include/bench.h | $(RV_BIN)
	$(RV_CC) $(RV_FLAGS) -DQNN_ACCEL -DQNN_BATCH=8 -o $@ $<

$(RV_BIN)/qnn_accel_nofence: src/qnn.c include/bench.h | $(RV_BIN)
	$(RV_CC) $(RV_FLAGS) -DQNN_ACCEL -DQNN_NO_FENCE -o $@ $<

# 你在任务 4 里自己写的那一版（qnn.c 里的 DOTP8_ACC == 9 分支）。
# 单独一个目标，改完只重编它一个，不碰上面五个参考版本。
$(RV_BIN)/qnn_dotp8my: src/qnn.c include/bench.h | $(RV_BIN)
	$(RV_CC) $(RV_FLAGS) -DDOTP8_ACC=9 -o $@ $<

$(RV_BIN)/qnn_dotp8: src/qnn.c include/bench.h | $(RV_BIN)
	$(RV_CC) $(RV_FLAGS) -DDOTP8_ACC=1 -o $@ $<

$(RV_BIN)/qnn_dotp8u4: src/qnn.c include/bench.h | $(RV_BIN)
	$(RV_CC) $(RV_FLAGS) -DDOTP8_ACC=1 -DDOTP8_UNROLL4 -o $@ $<

$(RV_BIN)/qnn_dotp8x2: src/qnn.c include/bench.h | $(RV_BIN)
	$(RV_CC) $(RV_FLAGS) -DDOTP8_ACC=2 -o $@ $<

$(RV_BIN)/qnn_dotp8x4: src/qnn.c include/bench.h | $(RV_BIN)
	$(RV_CC) $(RV_FLAGS) -DDOTP8_ACC=4 -o $@ $<

$(RV_BIN)/qnn_dotp8x4c: src/qnn.c include/bench.h | $(RV_BIN)
	$(RV_CC) $(RV_FLAGS) -DDOTP8_ACC=4 -DDOTP8_COMPACT -o $@ $<

# GNU make 在多条模式规则里选"stem 最短"的那条，所以下面三条规则
# 对 qnn_noalign_noroi / qnn_noalign / qnn 会各自命中正确的一条。
$(RV_BIN)/%_noalign_noroi: src/%.c include/bench.h | $(RV_BIN)
	$(RV_CC) $(RV_NOALIGN_FLAGS) -DNO_M5OPS -o $@ $<

$(RV_BIN)/%_noalign: src/%.c include/bench.h | $(RV_BIN)
	$(RV_CC) $(RV_NOALIGN_FLAGS) -o $@ $<

$(RV_BIN)/%_noroi: src/%.c include/bench.h | $(RV_BIN)
	$(RV_CC) $(RV_FLAGS) -DNO_M5OPS -o $@ $<

$(RV_BIN)/%: src/%.c include/bench.h | $(RV_BIN)
	$(RV_CC) $(RV_FLAGS) -o $@ $<

$(NAT_BIN)/%: src/%.c include/bench.h | $(NAT_BIN)
	$(NAT_CC) $(NAT_FLAGS) -o $@ $<

$(RV_BIN) $(NAT_BIN):
	mkdir -p $@

clean:
	rm -rf bin
