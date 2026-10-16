# XTrace(arXiv 2609.28769)在 sm_86 上真跑:二进制级 splicing 的保真度、可行性,以及拦在运行期前面的三条实现事实

**一句话总结论**:Xtrace 的公开实现(pip 包 `gwatch` 0.0.35)在本机 RTX 3060(sm_86,论文 Table 2 列出的受支持架构之一)上**能拼**:我们对本机 Triton 缓存里的 8 个 cubin(6 个内核名:vLLM 的 rotary、量化推理栈的 `awq_gemm`/`awq_dequantize`/`_forward`/`_backward`,以及我们自己的一个 matmul)做了 16 次逐站点拼接,总计 578/680 个站点成功,最重的一次把一个 664 条指令的内核拼进 77 个探针站点、新增 1176 条指令,**原文指令逐字保留率 97.1–100%、寄存器数在 16 次里一次都没变**;字节级相似度 82.2–97.4%,与论文的 94–98% 同量级,而少掉的那部分**86% 是控制位**(stall/记分板)被重排,不是分支——这修正了论文"变化的 2–6% 都是分支"的说法。但**跑不起来**:拼接后的 cubin 在真机启动即 `ILLEGAL_ADDRESS`,我做了完整的控制实验把范围逼到一小段地址算术上(见 §3.4),所以本线程交付的是**保真度与可行性的真数据 + 运行期开销的障碍报告**,没有开销数字。

---

## 1. 先讲清论文的机制(不看论文也能读懂)

### 1.1 它要修的是一个观测学问题

GPU 的内核级追踪(intra-kernel tracing)靠往 kernel 里插探针(probe)记录时间戳和运行状态。现在的工具(Neutrino、NVIDIA 官方 IKET)都在**编译前**插桩:探针和内核一起过编译器。两件事随之发生:

1. **编译器优化被打断**。探针把原本跨越区域边界的调度切断(论文 §2.2 的 FA-3 例子:编译器本来把两轮 unrolled SoftMax 交错、把 P-convert 与 SoftMax 重叠,插桩后这两件事都做不成了)。于是你追踪的二进制根本不是 GPU 真正执行的那个——论文用"二进制相似度"(追踪后二进制里与原二进制逐条相同的指令占比)量化,Neutrino 只有 8–20%,IKET 13–48%。
2. **探针抢资源**。探针要寄存器存数据,和内核自己的值抢寄存器文件;探针的 store 还会让后续指令等它的 scoreboard。论文测到 3.8–75.6% 的额外执行时间。开销直接污染被观测的流水线时序。

XTrace 的立场:**探针不该进编译器,应该进已编译好的二进制**(instruction splicing,借自 CPU 动态插桩的术语)。

### 1.2 地基:GPU 怎么排指令(承重概念)

现代 NVIDIA GPU **按序发射**,硬件不做寄存器互锁(不像 CPU 有乱序和重命名)。每一条指令里带若干**控制位(control bits)**,把三类冒险在编译期就解决掉:

- **结构冒险**:两条指令用同一条流水线(pipeline)。用 **stall(停顿计数)** 把它们的发射拉开若干周期。
- **静态数据冒险**:生产者的延迟编译期已知。同样用一个 stall,长度等于生产者延迟。
- **动态数据冒险**:生产者的延迟编译期未知(例如全局 load)。用一个 **scoreboard(记分板)** 登记这次发射,消费者指令在发射时"等第 k 号记分板清零"。每个 warp 的 scoreboard 数量有限(论文说 NVIDIA 是 6 个),编译器必须把它们分配好。

**活性分析(liveness analysis)**:反向数据流分析,对每个寄存器算出它的"活跃区间"——从写入到它最后一次被读。区间之外就是**死区间(dead period)**,这段里该寄存器的值已经没人要了。XTrace 只用死寄存器放探针的数据,所以既不和内核抢寄存器,也不制造新的寄存器依赖。

### 1.3 splicing 要做的三个决定与代价函数

在一个插入地址 `i^(p)`,探针是一小段指令 `I = {i^(1)…i^(7)}` 和它们之间的 RAW 依赖 `E_RAW`。拼接方案(splicing)是三个决定的组合:

- **D1 寄存器分配**:每个探针值 v 分到一个在 `i^(p)` 处已死的寄存器 ρ(v)。
- **D2 指令顺序**:一个尊重 `E_RAW` 的拓扑序 π。
- **D3 控制位**:为探针与内核指令之间新出现的冒险写 stall 与 scoreboard γ。

代价是插入地址的发射时间增量 `o = t[i^(p)] − t0[i^(p)]`(t 为拼接后的发射时间,t0 为原始)。因为动态冒险的实际等待无法静态预测,论文把每个动态依赖按其**微基准测得的完成延迟**折算进来,得到修正代价 **o′ = t′[i^(p)] − t0[i^(p)]**,其中 `t′[i^(k)] = max over i^(j)→i^(k) ( t′[i^(j)] + δ[i^(j)→i^(k)] )`,δ 把 stall 与"实测延迟修正"合在一起。这三个决定在编译器里分属不同 pass、用启发式各做各的(联合空间对整内核太大);但探针很小,论文就**联合枚举**这个空间取 o′ 最小者(Algorithm 1:拓扑序 × 干涉图着色 × 控制位),再给一个贪心版(按 RAW 链长度定价,先发射链最长的)加速。

一句话:**它把"插桩"变成了一次小规模的、有代价函数的二进制重写搜索**,搜索正确性靠活性分析(死寄存器)与编译器同一份冒险表(结构冒险按流水线占用、数据冒险按 pipe-to-pipe 表)。

### 1.4 运行期的那一半

- **INIT 探针**放在内核入口:从参数表取 trace buffer 地址、按线程算自己的槽位、把地址和环形缓冲游标存进寄存器(跨整个内核活着)。
- **RECORD 探针**在站点上记录时间戳/寄存器值并推进游标。
- **driver** 拦截 kernel launch 拿到 launch 规模(grid/block),据此算出 buffer 大小并分配,把 buffer 地址作为**追加的第 17 个参数**传给拼接后的内核,跑完再把 buffer 内容解析成 trace。

论文附录里还有一处细节值得记:时间戳用**每 SM 的时钟**(`%clock`,几十个周期、1 cycle 粒度),而不是 GPU 全局时钟(`%globaltimer`,要几百纳秒、32 ns 粒度、低 32 位容易回绕、要两个寄存器);代价是各 SM 时钟不同源,所以在每个 block 的头尾各读一次全局时钟做校准。

---

## 2. 工具链核对:论文的实现就是 `gwatch` 0.0.35

`pip install gwatch` 装到的就是这套系统(包 METADATA 的作者是论文一作 Zhuobin Huang,官网 g-watch.github.io 即论文脚注地址)。包内组件与论文的对应关系:

| 论文组件 | 包内位置 | 说明 |
| --- | --- | --- |
| binary disassembler(19 架构) | `gwatch/cuda/assemble/`+`common/assemble/`,指令表 `assets/cuda/sass/sm86.bin` 等 20 个架构文件 | `KernelDefSASS.get_list_instruction()` 给出反汇编文本 |
| 活性分析 | `common/assemble/analysis/liveness.py` + C++ `parse_register_liveness` / `get_register_live_intervals` / `collect_allocatable_register_pairs` | 我们实测:`get_live_registers_at_pc(pc, 寄存器类)` 直接给死/活集合 |
| **splicer** | `cuda/xtrace/_sass_instrumenter.py`(编排)+ 二进制里的 C++(`libgwatch_dark_cuda128_x86_64.so`,字符串里带 `runtime_control/kernel.cpp`) | `create_instrument_context(pc)` → `stage_instrument_operator(op, params)` → `finalize()`:算子先在 C++ 里联合调度,再落进内核 |
| 探针库 | `InstrumentOperator` 枚举 13 个算子(`INIT_TRACE_POINTER`/`CAPTURE_TRACE`/`STORE_TRACE_RECORD`/`STORE_TRACE_RECORD_WARP`/`CAPTURE_TRACE_ANCHOR` …) | 论文的 INIT/RECORD 两类探针 |
| runtime trace driver | `cuda/profile/context.py`(`start/stop_tracing_kernel_launch`、`register_trace_kernel_sass`)+ `cuda/xtrace/_buffer.py` | launch 拦截靠胶囊(CUPTI 回调) |
| recorder/trace 解析 | `cuda/xtrace/_schema.py`(记录布局)+ `_sass_instrumenter.collect()` | 逐站点 ID/时间戳/值 |
| **IKET 基线** | `cuda/iket/`(占位符机制、`patch_cubin`、`placeholders_of`) | 论文说"reimplemented",包里确实在 |
| Neutrino 意义上的对照 | `cuda/profile/`(PC sampling / 计数器) | 本机因驱动版本不可用,见 §3.4 |

**包与论文不一致的地方**(都是读代码/跑出来的,不是猜的):

1. **多了论文没写的机制:split splice(存store 下沉)**。`cuda/xtrace/_store_sink.py` 把一个站点拆成"站点处 CAPTURE_TRACE(只读时钟进 carrier 寄存器)+ 在同一个基本块里挑一个**同流水线竞争最小**的位置 STORE_TRACE_RECORD",还额外分配一个 handshake scoreboard 跨轮次保护 carrier。默认 `sass_splice_mode="auto"` 就是"有 carrier 就 split、没有就退回单体 store"。论文 §4.4 只描述单体探针的 stall/scoreboard 搜索,这层是发表版实现里的额外优化。
2. **搜索默认不是"最优"**。`_do_trace_sass` 默认 `schedule_mode="first_feasible"`(先贪心序,不成就换),`least_makespan`(论文 Algorithm 1 的枚举)要显式开,而且内部有 200000 序的上界。
3. **`libgwatch_dark_experimental_*.so` 没随包发布**,于是 `cuda/experimental` 里的 `get_pipeline_occupancy_of_opcode` 恒为 None —— split splice 的 sink 排序退化成"寄存器压力 + 就近",论文式的"按流水线竞争选 sink"实际未生效。
4. **Python 层与二进制有 API 漂移**:例如 `KernelDefSASS.get_register_state_at_pc()`(Python 版)在 .so 里不存在(应走 `get_live_registers_at_pc`)、`KernelDefSASS.exclude_instrument_registers` 只存在于 C++ 对象。用包自带的 Python 封装直接调会 AttributeError。
5. **AMD 侧不在这一构建里**:wheel 只带 `libgwatch_dark_cuda128_x86_64.so`(CUDA 12.8 / x86_64),论文的 CDNA/RDNA 支持不体现在这个 pip 包中;`scheduler/` 里 `import gwatch.libgwatch_scheduler` 也缺模块。

环境本身不需要 CUDA toolkit:gwatch 的原生库要 `libcudart.so.12`、`libcupti.so.12`、`libcheckpoint.so`,前两个由 torch 的 cu128 wheel 带的 `nvidia-*-cu12` 提供,最后一个是 triton 自带的 CUPTI shim。`code/gwenv.sh` 把它们放到 `LD_LIBRARY_PATH` 上。

---

## 3. 我们跑了什么

被测目标是本机 Triton 缓存里的 8 个真实 cubin(sm_86):vLLM 的 rotary embedding kernel,以及一批量化推理 kernel(`awq_gemm`×2、`awq_dequantize`、`_forward`×2、`_backward`)。这些不是玩具:它们是实际推理栈里编译出来的内核,只是规模比论文的 FA-3 小几倍到一个数量级(120–664 条指令 vs 2400+)。

### 3.1 静态拼接与二进制相似度(论文 §5.3 的度量,落在完全不同的内核与架构上)

拼接用工具自己的编排顺序:指针寄存器对在整个活范围内分配(优先 per-warp 的 uniform 寄存器),INIT 探针插在指令 1 之前,各站点探针按 pc **从大到小**依次插入(这样插一条只影响更深的 pc),最后把新镜像序列化、重新反汇编、与原镜像逐条对齐。

| 内核 | 原指令 | 寄存器 | 站点(成功/请求) | 新增探针指令 | 文本相似度 | 字节相似度 | 仅控制位变化 | 变化的块 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `_backward_kernel` | 192 | 28 | 15/24 | 246 | 0.9948 | 0.9115 | 16 | 2/5 |
| `_backward_kernel` | 192 | 28 | 5/5 | 96 | 1.0000 | 0.9740 | 5 | 1/5 |
| `_forward_kernel` | 224 | 34 | 26/28 | 411 | 0.9911 | 0.9018 | 20 | 3/11 |
| `_forward_kernel` | 224 | 34 | 10/12 | 171 | 0.9821 | 0.9464 | 8 | 4/11 |
| `_forward_kernel` | 272 | 38 | 18/34 | 291 | 1.0000 | 0.9596 | 11 | 1/11 |
| `_forward_kernel` | 272 | 38 | 7/16 | 126 | 0.9853 | 0.9669 | 5 | 4/11 |
| `awq_dequantize_kernel` | 552 | 68 | 69/69 | 1056 | 1.0000 | 0.9149 | 47 | 1/5 |
| `awq_dequantize_kernel` | 552 | 68 | 33/33 | 516 | 1.0000 | 0.9638 | 20 | 1/5 |
| `awq_gemm_kernel` | 656 | 56 | 69/82 | 1056 | 0.9710 | 0.8216 | 98 | 13/28 |
| `awq_gemm_kernel` | 656 | 56 | 77/88 | 1176 | 0.9710 | 0.8582 | 74 | 12/28 |
| `awq_gemm_kernel` | 664 | 56 | 70/83 | 1071 | 0.9714 | 0.8238 | 98 | 13/28 |
| `awq_gemm_kernel` | 664 | 56 | 77/88 | 1176 | 0.9714 | 0.8645 | 71 | 12/28 |
| `matmul_kernel`(自测) | 360 | 80 | 41/45 | 636 | 0.9917 | 0.8861 | 38 | 4/10 |
| `matmul_kernel`(自测) | 360 | 80 | 43/52 | 666 | 0.9944 | 0.9111 | 30 | 3/10 |
| `rotary_kernel` | 120 | 25 | 12/15 | 201 | 1.0000 | 0.8667 | 16 | 1/6 |
| `rotary_kernel` | 120 | 25 | 6/6 | 111 | 1.0000 | 0.9500 | 6 | 1/6 |

(同一内核两行 = 两种站点集:全部内存访问 `mem`,和每 8 条一站点 `every:8`。全套原始数字在 `code/results/summary.json`,图在 `code/results/xtrace_sm86.png`。)

**数字怎么读**:**寄存器数在 16 次拼接里一次都没变**(56→56、28→28、34→34、80→80、25→25),对应论文 Table 3 的"没有给内核加一个寄存器"——这条在我们这边也成立,而且是在跨架构、跨内核的前提下成立。文本级相似度 97.1–100%,论文在 FA-3/FA-4/FlashMLA/HSTU 上报 94–98%,我们落在它的上沿;站点级成功率 578/680 = 85%。

**尺寸感**:一条探针的形态是 INIT 序 21 条 SASS 指令(优先 per-warp 的 uniform 寄存器时)+ 每个站点 15–16 条(单体 store;模板里另有 per-thread general 寄存器的变体,站点探针缩到 12 条)。工具自己的搜索报告给出单次拼接的代价:二进制分析中位 **0.5–2.3 ms**、搜索中位 **0.1–0.3 ms**、探针的指令跨度(effective makespan)中位 **23 个周期**、最差 44(这些数字远小于论文报的 0.33–0.68 s / 2.6–4.9 ms,因为我们的内核只有 120–664 条指令,而论文是 2400+ 条的 FA-3/FA-4)。

### 3.2 比论文更严一步:字节级相似度暴露了控制位重排

论文的判据说"指令的操作数、**控制位**或块内位置任一不同就算改变"。但它的读数用的是反汇编文本,而反汇编文本不一定把所有控制位字段都印出来。我们额外做了**逐字节**对齐(16 字节原始编码;16 次拼接里有 8 次做了这一对照)。全部 655 条字节级变化的指令**恰好分解成两类,没有第三类**:

| 类别 | 条数 | 构成 |
| --- | --- | --- |
| 控制流位移被改写(论文说的那一类) | **92** | `BRA` 58、`BSSY` 34 —— 这些指令的**文本也变了** |
| 操作数不变、**只有控制位变了** | **563** | `LDS` 114、`LOP3` 69、`LEA` 60、`ISETP` 48、`IMAD` 44、`IADD3` 38、`ATOMG` 32、`PLOP3` 25、`LDG` 23、`LDSM` 20、`SHF` 14、`CCTL` 10、`MEMBAR` 8、`ERRBAR` 8、`BSYNC` 8、`FMUL` 8、`CS2R` 7、`FSETP` 6、`SHFL` 6 … |

- 两类之和正好等于字节级总数(563 + 92 = 655),也就是说:**除了分支位移,失真的全部就是控制位**。
- 且**控制位重排是主体:占 86.0%**(563/655)。文本相似度 100% 的那些拼接,字节相似度只有 86.7–97.4%;最重的一次(`awq_gemm` mem)一个内核里就有 **98 条**指令只改了控制位。
- **被重排的主要是访存指令周围的 stall/记分板间距**(`LDS`/`LDG`/`ATOMG`/`LDSM` 合计 189 条),而论文说"变化的 2–6% 都是分支指令"只对**文本级**成立。这不否定论文的机制(它本来就允许改写控制位),但**它的解释不完整**:在一台 3060 上,只要内核访存密集,控制位重排就是字节级失真的主体(563 vs 92)。

顺带一个方法论提醒:gwatch 自带的反汇编文本渲染与 `nvdisasm` 不一致(例如它把普通 `STG.E [R20.64]` 渲染成带 `memoryDescriptor[UR…]` 的形式,把 `c[0x0][0x0]` 之类的常量读也印得含糊)。我们两边都用 gwatch 的渲染做相似度(自比自,公平),但在读探针指令、判断地址算术时必须用 `nvdisasm`。

### 3.3 逐地址拼接可行性:论文说"没有一次退回",我们这边不是

论文 §5.5 报"53 次拼接全部靠死寄存器喂饱,没有任何 fallback";我们按**每个候选地址**单独试了一次拼接(每个内核在 step=4 的 40 个地址上,`rotary_kernel` 取了前 30 个;单片 store 探针,寄存器优先 uniform):

| 内核 | 指令数 | 寄存器 | 成功 | 失败都发生在哪一阶段 |
| --- | --- | --- | --- | --- |
| `awq_gemm_kernel` | 664 | 56 | 40/40 | — |
| `awq_dequantize_kernel` | 552 | 68 | 40/40 | — |
| `_forward_kernel` | 224 | 34 | 36/40 | schedule(4):寄存器够但凑不出可行调度 |
| `rotary_kernel` | 120 | 25 | 22/30 | schedule(8) |
| `_backward_kernel` | 192 | 28 | 20/40 | schedule(20) |

**失败的全部是"schedule"而不是"没寄存器"**:到 `finalize()` 时探针指令排不出满足工具的冒险规则(同一个存储顺序×寄存器×控制位的搜索空间)的方案,函数返回 "no feasible schedule?"。寄存器越紧、内核越小的越容易失败(25 寄存器、28 寄存器的两个内核明显差)。论文的 fallback(抬内核寄存器上限 / 退回 per-warp 寄存器)针对的是**寄存器不够**,对这种"调度不可行"没有覆盖——至少这个构建里没有自动降级。

同一内核对不同站点的成功率也不同(表 3.1 的 `7/16` vs `26/28`),说明这不是内核级属性,而是**地址级**属性:拼接能不能成,取决于那一点附近的死寄存器碎片与已有 scoreboard/流水线占用。

### 3.4 端到端跑起来这件事,卡在三层(附证据)

我想拿到两个运行期数字:追踪前后的**行为一致性**(输出是否逐位相同)与**开销**(中位墙钟比)。为此写了从 CUDA driver 直接启动 cubin 的 `cuda_launch.py`(torch/Triton 运行时都不经过),并用 gwatch 自己的 `TraceInstrumenter_SASS` 拼接 + `collect()` 解析。三层障碍:

1. **CUPTI 计数器不可用(环境层,可绕过)**:驱动 615.71.09(CUDA 13.4)比工具链(12.8)新,启动时报 `skipping device 0: CUPTI profiler not supported`。**追踪与二进制分析不受影响**,只是拿不到硬件计数器/PC sampling,所以论文里"探针自身的依赖等待占比"这类读数在我们这边做不了。
2. **Triton 3.6 的 launch 抓不到(工具层,已绕过)**:胶囊装的是 19 个经典 driver API 回调(字符串:`CUPTI hooks initialized successfully (19 driver API callbacks + resource callbacks)`),而 Triton 3.6 走 `cuLibraryLoadData`/`cuKernelGetFunction`,于是 `do_trace(dsl="triton")` 的 scout 阶段报 `failed to parse CUfunction due to missing CUmodule record … function(matmul_kernel)` 并**静默返回空结果**(`no launch matched 'matmul'; saw []`)。包自带的 `cuda/_examples/do_trace.py` 就是这个流程,所以**在这个工具链上按文档走 Triton 追踪是拿不到数据的**;换回经典 `cuModuleLoad`+`cuLaunchKernel` 后,scout 立刻能看到内核与 cubin(`captured launches: ['rotary_kernel']`,grid/block 也对)。
3. **拼接后的 cubin 一启动就 `ILLEGAL_ADDRESS`(实现层,未解决)**:这是真正的拦路处。控制实验排掉了大部分可能性:
   - 我的启动管线没问题:把**原 cubin** 用 gwatch 的 `add_argument` 追加同一个第 17 参数再序列化启动 → 正常跑完,输出 65535/65536 非零(与 kernelParams 方式的启动逐位一致)。
   - 参数元数据没问题:`cuobjdump -elf` 显示拼接后的 cubin 参数表有 **17 项**,第 16(0 起算)项 `Offset 0x58 / Size 0x8`,与探针 INIT 里读的 `ULDC.64 UR22, c[0x0][0x1b8]`(= 参数区基址 0x160 + 88)一致。
   - 拼接后的二进制没坏:文本相似度 1.000(120/120 原指令保留),字节级 0.983。
   - 探针的地址算术自洽(用 nvdisasm 读):`R20 = ptr + warp*0x800 + lane*0x40 + cursor*0x10`,游标在 4 处回绕,最多写到 `warp*2048+31*64+48`,完全落在 1 MiB 缓冲内。
   - 仍然 fault,且**放大缓冲 8×、缩到单 CTA、单 warp、`plain` 与 `split` 两种 splice、`sm` 与 `gpu` 两种时钟全部照 fault**。
   - 剩下最可疑的一处:INIT 里连续五条 `ULDC URxx, c[0x0][0x0]`,把**同一个常量**当作每一级的展开乘数(CTA 级与线程级都用它)。在这块卡上 `c[0x0][0x0]` 属于驱动保留的网格常量区(该内核自己的参数在 `c[0x0][0x160]` 之后,原内核从不读 0x0),用同一常量展开全局线程 id 只在很窄的 launch 形状下才对。而包内 `InstrumentOperator` 的 docstring 自称是 "a reusable **sm_90a** instrumentation operator",论文 Table 2 的 19 架构说的是**反汇编覆盖的指令数**(`#Instructions` 列),不是运行期算子——所以"Ampere 上能拼不能跑"是可预期的解释。我把它留成**待 H100 对照实验确认的假设**,不当结论。

因此本线程**没有开销数字**;论文 0.9–2.8% 的开销声称在我们的环境里未被验证也未被证伪。

### 3.5 与论文声称的逐条对照

| 论文声称 | 我们的测量(sm_86,RTX 3060,8 个真实 Triton cubin) | 判定 |
| --- | --- | --- |
| 保留 94–98% 原指令 | 文本级 97.1–100%;字节级 82.2–97.4% | **复现**(文本级偏上沿;字节级是更严的口径) |
| 变化的 2–6% 都是分支 | 文本级变化确实只有 `BRA`(58)/`BSSY`(34);但字节级还有 563 条**仅控制位变化**的指令,主体是 `LDS`/`LOP3`/`LEA`/`ISETP`/`ATOMG`/`LDG` | **部分修正**(控制位重排占字节级失真 86%) |
| 不给内核增加任何寄存器 | 16 次拼接全部寄存器数不变(56/28/34/80/25 → 同值) | **复现** |
| 死寄存器足够,53 次拼接无 fallback | 逐地址看:小内核 50–73% 地址可拼,大内核 100%;失败在调度不可行而非寄存器;整内核拼接的站点成功率 578/680 = 85% | **部分证伪**(与内核/地址相关) |
| probe 用最优 splicing 搜索 | 默认 `first_feasible`;`least_makespan` 需显式开且带 20 万序上界 | **实现与表述不一致** |
| 0.9–2.8% 开销 | 未测到(拼接后二进制启动即非法地址) | **未验证** |
| 19 架构支持 | 反汇编覆盖确实含 sm_86;运行期算子在这一构建上于 sm_86 失败 | **架构声称与实测不符** |

---

## 4. 机制认识之外的两点收获(我们自己的读法)

- **"观测工具不该扰动被观测对象"这条抽象的监视器不变式,在这里被拆成两笔可分开结算的账**:编译期干扰(二进制变了,追踪的证据基座不成立)与运行期扰动(时序变了,追踪的读数不可比)。XTrace 的设计主张就是把这两笔账分别压到零附近;我们的字节级测量说明**第一笔账没有它宣传的那么干净**——即使一条指令都不丢、寄存器一个都不加,它的控制位仍然会被重排,而控制位正是"时序"在二进制里的编码。要证明"观察到的是同一个执行",指令文本相似度不够,得连控制位一起比。
- **一个工具的支持矩阵要分三层看**:反汇编器覆盖(论文 Table 2 的指令数)、拼接器可拼(我们的 3.1/3.3)、运行期可信启动(我们的 3.4)。三层在图里是一句"supports 19 architectures",在实践中是三个可以独立崩掉的环节;评测报告里只写第一层的数字,复现者会在第三层被拦住。

---

## 5. 如何重跑

```bash
cd runs/2026-10-16-greenwam-onewm-xtrace-cortex/xtrace-2609.28769/code
source gwenv.sh                      # 设 LD_LIBRARY_PATH(gwatch 的 .so 要 CUDA 12 的 cudart/cupti)
                                      # GWATCH_PYTHON -> ../../../../code/xtrace-env/bin/python

# a) 静态拼接 + 二进制相似度(§3.1/3.2,不需要 GPU)
"$GWATCH_PYTHON" static_splice.py <origin.cubin> --sites mem --splice-mode plain \
    --out /tmp/spliced.cubin --json results/static_xxx.json
"$GWATCH_PYTHON" compare_binaries.py <origin.cubin> /tmp/spliced.cubin     # 文本/字节/控制位/块
bash run_static_sweep.sh                                                   # 全部缓存 cubin 一遍
"$GWATCH_PYTHON" summarize_results.py                                      # 汇总表

# b) 逐地址可行性(§3.3)
"$GWATCH_PYTHON" splice_feasibility.py <origin.cubin> --step 4 --limit 40 --reg-class auto

# c) 真机追踪(§3.4,当前在第三步 fault)
"$GWATCH_PYTHON" probe_rotary.py --sites sche            # 走工具自己的 launch 拦截(会被 Triton 3.6 拦住/被 SASS 探针 fault)
"$GWATCH_PYTHON" selflaunch_trace.py --sites mem --reps 40 --out results/self_mem.json
                                                          # 自己启动拼接镜像,不经 hijacker
"$GWATCH_PYTHON" launch_smoke.py                          # 只验启动管线:原内核 + 追加参数
```

环境:venv 在 `code/xtrace-env`(uv 管理,Python 3.12;没有 `pip` 模块,装包用
`uv pip install --python code/xtrace-env/bin/python <pkg>`),里面有 `gwatch 0.0.35`、
`torch 2.11.0+cu128`、`triton 3.6.0`、`numpy`。画图用的 matplotlib 在 `/tmp/plotenv`(临时,不进包)。
`code/xtrace-env` 本身被 gitignore,不进运行包。

脚本清单(`code/`):`gwenv.sh` 环境、`sasslib.py` 反汇编与相似度度量、`static_splice.py` 静态拼接、
`compare_binaries.py` 文本/字节/控制位三重对照、`control_flow_check.py` 分支目标等价性、
`splice_feasibility.py` 逐地址可行性、`splice_smoke.py` 最小拼接冒烟、`cuda_launch.py` 直接启动 cubin、
`launch_smoke.py` 启动管线自检、`probe_rotary.py` 走工具自身流程的端到端追踪、
`selflaunch_trace.py` 自己启动拼接镜像、`summarize_results.py` 汇总、`mkfigure.py` 出图、
`run_static_sweep.sh` 批量。原始结果在 `code/results/`(`*.json` 全量、`xtrace_sm86.png` 图)。

---

## 附录 A:一条真探针长什么样(rotary kernel,sm_86,`nvdisasm` 输出)

拼接在指令 1 之前插入 INIT 序(21 条),然后是站点探针(本例 16 条,时间戳-only)。这就是论文"探针是几条指令"的具体形态:

```asm
/*0010*/ ULDC UR16, c[0x0][0x0] ;            # ┐
/*0020*/ S2R R5, SR_CTAID.Z ;                # │ 展平全局线程 id:
/*0030*/ S2R R10, SR_CTAID.Y ;               # │ ctaid.z*UR16+ctaid.y …
/*0040*/ IMAD R11, R5, UR16, R10 ;           # │ (注意:五条 ULDC 读的都是
/*0050*/ ULDC UR17, c[0x0][0x0] ;            # │  c[0x0][0x0],见 §3.4 第 3 条)
/*0060*/ S2R R14, SR_CTAID.X ;               # │
/*0070*/ IMAD R6, R11, UR17, R14 ;           # │
/*0080*/ ULDC UR18, c[0x0][0x0] ;            # │
/*0090*/ S2R R7, SR_TID.Z ;                  # │
/*00a0*/ IMAD R12, R6, UR18, R7 ;            # │
/*00b0*/ ULDC UR19, c[0x0][0x0] ;            # │
/*00c0*/ S2R R3, SR_TID.Y ;                  # │
/*00d0*/ IMAD R13, R12, UR19, R3 ;           # │
/*00e0*/ ULDC UR20, c[0x0][0x0] ;            # │
/*00f0*/ S2R R0, SR_TID.X ;                  # │
/*0100*/ IMAD R4, R13, UR20, R0 ;            # ┘ R4 = 展平线程 id
/*0110*/ SHF.R.U32.HI R2, RZ, 0x5, R4 ;      # R2 = id>>5 = warp id
/*0120*/ ULDC.64 UR22, c[0x0][0x1b8] ;       # 取追加的第 17 个参数 = trace buffer 地址
/*0130*/ R2UR UR21, R2 ;                     # 搬到 uniform 寄存器
/*0140*/ UIMAD.WIDE.U32 UR60, UR21, 0x800, UR22 ;  # UR60 = buffer + warp*2048
/*0150*/ UMOV UR58, 0x0 ;                    # 游标 = 0     (以上 21 条 = INIT 序)
/*0160*/ CS2R R20, SR_GLOBALTIMERLO ;        # ┐ 读 64 位时间戳(每 SM 时钟见论文 §A.2.1)
/*0170*/ IMAD.WIDE.U32 R16, RZ, RZ, UR60 ;   # │ 地址 = 本 warp 基址
/*0180*/ MOV R22, UR58 ;                     # │ 本线程当前游标
/*0190*/ UIADD3 UR58, UR58, 0x1, UR0 ;       # │ 游标 +1
/*01a0*/ UMOV UR1, 0x4 ;                     # │ 每线程记录预算 = 4
/*01b0*/ S2R R8, SR_LANEID ;                 # │
/*01c0*/ IMAD.WIDE.U32 R16, R8, 0x40, R16 ;  # │ 地址 += lane*64
/*01d0*/ UISETP.GE.U32.AND UP2, UPT, UR58, UR1, UPT ;  # 游标到顶?
/*01e0*/ IMAD.WIDE.U32 R16, R22, 0x10, R16 ; # │ 地址 += 游标*16
/*01f0*/ ULDC.64 UR2, c[0x0][0x0] ;          # │ (模板遗留的死代码)
/*0200*/ MOV R9, 0x0 ;                       # │ 记录里的 site id
/*0210*/ STG.E [R16.64], R9 ;                # │ 记录 +0
/*0220*/ STG.E [R16.64+0x4], RZ ;            # │ 记录 +4(本探针不记录值)
/*0230*/ STG.E.64 [R16.64+0x8], R20 ;        # │ 记录 +8..15 时间戳 = 16 字节一条
/*0240*/ @UP2 UIADD3 UR58, UR58, -0x4, UR0 ; # ┘ 回绕(以上 16 条 = 站点探针)
/*0250*/ S2R R2, SR_CTAID.X ;                # ← 原内核的指令 1 从这里开始
```

读法:INIT 序把"我是谁"(展平线程 id → warp 基址)与 buffer 地址算进两个 uniform 寄存器(UR60 基址、UR58 游标),站点探针只用 16 条就完成"读时钟 → 算地址 → 存三条 → 推进并回绕游标",寄存器全部是插入点上已死的(这里 R0/R2/R3/R4/R5/R6/R7/R8/R9/R10/R11/R12/R13/R14/R16-R22 与 UR1/UR2/UR16-UR22/UR58/UR60-UR61 都在该点无活跃值;R2 恰好会被紧随其后的原指令 1 `S2R R2, SR_CTAID.X` 覆盖,所以借它做 scratch 是安全的)。整段没有一条 stall 被显式打印出来——`UISETP`/`IMAD.WIDE` 之间靠控制位里的记分板字段串起来,这正是 §3.2 里被改写的那些字节。

---

## 6. 好奇心清单(下一步值得追的)

1. **一条 H100/A100 对照就能定性 §3.4**:同一段拼接代码在 sm_90a 上是否正常启动?若正常,则 `ULDC c[0x0][0x0]` 的语义差异(Ampere vs Hopper 的参数/网格常量布局)就是根因;若也 fault,则是发布版本身的运行时缺陷。这一步只需一块 Hopper 卡 + 本包,成本很低。
2. **`do_trace(dsl="triton")` 的现代 Triton 兼容性**:把 Triton 换到 3.2 前后(经典 `cuModuleLoadData` 启动)再跑官方示例,可以确认 §3.4 第 2 层是"版本窗口问题"还是"从来没支持过 library API"。
3. **控制位重排是否可避免**:论文的 D3 只改"新引入的依赖",但我们看到 74 条内核指令的控制位被动过。能不能把"原指令控制位不变"加成一个硬约束(代价是 o′ 变大)?这是这篇论文最自然的一步后续研究,而且度量现成(我们的 byte-level 相似度)。
4. **调度不可行的边界**:§3.3 的失败集中在 25–28 寄存器的小内核。把"地址级可行性"画成(死寄存器数 × 活跃 scoreboard 数)的相图,就能给出"你这个内核上能插多少站点"的先验,而不是拼到一半才发现。
5. **store 下沉到底值多少**(§2 第 1 条):`plain` vs `split` 的对照实验本来是我们计划里的第一项(论文默认 auto 就是 split),现在被 §3.4 拦住了,一旦运行期修好应该立刻补上。
6. **agent 引导优化的增益怎么测**:论文的核心用例是"用 trace 引导 coding agent 优化 kernel,FA-3 上 3.9× 更少迭代、FA-4 +5.2–13.3%"。这需要一条 agent harness + 一个可跑的 trace 源;在本机我们连 trace 都拿不到,所以这条完全没有触碰——它是整篇论文最有价值也最难复现的部分。
7. **闭源 cuDNN kernel 的追踪边界**:论文附录 A.1 直接 trace 了 cuDNN SDPA/cuBLAS/TensorRT-LLM 的闭源 kernel。技术上这正是二进制插桩的强项,但"追踪厂商闭源二进制并据此改进竞品实现"在许可与工程伦理上有边界;值得单独想清楚(本机没有那些库,无法实测)。
