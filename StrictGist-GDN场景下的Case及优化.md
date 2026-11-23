# StrictGist-GDN 场景下的 Case 设计及 Ascend 910B（A2）性能优化解析

> **关联仓库与提交**：
> - 用例基线提交：`64af2068`（`test(chunk_gated_delta_rule_fwd): add 3 BSND varlen perf cases and layout support`）
> - 算子一轮优化提交：`7cda4da4`（`perf by Atg: A2芯片特定用例优化`）
> - 算子二轮优化提交：`1e98e84f`（`perf(chunk_gated_delta_rule_fwd): optimize arch22 single-chunk bs16 path (~15% speedup)`，基于 `perf/chunk_gated_delta_rule_fwd_bs16`）
> - `Batchsize = 16` 性能分支：`perf/chunk_gated_delta_rule_fwd_bs16`（将 `_perf.json` 中 `id: 2 ~ 4` 更新为 `Batchsize = 16` 规模并合入二轮优化）
> - 目标硬件平台：Ascend 910B（`ascend910b` / `DAV_2201`）

---

## 一、算子现网场景分析与性能用例设计

### 1.1 StrictGist-GDN 现网业务参数特征

在 StrictGist-GDN 实际推理/生成场景中，`chunk_gated_delta_rule_fwd` 算子面对的是一组**“多条短序列 pack 在同一个 batch 内、直接采用 BSND 排布、关闭中间量保存且（hit 模式下）零初始状态、仅输出 `o`”**的典型输入契约。现网参数全貌如下表所示：

| 参数 | 现网取值 | 现网语义说明 |
| :--- | :--- | :--- |
| `q` | `[1, T, 16, 128]` BF16 | `T = 330 / 410 / 490`，`BSND` 布局（16 个 K/Q 头，头维度 `Dk = 128`） |
| `k` | `[1, T, 16, 128]` BF16 | 同上 |
| `v` | `[1, T, 16, 128]` BF16 | 同上（16 个 V 头，头维度 `Dv = 128`，即 `Hk == Hv == 16`） |
| `g` | `[1, T, 16]` FP32 | log-decay 门控，负值 |
| `beta` | `[1, T, 16]` BF16 | 经 sigmoid 后的写入门控（范围 `[0, 1]`） |
| `aLogOptional` | `null` | `A_log` 已在外部 preprocess 中算进 `g` |
| `dtBiasOptional` | `null` | `dt_bias` 同上，已算进 `g` |
| `initialStateOptional` | `null`（hit） | hit 场景无初始状态；仅 `mode 2` 传 `[5, 16, 128, 128]` FP32 |
| `cuSeqlensOptional` | `[0, 66, 132, 198, 264, 330]` 等 | `tail=16` 时步长 66；`tail=32 / 48` 时对应步长 `82 / 98` |
| `chunkIndicesOptional` | `[0, 0, 0, 0, 0]` | 5 条序列，每条长度均 `<= 128`，因此每条序列只有 chunk 0 |
| `layout` | `"BSND"` | 输入输出直接保持序列优先（`[B, T, H, D]`） |
| `scale` | `1.0` | `1 / sqrt(128)` 已提前融合进 Q/K RMSNorm 的 `gamma` 里 |
| `chunkSize` | `128` | 分块大小 `BT = 128` |
| `useExp2` | `false` | `g` 使用自然底数 $\exp$ |
| `useQkL2norm` | `false` | Q/K 归一化在调用链上游完成 |
| `allowNegEigval` | `false` | 标准门控 Delta Rule 符号约定 |
| `stateVFirst` | `false` | 状态布局为 `[N, H, Dk, Dv]` |
| `oOut` | `[1, T, 16, 128]` BF16 | 主输出（`BSND`） |
| `finalStateOutOptional` | `null`（hit） | output-only，不输出终态 |
| `qHatOut` / `kHatOut` / `qRstdOut` / `kRstdOut` / `betaEffOut` / `gCumsumOut` / `aOut` / `hOut` | `null` | `SAVE_INTERMEDIATES = 0`，中间量在算子内部重算或消费 |
| `workspaceSize` / `executor` | 启动时查询一次后复用 | 3 个 mode 各一份 |

### 1.2 `id: 2 ~ 4` 性能用例参数设计说明

> **现网 `Batchsize` 与算子仓 `B` 的口径说明**：在 `varlen` 拼接排布下，算子输入张量第 0 维 **`B` 恒为 `1`**，所有并发请求按时间维首尾拼接在第 1 维 `T` 中，即 **`T = Batchsize × Seq_len`**（`Batchsize` 为 `cu_seqlens` 划分的独立请求条数）：
> - **`Batchsize = 5`（初版 `perf/chunk_gated_delta_rule_fwd_cases`）**：`T = 5 × 66/82/98 = 330 / 410 / 490`；
> - **`Batchsize = 16`（补测 `perf/chunk_gated_delta_rule_fwd_bs16`）**：保持 `B = 1` 与单条步长 `66 / 82 / 98 <= 128` 不变，总长度扩展为 **`T = 16 × 66/82/98 = 1056 / 1312 / 1568`**，`cu_seqlens_spec` 同步扩展为对应的 17 个端点（`0,66,...,1056` / `0,82,...,1312` / `0,98,...,1568`）。

在 `tests/atk/chunk_gated_delta_rule_fwd/atk_chunk_gated_delta_rule_fwd_perf.json` 中，保留原有的 `id: 0`（长序列 inference `BNSD`）和 `id: 1`（长序列 training `BNSD`）用于守住常规大 Shape 基线不回退，并新增 `id: 2`、`id: 3`、`id: 4` 三条用例，分别精确对应现网 `tail = 16 / 32 / 48` 三种单条长度规模（下表以 `Batchsize = 5` 为例，`Batchsize = 16` 仅将 `T` 与 `cu_seqlens` 等比例扩至 16 条序列）：

| 字段 | `id: 2` (`tail16`) | `id: 3` (`tail32`) | `id: 4` (`tail48`) |
| :--- | :--- | :--- | :--- |
| 用例名称 | `chunk_gated_delta_rule_fwd_bsnd_varlen_t330_tail16` | `chunk_gated_delta_rule_fwd_bsnd_varlen_t410_tail32` | `chunk_gated_delta_rule_fwd_bsnd_varlen_t490_tail48` |
| `q / k / v` shape & dtype | `[1, 330, 16, 128]` (`bf16`) | `[1, 410, 16, 128]` (`bf16`) | `[1, 490, 16, 128]` (`bf16`) |
| `g` shape & dtype & range | `[1, 330, 16]` (`fp32`, `[-1, 0]`) | `[1, 410, 16]` (`fp32`, `[-1, 0]`) | `[1, 490, 16]` (`fp32`, `[-1, 0]`) |
| `beta` shape & dtype & range | `[1, 330, 16]` (`bf16`, `[0, 1]`) | `[1, 410, 16]` (`bf16`, `[0, 1]`) | `[1, 490, 16]` (`bf16`, `[0, 1]`) |
| `scenario` / `is_varlen` | `"varlen"` / `true` | `"varlen"` / `true` | `"varlen"` / `true` |
| `cu_seqlens_spec` | `"0,66,132,198,264,330"`（5 条 × 66） | `"0,82,164,246,328,410"`（5 条 × 82） | `"0,98,196,294,392,490"`（5 条 × 98） |
| `layout` | `"BSND"` | `"BSND"` | `"BSND"` |
| `chunk_size` / `scale` | `128` / `1.0` | `128` / `1.0` | `128` / `1.0` |
| `disable_recompute` | `false` | `false` | `false` |
| 其余布尔属性 | `use_exp2=false`, `use_qk_l2norm_in_kernel=false`, `allow_neg_eigval=false`, `state_v_first=false` | 同左 | 同左 |

### 1.3 为什么这 3 个用例能够准确覆盖现网场景

1. **三档规模一一对应**：现网的 `T = 330 / 410 / 490` 恰好是由 5 条等长短序列（步长 `66 / 82 / 98`，对应不同 tail 长度 `16 / 32 / 48`）pack 而成。拆成 3 个独立用例而非合并为 1 个，可以完整观测从 `66 -> 82 -> 98` 的分块尾部 pad 与访存/计算缩放规律。
2. **真实 `BSND` 布局链路完整透传**：通过扩展 `executor_chunk_gated_delta_rule_fwd.py` 解析 `layout="BSND"` 与可选属性，`q, k, v` 以原始 `[1, T, 16, 128]` `BSND` 张量直接送入 `ascendc.chunk_gated_delta_rule_fwd(..., layout="BSND")`，不再经过虚假的 PyTorch 侧转置成 `BNSD`（`BNSD` 会触发额外的 `layout` 相关处理差异，导致性能失真）。
3. **现网可选参数的精确映射**：
   - `aLogOptional = null` / `dtBiasOptional = null` 已在现网预处理合入负值 `g`（`[-1, 0]`），`beta` 为 sigmoid 后正值（`[0, 1]`），用例直接构造对应形态的 `g` 和 `beta`；
   - `SAVE_INTERMEDIATES = 0`（8 个中间输出为 `null`）对应 `disable_recompute = false`；
   - `initialStateOptional = null`（hit）与 `finalStateOutOptional = null` 对应 ATK 执行器中 `scenario = "varlen"`（区别于 `"state_initial_only"` 等场景），且 `chunk_indices` 由 `canonical_chunk_indices(cu_seqlens, 128)` 生成，5 条长度 `66 / 82 / 98 <= 128` 的序列正好得到 `[0, 0, 0, 0, 0]`；
   - 唯一未单独覆盖的分支是现网表格中提及的 `mode 2`（带 `[5, 16, 128, 128]` 初始状态的场景），`id: 2 ~ 4` 聚焦覆盖最核心的零初态（hit）高频路径；如需验证 `mode 2`，可按 `scenario="state_initial_only"` 单独增补用例。

---

## 二、优化思路讲解（原理分析）

在优化前的基线（`64af2068`）上，通过 Profiling（`op_statistic.csv`）发现 `id: 2 ~ 4` 的耗时由三部分组成：
- `ChunkGatedDeltaRuleFwd` 融合主核约 `181 ~ 205 us`（占比 ~93%）；
- `Transpose` 独立算子约 `12.5 us`（占比 ~6%）；
- `Cast` 独立算子约 `2.1 ~ 2.4 us`（占比 ~1%）。

总耗时只有 ~200 us 左右，想要大幅压缩 30% 以上，既要消灭**核外多余的小算子调度与搬运**，更要砍掉**融合核内部因为“短序列其实只有 1 个 chunk”而白做的大量无效计算与同步**。

Commit `7cda4da4` 正是通过以下两项**通用性优化**做到这一点的（具体涉及的文件与符号改动见文末附录）：

---

### 优化一：将 `beta` 的类型转换与布局转置直接并入 Stage P 预处理

- **原先过程**：主核内部的向量/矩阵计算要求 `beta` 为 `FP32` 且采用 `BHT` 布局（`[B, Hv, T]`），但外部传入的 `beta` 通常是 `BF16` 且采用 `BTH` 布局（`[B, T, Hv]`）。因此 Host 侧在启动主核之前，必须先额外调度两个独立的 AI Vector 小算子：先执行一次 `Cast`（将 `beta` 从 `BF16` 转为 `FP32` 写回全局内存），再执行一次 `Transpose`（将 `[B, T, Hv]` 转置为 `[B, Hv, T]` 再写回全局内存），最后才进入 `ChunkGatedDeltaRuleFwd` 主核。对于 `T = 330 ~ 490` 这种轻量 Shape，额外启动这两个小算子会带来 2 次多余的全局内存读写以及相应的 Kernel 启动和调度等待开销（合计约 `14.7 ~ 14.9 us`）。
- **优化后过程**：主核的第一道预处理工序 Stage P（`GdnCumsumPrepare`）本来就需要读取 `BTH` 排布的 `g`，并预先计算出用于按序列/分块重排到 `BHT` 布局的 gather 索引表。优化后直接跳过 Host 侧的独立 `Cast` 和 `Transpose`，把原始的 `beta_bth` 直接送入主核，让 Stage P 在处理 `g` 的同时顺带读取 `beta_bth`，**直接复用同一份 gather 索引表在寄存器/片上缓存中完成 `BF16 -> FP32` 强转和 `BTH -> BHT` 重排并写入 workspace**，供后续阶段读取。这样把原本 3 次 Kernel 启动精简为 1 次，完全消除了核外 `Cast` 和 `Transpose` 的耗时。

---

### 优化二：零初始状态 + 单 Chunk 场景跳步（`singleChunkNoState` 快速路径）

这是本次性能跃升最核心的**算法级裁剪**。

#### 📐 结合 Chunkwise GDN 原理来看
在 Chunkwise Gated Delta Rule（GDN）中，长为 $L$ 的序列被按 `chunk_size`（记为 $BT=128$）切分成若干块 $c = 0, 1, \dots, N_c - 1$。在标准 WY 表示下，每个 chunk $c$ 的前向计算分为两大来源——**“块内贡献（intra-chunk）”**与**“历史状态贡献（inter-chunk）”**：

1. **Chunk 内局部求解**：计算块内衰减前缀和 $g_{\text{cumsum}}$、经 Householder/WY 变换求逆得到的 $A = (I + \text{strict\_tril}(\dots))^{-1}$，以及两个中间量：
   - $U_c = A_c (\beta_c \odot V_c)$（用于本块内部注意力输出）
   - $W_c = A_c (\beta_c \odot K_c \odot \exp(g_c))$（用于**扣除旧状态影响、计算新写入以传递给后续 chunk**）
2. **跨 Chunk 状态递推（`FwdH`）**：从初始状态 $H_0$ 开始，在每个 chunk 之间顺序传递状态：
   $$V_{\text{new}, c} = U_c - W_c H_c, \qquad H_{c+1} = H_c \cdot \exp(g_{\text{last}, c}) + (K_c \odot \exp(g_{\text{decay}, c}))^\top V_{\text{new}, c}$$
3. **最终输出汇总（`FwdO`）**：结合“块内注意力的结果”和“上一块传来的状态 $H_c$ 的查询结果”：
   $$O_c = \text{scale} \cdot \Big( (Q_c K_c^\top \odot M_{\text{causal}} \odot \exp(\Delta g_c))\, V_{\text{new}, c} + (Q_c \odot \exp(g_c))\, H_c \Big)$$

#### 🎈 ELI5 讲故事：“只有一页纸、不看前情、也不写续集”
把整个 GDN 想象成写一套连载日记（一条序列），每一页纸最多写 `128` 行（`BT = 128`）：
- 因为是一部很长的连载，第 1 页写完要总结一张**交接小纸条 $H_1$（状态）**给第 2 页用；第 2 页要先看看第 1 页留下的小纸条 $H_1$（做 `Q @ H`），再把自己的内容叠上去，写完再交新纸条 $H_2$。
- **但在 StrictGist-GDN（`id: 2 ~ 4`）里发生了什么？**
  虽然仓库一次打包了 5 篇日记（总长度 `330/410/490` 行），但打开目录（`cu_seqlens`）一看，每篇日记分别只有 `66`、`82` 或 `98` 行——**每篇都不到 1 页纸（$\le 128$），也就是说每一篇都只有第 0 页（Chunk 0）！**
  而且在 hit 模式下：
  1. 翻开第 0 页之前脑子是空的：**没有前情提要**（`initial_state == nullptr`，所以 $H_0 \equiv 0$）；
  2. 写完第 0 页就立刻交卷：**没有第 1 页，也不用保留续集小纸条**（`output_final_state == false`）。

既然 $H_0 \equiv 0$ 且没有下一页，再按通用多块流程跑会发生大量“对着空气干活”的浪费：
1. **Stage P 里算 `gLastExp` 和 `gDecay`**：这俩是专门给跨页更新 $H_{c+1}$ 准备的，现在根本不传下一页了，**白算！**
2. **`RecomputeWU` 里算 $W = A(\beta K \exp(g))$**：$W$ 是用来做 $U_0 - W_0 H_0$ 的，可是 $H_0 = 0$，**任何数乘以 0 都是 0**，$V_{\text{new}, 0}$ 直接就是 $U_0$，算 $W$ 的 Vector 准备循环和一次完整的 **Cube 矩阵乘法全是在白算！**
3. **`DispatchFwdH`（状态递推）以及它前后的核间全同步 `SyncAll`**：既然不用往第 1 页递推状态，也不输出 `final_state`，整个跨块状态模块和 `SyncAll` 屏障等待**全都不需要！**
4. **`FwdO` 里的 `Q @ H` 矩阵乘及向量融合**：公式第二项 $(Q_0 \odot \exp(g_0)) H_0 \equiv 0$，以前代码却还要启动一次 **Cube 矩阵乘算 `Q @ 0`**，然后再启动 Vector 对 `H` 做 `Exp`、广播、乘法和加法！现在只需要用 `Attn @ U` 乘以 `scale` 转回 `BF16` 直接输出即可！

这套剪枝**不是针对 `T=330/410/490` 的特化硬编码**，而是通用动态判定：只要满足“无初态、不输出终态、无 `gk`、且每个序列长度都不超过 `chunk_size`”的任意 batch/varlen/head/dtype 组合都能自动命中；一旦 `cu_seqlens` 里存在任何长度超过 `chunk_size` 的序列（例如长短混排），会自动安全回退到完整多块流程。

同时经源码路由边界核查（[aclnn_chunk_gated_delta_rule_fwd.cpp](file:///mnt/workspace/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.cpp)）：`UseA2PreparePath` 仅针对 `chunk_size == 64` 走拆分子算子路径；而 `chunk_size == 128` 对应的 `UsePreparedCumsum` 融合路径仅对 `Hv` 有行跨度上限（`CUMSUM_PREPARE_MAX_HEADS`），**对 `T`、`NT` 与 `Batchsize` 均无上限阈值**。因此当并发从 `Batchsize = 5` 扩大到 `Batchsize = 16`（`T = 1056 ~ 1568`、`NT = 16`）时，两项优化仍然 100% 命中，且核内跳步省去的冗余计算随任务数（`Batchsize × Hv`）等比例放大。

---

## 三、算子优化性能对比（`Batchsize = 16` 实测数据）

在同一台 Ascend 910B3（逻辑设备 `npu:0`）上，用 **CANN 9.1.0** 分别重编基线 `64af2068` 和优化版 `b915797f` 的 kernel。两版均使用分支 `perf/chunk_gated_delta_rule_fwd_bs16` 的同一份 ATK 性能用例，执行 `run_test_cpu.sh -op=chunk_gated_delta_rule_fwd -npu_device_id=0 -scope=performance`，每个版本测两轮，每轮每用例采样 20 次。表中取 `op_summary.csv` 的 `Task Duration(us)` 均值。本轮同环境对比覆盖 `id: 2 ~ 4`。

### 3.1 `Batchsize = 16` 设备侧总时延对比

**基线时延包含 `Cast`、`Transpose` 和融合 kernel `ChunkGatedDeltaRuleFwd`。** 每次调用将这三个步骤的设备时延相加。优化版在这三个用例上只有一个融合 kernel。表中负数表示时延下降，绝对收益按优化后均值减优化前均值计算，总加速比按优化前均值除以优化后均值计算。

| 用例 ID | 场景描述 | 优化前均值 (us) | 优化后均值 (us) | 绝对收益 (us) | 相对优化幅度 | 总加速比 |
| :---: | :--- | ---: | ---: | ---: | :---: | :---: |
| **`id: 2`** | StrictGist varlen `T=1056`（16×66，`BSND, BT=128`） | `315.604` | **`94.188`** | `-221.416` | **`-70.16%`** | `3.35x` |
| **`id: 3`** | StrictGist varlen `T=1312`（16×82，`BSND, BT=128`） | `341.867` | **`108.634`** | `-233.233` | **`-68.22%`** | `3.15x` |
| **`id: 4`** | StrictGist varlen `T=1568`（16×98，`BSND, BT=128`） | `383.934` | **`119.709`** | `-264.226` | **`-68.82%`** | `3.21x` |

### 3.2 重点用例（`id: 2 ~ 4`）子算子 Profiling 拆解

| 用例 ID | 优化前子算子构成及平均耗时（us / 占比） | 本轮优化后子算子构成及平均耗时（us / 占比） | 拆解收益分析 |
| :---: | :--- | :--- | :--- |
| **`id: 2`**（`T=1056`） | `ChunkGatedDeltaRuleFwd` `299.757`（94.98%）<br>`Cast` `3.567`（1.13%）<br>`Transpose` `12.280`（3.89%）<br>**合计 315.604 µs** | `ChunkGatedDeltaRuleFwd` `94.188`（100%）<br>独立 `Cast` 和 `Transpose` 均为 `0.000`<br>**合计 94.188 µs** | 两个独立预处理步骤合计节省 **15.847 µs**，融合 kernel 时延降低 **68.58%**。 |
| **`id: 3`**（`T=1312`） | `ChunkGatedDeltaRuleFwd` `325.577`（95.23%）<br>`Cast` `4.160`（1.22%）<br>`Transpose` `12.129`（3.55%）<br>**合计 341.867 µs** | `ChunkGatedDeltaRuleFwd` `108.634`（100%）<br>独立 `Cast` 和 `Transpose` 均为 `0.000`<br>**合计 108.634 µs** | 两个独立预处理步骤合计节省 **16.289 µs**，融合 kernel 时延降低 **66.63%**。 |
| **`id: 4`**（`T=1568`） | `ChunkGatedDeltaRuleFwd` `367.022`（95.60%）<br>`Cast` `4.732`（1.23%）<br>`Transpose` `12.180`（3.17%）<br>**合计 383.934 µs** | `ChunkGatedDeltaRuleFwd` `119.709`（100%）<br>独立 `Cast` 和 `Transpose` 均为 `0.000`<br>**合计 119.709 µs** | 两个独立预处理步骤合计节省 **16.912 µs**，融合 kernel 时延降低 **67.38%**。 |

表中数值由原始采样独立四舍五入，用显示值相加或相减时可能相差 0.001 µs。

### 3.3 复测结论

三个 StrictGist 用例的设备侧总时延分别降低 **70.16%、68.22%、68.82%**。ID 4 的绝对收益最大，每次调用节省 **264.226 µs**。单独比较融合 kernel，ID 2–4 的时延分别降低 **68.58%、66.63%、67.38%**。

原始汇总数据见远程工作区 `/mnt/workspace/gdn-bs16-60.11cE7k/cann91-native-vs-64af2068.csv`。

---

## 附录：Commit `7cda4da4` 核心代码改动说明

### 1. 优化一相关代码改动（Stage P 融合 `beta` 预处理）

1. **接口与 OpDef 扩展**（[chunk_gated_delta_rule_fwd_def.cpp](file:///mnt/workspace/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_host/chunk_gated_delta_rule_fwd_def.cpp)、[chunk_gated_delta_rule_fwd.h](file:///mnt/workspace/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_host/op_api/chunk_gated_delta_rule_fwd.h)、[chunk_gated_delta_rule_fwd.cpp](file:///mnt/workspace/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_host/op_api/chunk_gated_delta_rule_fwd.cpp)、[chunk_gated_delta_rule_fwd_arch35.cpp](file:///mnt/workspace/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch35/chunk_gated_delta_rule_fwd_arch35.cpp)）：
   - 在 OpDef 中新增可选输入 `beta_bth` 与可选属性 `beta_layout`（默认为 `0`），并在 arch35 核入口同步透传占位参数以对齐签名。
2. **Host 调度跳过独立 `Cast` 与 `Transpose`**（[aclnn_chunk_gated_delta_rule_fwd.cpp](file:///mnt/workspace/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.cpp)）：
   - 当走 `usePreparedCumsum` 且 `beta` 为 `FP32` 或与 `q` 同精度（`BF16`/`FP16`）时（`usePreparedBeta = true`），跳过 `l0op::Cast` 和 `TransposeContiguous`，直接将原始 `params.beta` 作为 `beta_bth` 传入，`betaLayout` 设为 `1`。
3. **Tiling 校验与 Workspace 预留**（[chunk_gated_delta_rule_fwd_arch22_tiling.cpp](file:///mnt/workspace/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_host/op_tiling/arch22/chunk_gated_delta_rule_fwd_arch22_tiling.cpp)、[chunk_gated_delta_rule_fwd_arch22_state_output_tiling.cpp](file:///mnt/workspace/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_host/op_tiling/arch22/chunk_gated_delta_rule_fwd_arch22_state_output_tiling.cpp)、[chunk_gated_delta_rule_fwd_arch22_struct.h](file:///mnt/workspace/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22_struct.h)）：
   - 校验 `beta_layout=1` 下的 `[B, T, Hv]` 形状并将其逻辑视作 `{B, Hv, T}` 提供给后续阶段，并在 Trailer（`Arch22ChunkGatedDeltaRuleFwdTrailer`）中新增 `betaBhtOffset`、`betaLayout`、`betaBthIsInputT` 字段。
4. **Device 核内融合实现**（[gdn_cumsum_prepare.hpp](file:///mnt/workspace/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/stages/cumsum/gdn_cumsum_prepare.hpp)）：
   - `GdnCumsumPrepare::KernelT<kStaticBT, BetaInputT>` 在 `ProcessMixed()` 中复用 `offsetBuf_` gather 索引表，直接完成 `beta` 的类型转换与 `BTH -> BHT` 转置写回 workspace 的 `betaBhtOffset`，后续阶段通过 `effectiveBeta` 无缝读取。

### 2. 优化二相关代码改动（`singleChunkNoState` 通用快速路径）

1. **Host Candidate 标记**（[chunk_gated_delta_rule_fwd_arch22_tiling.cpp](file:///mnt/workspace/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_host/op_tiling/arch22/chunk_gated_delta_rule_fwd_arch22_tiling.cpp)、[chunk_gated_delta_rule_fwd_arch22_struct.h](file:///mnt/workspace/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22_struct.h)）：
   - Host 侧当确认 `initial_state == nullptr && !output_final_state && gk == nullptr` 时，在 Trailer 中置位 `singleChunkCandidate = 1`。
2. **Device 动态条件判断与 `FwdH` 跳过**（[chunk_gated_delta_rule_fwd_arch22.cpp](file:///mnt/workspace/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp)）：
   - Kernel `RunPhase6` 入口检查：若 `singleChunkCandidate != 0`，Dense 模式下检查 `T <= BT`；Varlen 模式下遍历 `cu_seqlens` 检查每一条序列长度 `cuPtr[seq + 1] - cuPtr[seq] <= BT`。命中 `singleChunkNoState = true` 后跳过 `DispatchFwdH` 及中间 `SyncAll` 同步，直接以 `u` 作为 `vNew` 传入 `DispatchFwdO(..., true)` 后返回。
3. **Stage P 跳过衰减末尾项**（[gdn_cumsum_prepare.hpp](file:///mnt/workspace/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/stages/cumsum/gdn_cumsum_prepare.hpp)）：
   - 传入 `skipDecay = 1`，跳过不再使用的 `gLastExp` 与 `gDecay` 计算及写回。
4. **`RecomputeWU` 仅算 `U` 模式**（[recompute_w_u_fwd_struct.h](file:///mnt/workspace/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/stages/recompute_wu/op_kernel/recompute_w_u_fwd_struct.h)、[recompute_w_u_fwd_vector.h](file:///mnt/workspace/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/stages/recompute_wu/op_kernel/recompute_w_u_fwd_vector.h)、[recompute_w_u_fwd_cube.h](file:///mnt/workspace/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/stages/recompute_wu/op_kernel/recompute_w_u_fwd_cube.h)）：
   - 在 `RecomputeTaskRange` 中新增 `uOnly`，在 `!kConcatUW`（即 `BT != 64 || V != 128`，覆盖 `BT=128`）配置下开启 `uOnly = true`，跳过 `k * beta * exp(g)` 向量制备和 `W = A @ KbgExp` Cube 矩阵乘，并正确维护 `RecomputeWUKeepAMmad` 的 L1 A 矩阵驻留释放标志。
5. **`FwdO` 零状态极简输出**（[chunk_recompute_wu_fwd_ho.cpp](file:///mnt/workspace/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/pipeline/wu_ho/chunk_recompute_wu_fwd_ho.cpp)、[gdn_fwd_o_kernel.hpp](file:///mnt/workspace/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/stages/fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp)、[block_epilogue_gdn_fwdo_output.hpp](file:///mnt/workspace/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/stages/fwd_o/op_kernel/epilogue/block/block_epilogue_gdn_fwdo_output.hpp)）：
   - 通过 `ConfigureZeroH(true)` 启用 `zeroHEnabled_`，跳过第二次 Cube 矩阵乘 `Q @ H`，并在 Epilogue 中调用 `ProcessZeroHOutput`（同时支持 `nActual <= 128` 与 `> 128`），只对 `gmVWorkspace` 中的 `attn @ U` 结果执行 `Muls(scale) -> Cast -> CopyOutputToGm`，完全省略 `gmG` 读取、`Exp`、`Broadcast`、`gmHWorkspace` 读取、`Mul` 与 `Add`。

---

### 3. Q&A 补充说明

#### Q1：总长度 `T` 明明是 `330 / 410 / 490`，分块大小 `chunk_size = 128`，为什么还能算作“单 Chunk”放在一个块内完成？

并不是把 `330 / 410 / 490` 个 token 塞进同一个长度为 128 的块里，而是 **5 条互不相干的短序列首尾拼接（pack）在一起**，每条序列各自只占 1 个属于自己的块（一共 5 个独立的块，而不是 1 个块）。

以 `T = 330`（`cu_seqlens = [0, 66, 132, 198, 264, 330]`）为例：
- 它代表张量内有 5 条独立序列（Seq 0 ~ Seq 4），每条序列的长度均为 **`66 <= 128`**（`T = 410` 对应 5 条 × `82`，`T = 490` 对应 5 条 × `98`）；
- `varlen` 模式下序列之间绝不跨越分块，也不互相做 Attention 或传递隐状态 $H$。算子分块按**每条序列内部**从 0 开始切分，因此每条序列都只有各自的 **Chunk 0**（对应现网表中的 `chunkIndicesOptional = [0, 0, 0, 0, 0]`）；
- Kernel 入口（`RunPhase6`）在 `varlen` 下遍历检查每一条序列的长度 `cuPtr[seq + 1] - cuPtr[seq] <= 128`，确认没有任何序列产生 Chunk 1 后才激活 `singleChunkNoState`（如果换成一条长 330 的单序列 `cu_seqlens = [0, 330]`，则会切出 3 个 Chunk 并自动回退到完整跨块状态路径）。

---

#### Q2：在实际运算时，5 条序列是否完全独立、同时、并行运行？如果是整条长序列切多块，会做什么处理和掩盖？当前 5 条小序列在运行过程中有哪些优化与局限？

结论是：**这 5 条序列（准确说是 `5 条序列 × 16 个头 = 80 个任务`）在核内确实是相互独立、并行分摊在多个 AI Core 上同时运行的，并不是“1 个 AI Core 独占 1 条序列”。**

1. **实际运算时的并行粒度（Ascend 910B，`20 AIC + 40 AIV`）**：
   - **Stage P（`GdnCumsumPrepare`）**：任务队列按 `abc.B * abc.NT` 划分（每条序列每个 chunk 1 个任务，负责写完该 chunk 的全部 `Hv = 16` 个头）。因此在 `Batchsize = 5` 时只有 **5 个并行任务**跑在 40 个 AIV 上（`Batchsize = 16` 时为 16 个任务）；
   - **主计算阶段（`KKT -> SolveTri -> RecomputeWU -> FwdO`）**：任务队列按 **`B * Hv * NT = 1 × 16 × 5 = 80`** 二维打散（`Batchsize = 16` 时为 `16 × 16 = 256`）。这 80 个完全独立的 `(序列块, Head)` 任务均匀分摊到全部 20 个 AIC / 40 个 AIV 核上同时并行计算。
   - **实测印证**：当 `Batchsize` 从 `5` 扩到 `16`（计算量 `3.2x`）时，总耗时仅从 `128.7 / 137.7 / 150.8 us` 增至 `226.0 / 250.7 / 285.7 us`（`1.76 ~ 1.89x`），正说明 `Batchsize = 5` 时多核尚未喂满，`Batchsize = 16` 填补了 AI Core 的空闲并行槽位。

2. **整条长序列切成多个 Chunk 时的处理与掩盖**：
   - **块内计算可并行 + AIC/AIV 异构流水掩盖**：`Stage P`、`KKT`、`SolveTri`、`RecomputeWU` 只依赖单个 chunk 内部数据，所有 `(head, chunk)` 跨核并行，且同一阶段内 Cube（AIC）与 Vector（AIV）通过局部标志（如 `FRONT_READY_FLAG`）异步流水掩盖，配合 `RecomputeWUKeepAMmad` 将逆矩阵 `A` 驻留 L1 复用。
   - **跨 Chunk 状态递推必须串行 + HO 流水线掩盖**：到了 `FwdH` 阶段，同一条序列内部 Chunk 1 必须等待 Chunk 0 更新出 $H_1$ 才能递推 $H_2$，无法跨 chunk 并行。原工程为此设计了 **HO 流水线（`GdnHoPipeline` / `ResolveHoIdlePipeline` / `HoOwnerLocal`）**，通过空闲核接力或本核本地接力（`kOwnerLocal`），将 `FwdH` 的串行递推延迟与 `FwdO` 的计算/访存尽量重叠掩盖。

3. **当前 5 条并行小序列运行中的优化与尚存空间**：
   - **保留的原有优化**：80 个独立任务仍保留 AIC/AIV 异步流水掩盖、片上缓存/索引复用以及按真实序列长度（`66 / 82 / 98`）的有效边界计算。
   - **`7cda4da4` 新增的运行流程简化**：移除了原本针对多块状态交接的 `DispatchFwdH`、`ResolveHoIdlePipeline` 与全核屏障同步 **`SyncAll`**，使全部 80 个任务摆脱跨块状态耦合与全局同步阻塞，实现端到端无串行耦合流水执行。
   - **尚存的潜在调度空间**：目前 Stage P（`GdnCumsumPrepare`）仍按 `B * NT`（仅 5 个任务）调度，在 `Batchsize = 5` 时 40 个 AIV 中有 35 个在该阶段空闲；若后续需要进一步压榨极端小 Batch 时延，可考虑将 Stage P 队列按 `(B * NT, Hv)` 展开以提高 AIV 利用率（当前改动已大幅超过 10% 目标，暂未触动 Stage P 队列契约）。
