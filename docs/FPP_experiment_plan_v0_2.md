# 实验方案：Phase 1.1 — Phase 1 坍缩现象的机制诊断

**项目代号**：FixedPointProbe (FPP)，Phase 1.1
**文档版本**：v0.1
**日期**：2026-04-26
**前置文档**：`FPP_experiment_plan_v0_1.md` (Phase 1 主方案)、`Phase 1 实验报告` (在数据包中)
**对应文章**：`Self Context Is All AI Need?`

---

## 0. 文档定位

这份文档是 FPP 项目 Phase 1.1 的实验方案。它是 Phase 1 的诊断性后续，不是新项目。

Phase 1 已经给出了一个干净的负结果——预训练 GPT-2 small 在 mode-A 自迭代下，60 个输入全部坍缩到同一个 unigram-prior universal attractor。原始数据、报告、相似度矩阵都在 `data/raw/phase1_gpt2_small/` 和 `outputs/reports/phase1_gpt2_small/` 下。

**Phase 1.1 不是为了让 FPP 假说"重新成立"——是为了诊断 Phase 1 的 collapse 来自哪里**。当前的 universal attractor 现象有至少四个互不排斥的 alternative explanation：

1. **FPP 假说本身错了**——pretrained transformer 中根本没有语义性自维持结构
2. **接口错配**——last_hidden_state 不是 GPT-2 训练分布内的合法输入，自迭代把它推到 OOD 空间
3. **工程工件**——重复注入的 position embedding + LayerNorm 提供强收缩力
4. **观察对象错了**——语义可能存在于 transient trajectory，不是 terminal fixed point

这四个 explanation 在 Phase 1 的数据里**没被区分**。Phase 1.1 的目的是用 cheap 实验区分它们，让我们知道下一步去哪个方向才有价值。

文档目标读者依然是 Claude Code（执行）+ 未来人类合作者（决策）。沿用 v0.1 的章节布局以减少认知负担。

---

## 1. 实验目的

### 1.1 要回答的具体问题

**Phase 1 观察到的 universal attractor 坍缩，机制来自哪里？**

具体子问题：

- **Q1（transient 假说）**：在坍缩前的 transient 阶段（step 0-5），输入相关的语义信息是否还存在？信息何时被洗掉？
- **Q2（位置嵌入工件）**：如果去除每次迭代时重复注入的 position embedding，60 个输入是否仍然全部坍缩到同一个 universal attractor？
- **Q3（HTML outlier）**：idx=58（HTML 片段）是 60 个输入中唯一不收敛的——它的抗坍缩行为来自哪里？是否与其他显式嵌套结构（XML、JSON、LaTeX、纯括号）共享？
- **Q4（token 接口）**：如果用 mode C（hidden → argmax token → re-embed → next iteration）代替 mode A（hidden → hidden 直接）的接口，attractor 多样性是否恢复？

### 1.2 不回答的问题（明确划界）

为防止 scope creep，明确以下不在 Phase 1.1 范围内：

- 不回答"FPP-native 架构应该长什么样"（这是远期 hypothesis，不是 cheap experiment）
- 不实施任何带 external memory 的训练实验（这是 Phase 2 的范围）
- 不做 scale ladder（更大模型重复同样实验）——在机制未明前扩规模没意义
- 不做 mid-layer iteration——它的解读复杂度比看起来高，先排除其他 explanation 再考虑
- 不引入新的训练损失或 fine-tune 任何模型——Phase 1.1 全程 inference-only

### 1.3 成功标准

**实验成功 ≠ 看到 positive 信号让 FPP 假说复活**。成功的定义是：

- 对 Q1-Q4 给出基于数据的可信回答，无论 positive 还是 negative
- 至少能区分四个 alternative explanation 里的两个（哪个被支持、哪个被弱化）
- 输出一份 Phase 1.1 报告，明确标注哪些原 hypothesis 被实验拉远、哪些被推近
- 报告中诚实记录所有 negative 结果和未解之谜

如果 Phase 1.1 告诉我们"transient probe negative + position ablation 不改变 collapse + HTML outlier 是 trivial 现象"——那是对 FPP 假说的进一步压力测试，是有价值的负结果。下一步可能是认真考虑 FPP-native 方向。

如果 Phase 1.1 告诉我们"transient probe positive + position ablation 显著改变 attractor 多样性"——那么"GPT-2 完全无法承载 FPP 信号"的判断被反驳，下一步是在标准 transformer 上做更精细的 probe，而不是造新架构。

两种结果都让研究路径变得更清晰。这是 Phase 1.1 的真正价值。

---

## 2. 理论背景与 Phase 1 的具体观察

### 2.1 Phase 1 的关键数据回顾

详细数据见 `data/processed/phase1_gpt2_small/`。关键观察：

- **trained 模型**：59/60 在 10 步内收敛，effective_rank ≈ 1.02，所有 60 个 C* 之间 mean cosine similarity = 0.996
- **random init**：0/60 收敛，effective_rank ≈ 7，cosine similarity range [0.23, 1.00]
- **universal attractor 的 LM head 投影**：所有 32 个 token 位置的 top-5 都是 ` the`/`,`/`\n`/` and`/`.`（unigram prior）
- **HTML outlier (idx=58)**：唯一不收敛，effective_rank = 1.93，final_norm = 1312（其他 ≈ 1812），deltas 在 100 步内剧烈振荡
- **per-token norm 演化**：所有输入在 step 0 norm 范围 917-1360（差异 50%），在 step 5 已经是 1788-1794（差异 0.3%）

最后一项是 Phase 1.1 的关键起点——**输入差异在 step 5 内被压缩了 100 倍以上**。这暗示语义信息可能在 transient 中存在但被快速洗掉。

### 2.2 四个 alternative explanation 的具体形式

**Explanation 1 (FPP 假说错了)**：
- 预测：在所有可能的接口修正下，trained 模型仍然坍缩到 trivial attractor
- 反对它的证据：transient probe 给 positive、HTML outlier 显示某些输入能逃逸坍缩

**Explanation 2 (接口错配)**：
- 预测：用 mode C（token interface）代替 mode A（hidden interface）后 attractor 多样性恢复
- 反对它的证据：mode C 也坍缩到 unigram prior

**Explanation 3 (工程工件)**：
- 预测：去除 position embedding 重复注入后，universal attractor 消失或变成多种 attractor
- 反对它的证据：去除 position embedding 后仍然坍缩到同一个点

**Explanation 4 (观察对象错了——语义在 transient)**：
- 预测：linear probe 能从 step 0-3 的 hidden states 识别输入类别，从 step 10+ 不能
- 反对它的证据：linear probe 在所有 step 上都接近 chance level，或在所有 step 上都很高（说明分类基于位置编码而非语义）

注意 Explanation 2、3、4 互不排斥——可能多个同时为真。Phase 1.1 的目的不是选出唯一一个，是**收集足够证据让 alternative hypothesis 之间获得相对支持度**。

---

## 3. 核心实验设计

### 3.1 实验顺序的硬性承诺

**Phase 1.1 必须按以下顺序执行**：

1. **实验 A**：Transient Linear Probe（最优先）
2. **实验 B**：HTML Outlier 深挖
3. **实验 C**：Position Embedding Ablation
4. **实验 D**：Mode C Token Interface（最后）

这个顺序基于"成本/信息量比"——前面的实验更便宜、且每个的结果可能让后面某些实验不必做。Claude Code 不应自由调整这个顺序，除非某个实验在执行中出现明确的工程阻碍（见 §10）。

### 3.2 实验 A：Transient Linear Probe

**问题**：在坍缩前的 transient 阶段，输入的 6 个类别能否被 linear probe 识别？

**实施**：

- **数据来源**：直接复用 Phase 1 已保存的 representative_trained traces。如果保存了的不够（只有 6 个），需要重跑 Phase 1 实验但这次保存所有 60 个 trained traces 的完整 hidden state 序列。重跑的成本是几分钟。
- **probe 形式**：6-way logistic regression（输入是 hidden state、输出是 6 类之一）
- **probe 训练数据**：每个 step 训练一个独立的 probe。例如 step 3 的 probe 训练数据是 60 个输入在 step 3 的 hidden state（每个 hidden state 是 [seq_len=32, hidden_dim=768]，需要展平或 mean-pool 成单个向量）
- **probe 评估**：5-fold cross-validation 在同一组 60 个输入上。报告 accuracy。

**关键参数**：
- 在 step 0, 1, 2, 3, 5, 7, 10 上分别训练 probe（这些 step 覆盖 transient 到 post-convergence 整个阶段）
- pooling 方式：mean over seq_len 维度（最简单的，先做这个；如有必要再试 last token、first token、attention-weighted）
- random init 模型也做同样的 probe，作为 baseline

**预期结果分布**：

- **如果 transient probe 给 positive**（step 0-3 准确率显著高于 chance、step 10 退化到 chance）→ 强支持 Explanation 4。理论修订方向：从 fixed-point theory 转向 transient computation theory
- **如果所有 step 都接近 chance**（包括 step 0）→ probe 设计有问题（可能是 pooling 太粗）。需要尝试不同 pooling 方式
- **如果所有 step 都很高**（包括 step 10）→ probe 在测的不是语义，是某种 token-level 信号（可能是位置、长度、character distribution）。需要构造更难的对照
- **如果 step 0 高、step 10 仍然不低**→ unigram prior 也包含输入信息（不太可能但要排查）

**重要细节**：probe 的训练样本只有 60 个（每类 10 个），样本量极小。必须做 5-fold cross-validation 报告均值±标准差，而不是单次 split。如果数据量太小让 probe 不稳定，可以增加 input zoo 到每类 30 个（总共 180 个），重跑 Phase 1 收集 traces——这个工作量在 4070 Ti 上 30 分钟内完成。

**输出**：
- 每个 step 的 6-way accuracy（带 cross-validation std）
- 一张 line plot：横轴 step、纵轴 accuracy、两条线 trained vs random
- 一份 markdown 子报告 `phase1_1_transient_probe.md`

### 3.3 实验 B：HTML Outlier 深挖

**问题**：idx=58（HTML）是唯一不收敛的——这个抗坍缩行为是 HTML 特有的，还是某种更一般的"嵌套结构"现象？

**实施**：

- **构造扩展输入集**（Phase 1.1 input zoo）：
  - 5 个 HTML 变体（不同标签深度、不同内容长度）
  - 5 个 XML 文档
  - 5 个 JSON 嵌套结构
  - 5 个 LaTeX 公式
  - 5 个纯括号嵌套（无内容，纯 `((((  ))))`）
  - 5 个 Markdown 嵌套列表
  - 总共 30 个新输入
- **跑 mode-A iteration**：max_iter = 1000（远超 Phase 1 的 100），convergence_threshold = 1e-3
- **测量**：
  - 每个输入的收敛/不收敛
  - 不收敛的输入：trajectory 是周期的、混沌的、还是缓慢漂移的？
  - 收敛的输入：final attractor 是不是 universal attractor（与 Phase 1 的 unigram prior 一致）？

**关键问题**：

- 抗坍缩与"显式嵌套层数"是否相关？（如果是，HTML 抗坍缩可能是因为 token 分布反复出现 `<`、`>`、`/` 这类有强 position dependency 的字符）
- 抗坍缩与"内容长度"是否相关？
- HTML 不收敛到什么状态？effective_rank 1.93 意味着它停在某种 rank-2 流形上——是不是 limit cycle？

**输出**：
- 每个新输入的迭代轨迹和最终状态
- 收敛性 vs 嵌套深度的散点图
- 不收敛 trace 的 deltas 时间序列
- markdown 子报告 `phase1_1_html_outlier.md`

### 3.4 实验 C：Position Embedding Ablation

**问题**：Phase 1 每次迭代都通过 `inputs_embeds` 路径让 GPT-2 重新加上 position embedding。如果阻止这个重复注入，universal attractor 是否消失？

**实施**：两个变体并行测试。

**Variant C1: A-cancel-pos**（最便宜，先做）
- 在每次迭代前，从 `h_n` 中减去 position embedding `wpe`，再喂回模型
- 这样模型在 forward pass 中重新加上 wpe 后，相当于只加了一次（与原模型 first forward 同等）
- 工程实现：`h_n_corrected = h_n - model.transformer.wpe.weight[:seq_len]`，然后传 `inputs_embeds=h_n_corrected`
- 注意 dtype 一致

**Variant C2: A-posfree**（更干净，后做）
- 手动写一个 forward 函数，直接调用 transformer blocks，跳过 word embedding 和 position embedding 步骤
- 第一次 forward 仍然走标准路径（input_ids → embedding → blocks）得到 h_0
- 后续迭代直接 `h_{n+1} = blocks(h_n)`，不加 wpe
- 工程上需要 reproducing GPT-2 的 LayerNorm placement、attention mask 处理、final LayerNorm（`ln_f`）这些细节

**测试范围**：
- 用 Phase 1 的原 60 个输入
- 两个 variant 各跑 max_iter=100
- 同时跑 trained 和 random init

**预期结果分布**：

- **如果两个 variant 都仍然 universal attractor 坍缩**→ position embedding 不是主因，工件可能来自 LayerNorm
- **如果 attractor 多样性显著恢复**（cosine similarity 显著低于 0.99）→ position embedding 是主要工件来源
- **如果 C1 和 C2 给出不同结果**→ 工程实现有 bug，需要 debug
- **如果 trained 和 random 之间的差异变化**→ 训练对吸引子结构的塑造与 position 处理强耦合

**输出**：
- 两个 variant 在 60 个输入上的 cosine similarity 矩阵
- 与 Phase 1 原始结果的对比图
- markdown 子报告 `phase1_1_position_ablation.md`

### 3.5 实验 D：Mode C Token Interface

**问题**：如果用 token-level 接口（hidden → argmax token → re-embed → 下一轮 hidden）代替 hidden-level 接口，attractor 多样性是否恢复？

**实施**：

- 第一次 forward 正常：`input_ids → ... → h_0`
- 取 `h_0` 的 LM head 投影 → argmax → 得到 token IDs `t_1`
- `t_1` 作为下一轮的 input_ids → 完整 forward → `h_1`
- 重复 `t_2, t_3, ...`
- 收敛判据：连续两次 `t_n == t_{n+1}`（token 序列不变）

**关键参数**：
- max_iter = 50（token-level 收敛通常更快）
- 用 Phase 1 的原 60 个输入

**测量**：
- 收敛性：多少 input 收敛、多少不收敛
- 收敛的 input：final token sequence 是什么？是否 universal（所有输入到同一 token sequence）？
- 不收敛的：是 limit cycle 还是 chaotic？

**注意**：Mode C 引入离散化操作（argmax），这本身可能让动力学性质和 mode A 完全不同。结果不是 "mode A 的 cleaner version"，而是另一个独立系统。报告时要明确这一点。

**输出**：
- 60 个输入的 token trajectory
- 最终 token sequences 的多样性分析
- markdown 子报告 `phase1_1_mode_c.md`

### 3.6 不在 Phase 1.1 范围内的（明确划出）

以下都是合法的研究方向，但**Phase 1.1 不做**：

- LayerNorm Lipschitz/Jacobian diagnostic（如果实验 A-D 给出明确结果，这个可能不必做）
- Mid-layer iteration（在前面实验未定位机制前做这个解读复杂度太高）
- Scale ladder 到更大模型
- Pythia 模型（架构差异太大，Phase 1.1 完成后再考虑）
- Mode B（temperature sampling）—— 比 Mode C 多了随机性，先做 deterministic 的 Mode C
- Energy term 设计（这是 Phase 2 的范围）
- FPP-Native 架构设计（远期方向，不在任何 phase 内）

---

## 4. 数据记录与报告规范

### 4.1 必须保留的原始数据

每个实验保留：

- 完整 hidden state trajectory（所有 step 的 h_n，不只是 final state）
- 实验配置 JSON
- git commit hash 和时间戳

存储估计：
- 实验 A：复用 Phase 1 数据 + 60×8 个 probe 模型（很小）
- 实验 B：30 个新输入 × 1000 step × hidden state ≈ 3GB
- 实验 C：60 个输入 × 100 step × 2 variants × 2 (trained/random) ≈ 2.4GB
- 实验 D：60 个输入 × 50 step × token sequences（很小）

总数据量约 5-6GB。你的硬盘和 64GB RAM 都足够。

### 4.2 必须撰写的子报告

每个实验完成后写独立子报告（4 份），最后写一份**整合报告** `phase1_1_master_report.md`，整合 4 个子报告并给出对 Q1-Q4 的总体回答。

整合报告必须包含：

- 4 个 alternative explanation 的相对支持度（基于 Phase 1.1 数据）
- 哪些原 hypothesis 被推近、哪些被拉远
- 对 essay 的具体修订建议（哪一节、改成什么）
- 下一步建议（继续在 GPT-2 上做更精细 probe？转向 FPP-Micro？还是修订 essay 后暂停一段）

### 4.3 给 AI 协作者的反馈包

实验完成后准备 zip 包供后续讨论。包含：

- 4 份子报告 + 1 份整合报告
- 关键图表（PNG）
- 实验代码 snapshot（git tag）
- 已知问题和未解之谜列表

---

## 5. 分阶段验收门槛

### Phase 1.1 - A：Transient Linear Probe（约 1-2 天）

完成定义：
- 8 个 step 的 probe 全部训练并评估
- 5-fold cross-validation 结果稳定
- accuracy curve 图生成
- 子报告完成

**通过门槛**：能就 Q1 给出基于数据的回答（包括明确的 negative 也算通过）

### Phase 1.1 - B：HTML Outlier（约 1-2 天）

完成定义：
- 30 个新输入的 mode-A iteration 全部完成
- HTML 不收敛行为的 trajectory 分析完成
- 子报告完成

**通过门槛**：能就 Q3 给出基于数据的回答

### Phase 1.1 - C：Position Ablation（约 2-3 天）

完成定义：
- 两个 variant 都实施且数值上正确
- 60 输入 × 2 variants × 2 (trained/random) = 240 traces 全部完成
- 与 Phase 1 原始结果的对比完成
- 子报告完成

**通过门槛**：能就 Q2 给出基于数据的回答

### Phase 1.1 - D：Mode C（约 1 天）

完成定义：
- Mode C iteration 在 60 输入上完成
- token trajectory 多样性分析完成
- 子报告完成

**通过门槛**：能就 Q4 给出基于数据的回答

### Phase 1.1 - 整合（约 1-2 天）

完成定义：
- 整合报告完成
- 反馈包准备好
- 对 essay 的修订建议明确

**通过门槛**：报告读起来诚实而非包装；明确标注哪些 alternative explanation 被支持、哪些被弱化

**总时长预算**：6-10 天 wall-clock。GPU 实际计算时间 < 2 小时（实验主要在分析阶段花时间）。

---

## 6. 已知工程坑

### 6.1 沿用 Phase 1 的所有坑

参考 v0.1 §7。所有那些坑（dtype、attention_mask、position_ids、数值稳定性、GPU 内存、LayerNorm、position embedding）在 Phase 1.1 仍然适用。

### 6.2 Phase 1.1 特有的坑

**实验 A 特有的坑**：
- linear probe 用 60 样本训练 6 类分类器 → 极小样本，必须 cross-validation
- pooling 方式（mean / last / attention-weighted）会显著影响结果，先用 mean
- 如果 step 0 probe accuracy 不高，问题可能在 input zoo 设计而不是 transient 本身——需要更多区分度的输入

**实验 B 特有的坑**：
- HTML 输入容易被 GPT-2 tokenizer 切碎（每个 `<`、`>`、字母都是单独 token）→ 32 个 token 可能装不下完整 HTML 结构
- 解决：实验 B 可以放宽到 seq_len = 64 或 128，但要记录这个变化对结果的影响

**实验 C 特有的坑**：
- Variant C1（cancel-pos）的 `h_n - wpe` 操作要小心：wpe 是 [n_position, hidden_dim] 的矩阵，h_n 是 [batch, seq_len, hidden_dim]，需要正确 broadcast
- Variant C2（posfree）需要手写 forward，要复制 GPT-2 的所有细节包括 final LayerNorm `ln_f`。建议用 `model.transformer.h[i]` 直接调用每个 block，而不是重新实现
- 两个 variant 的 dtype 必须一致

**实验 D 特有的坑**：
- argmax 操作不可微但实验是 inference-only，所以无所谓
- token sequence 的"收敛"定义和 hidden state 不同——是离散的相等而不是连续的距离阈值
- 可能出现 limit cycle（token sequence 在 N 个状态间循环）→ 需要检测周期，不只是单步 stability

### 6.3 整体的元层次坑

**不要让结果驱动叙事**——最大的坑是看到某个实验给出 marginally interesting 的结果就把它放大成主要发现。Phase 1.1 的所有实验在小样本（60 个输入）上做，统计显著性都比较弱。报告时**严格区分"看到趋势"和"统计上稳健"**。

**不要在 phase 内 scope creep**——如果实验 A 给出意外结果，诱惑会是立即增加更多 probe 变体来"理解"它。**抑制这个诱惑**。Phase 1.1 的目的是给 4 个 alternative explanation 之间的相对支持度，不是给某一个找完整证据。意外现象记到报告，作为 Phase 1.2 候选议题。

---

## 7. 硬件约束

### 7.1 与 Phase 1 一致

参考 v0.1 §8。4070 Ti 12GB + 64GB RAM 完全胜任。

### 7.2 Phase 1.1 特有的资源需求

- 实验 A：CPU 训练 linear probe（每个 < 1 分钟）
- 实验 B：GPU 跑 30 输入 × 1000 step ≈ 1 小时
- 实验 C：GPU 跑 240 traces ≈ 30 分钟
- 实验 D：GPU 跑 60 traces ≈ 10 分钟

总 GPU 时间 < 2 小时。瓶颈在分析阶段不在计算。

---

## 8. 数据与代码组织

### 8.1 沿用 Phase 1 的项目结构

新代码进 `src/`，新数据进 `data/raw/phase1_1_*` 和 `data/processed/phase1_1_*`。

### 8.2 新增模块

```
src/
├── transient_probe.py       # 实验 A
├── extended_inputs.py       # 实验 B 的扩展 input zoo
├── posfree_iterate.py       # 实验 C 的两个 variant
└── mode_c_iterate.py        # 实验 D
```

每个模块独立可运行，避免模块间隐式耦合。

### 8.3 命名约定

延续 v0.1 §9.2：每次实验运行用唯一 ID。Phase 1.1 的运行 ID 形如：`{date}_phase1_1_{experiment}_{config_hash}`，例如 `20260428_phase1_1_transient_probe_a3f9c2`。

---

## 9. 给 Claude Code 的开发建议

### 9.1 推进顺序

严格按 §3.1 的实验顺序：A → B → C → D。**不要并行做多个实验**——每个实验的结果可能影响下一个的设计选择。

例如，如果实验 A 给出明显 positive（transient 携带语义），那么实验 D（mode C）的设计可能要调整：观察 token trajectory 的早期阶段而非只是终点。这种调整必须基于实验 A 的结果，所以必须先完成 A 才能开始 D。

### 9.2 在不确定时的默认选择

延续 v0.1 §10.2 的原则——**优先选保守、简单、可解读的方案**。

具体到 Phase 1.1：
- pooling 方式：mean over seq_len（如不行再考虑其他）
- probe 模型：sklearn 的 LogisticRegression（不要用神经网络，太复杂会引入额外变量）
- cross-validation：5-fold（不要用 LOO 之类，结果不稳定）
- 数据格式：跨实验保持一致，便于后续整合

### 9.3 何时停下来问

以下情况立即停止当前任务，向项目作者请示：

- 任何超过 §5 时间预算 50% 的工程问题
- 实验 A 给出意外结果（比如所有 step 都接近 chance、或所有 step 都很高），无法判断这是 probe 设计问题还是真实信号
- 实验 C 的两个 variant 给出显著不同的结果（暗示工程实现有 bug）
- 出现明显与本文档假设矛盾的现象
- 需要做超出文档范围的设计决策

### 9.4 不要做的事

- 不要在 Phase 1.1 内做 LayerNorm Lipschitz、mid-layer iteration、scale ladder——它们都不在范围内
- 不要修改 Phase 1 的数据或重新解读 Phase 1 报告——Phase 1.1 是补充，不是修订
- 不要为了让 FPP 假说复活而调参实验直到看到 positive 信号——这会污染 Phase 1.1 的诊断价值
- 不要省略 negative 结果——它们和 positive 结果同样重要
- 不要尝试训练任何东西——Phase 1.1 全程 inference-only

---

## 10. 对 essay 修订的协调

Phase 1.1 完成后，对应的 essay 修订建议：

### 在 Section 7 末尾加一段 "Update from initial probes"

简短描述（不超过两段）：
- Phase 1 给出了 mode-A self-iteration 在 GPT-2 small 上的 universal collapse 结果
- Phase 1.1 通过 transient probe / position ablation / HTML outlier / mode C 等诊断实验，区分了几个 alternative explanation
- 当前最受支持的解读是 [根据 Phase 1.1 实际结果填空]
- 链接到 GitHub 上的实验仓库

### 不改变 essay 主体的论证骨架

Phase 1 + Phase 1.1 的负结果**不**反驳：
- 自指结构是涌现的重要机制（Section 3）
- context-attention 耦合视角（Section 4）
- 当前 LLM 缺少跨对话稳态机制（Section 5）

它们**部分压力测试**：
- 训练让吸引子结构更丰富的预测（Section 4 末尾）—— 数据反向，需要修订表述

### 弱化 Section 4 的某些表述

ChatGPT 在前一轮反馈里建议的几处弱化（独立于 Phase 1.1 结果即可执行）：

- "context-attention 耦合不动点与场论是真同构" → "结构性类比"或"可用同一类自洽方程图式描述"
- "哥德尔自指是系统超越自身的必要条件" → "自指结构是许多形式系统产生不可判定性、反身性和元层级扩展的重要机制"
- "AI 无法产出真正新的整合视角" → "当前模型缺乏跨对话、跨实例、持续积累的自我整合机制"
- 第十节"文章本身就是存在性证明" → "概念性示例" 或 "生成过程上的类比证据"

这些弱化与 Phase 1.1 无关，但应该在同一次 essay 修订时一起做。

---

## 11. 一个 meta 层面的提醒

Phase 1 给出了一个干净的负结果。Phase 1.1 的任务是诊断这个负结果的来源。

**研究心理上要警惕的是**：诊断过程本身可能产生新的"理论升级"诱惑。当 Phase 1.1 的某个实验给出 marginally positive 信号，会有诱惑把它放大成"FPP 假说复活了！"。当所有实验都 negative 时，会有诱惑把它转化为"必须造 FPP-native 架构"。

两种诱惑都偏离了 Phase 1.1 的真正目标。

**Phase 1.1 的真正目标**：给 4 个 alternative explanation 之间的相对支持度，让下一步的研究方向**更基于证据**而非更基于直觉。

如果 Phase 1.1 的结果是"我们仍然不确定 collapse 来自哪里"，那也是合理的产出——它告诉我们下一步需要更精细的 probe 设计，而不是直接转向新架构。

保持这种克制——这是 Edison 式筛选研究的核心纪律。

---

## 12. 文档版本历史

- **v0.1**（2026-04-26）：初始版本。基于 Phase 1 报告和数据、ChatGPT 两轮反馈、Claude 反馈综合而成。覆盖 4 个 cheap 诊断实验（transient probe、HTML outlier、position ablation、mode C），明确划出 FPP-Micro 等远期方向不在范围内。

预期在 Phase 1.1 整合报告完成后更新到 v0.2，根据实际数据决定是 Phase 1.2 还是 Phase 2 还是远期方向。
