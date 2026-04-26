# 实验方案：Transformer 隐藏状态自迭代的不动点动力学

**项目代号**：FixedPointProbe（FPP）
**文档版本**：v0.1
**日期**：2026-04-26
**对应文章**：`Self Context Is All AI Need?` 第七节"实验之一"

---

## 0. 文档定位

这份文档是给 Claude Code 主持开发的具体实验项目方案。它的目标读者有两类：

- **Claude Code**：作为工程实施的主要执行者
- **未来的人类合作者**（包括项目作者本人 + 后续 AI 协作者）：作为理论上下文和设计决策的记录

文档刻意不堆砌伪代码——让 Claude Code 自己写出更精确的具体实现。文档提供的是**工程规范、约束条件、判据标准、验收门槛**。每个设计选择都附理由，让执行者在遇到工程问题时能做出和原作者一致的权衡。

文档分为：实验目的与背景（§1-2）、最小可行实验设计（§3）、扩展实验（§4）、数据记录与报告规范（§5）、分阶段验收门槛（§6）、已知工程坑（§7）、硬件约束（§8）、数据/代码组织（§9）。

---

## 1. 实验目的

### 1.1 要回答的具体问题

**当 Transformer 模型在 hidden state 空间中被反复自迭代时，是否表现出非平凡的不动点动力学？**

具体子问题：

- **Q1（存在性）**：迭代是否收敛？什么样的输入收敛、什么样发散？
- **Q2（多样性）**：不同输入是否收敛到不同的 attractors？还是所有输入都坍缩到相同的 trivial 状态？
- **Q3（训练依赖性）**：训练后的模型 vs 随机初始化的同结构模型，在迭代行为上有何区别？
- **Q4（语义对应）**：收敛到的状态 `C*` 在 embedding 空间里的位置是否对应于训练数据中的可识别语义模式？还是出现了某种意义上"训练数据外"的稳定结构？

### 1.2 不回答的问题（明确划界）

为了避免范围漂移，明确以下问题**不在本实验范围内**：

- 不回答"模型有没有意识"
- 不回答"自我对话能否产出训练数据中完全没有的概念"
- 不试图通过实验改进任何现有模型的下游性能
- 不构造新的训练目标或新的架构

本实验只用现成预训练模型做 inference-time 动力学分析。这是一个**概念验证级别**的实验，不是一个能直接 deploy 的工程产品。

### 1.3 成功标准

**实验成功 ≠ 看到 positive 结果**。成功的定义是：

- 实验代码可重复运行，结果可被他人复现
- 报告完整记录所有原始数据和分析过程
- 给出对四个核心问题（Q1-Q4）的可信答案，无论答案是 positive 还是 negative
- 暴露的所有方法论问题都被明确标注
- 在合理硬件预算内（见 §8）完成所有规定步骤

如果实验告诉我们"自迭代在 GPT-2 small 上根本不收敛"，那是一个完全成功的 negative 结果。如果它给出 positive 信号但混淆变量很多无法干净归因，那是一个有信息量的初步结果。两种情况都比"勉强搞出一个看起来很 promising 的图"更有价值。

---

## 2. 理论背景与实验合法性

### 2.1 实验的理论起点

详细论证见对应文章 `Self Context Is All AI Need?`。简要复述实验背后的假设链：

**核心假设**：Transformer 的语义不是由单独的 context 或单独的 attention 携带，而是由二者的**耦合稳态**携带。形式化为：

```
A = A_θ(C)         attention 由 context 诱导
C' = U_θ(C, A)    context 在 attention 作用下更新
```

合并：`C' = transformer_θ(C)`。这是一个自耦合动力系统，其不动点定义为 `C* = transformer_θ(C*)`。

**实验的逻辑**：如果耦合不动点假设成立，那么强制让训练好的 Transformer 在 hidden state 空间反复迭代应该展现出可观测的不动点动力学。如果根本没有不动点（迭代总发散），耦合假设有问题；如果有但 trivial（坍缩到常数），耦合假设成立但语义信息量不足；如果有非平凡 attractors 且训练让结构更丰富，耦合假设得到初步实证支持。

### 2.2 工程合法性

把 hidden states 喂回 Transformer 这个操作，HuggingFace 的标准 API 通过 `inputs_embeds` 参数原生支持。它绕过 embedding 层（`transformer.wte`），直接把 [batch, seq_len, hidden_dim] 的张量送入第一个 transformer block。

**重要**：标准 GPT-2 实现中，`inputs_embeds` 路径会跳过 word embedding 和 position embedding 中的前者。Position embedding 仍然会被加上。这意味着我们的迭代实际上是：

```
h_{n+1} = transformer_blocks(h_n + position_embedding)
```

不是纯粹的 `h_{n+1} = transformer_blocks(h_n)`。这个 position embedding 的存在不是 bug——它是模型本来设计的一部分——但需要在分析时被记住，因为它给了系统一个"位置依赖偏置"，可能影响动力学。

---

## 3. 最小可行实验（实验之一核心）

### 3.1 模型选择

**起步模型**：`gpt2`（GPT-2 small，124M 参数）

理由：
- 在 4070 Ti 12GB 上 fp16 占约 250MB，绰绰有余
- 单次前向 10-20ms，每个完整 trace（最多 100 次迭代）1-2 秒
- 60 个输入 × 训练版本 + 随机版本 ≈ 4 分钟全部跑完
- 是 mechanistic interpretability 研究的标准 toy model，有大量参考文献和分析工具
- 如果在最简单的模型上看不到现象，扩大规模也大概率看不到

**扩展验证模型**（在 GPT-2 small 给出明确信号后再用）：
- `gpt2-medium`（355M）
- `gpt2-large`（774M）
- `gpt2-xl`（1.5B）
- `EleutherAI/pythia-1.4b`（更现代的预训练）

**故意不用**：LLaMA、Mistral、Qwen 等带 KV cache 优化的模型。它们在自定义 forward pass 上有更多工程坑（见 §7）。等核心现象在 GPT-2 系列上稳定后再考虑扩展。

### 3.2 迭代机制

**具体做法（方式 A，推荐）**：在 hidden state 空间直接迭代，跳过 embedding 层。

```
1. 输入 token IDs → 完整 forward pass → 取 last_hidden_state h_0
2. 把 h_0 通过 inputs_embeds 喂回 model.transformer，得到 h_1
3. 重复 h_{n+1} = transformer(inputs_embeds=h_n)
4. 监测 ||h_{n+1} - h_n|| / ||h_n||，达到阈值则停止
```

**关键参数**：
- `max_iter = 100`（迭代上限）
- `convergence_threshold = 1e-3`（相对变化量阈值）
- 监测的范数：Frobenius norm（默认 PyTorch 的 `.norm()`）

**为什么用 last_hidden_state 而不是某个中间层**：last layer 是模型自己认为"已经处理完"的状态，是最自然的"耦合系统稳态"候选。如果实验给出意外结果，可以做 ablation 测试中间层。

**为什么不做方式 B（采样 token 喂回）或 C（argmax token 喂回）**：方式 B 引入采样随机性会让"是否收敛"判读复杂；方式 C 的 argmax 是离散操作会产生数值不连续。两者都偏离了"测耦合系统的不动点动力学"这个核心问题——它们测的是更复杂的混合系统。先做方式 A，等结果稳定再考虑变体。

### 3.3 输入集合（Input Zoo）

**总规模**：60 个精心构造的输入。不是大数据集——是一个可控的探针套件。

**6 个类别，每类 10 个**：

1. **随机 tokens（控制组）**：从 vocab 随机采样的 token 序列。控制变量——纯噪声不应该有有意义的吸引子。
2. **语法对但语义随机**：用合理语法连接随机词语。"The blue eight runs quickly the apple."
3. **常见自然语言**：日常英语句子。"The cat sat on the mat."
4. **Ambiguous 句子**：带指代歧义、需要上下文消歧的句子。"The trophy doesn't fit in the suitcase because it is too small."
5. **Nonsense 但语法正常**：Chomsky 经典。"Colorless green ideas sleep furiously."
6. **高度结构化文本**：代码片段或数学表达。`def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)`

**长度统一**：所有输入填充/截断到相同长度（建议 32 tokens）。这让不同输入的 hidden state 形状一致，便于横向比较。

**注意**：不要用大数据集。本实验目的不是统计显著性，是观察现象的形状。10 个高质量的精心构造输入比 1000 个随机数据点更有价值——它们让你能逐个研究每个 trace 的具体行为。

### 3.4 对照实验

**核心对照：训练 vs 随机初始化**。

```
1. 加载训练好的 GPT-2 small（HuggingFace 标准 checkpoint）
2. 加载相同架构但随机初始化的 GPT-2 small（用同样的 GPT2Config 但不调用 from_pretrained）
3. 对同一组 60 个输入，分别在两个模型上跑迭代
4. 比较各项指标的差异
```

随机初始化版本是 baseline。它告诉我们："架构本身的迭代行为是什么样"。任何 trained vs random 的差异，都是训练真正塑造的部分。

**注意**：随机初始化的具体方式必须和训练版本完全一致。建议使用 `GPT2LMHeadModel(GPT2Config())` 这种最简单的 PyTorch 默认初始化。如果想更严格，可以用同样的 random seed 多次初始化做平均。

### 3.5 必须测量的指标

每个 trace 必须记录以下数据：

**收敛性指标**：
- 是否收敛（boolean）
- 收敛时的迭代步数（如收敛）
- 整个 trace 的 `||h_{n+1} - h_n|| / ||h_n||` 序列

**稳态特征指标**（仅对收敛的 trace 计算）：
- `C*` 的 Frobenius norm
- `C*` 各个 token 位置的 hidden state norm 分布（看是否所有位置都坍缩或某些位置保留信息）
- `C*` 的 effective rank（用 SVD 算的有效秩，看维度坍缩情况）

**语义对应指标**：
- 把 `C*` 的每个 token 位置通过 unembedding 矩阵投影回 vocab logits，取 top-5 token——这告诉我们 `C*` "看起来像什么 token"
- 这个投影出来的 token 序列和原始输入序列的关系——是输入的某种 attractor、还是完全无关的某个固定模式？

**多样性指标**（跨 traces 计算）：
- 60 个 `C*` 状态之间的两两 cosine similarity 矩阵
- 用 hierarchical clustering 看 `C*` 是否聚成几类
- 类别和原始输入的 6 个类别是否对应

### 3.6 鲁棒性测试

主实验跑完后，做以下 robustness check：

**初始扰动测试**：取 5 个有代表性的输入，对每个输入做 10 次实验：
- 每次在 `h_0` 上加小扰动 `ε ~ N(0, 0.01·||h_0||)`
- 看 10 次的 `C*` 是否一致（同一吸引域）还是分散（chaotic）

**长 trace 测试**：取 5 个收敛快的输入，把 `max_iter` 设为 1000 而不是 100，看是否会出现"先看似收敛后发散"的现象。

这两个测试每个 trace 多花一点时间，但能告诉你"看到的收敛是真稳定还是假稳定"。

---

## 4. 扩展实验（仅在最小实验给出明确信号后做）

最小实验给出明确结果后，按以下顺序考虑扩展：

### 4.1 模型规模扩展

如果 GPT-2 small 上看到非平凡收敛行为，依次在 medium / large / xl / Pythia-1.4B 上重复实验。**只关心一个问题**：现象是否随规模 scale？是更明显（说明耦合不动点是真东西，越大越稳定），还是消失（说明 GPT-2 small 上看到的是工件）？

### 4.2 多种迭代机制对比

引入方式 B 和方式 C：
- 方式 B：sampling token 喂回
- 方式 C：argmax token 喂回

比较三种方式下的不动点结构差异。这给出"耦合系统的不动点"和"采样系统的不动点"在概念上的区分。

### 4.3 中间层迭代

不只迭代 last hidden state，也试试中间某层。这测试"耦合稳态在哪一层最显著"——可能不是最后一层。

### 4.4 干预实验（最有趣的一步）

如果某个输入收敛到 `C*`，那么：
- 在 `C*` 上做小扰动 `C* + δ`，会被拉回 `C*` 还是逃逸？这测吸引子的稳定性。
- 强制把 `C*` 投影到某个语义维度（比如改变某个 token 位置），看是否会重新收敛到不同的 `C*'`。这测吸引子的局部结构。

这一步把实验从"被动观察"升级为"主动干预"，更接近 mechanistic interpretability 的工作模式。

---

## 5. 数据记录与报告规范

### 5.1 必须保留的原始数据

实验运行时，必须保留（保存为文件）：

- 每个 trace 的完整 hidden state 序列（h_0, h_1, ..., h_n）。这是"原始数据"。可以用 .pt 或 .npy 格式保存。
- 实验配置（模型名称、seed、max_iter、threshold 等所有超参数）保存为 JSON
- 输入集合的精确内容（所有 60 个 input 的 token IDs 和原始文本）
- 每次运行的时间戳和 git commit hash（确保可复现）

存储估计：60 个输入 × 100 次迭代 × seq_len 32 × hidden_dim 768 × 4 bytes ≈ 600MB per model。整个实验所有数据约 1-2GB，64GB RAM 足够，磁盘上也不大。

### 5.2 必须产生的分析图表

每次完整实验后产生：

- **收敛性总览**：6 个输入类别 × (trained, random) = 12 个直方图，显示每类的收敛步数分布
- **轨迹可视化**：从每个类别选 1-2 个代表性 trace，画出 `||delta||` 随迭代次数的变化
- **C* 相似度矩阵**：60 × 60 cosine similarity heatmap（trained 和 random 各一张）
- **聚类结果**：在 `C*` 状态上做 hierarchical clustering，画出 dendrogram
- **`C*` token 投影**：把每个收敛状态的 top-5 投影 token 列成表格

### 5.3 必须撰写的实验报告

实验完成后写一份 markdown 格式报告（参考 Structoken 项目 v0.3 的报告风格）。报告必须包含：

- 实验配置完整描述
- 对 Q1-Q4 四个核心问题的直接回答（带支持证据）
- 所有发现的意外现象（无论正面负面）
- 明确标注的方法论限制
- 下一步建议（是扩展模型规模？深挖现象？放弃这个方向？）

**关键纪律**：报告里**不要省略 negative 结果**。如果 60 个输入里 50 个发散、10 个 trivial 收敛，那就如实报告。诚实的负结果比包装过的正结果更有长期价值——这是 Structoken 项目的核心方法论纪律之一。

### 5.4 准备给 AI 协作者的反馈包

每次实验完成后，准备一个 zip 包供后续讨论使用。包内容：
- 实验报告 markdown
- 关键图表（PNG）
- 代表性 trace 的原始数据（少数几个完整轨迹）
- 实验代码 snapshot（git tag）
- 已知问题和未解之谜列表

这个包是和 ChatGPT、Claude 反复讨论的输入。设计成 zip 是因为 chat 界面对单个长文档支持比对多个文件好。

---

## 6. 分阶段验收门槛

### Phase 0：基础设施（约 1-2 天）

完成定义：
- 能加载 GPT-2 small，做 inputs_embeds 形式的前向
- 能跑通一次完整 trace，保存 hidden state 序列
- 收敛性判断逻辑正确
- 至少 1 个测试输入能跑出可解释结果

**通过门槛**：在你的 4070 Ti 上单个 trace 在 5 秒内完成。

### Phase 1：最小实验完整跑通（约 2-3 天）

完成定义：
- 60 个输入 × (trained, random) = 120 个 trace 全部跑完
- 所有 §3.5 列出的指标都被计算
- 所有 §5.2 列出的图表都被生成
- 鲁棒性测试（§3.6）完成

**通过门槛**：能就 Q1-Q4 给出基于数据的初步回答。

### Phase 2：初步分析与第一份报告（约 2-3 天）

完成定义：
- 撰写完整实验报告
- 准备好反馈包
- 报告中明确标注了"哪些发现是稳的、哪些需要进一步验证、哪些可能是工件"

**通过门槛**：报告读起来诚实而非包装。包括 negative 结果和未解之谜。

### Phase 3 及以后：根据 Phase 2 反馈决定

Phase 2 报告会被发给 ChatGPT、Claude 等做交叉审视。**Phase 3 的具体方向由那次讨论决定**，可能是：
- 扩展到更大模型确认现象不依赖 GPT-2 small
- 深挖某个意外现象
- 引入干预实验
- 完全转向另一个角度（如果 Phase 1-2 揭示了新问题）

**重要**：Phase 3 不应该在 Phase 2 完成前规划。先得到第一批数据，再决定下一步。

---

## 7. 已知工程坑

提前列出以避免踩坑浪费时间：

### 7.1 inputs_embeds 的 dtype 问题

HuggingFace 模型默认 fp32，但 4070 Ti 在 fp16 下快得多。需要确保 `inputs_embeds` 张量和模型权重 dtype 一致。建议全程用 fp16 或 bf16。

### 7.2 attention_mask 在 inputs_embeds 路径下

当用 `inputs_embeds` 而非 `input_ids` 时，HuggingFace 不会自动构造 attention_mask。需要显式传入。所有位置都设为 1（不 mask 任何位置）。

### 7.3 position_ids

类似上面，需要显式传入 `position_ids = torch.arange(seq_len)`，否则可能出现意外行为。

### 7.4 数值稳定性

迭代深度大时（接近 100 次），hidden states 的 norm 可能爆炸或衰减。**实时监控**每次迭代后的 norm，如果超过 100 倍初始值或低于 0.01 倍初始值，停止迭代并标记为"数值不稳定"而非"发散"。

### 7.5 GPU 内存累积

深度迭代时如果不在每次迭代后调用 `torch.cuda.empty_cache()` 或正确管理 autograd graph（用 `with torch.no_grad():`），内存可能累积。本实验**全程**用 `torch.no_grad()`——我们不需要梯度。

### 7.6 LayerNorm 的特殊行为

GPT-2 在每个 block 输入处有 LayerNorm。这会让 hidden states 在每次迭代后被归一化到接近的 norm。这可能让"收敛"看起来很容易达到——但那不一定是真的不动点。**报告时**要分别报告"原始 norm" 和"LayerNorm 后 norm"的变化，避免被这个现象误导。

### 7.7 Position embedding 的累积

如前 §2.2 所述，每次迭代都会加 position embedding。这意味着 `h` 不是在自由演化，是被反复加上同一个偏置。这是设计的一部分，但分析时要注意——如果你看到强烈的位置依赖性，可能就是这个。

---

## 8. 硬件约束

### 目标硬件

- CPU：Ryzen 7 7800X3D
- RAM：64GB
- GPU：RTX 4070 Ti 12GB
- 系统：Windows 11

### 适配建议

**GPT-2 small**：fp32 也能舒服跑。无需特殊优化。

**GPT-2 medium / large**：建议 fp16。

**GPT-2 XL（1.5B）**：必须 fp16，可能需要 attention 实现优化（`attn_implementation="sdpa"`）。

**Pythia-1.4B**：fp16，注意 LayerNorm 可能要 fp32 否则数值不稳。

**所有模型**：开 `torch.backends.cudnn.benchmark = True` 加速。

### 不要做的事

- 不要在 4070 Ti 上跑 LLaMA-7B 之类——能跑但太慢，不适合 60 输入 × 大量 sweep 的实验
- 不要尝试训练任何东西——本实验是 inference-only

### 时间预算

- Phase 0-2 全部完成预计 5-8 天 wall-clock 时间（包括代码编写、debug、分析）
- 实际 GPU 计算时间不到 1 小时
- 主要时间在工程实现和数据分析

---

## 9. 数据与代码组织

### 9.1 推荐的项目结构

```
fpp/
├── README.md                  # 项目入口
├── requirements.txt
├── configs/                   # 实验配置 JSON
│   └── phase1_gpt2_small.json
├── src/
│   ├── iterate.py             # 核心迭代逻辑
│   ├── inputs.py              # 输入集合定义
│   ├── metrics.py             # 各项指标计算
│   ├── visualize.py           # 图表生成
│   └── report.py              # 报告自动生成辅助
├── notebooks/                 # 探索性分析（jupyter）
├── data/
│   ├── raw/                   # 原始 hidden state 序列（.pt）
│   └── processed/             # 计算好的指标（.npz）
├── outputs/
│   ├── figures/               # 生成的图表
│   └── reports/               # markdown 报告
└── tests/                     # 单元测试
```

### 9.2 命名约定

每次实验运行用一个唯一 ID：`{date}_{phase}_{model}_{config_hash}`，例如 `20260428_phase1_gpt2_small_a3f9c2`。所有输出（数据、图表、报告）都带这个 ID。这让多次实验易于区分和对比。

### 9.3 git 纪律

- 每个 phase 完成后打 git tag
- 实验报告包含 tag 名，便于复现
- raw hidden states 不进 git（太大），用 `.gitignore` 排除 `data/raw/`
- 但**指标文件、图表、报告**进 git，这些是真正需要保留的

---

## 10. 给 Claude Code 的开发建议

### 10.1 推进顺序

1. 先用一个最简化的输入跑通整个 pipeline，端到端验证可行（半天）
2. 然后扩展到完整的 60 输入和所有指标（一天）
3. 加上随机初始化对照（半天）
4. 鲁棒性测试（半天）
5. 报告生成（一天）

总计 3-4 天 wall-clock 时间能完成 Phase 0-2。

### 10.2 在不确定时的默认选择

遇到设计选择不明确时，**优先选保守、简单、可解读的方案**，而不是聪明、复杂、可能高效的方案。理由：本项目是探索性研究，可读性比性能更重要。

例如：
- 用 fp16 vs fp32：除非显存不够，fp32 更稳更易调试
- 用 batch 处理 vs 单样本：先单样本跑通，再优化 batch
- 用 numpy vs torch：分析阶段优先 numpy，更容易和其他工具配合

### 10.3 何时停下来问

以下情况立即停止当前任务，向项目作者请示：
- 任何超过 §6 时间预算 50% 的工程问题
- 出现明显与本文档假设矛盾的现象（比如所有 trace 都不收敛、或都瞬间坍缩）
- 需要做超出文档范围的设计决策（比如选用未在 §3.1 列出的模型）
- 发现项目代号、目录结构、命名约定上的根本性歧义

### 10.4 不要做的事

- 不要"优化"到偏离文档的实验设计
- 不要为了更好的视觉效果而修改原始数据
- 不要省略 negative 结果
- 不要在 Phase 2 完成前规划 Phase 3
- 不要尝试训练任何东西——本项目是 inference-only

---

## 11. 文档版本历史

- **v0.1**（2026-04-26）：初始版本，基于 `Self Context Is All AI Need?` 文章第七节"实验之一"的方案。覆盖最小可行实验设计、扩展实验候选、数据规范、验收门槛、工程坑列表、硬件约束。

预期在 Phase 2 第一份实验报告完成后更新到 v0.2，根据实际数据修正方案。
