# 实验方案：Phase 1.2 — Basin Selectors 与 Contraction Sources 的精细诊断

**项目代号**：FixedPointProbe (FPP)，Phase 1.2
**文档版本**：v0.1
**日期**：2026-04-26
**前置文档**：
- `FPP_experiment_plan_v0_1.md` (Phase 1 主方案)
- `FPP_phase1_1_experiment_plan.md` (Phase 1.1 主方案)
- `phase1_1_master_report.md` (Phase 1.1 整合报告)
**对应文章**：`Self Context Is All AI Need?`

---

## 0. 文档定位

这份文档是 FPP 项目 Phase 1.2 的实验方案。它建立在 Phase 1.1 的诊断基础上——**Phase 1.1 给出了清晰但不完全的图像**：

- ✅ Phase 1 的"single universal attractor"被证伪——存在至少两个独立 attractor（cos = 0.503）
- ✅ Position embedding 不是收缩源（ablation 后 cos = 0.998 to baseline）
- ✅ Mode C 不能恢复 attractor 多样性（0/60 token-level fixed points）
- ⚠️ Transient 携带语义信息但 step-10 仍有 0.40 残余 accuracy ——**残余的具体性质未知**
- ⚠️ Basin selection 看起来由 lexical register 主导，但 confound 太多（HTML 同时有大写、标签、特殊字符、嵌套结构）—— **真正的 basin selector 未知**
- ⚠️ Contraction source 在排除 position embedding 后落在 LayerNorm/MLP/Attention 之间——**具体哪个未知**

Phase 1.2 的目的是把这三个 ⚠️ 转化为 ✅ 或新的精确问题。**它不是为了让 FPP 假说"复活"——是为了把 GPT-2 mode-A 自迭代的动力学性质钉死，让我们对"GPT-2 是不是 FPP-native 系统"这个问题的回答有数据支持**。

文档目标读者依然是 Claude Code（执行）+ 未来人类合作者（决策）。沿用 Phase 1 / Phase 1.1 的章节布局。

---

## 1. 实验目的

### 1.1 要回答的具体问题

**Phase 1.1 留下的三个未解之谜**：

- **Q1（残余信号性质）**：transient probe 在 step 10 的 0.40 accuracy 是 (a) basin label 的可读性，还是 (b) sub-basin 内细致输入信息的残留？这两种解读对应不同的理论修订。
- **Q2（basin selector 因子）**：basin assignment 真正由什么因子驱动？是 (a) 首字母大小写、(b) 是否含 markup 标签、(c) 嵌套结构、(d) 标点密度、(e) token 频率分布、还是 (f) 它们的某种组合？
- **Q3（contraction source 模块定位）**：Phase 1.1 by elimination 把 LayerNorm 列为头号嫌疑。但这是逻辑推论而非正面证据。具体哪个子模块（LayerNorm / Attention / MLP / residual）在做 contraction 必须正面测出来。

**关键的 cross-cutting 问题**：

- **Q4（basin 是否独立于 position embedding）**：Phase 1.1 Experiment C 只在 Phase 1 的 60 个 natural-text 输入上做 position ablation。Capital basin 在 wpe 被去掉之后是否仍然存在？这一问题决定了 Capital basin 是否反映 trained blocks 的本质属性。

### 1.2 不回答的问题（明确划界）

为防止 scope creep：

- 不做 scale ladder（GPT-2 medium / large / Pythia）—— 在 GPT-2 small 的机制都没钉死前，扩大模型只是更昂贵地重复同样工件
- 不做 mid-layer iteration —— Phase 1.1 已经把 last-layer iteration 作为基线，先在这个基线上把机制弄清楚再考虑变体
- 不做 Mode B (temperature sampling) —— Mode C 已经测过 token interface，Mode B 引入随机性后解读复杂度高，留作 Phase 1.3 候选
- 不引入任何训练或 fine-tuning —— 全程 inference + linear probe
- **不实施 FPP-Native 架构**（这个方向是远期的，而且只有当 Phase 1.x 全部完成后给出"GPT-2 完全没有 FPP 信号"的结论时才进入认真考虑）

### 1.3 成功标准

**实验成功 ≠ 看到 positive 信号让 FPP 假说复活**。沿用 Phase 1.1 §1.3 的纪律——成功的定义是：

- 对 Q1-Q4 给出基于数据的可信回答（无论方向）
- 三个 ⚠️ 至少有两个变成 ✅ 或被精确化到下一阶段的问题
- 报告诚实记录所有 negative 和未解之谜
- 输出能让下一轮三方讨论（你 + ChatGPT + 我）有干净的数据基础

如果 Phase 1.2 给出"basin 完全由表层 token 频率决定 + LayerNorm 是 single dominant contraction source + 残余信号纯粹是 basin label"——那是对当前 mode-A probe 路线的进一步压力测试，下一步**严肃考虑** FPP-Native 方向才合理。

如果给出"basin 包含 trained blocks 的真实结构、残余信号包含 sub-basin 信息、contraction 是分布式的"——那意味着 GPT-2 中确实有更细的语义结构，下一步是在 mode-A 上做更精细的 probe。

两种结果都让"是否需要造 FPP-native 架构"这个决策**基于证据**而非基于直觉。

---

## 2. 工作流约定（写入文档以制度化）

从 Phase 1.2 起，整个项目（你 + Claude Code + 外部 AI 协作）按以下工作流推进。这一节的目的不是约束 Claude Code 的执行（执行还是按 §3-§9），是约束**评估和决策环节**——评估 Phase 1.2 报告时三方应该按这个顺序工作。

### 2.1 实验执行阶段

由 Claude Code 主导，按 §3 的顺序逐项完成。这一阶段**不引入外部 AI 反馈**——直到 Phase 1.2 整合报告完成。

### 2.2 独立评估阶段

实验报告完成后，**评估按以下顺序进行，不可并行**：

1. **每个评估者先独立看原始数据和报告**，形成自己的判断。判断必须包括：
   - 哪些 hypothesis 被支持、哪些被弱化
   - 数据是否支持报告的 conclusion
   - 报告漏掉了哪些 outlier、反常、cross-experiment 关联
   - 推荐的下一步实验
2. **独立判断写出来**（作为 markdown 文档或对话回应）
3. **再读其他评估者的判断**，做交叉印证

具体说，给我（Claude）发反馈时，应该让我先做独立判断，然后**才**给我看 ChatGPT 的回应。给 ChatGPT 时反过来。

### 2.3 交叉印证阶段

读完其他评估者的判断后，做的是**实质对比**而不是站队：

- 同意之处不必复述
- 不同意之处必须明确说出"我之前的判断是 X，看了 Y 后我现在判断是 Z"——这种 update 要可见
- 各自漏掉的现象要互相补足
- 仍然有分歧的部分要交给数据决定（即设计下一阶段实验来区分）

### 2.4 决策阶段

由你（项目作者）做最终决策，但决策必须基于：

- Claude Code 的执行报告
- 至少两个外部 AI 的独立评估
- 你自己对项目大方向的把握

这个工作流的核心防护是——**每次评估都从原始数据出发，而不是从其他 AI 的 framing 出发**。这预防的是 echo chamber failure：多 round 之后所有评估者都在同一个 narrative 的精致版本里前进，研究在不知不觉中朝错误方向走。

### 2.5 工作流的元目标

这一节存在的根本目的是——**让方法论纪律可被 audit**。研究项目的健康程度不是由"是否产出 positive 结果"衡量，是由"是否能在 6 个月后回看时还能解释每个判断的依据"衡量。这个工作流就是为了让那种回看可能。

---

## 3. 核心实验设计

### 3.1 实验顺序的硬性承诺

**Phase 1.2 必须按以下顺序执行**：

1. **实验 E**：Cross-basin Probe（最便宜，决定 Q1）
2. **实验 F**：模块级 Contraction Decomposition（决定 Q3）
3. **实验 G**：B × C 联合（决定 Q4，便宜到没理由不做）
4. **实验 H**：Factorial Input Zoo（决定 Q2，最贵但最系统）

理由（同 Phase 1.1）：前面实验的结果可能影响后面实验的设计。Claude Code 不应自由调整这个顺序。

### 3.2 实验 E：Cross-basin Probe（决定 Q1）

**问题**：Phase 1.1 transient probe 在 step 10 的 0.40 accuracy，是在测 basin label 还是 sub-basin 信息？

**实施**：

数据来源：完全复用 Phase 1.1 Experiment B 的 30 个 nested-zoo trace 数据（已保存）。零额外计算。

具体步骤：

1. 取 Experiment B 30 个输入在 step 10 的 hidden states（mean-pooled 到 768 维）
2. 给每个输入打两类 label：
   - **basin label**: lowercase basin / capital basin / hybrid basin（按 cos to phase1 universal 分类）
   - **fine label**: 6 个 nested category（html / xml / json / latex / pure_brackets / markdown_list）
3. 训练两个 logistic regression probe：
   - **Probe-basin**: 预测 basin label（3 类）
   - **Probe-fine**: 预测 fine label（6 类）
4. 关键的第三步——**控制 basin 后预测 fine label**：
   - 在每个 basin 内部分别训练 probe（lowercase basin 内训一个 6 类 probe，capital basin 内训一个 6 类 probe）
   - 这测量"在固定 basin 后，残余信号还能不能区分细致输入类别"

**关键判据**：

- 如果 Probe-basin accuracy 高（≥ 0.9）但 within-basin Probe-fine accuracy 低（接近 chance）→ **0.40 残余完全是 basin label**，没有 sub-basin 信息。这是对 FPP 假说的进一步压力——universal attractor 内部完全坍缩，只剩下 basin assignment 的离散信号。
- 如果 within-basin Probe-fine accuracy 仍然显著（如 ≥ 2× chance）→ **0.40 残余包含 sub-basin 信息**。这意味着 fixed point 不只是 basin label，还携带细致输入特征——对原 §4 hypothesis 是部分支持。

**Caveat**：每个 basin 内的样本量很小（lowercase basin 4 个、capital basin 9 个、hybrid basin 17 个）。within-basin probe 的统计噪声会很大。必须用 leave-one-out cross-validation 而不是 5-fold。报告时明确标注样本量约束。

**输出**：
- 三个 probe 的准确率（含 CV std）
- 子报告 `phase1_2_cross_basin_probe.md`

**预期成本**：半天（数据已有，只需训练 probe + 写报告）。

### 3.3 实验 F：模块级 Contraction Decomposition（决定 Q3）

**问题**：Phase 1.1 by elimination 把 LayerNorm 列为头号嫌疑。直接测每个子模块的 Lipschitz/Jacobian 局部放大率，看真正在做 contraction 的是哪一个。

**实施**：

把 GPT-2 small 的一个完整 transformer block 拆成 5 个子映射：

```
1. ln_1: pre-attention LayerNorm
2. attn: multi-head self-attention sublayer (含 attn 内部的 ln 和 residual)
3. ln_2: pre-MLP LayerNorm
4. mlp: MLP sublayer (含 GELU)
5. residual: 整个 block 的 residual stream（即 block 输入 → block 输出 的整体差）
```

加上 final `ln_f` 作为第 6 个候选。

**Jacobian 估计方法**：用 finite difference + power iteration 估计 spectral norm。对每个子映射 `T`：

```
σ_max(JT) ≈ max_ε ||T(h + ε) - T(h)|| / ||ε||
```

具体算法：
1. 取一个 fixed point `h*`（用 Phase 1.1 已有的 trained 模型 step-10 hidden state）
2. 用 power iteration 找最大 singular vector：随机初始化 `v_0`，迭代 `v_{k+1} = JT^T · JT · v_k / ||...||`，直到收敛
3. 得到的特征值开方就是 spectral norm，即局部 Lipschitz 常数

**测试位置**：
- 在 fixed point `h*` 处（看 attractor 局部的 contraction 强度）
- 在 step 1 的 hidden state 处（看 transient 中的 contraction 强度）
- 在初始 hidden state `h_0` 处（看 transient 起点的 contraction 强度）

**测试输入多样性**：
- Phase 1 的 60 个 natural-text 输入
- Phase 1.1 Experiment B 的 30 个 nested-zoo 输入（含 capital basin 和 lowercase basin）

**关键判据**：

- 如果某个子模块在所有测试点 Lipschitz 都 < 1，且其他子模块 Lipschitz 都 > 1 或 ≈ 1 —— **这个子模块就是 dominant contraction source**
- 如果多个子模块都 < 1 —— contraction 是分布式的，不能归因于单一模块
- 如果在 fixed point 处所有 Lipschitz < 1 但在 transient 起点 ≈ 1 —— contraction 是局部性质，不是全局

**特别检查**：trained 与 random init 的 Lipschitz 谱对比。如果 trained 的 LayerNorm 子模块 Lipschitz 显著低于 random init 版本，说明训练让 LayerNorm 变得更收缩——这是训练塑造 attractor 的具体机制。

**输出**：
- 5 个子模块的 Lipschitz 数值（按测试位置和输入分组）
- trained vs random 对比
- 子报告 `phase1_2_module_contraction.md`

**预期成本**：1-2 天（power iteration 实施 + 多输入跑遍 + 分析）。

### 3.4 实验 G：B × C 联合（决定 Q4）

**问题**：Phase 1.1 Experiment C 的 position ablation 只测了 Phase 1 的 60 个 natural-text 输入。Capital basin 在 wpe 被去除后是否仍然存在？

**实施**：

复用 Phase 1.1 已实施的 `posfree_iterate.py` 和 Experiment B 的 30 个 nested-zoo 输入。不写新代码，只是组合两个已有 pipeline。

具体：
1. Phase 1.1 Experiment B 的 30 个输入
2. 跑 C1 cancel-pos 变体的 mode-A iteration（复用 Phase 1.1 实现）
3. max_iter = 1000（HTML basin 在 Phase 1.1 需要 ~129 步收敛，max_iter=100 不够）
4. 比较新的 C* 和 Phase 1.1 Experiment B 的 C*

**关键判据**：

- 如果 cancel-pos 后所有 30 输入都坍缩到同一个 attractor —— Capital basin 是 position-embedding 工件，在去除 wpe 后消失。Basin 不反映 trained blocks 的本质。
- 如果 Capital basin 和 lowercase basin 都仍然存在 —— Basin 是 trained blocks 的本质属性，position embedding 不是 basin selector。
- 如果 basin 数量改变（变成 1 个或变成 3 个） —— 中间情况，需要更多分析

**输出**：
- 30 输入在 cancel-pos 下的 C*
- 与 Experiment B baseline 的 cosine similarity 对比
- 子报告 `phase1_2_basin_position_ablation.md`

**预期成本**：30 秒计算 + 半天分析。这是 Phase 1.2 最便宜的实验。

### 3.5 实验 H：Factorial Input Zoo（决定 Q2）

**问题**：Phase 1.1 Experiment B 的 nested-zoo 暗示 basin 由 lexical register 决定，但 confound 太多。Factorial design 直接测哪些表层因子是 basin selector。

**实施**：

设计正交因子输入。最小可行版本是 2³ × 3 = 24 个输入：

| 因子 | 水平数 | 具体水平 |
|---|---|---|
| 首字母大小写 | 2 | capital / lowercase |
| 是否含 markup 标签 | 2 | with `<...>` / without |
| 标点密度 | 2 | high (5+/seq_len) / low (≤1/seq_len) |
| 内容类别 | 3 | natural text / code / random tokens |

每个因子组合产出 1 个输入。共 2×2×2×3 = 24 个输入，每个 seq_len = 64（与 Experiment B 一致）。

**输入构造原则**：
- 同一个内容类别下的两个变体（含/不含 markup、首字母大小写不同）必须**只在指定因子上不同**
- 例如 capital + with markup + high punct + natural text vs lowercase + with markup + high punct + natural text，差异只在首字母大小写
- 这要求 input zoo 的精心设计——Claude Code 可能需要花半天单独构造这个 zoo 并人工检查

**测试**：每个输入跑 mode-A iteration（max_iter = 1000），记录 final C* 并与 Phase 1 universal、Phase 1.1 capital basin 比较。

**关键判据**：

- ANOVA 风格分析每个因子对 "to capital basin" vs "to lowercase basin" 的影响：
  - **首字母大小写效应**：固定其他因子，只改大小写，basin 是否变化？
  - **markup 效应**：固定其他因子，只改是否有标签，basin 是否变化？
  - **标点密度效应**：同上
  - **内容类别效应**：同上

- 报告每个因子的"basin shift 概率"——例如：在 24 个输入中，"capital → lowercase basin"的 16 对比较里，有多少是首字母大小写翻转引起、多少是其他因子引起。

**输出**：
- 24 个输入及其 basin assignment
- 4 个因子的 effect size 分析
- 子报告 `phase1_2_factorial_zoo.md`

**预期成本**：1 天构造 input zoo + 30 秒计算 + 1 天分析 = 2-3 天总。

### 3.6 不在 Phase 1.2 范围内的（明确划出）

- **Mode B (temperature sampling)**：Mode C 已经测过 token interface，Mode B 引入随机性解读复杂度高
- **Mid-layer iteration**：Phase 1.1 已经把 last-layer iteration 作为基线，先弄清楚它再考虑变体
- **Scale ladder**：在 contraction source 和 basin selector 都未明前不做
- **任何训练或 fine-tuning**：Phase 1.x 全程 inference-only
- **FPP-Native 架构设计**：远期方向，不在任何 Phase 1.x 内

---

## 4. 数据记录与报告规范

### 4.1 必须保留的原始数据

每个实验保留：
- 完整 hidden state trajectory（所有 step 的 h_n）
- 实验配置 JSON
- git commit hash 和时间戳
- Probe 模型的训练/评估 logs（fold-level accuracy）
- Jacobian power iteration 的收敛 logs（实验 F）
- Factorial zoo 的精确文本（实验 H）

存储估计：约 2GB（多数是实验 G 的 30×1000-step traces）。

### 4.2 必须撰写的子报告

每个实验完成后写独立子报告（4 份），最后写整合报告 `phase1_2_master_report.md`。

整合报告必须包含：

- 对 Q1-Q4 的直接回答（带证据强度评估）
- 4 个 alternative explanation（沿用 Phase 1.1 §1.1）的最新 verdict
- 对 essay §4-§7 的具体修订建议
- 下一步建议（Phase 1.3？Phase 2？转向 FPP-Native？还是 essay 修订后暂停一段？）

### 4.3 给 AI 协作者的反馈包

整合报告完成后准备 zip 包，沿用 Phase 1 / Phase 1.1 的格式。包内容：

- Phase 1.2 整合报告
- 4 份子报告
- 关键图表（PNG）
- 实验代码 snapshot（git tag）
- 已知问题和未解之谜列表

**注意**：按 §2 工作流约定，反馈包发给外部 AI 时**不附带任何其他 AI 的判断**，让每个评估者从原始数据出发独立形成判断。

---

## 5. 分阶段验收门槛

### Phase 1.2 - E：Cross-basin Probe（约 0.5-1 天）

完成定义：3 个 probe 全部训练并评估，子报告完成。

通过门槛：能就 Q1 给出基于数据的回答（包括明确的 negative 也算通过）。

### Phase 1.2 - F：模块级 Contraction Decomposition（约 1-2 天）

完成定义：5 个子模块在 3 个测试位置的 Lipschitz 全部估计，trained vs random 对比完成，子报告完成。

通过门槛：能就 Q3 给出基于数据的回答（明确指出 contraction source）。

### Phase 1.2 - G：B × C 联合（约 0.5 天）

完成定义：30 输入在 cancel-pos 下的 mode-A iteration 完成，basin 比较完成，子报告完成。

通过门槛：能就 Q4 给出基于数据的回答。

### Phase 1.2 - H：Factorial Zoo（约 2-3 天）

完成定义：24 输入 zoo 设计 + 跑完 + 分析完成，子报告完成。

通过门槛：能就 Q2 给出 basin selector 的因子分析结果。

### Phase 1.2 - 整合（约 1-2 天）

完成定义：整合报告 + 反馈包准备好。

通过门槛：报告读起来诚实而非包装；明确标注哪些 explanation 被支持、哪些被弱化。

**总时长预算**：5-8 天 wall-clock。GPU 实际计算时间 < 1 小时。

---

## 6. 已知工程坑

### 6.1 沿用 Phase 1 / Phase 1.1 的所有坑

参考前置文档。所有那些坑（dtype、attention_mask、position_ids、数值稳定性、GPU 内存、LayerNorm、position embedding）在 Phase 1.2 仍然适用。

### 6.2 Phase 1.2 特有的坑

**实验 E 特有的坑**：
- 每个 basin 内的样本极少（lowercase basin 4 个）。within-basin probe 的方差会很大。**必须用 leave-one-out CV** 而不是 5-fold。报告时明确标注这个限制。
- "basin label" 的定义本身有自由度——按 cos to phase1 universal 阈值切分有多种方式。建议用 cos > 0.9 → lowercase basin、cos < 0.7 → capital basin、其他 → hybrid 这个分法，与 Phase 1.1 一致。

**实验 F 特有的坑**：
- Power iteration 在 high-dim space (768) 上可能收敛慢。建议 max_iter=500 + tolerance=1e-4。如果不收敛要明确标注。
- 不同子模块的"输入维度"可能不一致（attention sublayer 输入是整个 sequence 的 hidden states，shape [seq_len, 768]；MLP 是逐位置的 [768]）。需要明确每个子模块的 Jacobian 是在哪个维度上算的。
- LayerNorm 是非线性的（mean-and-variance normalization），但局部 Jacobian 仍然有定义，只是不是简单的 affine。Power iteration 仍适用但要小心 numerical edge cases（如 hidden state 已经接近 normalized 时）。

**实验 G 特有的坑**：
- max_iter 必须设到 1000（HTML basin 需要 ~129 步），不能用 Phase 1 的 100。
- cancel-pos 的工程实现要和 Phase 1.1 完全一致——如果有任何修改要明确记录。

**实验 H 特有的坑**：
- Factorial zoo 的精确控制要求高。**人工检查**每个输入是否真的只在指定因子上不同。建议 Claude Code 先生成 24 个候选，让你（项目作者）人工 review 后再跑实验。
- 用 GPT-2 的 tokenizer 时，"标点密度" 这个因子的实际效应可能和你设想的不同——某些标点会被 merge 到 word token，某些会单独成 token。需要在 input zoo 文档里附 token IDs。

### 6.3 整体的元层次坑

**不要让结果驱动叙事**——延续 Phase 1.1 §6.3 的纪律。Phase 1.2 实验在小样本上做（24-30 输入），统计显著性弱。报告时严格区分"看到趋势"和"统计上稳健"。

**不要在 phase 内 scope creep**——Phase 1.2 已经从 4 个 cheap 实验扩展到包括 factorial zoo（贵一些）。不要再扩——如果发现新的有趣方向，记下来作为 Phase 1.3 候选。

**警惕"鼓舞人"效应**——这是研究心理上最隐蔽的滑坡。如果 Phase 1.2 的某个实验给出 marginally positive 信号，会有诱惑把它放大成"FPP 越来越接近被验证"。**抑制这个诱惑**。Phase 1.2 的目标是把三个 ⚠️ 转化为 ✅ 或新问题，不是给某一方向找完整证据。

---

## 7. 硬件约束

沿用 Phase 1 / Phase 1.1。4070 Ti 12GB + 64GB RAM 完全胜任。

Phase 1.2 特有：
- 实验 E：CPU only（probe 训练）
- 实验 F：GPU + power iteration，每个子模块每个测试点 < 1 分钟，总共 < 30 分钟
- 实验 G：GPU 30 秒（30 输入 × 1000 max_iter）
- 实验 H：GPU 30 秒（24 输入 × 1000 max_iter）+ 人工检查 input zoo 半天

总 GPU 时间 < 1 小时。瓶颈仍然在分析阶段。

---

## 8. 数据与代码组织

### 8.1 沿用 Phase 1 / Phase 1.1 项目结构

新代码进 `src/`，新数据进 `data/raw/phase1_2_*` 和 `data/processed/phase1_2_*`。

### 8.2 新增模块

```
src/
├── cross_basin_probe.py       # 实验 E
├── module_jacobian.py         # 实验 F
├── basin_position_ablation.py # 实验 G（复用 Phase 1.1 的 posfree）
└── factorial_zoo.py           # 实验 H（含 input zoo 构造）
```

### 8.3 命名约定

延续之前 Phase。运行 ID 形如 `{date}_phase1_2_{experiment}_{config_hash}`。

---

## 9. 给 Claude Code 的开发建议

### 9.1 推进顺序

严格按 §3.1：E → F → G → H。**不要并行**。

特别注意——实验 F 是 Phase 1.2 工程上最复杂的（power iteration + Jacobian estimation + 多子模块拆分）。如果在 F 上遇到工程困难，可以先跳过 F 做 G（最便宜）和 E（已经决定优先级），但**绝对不要先跳到 H**——H 依赖 input zoo 的精心构造，如果设计不对会浪费整个实验的成本。

### 9.2 在不确定时的默认选择

延续之前文档的原则——**优先选保守、简单、可解读的方案**。

具体到 Phase 1.2：
- Probe 仍用 sklearn LogisticRegression
- Jacobian 用 finite difference + power iteration（不要用 PyTorch autograd——那需要保留 graph，对于很深的迭代轨迹太贵）
- Factorial zoo 设计：先生成 24 个候选交给项目作者 review，不要自动跑

### 9.3 何时停下来问

立即向项目作者请示的情况：

- 任何超过 §5 时间预算 50% 的工程问题
- 实验 F 的 Jacobian 估计在某些子模块上不收敛
- 实验 H 的 factorial zoo 在 Claude Code 自己构造时不能干净地 isolate 各因子
- 出现明显与本文档假设矛盾的现象
- 需要做超出文档范围的设计决策

### 9.4 不要做的事

- 不要在 Phase 1.2 内做 Mode B、mid-layer iteration、scale ladder
- 不要修改前序 Phase 的数据或重新解读前序报告
- 不要为了让某个 hypothesis "复活" 而调参
- 不要省略 negative 结果
- 不要尝试训练任何东西

---

## 10. 对 essay 修订的协调

Phase 1.2 完成后，对应的 essay 修订建议：

### 10.1 §7 末尾的 "Initial Probe Update" 要更新

Phase 1.1 给出的版本承诺写入"multi-basin attractor by lexical register"。Phase 1.2 后这段要再更新，加入：

- Q1 的回答（残余信号是 basin label 还是 sub-basin 信息）
- Q3 的回答（contraction source 具体在哪个子模块）
- Q4 的回答（basin 是否独立于 position embedding）
- Q2 的回答（basin selector 的因子分析）

### 10.2 §4 的耦合公式部分可能需要更精细

如果实验 F 显示 contraction 主要来自 LayerNorm、attention 子模块的 Jacobian 反而 > 1（即 attention 在抗坍缩）——那么 §4 的"context-attention 耦合"这个 framing 实际上是**区分两种不同力**：attention 在保持差异、LayerNorm 在做归一化。这比当前 essay 的"统一耦合系统"更精确。如果数据支持这种解读，§4 应该相应修订。

### 10.3 关于 FPP-Native 的提及

Phase 1.2 完成后，如果数据显示"GPT-2 中确实有可读的 sub-basin 结构"——essay 应该**淡化** "需要 FPP-Native 架构" 的暗示，强调"在现有架构上做更精细的 probe"是更接近的下一步。

如果数据显示"GPT-2 完全坍缩到 register-level prior，sub-basin 信息为零"——essay 可以更明确地把 FPP-Native 列为"经过 Phase 1.x 排除现有架构所有可能后的合理方向"。

这两个 scenario 对应不同的 essay 修订。在 Phase 1.2 完成前不预先承诺方向。

---

## 11. 一个 meta 层面的提醒

Phase 1.1 的 master report 显示 Claude Code 把方法论纪律内化得很好——它没有把多 basin 发现 inflate 为"FPP 复活"。Phase 1.2 的执行需要**继续保持**这种克制。

研究心理上要警惕的具体诱惑：

- **"鼓舞效应"**：Phase 1.1 给出比 Phase 1 更丰富的图像，会让研究者无意识地觉得"事情在往正确方向走"。Phase 1.2 如果 cross-basin probe 给 positive、Lipschitz 显示 LayerNorm 是 dominant source、factorial zoo 给出干净因子分析——这种"一切都解释通了"的感觉本身是个警告信号。在 Edison 式筛选研究里，"一切都解释通了"通常意味着我们漏掉了什么。

- **"过度因果归因"**：实验 H 的 factorial zoo 会给出"哪个因子是 basin selector"的相对效应。但小样本（24 输入）的因子分析有相当大的不确定性。报告时不要把"首字母大小写在我们 24 输入中是最强 basin selector"写成"首字母大小写决定 basin"。

- **"模块级误读"**：实验 F 的 Jacobian 是局部的、依赖测试点的。在 fixed point 处 LayerNorm Lipschitz < 1 不等于"LayerNorm 总是 contraction source"。它只意味着"在这个特定的 fixed point 周围 LayerNorm 在收缩"。报告时严格区分局部和全局 claim。

保持这些纪律——这是 Edison 式筛选研究的核心。

---

## 12. 文档版本历史

- **v0.1**（2026-04-26）：初始版本。基于 Phase 1.1 master report、ChatGPT 反馈、Claude 独立判断（按 §2 工作流）综合而成。覆盖 4 个实验：cross-basin probe、模块级 Jacobian decomposition、B×C 联合、factorial input zoo。明确划出 Mode B / mid-layer / scale ladder / FPP-Native 不在范围内。新增 §2 工作流约定，把"独立判断 → 相互印证 → 决策"制度化。

预期在 Phase 1.2 整合报告完成后更新到 v0.2，根据实际数据决定是 Phase 1.3、Phase 2、还是远期方向。
