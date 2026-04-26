# 实验方案：Phase 1.3 — 机制断言的三角验证与盲点排查

**项目代号**：FixedPointProbe (FPP)，Phase 1.3
**文档版本**：v0.2
**日期**：2026-04-27
**前置文档**：
- `FPP_experiment_plan_v0_1.md` (Phase 1)
- `FPP_phase1_1_experiment_plan.md` (Phase 1.1)
- `FPP_phase1_2_experiment_plan.md` (Phase 1.2)
- `phase1_2_master_report.md` (Phase 1.2 整合报告)
**对应文章**：`Self Context Is All AI Need?`

---

## 0. 文档定位

这份文档是 FPP 项目 Phase 1.3 的实验方案。它建立在 Phase 1.2 给出的"机制图像"之上，目的是**对这个图像做三角验证（triangulation）和盲点排查**——不是为了让 FPP 假说复活，也不是为了进一步推进 essay，是为了在我们做出更强的理论 commitment 之前，先把 Phase 1.2 的 claim 钉到无懈可击。

Phase 1.2 给出的合成图像非常干净：

> 训练后的 GPT-2 small mode-A 自迭代有**一个架构吸引子 + 一个由 wpe 维持的卫星结构**；卫星 trigger 是 angle-bracket markup tokens；收缩源是 trained LayerNorm（σ_max ≈ 0.12）；step-10 残余信号主要是 basin label。

但 Phase 1.2 自己（master report §6）和外部 AI 评估都注意到——**这个图像 suspiciously coherent**。Phase 1.3 必须正面处理"太干净是不是过度拟合"这个 meta 问题。

特别地——Phase 1.2 评估暴露了一个被两个独立 AI 评估者**都漏掉**的现象：lowercase basin 收敛 10 步、HTML/markup-induced 结构收敛 129 步，13× 速度差异。这个现象暗示这两个所谓的"basin"在动力学性质上**根本不对等**——把它们用同一个词描述本身就是概念污染。Phase 1.3 把这个被双盲点漏掉的现象作为一等公民处理。

文档目标读者依然是 Claude Code（执行）+ 未来人类合作者（决策）。沿用前几个 phase 的章节布局。

### 0.1 命名约定（重要）

为避免 Phase 1.2 的"capital basin"命名导致的概念污染（同一个词描述两种动力学性质完全不同的现象），Phase 1.3 全程使用以下命名：

- **lowercase attractor**：Phase 1.2 中收敛 10 步、norm ≈ 2563、rank ≈ 1.02 的强吸引子。这个命名保留——它的动力学地位由 Phase 1.2 数据干净支持。
- **HTML-induced metastable structure** / **wpe-supported candidate**：原 Phase 1.2 称为"capital basin"的对象。**本体地位 provisional**——它可能是真 fixed point、metastable transient、limit cycle、或 wpe-forced equilibrium。Phase 1.3 实验 J 的目的是确定它到底是哪一种。

**Claude Code 在所有 Phase 1.3 输出中必须使用这套精确命名**。如果 Phase 1.3 实验给出明确 verdict，Phase 1.x 总结时再统一更新到最终命名。

### 0.2 工作流回顾

按 Phase 1.2 §2 确立的"独立判断 → 相互印证 → 决策"工作流。Phase 1.2 报告完成后我和 ChatGPT 独立评估，两份判断 80% 重合但漏的东西不同：

- Claude 漏：angle-bracket × position 交互矩阵
- ChatGPT 漏：spectral norm vs spectral radius 的方法论区别、ln_f 在 mode-A 动力学中的不相关性
- **两者都漏**：13× 收敛速度差异这个具体数字应该解释什么

第三类（双方都漏）尤其重要——它说明这个工作流虽然 robust，仍然可能有共同盲点。Phase 1.3 把这个被双盲点漏掉的现象作为一等公民实验（实验 J）。

后续——在 ChatGPT 看了 Claude 的 13× 慢观察后，它进一步深化为"basin 概念污染需要语言层面修订"+ "WPE shutoff after capture 是关键诊断实验"+ "tail convergence rate fitting 量化吸引强度"。这次 cross-round complementarity 让 Phase 1.3 的实验 J 设计比单一 AI 能给出的版本扎实得多。

---

## 1. 实验目的

### 1.1 要回答的具体问题

Phase 1.2 留下的几个未解之谜：

- **Q1（markup 还是 angle-bracket）**：basin selector 是 "any tag-like character" 还是 "angle bracket BPE token IDs specifically"？这两种解读对应非常不同的机制。
- **Q2（LayerNorm contraction 的根源）**：trained LayerNorm σ_max ≈ 0.12 是因为 (a) learned γ < 1，(b) fixed-point hidden state norm 大让 1/σ_x 小，还是 (c) 两者耦合？
- **Q3（spectral norm vs spectral radius）**：Phase 1.2 测的 σ_max 不是严格的 contraction 指标。spectral radius (|λ_max|) 才是决定动力学稳定性的量。两者数值差异有多大？"asymmetric coupling"的解读在 spectral radius 上是否仍成立？
- **Q4（HTML-induced structure 的本体地位）**：原"capital basin"是 (a) 真 fixed point、(b) 长寿命 metastable transient、(c) limit cycle、还是 (d) 由 wpe 持续 forcing 维持的非自主平衡点？这四种可能对应不同的理论修订。
- **Q5（位置 × token 交互）**：angle-bracket 在 sequence 不同位置（开头 / 中间 / 结尾）是否同等触发结构？是 token 触发还是 position-token 共同触发？

### 1.2 不回答的问题（明确划界）

为防止 scope creep：

- 不做 scale ladder（GPT-2 medium / large）—— Phase 1.x 完成前不扩规模
- 不做 cross-architecture（Pythia / LLaMA）—— 同上理由，留到 Phase 2 候选
- 不做 mid-layer iteration —— 仍然是 last-layer iteration 为基线
- 不做 Mode B (temperature sampling) —— Mode C 已经测过 token interface
- 不实施 FPP-Native 架构 —— 远期方向
- 不做大规模扰动 basin-volume 实验 —— 高维扰动的解读复杂度高，需要先有 spectral radius 信息（实验 M）才能干净处理。作为 Phase 1.4 候选

### 1.3 essay 修订冻结

**Phase 1.3 完成后不修订 essay**。这是有意识的纪律。每次实验后修订 essay 的诱惑很强（前几次都做了），但多轮小修会让 essay 变成补丁堆叠。

Phase 1.3 完成后写一份"Phase 1.x 总结报告"，把所有 mechanism findings 合在一起。**那时再统一修订 essay**，做一次干净的 §4 重写而不是再补一段。

如果 Phase 1.3 给出意外的 reversal（比如实验 J 显示 HTML structure 长 trace 后漂回 lowercase，整个"两个 basin"图像被推翻），essay 需要更大幅度修订——但仍然等 Phase 1.x 完成后统一做。

### 1.4 成功标准

延续之前的纪律——**实验成功 ≠ 看到 positive 信号**。成功的定义是：

- 对 Q1-Q5 给出基于数据的可信回答
- Phase 1.2 的"机制图像"要么获得三角验证（多角度独立支持），要么暴露具体不一致点
- HTML-induced structure 的本体地位有 verdict（fixed point / metastable / limit cycle / wpe-forced 四选一或新选项）
- 报告诚实记录所有 negative 和未解之谜

---

## 2. 工作流约定（沿用并补充）

延续 Phase 1.2 §2 的"独立判断 → 相互印证 → 决策"工作流。

### 2.1 关于双方都漏的盲点的特别约定

**当某个现象被两次审视都漏掉，它在下次评估中应该被特别 flag**。

具体到 Phase 1.3——Phase 1.2 时双方都漏的"13× 慢"现象，在 Phase 1.3 实验设计里有专门实验（§3.3），且在评估 Phase 1.3 报告时双方必须明确回答"这个现象是否被解释了"。

如果 Phase 1.3 完成后这个现象**再次**没被认真讨论，那么工作流本身需要被修订——也许需要引入第三个评估者，或者引入"明确盲点检查"步骤。

### 2.2 关于 cross-round complementarity 的观察

Phase 1.2 评估后我和 ChatGPT 跨 round 互补——我前一轮注意到 13× 慢现象，ChatGPT 在看到这个观察后接过去做了实质性深化（"basin"概念污染、WPE shutoff 实验、tail convergence rate）。这种 cross-round complementarity 不是 §2 工作流设计时明确预期的收益，但实战中证明很有价值。

**对 Phase 1.3 的含义**：实验 J 的设计直接吸收了这个 cross-round 互补的产物。Claude Code 在执行时应该把实验 J 看作"经过两轮独立审视后的最稳设计"，不要因为"看起来过于复杂"而简化。

---

## 3. 核心实验设计

### 3.1 实验顺序与分组

Phase 1.3 有 5 个实验。**它们不是单线 1→5**，而是分两组：

**Group Cheap**（3 个实验，半天到一天打包做完）：
- 实验 I：Markup-type triangulation
- 实验 J：HTML-induced structure 本体诊断（多阶段实验）
- 实验 K：Layer-by-layer LayerNorm σ_max

**Group Expensive**（2 个实验，各 1-3 天）：
- 实验 L：LayerNorm γ/state decomposition
- 实验 M：Spectral radius measurement

**执行原则**：

1. **先把 Group Cheap 三个全做完**——这是已有 infrastructure 加少量新代码，可以打包推进
2. 看 Group Cheap 结果决定 Group Expensive 哪个先做
3. **关键决策**：
   - 如果实验 I 显示 markup 不只是 angle-bracket（其他 brackets 也触发结构），先做 L
   - 如果实验 J 显示 HTML structure 长 trace 漂移或对 wpe 切断敏感，先做 M（用 spectral radius 量化吸引强度差异）
   - 如果两者都给出 stable Phase 1.2 验证，按 L → M 顺序

Claude Code 在 Group Cheap 完成后**必须停下**向项目作者请示 Group Expensive 顺序，不要自己决定。

### 3.2 实验 I：Markup-type Triangulation（决定 Q1，半天）

**问题**：Phase 1.2 H 显示"markup 是 sole basin selector"，但 markup 仅指 `<...>`。这是 (a) 一般 markup 现象，还是 (b) 仅 angle-bracket BPE tokens 现象？

**实施**：

构造 30 个测试输入，按以下设计：

| 类别 | 例子 | n |
|---|---|---|
| Angle brackets (baseline) | `<tag>text</tag>` | 5 |
| Square brackets | `[tag]text[/tag]` | 5 |
| Curly braces | `{tag}text{/tag}` | 5 |
| Parentheses | `(tag)text(/tag)` | 5 |
| Quotes | `"tag"text"tag"` | 5 |
| Isolated single chars | `<` only, `>` only, `[` only 等 | 5 |

每个输入跑 mode-A iteration（max_iter=1000，seq_len=64），记录 final state 与 lowercase attractor、HTML metastable structure 的 cos 距离。

**关键判据**：

- 如果只有 angle brackets 进入 HTML metastable structure → "markup-induced" 实际是 "angle-bracket BPE token-induced"。Phase 1.2 H 的 framing 需要 narrowing
- 如果其他 brackets 也进入某个**不同**的 structure（不是 lowercase 也不是 HTML） → 多结构地形是真实的，每种 bracket type 可能对应自己的吸引结构
- 如果只有 `<...>` 完整 tag pattern 触发，单独 `<` 或 `>` 不触发 → 触发条件是 token 序列**模式**而非单 token

**输出**：
- 30 个 input 的 final state 和距离两个已知吸引结构的 cos
- 每种 bracket type 的实际 BPE token IDs 列表
- 子报告 `phase1_3_markup_triangulation.md`

**预期成本**：30 秒计算 + 半天分析。

### 3.3 实验 J：HTML-induced Structure 本体诊断（决定 Q4，1-2 天）

**问题（被双盲点漏掉的现象正面处理）**：原 Phase 1.2"capital basin"是真 fixed point、metastable transient、limit cycle、还是 wpe-forced equilibrium？

实验 J 是 Phase 1.3 信息密度最高的实验。它有**四个子阶段**，所有阶段共用 forward pass infrastructure。

#### 3.3.1 J.1 — Long-horizon stability test（决定它是不是 transient drift）

从 Phase 1.2 H 的 12 个 HTML-markup 输入和 12 个 lowercase 输入中各取 4 个代表，跑超长 trace：

```
max_iter = 10000
threshold = 1e-5  # 比 Phase 1.2 严格 100 倍
save_every = 100  # 保留中间状态便于轨迹分析
```

**测量指标**（每 100 步记录一次）：

- `‖h_n - h_{n-1}‖`（绝对残差）
- `‖h_n - h_{n-1}‖ / ‖h_n‖`（相对残差）
- `cos(h_n, h_{lowercase})`（与 lowercase attractor 的 cos）
- `cos(h_n, h_{HTML_129})`（与 step 129 时 HTML state 的 cos）
- `effective_rank(h_n)`
- top-5 logits 投影
- `KL(softmax(logits_n) || unigram_prior)`

**关键判据**：

| 长 trace 行为 | 解读 |
|---|---|
| step 130-10000 残差稳定在 < 1e-4 | HTML structure 是真 fixed point |
| step ~500-2000 后开始单调漂向 lowercase | HTML 是 metastable transient（ghost attractor） |
| top logits 在几个 token 间周期切换、但 hidden 残差稳定 | hidden-space 稳定但 logits 不稳定 |
| `‖h_n - h_{n-2}‖ < ‖h_n - h_{n-1}‖`（隔一步更近） | 存在 limit cycle |
| 残差不降但 cos 数据稳定 | hidden 沿某流形漂移但语义稳定 |

#### 3.3.2 J.2 — WPE shutoff after capture（决定 wpe 是否持续维持）

这是关键诊断实验。流程：

1. 用 normal mode-A 从 HTML 输入跑到 step 200（确保已经"在"HTML structure 内）
2. **从 step 200 开始切换到 cancel-pos 模式**（继续不重新加 wpe）
3. 继续跑 1000 步
4. 监测 `cos(h_n, h_{HTML_200})` 和 `cos(h_n, h_{lowercase})`

**关键判据**：

| 切换 wpe 后行为 | 本体地位 verdict |
|---|---|
| 仍稳定停在原位置（cos 不变） | HTML structure 是 trained-blocks 内生属性，wpe 只起初始化作用 |
| 慢慢漂移到 lowercase（cos 单调上升朝 1.0） | HTML structure 依赖 wpe **持续维持**，是 wpe-forced equilibrium |
| 立刻（< 50 步）掉到 lowercase | HTML structure 基本是 wpe artifact，没有内生支撑 |
| 进入震荡 | wpe 与主收缩力构成强耦合系统，关掉 wpe 释放振荡模式 |

J.2 的诊断力比 J.1 强一档——J.1 只能告诉我们"在持续 wpe forcing 下是否稳定"，J.2 直接区分"内生 vs wpe-依赖"两种本体。**两个一起做才完整**。

#### 3.3.3 J.3 — Tail convergence rate fitting（量化吸引强度差异）

利用 J.1 的长 trace 数据（已存在不需要重跑），在收敛 tail 段拟合：

```
log ‖h_n - h*‖ ≈ a + n · log λ
```

得到有效收敛率 λ。关键比较：

- λ_lowercase（lowercase attractor 的 tail rate）
- λ_HTML（HTML structure 的 tail rate）

**关键判据**：

| λ 比较 | 解读 |
|---|---|
| λ_HTML ≈ λ_lowercase | 两者吸引强度相同，129 vs 10 差异来自起点距离而非 attractor 性质 |
| λ_HTML >> λ_lowercase（如 0.95 vs 0.2） | HTML 是弱吸引子（局部 spectral radius 接近 1），lowercase 是强吸引子 |
| λ_HTML 不稳定（拟合不出干净直线） | HTML 不是 exponential convergence，可能是 power-law 漂移或 oscillatory |

J.3 把"129 步 vs 10 步"这种粗糙数字升级为可量化的吸引强度比较。**几乎免费**——已有 J.1 trace 数据就能算。

#### 3.3.4 J.4 — Cycle diagnostic 副产品（检查 limit cycle）

利用 J.1 的长 trace 数据，对每个 trace 计算 `‖h_n - h_{n-k}‖` for k ∈ {1, 2, 3, 4, 8, 16}。如果某个 k > 1 的距离比 k=1 更小，说明存在周期 k 的 cycle。

**关键判据**：

- 所有 k 的距离单调递增 → 没有 cycle 信号（最常见）
- 某个 k 的距离明显小于 k=1 → 存在 limit cycle，周期为该 k
- 距离都大致相等 → 可能是慢漂移而非 fixed point

J.4 是 J.1 数据的副产品，零额外计算。如果 J.4 显示 cycle 信号，留作 Phase 1.4 候选深挖；如果没有，作为 negative 信号记录。

#### 3.3.5 实验 J 整体输出

- 8 个长 trace（4 HTML + 4 lowercase，10000 步）的完整中间状态（每 100 步存）
- 4 个 HTML trace 在 step 200 后切换 cancel-pos 的 1000-step 续跑数据
- λ_lowercase 和 λ_HTML 的拟合
- ‖h_n - h_{n-k}‖ for k ∈ {1, 2, 3, 4, 8, 16} 的轨迹
- 子报告 `phase1_3_basin_diagnosis.md`，覆盖 J.1-J.4 全部

**预期成本**：2-3 小时 GPU（J.1 + J.2）+ 1 天分析。这是 Phase 1.3 GPU 时间最长的实验。

### 3.4 实验 K：Layer-by-Layer LayerNorm σ_max（半天）

**问题**：Phase 1.2 F 只测了 layer 6 的 ln_1 / ln_2 σ_max。所有 12 层的分布是什么？是均匀的还是某些层特别强收缩？

**实施**：

在 Phase 1.2 F 的同样 10 个测试输入上，对所有 12 个 transformer block 各测 ln_1 和 ln_2 的 σ_max（at h_fixed only，h_0 / h_1 不必扩展）。

```
n_layers = 12
n_inputs = 10
n_modules_per_layer = 2  # ln_1, ln_2
total = 12 * 10 * 2 = 240 power iterations
```

**关键测量**：

- 每层的 ln_1 / ln_2 σ_max 分布
- trained vs random 在每层的 gap
- σ_max 是均匀小（每层都收缩），还是集中在少数层（某些层强收缩、其他层接近 1）？

**关键判据**：

- **均匀分布**（所有层 σ_max ≈ 0.12-0.20）→ contraction 是分布式的，所有层共同贡献
- **集中分布**（少数层 σ_max < 0.1，其他层 ≈ 1.0）→ contraction 主要由少数 layer 驱动
- **梯度分布**（早层 σ_max 大、晚层 σ_max 小，或反过来）→ contraction 在网络深度上有方向性

**为什么便宜**：power iteration 的代码已经写好（Phase 1.2 F 的 `module_jacobian.py`），只需把 layer index 从 6 扩展到 0..11。

**输出**：
- 12 × 2 = 24 个 σ_max 分布
- 与 random init 对比
- 子报告 `phase1_3_layer_jacobian.md`

**预期成本**：30 分钟 GPU + 半天分析。

### 3.5 实验 L：LayerNorm γ/State Decomposition（决定 Q2，1-2 天）

**问题**：trained LayerNorm σ_max ≈ 0.12 是因为 (a) learned γ < 1，(b) fixed-point hidden state norm 大让 1/σ_x 小，还是 (c) 两者耦合？

**实施**：

LayerNorm 的 Jacobian 形式：

```
∂y/∂x = (γ / σ_x) * (I - 1/d · 11ᵀ - (x-μ)(x-μ)ᵀ / (d·σ_x²))
```

σ_max 主要由 `γ / σ_x` 这个 prefactor 决定。Phase 1.2 显示 σ_max ≈ 0.12，但没分解 γ 和 σ_x 各贡献多少。

做四组对照实验：

| 条件 | γ | hidden state | 测什么 |
|---|---|---|---|
| **A1** | trained | 原 h_fixed (norm ≈ 2563) | Baseline，匹配 Phase 1.2 F |
| **A2** | γ=1 | 原 h_fixed | 看 hidden state norm 单独贡献 |
| **A3** | trained | rescaled h_fixed (norm 缩到 100) | 看 learned γ 单独贡献 |
| **A4** | γ=1 | rescaled h_fixed | Baseline 对照（应该 ≈ 1.0） |

**关键测量**：分别算 4 个条件下 ln_1 的 σ_max，比较：

- **σ_max(A1) / σ_max(A4)**：总收缩强度
- **σ_max(A2) / σ_max(A4)**：state norm 单独贡献
- **σ_max(A3) / σ_max(A4)**：γ 单独贡献
- **σ_max(A1) / [σ_max(A2) × σ_max(A3) × σ_max(A4)⁻¹]**：耦合项

**关键判据**：

- 如果 **σ_max(A3) ≈ σ_max(A1)**（γ 单独足够给出 0.12）→ learned γ 是主要 contraction source，Phase 1.2 报告的因果归因正确
- 如果 **σ_max(A2) ≈ σ_max(A1)**（state norm 单独足够给出 0.12）→ contraction 主要来自 fixed-point 的大 norm，γ 不是关键。Phase 1.2 报告的归因需要修正
- 如果 **σ_max(A2) ≈ σ_max(A3) ≈ √(σ_max(A1))**（两者各贡献一半，乘性耦合）→ 是耦合现象，γ 和 state norm 都不可单独解释收缩

**工程注意**：

- "γ=1" 的实现：用 `model.transformer.h[6].ln_1.weight.data.fill_(1.0)`（在副本上），完成实验后恢复
- "rescaled h" 的实现：`h_rescaled = h_fixed * (target_norm / h_fixed.norm())`，target_norm 取 100（与 normal h_0 / h_1 量级一致）
- 必须用模型副本（`copy.deepcopy(model)`），不要污染主模型

**输出**：
- 4 个条件下的 σ_max 数值
- decomposition 归因
- 子报告 `phase1_3_ln_decomposition.md`

**预期成本**：1-2 天（4 组实验 × 10 输入 ≈ 1 小时 GPU + 工程实现 + 分析）。

### 3.6 实验 M：Spectral Radius Measurement（决定 Q3，2-3 天）

**问题**：Phase 1.2 用 power iteration on J^T·J 测 σ_max（spectral norm）。决定动力学稳定性的是 ρ(J) = |λ_max(J)|（spectral radius），不是 σ_max。两者可能数值差距很大，特别是 J 不对称时。

具体在 Phase 1.2 中——trained full_stack_posfree σ_max = 16.3 但实验 100% 收敛，意味着 ρ(J) 远小于 16.3。这个 gap 多大？

**实施**：

用 J 本身的 power iteration（不是 J^T·J）。这测最大特征值的模 |λ_max|。

```python
# 伪代码
v = random_unit_vector
for _ in range(max_iter):
    v_new = J @ v  # 用 jvp，不构造 J 矩阵
    lambda_est = (v_new @ v) / (v @ v)  # Rayleigh quotient
    v = v_new / v_new.norm()
return abs(lambda_est)
```

**注意工程难点**：

- J 不对称时，power iteration 收敛速度依赖 |λ_1| / |λ_2| 比值。如果 |λ_2| 接近 |λ_1|，收敛慢
- 复特征值（J 不对称）会让 power iteration 给出**振荡**而非收敛——可能需要用 shifted power iteration 或 Arnoldi 方法
- 数值稳定性：double precision 是必须的

**Fallback**：如果直接 power iteration 不收敛，用 Arnoldi 方法（scipy.sparse.linalg.eigs）在低维 Krylov 子空间里找特征值。

**测试范围**：

- Phase 1.2 F 的 10 个输入
- 三个测试点：h_0、h_1、h_fixed
- 子模块：ln_1、attn-sublayer、mlp-sublayer、full block、full stack
- trained 和 random init

总计 10 × 3 × 5 × 2 = 300 次 spectral radius 测量。

**关键判据**：

- 比较 ρ(J) vs σ_max(J)（用 Phase 1.2 F 的数据）
- 如果 ρ(J) << σ_max(J)（比如 σ_max=16 但 ρ=0.5）→ spectral norm 严重高估了动力学影响。Phase 1.2 的"asymmetric coupling"解读得到强支持（attention 沿某些方向 stretch 但不影响 stability）
- 如果 ρ(J) ≈ σ_max(J) → spectral norm 是动力学的合理代理。Phase 1.2 的解读不需要修正
- 关键的 sub-module：ln_1 的 ρ vs σ_max。如果两者都 << 1，"LayerNorm 收缩"的归因就严格了。如果 ρ 比 σ_max 大，需要重新审视

**为什么这是最贵的实验**：power iteration on 不对称 J 的工程难度比对称的 J^T·J 高一个数量级。可能需要尝试多个 numerical methods 直到找到一个 robust 的。

**输出**：
- ρ(J) vs σ_max(J) 对照表
- 对 Phase 1.2 解读的影响评估
- 子报告 `phase1_3_spectral_radius.md`

**预期成本**：2-3 天（包括 numerical method 探索 + 实验跑完 + 分析）。

---

## 4. 数据记录与报告规范

延续前几个 phase。每个实验保存：完整 hidden state 数据（实验 J 特别多）、配置 JSON、git commit、Jacobian power iteration logs（实验 K/L/M）。

每个实验写独立子报告，最后写整合报告 `phase1_3_master_report.md`。

整合报告必须包含：

- 对 Q1-Q5 的回答（带证据强度）
- Phase 1.2 机制图像的最终 verdict（验证、修正、推翻）
- HTML-induced structure 的本体地位 verdict（fixed point / metastable / limit cycle / wpe-forced）
- 13× 慢之谜的具体解释
- 对 essay 修订的影响（注：Phase 1.3 不实施修订，但要列出修订点供 Phase 1.x 总结时使用）
- 是否进入 Phase 1.4 还是开始 Phase 2 / FPP-Native 讨论的建议

### 4.1 反馈包

按 §2 工作流，整合报告完成后准备 zip 包发给评估者。**不附其他 AI 回应**。

---

## 5. 分阶段验收门槛

### Phase 1.3 - Group Cheap（实验 I+J+K，约 2-3 天）

完成定义：三个实验全部跑完，三份子报告完成。

**通过门槛**：
- 实验 I：能就 Q1（markup vs angle-bracket）给出回答
- 实验 J：能就 Q4（HTML structure 本体地位）和 13× 慢之谜给出回答
- 实验 K：能就"contraction 在层间分布"给出回答

完成后**停下来向项目作者请示**，决定 Group Expensive 的顺序。

### Phase 1.3 - 实验 L（约 1-2 天）

完成定义：4 组对照全部跑完，子报告完成。

**通过门槛**：能就 Q2（γ vs state norm 的 decomposition）给出回答。

### Phase 1.3 - 实验 M（约 2-3 天）

完成定义：spectral radius 测量在所有 sub-modules / 测试点完成（如果 numerical method 卡住，至少完成 ln_1 + full block 这两个最关键的）。

**通过门槛**：能就 Q3（spectral norm vs spectral radius gap）给出回答。

### Phase 1.3 - 整合（约 1 天）

完成定义：整合报告 + 反馈包准备好。

**通过门槛**：报告诚实记录，13× 慢之谜有了答案，Phase 1.2 机制图像的最终 verdict 明确（无论 verdict 方向），HTML structure 本体地位有 verdict。

**总时长预算**：6-9 天 wall-clock。GPU 实际计算时间 < 5 小时（实验 J 是最长的 2-3 小时 + 实验 M 的 numerical exploration）。

---

## 6. 已知工程坑

### 6.1 沿用前几个 phase 的所有坑

参考前置文档。所有 dtype / attention_mask / position_ids / 数值稳定性 / GPU 内存等坑仍然适用。

### 6.2 Phase 1.3 特有的坑

**实验 I 特有**：
- "Square brackets" `[tag]` 在 GPT-2 BPE 里可能被切碎为多个 token（`[`、`tag`、`]`），具体的 token IDs 决定 selection 行为
- 必须在子报告里附每种 bracket type 的实际 token IDs，让"是 token 触发还是 character 触发"这个区分有数据依据

**实验 J 特有（最重要）**：

- **J.1 长 trace 数值稳定性**：fp32 下 10000 步可能漂移。每 100 步检查 `‖h_n‖` 是否爆炸/衰减异常。如果 norm 变化超过初始值 10× 或 0.1×，标记为"数值不稳定"而非"动力学发散"
- **J.1 不要 early stop**：哪怕 step 200 后看似收敛，也跑满 10000 步。否则错过 metastable drift 信号
- **J.2 切换 wpe 模式时的 hidden state 一致性**：从 normal mode-A 切换到 cancel-pos 时，h_200 是已经加过 wpe 的状态。在 cancel-pos 模式下喂回时，要确保 wpe 不被再次加上
- **J.3 tail rate 拟合的稳定性**：拟合区间应该在收敛后段（lowercase 用 step 5-10，HTML 用 step 100-300），太早会被起始 transient 污染。如果发现拟合 R² < 0.9，标注为"非 exponential 收敛"
- **J.4 cycle 检测的 false positive**：在 norm 衰减的早期阶段，所有 ‖h_n - h_{n-k}‖ 都会比较小，可能伪造 cycle 信号。只在 stable 段（J.1 的后半）做 cycle 检测

**实验 K 特有**：
- 12 层每层都要 power iteration，注意每层的 Jacobian 维度（但 GPT-2 hidden_dim = 768 在每层都一样）
- 不同层的 LayerNorm γ 数值范围可能差很大，不要假设所有层都在同一个 σ_max 范围

**实验 L 特有**：
- 修改 LayerNorm γ 时**必须使用模型副本**（`copy.deepcopy(model)`）
- "rescaled h_fixed" 的具体 norm 选择有自由度，建议选 100，在子报告里说明
- LayerNorm 的 β（bias）也是 learned，实验 L 只控制 γ，β 保持原样

**实验 M 特有（工程上最难）**：
- 不对称矩阵的 power iteration 数值不稳定，可能出现：
  - 收敛到正确特征值但慢
  - 振荡（复特征值）
  - 收敛到次大特征值（如果初始向量与最大特征向量正交）
- **建议**：先在小规模（单 LayerNorm，d=768）上验证 numerical method，再扩展到 full stack
- **Fallback 方案**：如果 power iteration 在某些位置不收敛，用 scipy.sparse.linalg.eigs（Arnoldi）在 Krylov 子空间求 top-k 特征值
- 计算成本：full stack 的 J 是 (seq_len × hidden_dim)² 维，但用 jvp 可以避免显式构造

### 6.3 整体的元层次坑

延续前几个 phase 的纪律。Phase 1.3 特有的几个：

**"已经验证了 Phase 1.2 图像"的诱惑**——如果实验 I 显示只有 angle bracket 触发、实验 J 显示 HTML structure 长 trace 稳定且 wpe 切断后仍存在、实验 L 显示 γ 是主因、实验 M 显示 ρ << σ_max——这种"全部验证"的体验本身是个警告。**抑制把这种 coherence 当作"机制确认"**。

**对实验 J 的特殊警惕**——实验 J 是 Phase 1.3 信息密度最高的实验，也是最容易过度解读的。J.1 + J.2 + J.3 + J.4 四个子阶段如果给出"完美"的协同结论（HTML 是 wpe-forced，wpe 切断后立即漂移，λ_HTML 接近 1.0，无 cycle 信号）——这种完美本身值得多审视一次，因为这正是实验设计预期会"想看到的"结果。**严格区分"实验给出预期 positive"和"实验给出独立 positive"**。

**"FPP-Native 终于该上场了"的诱惑**——Phase 1.x 完成后会有强诱惑说"现在可以开始造 FPP-native 系统"。**Phase 1.3 不允许这个跳跃**。

---

## 7. 硬件约束

延续前几个 phase。4070 Ti 12GB + 64GB RAM 完全胜任。

Phase 1.3 GPU 时间估计：
- 实验 I：30 秒
- 实验 J：2-3 小时（J.1 长 trace 1-2 小时 + J.2 续跑 30 分钟）
- 实验 K：30 分钟
- 实验 L：1 小时
- 实验 M：2-4 小时（含 numerical method 探索）

总 GPU 时间约 5-8 小时。瓶颈仍然在分析阶段。

---

## 8. 数据与代码组织

延续前几个 phase。新代码进 `src/`：

```
src/
├── markup_triangulation.py      # 实验 I
├── basin_diagnosis.py           # 实验 J（J.1 + J.2 + J.3 + J.4 整合）
├── layer_jacobian.py            # 实验 K
├── ln_decomposition.py          # 实验 L
└── spectral_radius.py           # 实验 M（最复杂）
```

实验 J 的 J.1-J.4 共用一个模块，因为它们共用 forward pass 和数据存储 infrastructure。J.3 和 J.4 是 J.1 的 post-hoc 分析，不需要额外 GPU 时间。

---

## 9. 给 Claude Code 的开发建议

### 9.1 推进顺序

**严格按 §3.1 的分组**：先 Group Cheap (I+J+K)，停下请示，再 Group Expensive (L 或 M)。

**实验 J 在 Group Cheap 内的子顺序**：J.1（长 trace）→ J.2（wpe 切断续跑）→ J.3（tail rate 拟合，post-hoc）→ J.4（cycle 检测，post-hoc）。J.1 必须先完成才能做 J.2 / J.3 / J.4。

实验 M 在工程上是 Phase 1.3 最难的。如果在 numerical method 上卡住超过半天，**停下来请示**——可以接受 fallback 到 scipy.sparse.linalg.eigs，可以接受只测部分子模块，但不要自己决定降级策略。

### 9.2 在不确定时的默认选择

延续前几个 phase——保守、简单、可解读。

特别到 Phase 1.3：
- 实验 J 的长 trace：宁可保存太多中间状态（每 100 步存）也不要丢数据
- 实验 J.2 的 wpe 切换：先用 cancel-pos 变体（更便宜），如果有时间再补 posfree 变体
- 实验 L 的"rescaled h"：用 norm=100 这个简单值，不要追求"最自然"的归一化
- 实验 M 的 power iteration：如果不收敛，提前停止给上界估计而不是 retry 100 次

### 9.3 何时停下来问

**新增的停下来问的情况**（Phase 1.3 特有）：

- 实验 I 给出"只有 angle bracket 触发"和"多种 bracket 都触发"之外的第三种结果（比如某些 bracket 触发新结构）—— 设计本来没考虑这种情况
- 实验 J.1 显示 HTML structure 在 step ~500 漂移，但漂移方向**不是**朝 lowercase —— 暗示有第三个我们不知道的 attractor
- 实验 J.2 的 wpe 切换给出意外结果（比如 HTML structure 在切换后**强化**而不是弱化或保持）—— 完全超出四种预期 verdict
- 实验 J.4 cycle 检测显示某个 k > 1 的距离明显小，但不是 k=2 这种简单偶分裂 —— 复杂周期结构需要深入分析
- 实验 M 的 spectral radius 显示 ρ > 1（应该 < 1 才对，否则系统不该收敛）—— numerical bug 或者更深的发现
- 任何超过 §5 时间预算 50% 的工程问题

### 9.4 不要做的事

- 不要为了节省时间跳过 Group Cheap 直接做 Group Expensive
- 不要在 Phase 1.3 内修订 essay
- 不要扩展实验范围（不做 cross-architecture / scale ladder）
- 不要把 Phase 1.2 的解释当作既定事实——本 phase 的目的是验证它，不是基于它推进
- 不要简化实验 J 为单一阶段（J.1 + J.2 + J.3 + J.4 必须全部做）
- 不要省略 negative 结果

---

## 10. 不修订 essay 的理由（明确说出）

Phase 1.1 完成后我们承诺修订 essay。Phase 1.2 完成后我们再次承诺修订。Phase 1.3 完成后**不再承诺修订**。

理由：

1. **多轮小修会让 essay 变成补丁堆叠**。Section 7 的"Update from initial probes"已经写过两轮，再加一轮会让它比原文更长。
2. **Phase 1.x 还没完成**——如果实验 J 显示 HTML structure 是 metastable transient、Phase 1.2 的"两个 basin"图像需要修正，那 essay §4 的"context, attention, normalization"修订就要再调整。等 Phase 1.x 真正完成再统一处理更干净。
3. **文章的姿态本来就是"提出方向"而不是"实时报告进度"**。每次实验都更新 essay 让它变得越来越像研究 log，偏离 essay 体裁。

具体做法：

- Phase 1.3 整合报告里**列出**对 essay 的修订点（哪一节、改什么），但**不实施**
- Phase 1.x 完成后写"FPP Phase 1.x Summary"独立文档，整合所有 mechanism findings
- **基于那份 Summary**重写 essay §4 + §7 + §10，做一次干净的修订

如果 Phase 1.3 给出意外的 reversal（彻底推翻 Phase 1.2 图像），可以提前修订——但要明确这是 reversal 触发的，不是常规迭代。

---

## 11. Meta 层面的提醒

Phase 1.3 的设计有一部分动机是处理工作流自身的盲点（"13× 慢"被两次审视都漏掉）。这件事本身值得被记录为方法论 record：

**两个独立 AI 评估者可能有共同盲点**——这次的盲点是被"两个干净 basin"的简洁性 attract 走，没注意到这两个所谓 basin 在动力学性质上**根本不对等**。这种"共同 framework 内的共同盲点"是 §2 工作流的剩余风险。

**对策**：

1. 实验 J 把这个盲点变成正式实验
2. §2.1 添加"两次审视都漏的现象在下次评估必须被特别 flag"
3. §0.1 引入精确命名，避免"basin"这种语言层面的概念污染
4. 长期看，可能需要引入第三个 AI 评估者（或某种 adversarial probe）来识别共同 framework 内的盲点

**Phase 1.3 完成后，应该评估这套工作流本身的有效性**——如果 13× 慢之谜被实验 J 干净解决，工作流仍然有效；如果它揭示的现象超出当前 framework，那么工作流本身需要演化。

**关于 cross-round complementarity**——Phase 1.2 评估后我和 ChatGPT 在 cross-round 互补，让实验 J 的设计比单一 AI 能给出的版本扎实得多。这种 cross-round complementarity 是个意外收益。但要警惕——**工作流自身正在变得越来越好**这种感觉本身可能是个警告。每一轮都在制度化纪律、每一轮的发现都在前一轮基础上深化——这种"工作流越来越成熟"的体验可能让我们对工作流本身的信任过度增长，掩盖某种系统性偏差。Phase 1.x 总结报告时应该单独评估"工作流自身是否需要演化"。

延续前几个 phase 的所有 meta 警告：

- "鼓舞效应"——某个干净的发现让研究心理向 positive 漂移
- "一切都解释通了"——这种感觉本身是警告而非证据
- "过度因果归因"——小样本（24 输入、10 traces）不支持强 causal 主张
- "局部 vs 全局 claim 混淆"——Jacobian 是局部的，不能直接外推到全局
- **新增**："basin"等语言层面的概念污染——同一个词描述动力学性质完全不同的对象会让理论不严格

---

## 12. 文档版本历史

- **v0.2**（2026-04-27）：初始执行版本。基于 Phase 1.2 master report、Claude 与 ChatGPT 独立评估的合并、特别是双方都漏的"13× 慢"盲点和 ChatGPT 后续在看到 Claude 观察后的进一步深化（"basin" 概念污染、WPE shutoff after capture 实验、tail convergence rate fitting）。覆盖 5 个实验：markup-type triangulation、HTML-induced structure 多阶段诊断（J.1 长 trace + J.2 wpe 切断续跑 + J.3 tail rate 拟合 + J.4 cycle 检测）、layer-by-layer LayerNorm、γ/state decomposition、spectral radius。引入 Group Cheap / Group Expensive 分组执行、§0.1 精确命名约定、§10 essay 修订冻结、§11 工作流盲点反思与 cross-round complementarity 观察。

预期在 Phase 1.3 整合报告完成后更新到 v0.3，根据实际数据决定 Phase 1.x 总结的时机和形式。
