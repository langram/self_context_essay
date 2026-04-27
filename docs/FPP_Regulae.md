# FPP Regulae v0.2 — 项目研究方法论

**项目代号**：FixedPointProbe (FPP)
**文档版本**：v0.2
**日期**：2026-04-27
**性质**：方法论 record，非技术方案
**目标读者**：项目作者、AI 协作评估者（Claude / ChatGPT）、未来研究者

---

## 0. 文档定位与根本姿态

### 0.1 文档定位

这份文档是 FPP 项目从 Phase 1 到 Phase 1.3 实战中暴露的**真实失败模式 + 对应的可执行 discipline** 的成文化。每条 Regula 都有具体 case study 作为 grounding。

文档刻意保持简短。7 条 Regula 是上限——更多规则会让方法论本身变成另一种本轮系统。

文档自身有边界：基于 Phase 1.x 实战经验，未必适用于所有 ML 研究；可能在执行 1-2 phase 后被实战暴露不足。这件事 upfront 说出来比假装权威更诚实。

### 0.2 根本姿态：Invariant-first, fit-second

这是所有 7 条 Regula 的根本组织原则。**所有 Regula 都是这一姿态的具体化，不是 7 个并列检查项**。

具体说：

- **先找不变量**：什么现象可能跨模型、跨编码、跨任务、跨状态空间保持？
- **再做拟合**：当前 GPT-2 small / Mode-A 结果如何解释？
- **最后才理论化**：哪些局部解释有资格上升为 FPP 理论的一部分？

一句话——**拟合是局部工具，不变量才是理论候选**。

为什么这个姿态比"7 条规则"更根本——Regulae 最容易变成 cosmetic compliance（每个 phase 报告里机械填 7 条 checklist）。**只有"先找不变量"这种姿态被内化，Regula 才不会退化为本轮的一种新形式**。

### 0.3 Regulae 防御的两类风险

FPP 项目面对两类风险：

**第一类：研究本身的本轮化（Ptolemization of research）**

研究者持续修正现有 framework 来 fit 新数据，每次修正看似合理，累积起来是越来越复杂的 framework，但**研究对象本身可能选错了**。FPP 的具体风险——一直在 GPT-2 small mode-A 这个特定 setup 上工作，所有修正都在 setup 内做。

**第二类：评估流程的本轮化（Ptolemization of evaluation）**

评估者持续修正现有评估 framework 来处理新出现的盲点，每次都更严格，但**评估流程本身可能在错误层级工作**。把多 AI 评估当作"独立性 + 互补性"够了，可能漏掉一个更基础的事实——所有 LLM 评估者都有共同的拟合倾向。

Regulae 同时防御这两类风险。

---

## 1. 七条 Regulae

### Regula I — 最少假设

**"每个 phase 整合报告必须包含 'minimum-component alternative' 节"**

具体形式：给出能 explain 同样观测数据的最少假设替代解释。如果当前 framework 的 component 数远超最少 alternative，必须显式说明为什么需要这些额外 component。

**Case study (Phase 1.3)**：master report 给的机制图像有 7+ component（lowercase strong attractor + HTML wpe-supported equilibrium + third intermediate regime + LayerNorm contraction + attn/mlp expansion + asymmetric coupling + ...）。Regula I 强制问"是否存在 3-component 解释"。后来 J.3 拟合错误暴露时，"HTML 不是真吸引子" component 被推翻，但之前没有 minimum alternative 节，错误直接污染下游。

**Anti-pattern**：把 minimum alternative 节写得 cosmetic（"alternative：宇宙是模拟"），用形式 compliance 应付。

---

### Regula II — 同效同因

**"声称为 'transformer mechanism' 的 claim 必须通过 cross-architecture 实验。声称为 'semantic invariant' 的 claim 必须通过同构样本测试"**

具体形式：

- "transformer mechanism" claim：在至少一个其他架构（Pythia / LLaMA / 甚至 RNN）上做最小复现，验证现象不是 GPT-2 specific
- "semantic invariant" claim：同一抽象结构通过多种编码（HTML / JSON / Lisp / 自然语言）测试，验证不是 surface encoding artifact

**Case study (Phase 1.1-1.3)**：所有 phase 都只在 GPT-2 small 上做。"LayerNorm 是 transformer contraction source" 这种 claim 被 phrasing 成 transformer general property，但 evidence scope 只有 GPT-2 small。Phase 1.4 之前必须做一个 cross-architecture 对照，否则 verdict 不能从 candidate 升级。

**Anti-pattern**：用"留作 future work"无限延后。Regula II 要求**Phase 1.x 完成前**至少做一次最小 cross-arch replication。

---

### Regula III — 证据范围标注

**"每个 verdict 必须明确标注 'evidence scope'，从 evidence scope 外推必须有显式 justification"**

具体形式：每个 claim 后面跟 (scope: X inputs on Y model with Z setup)。如果 claim 想超出 scope，必须明说"我们 generalize 到 [更广 X]，依据是 [具体 reason]"。

**Case study (Phase 1.3)**：master report 的 "Phase 1.2 two-attractor framing was wrong" 是基于 4 HTML rep + 4 lowercase rep 的 J.3 拟合得出的。Evidence scope 是 8 个 traces，但被 phrasing 成对整个 framework 的颠覆。J.3 拟合错误污染下游 verdict。

**Anti-pattern**：把 scope 标注做成 boilerplate（每个 claim 都加同样 scope label）。要求每个 claim 的 scope 是具体不同的。

---

### Regula IV — 可证伪条款

**"每个 verdict 必须包含 explicit 'what would falsify this' 条款"**

具体形式：report 里每个核心 claim 后明说"以下观测会 falsify 它：[具体可观测信号]"。如果 verdict 后续被 falsify，trigger 明确的"verdict revoked"协议而不是悄悄 corrected。

**Case study (Phase 1.3)**：master report 的 "HTML is wpe-supported marginal equilibrium" 没有可证伪条款。J.3 拟合错误暴露时，没有 protocol 让这个 verdict 被显式 revoke——它只是被 corrigendum 替换。

**Anti-pattern**：可证伪条款写得 vague（"如果数据完全不同"）。要写具体可观测信号。

---

### Regula V — 远处工作（oscillation discipline）

**"每个评估者必须在每次评估中完成远处工作。这是责任，不是 contribution"**

具体形式——这不是写"近处：xxx 远处：xxx"的两段式模板，是**在关键判断时强制切换 mode**：

- **托勒密检查**：当前解读是不是在已选对象上加本轮？如果换对象/坐标系/层级，这些数据怎么解读？
- **不变量假设**：如果 verdict 在另一架构/编码/任务上 replicate，最可能哪部分被推翻？

如果评估者答不出来（"我不知道还能换什么对象"），必须明说"我在 framework 内、需要外部 perspective"——这本身是 valuable signal。

**关于责任分配的明确**：远处工作是**每个 AI 评估者的 mandatory responsibility**。不能把这件事 outsource 给用户或其他评估者。即使做得不好，也必须自己尝试做。**用户的 perspective injection 是 valuable additional input，不是远处工作的 load-bearing element**（参见 §3 D3）。

**Case study (Phase 1.1-1.3)**：所有评估都在做近处工作，从未显式做远处检查。直到项目作者主动 push back（"我们是不是在错误坐标系"），才有人做这件事。Regula V 让这个动作不依赖 ad hoc 的人类介入，而是每个评估者每次必做。

**Oscillation 不是模板**：这件事 ChatGPT 在 v0.1 review 中的关键洞察——双焦距框架（A/B/C/D 五段式）即使是 ChatGPT 自己提议的，仍然有变成 cosmetic 模板的风险。oscillation discipline 的真意是在**关键判断处主动切换**："当前解释如果完全错，最可能是因为对象错、指标错、窗口错、还是层级错？" 这比固定模板更重要。

**Anti-pattern**：把回答写得 cosmetic（"this is a deep question, will return to it"）。要求给出具体的 alternative object 候选或者明说不知道。

---

### Regula VI — 拟合-不变量分层

**"理论 claim 分层：拟合层 / 机制层 / 不变量层。每个 claim 必须明确标注在哪一层"**

具体形式——

- **拟合层**：解释当前数据。例如 "GPT-2 small mode-A 中 HTML-like input 进入一个 slower attractor"
- **机制层**：解释模块作用。例如 "在当前设置中，normalization 介导收缩，attention/MLP 更像关系展开或局部扩张组件"
- **不变量层**：跨系统保持。例如 "自迭代认知系统若缺少 memory / anchor / selection function，倾向坍缩到 architecture prior"

第三类才是真正理论候选。前两类不能自动升级。**拟合层 → 不变量层的升级必须经过 Regula II 验证**。

**Case study (Phase 1.3)**：master report 的 "(context, attention, normalization) asymmetric coupling" 被作为对 essay §4 的修订建议（语气是不变量层），但实际是基于 GPT-2 small mode-A 的拟合（实际是拟合层）。在 cross-architecture 验证之前不应升级。

**Anti-pattern**：所有 claim 都标 "fit-level" 来逃避升级压力。拟合层 claim 仍然必须经过 Regula I（最少假设）和 Regula IV（可证伪）。

---

### Regula VII — 关键统计量复核

**"评估者不能只接受 master report 给的核心数字。涉及核心 verdict 的关键统计量必须独立 recompute 至少一个 sanity check"**

具体形式——遇到核心结论依赖的关键数字（fitting window、λ、R²、effective rank、spectral radius、cosine similarity、KL divergence、probe accuracy 等），评估者至少要做一项 sanity check：

- 拟合窗口是否在 transient 而不是 noise plateau？
- 同样方法用于对照组是否给出荒谬结论？
- 低 R² 是现象本身，还是窗口错？
- 指标是否受 norm / scale / layer / tokenizer 影响？

**Case study (Phase 1.3 J.3)**：master report 用 [200, 800] 窗口拟合得到 HTML λ=1.0001, R²=0.011，结论是 "HTML 不是真吸引子"。Claude 在评估时独立 recompute，发现 lowercase 在同窗口也是 R²=0.000——按报告逻辑 lowercase 也"不是真吸引子"，显然荒谬。这暴露窗口选错。J.3 拟合错误的根本原因是 ChatGPT 第一轮评估时**没有 recompute 关键数字**，直接接受了报告叙事。

**为什么从 v0.1 的 D2 升格为 Regula**：J.3 事件证明，没有这条规则，所有其他 Regula 都建立在错的数字上。它不是辅助 discipline，是防止错误传播的核心机制。ChatGPT 在 v0.1 review 中的关键升格 proposal。

**Anti-pattern**：用"trust the report" 逃避复核。每次评估都必须显式说明做了哪个 sanity check。

---

## 2. Operational utilities（不构成 Regula）

以下三项是辅助操作机制，不是关于 verdict epistemics 的根本原则，所以不算 Regula。

### U1: Targeted search at decision points

**触发条件**——

1. 出现 specific 数学/概念结构时（找 prior framework 是否存在）
2. 提出 candidate 不变量时（找已有间接证据）
3. 在 framework 边界挣扎时（找方法论 prior work）

**执行规则**——

- Search 结果作为 candidate 进入 pool，**不直接进入 verdict**
- Search 后必须独立 process 再 share，避免 anchoring
- Search 不是常规步骤，是 exception——默认仍是基于已有 context 推理

**为什么是 utility 不是 Regula**：这件事 ChatGPT 在 v0.1 review 中指出——search 是扩展 candidate pool 的操作机制，和"如何形成 verdict"的 Regula 在不同 conceptual level。不应该和前 7 条并列。

**Case study (将在 Phase 1.4 第一次 trigger)**：search 之前不做、不探索性使用、不 broad search 找 inspiration。

---

### U2: 用户 perspective injection

**承认人类决策者不只是 vote 角色，而是 inject perspective 角色**。

U2 有两个互补的方面：

**方面一：责任分配（about who is responsible）**

用户可以提出"两个 AI 可能都没想到的层面"问题。这件事**不是用户的 mandatory contribution**——AI 评估者的远处工作责任仍然在 Regula V 下，不能把这件事 outsource 给用户。如果用户没有提出远处问题，AI 仍然必须自己执行 Regula V 的远处检查。

**方面二：处理优先级（about how AI handles user's perspective）**

当用户**确实**提出远处问题时，AI 评估者必须把它当作 **framework-level intervention** 处理，而不是当作普通 follow-up question。具体说——

- AI 不能只在原 framework 内回答用户的远处问题
- AI 必须重新审视当前 verdict，可能撤销或大幅修订
- AI 不能用"this is a deep question, will return to it"等 cosmetic 应答方式应付
- 如果用户的远处问题让 AI 无法在原 framework 内回答，AI 必须明说"原 framework 内答不了这个问题，需要重新选对象/坐标系"

**两个方面合起来一句话**：远处工作是 AI 的义务（方面一），用户远处问题是增强项但 AI 必须当 framework intervention 处理（方面二）。

**为什么是 utility 不是 Regula**：这件事 Claude 在 v0.1 → ChatGPT review → Claude push back 的迭代中明确——把"用户提供远处问题"作为强制流程，会让 AI 评估者把责任 outsource，让用户成为 single point of failure。正确的 framing 是"用户 perspective 是 valuable additional input"，不是"流程的强制 component"。但 ChatGPT 后续补充的"处理优先级"是 valid 的——保留为方面二。

**Case study (Phase 1.x 全程)**：J.3 拟合错误是 Claude 在 Regula VII 工作时发现的；"我们是不是在错坐标系"是用户提出的；ChatGPT 把这件事内化产出了 oscillation discipline 这个升级。**这是 valuable additional perspective 的 instance**——用户提出后 ChatGPT 没有当作普通 follow-up，而是把它当 framework intervention 处理（按 U2 方面二），结果产出了对 v0.1 的实质改进。

---

### U3: Framework 复杂度 monitoring

每个 phase 完成时，记录 framework 当前 explanatory component 数。如果连续 3 个 phase 这个数 monotonically 上升，trigger Regula I 的强化检查（强制构造 5-component 以下的 alternative framework）。

**为什么是 utility 不是 Regula**：这件事是 Regula I 的 supporting infrastructure，不是独立原则。

---

## 3. 共同执行的 disciplines

**D1: 工作流的"独立先于印证"**

延续 Phase 1.2 §2 工作流——每个 AI 评估者先独立看原始数据形成判断，**之后**才看其他评估者的回应。Phase 1.3 J.3 拟合错误事件中被验证有真实保护作用——Claude 因为先看数据没被 master report 的错误叙事 anchor。

**D2: 用户作为 perspective injector**

参见 U2。这件事是 valuable additional input，不是流程的强制 component。

**D3: Regulae 自身的 audit**

Phase 1.4 完成后做第一次 Regulae audit——某条 Regula 是否太严或太松，是否有未覆盖的失败模式。Regulae 自己是 working hypothesis（按 Regula IV 自己的标准），不是 timeless truth。

---

## 4. Regulae 之间的关系

7 条 Regulae 不是平行的——它们形成一个**判断流水线**：

```
data → Regula VII (统计量复核) → trustable data
trustable data → Regula I (最少假设) ↘
                 Regula III (scope)  → working hypothesis
working hypothesis
  → Regula IV (可证伪条款) → verdict candidate
  → Regula V (远处工作 oscillation) → 是否本轮化检查
  → Regula VI (分层) → 拟合层 / 机制层 / 不变量层标注
verdict candidate
  → Regula II (同效同因) → 升级为 hypothesis (如通过)

Utilities (并行):
  U1: targeted search 在触发条件下扩展 candidate pool
  U2: 用户 perspective injection（valuable but not load-bearing）
  U3: framework 复杂度 monitoring
```

**实际工作时不需要按这个顺序机械执行**。这个流水线是 retrospective 检查的——每个 phase 完成后，回看每个 verdict 是否经过了所有 7 条 Regula。

**关键依赖关系**：

- Regula VII 是 prerequisite——其他 6 条都依赖统计量可信
- Regula V 是 oscillation discipline，不是单步检查（在所有阶段都可能 trigger）
- Regula II 是升级关卡——只有通过它，拟合层 claim 才能升级到不变量层
- 根本姿态（§0.2 invariant-first）渗透所有 Regula，不是某条 Regula 单独承担

---

## 5. Regulae 自身的 epistemics

这套 Regulae 不是 timeless truth，是基于 Phase 1.x 实战的 best guess。它有以下 known limitations：

**它不会让 AI 评估者变成牛顿或爱因斯坦**——只让评估者更接近"有 epistemics discipline 的工作伙伴"。这是有真实价值的目标，但和"产出 breakthrough insight"是两回事。

**它可能在执行 1-2 phase 后被实战暴露不足**——某条 Regula 可能太严或太松，某个 anti-pattern 可能没料到。需要在 phase 完成时 audit Regulae 本身。

**它不能替代 sustained focus**——真正的 integration moment 仍然需要项目作者的长期沉浸。Regulae 的作用是让 sustained focus 不被 AI 评估者的拟合 mode 稀释。

**它有元层次的本轮化风险**——把 Regula 机械执行，每个评估都填 7 条 checklist，这本身就是 cosmetic compliance 而非真正的方法论。Regulae 的真实价值在于改变默认推理 mode，让"换坐标系"成为可能而不仅是 nice to have。**机械执行 Regula 是托勒密化 Regula 本身**。

**v0.2 相对 v0.1 的主要改进**：

1. 根本姿态（invariant-first, fit-second）从 §6 的隐含原则升格为 §0 的根本组织原则
2. Regula VII（关键统计量复核）从 D2 升格为正式 Regula
3. Regula VIII（targeted search）降级为 utility U1
4. Claude push back ChatGPT 的 Regula IX 提议——"用户作为 perspective injector"保留为 utility U2 而非 Regula（远处工作责任在 Regula V，不能 outsource）
5. ChatGPT 后续补充——U2 在"责任分配"基础上加了"处理优先级"维度（用户远处问题不能当普通 follow-up 处理）
6. 双焦距框架明确为 oscillation discipline 而非模板（ChatGPT 自己提议的反 cosmetic compliance）

---

## 6. 给评估者（Claude / ChatGPT）的接入说明

期望评估者在 Phase 评估中：

1. **每次评估必做 Regula VII**——独立 recompute 至少一个核心数字。这是最 load-bearing 的一条，没有它其他 Regula 全部建立在错数字上。
2. **每次评估必做 Regula V 的 oscillation**——在关键判断处切换 mode 做托勒密检查。如果答不出来明说"在 framework 内、需要外部 perspective"，不要 cosmetic 应付。
3. **遇到 framework 复杂度上升时主动 flag**（U3）——明确指出"这是托勒密化的早期信号"。
4. **接受 U2 的用户 perspective injection 作为 valuable input**——但不依赖它，自己仍要做远处工作（U2 方面一）。当用户**确实**提出远处问题时，必须当作 framework intervention 处理而非普通 follow-up（U2 方面二）。
5. **Search 仅在 U1 触发条件下做**——不为了"全面"在每次评估前做 broad search。

---

## 7. 这份文档的产生方式（meta-record）

v0.1 的产生路径：
- Phase 1.1-1.3 实际暴露的失败模式
- Claude 在 Phase 1.3 评估时识别出 J.3 拟合错误
- 项目作者提出"AI 倾向于拟合可能造成本轮化"
- ChatGPT 提出"双焦距评价"和"invariant-first"
- 项目作者提出"web search 引入前沿 candidate idea"
- Claude 综合生成 v0.1

v0.2 的额外路径：
- ChatGPT review v0.1 后提出 5 项实质改进（oscillation 反模板、统计量复核升格、targeted search 重定位、Regula IX 提议、双焦距改造）
- Claude push back Regula IX（远处工作责任不能 outsource 给用户）
- ChatGPT 同意 push back
- Claude 综合生成 v0.2

这个生成路径本身是 FPP Regulae 想推广的工作模式的 instance——**人类作为 perspective injector + 多 AI 评估者作为 disciplined collaborators + 长期 sustained focus + 关键 moment 的 codification**。

---

## 8. 立即落地

**Phase 1.3 J.3 corrigendum**（最近的工作）：
- 按 Regula VII，重新做 J.3 拟合（在正确 transient window 上）
- 按 Regula III，明确 corrigendum 的 evidence scope
- 按 Regula IV，给出 corrigendum 的可证伪条款
- 按 Regula V，corrigendum 后必答 oscillation 检查
- 按 Regula I，给 corrigendum 的 framework 提供 minimum alternative

**Phase 1.4 design**（J.3 修正后）：
- 按 Regula II，必须包含至少一个 cross-architecture 实验
- 按 U1，在 design 前 trigger 一次 targeted search（DEQ + mechanistic interp + dynamical systems 在 LLM 上的应用）
- 按 Regula VI，明确每个实验的 claim 是哪一层
- 按根本姿态 §0.2，Phase 1.4 核心实验应该是 invariant probe（同构样本测试）而非继续在 GPT-2 small mode-A 上加 phase

---

## 9. 文档版本历史

- **v0.1**（2026-04-27 上午）：初始版本。基于 Phase 1.1-1.3 失败模式 + 项目作者方法论 push back + ChatGPT 双焦距评价框架。覆盖 7 条 Regula + 4 条 disciplines。
- **v0.2**（2026-04-27 下午）：基于 ChatGPT 对 v0.1 的 review 加 Claude 对其 Regula IX 提议的 push back 加 ChatGPT 后续接受 push back 时附加的 U2 处理优先级补充。Regula VII（统计量复核）从 discipline 升格；targeted search 降为 utility；根本姿态 §0.2 显式化；oscillation discipline 反模板；Regula IX 提议被 push back 后保留为 U2 utility（含两个方面：责任分配 + 处理优先级）。

预期在 Phase 1.4 完成后做第一次 Regulae audit（按 D3）。除非三方任何一方觉得有重大调整必要，方法论讨论暂停在此版本。
