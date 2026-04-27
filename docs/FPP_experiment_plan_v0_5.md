# Phase 1.3 J.3 Corrigendum — 任务文档

**项目代号**：FixedPointProbe (FPP)
**任务类型**：错误修正（corrigendum）
**版本**：v0.1
**日期**：2026-04-27
**前置文档**：
- `phase1_3_master_report.md`
- `phase1_3_basin_diagnosis.md`
- `FPP_Regulae_v0_2.md`（本任务按此 Regulae 执行）

---

## 0. 任务定位

Phase 1.3 master report 给出的核心 verdict 之一——**"HTML structure is not a real fixed point" / "wpe-supported marginal equilibrium"**——基于 J.3 的拟合结果（HTML λ ≈ 1.0001, R² = 0.011）。

这个 verdict 在外部评估（按 FPP Regulae §6 工作流）中被发现**基于错误的拟合窗口**。具体说：

- J.3 用 window [200, 800] 拟合 HTML 的 tail rate
- 但 HTML 在 step ~200 之后已经达到浮点噪声 plateau（delta 稳定在 ~2e-7，std ~3e-8）
- 同样窗口 [200, 800] 拟合 lowercase 也得到 R² = 0.000——按报告逻辑 lowercase 也"不是真吸引子"，显然荒谬
- HTML 真正的指数衰减阶段在 step ~110-180，那段窗口拟合得到 λ ≈ 0.87-0.89, R² > 0.99

**所以 Phase 1.3 master report 关于 HTML 本体地位的 verdict 需要被显式 revoke 并 corrigendum**。

这个任务是按 FPP Regulae v0.2 §8 立即落地清单的第一项。

---

## 1. 任务范围

### 1.1 必做事项

**Step 1：J.3 重新拟合**

对 8 个长 trace（4 HTML + 4 lowercase）和 30 个 markup triangulation traces（特别是落入第三 endpoint 的 5 个 square brackets + 5 个 isolated chars）做 tail rate 重新拟合。每个 trace 使用**自动检测的 transient window** 而非固定窗口。

具体算法：

```
对每个 trace 的 deltas[]:
  1. 估计噪声 floor:
     ε_floor = median(deltas[max(0, n-1000):n])
     # 假设最后 1000 步是 plateau

  2. 找到 transient 上界 step_high:
     deltas[step_high] > 100 * ε_floor 的最大 step
     # 必须显著高于 noise floor 才算 transient

  3. 找到 transient 下界 step_low:
     最早的 step 使得后续是单调下降段
     # 跳过初始 chaotic 阶段

  4. 在 [step_low, step_high] 上拟合:
     log delta_t = a + t * log λ
     报告 λ, R², 拟合 step 数

  5. 输出诊断信息:
     - noise floor 数值
     - transient window 起止
     - 是否存在 oscillation phase（initial chaotic 阶段长度）
     - 如果 R² < 0.95，标注"非干净 exponential，需进一步分析"
```

**Step 2：对照实验——故意用错窗口验证**

为了证明 J.3 的窗口选择问题真实存在，做一次对照——

```
对 lowercase trace:
  - 用 [200, 800] 窗口拟合 (master report 用的窗口策略)
  - 用 [5, 50] 窗口拟合 (正确 transient 窗口)
  - 报告两个 λ 和 R² 的对比

对 HTML trace:
  - 用 [200, 800] 窗口拟合 (master report 用的窗口)
  - 用 [110, 180] 窗口拟合 (正确 transient 窗口)
  - 报告两个 λ 和 R² 的对比
```

这一步是为了让 corrigendum 的论证不只是"用了对的窗口得到了不同结论"，而是**证明窗口选择本身是关键变量**——正确窗口下两个 basin 都是真指数收敛，错误窗口下两个 basin 都看起来"不收敛"。

**Step 3：Verdict 修订**

基于 Step 1-2 的数据，做以下显式修订：

- 撤销："HTML is wpe-supported marginal equilibrium, NOT a real fixed point"
- 替换为：[基于实际数据的新 verdict，预计是 "HTML is a weaker but genuine exponential attractor with longer transient phase"]
- 撤销：Phase 1.2 "two attractors" framing was wrong
- 替换为：Phase 1.2 "two attractors at different convergence rates" framing was correct, but characterized the rate difference as 13× wall-clock steps rather than as differential local contraction strength

**Step 4：可证伪条款（按 Regula IV）**

新 verdict 必须配可证伪条款：

```
Verdict: HTML is a weaker but genuine exponential attractor.
Falsifiable by:
  - HTML transient λ in correctly-detected window has R² < 0.95
  - HTML λ depends sensitively on choice of transient window
    (varies by > 0.1 across reasonable window choices)
  - HTML in cancel-pos / posfree mode loses exponential pull
    (this is a separate question from "wpe-supported"—the
     attractor may still be exponential in cancel-pos mode
     but at different λ)
```

**Step 5：Evidence scope 标注（按 Regula III）**

新 verdict 的 evidence scope 必须明确：

```
Scope: GPT-2 small (124M params), mode-A hidden state self-iteration,
       8 long traces (4 HTML + 4 lowercase) + 30 markup triangulation traces,
       seq_len 64, fp32, position embedding active
Generalization restrictions:
  - Not yet validated on GPT-2 medium / Pythia (Regula II hold)
  - Not yet validated on cross-encoding isomorphic samples (Regula II hold)
  - The "asymmetric coupling" interpretation depends on Phase 1.3 M data
    which uses spectral norm not spectral radius for some sub-modules
    (separate caveat from this corrigendum)
```

### 1.2 必不做事项

**不要做的事**：

- **不要扩展 corrigendum 范围**——这次只修 J.3 的 fitting 错误，不顺带修订其他 verdict
- **不要做新实验** ——所有数据来自已存 traces，corrigendum 是 reanalysis 不是 new experiment
- **不要修订 essay**（按 Phase 1.3 plan §10 essay 修订冻结）
- **不要把 corrigendum 写成"Phase 1.3 整体修订"** ——它范围严格限于 J.3 + Q4 verdict + 直接受 Q4 影响的下游 framing

### 1.3 不在 corrigendum 范围但需要后续处理的（仅记录，不实施）

- **Third endpoint 的 transient λ 没测过** ——M 实验给了它 ρ=0.91 但没测 actual tail rate。Phase 1.4 候选实验。
- **Cross-architecture 验证** ——按 Regula II 是 Phase 1.4 必做项。这次 corrigendum 不涉及。
- **Spectral norm vs spectral radius 在 sub-module 层面的解读** ——Phase 1.2 F 的"asymmetric coupling"解读需要被 Phase 1.3 M 的数据修正。这是另一个 separate issue，不在 corrigendum 范围。

把这三件事**显式记录在 corrigendum 报告里**作为"已知未处理 issue"，但不在 corrigendum 内实施。

---

## 2. 按 FPP Regulae v0.2 的执行约束

### 2.1 必做的 Regulae 检查

**Regula I（最少假设）**：corrigendum 给出的新机制图像必须有 minimum-component alternative。预计修订后图像是 "two genuine exponential attractors with different convergence rates + LayerNorm contraction" (4-component)，比原 master report 的 7+component 更经济。

**Regula III（evidence scope）**：每个修订 verdict 都标注 scope（见 Step 5）。

**Regula IV（可证伪条款）**：每个修订 verdict 都配 falsification 条件（见 Step 4）。

**Regula V（远处工作 oscillation）**：corrigendum 报告末尾必答两个问题——

1. 托勒密检查：J.3 拟合错误暴露后，我们是否仍在错误对象上工作？是否应该考虑"hidden state 自迭代"本身就不是合适的 self-context dynamics 投影面？
2. 不变量假设：HTML "weaker exponential attractor" 这个修订 verdict，如果在 Pythia / LLaMA 上 replicate，最可能哪部分被推翻？

**Regula VI（拟合-不变量分层）**：corrigendum 的所有 verdict 必须明确标注是拟合层还是机制层。预计全部是拟合层（基于 GPT-2 small mode-A 的现象描述）。

**Regula VII（关键统计量复核）**：corrigendum 本身就是 Regula VII 的应用——独立 recompute J.3 拟合后发现 master report 的窗口错误。corrigendum 完成后，**评估者再次复核 corrigendum 的核心数字**（按同样的 Regula VII），形成双重验证。

### 2.2 不做的 Regulae 工作

- 不做 Regula II 工作（cross-architecture）——这是 Phase 1.4 的事
- 不 trigger U1（targeted search）——corrigendum 范围不需要外部 candidate
- 不需要 U2（用户 perspective injection）——corrigendum 是技术修正，不需要新 perspective

---

## 3. 输出要求

### 3.1 文档输出

`reports/phase1_3_J3_corrigendum.md` —— 包含：

1. **Executive summary**：J.3 拟合错误描述 + 修订前后 verdict 对比（一页内）
2. **Step 1-5 完整数据**：所有重新拟合的 λ / R² / window 详情
3. **对照实验数据（Step 2）**：错误窗口 vs 正确窗口的并列展示
4. **修订后的机制图像**：基于 corrigendum 的最新 framework
5. **Regulae 合规检查**：按 §2.1 列出的每条 Regula 显式回答
6. **已知未处理 issue**（§1.3 三项）：明确记录为 Phase 1.4 候选

### 3.2 数据输出

```
data/phase1_3_J3_corrigendum/
  ├── refits/
  │   ├── lowercase_traces_refit.json  # 4 个 lowercase 重拟合
  │   ├── html_traces_refit.json        # 4 个 HTML 重拟合
  │   └── third_endpoint_refit.json     # markup triangulation 中第三 endpoint
  ├── window_comparison.json            # Step 2 对照实验
  └── revised_verdict.json              # Step 3 修订 verdict
```

### 3.3 不需要的 outputs

- 不需要新的 figures（除非现有的有错误需要重做）
- 不需要新的 trace data（全部用 Phase 1.3 已存数据）

---

## 4. 验收门槛

corrigendum 完成的判据：

1. ✅ J.3 重新拟合在 8 个长 trace + 30 个 markup triangulation traces 上完成
2. ✅ Step 2 对照实验明确展示窗口选择是关键变量
3. ✅ Verdict 修订显式（不是悄悄替换，要明确标注 "previously stated X, now revoked, replaced by Y"）
4. ✅ 修订 verdict 配可证伪条款
5. ✅ 所有 §2.1 的 Regulae 检查被显式回答
6. ✅ §1.3 已知未处理 issue 被明确记录
7. ✅ corrigendum 报告读起来诚实而非辩护——明说"原 verdict 错了，原因是窗口选择"，不要写成"原 verdict 部分正确，需要 nuance"

---

## 5. 时间预算

约 1-2 天 wall-clock。GPU 时间 ≈ 0（全部 reanalysis）。

如果在自动 transient window 检测算法上卡住超过半天，向项目作者请示——可以接受 fallback 为手动指定多个候选窗口然后报告 sensitivity。

---

## 6. 给 Claude Code 的额外提醒

这次任务是修正你之前一份报告里的错误。这种任务有特殊心理动态——容易出现"辩护性叙事"的诱惑（"原 verdict 不算完全错，只是需要 nuance"）。**抑制这个诱惑**。诚实修正比辩护性叙事更有长期价值。

报告写作上，参考 ChatGPT 在 v0.1 review 中提的"承认错误本身比修正错误更重要"的精神——

- 用清晰的 "previously stated X, now revoked, replaced by Y" 句式，不要用"refining the previous understanding"这种淡化语气
- 在 executive summary 第一段就明确说原 verdict 错在哪里（窗口选择），不要把这件事埋在 detail 里
- §6 末尾承认这次错误暴露的工作流问题——Regula VII（统计量复核）的升格直接来自这次事件

这次 corrigendum 本身将作为 FPP Regulae v0.2 的 case study 在未来被引用。**写得诚实让 Regulae 文档更扎实，写得辩护性会让 Regulae 文档失去 grounding**。

---

## 7. 文档版本

- **v0.1**（2026-04-27）：初始版本。基于 Phase 1.3 master report J.3 错误 + Claude / ChatGPT 独立评估 + 项目作者最终决策。
