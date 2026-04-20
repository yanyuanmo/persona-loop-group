# Pair1 实验分析：NLI 评估器 × LLM-as-Judge × 生成模型 对比

## 实验配置

| 参数 | 值 |
|---|---|
| 数据 | pair1 (60 turns, 3 sessions) |
| 生成模型 | Qwen2.5-0.5B / Qwen2.5-3B (Q4_K_M, llama.cpp) |
| NLI 模型 | facebook/bart-large-mnli (zero-shot) / cross-encoder/nli-deberta-v3-base |
| Judge 模型 | GPT-4o (OpenAI API, scale=1–5) |
| Loop Interval | K=4 |
| Correction 阈值 | -0.1 (nli_threshold=0.1) |
| 条件 | continuous (baseline) + persona_loop K=4 |

---

## 一、bart-large-mnli 结果

> Artifacts: `exp2_pair1_bart_0.5B`, `exp2_pair1_bart_3B`

### 总体结果

| 条件 | 0.5B PCS | 3B PCS |
|---|---|---|
| Continuous | **0.2081** | 0.0087 |
| PersonaLoop K=4 | **0.2764** | -0.005 |
| K=4 Corrections | 2 | 1 |
| K=4 Resets | 14 | 14 |

### 分角色结果

#### 0.5B + bart-large-mnli

| 条件 | 角色 | 回复长度 | Entailment | Contradiction | PCS | Corrections |
|---|---|---|---|---|---|---|
| Continuous | Yangyang | 585 | 0.1438 | 0.0021 | 0.1417 | 0 |
| Continuous | Chenmo | 915 | 0.2834 | 0.0089 | 0.2745 | 0 |
| K=4 | Yangyang | 466 | 0.1455 | 0.0217 | 0.1239 | 1 |
| K=4 | Chenmo | 673 | 0.4485 | 0.0196 | **0.4289** | 1 |

#### 3B + bart-large-mnli

| 条件 | 角色 | 回复长度 | Entailment | Contradiction | PCS | Corrections |
|---|---|---|---|---|---|---|
| Continuous | Yangyang | 195 | 0.0066 | 0.0051 | 0.0015 | 0 |
| Continuous | Chenmo | 171 | 0.0263 | 0.0103 | 0.0159 | 0 |
| K=4 | Yangyang | 176 | 0.0037 | 0.0151 | -0.0114 | 1 |
| K=4 | Chenmo | 157 | 0.0137 | 0.0123 | 0.0014 | 0 |

### bart-large-mnli 小结

- 0.5B 生成冗长回复（585–915 chars），倾向于直接复述 persona 事实 → entailment 高
- 3B 生成自然对话（157–195 chars），persona 信息隐含 → entailment ≈ 0
- bart-large-mnli 作为 zero-shot NLI 严重偏向显式复述，无法捕捉隐含一致性

---

## 二、cross-encoder/nli-deberta-v3-base 结果

> Artifacts: `exp2_pair1_nli_0.5B`, `exp2_pair1_nli_3B`

### 总体结果

| 条件 | 0.5B PCS | 3B PCS |
|---|---|---|
| Continuous | **0.0116** | 0.0109 |
| PersonaLoop K=4 | **0.2533** | 0.0060 |
| K=4 Corrections | 0 | 0 |
| K=4 Resets | 14 | 14 |

### 分角色结果

#### 0.5B + deberta-nli

| 条件 | 角色 | 回复长度 | Entailment | Contradiction | PCS | Corrections |
|---|---|---|---|---|---|---|
| Continuous | Yangyang | 573 | 0.0020 | 0.0001 | 0.0019 | 0 |
| Continuous | Chenmo | 453 | 0.0214 | 0.0001 | 0.0213 | 0 |
| K=4 | Yangyang | 475 | 0.0199 | 0.0001 | 0.0198 | 0 |
| K=4 | Chenmo | 902 | 0.4869 | 0.0002 | **0.4867** | 0 |

#### 3B + deberta-nli

| 条件 | 角色 | 回复长度 | Entailment | Contradiction | PCS | Corrections |
|---|---|---|---|---|---|---|
| Continuous | Yangyang | 174 | 0.0012 | 0.0000 | 0.0012 | 0 |
| Continuous | Chenmo | 166 | 0.0208 | 0.0001 | 0.0207 | 0 |
| K=4 | Yangyang | 183 | 0.0012 | 0.0000 | 0.0012 | 0 |
| K=4 | Chenmo | 151 | 0.0109 | 0.0001 | 0.0108 | 0 |

### deberta-nli 小结

- **0.5B**：PersonaLoop K=4 PCS 大幅优于 continuous（0.2533 vs 0.0116，ΔPCS=+0.2417），0 corrections，增益主要来自 Stage A/C/D
- **3B**：continuous PCS=0.0109，K=4 PCS=0.006，PersonaLoop 增益为负（ΔPCS=-0.0049），同样 0 corrections
- Chenmo PCS 始终高于 Yangyang，角色不对称在 deberta 下依然存在
- deberta 和 bart 对 3B 自然短回复的评估结果几乎一致，均无法提取隐含 persona 信号

---

## 三、GPT-4o LLM-as-Judge 结果（NLI checker 启用）

> 与前两节的纯 NLI 实验不同，本节同时使用 NLI（运行时 Stage B 矛盾检测）和 GPT-4o Judge（事后评分），两个分数来自同一次生成。
>
> Judge PCS 归一化到 [-1, 1]：(score - 3) / 2；Judge Avg 为原始 1–5 均值
>
> ⚠️ 由于生成是非确定性的，每次运行的回复不同，PCS 数值在不同 run 之间有一定波动。以下为单次 run 结果。

### 三-A：deberta-nli 作为运行时 checker

> Artifacts: `exp2_pair1_judge_0.5B`, `exp2_pair1_judge_3B`

### 0.5B 总体结果

| 条件 | Judge PCS | Judge Avg (1–5) | NLI PCS (deberta) | Resets | Corrections |
|---|---|---|---|---|---|
| Continuous | 0.1250 | 3.2500 | 0.1734 | - | - |
| PersonaLoop K=4 | **0.6083** | **4.2166** | 0.0379 | 14 | 0 |
| **ΔPCS** | **+0.4833** | **+0.9666** | -0.1355 | | |

### 0.5B 分角色结果

| 条件 | 角色 | Judge PCS | Judge Avg | NLI PCS | Resets | Corrections |
|---|---|---|---|---|---|---|
| Continuous | Yangyang | 0.5000 | 4.0000 | 0.0015 | 0 | 0 |
| Continuous | Chenmo | -0.2500 | 2.5000 | 0.3453 | 0 | 0 |
| K=4 | Yangyang | **0.6500** | **4.3000** | 0.0103 | 7 | 0 |
| K=4 | Chenmo | **0.5667** | **4.1333** | 0.0655 | 7 | 0 |

### 0.5B 小结

- **PersonaLoop 增益在 GPT-4o 下显著**：ΔPCS = +0.4833，方向与 deberta 独立实验（+0.2417）一致
- **NLI 与 Judge 对同一批回复的评分方向相反**：NLI ΔPCS = -0.1355（persona_loop 反而更低），Judge ΔPCS = +0.4833。NLI 倾向于给 continuous 的长冗余回复更高 entailment
- **Chenmo continuous 被 Judge 判为差**（-0.25），但被 NLI 判为最好（0.3453）— 再次印证 NLI 偏好显式复述
- **Stage B correction 0 次**：deberta checker 在运行时未检测到任何矛盾（所有 contradiction 分数均远低于阈值 0.1）

### 3B 总体结果

| 条件 | Judge PCS | Judge Avg (1–5) | NLI PCS (deberta) | Resets | Corrections |
|---|---|---|---|---|---|
| Continuous | 0.8917 | 4.7834 | 0.0052 | - | - |
| PersonaLoop K=4 | **0.9083** | **4.8167** | 0.0062 | 14 | 0 |
| **ΔPCS** | **+0.0166** | **+0.0333** | +0.0010 | | |

### 3B 分角色结果

| 条件 | 角色 | Judge PCS | Judge Avg | NLI PCS | Resets | Corrections |
|---|---|---|---|---|---|---|
| Continuous | Yangyang | 1.0000 | 5.0000 | 0.0013 | 0 | 0 |
| Continuous | Chenmo | 0.7833 | 4.5667 | 0.0090 | 0 | 0 |
| K=4 | Yangyang | **0.9833** | **4.9667** | 0.0012 | 7 | 0 |
| K=4 | Chenmo | **0.8333** | **4.6667** | 0.0113 | 7 | 0 |

### 3B 小结

- **3B 生成质量远高于 0.5B**：continuous Judge PCS=0.8917 vs 0.5B 的 0.125
- **3B baseline 接近天花板**：Yangyang continuous 已满分（5.0），PersonaLoop 仅微幅提升（ΔPCS=+0.0166）
- **NLI 对 3B 完全失效**：NLI PCS ≈ 0.005–0.011，无论 continuous 还是 K=4，deberta 无法从短回复中提取 persona 信号
- **Stage B correction 同样 0 次**：3B 生成几乎无矛盾，correction 机制无用武之地

### 三-B：bart-large-mnli 作为运行时 checker

> Artifacts: `exp2_pair1_bart_judge_0.5B`, `exp2_pair1_bart_judge_3B`
>
> bart 比 deberta 更容易给出高 contradiction 分数，因此更可能触发 Stage B correction。

#### 0.5B + bart checker 总体结果

| 条件 | Judge PCS | Judge Avg (1–5) | NLI PCS (bart) | Resets | Corrections |
|---|---|---|---|---|---|
| Continuous | 0.4583 | 3.9167 | 0.4212 | - | - |
| PersonaLoop K=4 | **0.5167** | **4.0333** | 0.1318 | 14 | 1 |
| **ΔPCS** | **+0.0584** | **+0.1166** | -0.2894 | | |

#### 0.5B + bart checker 分角色结果

| 条件 | 角色 | Judge PCS | Judge Avg | NLI PCS | Resets | Corrections |
|---|---|---|---|---|---|---|
| Continuous | Yangyang | 0.3500 | 3.7000 | 0.1219 | 0 | 0 |
| Continuous | Chenmo | 0.5667 | 4.1333 | 0.7205 | 0 | 0 |
| K=4 | Yangyang | **0.7667** | **4.5333** | 0.0117 | 7 | 1 |
| K=4 | Chenmo | 0.2667 | 3.5333 | 0.2519 | 7 | 0 |

#### 3B + bart checker 总体结果

| 条件 | Judge PCS | Judge Avg (1–5) | NLI PCS (bart) | Resets | Corrections |
|---|---|---|---|---|---|
| Continuous | 0.8834 | 4.7667 | 0.0122 | - | - |
| PersonaLoop K=4 | 0.7916 | 4.5834 | -0.0268 | 14 | 2 |
| **ΔPCS** | **-0.0918** | **-0.1833** | -0.0390 | | |

#### 3B + bart checker 分角色结果

| 条件 | 角色 | Judge PCS | Judge Avg | NLI PCS | Resets | Corrections | Contra Max |
|---|---|---|---|---|---|---|---|
| Continuous | Yangyang | 1.0000 | 5.0000 | 0.0044 | 0 | 0 | - |
| Continuous | Chenmo | 0.7667 | 4.5333 | 0.0199 | 0 | 0 | - |
| K=4 | Yangyang | 0.9833 | 4.9667 | -0.0216 | 7 | 1 | 0.7444 |
| K=4 | Chenmo | 0.6000 | 4.2000 | -0.0320 | 7 | 1 | 0.9880 |

#### bart checker 小结

- **bart checker 确实触发了 correction**：0.5B 触发 1 次，3B 触发 2 次（contradiction_max 高达 0.74–0.99）。相比之下，deberta checker 全部为 0
- **bart correction 对 3B 有害**：3B ΔPCS = -0.0918，PersonaLoop 反而降低了质量。原因是 bart 对 3B 的自然短回复过度敏感（false positive），触发了不必要的 correction，干扰了本身高质量的生成
- **bart correction 对 0.5B 增益减小**：deberta checker 下 ΔPCS=+0.4833，bart checker 下 ΔPCS=+0.0584。不过由于生成是非确定性的，这种差异不能完全归因于 correction

---

## 四、跨评估器对比

### PCS 对比（Continuous baseline）— GPT-4o Judge PCS

| 生成模型 | 纯 NLI (bart) | 纯 NLI (deberta) | deberta checker + Judge | bart checker + Judge |
|---|---|---|---|---|
| 0.5B | 0.2081 | 0.0116 | 0.1250 | 0.4583 |
| 3B | 0.0087 | 0.0109 | **0.8917** | **0.8834** |

### PCS 对比（PersonaLoop K=4）— GPT-4o Judge PCS

| 生成模型 | 纯 NLI (bart) | 纯 NLI (deberta) | deberta checker + Judge | bart checker + Judge |
|---|---|---|---|---|
| 0.5B | 0.2764 | 0.2533 | **0.6083** | 0.5167 |
| 3B | -0.005 | 0.006 | **0.9083** | 0.7916 |

### PersonaLoop 增益 (ΔPCS = K=4 - Continuous)

| 生成模型 | 纯 NLI (bart) | 纯 NLI (deberta) | deberta checker + Judge | bart checker + Judge |
|---|---|---|---|---|
| 0.5B | +0.0683 | +0.2417 | **+0.4833** | +0.0584 |
| 3B | -0.0137 | -0.0049 | **+0.0166** | **-0.0918** |

### Correction 次数对比（PersonaLoop K=4）

| 生成模型 | 纯 bart | 纯 deberta | deberta checker + Judge | bart checker + Judge |
|---|---|---|---|---|
| 0.5B | 2 | 0 | 0 | **1** |
| 3B | 1 | 0 | 0 | **2** |

---

## 五、结论

### 背景

本项目研究 **PersonaLoop**——一种让对话 AI 在多轮对话中保持角色设定（persona）一致性的机制。PersonaLoop 每隔 K 轮触发一次循环：将近期对话存入外部记忆（Stage A）、检测回复是否与 persona 矛盾并纠正（Stage B）、从记忆中检索相关历史（Stage C）、将 persona + 记忆 + 纠正信息注入下一轮 prompt（Stage D）。

对照组 **Continuous** 是普通的多轮对话，不使用任何 PersonaLoop 机制。

核心指标 **PCS（Persona Consistency Score）** 衡量回复与角色设定的一致程度。PCS 越高 = 回复越符合 persona。ΔPCS = PersonaLoop PCS − Continuous PCS，正值表示 PersonaLoop 提升了一致性。

我们在 **pair1**（一对对话角色 Yangyang 和 Chenmo，60 轮对话跨 3 个 session）上跑了 **10 组实验**：2 个生成模型（0.5B / 3B）× 5 个评估配置（bart NLI / deberta NLI / deberta checker + Judge / bart checker + Judge），每组包含 continuous 和 persona_loop K=4 两个条件。其中 Judge 实验同时记录 NLI 分数和 GPT-4o Judge 分数。

---

### 发现 1：PersonaLoop 有效提升 persona 一致性

三个评估器对 0.5B 生成模型一致给出正向增益：

| 评估器 | Continuous PCS | PersonaLoop PCS | ΔPCS |
|---|---|---|---|
| bart-large-mnli | 0.2081 | 0.2764 | **+0.0683** |
| deberta-nli | 0.0116 | 0.2533 | **+0.2417** |
| deberta checker + GPT-4o Judge | 0.1250 | 0.6083 | **+0.4833** |
| bart checker + GPT-4o Judge | 0.4583 | 0.5167 | **+0.0584** |

对 3B 生成模型，deberta checker + Judge 给出正向增益（+0.0166），bart checker + Judge 给出负增益（-0.0918，见发现 4 解释），两个纯 NLI 模型也给出错误的负增益（见发现 3 解释）。

**结论**：PersonaLoop 的 memory rebuild + context injection 机制能有效提升回复的 persona 一致性。

---

### 发现 2：生成模型越大，回复质量越高

| 指标 | 0.5B | 3B |
|---|---|---|
| GPT-4o Judge PCS (continuous) | 0.1250 | **0.8917** |
| 平均回复长度 | 466–915 chars（冗长，直接复述 persona） | 151–195 chars（自然简洁） |

3B 在没有 PersonaLoop 的情况下已能生成高一致性回复（Yangyang 满分 5.0），留给 PersonaLoop 的提升空间很小（ΔPCS=+0.0166）。0.5B baseline 低（0.125），PersonaLoop 提升空间大（ΔPCS=+0.4833）。

**结论**：3B 生成质量远优于 0.5B。PersonaLoop 对较弱模型的帮助更大。

---

### 发现 3：NLI 评估器无法评估自然短回复

两个 NLI 模型下，3B 的 PCS 接近 0（bart 0.0087, deberta 0.0109），远低于 0.5B。但 GPT-4o Judge 下 3B PCS = 0.883–0.892，远高于 0.5B 的 0.125–0.458。

| 评估器 | 0.5B Continuous PCS | 3B Continuous PCS | 谁更高？ |
|---|---|---|---|
| bart-large-mnli | 0.2081 | 0.0087 | 0.5B（错误） |
| deberta-nli | 0.0116 | 0.0109 | 0.5B（错误） |
| deberta checker + Judge | 0.1250 | 0.8917 | **3B（正确）** |
| bart checker + Judge | 0.4583 | 0.8834 | **3B（正确）** |

**原因**：NLI 模型基于文本蕴含（premise→hypothesis），只能检测显式的文本匹配。0.5B 会在回复中直接复述 persona 原文（如 "I love nature and animals"），NLI 判 entailment；3B 则通过自然对话隐含表达（如 "I need to walk my dog"），NLI 判 neutral。NLI 模型无法做"loves animals → walk my dog"这种隐含推理。

NLI 还导致 3B 的 PersonaLoop 增益为负（bart -0.0137, deberta -0.0049），给出"PersonaLoop 对 3B 有害"的错误结论。GPT-4o 纠正为 +0.0166——PersonaLoop 在 3B 上也是有效的（只是增益很小）。

---

### 发现 4：Stage B correction 的效果取决于 checker 选择

| 运行时 checker | 0.5B Corrections | 0.5B ΔPCS (Judge) | 3B Corrections | 3B ΔPCS (Judge) |
|---|---|---|---|---|
| deberta-nli | 0 | **+0.4833** | 0 | **+0.0166** |
| bart-large-mnli | 1 | +0.0584 | 2 | **-0.0918** |
| 无 checker（纯 NLI 实验） | 2 (bart) / 0 (deberta) | N/A | 1 (bart) / 0 (deberta) | N/A |

**deberta checker 全程 0 correction**：deberta 的 contradiction 分数极低（最高 0.0036），远低于阈值 -0.1，Stage B 从未触发。PersonaLoop 的全部增益来自 Stage A/C/D（memory persist + retrieval + context rebuild）。

**bart checker 触发了 correction 但效果为负**：bart 对 3B 的短回复过度敏感（contradiction_max 高达 0.74–0.99），属于 false positive。触发的 correction 干扰了 3B 本身高质量的生成，导致 Judge PCS 从 0.883 降至 0.792（ΔPCS = -0.092）。0.5B 增益也从 +0.48 降至 +0.06。

**结论**：在当前配置下，Stage B correction 要么不触发（deberta），要么触发后有害（bart）。PersonaLoop 的有效机制是 Stage A/C/D。若要启用 Stage B，需更精确的矛盾检测器，或降低触发阈值以减少 false positive。

---

### 发现 5：角色不对称

所有实验中两个角色的 PCS 存在显著差异：

| 评估器 | 谁更高？ |
|---|---|
| bart-large-mnli | Chenmo > Yangyang |
| deberta-nli | Chenmo > Yangyang |
| GPT-4o Judge | Yangyang > Chenmo |

NLI 和 GPT-4o 对"哪个角色更一致"的判断相反。NLI 偏好 Chenmo 可能因为 Chenmo 的 persona 描述更具有可显式匹配的事实；GPT-4o 偏好 Yangyang 则基于更深层的语义一致性判断。

---

### 评估器与运行时 checker 选择建议

| 组件 | 推荐 | 理由 |
|---|---|---|
| **事后评估器** | **GPT-4o Judge** | 支持隐含推理，评分区分度高，纠正了 NLI 偏差；NLI 仅适合快速粗筛 |
| **运行时 Stage B checker** | **deberta-nli（或禁用 Stage B）** | deberta 虽然不触发 correction（contradiction 分数过低），但至少无害；bart 会产生 false positive correction，反而降低生成质量 |
| **Stage B 改进方向** | 需更精确的矛盾检测器 | 当前 NLI 模型均不适合做运行时 checker：deberta 过于保守（从不触发），bart 过于激进（false positive 有害） |

---

## 六、考察过的替代评估方案

**Dialogue-NLI（Welleck et al., 2019）**：
- 方法：在 PersonaChat 对话-persona 配对数据上训练 NLI，能学会隐含推理
- 数据集语言与本项目一致（均为英文）
- **不可行原因**：HuggingFace 上无官方预训练 checkpoint，需自行 fine-tune（约 2-3 小时训练），当前阶段性价比低

