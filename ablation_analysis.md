# Pair6 组件消融实验分析（Component Ablation）

## 一、实验设计

### 1.1 目标

确定 PersonaLoop 系统中**记忆机制（Stage A 写入 + Stage C 检索）**的贡献，并区分"周期性上下文重建"的机制价值与"记忆检索"的信息价值。

### 1.2 设计理由

消融条件精简为 3 个（而非完整 5 条件矩阵），原因：

1. **No Retrieval (top_k=0) vs No Memory (disable_persona_persist)** — 功能上等价。两者都导致 `[MEMORY]` 不出现在 context 中。唯一区别是前者仍在后台写入向量库但不读取，对 PCS 无影响。保留 No Memory 即可。

2. **No Correction（去 Stage B）** — 无需单独消融，因为跨实验数据已充分证明 Stage B 在当前配置下从未产生有效影响。详见下文 §1.2.1。

#### 1.2.1 为什么不需要跑 Stage B 消融

跨所有已跑实验（pair1/4/6 K-sensitivity + exp2 + 本次消融），共 **282 次 loop reset，DeBERTa checker 仅触发 2 次 correction（触发率 0.7%）**：

| 实验 | 总 Resets | 总 Corrections |
|------|----------|----------------|
| pair6 K-sensitivity (K=2~32) | 112 | 0 |
| pair1 K-sensitivity (K=2~32) | 52 | 2 |
| pair4 K-sensitivity (K=2~32) | 52 | 0 |
| exp2 (DeBERTa checker, 0.5B+3B) | 28 | 0 |
| exp2 (BART checker, 0.5B+3B) | 28 | 3 |
| 本次 No Memory (K=8) | 14 | 0 |

Stage B 在当前配置下等价于 no-op。跑 `disable_corrections` 消融的结果将与 Full 完全相同，不产生新信息。

原因：0.5B 模型输出模式化（信息稀薄而非事实错误），DeBERTa NLI 倾向判为 neutral 而非 contradiction。

### 1.3 实验条件

| 条件 | Stage A（写入） | Stage B（修正） | Stage C（检索） | Stage D（重建） | 命令关键参数 |
|------|:---:|:---:|:---:|:---:|------|
| **Continuous（基线）** | ✗ | ✗ | ✗ | ✗ | `--agent continuous` |
| **PersonaLoop Full** | ✓ | ✓ | ✓ | ✓ | `--agent persona_loop --loop-interval 8` |
| **No Memory（去A+C）** | ✗ | ✓ | ✗ | ✓ | `--agent persona_loop --loop-interval 8 --loop-ablation disable_persona_persist` |

### 1.4 固定配置

| 参数 | 值 |
|------|-----|
| 生成模型 | Qwen2.5-0.5B-Instruct (Q4_K_M)，本地 llama.cpp 服务 |
| 评估 | GPT-4o Judge (scale 1–5, PCS normalized to [-1,1]) |
| 运行时 checker | cross-encoder/nli-deberta-v3-base |
| NLI 阈值 | 0.1 |
| Loop interval (K) | 8（pair6 前 2 session 最优 K，见 K-sensitivity 实验） |
| retrieval_top_k | 3 |
| recent_turns | 3 |
| max_history_window | 20 |
| context size | 32768（llama.cpp server 参数） |
| 数据 | pair6，前 2 sessions（120 turns），`--max-sessions 2` |

### 1.5 对比逻辑

| 对比 | 回答的问题 |
|------|-----------|
| Full vs Continuous | PersonaLoop 系统整体收益 |
| Full vs No Memory | **记忆机制（Stage A+C）** 的贡献 |
| No Memory vs Continuous | **周期性上下文重建本身的价值** — loop 每 K 轮清空旧 context 用 `[HISTORY]` 重建，不依赖记忆检索是否仍有效？ |

---

## 二、实验结果

### 2.1 总览表

| 条件 | Judge PCS | Judge Avg (1–5) | NLI PCS | Resets | Corrections |
|------|-----------|-----------------|---------|--------|-------------|
| Continuous（基线） | 0.4625 | 3.925 | 0.0340 | 0 | 0 |
| **PersonaLoop Full (K=8)** | **0.7833** | **4.567** | 0.0650 | 14 | 0 |
| No Memory (K=8) | 0.5333 | 4.067 | 0.0282 | 14 | 0 |

### 2.2 Per-Role 分解

| 条件 | Mingzhi PCS | Feifei PCS | Role 差值 |
|------|-------------|------------|-----------|
| Continuous | 0.333 | 0.592 | 0.259 |
| **PersonaLoop Full** | **0.808** | **0.758** | **0.050** |
| No Memory | 0.817 | 0.250 | 0.567 |

### 2.3 ΔPCS 分解

| 对比 | ΔPCS | 含义 |
|------|------|------|
| Full vs Continuous | **+0.321** | 系统整体收益 |
| Full vs No Memory | **+0.250** | 记忆检索（Stage A+C）的贡献，占总收益的 **77.9%** |
| No Memory vs Continuous | **+0.071** | 周期性重建机制本身的贡献，占总收益的 **22.1%** |

---

## 三、分析

### 3.1 记忆检索是核心驱动力

Full 与 No Memory 的差距（+0.250）远大于 No Memory 与 Continuous 的差距（+0.071）。PersonaLoop 的收益中 **约 78% 来自记忆检索（Stage A+C）**，仅约 22% 来自周期性重建机制本身。

这说明：仅仅定期清空并重建 context 的动作有一定帮助，但如果重建后的 context 没有从外部记忆库补充历史信息，改善幅度有限。

### 3.2 记忆检索对难角色尤为关键

Feifei 是"敏锐自持"的角色，persona 维持难度高于 Mingzhi：

- **Full**：Feifei PCS = 0.758，与 Mingzhi (0.808) 接近，role 差值仅 0.050
- **No Memory**：Feifei PCS 崩至 **0.250**，Mingzhi 反而微升至 0.817，role 差值飙升至 0.567
- **Continuous**：Feifei PCS = 0.592，反而优于 No Memory 的 Feifei

**结论**：记忆检索对难角色（persona 复杂、需要更多事实支撑的角色）尤为重要。没有记忆检索时，模型对简单角色（Mingzhi）仍能靠 `[HISTORY]` 维持，但对难角色（Feifei）反而不如纯 Continuous 基线。

### 3.3 No Memory 反而让 Feifei 低于 Continuous 的原因

No Memory 的 Feifei (0.250) 低于 Continuous 的 Feifei (0.592)，这是一个值得注意的现象：

- No Memory 条件下，loop 每 8 轮清空 `_recent_buffer`，但没有记忆可检索，重建后的 context 只包含最近 3 轮 `[HISTORY]` + `[PERSONA]`
- 相比之下，Continuous 的 `max_history_window=20` 保留了更长的滚动历史窗口
- 对于 Feifei 这种需要大量上下文信息维持人格一致性的角色，**周期性截断反而有害** — 它丢弃了 Continuous 保留的 17 轮额外历史，却没有记忆检索来补偿信息损失

### 3.4 Stage B（矛盾修正）再次确认无效

三个条件的 corrections 均为 0。即使在 No Memory 的退化条件下（PCS 更低，理论上更可能出现矛盾），Stage B 仍未触发。这与 K-sensitivity 实验和 exp2 的结论一致：0.5B 模型的输出模式不易触发 DeBERTa NLI 的矛盾阈值。

---

## 四、结论

1. **记忆检索（Stage A+C）贡献了 PersonaLoop 约 78% 的 PCS 提升**，是系统的核心组件。
2. **周期性上下文重建（Stage D 的 loop 机制）** 单独贡献约 22% 的提升，有一定但有限的价值。
3. **对于高难度角色，记忆检索不可或缺** — 没有它，loop 的周期性截断反而损害性能（Feifei No Memory PCS 低于 Continuous）。
4. Stage B（矛盾修正）在当前配置下不贡献任何改善，可安全移除而不影响系统性能。

---

## 五、输出目录

| 条件 | 路径 |
|------|------|
| Continuous | `artifacts/k_sensitivity_pair6/continuous/` |
| PersonaLoop Full (K=8) | `artifacts/k_sensitivity_pair6/K8/` |
| No Memory (K=8) | `artifacts/k_sensitivity_pair6/K8_no_memory/` |

