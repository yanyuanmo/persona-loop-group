# Persona Loop — Implementation Plan

## 当前状态

核心流程（Stage A→B→C→D）已实现，但与 proposal 有两处明显缺口：

---

## TODO: 未实现特性

### TODO-1: Stage A — Persona 相关性过滤后再存入向量 DB

**Proposal 原文：**  
> 「将最近 K 轮中**与人格相关**的对话片段存入向量数据库；高相关度优先，低相关度降权」

**现状：**  
```python
# persona_loop_agent_v2.py，Stage A
for snippet in self._recent_buffer:
    self.memory.add(text=snippet)  # 无差别全存，无过滤
```

**方案：**  
在存入之前，用 NLI 或 embedding 余弦相似度判断每条 snippet 与 `persona_summary` 的相关性，只存超过阈值的片段（或给分数高的片段加权 boost）。

**影响：** 中等。向量库内容更干净，Stage C 检索质量提升；但 K 轮内若人格相关内容少，库可能变稀疏。

---

### TODO-2: Stage C — NLI Reranking（检索后用 NLI 对人格支持度重排）

**Proposal 原文：**  
> 「检索时**不仅看语义相关性，还用 NLI 评估候选片段对人格的支持程度，优先召回最能强化当前人格的内容**」

**现状：**  
```python
# persona_loop_agent_v2.py，Stage C
retrieved = self.memory.search(query=user_prompt, top_k=self.retrieval_top_k)
# 纯 cosine 相似度，无 NLI reranking
```

**方案：**  
先 `search(query, top_k=retrieval_top_k * 2)` 召回双倍候选，再对每条候选计算 `NLI(premise=snippet, hypothesis=persona_summary)` 的 entailment 分数，按 `(entailment - contradiction)` 降序取前 `top_k` 条。

**影响：** 较大。过滤掉与 persona 矛盾的历史片段，避免把 "有矛盾的旧历史" 注入上下文。NLI 调用次数 = `retrieval_top_k * 2` per loop，开销可控。

---

## 实验计划

### Exp-1: 当前配置基线对比（已跑 / 进行中）

| 组 | Agent | 说明 |
|---|---|---|
| cont | continuous | 无记忆，no loop |
| pl   | persona_loop | 当前实现（Stage A全存，Stage C无rerank） |

### Exp-2: 消融 Stage B（NLI 修正提示）

设置 `--loop-nli-threshold 999` 禁用 Stage B，观察 PCS 变化。

### Exp-3: 补齐 Stage A 过滤后的效果

实现 TODO-1 后，比较 `pl` vs `pl_stageA_filtered`。

### Exp-4: 补齐 Stage C NLI Reranking 后的效果

实现 TODO-2 后，比较 `pl` vs `pl_stageC_nli`。

---

## 优先级

1. **先跑 Exp-1**（当前代码直接跑，建立基线）
2. **实现 TODO-2**（NLI reranking，影响大，实现难度低）
3. **实现 TODO-1**（persona 过滤，需要额外 NLI/embedding 调用）
4. **跑 Exp-2/3/4**（消融，验证各组件贡献）
