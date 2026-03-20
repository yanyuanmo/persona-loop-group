**Project Proposal**

_Persona Loop: Maintaining Persona Consistency in Long-horizon Dialogues via Periodic Context Re-initialization_

# 1\. Introduction

在人机对话系统中，赋予对话代理（dialogue agent）一致的人格（persona）是构建可信赖、有温度的交互体验的关键。无论是心理健康咨询、教育辅导还是虚拟社交，用户都期望对话代理在多轮交互中表现出连贯的性格特征、稳定的偏好和一致的价值观 \[1\]\[17\]。然而，现有的大语言模型（LLM）在长对话中普遍存在人格漂移（persona drift）的问题：随着对话轮次的增加，模型会逐渐偏离最初设定的人格描述，产生自相矛盾的回复 \[9\]\[16\]。

这一问题的根源在于 LLM 的上下文窗口（context window）机制。对话历史在 context 中不断累积，人格描述在 prompt 中的相对占比持续下降，注意力机制对早期信息的关注度衰减，最终导致人格信息被稀释。实证研究表明，随着对话轮次的增加，人格一致性会显著衰减 \[9\]。在超长对话（50 轮以上）中，即便是 GPT-4 等前沿模型，在不加任何记忆机制的情况下，长对话问答的 F1 分数也仅有 32.4（满分 100），远低于人类水平的 87.9 \[19\]。

现有方法可以从四个层面加以分类。在 Prompt 层面，最基础的做法是在 system prompt 中直接注入人格描述 \[1\]，但对话一长模型就会偏离设定；周期性人格提醒每隔 N 轮重新注入人格描述以缓解遗忘 \[9\]；PICLe 通过贝叶斯推断选择最优 few-shot 示例引导模型的行为与目标人格对齐 \[3\]；Chain of Persona 让模型在生成前先基于角色特征做推理式自问自答 \[4\]。在训练层面，NLI 引导的强化学习将一致性得分作为 reward 信号训练生成器 \[5\]；对比学习通过构造符合与不符合人格的正负样本对增强人格感知 \[4\]；质量分数条件训练（SBS）将回复与人格的语义相似度作为条件注入训练 \[6\]；LoRA 微调以较低成本让模型内化特定角色的行为模式 \[7\]；伪偏好调优利用自动生成的好坏回复对，以 DPO 算法让模型偏好一致的回复 \[8\]。在记忆与检索层面，滑动窗口摘要定期将对话历史压缩以节省上下文空间 \[10\]；RAG 将历史对话存入向量数据库按需检索注入上下文 \[11\]；长期记忆模块专门从对话中抽取人格特征存入结构化记忆 \[13\]；Persona-DB 构建多层级的人格数据库，按需取用不同粒度的信息 \[12\]；ID-RAG 利用知识图谱组织人格信息 \[18\]。在推理时控制层面，NLI Reranking 生成多个候选回复后用 NLI 模型过滤矛盾项 \[2\]；语用自我意识方法基于 Rational Speech Acts 框架让模型模拟虚拟听众判断回复是否矛盾 \[14\]；动态人格更新则周期性地处理最近的对话历史，输出修订后的人格描述重新注入 prompt。此外，Post Persona Alignment（PPA）提出先生成通用回复再检索人格记忆做后对齐的两阶段范式 \[20\]。

然而，我们注意到一个重要的研究空白：上述所有方法都在一个持续增长的上下文中做局部修补--无论是检索注入、摘要压缩还是输出过滤，都没有从根本上解决 context window 中人格信息被稀释的结构性问题。与此同时，在 AI 代码生成领域，Geoffrey Huntley 提出的 Ralph Loop 技术 \[22\] 展示了一种截然不同的范式：与其在一个不断退化的上下文中挣扎，不如周期性地重建整个上下文，每次都从干净、确定性的起点重新出发。

在评估人格一致性方面，研究者发展了多种互补的方法。Welleck et al. \[2\] 最早提出 Dialogue NLI，将一致性检测建模为自然语言推断任务，通过判断回复与人格描述之间的蕴含、中立或矛盾关系来量化一致性。Abdulhai et al. \[9\] 进一步定义了三种细粒度指标：Prompt-to-line Consistency 衡量每轮回复与初始人格设定的全局一致性，Line-to-line Consistency 检测相邻轮次之间的局部矛盾，Q&A Consistency 通过重复提问测试信念的稳定性。在人格特质层面，Jiang et al. \[15\] 提出让 LLM 完成 Big Five 人格量表（BFI），直接测量其表达的人格特质是否与目标设定吻合；Wang et al. \[16\] 则通过心理学访谈的方式（InCharacter）评估角色扮演代理的人格忠实度。此外，以 GPT-4 为代表的 LLM-as-Judge 方法和人工评估也被广泛采用。本项目将直接复用上述已有的评估方案，不另行构建新的评估体系。

受此启发，我们提出 Persona Loop 框架，将周期性上下文重建引入人格一致性维护。常规对话轮次中，模型正常基于当前上下文生成回复，不做额外操作。每隔 K 轮触发一次重置：先将最近 K 轮中与人格相关的对话片段存入外部向量数据库，再用 NLI 模型检查这些回复是否与人格描述存在矛盾，然后清空上下文中的历史对话，将人格描述重新放到最前面，拼上矛盾修正提示、从向量数据库检索到的相关历史、以及最近几轮原始对话，组成一个全新的上下文继续对话。这一设计同时解决两个问题：定期重置防止人格描述被越来越长的对话历史稀释，外部向量数据库保证模型不会因为上下文清空而丢失重要的历史记忆。据我们所知，目前尚无工作系统性地研究这种周期性上下文重建策略对人格一致性的影响。

# 2\. Objectives

**目标一：实现 Persona Loop 框架。** 一个纯推理层面的、不需要修改模型权重的周期性上下文重建系统，结合外部记忆检索和 NLI 一致性检查，可应用于任意 LLM。

**目标二：横向对比，对比验证 Persona Loop 相对于现有方法的效果。** 在多个对话长度条件下（10、30、50、100 轮），将 Persona Loop 与若干代表性基线方法进行对比实验，使用领域内已有的成熟评估指标衡量人格一致性和回复质量。

**目标三：纵向对比，对比Persona Loop方法更改自身变量的表现。**超参数方面重点分析 loop 间隔 K 值的影响。

**Bonus：通过消融实验理解各组件的作用。** 逐一移除框架中的三个核心组件，分析每个组件对最终一致性提升的贡献，明确哪些设计决策是关键的、哪些是锦上添花的。

# 3\. Research Plan

项目精力集中在框架实现和实验验证上，评估直接采用领域内已有的成熟指标，不构建新的评估体系。

## Task 1: 数据准备与超长对话合成

采用 PERSONA-CHAT \[1\] 和 ConvAI2 \[21\] 作为基础数据集，LoCoMo \[19\] 作为长对话评估基准。由于现有数据集的对话长度多为 5-15 轮，不足以充分暴露人格漂移问题，我们将利用 LLM 基于已有人格描述合成 50-100 轮的超长对话，覆盖多种人格类型，预计合成 100+ 段。同步完成数据预处理和统一格式化。

## Task 2: Persona Loop 核心框架实现

实现 Persona Loop 的完整流程。常规轮次中模型直接基于当前上下文生成回复，不与向量数据库交互。每隔 K 轮触发重置，依次执行：（a）将最近 K 轮中与人格相关的对话片段存入向量数据库；（b）用 NLI 模型批量检查最近 K 轮回复有无人格矛盾，若有则生成修正提示；（c）清空上下文，按优先级重建--人格描述、修正提示、从向量数据库检索到的相关历史片段、最近几轮原始对话。检索时不仅看语义相关性，还用 NLI 评估候选片段对人格的支持程度，优先召回最能强化当前人格的内容。同时实现一个控制器追踪轮次并协调以上流程。

## Task 3: Baseline 实现

实现若干代表性的基线方法用于对比，初步计划包括：（1）不做任何处理的 Continuous Context；（2）每 K 轮重复人格描述但不清空历史的 Periodic Reminding \[9\]；（3）基于普通语义相似度检索的 RAG；（4）定期将旧历史压缩为摘要的 Sliding Window Summary \[10\]；（5）先生成后对齐的 Post Persona Alignment \[20\]。所有方法与 Persona Loop 使用相同的底层模型和人格描述，确保公平对比。具体的 baseline 选择将在阅读相关文献后进一步确认和调整。

## Task 4: 实验

主实验在多个方法、多个对话长度的组合下进行执行，也就是前面说到的横向和纵向对比目标。评估方面采用现有方案，具体的评估方案选择将在阅读相关文献后进一步确认和调整。如果时间充足我们将进行消融实验。

## Task 5: 论文 + slides

完成paper 和演示需要的slides。

# 4\. Timeline

本项目总工期约 8 周，三人并行开发。启动时间大概从2月17号开始。

| **Tasks**                                          | **Deliverable/s**                                   | **Due**             |
| -------------------------------------------------- | --------------------------------------------------- | ------------------- |
| Task 1: Data preparation & long dialogue synthesis | Processed datasets; 100+ synthesized long dialogues | 3月3日 （用时14天） |
| Task 2: Persona Loop core framework                | Runnable pipeline code                              | 3月24日             |
| Task 3: Baseline implementation                    | Runnable baseline methods for fair comparison       | 3月24日             |
| Task 4: Experiments                                | Main results, ablation analysis                     | 4月7日              |
| Task 5: Paper & slides                             | Paper doc & slides                                  | 4月14日             |

# 5\. References

\[1\] Zhang, S., Dinan, E., Urbanek, J., Szlam, A., Kiela, D., & Weston, J. (2018). Personalizing Dialogue Agents: I have a dog, do you have pets too? ACL 2018.

\[2\] Welleck, S., Weston, J., Szlam, A., & Cho, K. (2019). Dialogue Natural Language Inference. ACL 2019.

\[3\] Choi, H. K., & Li, Y. (2024). PICLe: Eliciting Diverse Behaviors from Large Language Models with Persona In-Context Learning. ICML 2024.

\[4\] Ji, K., Lian, Y., Li, L., Gao, J., Li, W., & Dai, B. (2025). Enhancing Persona Consistency for LLMs' Role-Playing using Persona-Aware Contrastive Learning. ACL 2025 Findings.

\[5\] Song, H., Zhang, W., Cui, Y., Wang, D., & Liu, T. (2020). Generating Persona Consistent Dialogues by Exploiting Natural Language Inference. AAAI 2020.

\[6\] Saggar, A., Darling, J. C., Dimitrova, V., Sarikaya, D., & Hogg, D. C. (2025). Score Before You Speak: Improving Persona Consistency in Dialogue Generation using Response Quality Scores. ECAI 2025.

\[7\] Enhancing Persona Consistency with Large Language Models. CNIOT 2024, ACM (DOI: 10.1145/3670105.3670140).

\[8\] Takayama, J., Ohagi, M., Mizumoto, T., & Yoshikawa, K. (2025). Persona-Consistent Dialogue Generation via Pseudo Preference Tuning. COLING 2025.

\[9\] Abdulhai, M., Cheng, R., Clay, D., Althoff, T., Levine, S., & Jaques, N. (2025). Consistently Simulating Human Personas with Multi-Turn Reinforcement Learning. NeurIPS 2025 (arXiv:2511.00222).

\[10\] Xu, J., Szlam, A., & Weston, J. (2022). Beyond Goldfish Memory: Long-Term Open-Domain Conversation. ACL 2022.

\[11\] Huang, Q., Fu, S., Liu, X., Wang, W., Ko, T., Zhang, Y., & Tang, L. (2023). Learning Retrieval Augmentation for Personalized Dialogue Generation. EMNLP 2023.

\[12\] Sun, C., Yang, K., Gangi Reddy, R., et al. (2025). Persona-DB: Efficient Large Language Model Personalization for Response Prediction with Collaborative Data Refinement. COLING 2025.

\[13\] Yi, Z., et al. (2025). A Survey on Recent Advances in LLM-based Multi-turn Dialogue Systems. ACM Computing Surveys, 2025.

\[14\] Kim, H., Kim, B., & Kim, G. (2020). Will I Sound Like Me? Improving Persona Consistency in Dialogues through Pragmatic Self-Consciousness. EMNLP 2020.

\[15\] Jiang, H., Zhang, X., Cao, X., Breazeal, C., Roy, D., & Kabbara, J. (2024). PersonaLLM: Investigating the Ability of Large Language Models to Express Personality Traits. NAACL 2024 Findings.

\[16\] Wang, X., Xiao, Y., Huang, J., et al. (2024). InCharacter: Evaluating Personality Fidelity in Role-Playing Agents through Psychological Interviews. ACL 2024.

\[17\] Yu, et al. (2024). Two Tales of Persona in LLMs: A Survey of Role-Playing and Personalization. EMNLP 2024 Findings.

\[18\] Platnick, D., et al. (2025). ID-RAG: Identity Retrieval-Augmented Generation for Long-Horizon Persona Coherence in Generative Agents. LLAIS Workshop @ ECAI 2025.

\[19\] Maharana, A., Lee, D.-H., Tulyakov, S., Bansal, M., Barbieri, F., & Fang, Y. (2024). Evaluating Very Long-Term Conversational Memory of LLM Agents. arXiv 2024 (LoCoMo).

\[20\] Chen, Y.-P., Nishida, N., Nakayama, H., & Matsumoto, Y. (2025). Post Persona Alignment for Multi-Session Dialogue Generation. EMNLP 2025 Findings.

\[21\] Dinan, E., Logacheva, V., Malykh, V., Miller, A., Shuster, K., Urbanek, J., Kiela, D., Szlam, A., Serban, I., Lowe, R., Prabhumoye, S., Black, A. W., Rudnicky, A., Williams, J., Pineau, J., Burtsev, M., & Weston, J. (2019). The Second Conversational Intelligence Challenge (ConvAI2). arXiv preprint arXiv:1902.00098.

\[22\] Huntley, G. (2025). Ralph Wiggum as a Software Engineer. <https://ghuntley.com/ralph/>