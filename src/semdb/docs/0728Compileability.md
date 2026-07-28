结论：BARGAIN 的统计认证机制可以迁移，但不能直接套用。它适合作为“生成代码是否值得信任”的认证内核；要用于本项目，还需要解决代码自适应生成、缺少逐样本置信分数、LLM oracle 噪声和多算子
  组合四个问题。

  我查看了当前设计和已有保证方案，包括 docs/PROXY_GUARANTEE_PLAN.md、docs/VALSET_GUARANTEE_IMPL.md 和 docs/0727CurrentDesign.md。现有采样、oracle label cache、validation trace 和
  refinement loop 已经提供了基础；真正缺的是严格的“搜索—认证”边界，以及 optimizer 可使用的 compilability 定义。

  ## 1. 先明确要保证什么

  对固定数据快照 (D)、冻结的 LLM operator (O) 和最终生成代码 (p)，首先研究：

  [
  A_D(p,O)=\frac{1}{|D|}\sum_{x\in D}\mathbf{1}[p(x)=O(x)]
  ]

  认证器输出：

  [
  \Pr(A_D(p,O)\ge T)\ge 1-\delta
  ]

  这里必须区分两种性质：

  - Fidelity：Python code 是否复现冻结 LLM 的行为。
  - Semantic correctness：LLM 和 Python code 是否都符合真实语义。

  BARGAIN主要解决第一种。即使代码和 LLM 达到 99% agreement，也不等于真实正确率达到 99%。如果要声明 semantic correctness，必须额外引入人工 gold labels、可靠参考系统或 oracle noise
  bound。

  统计保证本身不会让代码更准确。完整系统应分成：

  1. Code synthesis：生成和改进代码。
  2. Certification：判断最终代码能否被信任。
  3. Routing：根据认证结果选择 compile、LLM interpret 或 hybrid。

  ## 2. BARGAIN 能搬什么、不能搬什么

  BARGAIN (https://arxiv.org/abs/2509.02896)将廉价 proxy 与昂贵 oracle 组成 cascade，通过 proxy score、无放回自适应采样和 anytime-valid 统计检验寻找可信阈值。这部分与本项目高度契合。

   BARGAIN 组件                               迁移结论
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   固定数据集上的有限总体保证                 可直接采用
  ─────────────────────────────────────────  ────────────────────────────────────────────
   LLM 作为 oracle、Python code 作为 proxy    可采用
  ─────────────────────────────────────────  ────────────────────────────────────────────
   自适应采样和 anytime-valid 检验            可采用，但必须匹配实际采样设计
  ─────────────────────────────────────────  ────────────────────────────────────────────
   高置信样本运行代码、其余交给 LLM           可直接形成 hybrid routing
  ─────────────────────────────────────────  ────────────────────────────────────────────
   Accuracy/precision/recall 分别认证         可采用
  ─────────────────────────────────────────  ────────────────────────────────────────────
   Proxy 自带连续置信分数                     当前代码没有，需要新增
  ─────────────────────────────────────────  ────────────────────────────────────────────
   Proxy 在认证前固定                         当前 refinement 会自适应选择代码，需要隔离
  ─────────────────────────────────────────  ────────────────────────────────────────────
   Oracle 被视为固定真值                      当前 LLM 可能随机、漂移，需要重新定义
  ─────────────────────────────────────────  ────────────────────────────────────────────
   每条记录独立预测                           join、top-k、aggregation 不完全满足

  最大的三个障碍是：

  1. 代码通常只有 hard prediction，没有逐样本 score。没有 score 时，BARGAIN只能退化为“整段代码全部可信或全部不可信”，无法实现高覆盖率 cascade。
  2. 当前 codegen 根据 validation mismatch 反复修改代码。如果最后仍用这些样本认证，会产生 selection bias。必须把数据分为 SELECT 和密封的 CERT；或者使用 Learn Then Test
     (https://arxiv.org/abs/2110.01052) 一类能处理多候选选择的校准方法。

  3. LLM oracle 并不是真值。MVP阶段应冻结 model、prompt、decoding、版本和输入序列化方式，保证的是“相对于该 oracle 的 fidelity”；之后再研究多次投票、人工锚点和 oracle error bound。

  因此，BARGAIN迁移结论是：统计内核可迁移，端到端保证不可直接迁移。

  ## 3. 把 compilability 定义成可优化的量

  不要把 compilability 定义为“代码能不能生成”或 validation accuracy，而应定义为：

  > 在给定准确率阈值、失败概率和预算下，能够被统计认证、从而安全交给代码执行的数据比例。

  对代码 (p) 的样本分数 (s_p(x))，定义 certified coverage：

  # [
  \operatorname{Comp}(p;T,\delta)

  \max_\tau
  \frac{|{x:s_p(x)\ge\tau}|}{|D|}
  ]

  约束是在该区域内能认证：

  [
  A_{{s_p(x)\ge\tau}}(p,O)\ge T
  ]

  这样 optimizer 的动作自然变为：

  - compile-only：全体数据都可认证。
  - hybrid：高置信区域运行代码，其余调用 LLM。
  - interpret-only：认证覆盖率太低或编译成本无法摊销。
  - recompile/reject：代码运行失败或 pilot 结果显示仍有改进空间。

  对应成本模型：

  [
  C =
  C_{\text{compile}}
  +C_{\text{select}}
  +C_{\text{cert}}
  +N C_{\text{code}}
  +N(1-\text{coverage})C_{\text{LLM}}
  ]

  这使 compilability 能与 selectivity、cardinality、latency 一起成为一等 cost-model property。类似的成本与质量联合优化可参考 Abacus (https://arxiv.org/abs/2505.14661) 和 Palimpzest
  (https://arxiv.org/abs/2405.14696)，但它们不能替代统计认证。

  ## 4. 核心研究问题

  建议收敛为四个 RQ：

  - RQ1：固定代码的 fidelity certification
    最终代码冻结后，多少 oracle labels 可以认证 accuracy、precision 或 recall 达标？

  - RQ2：代码置信分数与 selective execution
    能否为生成代码构造有效的逐样本风险排序，使 BARGAIN-style hybrid 比 all-code 和 all-LLM 更便宜？

  - RQ3：operator compilability prediction
    在编译前和小规模 pilot 后，能否预测某个 operator 最终可达到的 certified coverage，并据此 routing？

  - RQ4：query-level composition
    filter、map、join 等单算子保证如何组合成端到端查询保证？数据漂移后证书何时失效？

  ## 5. 分阶段研究计划

  ### Phase 0：统计契约和研究边界

  先限定：

  - 固定数据快照；
  - 单个 filter/map，随后扩展到 pairwise join；
  - 固定 LLM oracle；
  - SELECT 只用于生成和 refinement；
  - CERT 只用于最终认证；
  - SemBench gold truth 只能用于研究评估，不能进入 codegen 或 certification。

  产物是正式的 estimand、certificate schema、oracle contract 和数据泄漏规则。

  ### Phase 1：BARGAIN 可迁移性验证

  使用固定的已有 Python operator，在可获得完整 oracle labels 的 benchmark 上模拟不同采样过程：

  - uniform without replacement；
  - anytime-valid betting test；
  - Clopper–Pearson/Hoeffding 基线；
  - 当前 stratified/weighted sampling。

  重点不是平均 accuracy，而是重复实验中的：

  - false-certificate rate；
  - empirical guarantee violation；
  - certification sample size；
  - optional stopping 是否仍有效。

  只有 empirical violation 能控制在 (\delta) 附近，才能进入下一阶段。针对 semantic operators 的其他统计路线可对照 LOTUS (https://www.vldb.org/pvldb/vol18/p4171-patel.pdf) 和 SUPG
  (https://www.vldb.org/pvldb/vol13/p1990-kang.pdf)。

  ### Phase 2：All-or-nothing certificate MVP

  先不做逐样本置信分数：

  - codegen 在 SELECT 上完成所有迭代；
  - 冻结唯一 final code；
  - 在未见过的 CERT 上进行一次认证；
  - 认证通过则整段代码上线，否则全部回退 LLM。

  同时实验多个生成代码候选时的 multiplicity correction，证明“best-of-K 后认证”不会产生虚假保证。

  ### Phase 3：置信分数和 BARGAIN hybrid

  研究代码可用的 confidence signals：

  - 多个独立生成程序是否一致；
  - 命中的代码分支属于精确规则、模糊规则还是 fallback；
  - 字符串距离、数值阈值 margin、实体匹配 margin；
  - assertion/precondition 是否满足；
  - 输入是否超出 synthesis 时见过的 value space；
  - 显式 ABSTAIN；
  - 小规模 pilot 上训练的 error predictor。

  评价重点是 selective-risk curve 和 certified coverage，而不是 score 的表面数值。若这些分数不能把错误样本排到低置信区域，BARGAIN hybrid 就没有实际价值。

  ### Phase 4：Compilability model 与 operator router

  建立 operator-level 数据集，标签为：

  - 最大 certified coverage；
  - 达标所需 oracle labels；
  - compile/certification cost；
  - 最优动作：compile、hybrid 或 interpret。

  特征分两类：

  - 编译前：operator type、predicate ambiguity、是否需要世界知识、closed/open vocabulary、模态、pairwise 程度、数据熵和基数。
  - 编译后：preflight 结果、branch coverage、abstain rate、program agreement、score 分布、pilot confidence interval。

  比较规则模型、监督分类器和成本敏感 router，并报告相对于 oracle-best route 的 regret。

  ### Phase 5：算子组合

  逐类处理，不能强行共用一个 accuracy guarantee：

  - filter：accuracy/precision/recall；
  - map/extraction：exact match 或 bounded loss；
  - join：candidate-generation recall 与 pair-classification precision 分开；
  - aggregation：prediction-powered inference 或 query-answer confidence interval；
  - top-k/ranking：单独研究 recall@k/nDCG 保证。

  多算子查询需要进行 (\delta) allocation，或推导针对查询结果的直接保证。

  ### Phase 6：漂移和证书复用

  证书必须绑定：

  - query/operator specification；
  - generated code 与 helper hash；
  - oracle model/prompt/version；
  - dataset snapshot或适用分布；
  - sampling design；
  - (T,\delta) 和认证指标。

  发生数据漂移、代码变化或 oracle 升级时，应触发 shadow sampling 或重新认证。

  ## 6. 当前 codegen 迭代信息流需要怎样调整

  根据当前设计，首轮由 signature/API/solver 信息生成代码，后续 solver refinement 主要使用：

  - 当前最佳代码；
  - validation sampled inputs；
  - LLM oracle labels；
  - 代码执行结果；
  - mismatch、exception 和 validation metrics/traces。

  这些信息都可以继续用于提高代码准确率，但必须全部归入 SELECT 侧。CERT 的输入、标签、逐样本结果和中间统计量不得反馈给 codegen；否则证书就再次变成训练集分数。

  ## 7. 成功与否的判据

  研究不应以“平均准确率提高”作为唯一成功条件。核心指标应是：

  - 在重复实验中，错误认证概率是否不超过 (\delta)；
  - 达到 (T) 时需要多少 oracle calls；
  - certified coverage；
  - 总成本和 latency；
  - router regret；
  - 因编译但无法认证造成的浪费；
  - oracle noise 对结论的敏感度。

  关键否证条件：

  - 置信分数在多个 workload 上与随机排序无显著区别；
  - certification 成本加 compilation 成本长期高于直接 LLM；
  - oracle 自身不稳定性高于 code–oracle disagreement；
  - 单算子证书无法转化为查询级保证。

  Phase 1 的审查结论是：方向可行，但 MVP 必须收窄到固定数据快照、冻结 oracle、单算子和 fidelity guarantee。 这与编译式 semantic operator 论文提出的 fidelity、confidence score 和
  per-operator routing 问题完全对齐，原始 vision paper (https://arxiv.org/abs/2607.13407)也将它们列为核心开放问题。

  本轮没有修改代码。下一步应先完成 Phase 0 的 statistical contract 和 Phase 1 的 BARGAIN assumptions matrix，再决定具体实现接口。