# 采样验证集 + 统计保证驱动的迭代代码生成 — 实现计划 (semdb only)

配套研究文档：`docs/PROXY_GUARANTEE_PLAN.md`（BARGAIN / LOTUS / MOAR 的理论定位）。
本文只谈**工程落地**：改哪些文件、数据格式、命令、判据、测试。

---

## 0. 现状（读代码所得，作为改动基线）

| 组件 | 位置 | 现在做什么 |
|---|---|---|
| refine 循环 | `orchestrator.mjs:823 refineLoop` | iter_0 生成 → 运行 → 打分；iter 1..N 带 feedback 重生成 → keep/rollback |
| 改进判据 | `orchestrator.mjs:177 checkSemdbImprovement` | 正确性优先，其次 **裸 F1 arg-max** |
| 停止判据 | `orchestrator.mjs:189 shouldContinueSemdb` | maxIter / F1==1.0 / 连续 2 轮不改进 |
| 打分（compiled） | `orchestrator.mjs:786 scoreWithDiff` → `evaluate.py --emit-diff` | 对 **SemBench ground truth** 算 F1 + FP/FN 样例 |
| 打分（direct/val 模式） | `orchestrator.mjs:765 scoreInference` → `evaluate.py --score-inference` | 对**手工 `val.json`** 算 per-row accuracy + mistakes |
| val 文件格式 | `tests/fixtures/val_q3a.json` | `{query, attr, labels: {id: value}}` |
| 反馈渲染 | `renderFeedback` | 失败→修错优先；否则 F1/accuracy + FP/FN/mistakes + history |
| 遥测 | `telemetry.json` | phases（每 agent 的 ms / tokens / cost / calls）、`refine.f1_history` |

**四个必须补的洞：**

1. **验证集是手工的**——没有采样设计、没有 LLM 标注、没有权重、没有可复现的 seed。
2. **选择偏差没被处理**——K 个候选在同一份标签上 arg-max，胜者的验证分是乐观偏估；且当前 compiled 分支直接用 GT 进环，论文口径上不成立。
3. **反馈只有质量一维**——cost / latency / 编译错误分类 / 程序分支归因都没进 prompt，agent 无法针对它们优化。
4. **没有任何区间**——所有决策用点估计，n=5..120 时噪声远大于迭代间真实差异（BARGAIN 已经证明 CLT 型保证在小样本下 >75% 概率失效）。

---

## 1. 目标与不做的事

**做：**
- 从语料按 **uniform / stratified / importance** 三种设计抽样，用**强 LLM (k-vote) 标注**，产出可复现、带权重、带 provenance 的验证集；SELECT / CERT 两份不相交。
- 每轮生成的代码产出一张 **scorecard**：quality（带置信下界）/ cost / latency / correctness / compile-error 分类 / 分支归因。
- 用 **配对显著性判据**替代裸 arg-max 决定 keep-or-rollback，用 **LTT + Bonferroni** 在密封 CERT 集上出最终证书。
- feedback 里把这些信号结构化喂回 codegen agent。

**不做（本期）：**
- 不改 BARGAIN 的覆盖率切点算法进主循环（放 Phase 4，需要程序输出 per-row score）。
- 不碰 gendb / VADAR 上游、不碰 compiled 模式的 GT 评测口径（GT 只做 meta-evaluation）。

---

## 2. 新增 / 改动文件清单

```
src/semdb/
  build_valset.py        新增  采样设计 + 调用 oracle 标注 + 写 select.json / cert.json
  oracle_label.py        新增  强 LLM 标注器：k-vote、磁盘缓存、预算闸、成本/延迟核算
  certify.py             新增  有限样本区间（Clopper–Pearson / WSR-betting）+ LTT 证书
  scorecard.py           新增  把一轮的 run/compile/quality/cost/latency 归一成 scorecard.json
  evaluate.py            改    --score-inference 支持 weights + 分层估计 + 分支归因字段
  orchestrator.mjs       改    新 flags、acceptIteration、renderFeedback v2、compile 前置闸、telemetry
  semdb.config.mjs       改    valset / oracle / certify 默认值
  tests/                 新增  见 §9
  docs/VALSET_GUARANTEE_IMPL.md  本文
```

---

## 3. Phase 1 — 验证集构造（`build_valset.py` + `oracle_label.py`）

### 3.1 抽样设计

三种，`--method` 选择，**全部无放回（WoR）+ 固定 seed**：

- **`uniform`**（默认）：SRSWOR。估计量 = 简单均值，区间 = Clopper–Pearson（含有限总体修正 FPC）。最省事，先跑通。
- **`stratified`**：分层依据必须是**在生成任何程序之前就可得**的廉价信号，避免"用被测对象定义分层"：
  - 文本：长度分桶 × 关键词命中 × sentence-embedding k-means（k=4~8）簇 id；
  - 图像：CLIP embedding k-means 簇 id（复用 `semvision.py` 的 encoder）。
  - 分配：先导样本估计层内方差 → Neyman 分配；样本不足则按比例分配。估计量 = 分层加权均值，区间 = 各层 CP 按 δ/H 联合，或对加权有界变量用 WSR。
- **`importance`**：以 iter_0 程序的 per-row 置信/margin 为提议分布（`PROXY_GUARANTEE_PLAN.md` §4.1 的信号 A/C），处理**类别不平衡**（正例 ~6%，recall 界否则全落在负例上）。估计量 = 自归一化 IS，权重截断到 `w_max`，区间用 WSR（对有界加权变量有效；**CP 在此不成立**，代码里要 assert 住）。

> 设计原则：估计量和区间必须与抽样设计匹配。这是最容易写错、也最容易被 reviewer 抓的地方，`certify.py` 用 `design.method` 分派，不允许调用方自选。

### 3.2 SELECT / CERT 两份预算

- `select.json`：循环里随便用、随便看，**不产生任何对外声称**。默认 n=100~150。
- `cert.json`：密封，orchestrator 全程不读；只在跑完后由 `certify.py` 触碰一次。默认 n=60~100。
- 两者从同一设计中一次性 WoR 抽出，保证不相交；`build_valset.py` 写 `split_manifest.json` 记录 id 归属，任何一侧被读取都留痕。

### 3.3 Oracle 标注器

`oracle_label.py`：

```python
label_rows(rows, *, attr, value_space, query_nl, model, endpoint, k=3,
           budget_calls, cache_dir) -> dict[id, LabelRecord]
```

- 复用现成 HTTP 客户端：文本走 `semextract.gen_endpoint`（guided-JSON），图像走 `semvision` 的 VLM 路径——**不新开客户端**。
- **k-vote self-consistency**（默认 k=3，temperature>0）：多数票为标签，同时存全部投票 → 后续估计 oracle 自身错误率 η（Phase 5）。
- **磁盘缓存**：`runs/_labels/<corpus>/<sha1(attr|value_space|query_nl|model|k)>.json`，键 `(corpus, row_id)`。同一语料的 q3a–q3g 共享属性 → 标签一次买、多查询摊销（研究文档 §4.7）。
- **预算闸**：`--label-budget` 为硬上限，超出直接抛错而不是静默截断；记录 `cost_usd` / `ms` / `calls` 进 provenance。

### 3.4 输出格式（向后兼容现有 `val.json`）

```json
{
  "query": "q3a", "attr": "genre", "split": "select",
  "labels": { "m1": "comedy", "m17": "drama" },
  "weights": { "m1": 20.0, "m17": 12.5 },
  "design": { "method": "stratified", "seed": 7, "N": 2000, "n": 120,
              "strata": { "c0": {"N_h": 800, "n_h": 40}, "c1": {...} },
              "stratum_of": { "m1": "c0" } },
  "provenance": { "oracle_model": "...", "k": 3,
                  "agreement": { "m1": 1.0, "m17": 0.67 },
                  "cost_usd": 0.31, "ms": 42000, "calls": 360 }
}
```

`labels` 之外全部可选 → 现有 `tests/fixtures/val_q3a.json` 和 `--val-file` 路径**不需要改**；`weights` 缺失时 `score_inference` 退化为现在的未加权行为。

### 3.5 命令

```bash
python3 src/semdb/build_valset.py \
  --benchmark mmqa --query q3a \
  --corpus src/semdb/runs/_corpus/<corpus>.csv --id-col id --text-col plot \
  --attr genre --value-space "comedy,drama,documentary,..." \
  --method stratified --n 120 --cert-n 60 --seed 7 \
  --oracle-model Qwen/Qwen3-32B-Instruct --endpoint http://127.0.0.1:8000/v1 --k 3 \
  --label-budget 600 --out src/semdb/runs/_val/mmqa-q3a/
# → select.json, cert.json, split_manifest.json, labels_cache 写入 runs/_labels/
```

---

## 4. Phase 2 — 每轮 scorecard（五维信号）

`scorecard.py` 汇总一轮，写 `iter_N/scorecard.json`：

```json
{
  "iter": 2,
  "compile": { "ok": false, "stage": "static",
               "error_class": "NameError", "file": "solve_q3a.py", "line": 87,
               "snippet": "...", "traceback_tail": "..." },
  "run":     { "status": "ok", "rows_in": 120, "rows_out": 118,
               "row_exception_rate": 0.017, "top_exception": "KeyError('genre')" },
  "quality": { "estimator": "stratified", "n": 120, "point": 0.842,
               "lcb": 0.771, "ucb": 0.897, "delta": 0.05, "method": "wsr",
               "precision": {"point":0.88,"lcb":0.79}, "recall": {"point":0.71,"lcb":0.58},
               "paired_vs_best": { "wins": 14, "losses": 5, "p_value": 0.032, "diff_lcb": 0.021 } },
  "cost":    { "agent_usd": 0.42, "agent_tokens": {...},
               "runtime_llm_calls": 34, "runtime_usd": 0.006, "oracle_calls_used": 0 },
  "latency": { "codegen_ms": 61000, "exec_ms": 8400, "exec_ms_per_row": 71 },
  "strata":  [ {"branch":"regex_hit","n":60,"acc":0.95},
               {"branch":"clip_score","n":40,"acc":0.80},
               {"branch":"fallback",  "n":20,"acc":0.35} ]
}
```

三处需要新采集的信号：

1. **compile 前置闸**（orchestrator，跑之前）：`py_compile` → 现有 `validateOfflineVadarFile` → import smoke（`python3 -c "import solve_q3a"` 于隔离 cwd）。任何一步失败即 `compile.ok=false`，**不浪费一次执行**，直接进 fix-first 反馈，并带上精确行号+源码片段（现在只给 stderr 尾巴）。
2. **per-row 异常率**：solver 契约里已有 `trace_<q>.json`，扩一个 `errors: {id: "ExceptionClass: msg"}` 字段 —— 一行炸掉不再等于整轮 crash，可以定位到具体行/分支。
3. **分支归因**（研究文档 §4.6，只有"proxy 是程序"才做得到）：`trace` 增 `branch: {id: "regex_hit"}`。prompt 契约要求 solver 在每个判定分支打一个短标签。它同时是**分层反馈**和 Phase 4 置信分数的原料。

---

## 5. Phase 2 — 判据：把 arg-max 换成配对显著性

改 `checkSemdbImprovement` → `acceptIteration(best, next, cfg)`，字典序：

1. **gate**：`compile.ok` 且 `run.status==="ok"` 的候选严格优于不满足者（保持现有 correctness-first 语义）。
2. **quality**：**同一批已标注行上的配对比较**——两程序在同一 val 行上的对错构成配对样本，用 McNemar 精确检验 / 不一致对上的精确二项区间。仅当 `diff_lcb > 0`（即 δ 水平下确有提升）才 promote。
   - 为什么配对：两个独立 CI 在 n=120 时几乎必然重叠，会把循环卡死；配对只看 discordant pairs，方差小一个量级。
   - 兼作**噪声地板**（研究文档 §3.2(d) ladder）：提升不超过噪声就不 promote，也就不泄露信息，选择偏差随之被压住。
3. **tie-break**：`cost.agent_usd + runtime_usd` 更低 → `latency.exec_ms_per_row` 更低。
4. **Pareto 记录**：即使未 promote，也把 (quality_lcb, cost, latency) 三元组写入 `pareto.json`，供最终多目标选择与画图。

停止判据同步：`shouldContinueSemdb` 增两条——(a) `quality.lcb >= target` 即可提前停（不必等 point==1.0）；(b) 连续 2 轮 `diff_lcb <= 0` 视为 stall（比现在的 `improved` 布尔更稳）。

---

## 6. Phase 2 — 证书（`certify.py`，循环之外）

```bash
python3 src/semdb/certify.py --run-dir src/semdb/runs/mmqa-q3a \
  --cert-file src/semdb/runs/_val/mmqa-q3a/cert.json \
  --target 0.90 --delta 0.05 --out certificate.json
```

- 输入：本次 run 里**实际被 promote 过**的 K 个候选（从 `pareto.json` 读，K 通常 2~6）。
- 方法：Learn-then-Test —— 每个候选一个假设 `H_i: risk_i > 1-target`，用有限样本界（uniform→Clopper–Pearson；stratified/importance→WSR-betting）逐个检验，Bonferroni 用 δ/K。存活者即在 δ 水平被认证。
- 输出：
  ```json
  { "certified": true, "program": "iter_3", "target": 0.90, "delta": 0.05,
    "lcb_cert": 0.912, "n_cert": 60, "K_tested": 4, "correction": "bonferroni",
    "naive_select_score": 0.958, "optimism_gap": 0.046 }
  ```
- `optimism_gap` = SELECT 上的胜者分 − CERT 上的认证下界，就是研究文档 §6 那张"selection-bias money plot"的数据点，**用现有 run 就能先画出来**。
- 明确写 `certified: false` 而不是硬凑——mmqa 若干查询 gold 只有 1~13 条，在任何合理 δ 下都不可认证，如实报告。

---

## 7. Phase 2 — 反馈 v2（`renderFeedback`）

在现有块基础上加四段，且**结构化、可截断**：

```
## OBJECTIVE
maximize per-row accuracy (95% LCB) on a sampled+LLM-labeled validation set,
subject to cost <= $X / query and exec <= Y ms/row.

## LAST RUN — SCORECARD
accuracy 0.842 (95% LCB 0.771, n=120 labeled rows)   [prev best 0.821 / LCB 0.744]
cost  $0.42 agent + 34 runtime LLM calls   latency 61.0s codegen / 71 ms per row
compile OK · 2/120 rows raised KeyError('genre')

## WHERE THE ERROR LIVES  (accuracy by program branch)
  regex_hit    60 rows  0.95
  clip_score   40 rows  0.80
  fallback     20 rows  0.35   <-- 65% of all errors are here
## MISLABELED ROWS (stratified sample, 5 per branch)
  - id=... predicted=... expected=... text="..."

## GUARDRAILS
These 120 ids are a RANDOM SAMPLE of a 2000-row corpus. Do NOT special-case any id,
text snippet, or expected value from the list above — the program is scored on the
full corpus. Improve the RULE, not the sample.
```

- mistakes 采样从"前 15 条"改为**按分支分层采样**，保证少数分支的错误不被多数分支淹没。
- `## GUARDRAILS` 配一个静态检查（见 §9）：生成代码若出现 val 集里的 id/字面量，该轮直接判负。这是真实的 reward-hack 面，必须堵。

---

## 8. Phase 2 — orchestrator 接线

新增 flags（全部有默认值，不传时行为与现在完全一致）：

| flag | 默认 | 作用 |
|---|---|---|
| `--val-file <p>` | 已有 | SELECT 集（现在起可带 weights/design） |
| `--cert-file <p>` | null | 密封 CERT 集路径，只写进 telemetry，循环不读 |
| `--accept-rule {f1,paired-lcb}` | `f1` | 切到新判据；先默认旧行为，验证后翻转 |
| `--delta <d>` | 0.05 | 区间/检验的 δ |
| `--quality-target <t>` | null | 达到 LCB≥t 提前停 |
| `--cost-budget-usd <c>` | null | 进 feedback 的 objective 行 + tie-break |

`semdb.config.mjs` 加一块 `guarantee: { delta, acceptRule, valN, certN, oracleModel, oracleK, labelBudget, samplingMethod }`。

`telemetry.json` 的 `refine` 块扩展：`{ mode, objective, val: {file, n, method, seed}, cert: {file, n}, scorecards: [...], pareto: [...], certificate: {...} }`。

---

## 9. 测试（沿用现有约定：py 用 pytest，纯函数逻辑用 node 脚本）

| 文件 | 断言 |
|---|---|
| `tests/test_build_valset.py` | 同 seed 可复现；select ∩ cert = ∅；WoR 无重复；分层分配 Σn_h = n；权重 Σ 1/π = N |
| `tests/test_oracle_cache.py` | 缓存键命中/未命中；k-vote 多数票与 agreement 计算；超预算抛错而非截断 |
| `tests/test_certify.py` | CP 下界对齐已知数值表；**蒙特卡洛覆盖率检验**（2000 次试验，名义 95% 的实际覆盖 ≥95%）；加权估计量无偏；Bonferroni 随 K 单调收紧；importance 设计调用 CP 时 assert 报错 |
| `tests/test_scorecard.py` | compile 错误分类（SyntaxError/ImportError/NameError）；per-row 异常率；分支归因聚合 |
| `tests/test_accept_rule.mjs` | 纯函数 `acceptIteration`：gate 优先级、diff_lcb≤0 不 promote、cost/latency tie-break |
| `tests/test_feedback_v2.mjs` | 渲染包含 objective/scorecard/branch/guardrails 四段；分层采样 mistakes |
| `tests/test_val_leakage.py` | 生成代码含 val id 字面量 → 该轮判负 |

回归：`test_refine_pure.mjs`、`test_val_feedback.mjs` 必须继续通过（默认 `--accept-rule f1` 时行为不变）。

---

## 10. Phase 3+（可选，按研究文档优先级）

- **P3 标签效率**：分歧驱动采样（K 个程序不一致的行信息量最大）+ 按程序分支分层 + 候选 racing（successive halving）。指标：*认证到 (T, δ) 所需标签数*。
- **P4 覆盖率刻度**：程序输出 `(label, conf)` 或 `ABSTAIN` → 接 BARGAIN_R/_P 选切点；"代码不够好"变成"oracle 调用多一点"的**成本刻度**，循环目标改为"固定认证质量下的 oracle 成本"。
- **P5 噪声 oracle**：人工 anchor set 估 η → 界传播 `true ≥ T − η` → DSL/PPI 双层估计。产出"保证退化曲线"。
- **P6 meta-evaluation**：SemBench GT **只做元评测，永不进环**。核心图：(i) 标签数 vs 认证质量；(ii) 固定 T 下省下的 oracle 调用；(iii) ≥100 次试验的**经验违约率**（SUPG 在此失败 >75%）；(iv) optimism_gap 图（**今天就能画**）。

---

## 11. 里程碑与验收

| 里程碑 | 内容 | 验收 |
|---|---|---|
| M1（~1d） | scorecard + compile 前置闸 + feedback v2，**不需要任何新标签** | 在既有 `runs/mmqa-q2a` 上重跑，iter 级 scorecard 齐全；compile 错误能定位到行 |
| M2（~2d） | `oracle_label.py` + `build_valset.py`（uniform 先行） | q3a 上产出 select/cert，标签成本与缓存命中可复现 |
| M3（~2d） | `certify.py` + `--accept-rule paired-lcb` 接线 | 蒙特卡洛覆盖率测试通过；q3a 产出 certificate.json 与 optimism_gap |
| M4（~2d） | stratified / importance 设计 + 分层估计 | 同预算下 CI 宽度较 uniform 显著更窄（在 q3a/q3b 上量化） |

---

## 12. 风险与诚实的边界

- **oracle 有噪声**——BARGAIN/LOTUS 都把 oracle 定义为真值，我们不能。M1–M4 的所有"保证"都是**对 oracle 的一致性保证**，写进证书文案里，别写成对真值的保证；真值修正留 P5。
- **小 gold / 小语料**：n<50 时任何 T≥0.9 都不可认证——`certify.py` 要能直接回答"该目标在此样本量下统计上不可达"，而不是给一个虚假区间。
- **配对检验的多重性**：每轮都做一次 promote 检验 → 迭代数 I 次比较。用 δ_promote = δ/2 与 δ_cert = δ/2 拆分，或直接声明 promote 检验只是启发式（不对外声称），对外数字全部来自密封 CERT 集。倾向后者，更干净。
- **非成员类指标**（聚合/排序/ARI，ecomm）不适配 precision/recall 目标 → 走 PPI，或明确 scope out，别硬套。
- **join 类查询**标注单位是 pair（M×N），需要先 blocking，本期不覆盖。
