# Multi-call-site 与 Self-join Validation 实现计划

状态：设计完成，尚未开始修改运行代码。

目标是让 SemDB 在使用 `--val-rate` 时能够安全运行 EComm 的复杂查询：

- q7/q9：带确定性前缀过滤的 AI self-join。
- q10/q11：多个 per-row filter 与 pairwise join predicate 的组合。
- q12：`AI.IF` 与 `AI.GENERATE` 的组合。
- q14：per-row filter、pairwise join、`AI.SCORE` 和 group-wise top-1 的组合。

最终要求不是“分别测量每个 AI predicate”，而是在不向 code-generation agent
泄露完整 benchmark ground truth 的前提下，同时提供：

1. 每个 AI call site 的 operator-level 指标与错误样本。
2. 整条查询的 query-level 指标与错误样本。
3. 可审计的 sampling population、inclusion probability 和 Oracle 调用成本。

## 1. 当前能力与缺口

当前已有组件可以继续复用：

- `predicate.py`
  - 能发现 `AI.IF`、`AI.GENERATE`、`AI.CLASSIFY`、`AI.SCORE`。
  - 能区分 `per_row` 与 `pairwise` call site。
  - 能解析 alias、base relation、引用列、prompt、输出类别和源代码位置。
- `build_valset.py`
  - 支持 uniform、stratified、importance sampling。
  - 支持 inclusion probability、Horvitz–Thompson weight、SELECT/CERT split。
  - 能调用 Oracle 并缓存 label。
- `build_pairs.py`
  - 能构造 cross-table pair frame。
  - 能构造 self-join 的 image-image pair frame。
  - 已有 pair similarity 与 `--only-ids` 接口。
- `evaluate.py`
  - 能对单个 `trace.rows[id]` 做加权 boolean predicate 评分。
  - 能对最终 SemBench 结果做 scenario-specific evaluation。
- `orchestrator.mjs`
  - 已有 validation cache、refinement loop、telemetry 和 full-GT leakage guard。

当前阻塞点：

- `build_valset.py` 一次只能选择一个 call site。
- `select.json` 只表达一组 `id -> label`。
- solver trace 只表达一组 `trace.rows[id] -> prediction`。
- self-join frame 不能自动执行 SQL 中 AI predicate 之前的 CTE、普通 join 和 filter。
- 现有 self-join frame 默认是无序、无对角线的 `N choose 2`，但并不总是符合 SQL
  语义。
- 多个 predicate 独立抽样后不能直接组合成 query-level F1，因为不同 site 的样本
  通常不是同一批 candidate tuples。
- `AI.GENERATE`、`AI.SCORE`、group-wise top-k 不能用 boolean accuracy 统一处理。

## 2. 关键设计决定

### 2.1 Validation 的基本抽样单位是 query candidate

默认 estimand 定义为“整条查询在 AI operators 之前的 candidate population 上的
误差”，而不是某一个 predicate 的误差。

不同查询的 candidate unit：

| Query shape | Candidate unit |
|---|---|
| 单表 semantic filter/map | 一行 |
| 单个 cross-table/self join | 一对 rows |
| q10 三路组合 | 一个 `(image1, image2, image3)` tuple |
| q11 四路组合 | 一个 `(image1, image2, image3, image4)` tuple |
| q12 filter + generate | 一个 product row |
| q14 grouped argmin/top-1 | 一个 left-side product/group |

因此：

```text
validation_rows = ceil(candidate_population_size × val_rate)
```

这个定义保持之前的 join 约定：单个 pairwise join 的 candidate population 就是
完整 pair population。

新增 `--val-n` 作为固定 candidate 数量的替代接口。`--val-rate` 与 `--val-n`
互斥。系统不得静默 top-k、静默降低 rate 或静默改变 population。

### 2.2 先抽 query candidates，再展开 call-site inputs

不能为五个 predicates 各自独立抽 5%，然后假设这些样本能组成 query F1。正确流程：

```text
deterministic relational prefix
        ↓
query candidate population
        ↓
sample candidate tuples/groups
        ↓
展开每个 sampled candidate 所依赖的 site inputs
        ↓
按 (site fingerprint, input key) 去重
        ↓
Oracle labels + solver predictions
        ↓
执行 query composition
        ↓
operator metrics + query-level metrics
```

例如 q10 的一个 candidate tuple `(a,b,c)` 依赖：

```text
site0(a) ∧ site1(b) ∧ site2(c) ∧ site3(a,b) ∧ site4(b,c)
```

如果多个 sampled tuples 使用同一 row/pair，Oracle 只标注一次。

### 2.3 Sampling 与 operator evaluation 分离

一份 query sample 决定 query-level inclusion probability。各 call site 的展开 frame
用于复用 label 和产生 operator feedback，但不能把 site frame 自己的命中率冒充
query-level F1。

### 2.4 确定性前缀使用嵌入式关系执行，不引入新的持久化数据库

建议新增一个只在 planning/validation 阶段使用的 DuckDB in-memory connection：

```text
materialized CSV/Parquet → temporary DuckDB views → deterministic SQL prefix
```

它只负责普通 projection/filter/equi-join/CTE，不成为新的 physical storage，也不改变
solver 的执行后端。使用 SQLGlot 解析 BigQuery SQL，并在解析前把 AI calls 替换为有
类型的 placeholder。

这样比为每个 query 手写 pandas filter 更容易验证 SQL 语义，同时仍保持所有
validation artifacts 为普通 CSV/JSON。

### 2.5 Query-level 与 operator-level 指标必须同时保留

Refinement 的主排序信号使用 query-level metric。operator metrics 用于定位错误和
生成 feedback：

```text
query-level F1 下降
  ├── site0 recall 低
  ├── site3 false positives 多
  └── final composition/output lineage 错误
```

`--val-call-site N` 保留为明确的 operator-debug 模式；只有显式指定时才允许只验证
单个 predicate。

## 3. 新的中间表示与文件格式

### 3.1 QueryValidationPlan

新增 `validation_plan.py`，从 SQL 和 benchmark metadata 生成
`validation_plan.json`：

```json
{
  "version": 2,
  "benchmark": "ecomm",
  "query": "q10",
  "candidate": {
    "unit": "tuple",
    "aliases": ["images1", "images2", "images3"],
    "key_columns": ["images1.id", "images2.id", "images3.id"],
    "population_size": 216000,
    "ordered": true,
    "include_diagonal": true
  },
  "sites": [
    {
      "site_id": "s0",
      "source_index": 0,
      "kind": "if",
      "shape": "per_row",
      "input_aliases": ["images1"],
      "input_key": ["images1.id"],
      "fingerprint": "sha256:..."
    }
  ],
  "composition": {
    "kind": "boolean_expression",
    "expression": ["and", "s0", "s1", "s2", "s3", "s4"]
  }
}
```

Plan 必须保存：

- SQL hash 和 materialized input hashes。
- 每个 alias 的 base relation/CTE。
- 普通 join/filter。
- AI site 的 source index、kind、shape、prompt fingerprint 和输入列。
- candidate ordering、self-pair、outer-join 和 NULL 语义。
- projection/output key。
- group/order/limit 信息。
- 无法安全解析的 construct 及明确拒绝原因。

### 3.2 Query candidate sample

新增：

```text
_val/ecomm-q10/<design>/candidates.csv
```

推荐列：

```text
candidate_id,a0_id,a1_id,a2_id,pi,weight,stratum
```

`candidate_id` 内部使用无歧义的 versioned encoding，例如 length-prefixed JSON
tuple；最终 SemBench output ID 仍按原 SQL 投影为 `a-b-c`。不要继续假设 `-` 永远
不会出现在任意一侧 ID 中。

大 Cartesian product 不应先完整 materialize。uniform sampling 可以：

1. 在 `[0, N)` 上抽 mixed-radix integer indexes。
2. 将 index 解码为各 alias row index。
3. 只 materialize sampled candidates。

importance/stratified sampling 使用可计算的 factor score，并完整记录 proposal 与
`pi`。任何零 inclusion probability 的 pruning 都必须作为不同 estimand 明确命名。

### 3.3 Multi-site validation bundle

新增：

```text
validation_bundle.json
site_s0.json
site_s1.json
...
```

`validation_bundle.json`：

```json
{
  "version": 2,
  "query": "q10",
  "plan": "validation_plan.json",
  "candidates": "candidates.csv",
  "sites": {
    "s0": "site_s0.json",
    "s1": "site_s1.json"
  },
  "select_candidate_ids": ["..."],
  "cert_candidate_ids": [],
  "design": {
    "N": 216000,
    "n": 10800,
    "unit": "output_candidate"
  }
}
```

每个 `site_sN.json` 继续复用当前 `labels/weights/provenance` 思路，但增加：

- `site_id`、source index、kind、shape 和 fingerprint。
- `input_key -> label`。
- 哪些 query candidates 引用了该 input。
- Oracle cache key。
- boolean/categorical/free-form/numeric/ranking label type。

### 3.4 Trace v2

Direct solver 输出：

```json
{
  "trace_version": 2,
  "query": "q10",
  "sites": {
    "s0": {"rows": {"6100": "true"}},
    "s3": {"rows": {"6100|7935": "true"}}
  },
  "outputs": {
    "<candidate-id>": {
      "selected": true,
      "projected_id": "6100-7935-10579"
    }
  }
}
```

兼容策略：

- 单 site query 继续接受现有 `trace.rows`。
- v2 bundle 运行时必须提供 `trace.sites`。
- evaluator 对缺失 site/key 计为 missing prediction，不得跳过。
- `trace.outputs` 用于检查 relational composition 和 final projection；不包含
  expected label。

## 4. Part A：Deterministic Self-join Pair Frame

### 4.1 建立 normalized EComm product view

当前 `styles_details.csv` 将 nested structs 字符串化，`IMAGES.csv` 只保留
`id,filename`。扩展 `materialize.py`，额外生成：

```text
ecomm_products.csv
```

至少包括：

```text
id
filename
price
baseColour
colour1
colour2
brandName
productDisplayName
description
masterCategory
```

从原 Parquet 中直接 flatten nested fields，不要从 CSV 中用 `eval()` 还原 Python
字符串。`id` 与 image mapping 的 cardinality/uniqueness 必须在 materialization
阶段验证。

### 4.2 SQL AI-call masking

扩展 `predicate.py` 暴露每个 AI call 的 source span。新增 typed masking：

```text
AI.IF / AI.CLASSIFY → TRUE placeholder
AI.SCORE            → numeric placeholder
AI.GENERATE         → string/JSON placeholder
```

masking 只用于 SQL AST/relational-prefix 构造，Oracle prompt 仍来自原始 SQL。

### 4.3 生成 deterministic relation

新增 `frame_builder.py`：

1. 注册 normalized CSV/Parquet 为 DuckDB temporary views。
2. 执行 CTE 内 AI 之前的普通 predicates。
3. 执行 ordinary equi-joins。
4. 投影 AI call 需要的 columns 和稳定 row IDs。
5. 写出 `deterministic_rows.parquet` 或流式传递给 candidate sampler。

安全边界：

- 只执行 allowlist 中的 deterministic relational operators。
- 遇到 correlated subquery、window semantics、无法翻译的 nested expression 时
  fail closed。
- 不得把 AI predicate 替换成 `TRUE` 后直接执行整个 query，因为这可能改变
  outer-join、grouping 和 limit 语义。

### 4.4 从 SQL 推导 pair domain

pair domain 不能固定为 `N choose 2`：

| Query | SQL semantics | Pair population |
|---|---|---|
| q7 | ordered self-join，允许 `p1 = p2` | `N²` |
| q9 | ordered self-join，显式 `p1 != p2` | `N(N-1)` |
| symmetric predicate optimization | 可以只标一份 canonical pair | Oracle 去重，不改变输出 domain |

当前 sf250 materialization 的回归基线：

- q7：`price <= 500` 后 41 rows，candidate population 为 `41² = 1,681`。
- q9：当前 deterministic filter + mapped image 后 18 rows，ordered non-self
  population 为 `18 × 17 = 306`。

这些数字应由 frame builder 重新计算并写入 manifest，测试不应把它们当成跨 scale
固定常量。

扩展 `build_pairs.py`：

- `--ordered`
- `--include-diagonal`
- `--exclude-diagonal`
- text-text、image-image、text-image input columns
- canonical Oracle label reuse
- output-direction expansion

q7 ground truth 已证明 diagonal 和方向不可丢失：93 个输出中有 41 个 self-pairs，
其余正 pair 同时存在正反方向。q9 则是 6 个无序正 pair的 12 个方向输出。

### 4.5 Self-join acceptance criteria

- q7/q9 在 `--val-rate` 下不再被 planner skip。
- `design.N` 等于确定性前缀之后的真实 ordered pair population。
- `design.n = ceil(design.N × val_rate)`。
- `candidates.csv` 不含前缀 filter 已排除的 rows。
- q7 包含 diagonal；q9 不包含 diagonal。
- symmetric label reuse 只能减少 Oracle calls，不能减少 query candidate 数量。
- solver 缺少任意 sampled pair prediction 时计为错误。
- refinement 期间只暴露 SELECT labels；CERT/full GT 不进入 feedback。

## 5. Part B：Multi-call-site Validation 与 Query Composition

### 5.1 Call-site dependency graph

扩展 `predicate.py` 或新增 `validation_plan.py`，产生：

```text
alias relation graph
AI site dependency graph
final relational composition
```

每个 site 必须知道：

- 输入 aliases 和 key。
- prompt 中列的顺序。
- label type。
- 是否与另一个 site 语义等价、可以共享 label。
- 是否影响 row selection、join edge、projection、group ordering 或 aggregation。

site fingerprint 至少包含：

```text
operator kind
normalized prompt
choices/output schema
ordered input roles
model/protocol version
```

不能仅根据 prompt 文本去重：方向敏感的输入角色不能交换。

### 5.2 Oracle label expansion 与去重

从 sampled query candidates 展开 required site inputs：

```text
candidate c1 → s0(a), s1(b), s2(c), s3(a,b), s4(b,c)
candidate c2 → s0(a), s1(d), s2(c), s3(a,d), s4(d,c)
```

得到唯一 `(site fingerprint, input key)` 集合后再调用 Oracle。cache 从当前
`query -> labels.json` 升级为：

```text
oracle model + protocol + site fingerprint + normalized input key
```

这样 q10/q11 中重复的 brand/color pair predicate 能安全复用，跨 iteration 和提高
validation rate 时也只补标新增 inputs。

### 5.3 类型化 site evaluator

| Site kind | Operator metric |
|---|---|
| `AI.IF` | weighted precision/recall/F1/accuracy |
| `AI.CLASSIFY` | weighted accuracy、macro/micro F1、confusion matrix |
| `AI.GENERATE` | exact/normalized match；有 schema 时做 parse/schema/field-level score |
| `AI.SCORE` | Spearman/Kendall、pairwise order accuracy、top-k agreement |

每种 evaluator 必须输出统一 envelope：

```json
{
  "site_id": "s0",
  "kind": "if",
  "n": 60,
  "metrics": {},
  "mistakes": []
}
```

### 5.4 Query composition evaluator

新增 `compose_validation.py`：

1. 读取 plan、candidate sample、site Oracle labels 和 solver site predictions。
2. 分别执行 expected composition 与 predicted composition。
3. 保留 candidate-to-output lineage。
4. 使用 candidate sampling weights 计算 query-level confusion matrix。
5. 生成有限条、带 site attribution 的 feedback examples。

Boolean relational query：

```text
expected_selected = compose(expected site labels)
predicted_selected = compose(predicted site predictions)
```

输出：

```text
weighted TP/FP/FN/TN
precision/recall/F1
estimated positive population
site-attributed false positives/false negatives
```

SQL NULL/outer join 语义必须显式实现，不能用 Python truthiness 代替。

### 5.5 q10/q11

q10 deterministic prefix 当前约 60 image-products：

```text
candidate population = 60³ = 216,000
5% candidate sample = 10,800 tuples
```

但 Oracle 调用按唯一 site inputs 去重，因此不等于 `10,800 × 5`。

q11 在 sf250 上的理论 root population 是：

```text
250⁴ = 3,906,250,000 tuples
5% = 195,312,500 sampled tuples
```

这个 rate 虽然定义清楚，但不适合直接物化。实现必须：

- mixed-radix sampling，不生成完整 Cartesian table。
- chunked/streaming composition。
- 在执行 Oracle 前打印 root N、sample n、unique site inputs 和预计调用数。
- 新增 `--max-val-calls`/`--max-val-candidates` guard；超预算时 fail closed，并建议
  显式使用 `--val-n`，不能静默降采样。

候选 sample 极稀疏时使用合法的 importance proposal，并通过 `pi/weight` 回推
population metric。proposal 必须具有 full support。

### 5.6 q12：Filter + Generate

candidate unit 是 product row：

```text
expected output =
  if expected_filter(row):
      expected_generate(row)
  else:
      no output
```

需要：

- 为 generation Oracle 明确 JSON schema。
- 规范化 JSON key order、whitespace、case 的规则要与 SemBench evaluator 一致。
- trace 保存 `input row id -> generated output` lineage。
- 同时报告 filter F1、generation exact/schema/field score 和 final retrieval F1。

不能只验证 q12 的 `AI.IF`，否则无法发现 generator 输出错误。

### 5.7 q14：Filter + Join + Score + Group-wise Top-1

q14 的 candidate unit 是经过 `price < 130` 后的 left product group。对于每个被抽到的
group，需要：

1. 标注 image 是否为 white socks。
2. 标注/预测 `(style, image)` 是否匹配。
3. 对 surviving candidates 获取 expected/predicted score 或 ordering。
4. 在组内执行 SQL 的 order direction 与 `LIMIT 1`。
5. 比较 expected top-1 与 predicted top-1。

因为 group top-1 依赖同一组的其他 candidates，不能随机抽独立 pairs 后直接声称
top-1 accuracy。采用 cluster sampling：

```text
sample left groups
→ 对被抽中的 group 展开全部 right candidates
→ 组内完整 composition
```

sf250 当前 `price < 130` 后只有约 6 个 mapped groups；`--val-rate 0.05` 只抽到 1
个 group，通常不足以稳定 refinement。系统应给出诊断并建议 `--val-n 6`，而不是
静默增加样本。

## 6. Orchestrator 与 Refinement 改造

### 6.1 CLI

新增：

```text
--val-n N
--val-unit output-candidate
--max-val-calls N
--max-val-candidates N
--val-plan-only
```

保留：

```text
--val-call-site N
```

其语义明确为 operator-debug，不产生 query-level score。

### 6.2 Planning preflight

在任何 Oracle call 前打印：

```text
deterministic row counts
candidate unit/domain
population N
sample n
各 site 展开后的 unique inputs
预计 Oracle calls/cache hits
预计 frame bytes
```

`--val-plan-only` 只生成 plan/cardinality/cost report，不调用 agent/Oracle。

### 6.3 Refinement signal

`refineLoop` 主分数改为：

```text
query_metrics.f1
```

feedback 依次包含：

1. query-level FP/FN candidates。
2. 导致该错误的 site predictions。
3. 每个 site 的聚合 P/R/F1 或相应 typed metric。
4. runtime/trace contract 错误。

仍然严格运行用户请求的全部 iterations；validation F1=1 不提前停止。

### 6.4 Telemetry

`telemetry.json` 增加：

```json
{
  "validation": {
    "unit": "output_candidate",
    "population": 216000,
    "sample_n": 10800,
    "expanded_site_inputs": 3780,
    "oracle_calls": 3700,
    "oracle_cache_hits": 80,
    "planning_ms": 0,
    "frame_build_ms": 0,
    "sampling_ms": 0,
    "oracle_ms": 0,
    "composition_ms": 0
  },
  "refine": {
    "f1_history": [],
    "site_metric_history": {}
  }
}
```

`results.csv` 保留 query-level `val_f1_iter_N`，另以 compact JSON column 保存
site history，避免为不同 query 的不同 site 数量创建不稳定列集合。

## 7. 分阶段代码实施顺序

### Phase 0：Contracts 与 fixtures

计划修改/新增：

- `predicate.py`
- `validation_plan.py`
- `tests/fixtures/ecomm/*.sql`
- `tests/test_validation_plan.py`

交付：

- source spans、typed sites、candidate-domain plan。
- q7/q9/q10/q11/q12/q14 golden plan JSON。
- 对无法支持的 SQL fail closed。

### Phase 1：Normalized materialization 与 deterministic frame

计划修改/新增：

- `materialize.py`
- `frame_builder.py`
- `tests/test_frame_builder.py`

交付：

- `ecomm_products.csv`。
- CTE/equi-join/filter 执行。
- q7/q9 deterministic row IDs 和 cardinality。
- input hash/cache invalidation。

### Phase 2：Self-join end-to-end

计划修改：

- `build_pairs.py`
- `build_valset.py`
- `oracle_label.py`
- `orchestrator.mjs`
- solver prompts
- pair validation tests

交付：

- q7 ordered+diagonal。
- q9 ordered+non-diagonal。
- full pair population × validation rate。
- symmetric Oracle label reuse。
- query-level weighted F1。

这是第一个可独立合并、可运行 benchmark 的 milestone。

### Phase 3：Multi-site boolean composition

计划新增/修改：

- `build_validation_bundle.py`
- `compose_validation.py`
- `evaluate.py`
- `orchestrator.mjs`
- direct solver trace contract/prompts

交付：

- q10 与 q11 plan/sample/site expansion。
- per-site metrics。
- tuple-level query F1。
- mixed-radix sampling、streaming 和 budget preflight。

### Phase 4：Typed operators

交付：

- q12 `IF + GENERATE`。
- q14 `IF + pairwise IF + SCORE + group top-1`。
- JSON/schema/ranking evaluators。
- cluster sampling。

### Phase 5：Telemetry、兼容和文档

交付：

- timing/cost/site metric history。
- results sync。
- v1 single-site validation regression。
- CLI/USAGE 文档和 EComm reproduction commands。

## 8. 测试计划

### 8.1 Unit tests

- SQL scanner 的 source span、prompt、input role 和 fingerprint。
- q7/q9 ordered/diagonal domain 推导。
- mixed-radix tuple index round-trip。
- sample size严格等于 `ceil(N × rate)`。
- unequal-probability sample 的 `pi/weight` 正确。
- repeated site inputs 去重但 candidate 数量不变。
- boolean three-valued composition。
- generation normalization/schema evaluation。
- ranking/top-1 group evaluation。
- missing trace keys 计错。

### 8.2 Integration tests

使用小型 synthetic relations：

- 两表 join。
- ordered self-join with diagonal。
- ordered self-join without diagonal。
- 三路 `filter ∧ filter ∧ pair predicate`。
- filter + generate。
- grouped filter + join + score + top-1。

每个测试同时构造：

- full exhaustive expected result。
- sampled/weighted estimate。
- intentionally broken solver trace。

检查 sampled evaluator 的方向正确，并在重复 seeds 的统计测试中接近 exhaustive
metric。

### 8.3 EComm sf250 smoke tests

- `q7 --val-rate 0.05`
- `q9 --val-rate 0.05`
- `q10 --val-rate 0.05`
- `q11 --val-plan-only --val-rate 0.05`
- `q11 --val-n <bounded-value>`
- `q12 --val-rate 0.05`
- `q14 --val-n 6`

检查：

- 不再出现当前 six-query planner skips。
- Oracle call count与 preflight estimate 一致。
- 每轮产生 query F1 和 site metrics。
- 五轮完整执行。
- full GT 只在最终 benchmark scoring 使用，不进入 refinement feedback。

## 9. 风险与防护

### SQL 解析错误

风险：BigQuery nested fields、`EXTERNAL_OBJECT_TRANSFORM`、三引号和 named arguments。

防护：原 SQL call span masking + SQLGlot AST + allowlist；无法证明安全时拒绝，不使用
regex 直接重写整个 SQL。

### Cartesian explosion

风险：q11 root domain 达十亿级。

防护：mixed-radix sample、chunked composition、label dedupe、plan-only cost、显式 budget
guard 和 `--val-n`。

### 稀有 positives

风险：uniform sample 没有 positives，F1 无法指导 refinement。

防护：full-support importance sampling、certainty anchor + residual random strata、HT
weights；仍然拒绝 single-class SELECT，除非用户明确允许。

### Label reuse 错误

风险：看似相同的 prompt 实际输入角色或 output schema 不同。

防护：fingerprint 包含 kind、normalized prompt、schema、choices 和 ordered roles。

### Query-level metric 偏差

风险：独立 site samples、top-k pruning 或不完整 group candidates 改变 estimand。

防护：root candidate sampling；ranking 使用完整 sampled groups；所有 `pi`、frame
restriction 与 population hash 写入 provenance。

## 10. Definition of Done

只有满足以下条件才算两个部分真正实现：

- q7/q9 自动执行 deterministic prefix 并建立符合 SQL 方向/对角线语义的 self-join
  validation frame。
- q10/q11/q12/q14 自动建立 multi-site bundle，不要求用户逐个指定
  `--val-call-site`。
- `--val-rate` 始终基于明确记录的 root candidate population。
- query-level metric 来自同一批 sampled candidates 的完整 composition。
- 每个 site 同时有独立 metrics 与 actionable feedback。
- Oracle label cache 可按 semantic site/input 安全复用。
- 所有高成本运行先输出 cardinality/call-cost preflight。
- refinement 不读取 full benchmark ground truth。
- validation F1=1 时仍完成全部指定 iterations。
- telemetry/results.csv 保存 query-level history、site-level history和各阶段时间。
- 现有单 predicate MMQA/Cars/EComm queries 与 v1 val files 全部 regression tests
  通过。

## 11. 推荐的第一批实现边界

第一批代码建议只完成 Phase 0–2，也就是先让 q7/q9 正确运行：

```text
SQL plan
→ deterministic EComm product rows
→ ordered self-join domain
→ sampled pair bundle
→ Oracle labels
→ query-level F1/refinement
```

原因是这条路径会先建立 multi-site 后续也需要的 plan、frame 和 candidate-domain
基础，同时 pair population 较小，容易与 exhaustive result 对照验证。确认 q7/q9
正确后，再把相同 candidate/bundle contract 扩展到 q10/q11，而不是直接在
orchestrator 中为六个 query 添加特例。
