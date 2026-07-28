# SemDB 与 SemBench 指标对照

本文档以以下 SemBench 实现为唯一事实来源：

- `SemBench/src/evaluator/generic_evaluator.py`
- `SemBench/src/scenario/<workload>/evaluation/evaluate.py`
- `SemBench/files/ecomm/queries/q*.toml` 中的 `accuracy_metric`

SemDB 的实现位于 `scenario_metrics.py`（movie、animals、cars、medical、
ecomm）和 `evaluate.py`（mmqa）。`scenario_metrics_validate.py` 会直接调用
SemBench evaluator 做数值 parity 测试。

## 指标族和输出字段

| SemBench 返回类型 | SemBench 原始字段 | SemDB 兼容短字段 | 优化方向 |
|---|---|---|---|
| `QueryMetricRetrieval` | `precision`, `recall`, `f1_score` | `precision`, `recall`, `f1` | 越大越好 |
| `QueryMetricAggregation` | `relative_error`, `absolute_error`, `mean_absolute_percentage_error` | `relative_error`, `absolute_error`, `mape` | 越小越好 |
| `QueryMetricRank` | `spearman_correlation`, `kendall_tau` | `spearman`, `kendall` | 越大越好 |
| `SingleAccuracyScoreWithRetrievalDetails` | `accuracy`, `metric_type=f1-score`, `precision`, `recall`, `f1_score` | `accuracy`, `precision`, `recall`, `f1` | 越大越好 |
| `SingleAccuracyScore` | `accuracy`, `metric_type=adjusted-rand-index` | `accuracy`, `ari`, `adjusted_rand_index` | 越大越好 |

SemDB 同时写出 SemBench 原始字段和已有短字段。`metric_family` 保存上表中的
SemBench 返回类型；`variant`/`metric_variant` 只说明同一指标族内部的匹配方式，
不再把 animals 的特殊单行匹配误称为另一个 `top1` 指标。

聚合查询同时返回三项误差。迭代选择使用 `relative_error`；排序查询同时返回
Spearman 和 Kendall，迭代选择使用 `spearman_correlation`。这是 SemDB 为“选择
最佳 iteration”指定的主指标，SemBench 本身没有在这两个字段中另行指定主次。

## MMQA

MMQA 的所有查询均返回 `QueryMetricRetrieval`：
`precision`、`recall`、`f1_score`。不同查询只在结果规范化和匹配单位上不同。

| Query | 指标 | 匹配单位/特殊规则 |
|---|---|---|
| q1 | Precision / Recall / F1 | director 字符串 membership |
| q2a | Precision / Recall / F1 | `(ID, image_id)` tuple set |
| q2b | Precision / Recall / F1 | `(ID, image_id[, color])` tuple set |
| q3a | Precision / Recall / F1 | title membership |
| q3b | Precision / Recall / F1 | title membership |
| q3c | Precision / Recall / F1 | title membership |
| q3d | Precision / Recall / F1 | title membership |
| q3e | Precision / Recall / F1 | title membership |
| q3f | Precision / Recall / F1 | title membership |
| q3g | Precision / Recall / F1 | title membership |
| q4 | Precision / Recall / F1 | `(genre, movie)` membership |
| q5 | Precision / Recall / F1 | actor/output membership |
| q6a | Precision / Recall / F1 | airline membership |
| q6b | Precision / Recall / F1 | airline membership |
| q6c | Precision / Recall / F1 | airline membership |
| q7 | Precision / Recall / F1 | `(airline, image_id)` tuple set |

## Movie

| Query | SemBench 返回类型 | 返回指标 | 计算变体 |
|---|---|---|---|
| Q1 | `QueryMetricRetrieval` | Precision / Recall / F1 | 第一列集合匹配，预测取前 5 行 |
| Q2 | `QueryMetricRetrieval` | Precision / Recall / F1 | 第一列集合匹配，预测取前 5 行 |
| Q3 | `QueryMetricAggregation` | Relative error / Absolute error / MAPE | 单值聚合 |
| Q4 | `QueryMetricAggregation` | Relative error / Absolute error / MAPE | 单值聚合 |
| Q5 | `QueryMetricRetrieval` | Precision / Recall / F1 | review pair 匹配，预测取前 10 行 |
| Q6 | `QueryMetricRetrieval` | Precision / Recall / F1 | review pair 匹配，预测取前 10 行 |
| Q7 | `QueryMetricRetrieval` | Precision / Recall / F1 | review pair 匹配，无 LIMIT |
| Q8 | `QueryMetricAggregation` | Relative error / Absolute error / MAPE | 按 sentiment 比较 count；相对误差取类别平均 |
| Q9 | `QueryMetricRank` | Spearman correlation / Kendall tau | 共同 ID 上比较第二列 score |
| Q10 | `QueryMetricRank` | Spearman correlation / Kendall tau | 共同 ID 上比较第二列 score |

## Animals

| Query | SemBench 返回类型 | 返回指标 | 计算变体 |
|---|---|---|---|
| Q1 | `QueryMetricAggregation` | Relative error / Absolute error / MAPE | 单值聚合 |
| Q2 | `QueryMetricAggregation` | Relative error / Absolute error / MAPE | 单值聚合；audio-only |
| Q3 | `QueryMetricRetrieval` | Precision / Recall / F1 | 必须恰好输出一个 city，且可命中任一并列 GT city |
| Q4 | `QueryMetricRetrieval` | Precision / Recall / F1 | 与 Q3 相同；audio-only |
| Q5 | `QueryMetricRetrieval` | Precision / Recall / F1 | 公共列上的逐行 greedy matching |
| Q6 | `QueryMetricRetrieval` | Precision / Recall / F1 | 公共列上的逐行 greedy matching |
| Q7 | `QueryMetricRetrieval` | Precision / Recall / F1 | 公共列上的逐行 greedy matching |
| Q8 | `QueryMetricRetrieval` | Precision / Recall / F1 | 公共列上的逐行 greedy matching |
| Q9 | `QueryMetricRetrieval` | Precision / Recall / F1 | 公共列上的逐行 greedy matching |
| Q10 | `QueryMetricRetrieval` | Precision / Recall / F1 | 必须恰好输出一个 `(city, station)` 并命中 GT |

注意：Q3、Q4、Q10 的分数通常为 0 或 1，但 SemBench 返回的仍然是
`QueryMetricRetrieval`，不是 ARI，也不是独立的 top-1 metric。

## Cars

| Query | SemBench 返回类型 | 返回指标 | 计算变体 |
|---|---|---|---|
| Q1 | `QueryMetricRetrieval` | Precision / Recall / F1 | `car_id` set |
| Q2 | `QueryMetricRetrieval` | Precision / Recall / F1 | 去重后的 `car_id` set；audio-only |
| Q3 | `QueryMetricRetrieval` | Precision / Recall / F1 | `vin` set；围绕 `LIMIT 10` 对 GT 做 `random_state=42` 平衡采样 |
| Q4 | `QueryMetricAggregation` | Relative error / Absolute error / MAPE | 单值聚合 |
| Q5 | `QueryMetricAggregation` | Relative error / Absolute error / MAPE | 单值聚合 |
| Q6 | `QueryMetricRetrieval` | Precision / Recall / F1 | `car_id` set |
| Q7 | `QueryMetricRetrieval` | Precision / Recall / F1 | `car_id` set |
| Q8 | `QueryMetricRetrieval` | Precision / Recall / F1 | `car_id` set；围绕 `LIMIT 100` 对 GT 做 `random_state=42` 平衡采样 |
| Q9 | `QueryMetricRetrieval` | Precision / Recall / F1 | `car_id` set |
| Q10 | `QueryMetricRetrieval` | macro Precision / macro Recall / macro F1 | `problem_category` 多分类，`average=macro` |

## Medical

| Query | SemBench 返回类型 | 返回指标 | 计算变体 |
|---|---|---|---|
| Q1 | `QueryMetricRetrieval` | Precision / Recall / F1 | `patient_id` set |
| Q2 | `QueryMetricRetrieval` | Precision / Recall / F1 | 去重后的 `patient_id` set；audio-only |
| Q3 | `QueryMetricRetrieval` | Precision / Recall / F1 | `patient_id` set；围绕 `LIMIT 5` 对 GT 做 `random_state=42` 平衡采样 |
| Q4 | `QueryMetricAggregation` | Relative error / Absolute error / MAPE | 单值聚合 |
| Q5 | `QueryMetricAggregation` | Relative error / Absolute error / MAPE | 单值聚合 |
| Q6 | `QueryMetricRetrieval` | Precision / Recall / F1 | `patient_id` set |
| Q7 | `QueryMetricRetrieval` | Precision / Recall / F1 | `patient_id` set |
| Q8 | `QueryMetricRetrieval` | Precision / Recall / F1 | `patient_id` set；围绕 `LIMIT 100` 对 GT 做 `random_state=42` 平衡采样 |
| Q9 | `QueryMetricRetrieval` | Precision / Recall / F1 | `patient_id` set |
| Q10 | `QueryMetricRetrieval` | macro Precision / macro Recall / macro F1 | `text_diagnosis` 多分类，`average=macro` |

`files/medical/query/bigquery/Q11.sql` 存在，但当前 SemBench
`MedicalEvaluator` 只 dispatch Q1–Q10，因此 Q11 没有可对齐的权威指标，SemDB
也不应为它静默使用 F1。

## Ecomm

Ecomm 不按 SQL 形状推断指标，而是严格读取每个 TOML 的
`[definition].accuracy_metric`。

| Query | TOML `accuracy_metric` | SemBench 返回类型 | 返回指标 |
|---|---|---|---|
| Q1 | `f1-score` | `SingleAccuracyScoreWithRetrievalDetails` | accuracy (= F1), Precision, Recall, F1 |
| Q2 | `f1-score` | `SingleAccuracyScoreWithRetrievalDetails` | accuracy (= F1), Precision, Recall, F1 |
| Q3 | `adjusted-rand-index` | `SingleAccuracyScore` | accuracy (= ARI) |
| Q4 | `adjusted-rand-index` | `SingleAccuracyScore` | accuracy (= ARI) |
| Q5 | `adjusted-rand-index` | `SingleAccuracyScore` | accuracy (= ARI) |
| Q6 | `adjusted-rand-index` | `SingleAccuracyScore` | accuracy (= ARI) |
| Q7 | `f1-score` | `SingleAccuracyScoreWithRetrievalDetails` | accuracy (= F1), Precision, Recall, F1 |
| Q8 | `f1-score` | `SingleAccuracyScoreWithRetrievalDetails` | accuracy (= F1), Precision, Recall, F1 |
| Q9 | `f1-score` | `SingleAccuracyScoreWithRetrievalDetails` | accuracy (= F1), Precision, Recall, F1 |
| Q10 | `f1-score` | `SingleAccuracyScoreWithRetrievalDetails` | accuracy (= F1), Precision, Recall, F1 |
| Q11 | `f1-score` | `SingleAccuracyScoreWithRetrievalDetails` | accuracy (= F1), Precision, Recall, F1 |
| Q12 | `f1-score` | `SingleAccuracyScoreWithRetrievalDetails` | accuracy (= F1), Precision, Recall, F1 |
| Q13 | `f1-score` | `SingleAccuracyScoreWithRetrievalDetails` | accuracy (= F1), Precision, Recall, F1 |
| Q14 | `f1-score` | `SingleAccuracyScoreWithRetrievalDetails` | accuracy (= F1), Precision, Recall, F1 |

Q15 和 Q17 是 `.toml.draft`，不属于当前正式 query 集合。

## Iteration feedback 与最终评分

每轮 codegen iteration 没有访问 SemBench held-out full-query ground truth；它使用
validation sample 上的 oracle 标签。因此 feedback 明确分成两层：

1. `VALIDATION OBJECTIVE`：按 query 指标族重建的可优化目标。Ecomm Q3–Q6
   返回 ARI；Movie Q9–Q10 返回 Spearman 并附 Kendall；聚合 query 返回
   relative/absolute error 和 MAPE；Cars/Medical Q10 返回 macro P/R/F1；
   Animals Q3/Q4/Q10 返回 SemBench 定义的 retrieval F1（特殊单行匹配）。
2. `PER-ROW INFERENCE FIDELITY`：validation sample 上的 operator 级
   accuracy、Precision、Recall、F1 和错误样本，作为诊断信息，不冒充最终 query
   metric。

Movie Q8、Cars Q5、Medical Q5 的最终聚合依赖多个 semantic call site，而当前
trace 只标注一个 call site。此时 feedback 返回
`query_metric_unavailable`，不会用错误的 predicate F1 代替最终聚合指标。

最终 full-corpus 执行若有 SemBench ground truth，则 feedback 和 telemetry 写入
完整的 `query_metrics`，而不仅是用于 iteration 选择的单一 objective。

## 当前支持范围

SemDB `benchmarks.mjs` 当前支持 mmqa、movie、cars、medical、animals、ecomm。
SemBench 的 lro 是 retrieval/entity-matching workload，不在当前 compiled
`AI.IF`/`AI.GENERATE` 执行路径内；且本地 checkout 仅有其 evaluator `.pyc`，
没有可审计的 evaluator 源码。因此本次没有把 lro 指标接入 SemDB。历史
`files/lro/metrics/*.json` 显示 Q113、Q121、Q202、Q305 报告
Precision/Recall/F1，但这不能替代源码级 parity 验证。
