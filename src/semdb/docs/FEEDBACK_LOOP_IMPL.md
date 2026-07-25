# 多维执行反馈驱动的迭代代码生成 — 代码修改计划

分支：`claude/valset-guarantee-feedback`
配套：`VALSET_GUARANTEE_IMPL.md`（采样验证集与统计保证）、`PROXY_GUARANTEE_PLAN.md`（理论定位）

**本文范围**：把「生成代码在验证集上的 quality / latency / cost + 编译错误 + 运行日志」采集出来，
结构化成 scorecard，再渲染成反馈喂回 codegen agent。**不含**统计保证部分（区间、认证、
配对检验）——那部分依赖本文产出的 scorecard 作为输入，排在其后。

---

## 0. 当前链路的四个具体障碍（读代码所得）

| # | 位置 | 问题 |
|---|---|---|
| B1 | `orchestrator.mjs:1121` `runSolver` | `stdio: ["inherit","inherit","pipe"]` —— **stdout 被直接扔给终端，程序自己打的日志一条都拿不到**；只有 stderr 尾部 40 行进反馈 |
| B2 | `orchestrator.mjs:1162` `regenSolver` | feedback 是 `sql + "\n\n" + feedback` **塞进 `{{query_sql}}` 模板变量**传的 —— prompt 里那段本该是 SQL 的代码块混进了反馈文本，agent 看到的是被污染的查询 |
| B3 | `makeRecorder` / `phases[]` | 成本、耗时、token 记在 **run 级**数组里，没有按 iteration 切分 → 无法回答"这一轮花了多少" |
| B4 | `runSolver` 只做 `validateOfflineVadarFile` | 那是**离线合规正则检查**，不是编译检查。语法错、未定义名都要等真跑一遍才炸，浪费一次执行，且只能拿到 stderr 尾巴，**没有行号+源码上下文** |

另外 `refineLoop` 传给 `renderFeedback` 的只有 `{status, f1, metrics, diff, stderrTail, history}`，
没有承载 cost/latency/branch 的位置——outcome 对象需要扩。

---

## 1. 反馈内容的设计依据（参考其他 codegen agent）

调研结论，按对本项目的可迁移性排序：

| 来源 | 有效机制 | 我们怎么用 |
|---|---|---|
| **LDB** (LLM Debugger) | 把程序切成 basic block，回传**中间变量值**，而不只是最终输出 diff。中间状态比最终 diff 信息量高一个量级 | → §4 的 **branch 归因**：solver 在每个判定分支打短标签，反馈按分支聚合准确率。这是"proxy 是程序"才有的红利 |
| **Self-Debugging** (Chen et al.) | 让模型**先解释代码再改**（rubber-duck），比直接喂 error 显著更好 | → 反馈末尾强制 `## YOUR TASK` 要求先写一句失败假设再动手 |
| **AlphaCodium** | 迭代时**锚定已通过的用例**，防止修 A 弄坏 B | → 反馈里同时给出**当前正确的分支/行数**，不只给错的 |
| **SWE-agent (ACI)** | 观察必须**短、结构化、确定性**；截断的 traceback + 行号上下文 >> 原始 dump | → 所有段落有硬行数预算，总长 ≤ 120 行；traceback 只留最后一帧 + 源码 ±3 行 |
| **Reflexion** | 跨轮次的语言化反思记忆 | → `## HISTORY` 扩成"每轮做了什么改动 → 结果如何"，而不是现在的纯 F1 数列 |
| **EffiBench / ECCO / SWE-Perf** | 效率导向的 codegen 需要**显式回传 runtime/memory profile**，否则模型完全不优化效率 | → cost/latency 必须进 `## OBJECTIVE` 和 scorecard，且给出预算对比。**只报数字不给目标，模型不会优化它** |

**一条硬结论**：多目标必须在 prompt 里写成**约束式目标**（"maximize quality subject to
cost ≤ X"），而不是把三个数并排列出来让模型自己权衡——后者的实测行为是模型只盯第一个数。

---

## 2. 改动文件清单

```
src/semdb/
  orchestrator.mjs                     改  B1-B4 + scorecard 组装 + renderFeedback v2
  preflight.py                         新  编译前置闸（syntax / offline / static-name）
  evaluate.py                          改  --score-inference 增 branch 聚合与 error 统计
  agents/vadar-solver/prompt-text.md   改  trace 契约增 branch/errors 字段
  agents/vadar-solver/user-prompt-text.md 改  新增 {{feedback}} 槽位（不再污染 query_sql）
  agents/vadar-solver/user-prompt.md   改  同上（image 版）
  agents/code-generator/user-prompt.md 改  同上（compiled 模式）
  semdb.config.mjs                     改  feedback / budget 默认值
  tests/test_preflight.py              新
  tests/test_scorecard.mjs             新
  tests/test_feedback_v2.mjs           新
  tests/test_evaluate_branch.py        新
```

> **与 `VALSET_GUARANTEE_IMPL.md` §4 的一处偏离**：原稿计划新建 `scorecard.py`。改为
> **orchestrator 里用 JS 组装 scorecard**，因为 compile / run / cost / latency 四类事实全在
> JS 侧，只有 quality+branch 需要 Python（trace × val 的 join，已在 `evaluate.py` 里）。
> 新开一个 Python 文件会把同一张卡的组装逻辑劈成两半。

---

## 3. PR-1：编译前置闸 + 抽样设计 ✅ 已实现

> **状态**：已落地在 `claude/valset-guarantee-feedback`。
> 新增 `preflight.py` / `sampling.py` / `build_valset.py`，
> 测试 `test_preflight.py`(13) / `test_sampling.py`(31) / `test_build_valset.py`(20) /
> `test_preflight_gate.mjs`(10) 全绿，既有 `.mjs` 回归全过。
> 实现过程中相对本节原稿的两处修正记在 §3.3。

### 3.1 `src/semdb/preflight.py`

```
python3 preflight.py <solve_file> [--helpers <path>] --out <preflight.json>
```

三级检查，**任一级失败即停并返回结构化定位**：

| 级 | 手段 | 抓到的错 |
|---|---|---|
| 1 syntax | `py_compile.compile(path, doraise=True)` | `SyntaxError` / `IndentationError` — 有精确 lineno/offset |
| 2 static-name | `pyflakes` API（可选依赖，缺失则跳过并在 json 里标 `skipped`） | 未定义名、未使用导入、shadow —— **不执行代码**就能抓到大部分 `NameError` |
| 3 offline | 复用现有 `offlineVadarViolations` 的规则（Python 侧重实现或由 JS 先跑） | 网络/endpoint 违规 |

输出：
```json
{ "ok": false, "stage": "syntax", "error_class": "SyntaxError",
  "file": "solve_q3a.py", "line": 87, "col": 12,
  "message": "invalid syntax",
  "context": ["85|     for row in rows:", "86|         v = classify(row)",
              "87|     if v == 'comedy'", "88|         out.append(row)"],
  "pyflakes": [{"line": 42, "code": "F821", "message": "undefined name 'genre_map'"}] }
```

> **为什么不做 import smoke**：`solve_<q>.py` 的模块级代码不保证被 `if __name__` 守住，
> import 一次可能真的跑起来（写文件、读数据）。静态的 pyflakes 覆盖了同一类错误
> （`F821 undefined name` ≈ `NameError`、`F401` ≈ 死导入）且**零副作用**，是更好的取舍。
> 原稿 `VALSET_GUARANTEE_IMPL.md` §4 写的 "import smoke" 据此取消。

### 3.2 orchestrator 接线

`runSolver` 开头，`validateOfflineVadarFile` 之后、`spawnSync` 之前：

```js
const pf = runPreflight(iterCode, helpersPath, resolve(iterDir, "preflight.json"));
if (!pf.ok) return { status: "crash", stage: "compile", preflight: pf, stderr: pf.message, execMs: 0 };
```

**收益**：编译错不再消耗一次全量执行（val 模式下 120 行 × LLM 调用），且反馈能给到行号+源码上下文。

实际接线（`orchestrator.mjs`）：

- `runPreflight(paths, outPath)` —— 调 `preflight.py`，写 `iter_N/preflight.json`，
  返回 `{ok, stage, text, report}`。**永不抛异常**：`preflight.py` 缺失或崩溃时返回
  `stage:"skipped"` 并放行。一个静态检查器挂掉不该阻断代码生成。
- 挂在 `runSolver` 和 `runCompiled` 里，位于既有 `validateOfflineVadarFile`（合规）之后、
  `spawnSync`（执行）之前。
- `run.stage="compile"` 经 `scoreIter` 透传进 `renderFeedback`，让 fix-first 块说
  "DID NOT COMPILE / was NOT executed" 而不是 "crashed"——**没跑过的程序不能说它崩了**，
  否则 agent 会去找一个不存在的运行时原因。

### 3.2b 首次端到端运行：拿到了一个真实的 optimism gap

`mmqa/q3a`，importance 设计，n=60，`--max-iterations 3`：

| | 循环看到的（SELECT，60 行） | 全语料真值（200 行） |
|---|---|---|
| iter_0 | accuracy 0.9833 (59/60) | — |
| iter_1 | **accuracy 1.0000 (60/60)** | — |
| 停止原因 | `Perfect F1 reached` | — |
| 最终 | — | **F1 0.88** (P=0.917 R=0.846, tp=11 fp=1 **fn=2**) |

**这正是 §5 要修的判据缺陷的实物证据**：循环在 n=60 上看到点估计 1.0 就宣布完美并提前停机，
而程序在语料上漏掉了 2 个正例。

- 60/60 全对时，单侧 95% Clopper–Pearson 下界是 **0.9513**，不是 1.0。
  `shouldContinueSemdb` 若按 `quality.lcb >= target` 判定，target=0.98 时会继续迭代而不是停。
- 两个数字口径不同（SELECT 是 60 行的 per-row accuracy，全语料是 13 条 gold 上的集合 F1），
  不能直接相减当 optimism gap 用。**要画那张图必须让两侧同口径**——这是 `certify.py` 的
  前置条件，记在这里以免以后拿这两个数硬凑。
- 本次用的是 pps 样本，未加权 accuracy 本就不是语料 accuracy 的估计（见 §3.4 警告框），
  所以这里的 1.0 连"SELECT 上的语料准确率估计"都算不上，只是"这 60 行全对"。

### 3.3 实现中对本节原稿的两处修正

1. **取消 import smoke**（原稿 §3.1 第 3 级）。理由已写在 `preflight.py` 模块 docstring：
   import 一次会执行模块级代码。改用 pyflakes 的 `UndefinedName` 静态覆盖同一错误类。
2. **离线合规检查不进 preflight.py**。那份正则表在 `orchestrator.mjs` 的
   `VADAR_RUNTIME_FORBIDDEN`，Python 侧重写一份必然漂移。由调用方按顺序组合。

### 3.4 抽样设计（`sampling.py` + `build_valset.py`）

三种设计，全部无放回 + 固定 seed，每次抽样都带回**一阶包含概率 `pi_i`**，
HT 权重 `w_i = 1/pi_i`，不变量 `sum w_i == N` 由 `Sample.check()` 断言：

| 设计 | `pi_i` | 何时用 |
|---|---|---|
| `uniform` | `n/N` | 默认，先跑通 |
| `stratified` | `n_h/N_h` | 层与被估量相关时降方差 |
| `pareto_pps` | `≈ n·q_i` | **类别不平衡**：正例 ~6% 时把标签花在信息量大的行上 |

关键实现决定：

- **一次抽样再切分**（`split_sample`）。分两次抽要么重叠（密封集失效），要么第二次只能
  从"总体减去第一次"里抽——那认证的是子总体而不是语料。随机划分一个合法样本，
  两半各自仍是同设计的合法样本，`pi` 按划分比例缩放。
- **每层至少 1 行**。`n_h = 0` 让该层单元 `pi = 0`，无偏估计量对它根本没定义。
  过薄的层自动**合并**（`collapse_small_strata`，survey 标准做法：层变粗，不引入偏差）；
  合并也救不了的（层数 > n）直接报错，给出可执行的修改建议，而不是悄悄降级。
- **均匀混合而非权重截断**。提议分布 `q_i = (1-ε)·s_i/Σs + ε/N`，均匀分量把权重
  从设计上界在 `N/(n·ε)`；事后截断也能压住权重，但代价是有偏。
- `pareto_pps` 的 `sum 1/pi == N` **只在期望意义上成立**（Rosén 的恒等式是渐近的），
  所以 `check()` 对它只做合理性区间断言。基于该设计的区间必须对**加权有界变量**有效，
  **Clopper–Pearson 在此不适用**。

> ⚠️ **当前 loop 算的是未加权准确率**。SRSWOR 与比例分层下未加权均值无偏，
> **pps 下不是**——它会按超采样比例高估那部分区域。`select.json` 里的 `weights`
> 已经带上了正确估计量所需的一切，但 `evaluate.py --score-inference` 还没用它。
> 所以 `--method importance` 现在会打印警告，且**端到端推荐用 `uniform`**，
> 直到加权估计量落地（下一期）。

`--label-source gt` 从 benchmark GT 打标签，**仅用于打通管线**，每次运行都打印警告：
这样建的验证集等于让 refine 循环读答案，不支持任何 oracle-free 的声称。
真正的 `--label-source oracle` 随 `oracle_label.py` 落地。

---

## 4. PR-2：采集运行期信号（B1 + B3 + trace 契约）

> **状态**：日志采集部分（B1 + 过滤 + 反馈接线 + solver 日志契约）已实现，
> 见 §4.0 / §4.1。cost 切片（B3）与 trace 契约扩展（§4.2）尚未做。

### 4.0 一个必须先讲的发现：只做捕获是根空管道

把 stdout 接出来之后，拿上一轮真实生成的 `solve_q3a.py` 一跑：

```
--- exit 0 execMs 100
--- raw stdout lines: 1 | stderr lines: 1
--- filtered: { "lines": [], "total": 0, "omitted": 0 }
```

**生成的程序一行日志都不打。** solver 的 prompt 从来没要求它说话，于是整条日志通道
恒为空——采集代码写得再好也传不出一个 bit。

所以"日志采集"的完整形态是**两半**，缺一不可：

1. **捕获侧**（orchestrator）：把输出接出来、落盘、过滤成可入 prompt 的几行；
2. **产出侧**（solver prompt 契约）：**要求程序报告自己的执行情况**。

第 2 半才是信息的来源。`agents/vadar-solver/prompt-text.md` 因此新增
`### DIAGNOSTIC LOGGING — REQUIRED`：分支计数、带上限的 WARN、每行异常捕获后继续、
以及收尾的 `rows_in/rows_out/elapsed`。它同时是 §4.2 分支归因的原料——分支标签在这里
就已经要求打出来了。

### 4.1 stdout 不再丢（B1）

`runSolver` / `runCompiled` 改为 `stdio: ["inherit","pipe","pipe"]`，两路都收，写
`iter_N/run.log`，同时 tee 到控制台（保留现在的交互观感）：

```js
const t0 = process.hrtime.bigint();
const s = spawnSync("python3", sArgs, { stdio: ["inherit", "pipe", "pipe"], maxBuffer: 64 * 1024 * 1024 });
const execMs = Number((process.hrtime.bigint() - t0) / 1000000n);
const stdout = (s.stdout || "").toString(), stderr = (s.stderr || "").toString();
process.stdout.write(stdout);                       // 保持现有可观测性
await writeFile(resolve(iterDir, "run.log"), stdout + "\n--- stderr ---\n" + stderr);
```

日志进反馈时**不是全量**，而是 `filterRunLog()` 三级削减（纯函数，可测）：

1. **留信号行** —— 匹配 `RUN_LOG_SIGNALS` 的行，**外加最后 8 行**（崩溃信息往往不含关键词，
   而且它总在末尾）；
2. **相同行折叠成计数** —— `WARN no keyword matched   (× 180)` 占一行而不是 180 行。
   折叠在**截断之后**做，所以只有尾部不同的长行也能合并；
3. **封顶并如实报告丢弃量** —— `total` / `omitted` 一起返回，
   截断过的视图绝不能读起来像完整的。

> 正则的一个坑：`\berror\b` **抓不到 `KeyError`**（"Key" 和 "Error" 之间没有词边界），
> 而 Python 异常名恰恰是最该抓的行。所以用 `\w*(?:error|exception)\b`。
> 这条是被测试逼出来的，不是想出来的。

实际接线：`runPythonLogged(argv, logPath, label)` 统一 `runSolver` / `runCompiled` 两条路径，
写 `iter_N/run.log`（含命令行 + stdout + `--- stderr ---` + stderr）、计 `execMs`、
把 stdout 回显到控制台。代价是**输出不再实时流式**，要等进程结束才一次性打印；
换来的是这些输出第一次真正存在于反馈可及的地方。

`maxBuffer` 设 64MB；ENOBUFS 或 spawn 失败时把原因写进 stderr 而不是当作"跑完了但没输出"。

### 4.2 solver trace 契约扩展

`trace_<q>.json` 现有 `{id: predicted}`，扩成：

```json
{ "predictions": { "m1": "comedy", "m17": null },
  "branch":      { "m1": "regex_hit", "m17": "fallback" },
  "errors":      { "m17": "KeyError: 'genre'" },
  "stats":       { "rows_in": 120, "rows_out": 118, "runtime_llm_calls": 0 } }
```

- **向后兼容**：`evaluate.py` 读到旧格式（顶层就是 id→value 的 map）时按原样处理，
  `branch`/`errors` 视为空。现有 fixture 与测试不改。
- prompt 契约（`agents/vadar-solver/prompt-text.md`）加一节，要求：
  1. 每行判定后记录一个**短分支标签**（`snake_case`，≤ 20 字符，取值不超过 8 种）；
  2. 单行异常必须 `try/except` 捕获后记进 `errors` 并继续，**不允许整轮 crash**；
  3. 标签要描述**判定路径**（`regex_hit` / `clip_above_thresh` / `fallback`），
     不是描述结果（不要 `is_comedy`）。

### 4.3 每轮 cost / latency 切片（B3）

`refineLoop` 增一个可选参数 `costProbe`：

```js
// runQuery 里：
const costProbe = () => ({
  ms:     phases.reduce((s, p) => s + p.duration_ms, 0),
  usd:    phases.reduce((s, p) => s + p.cost_usd, 0),
  calls:  phases.reduce((s, p) => s + p.llm_calls, 0),
  tokens: phases.reduce((s, p) => s + (p.tokens?.total ?? 0), 0),
});
// refineLoop 里，每轮 regen 前后各取一次快照，差值即本轮 codegen 开销
const before = costProbe(); await regen(...); const after = costProbe();
const codegen = { ms: after.ms - before.ms, usd: after.usd - before.usd, ... };
```

---

## 5. PR-3：scorecard 组装 + evaluate.py 的 branch 聚合

### 5.1 `evaluate.py --score-inference` 增量输出

在现有 `score_inference()`（`evaluate.py:235`）里，当 trace 带 `branch` 时额外聚合：

```json
{ "accuracy": 0.842, "n": 120, "correct": 101,
  "mistakes": [...], "n_mistakes": 19,
  "by_branch": [ {"branch":"regex_hit","n":60,"correct":57,"acc":0.95,"err_share":0.16},
                 {"branch":"fallback","n":20,"correct":7,"acc":0.35,"err_share":0.68} ],
  "errors": { "n_rows_errored": 2, "top": [["KeyError: 'genre'", 2]] },
  "missing": { "n": 0 } }
```

- `err_share` = 该分支贡献的错误 / 总错误 —— 反馈里直接指出"65% 的错误来自这一个分支"。
- **mistakes 采样从"前 15 条"改成按分支分层**：每分支至少 2 条、总数仍受 `--diff-cap` 约束。
  否则占多数的分支会把少数分支的错误全挤掉，而少数分支往往才是问题所在。

### 5.2 scorecard 组装（orchestrator，JS 侧）

`scoreIter` 的返回值扩成携带 scorecard，写 `iter_N/scorecard.json`：

```js
{ iter, 
  compile: { ok, stage, error_class, line, context[], pyflakes[] },   // ← preflight.json
  run:     { status, exec_ms, exec_ms_per_row, rows_in, rows_out,
             n_rows_errored, top_errors[], log_excerpt[] },           // ← run.log + trace.stats
  quality: { metric: "accuracy"|"f1", point, n, correct, by_branch[], mistakes[] }, // ← diff.json
  cost:    { codegen_usd, codegen_tokens, codegen_ms, runtime_llm_calls },          // ← costProbe 差值
  budget:  { cost_usd: <--cost-budget-usd>, ms_per_row: <--latency-budget-ms> } }
```

留好 `quality.lcb` / `quality.paired_vs_best` 两个字段位（本期填 `null`），
统计保证那一期直接往里写，不用再动 scorecard 的结构。

---

## 6. PR-4：`renderFeedback` v2 —— 具体喂什么

纯函数，签名从 `renderFeedback(prev)` 改为 `renderFeedback(scorecard, { best, history, budget })`。
**旧路径保留**，由 `--feedback v1|v2`（默认先 `v1`）切换，回归测试不动。

### 6.1 分支 A：编译失败 → 只给编译信息，其他一律不给

```
## COMPILE FAILED — FIX THIS FIRST, NOTHING ELSE
SyntaxError at solve_q3a.py:87:12 — invalid syntax
    85 |     for row in rows:
    86 |         v = classify(row)
 >> 87 |     if v == 'comedy'
    88 |         out.append(row)
Also flagged (static): F821 undefined name 'genre_map' at line 42

The program was NOT executed, so there are no quality numbers this round.
Fix the error above. Do not change anything else.
```

理由（SWE-agent ACI）：编译不过时给 quality/cost 数字是纯噪声，且会诱导模型同时做两件事。

### 6.2 分支 B：跑起来了 → 完整 scorecard

行数预算写死在渲染器里，超出即截断并标 `(N more omitted)`：

```
## OBJECTIVE                                                          [3 行]
Maximize per-row accuracy on the validation set,
subject to: codegen cost <= $0.50/iteration, exec <= 100 ms/row.
Current: accuracy 0.842 | cost $0.42 (OK) | 71 ms/row (OK)

## SCORECARD — iter 2 vs best so far (iter 1)                         [5 行]
  accuracy   0.842   (was 0.821)   +0.021    <- primary
  cost       $0.42   (was $0.38)   +$0.04    within budget
  latency    71 ms/row (was 64)    +7 ms     within budget
  compile    OK      runtime 8.4 s over 120 rows
  errors     2/120 rows raised KeyError: 'genre'  (rows still counted as wrong)

## WHERE THE ERRORS LIVE — accuracy by program branch                 [<=10 行]
  regex_hit          60 rows   0.95     16% of all errors
  clip_above_thresh  40 rows   0.80     16% of all errors
  fallback           20 rows   0.35     68% of all errors   <-- focus here
  Branches that are already good: regex_hit, clip_above_thresh.
  Do NOT change their logic — you will regress rows that currently pass.

## WRONG ROWS — stratified by branch, 19 total, showing 8              [<=16 行]
  [fallback]   id=m17  predicted=MISSING  expected=drama   text="..."
  [fallback]   id=m41  predicted=comedy   expected=drama   text="..."
  [regex_hit]  id=m3   predicted=comedy   expected=documentary  text="..."

## RUNTIME LOG — filtered (WARN/ERROR/fallback/retry)                  [<=25 行]
  WARN  no genre keyword matched for 18 rows, using fallback
  ERROR KeyError: 'genre' on id=m17
  ... (14 more omitted)

## HISTORY — what was changed and what happened                        [<=6 行]
  iter 0  initial              accuracy 0.774  $0.31  58 ms/row
  iter 1  widened regex set    accuracy 0.821  $0.38  64 ms/row  KEPT
  iter 2  added clip fallback  accuracy 0.842  $0.42  71 ms/row  KEPT

## GUARDRAILS                                                          [4 行]
The rows above are a RANDOM SAMPLE of a 2000-row corpus. Do NOT special-case
any id, exact text snippet, or expected value from this list — the program is
scored on the FULL corpus. Improve the RULE, not the sample.

## YOUR TASK                                                           [4 行]
1. State in one sentence WHY the `fallback` branch is failing.
2. Then edit solve_q3a.py in place to fix that branch.
3. Keep writing trace_q3a.json with predictions/branch/errors.
```

各段的依据（对应 §1）：`OBJECTIVE` 用约束式写法（EffiBench）；`by_branch` 的
"already good, don't touch" 是 AlphaCodium 的锚定；`YOUR TASK` 第 1 步是 Self-Debugging
的强制解释；`HISTORY` 带"改了什么"是 Reflexion；行数预算是 SWE-agent ACI。

### 6.3 `HISTORY` 的"做了什么改动"从哪来

现在 history 只有 F1 数列。两个选项：

- **(a) 便宜**：`git diff --stat` 式的行数差 + 变更函数名（用 `ast` 比对两轮 solve 文件的
  顶层函数体哈希）→ `"changed: classify_genre(), +12/-4 lines"`。
- **(b) 更好**：要求 agent 每轮在 `iter_N/CHANGE.md` 写一行意图，反馈里回读。

**取 (b)，(a) 兜底**——(b) 只需 prompt 加一句、成本为零，且拿到的是意图而非表象；
agent 没写时退回 (a)。

---

## 7. PR-5：prompt 槽位与判据微调

### 7.1 修 B2 —— feedback 不再污染 query_sql

`user-prompt-text.md` / `user-prompt.md` / `code-generator/user-prompt.md` 末尾加：

```markdown
{{feedback}}
```

`regenSolver` 改为 `solverVars(...)` 里传 `{ query_sql: sql, feedback }`，
iter_0 时 `feedback: ""`。这样 `## Query` 代码块永远只有 SQL。

### 7.2 判据：F1 打平时用 cost/latency 破平

本期**不动**统计判据（留给保证那一期），只在 `checkSemdbImprovement` 加一条尾部规则：

```js
if (prevOk && nextOk) {
  const dq = (next.f1 ?? -1) - (prev.f1 ?? -1);
  if (dq > 0) return true;
  if (dq < 0) return false;
  return next.cost_usd < prev.cost_usd ||                 // 质量相同 → 更便宜
        (next.cost_usd === prev.cost_usd && next.exec_ms_per_row < prev.exec_ms_per_row);
}
```

同时无论是否 promote，都把 `(quality, cost, latency)` 三元组追加进 `pareto.json`。

### 7.3 config

`semdb.config.mjs` 加：

```js
feedback: {
  version: "v1",              // "v1" | "v2"，验证通过后翻默认
  maxLogLines: 25, maxMistakeRows: 8, minRowsPerBranch: 2, maxHistoryRows: 6,
  costBudgetUsd: null,        // null = 只报告不约束
  latencyBudgetMsPerRow: null,
}
```

---

## 8. 测试

| 文件 | 断言 |
|---|---|
| `tests/test_preflight.py` | SyntaxError 定位到正确行列且带 ±3 行上下文；pyflakes 缺失时 `skipped` 而非报错；干净文件 `ok:true`；离线违规被抓 |
| `tests/test_evaluate_branch.py` | branch 聚合的 acc / err_share 正确；**旧格式 trace 仍能打分**（向后兼容）；mistakes 按分支分层且每分支 ≥ minRowsPerBranch |
| `tests/test_scorecard.mjs` | 纯组装函数：preflight 失败时 quality 为 null 且不编造；cost 差值切片正确；缺字段不抛异常 |
| `tests/test_feedback_v2.mjs` | 编译失败分支只渲染编译段；正常分支七段齐全且顺序固定；各段行数不超预算；超出时带 `(N more omitted)`；预算为 null 时 OBJECTIVE 不写约束句 |

回归：`test_refine_pure.mjs`、`test_val_feedback.mjs`、`test_vadar_offline_guard.mjs`
在 `feedback.version="v1"` 下必须**逐字节等价**地通过。

---

## 9. 落地顺序与验收

| PR | 内容 | 独立验收方式 |
|---|---|---|
| **PR-1** | `preflight.py` + 编译前置闸 | 故意写坏一个 `solve_*.py`，确认不执行就报错且带行号 |
| **PR-2** | stdout/run.log + exec_ms + cost 切片 + trace 契约 | 在既有 `runs/mmqa-q2a` 重跑，`iter_N/run.log` 与 cost 切片齐全 |
| **PR-3** | evaluate branch 聚合 + scorecard.json | scorecard 五个块齐全；`by_branch` 与手算一致 |
| **PR-4** | renderFeedback v2 + `{{feedback}}` 槽位 | 打印一轮真实反馈人工审阅；A/B：v1 vs v2 各跑 3 次 5 轮，比最终 accuracy 与达标轮数 |
| **PR-5** | cost/latency 破平 + pareto.json | 构造质量打平的两轮，确认选便宜的那个 |

**PR-1 ~ PR-3 不需要任何新标签、不需要 LLM 调用即可验收**，可以直接在现有 run 目录上重放。

---

## 10. 已知风险

- **branch 标签靠 agent 自觉**。它可能全打 `main` 或每行一个不同标签。缓解：`evaluate.py`
  检测到 `distinct(branch) > 8` 或 `== 1` 时在 scorecard 标 `branch_quality: "degenerate"`，
  反馈里换回不分层的渲染，而不是给一张没用的表。
- **反馈变长 → agent 注意力稀释**。这就是行数预算存在的原因；A/B（PR-4 验收）必须真跑，
  "信息更多"不等于"效果更好"，这是 SWE-agent 那条经验的核心。
- **cost 反馈可能诱发退化**：模型为省钱把 LLM 调用删光导致质量崩。因此 cost 只能是
  **约束**（超预算才提），永远不能与 quality 并列为"待权衡的目标"；破平规则也只在
  质量严格相等时生效。
- **val 模式下 runtime cost 恒为 0**（solver 离线），cost 维度此时只反映 codegen 开销；
  compiled 模式才有 extraction 的 runtime 成本。反馈文案要按模式区分，别在 val 模式下
  写 "runtime LLM calls: 0" 误导模型以为可以随便调 LLM。
