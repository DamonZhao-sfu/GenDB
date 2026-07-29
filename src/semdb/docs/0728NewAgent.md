GenDB SemDB Planner–Generator–Optimizer 架构修改计划

目标仓库：DamonZhao-sfu/GenDB基线分支：vis-operator交付性质：可直接交给 Codex 执行的代码修改计划；本文件不修改仓库代码建议实现分支：agent-pgo-architecture

0. 给 Codex 的执行指令

从 vis-operator 创建新分支 agent-pgo-architecture，按本文阶段顺序实现。不要删除旧的 VADAR Signature/API/Solver 路径；保留 --agent-architecture legacy 作为回滚入口。新架构先只替换 --direct 路径，不改变非 DIRECT 的 Schema Designer → Extractor → Code Generator 流程。

完成每个阶段后运行该阶段测试。不要在同一提交中同时修改 statistical certification、CERT 数据读取策略或跨查询 Agent Memory。新 Agent 只能读取 SELECT/validation 反馈，不能读取 CERT 或最终 ground truth。

建议先执行：

git switch vis-operator
git switch -c agent-pgo-architecture

1. 最终架构决策

采用三个逻辑 Agent：

Query Planner：合并当前 VADAR Signature 和 VADAR API 的规划职责，但不直接写 Python。输出类型化 plan.json。

Code Generator：根据 plan.json 生成 helper 和 end-to-end solver；也负责依据 Optimizer 的结构化 action 修改代码。

Optimizer：读取执行、preflight、SELECT validation、历史指标和错误样本，诊断失败原因并输出 optimizer_action.json；不直接编辑 Python。

flowchart TD
    Q["SQL + table metadata + local primitives"] --> P["Query Planner"]
    P --> IR["Typed plan.json"]
    IR --> G["Code Generator"]
    G --> C["helpers.py + solve_q.py"]
    C --> R["Deterministic preflight / run / score"]
    R --> F["iteration_feedback.json"]
    F --> O["Optimizer"]
    O --> A{"Action"}
    A -->|"PATCH_CODE"| G
    A -->|"REPLAN"| P
    A -->|"STOP"| B["Promote best candidate"]

为什么不把 Code Generator 和 Optimizer 合并

两者保持独立，但允许使用同一个底层模型。

Generator 的目标是忠实实现 plan；Optimizer 的目标是依据外部证据诊断偏差。

分离后可以区分“规划错误”和“实现错误”，并记录哪个环节真正带来提升。

Optimizer 不直接编辑代码，避免一边解释指标、一边无约束重写程序。

可以给两者不同工具权限、上下文和温度/effort。

物理上无需部署两套模型；semdb.config.mjs 可以把两个 config key 映射到相同 model id。

2. 必须保持的系统不变量

生成程序继续是离线程序：禁止运行时网络、远程 LLM、API key 和 endpoint。

--only-ids、trace_<q>.json、pairwise key、ordered self-pair、结果 CSV projection 等现有契约必须保留。

编译/preflight 失败时不运行全 corpus。

SELECT validation 与 CERT/最终 ground truth 严格隔离。

Optimizer 不能硬编码 validation row id、label 或错误样本。

每轮 candidate 必须自包含 helper 与 solver；不能像当前实现一样永久复用 iter_0 helper。

最佳 candidate 的 plan、helper、solver、manifest 必须一起 promotion。

legacy 路径的输出、CLI 和测试保持可用。

新架构只在 DIRECT 路径启用；非 DIRECT 路径本次不重构。

3. 当前代码锚点

当前位置

当前职责

修改方向

src/semdb/orchestrator.mjs::runPhase

读取 prompt 并调用 provider；强制 useSkills: false

支持每个 Agent 的固定 skillPath，通过 domainSkillsPrompt 显式注入

renderFeedback

把错误、指标和样本拼成自由文本

保留 legacy renderer；新增结构化 feedback builder

refineLoop

Generator 自己读反馈并重写代码

保留 legacy；新增 PGO loop

runQueryDirect::gen3Agents

Signature → API → Solver

PGO 模式改成 Planner → Generator

runQueryDirect::regenSolver

只重新调用 Solver，helper 固定在 iter_0

删除 PGO 对该路径的依赖；每轮生成完整 candidate

src/semdb/semdb.config.mjs

没有新三个 Agent 的模型项

增加 planner/generator/optimizer 配置

src/gendb/providers/*.mjs

已支持 domainSkillsPrompt

不改 provider API，只复用现有接口

4. 目标文件树

新增：

src/semdb/
├── agent-runtime/
│   ├── contracts.mjs
│   ├── feedback.mjs
│   ├── pgo-loop.mjs
│   └── skill-loader.mjs
├── contracts/
│   ├── semantic-plan.schema.json
│   ├── candidate-manifest.schema.json
│   ├── iteration-feedback.schema.json
│   └── optimizer-action.schema.json
├── agents/
│   ├── query-planner/
│   │   ├── index.mjs
│   │   ├── prompt.md
│   │   └── user-prompt.md
│   ├── semantic-code-generator/
│   │   ├── index.mjs
│   │   ├── prompt.md
│   │   └── user-prompt.md
│   └── semantic-optimizer/
│       ├── index.mjs
│       ├── prompt.md
│       └── user-prompt.md
├── skills/
│   ├── plan-semantic-query/
│   │   └── SKILL.md
│   ├── generate-semantic-program/
│   │   └── SKILL.md
│   └── optimize-semantic-program/
│       └── SKILL.md
└── tests/
    ├── test_agent_contracts.mjs
    ├── test_skill_loader.mjs
    ├── test_optimizer_routing.mjs
    └── test_pgo_loop.mjs

修改：

package.json
package-lock.json                 # 若仓库存在
src/semdb/orchestrator.mjs
src/semdb/semdb.config.mjs
src/semdb/docs/0727CurrentDesign.md
src/semdb/docs/0728Compileability.md

不要删除：

src/semdb/agents/vadar-signature/
src/semdb/agents/vadar-api/
src/semdb/agents/vadar-solver/

5. 结构化 Agent 契约

5.1 semantic-plan.schema.json

Planner 的唯一权威输出。最低字段：

字段

类型

说明

schema_version

string

初始为 "1.0"

query_id

string

如 q3a

plan_version

integer

初始为 1；replan 单调增加

parent_plan_version

integer/null

初始为 null

modality

enum

image、text、mixed

compilability.class

enum

exact、bounded_approximation、not_compilable

compilability.obligations

array

要使代码可执行必须满足的条件

compilability.unresolved

array

无法用当前 primitive 解决的问题

semantic_sites

array

SQL/NL 中每个 AI call site

helper_dag

array

有类型的 helper 节点和 primitive 绑定

relational_plan

array

filter/join/group/project/order 等顺序

trace_contract

object

row/pair/tuple 的 key、value、coverage

runtime_contract

object

输入、输出、offline、允许 import

validation_contract

object

sampling unit、id fields、禁止读取的数据

assumptions

array

明示假设

invariants

array

Generator 必须保持的语义约束

每个 semantic_sites[] 至少包括：

{
  "site_id": "site_0",
  "operator": "AI_FILTER",
  "predicate": "natural-language predicate",
  "inputs": [
    {"table": "images", "column": "filename", "type": "image_ref"}
  ],
  "sampling_unit": "row",
  "output_type": "boolean",
  "value_space": null
}

每个 helper_dag[] 至少包括：

{
  "helper_id": "h0",
  "name": "matches_predicate",
  "args": [{"name": "image_path", "type": "path"}],
  "return_type": "boolean",
  "depends_on": [],
  "primitive_steps": [
    {
      "primitive": "classify",
      "source": "src/semdb/vadar/predefined.py",
      "inputs": ["image_path"],
      "output_type": "label"
    }
  ],
  "confidence_signal": {
    "source": "primitive_score",
    "range": [0.0, 1.0],
    "higher_is_more_confident": true
  }
}

规则：

Planner 不输出 Python。

not_compilable 不能被 Generator 静默改写为“似乎可运行”的启发式。

未知值空间不能从 validation label 反推。

pairwise/self-join 的 sampling unit 和 key 构造必须写入 plan。

5.2 candidate-manifest.schema.json

每轮完整候选的清单：

{
  "schema_version": "1.0",
  "candidate_id": "q3a-iter-2",
  "query_id": "q3a",
  "iteration": 2,
  "plan_version": 1,
  "parent_candidate_id": "q3a-iter-0",
  "trigger_action": "PATCH_CODE",
  "artifacts": {
    "plan": "plan.json",
    "helpers": "_semantic_helpers_q3a.py",
    "solver": "solve_q3a.py"
  },
  "hashes": {
    "plan_sha256": "...",
    "helpers_sha256": "...",
    "solver_sha256": "..."
  }
}

Generator 完成后由 orchestrator 计算 hash；不要让 Agent 自报 hash。

5.3 iteration-feedback.schema.json

Optimizer 的唯一输入反馈文件。最低字段：

{
  "schema_version": "1.0",
  "query_id": "q3a",
  "candidate_id": "q3a-iter-1",
  "iteration": 1,
  "execution": {
    "status": "ok",
    "stage": "score",
    "preflight": null,
    "stderr_tail": "",
    "runtime_ms": 1234
  },
  "objective": {
    "name": "f1",
    "value": 0.74,
    "precision": 0.81,
    "recall": 0.68,
    "weighted": true
  },
  "errors": {
    "false_positive_total": 8,
    "false_negative_total": 13,
    "mistakes": []
  },
  "runtime_branches": {},
  "history": [],
  "data_boundary": {
    "source": "select_validation",
    "cert_accessed": false,
    "full_ground_truth_accessed": false
  }
}

实现时复用当前 scoreInference、scoreWithDiff 和 filterRunLog 的结果。样本数继续受 defaults.refineSampleCap 限制。

5.4 optimizer-action.schema.json

Optimizer 只能输出以下 action：

PATCH_CODE
REPLAN
STOP

最低字段：

{
  "schema_version": "1.0",
  "query_id": "q3a",
  "candidate_id": "q3a-iter-1",
  "action": "PATCH_CODE",
  "diagnosis": {
    "category": "threshold_or_mapping",
    "summary": "Recall loss is concentrated in ...",
    "evidence": ["feedback.errors.false_negative_total", "mistake:sample-3"]
  },
  "targets": [
    {
      "artifact": "helpers",
      "symbol": "matches_predicate",
      "intent": "broaden positive condition without changing result projection"
    }
  ],
  "preserve": [
    "trace key format",
    "offline-only runtime",
    "SQL SELECT projection"
  ],
  "expected_effect": {
    "primary_metric": "recall",
    "direction": "increase",
    "risk": "precision may decrease"
  }
}

REPLAN 必须包含 replan_reason 和证据，且只能在 maxReplans 预算内触发。STOP 表示没有安全、证据支持的修改，或已达到目标；orchestrator 仍以历史最佳 candidate 为最终结果。

6. 逐文件修改计划

6.1 src/semdb/semdb.config.mjs

增加：

directAgentArchitecture: "pgo",
enableAgentSkills: true,
maxReplans: 1,

在 Claude/Codex 的 agentModels 与 agentEffortLevels 中增加：

query_planner
semantic_code_generator
semantic_optimizer

建议 effort：

Agent

Claude

Codex

Query Planner

high

high

Code Generator

medium

medium

Optimizer

high

high

三个 key 可以映射到同一 model id；逻辑隔离不要求模型隔离。

6.2 src/semdb/orchestrator.mjs::parseArgs

增加：

--agent-architecture legacy|pgo
--max-replans <N>
--no-agent-skills

校验：

非法 architecture 立即报错。

max-replans 必须是非负整数。

这些选项只影响 --direct；非 DIRECT 运行时打印一次“ignored”警告或直接忽略。

6.3 src/semdb/agent-runtime/skill-loader.mjs

实现：

loadAgentSkill(skillPath) -> {
  name,
  description,
  body,
  prompt
}

要求：

解析 YAML frontmatter，但只允许 name、description。

校验 name 是 lowercase hyphen-case，长度小于 64。

prompt 明确标记这是绑定给当前 role 的 procedural skill。

不扫描用户目录，不做自动 skill 选择，不读取任意 .claude/skills。

文件不存在、frontmatter 错误或 skill name 与 Agent config 不匹配时 fail closed。

6.4 src/semdb/orchestrator.mjs::runPhase

修改逻辑：

const skillEnabled =
  opts.useSkills ?? agentConfig.useSkills ?? args.enableAgentSkills ?? false;

const domainSkillsPrompt =
  skillEnabled && agentConfig.skillPath
    ? (await loadAgentSkill(agentConfig.skillPath)).prompt
    : undefined;

调用 runAgent 时传：

useSkills: skillEnabled,
domainSkillsPrompt,

不要给 Claude Agent 开放动态 Skill 工具；每个 role 只注入自己的固定 skill。这样 Claude 和 Codex provider 都通过已有的 domainSkillsPrompt 得到相同内容。

--dry-run 时打印：

Agent name

bound skill name

rendered user prompt

不要打印 validation label 的完整内容。

6.5 新 Agent config

query-planner/index.mjs：

{
  name: "Semantic Query Planner",
  configKey: "query_planner",
  promptPath,
  userPromptPath,
  skillPath,
  skillName: "plan-semantic-query",
  useSkills: true,
  allowedTools: ["Read", "Write", "Glob", "Grep"]
}

semantic-code-generator/index.mjs：

{
  name: "Semantic Code Generator",
  configKey: "semantic_code_generator",
  promptPath,
  userPromptPath,
  skillPath,
  skillName: "generate-semantic-program",
  useSkills: true,
  allowedTools: ["Read", "Write", "Edit", "Glob", "Grep", "Bash"]
}

semantic-optimizer/index.mjs：

{
  name: "Semantic Optimizer",
  configKey: "semantic_optimizer",
  promptPath,
  userPromptPath,
  skillPath,
  skillName: "optimize-semantic-program",
  useSkills: true,
  allowedTools: ["Read", "Write", "Glob", "Grep"]
}

Codex provider 当前不使用 allowedTools 做强制隔离，因此 prompt、输出路径校验和 postcondition 也必须存在。

6.6 三个 Agent 的 prompt

prompt 只放：

role identity

hard constraints

唯一输出路径/格式

禁止行为

procedural workflow 放进对应 SKILL.md，避免 prompt 与 skill 重复。

Planner user prompt 输入：

query_id
query_sql
query_nl
modality
tables_doc
local_primitive_files
trace_contract
plan_path
previous_plan_path (replan only)
optimizer_action_path (replan only)

Generator user prompt 输入：

query_id
plan_path
parent_candidate_manifest_path (optional)
optimizer_action_path (optional)
helpers_path
solve_path
manifest_draft_path
table/data paths

Optimizer user prompt 输入：

query_id
plan_path
candidate_manifest_path
iteration_feedback_path
history_manifest_paths
optimizer_action_path
remaining_iteration_budget
remaining_replan_budget

6.7 src/semdb/agent-runtime/contracts.mjs

在 package.json 增加 "ajv": "^8.17.1"，并编译四个 JSON Schema。同步更新 lockfile；不能只检查“文件存在”。

导出：

readAndValidatePlan(path)
readAndValidateManifest(path)
readAndValidateFeedback(path)
readAndValidateOptimizerAction(path)
writeJsonAtomic(path, value)
sha256File(path)

writeJsonAtomic 先写同目录临时文件再 rename，避免 Agent/进程中断留下半个 JSON。

postcondition：

Planner 返回后，plan.json 必须合法。

Generator 返回后，helper、solver、manifest 必须存在；manifest hash 由 orchestrator 重写。

Optimizer 返回后，action 必须合法，且 query/candidate id 与当前轮一致。

6.8 src/semdb/agent-runtime/feedback.mjs

从当前 renderFeedback 中抽取结构化逻辑：

buildIterationFeedback({
  query,
  candidate,
  runOutcome,
  scoreOutcome,
  history,
  dataBoundary
})

保留原 renderFeedback export，供 legacy 测试和路径使用。可以让它调用 buildIterationFeedback 后再渲染为旧文本，但不要改变现有输出断言。

隐私/无泄漏规则：

SELECT 模式可以放 capped mistakes。

CERT 模式永远不进入 optimizer loop。

PGO 中 full ground truth 只允许在最终冻结后做一次 post-hoc evaluation，不能进入 optimizer feedback。没有 SELECT signal 时，PGO 退化为 Planner + Generator 单次运行。

data_boundary 必须记录反馈来源。

6.9 src/semdb/agent-runtime/pgo-loop.mjs

不要继续给旧 refineLoop 增加分支。新增独立：

runPgoLoop({
  args,
  query,
  runDir,
  createInitialPlan,
  replan,
  generateCandidate,
  optimize,
  executeCandidate,
  scoreCandidate,
  promoteCandidate
})

伪代码：

plan = await createInitialPlan(iter0)
assertValid(plan)

best = await generateCandidate(iter0, plan, null)
bestOutcome = await executeAndScore(best)

for iteration in 1..maxIterations:
  feedback = buildIterationFeedback(bestOutcome, history)
  action = await optimize(best.plan, best.manifest, feedback)

  if action == STOP:
    break

  if action == REPLAN:
    assert(replansUsed < maxReplans)
    plan = await replan(previousPlan, action)

  next = await generateCandidate(nextDir, plan, action, parent=best)
  nextOutcome = await executeAndScore(next)

  if checkSemdbImprovement(bestOutcome, nextOutcome):
    best = next
    bestOutcome = nextOutcome

await promoteCandidate(best)

关键语义：

Optimizer 诊断当前 best anchor；history 同时记录所有被拒绝的 trial 及其 action，防止重复失败。

Generator 的 parent 是 best candidate，防止连续劣化。

REPLAN 后 plan_version += 1，并完整重新生成 helper+solver。

compile/runtime failure 也允许 Optimizer 给 PATCH_CODE。

没有可测量 SELECT signal 时，PGO 退化为 Planner + Generator 单次运行；不要让 Optimizer猜测质量。

promotion 使用 manifest 指定的所有 artifact，而不是只复制 solver。

使用现有 checkSemdbImprovement 作为第一版 best-selection 规则，避免在架构 PR 中同时改变目标函数。

6.10 src/semdb/orchestrator.mjs::runQueryDirect

拆成两个入口：

runQueryDirectLegacy(...)
runQueryDirectPgo(...)

公共部分继续复用：

validation frame 构造

trace contract 构造

runSolver

scoreIter

telemetry

建议先从超长函数中抽出：

prepareDirectValidationContext(...)
buildDirectTraceContract(...)
makeDirectExecutor(...)
makeDirectScorer(...)

PGO initial artifacts：

iter_0/
├── plan.json
├── _semantic_helpers_<q>.py
├── solve_<q>.py
├── candidate_manifest.json
├── preflight.json
├── run.log
├── trace_<q>.json
├── diff.json
└── iteration_feedback.json

后续：

iter_N/
├── plan.json
├── optimizer_action.json
├── _semantic_helpers_<q>.py
├── solve_<q>.py
├── candidate_manifest.json
├── preflight.json
├── run.log
├── trace_<q>.json
├── diff.json
└── iteration_feedback.json

run root promotion：

plan.json
_semantic_helpers_<q>.py
solve_<q>.py
candidate_manifest.json

6.11 telemetry

telemetry.json 增加：

{
  "agent_architecture": "pgo",
  "plan_versions": 2,
  "replans": 1,
  "optimizer_actions": {
    "PATCH_CODE": 3,
    "REPLAN": 1,
    "STOP": 1
  },
  "best_candidate_id": "q3a-iter-3",
  "phases": []
}

phases 使用：

query_planner
semantic_code_generator
semantic_optimizer

不要把 row-level mistakes、validation labels 或 CERT 内容写入跨查询 telemetry。

7. 测试计划

7.1 test_agent_contracts.mjs

覆盖：

四个最小合法 JSON 通过。

缺少 schema_version、错误 enum、query id 不一致失败。

REPLAN 缺少 reason 失败。

manifest hash 不匹配失败。

not_compilable plan 不能进入 Generator。

7.2 test_skill_loader.mjs

覆盖：

三个 SKILL.md 可解析。

frontmatter 只有 name 和 description。

Agent config 的 skillName 与文件一致。

未知 frontmatter 字段、camelCase name、缺 description 失败。

--no-agent-skills 时不注入。

7.3 test_optimizer_routing.mjs

纯函数测试：

compile failure + PATCH_CODE → Generator，不触发 Planner。

semantic mismatch + REPLAN → Planner，然后 Generator。

replan budget 用完后再次 REPLAN → fail closed 或转换成明确 STOP，不能静默继续。

STOP → promotion best。

action candidate id 不匹配 → 拒绝。

7.4 test_pgo_loop.mjs

使用 fake callbacks，不调用真实模型：

planner 只调用一次；replan action 时多调用一次。

generator 每个 candidate 调用一次。

optimizer 在有 signal 后调用。

劣化 candidate 不覆盖 best。

best promotion 同时复制 plan/helper/solver/manifest。

没有 signal 时 single-shot。

compile failure 可以进入下一轮修复。

max iteration/replan budget 生效。

7.5 回归测试

至少运行：

node src/semdb/tests/test_refine_pure.mjs
node src/semdb/tests/test_val_feedback.mjs
node src/semdb/tests/test_agent_contracts.mjs
node src/semdb/tests/test_skill_loader.mjs
node src/semdb/tests/test_optimizer_routing.mjs
node src/semdb/tests/test_pgo_loop.mjs

再做两次 dry-run：

node src/semdb/orchestrator.mjs ... --direct --agent-architecture legacy --dry-run
node src/semdb/orchestrator.mjs ... --direct --agent-architecture pgo --dry-run

最后选择一个 image row-filter、一个 pairwise image join、一个 text query 做小规模真实运行。验收时比较：

是否生成正确 trace key。

是否只对 --only-ids 执行 semantic inference。

是否通过 offline validator。

是否保持最终 CSV projection。

PGO telemetry 是否能还原每次 action 和 plan version。

8. 分阶段实现与提交边界

Phase A：契约和 Skill 注入，不改变运行路径

实现 schemas、contracts、skill loader、三个 SKILL.md 和测试。legacy 行为必须完全不变。

验收：contract/skill tests 通过，legacy tests 通过。

Phase B：新 Agent scaffold

增加三个 Agent config 和 prompts；新增 config keys。实现 dry-run，验证三个阶段各自产出正确路径。

验收：PGO dry-run 不需要 endpoint/GT；输出中显示绑定 skill。

Phase C：PGO loop

实现 runPgoLoop、structured feedback、完整 candidate artifact 和 promotion。

验收：fake callback 测试覆盖 routing、rollback、replan budget。

Phase D：接入 DIRECT

抽取 DIRECT 公共 validation/execution/scoring 逻辑，接入 runQueryDirectPgo，保留 legacy。

验收：三类小规模 query 运行成功。

Phase E：文档与默认值

更新 0727CurrentDesign.md、0728Compileability.md，记录新架构和 artifact contracts。分支内默认 pgo，保留 CLI legacy rollback。

9. 本次明确不做

不修改 confidence interval、precision/recall/F1 guarantee 的统计实现。

不让 Optimizer 读取 CERT。

不引入跨 query、跨 benchmark 的长期 Agent Memory。

不删除旧 Agent。

不重构非 DIRECT pipeline。

不让 Optimizer 直接改 Python。

不把 Skill 放入用户级或 provider 专属目录。

Memory 的预留接口

本 PR 只把 plan.json、optimizer_action.json、candidate_manifest.json 和结构化 feedback 做好，这些就是未来 Memory 的安全数据源。后续若实现 Memory，只存：

query shape 与 sampling unit；

primitive/plan pattern；

failure taxonomy；

action 类型；

聚合 metric delta；

provider/model/version。

不要存 validation row 文本、row id、label、CERT 或完整生成代码。Memory retrieval 也应只注入 Planner/Optimizer，不直接注入 Generator。

10. Definition of Done

--direct --agent-architecture pgo 实际走 Planner → Generator → run/score → Optimizer。

Planner 不生成 Python；Optimizer 不编辑 Python。

每个 Agent 都绑定且加载唯一一个仓库内 SKILL.md。

四类 JSON contract 都在运行时校验。

每轮 candidate 的 helper/solver/plan 一致并可整体回滚。

legacy DIRECT 与非 DIRECT 均可继续运行。

新旧单元测试全部通过。

SELECT/CERT boundary 可从 telemetry 和 feedback contract 验证。

至少完成 image row、pair join、text 三类 smoke test。

Appendix A：三个 Agent 的完整 SKILL.md

下面内容应原样创建到对应路径。YAML frontmatter 只包含 name 和 description。

A.1 src/semdb/skills/plan-semantic-query/SKILL.md

---
name: plan-semantic-query
description: Plan GenDB SemDB semantic SQL operators into a typed, offline-executable physical plan. Use for initial planning or evidence-driven replanning when identifying AI call sites, determining row/pair/tuple sampling units, selecting local VADAR primitives, defining helper DAGs, preserving relational semantics, specifying trace contracts, or deciding that a query is not compilable with the available local runtime.
---

# Plan Semantic Query

Produce a typed semantic plan. Do not write implementation code.

## Read required inputs

Read the query SQL and natural-language description, table metadata, local primitive implementations, trace contract, and output schema. On a replan, also read the previous plan and the optimizer action.

Treat the primitive implementation files as authoritative. Never invent a function, parameter, return type, score meaning, or model capability.

## Build the plan

1. Identify every semantic call site in the SQL.
2. Normalize each call site into its input columns, predicate, output type, and sampling unit.
3. Distinguish row, cross-table pair, ordered self-pair, grouped, and tuple domains.
4. Preserve deterministic relational operations before and after semantic evaluation.
5. Classify compilability as `exact`, `bounded_approximation`, or `not_compilable`.
6. List every compilability obligation and unresolved semantic requirement.
7. Build a typed helper DAG. Give every helper explicit arguments, return type, dependencies, primitive bindings, and confidence signal.
8. Define the relational plan in execution order.
9. Define runtime inputs, result projection, trace key/value semantics, and `--only-ids` filtering location.
10. State assumptions and invariants explicitly.

## Apply planning rules

- Use only local, offline primitives available in the repository.
- Keep value spaces explicit. Read closed value spaces from database columns when allowed; do not infer them from validation labels.
- Preserve ordered-pair direction and diagonal rules for self-joins.
- Apply `--only-ids` after forming the correct validation unit and before semantic inference.
- Require a trace entry for every evaluated validation unit, including negative decisions.
- Keep physical trace identity separate from the SQL result projection.
- Reject silent fallback to a different semantic predicate.
- Mark a plan `not_compilable` when required information or capability is unavailable.

## Replan from evidence

Preserve all unaffected plan sections. Increment `plan_version`, set `parent_plan_version`, and change only what the optimizer evidence supports.

Do not replan merely because one sample is difficult. Require evidence of a plan-level problem such as a wrong sampling unit, missing input, invalid primitive capability, wrong value space, or impossible trace contract.

## Validate before finishing

Confirm that:

- every semantic site is represented;
- every helper dependency resolves;
- every primitive exists;
- every type edge is compatible;
- the relational plan preserves SQL projection and ordering semantics;
- trace keys match the validation unit;
- the runtime remains offline;
- no CERT, final ground truth, or validation label was used for planning.

Write only the requested `plan.json`. Make it conform to `src/semdb/contracts/semantic-plan.schema.json`.

A.2 src/semdb/skills/generate-semantic-program/SKILL.md

---
name: generate-semantic-program
description: Generate or revise a GenDB SemDB offline semantic program from a validated typed plan. Use when creating helper functions and an end-to-end solver, applying a structured optimizer patch, regenerating after a replan, preserving trace and result contracts, or repairing compile/runtime defects without changing the query's planned semantics.
---

# Generate Semantic Program

Implement the validated plan as a complete candidate. Do not redesign the query.

## Read required inputs

Read the validated `plan.json`, exact local primitive source files, table metadata, required output paths, and any parent candidate manifest. When revising code, also read the structured optimizer action and the parent helper and solver files.

If the plan is `not_compilable`, stop without generating a misleading program.

## Generate a complete candidate

1. Map every helper DAG node to one Python function with matching argument and return types.
2. Generate the helper module at the requested path.
3. Generate the end-to-end solver at the requested path.
4. Implement deterministic relational operations in the plan's stated order.
5. Implement the exact SQL result projection and CSV shape.
6. Implement `--only-ids` at the plan-specified sampling-unit boundary.
7. Write `trace_<q>.json` with one entry for every evaluated row, pair, or tuple.
8. Add concise stderr branch diagnostics without printing secrets or complete validation data.
9. Handle individual row/pair failures without corrupting unrelated results.
10. Leave candidate metadata for the orchestrator to hash and finalize.

## Apply optimizer actions

For `PATCH_CODE`, edit only the named artifact or symbol and preserve every listed invariant. Use the parent candidate as the starting point.

For `REPLAN`, regenerate the complete helper and solver from the new plan. Do not mix helper code from an older plan version.

Do not implement requests unsupported by the plan. If an action conflicts with the plan, fail with a clear `NEEDS_REPLAN` diagnostic.

## Enforce runtime constraints

- Do not call a network service, remote LLM, endpoint, shell download, or external API.
- Do not read CERT or final ground truth.
- Do not hardcode validation ids, labels, expected outputs, or mistake rows.
- Do not invent repository APIs.
- Do not silently weaken the semantic predicate.
- Do not change pair direction, diagonal inclusion, trace key format, or result projection.
- Keep confidence values aligned with the primitive's documented score semantics.

## Verify before finishing

Run syntax and static preflight checks that do not execute the full corpus. Confirm imports and referenced names resolve. Confirm helper and solver plan versions match.

Write only the requested helper, solver, and manifest draft paths. The orchestrator owns execution, scoring, hashes, best-candidate selection, and promotion.

A.3 src/semdb/skills/optimize-semantic-program/SKILL.md

---
name: optimize-semantic-program
description: Diagnose a GenDB SemDB candidate from structured preflight, runtime, SELECT-validation, metric, trace, and history evidence, then choose a bounded code patch, plan revision, or stop action. Use during iterative optimization to separate implementation defects from plan defects, target precision/recall/F1 failures, protect validation integrity, and produce an optimizer action without editing program files.
---

# Optimize Semantic Program

Diagnose the latest candidate and write one structured action. Do not edit Python or rewrite the plan.

## Read required inputs

Read the current plan, candidate manifest, iteration feedback, bounded history, remaining iteration budget, and remaining replan budget.

Treat `data_boundary` as a hard policy. Stop if feedback includes CERT or unauthorized final-ground-truth information.

## Diagnose in order

1. Check contract and preflight failures.
2. Check runtime failures and missing outputs.
3. Check trace coverage, key format, and sampling-unit errors.
4. Check relational projection and join/filter placement.
5. Check primitive arguments, value-space mapping, prompts, thresholds, and confidence semantics.
6. Compare precision and recall to distinguish over-broad from under-broad behavior.
7. Use capped mistakes as supporting evidence, not as cases to memorize.
8. Compare history to avoid repeating a failed action.

## Choose one action

Choose `PATCH_CODE` when the plan is sound and the defect is in implementation, mapping, threshold, prompt wording, control flow, trace handling, or error handling.

Choose `REPLAN` only when evidence shows a plan-level defect such as a wrong sampling unit, missing semantic input, invalid primitive choice, wrong value space, impossible helper type, or incorrect relational placement. Require remaining replan budget.

Choose `STOP` when no evidence-supported safe change remains, the candidate satisfies the configured goal, the budget is exhausted, or the query is not compilable with available primitives.

## Bound the proposed change

Name the target artifact and symbol. State the evidence path, intended change, preserved invariants, expected metric direction, and regression risk.

Prefer one falsifiable change per iteration. Do not request an unrelated rewrite.

## Protect validation integrity

- Never encode row ids, expected labels, false-positive rows, or false-negative rows into code.
- Never infer a hidden label rule from a small validation sample.
- Never request access to CERT or final ground truth.
- Never optimize accuracy alone when class imbalance makes precision/recall/F1 available.
- Never claim a statistical guarantee from point estimates.
- Never change the validation unit or metric definition to make the score look better.

## Validate before finishing

Confirm that the action references the current query and candidate, conforms to the remaining budgets, cites available evidence, and preserves offline runtime, trace identity, and SQL result semantics.

Write only the requested `optimizer_action.json`. Make it conform to `src/semdb/contracts/optimizer-action.schema.json`.

Appendix B：建议的 Codex 最终验收报告格式

实现完成后，Codex 应返回：

新分支名和 commit 列表。

新增/修改文件表。

架构行为摘要。

每个测试命令及结果。

三个 smoke test 的 query、modality、validation shape、best iteration、F1 变化。

尚未完成或有风险的项目。

明确确认没有删除 legacy 路径、没有启用 CERT feedback、没有实现跨查询 Memory。