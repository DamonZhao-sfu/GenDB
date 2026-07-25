# Synthesized code as a certified proxy: research + implementation plan

**Question.** GenDB / BespokeOLAP-style synthesis loops assume a *free, exact* oracle (run
the SQL, diff the answer). Semantic operators have no such oracle: the only judge is an LLM,
which is **expensive** and **noisy**. How do we build a usable synthesis loop under that
feedback, and can we certify the synthesized program well enough to *replace* the LLM?

---

## 1. What LOTUS and BARGAIN actually give us

| | LOTUS (VLDB '25) | BARGAIN (SIGMOD/PACMMOD '25) |
|---|---|---|
| Oracle `O` | large LLM (e.g. Llama-3-70B), **defined as ground truth** | expensive LLM (GPT-4o / Sonnet), **defined as ground truth** |
| Proxy `P` | small-LLM log-probs (rescaled to quantiles); embedding similarity for join/group-by | cheap LLM (GPT-4o-mini / Haiku) + its confidence |
| Guarantee | `P(recall ≥ γ_R) ≥ 1−δ` and `P(precision ≥ γ_P) ≥ 1−δ` for the filter cascade, w.r.t. the gold plan | three targets: accuracy `P(A(Ŷ) ≥ T) ≥ 1−δ`, precision `P(𝒫 ≥ T) ≥ 1−δ`, recall `P(ℛ ≥ T) ≥ 1−δ` |
| Mechanism | learn two thresholds `τ+`, `τ−` on the proxy score; rows in between go to the oracle | scan candidate thresholds `ρ` (proxy-score percentiles) top-down, accept the first that passes a test |
| Statistics | sample ~0.01% (min 100), importance sampling mixed with uniform, **CLT** approximation, δ/2 split across the two thresholds | **anytime-valid sequential test** (Waudby-Smith–Ramdas betting/EB bound, variance-adaptive), adaptive sampling *without replacement*, selection handled by a union bound with tolerance `η` (default `η=0`, justified by monotone precision curves) |
| Reported | matches/beats hand-built pipelines, up to 3.6× faster | up to **86%** more cost reduction than SOTA; SUPG (CLT-based) **misses its target >75% of the time at δ=0.1** |

### 1.1 BARGAIN's actual control flow (read from source, `ucbepic/BARGAIN@HEAD`)

It is **not** "label a fixed validation set, run the proxy on it, accept/reject". `BARGAIN_A.process()`:

1. **Proxy on the whole corpus first** → `(pred, score)` per row; sort **descending by score**.
2. Candidate cut points = `M=20` evenly spaced prefixes (`thresh_step = N//M`).
3. For each prefix, from smallest to largest coverage:
   - **Adjust the target for the prefix.** Rows sent to the oracle are correct by construction, so
     the proxy-handled prefix must clear a *higher* bar:
     `new_target = (T·(n_proxy+n_oracle) − n_oracle) / n_proxy`; skip the prefix if that exceeds 1.
   - **Buy labels adaptively, 10 at a time, without replacement, from *inside* that prefix**
     (`WoR_Sampler.sample(thresh, 10)`).
   - The oracle call is `oracle_func(input, proxy_output) -> (is_correct, correct_answer)` — i.e.
     the oracle **validates the proxy's answer** (LLM-as-judge on the proxy output), it does not
     merely produce an independent label.
   - Test with a **betting confidence sequence** (Waudby-Smith–Ramdas): `k_t = Π(1 + λ_i(X_i − m_i))`,
     rejected when `k_t ≥ 1/α`; finite-population correction `m_i = (N·T − S_{i−1})/(N − (i−1))`;
     `λ_i = √(2 log(2/α) / (i·log(i+1)·σ̂²_{i−1}))` — the `i·log(i+1)` form is the **anytime-valid**
     variant, which is why it may peek after every batch of 10.
   - Cheap prune: after ≥50 samples, if `mean − std < target`, abandon this prefix.
   - Stop at the **first prefix that fails**; keep the last one that passed.
4. Output = proxy answers on the un-sampled part of the winning prefix, oracle answers everywhere
   else — **the calibration labels are reused in the final answer**, so no budget is wasted.
5. Multiplicity across the `M` prefixes is handled by testing at `δ/η` (union bound with tolerance
   `η`, default `η=0` → `η+1=1`, justified by the empirical monotonicity of the score).

`BARGAIN_P` (precision target) is the same shape, sorting by positive-class probability
`x = p·s + (1−p)(1−s)`, testing the prefix's positive rate against `T`, and returning the largest
passing prefix ∪ the sampled rows the oracle called positive (maximizing recall at fixed precision).

So the flow you described — *build a validation set, test the proxy once, accept/reject* — is the
special case `M=1` with a fixed sample size. Three things are different in the real algorithm:
**(a)** the decision is not "is the proxy good enough" but "**how much coverage can I trust it
with**"; **(b)** labels are bought adaptively inside the region under test, not sampled up-front;
**(c)** the oracle validates the proxy's output and those labels land in the final result.

Two things to internalize:

1. **The guarantee is about *where to cut a score*, not about whether a program is
   semantically right.** Both papers assume the proxy emits a *continuous, roughly
   monotone* confidence. The theory is a threshold-selection theory.
2. **Both explicitly assume the oracle is noise-free** — expensive, but truth. That is the
   assumption that does not hold for us, and it is the largest open slot.

BARGAIN's lesson for us is also methodological: CLT-based guarantees (SUPG, and LOTUS's
cascade estimation) *empirically fail* at small sample sizes. Any bound we build must be
finite-sample (Hoeffding/Bernstein/betting/Clopper–Pearson), not asymptotic — our gold sets
are tiny (SemBench mmqa q3b has **3** gold items).

---

## 2. Mapping our setting onto theirs

| Their object | Our object | Consequence |
|---|---|---|
| oracle `O(x)` | strong LLM judging one row against the query predicate | expensive **and noisy** → §5 |
| proxy `P(x)` | **the synthesized program** (`solve_q.py` / `compiled_q.py`) | ~free per row after synthesis, deterministic, reusable across queries over the same corpus |
| proxy score | *(missing)* | a program emits a **hard label**, so there is no threshold to tune → §4.1 |
| candidate set `𝒞` | the programs the refinement loop generates | chosen **adaptively using the same labels** → §3 |

So: yes, the generated code is exactly a proxy in the LOTUS/BARGAIN sense — but it is a
*hard-label, adaptively-selected* proxy, and our oracle is noisy. Those three deltas are
the research content.

---

## 3. Your sampling idea: correct, and the guarantee slot is open

> 通过采样来用 LLM 标注，然后让 LLM 生成的代码跑这部分数据来验证语义正确性

### 3.1 What MOAR actually does (paper read in full)

MOAR (*Multi-Objective Agentic Rewrites for Unstructured Data Processing*, arXiv:2512.02289v4,
Wei, Shankar, Zeighami, Chung, Ozcan, Parameswaran) is a **multi-objective query optimizer**
for DocETL, not a verification method:

- **30+ rewrite directives** (18 new), in 5 categories. Code synthesis is one category —
  directives ⑥ Code Substitution (`o_x ⇒ code_op_x`), ⑦ Code Sub. Reduce
  (`reduce ⇒ code_reduce + map`), ⑧ Doc Compression via regex `code_map`, ⑨ Head/Tail —
  plus ⑫ Cascade Filtering, which injects cheap `code_filter` / small-model pre-filters
  before an expensive `filter`.
- **Search** = UCT/MCTS over a tree of complete pipelines with progressive widening
  (`W(n) = max(2, 1+√n)`), an LLM agent choosing + instantiating the directive, a custom
  utility based on *marginal accuracy contribution* to the Pareto frontier.
- **Budget** `B = 40` pipeline evaluations. Each candidate is executed on a **sample
  `D_O ⊂ D` of 40 documents** to get `(ĉ(P), â(P))`; the paper's reported numbers come from a
  **held-out `D_T = D \ D_O` of 100 documents**.
- **Accuracy `a(P)` is user-supplied**: "If `a` requires ground truth labels, the user
  provides them, **or `a` can be an LLM-as-judge implementation**."

**There is no statistical guarantee — and the paper says so explicitly:**

> "Note, however, that in all of these cases, unlike the relational setting, equivalence is
> *not guaranteed*; rewriting could lead to a plan that has worse accuracy than the original.
> Thus, we still need to evaluate each rewritten plan for accuracy (and cost)."

> "'Semantic soundness' requires that a rewritten pipeline accomplish the same task as the
> original. **Unlike type soundness, semantic soundness cannot be formally verified.**"

So MOAR = *search + empirical estimate on a 40-doc sample*. No confidence interval, no `δ`,
no correction for the ~40 adaptive comparisons made against that one sample. Definition 2.1 in
the paper is a Pareto-set definition, not a probabilistic statement. (An earlier draft of this
plan claimed MOAR proves a matching guarantee — that came from an unreliable PDF summary and is
wrong. The guarantee slot is **open**.)

**Their own data shows the oracle is noisy**, which is free ammunition for §4.3: on Biodex they
LLM-judged their pipeline's mismatches and found **37 of 132 false negatives (28%) were errors
in the ground truth**, and of 192 false positives only **8 (4.2%) were true model errors**;
correcting the GT moved RP@5 from 37.9% → 97.5%.

### 3.2 The bug that no existing system addresses

The loop generates programs `p₁…p_K` and keeps the arg-max. That score is an **optimistically
biased** estimate of the winner's true quality — adaptive data analysis / selection bias, worse
the more iterations we run. MOAR runs ~40 adaptive comparisons against one 40-document sample
with no correction; it mitigates this *only* by reporting paper numbers on a held-out `D_T`, i.e.
the optimizer itself has no notion of the bias. Four fixes, in increasing sophistication:

**The bug in the naive version.** The loop generates program `p₁…p_K` and keeps the one with
the best validation score. The winner's validation score is then an **optimistically biased**
estimate of its true quality — classic adaptive-data-analysis / selection bias. Reporting it
as "validated accuracy" is wrong, and it gets worse the more iterations we run (i.e. exactly
when the loop is working hardest). Four fixes, in increasing sophistication:

- **(a) Two budgets.** A *selection* set (reused freely by the loop, no claims made) and a
  *certification* set (fresh labels, touched once, produces the number we report). Trivial,
  costs labels.
- **(b) Learn-then-Test (LTT).** Treat each program as a candidate configuration `λ`, risk =
  per-row error, test `H_λ: risk(λ) > α` with a finite-sample UCB and Bonferroni over the
  `K` candidates actually tested. Any surviving `λ` is certified at level δ. `K` is ~5–20 for
  us, so the correction costs only `log K` — **cheap, and reuses one label set**. This is the
  cleanest drop-in for our loop.
- **(c) Anytime-valid sequential testing** (BARGAIN's WSR tool). Lets us *peek* after every
  new label and stop as soon as a candidate is certified — turns "how many labels do I need?"
  into an adaptive stopping rule instead of a fixed sample size.
- **(d) Reusable holdout / Ladder.** Only reveal a candidate's score when it beats the
  incumbent by more than the noise floor; caps information leakage per iteration.

### 3.3 Code audit of `github.com/ucbepic/docetl` (clone at `ae7f10f`, 2026-07)

Agentic Python codegen-as-proxy **does exist** in the repo; a BARGAIN-backed guarantee engine
**also** exists; **they are not connected.**

| Subsystem | Files | Landed |
|---|---|---|
| `code_map` / `code_filter` / `code_reduce` operators (execute Python as an op) | `docetl/operations/code_operations.py` (378 L) | pre-existing |
| MOAR search (UCT, Pareto frontier, `simulate(node) → (cost, accuracy)`) | `docetl/moar/MOARSearch.py` (1308 L), `Node.py`, `ParetoFrontier.py`, `optimizer.py`, `search_utils.py` | PR #464, 2025-11-28 |
| Directive ⑦ — agent synthesizes `transform()` to replace a `reduce` | `reasoning_optimizer/directives/swap_with_code.py` (339 L) | with MOAR |
| Directives ⑧/⑨ — regex `code_map` compression, head/tail | `deterministic_doc_compression.py` (350 L), `take_head_tail.py` (321 L) | with MOAR |
| Directive ⑫ — inject synthesized `code_filter` pre-filters before an expensive `filter` | `cascade_filtering.py` | with MOAR |
| **BARGAIN cascade engine** — imports `BARGAIN.process.BARGAIN_A/_P/_PR/_R`, betting confidence sequences, without-replacement adaptive sampling | `docetl/operations/utils/cascade.py`, `cascade_runner.py`, `docs/design/model-cascade.md` | PR #491, **2026-06-07** |

How synthesized code is validated today: `swap_with_code.llm_instantiate()` passes a
`validation_func` that is a **no-op** ("Basic validation is handled by Pydantic validators") —
i.e. schema/type checking only, **no comparison against the original operator's outputs**. The
only accuracy signal is pipeline-level `simulate()` on the 40-doc sample. A repo-wide grep for
`hoeffding|bernstein|clopper|bootstrap|confidence.interval|p_value|bonferroni` in the optimizer
path returns **nothing**.

The guarantee engine is opt-in per operator (`filter` / `resolve` / `equijoin`):

```yaml
cascade:
  proxy_model: gpt-4o-mini      # <-- must be a MODEL
  guarantee: recall             # accuracy | precision | recall
  target: 0.95
  delta: 0.05
  label_budget: 400
```

with `proxy_predict(item) -> (label, confidence)` implemented as a **single-token decode with
logprobs** on the cheap model (or an embedding model + logistic head). `CascadeConfig.proxy_model`
is a `str` — **there is no way to plug a synthesized program in as the proxy.**

**The near-miss is the opportunity.** Directive ⑫ inserts synthesized `code_filter` pre-filters
whose docstring says they should "prioritize high recall (rarely rejecting valid documents)" —
and it achieves that *by prompting the agent to be conservative*. A BARGAIN engine that can
**certify** exactly that recall at `1−δ` sits in the same repository, unusable for this purpose
for one structural reason: **a program emits a hard label, not a confidence score** (§4.1).
Closing that gap is a concrete, verifiable contribution against a well-known system, and the
`proxy_predict` callable seam is explicitly designed as a plug point.

**Certify precision and recall separately, not F1.** Set-F1 is a ratio of dependent counts
over a gold set of size 1–13 — the CI is nearly vacuous. BARGAIN and LOTUS both certify
*precision* / *recall* on the positive class, or per-row accuracy. We should mirror that and
translate to F1 only for reporting.

---

## 4. Other directions (beyond sample-and-validate)

### 4.1 Make the program emit a per-row score → then BARGAIN applies verbatim

This is the load-bearing piece, so it gets the detail. **What BARGAIN actually needs is much
weaker than a calibrated probability**: it sorts rows by the score, cuts at one of `M=20`
prefixes, and *measures* the prefix's true quality with oracle labels. So the score only has to
**rank** rows such that quality tends to decrease along the order; miscalibration is irrelevant
and non-monotonicity is absorbed by the `δ/η` union bound over thresholds. A score that is merely
*rank-informative* is enough.

| # | Signal | How to get it from a synthesized program | Cost | Failure mode |
|---|---|---|---|---|
| A | **Cross-program agreement** (per row) | synthesize `K` programs (GenDB's VADAR Signature→API→Solver already yields structurally different ones); score = size of the majority-output cluster / `K` | `K ×` ~free execution | correlated errors: all `K` share one misreading → over-confident. Diversify by construction (different signatures/helper sets), and verify rank quality on the oracle sample |
| B | **Rule provenance / branch tier** (per row) | instrument the program: exact match ≫ regex hit ≫ fuzzy match ≫ default/fallback branch → ordinal tier | prompt-contract change | tiers may be coarse (few distinct values → few usable thresholds) |
| C | **Matching margin** (per row) | top1−top2 gap already computed inside our helpers: lexical distance in `vadar/predefined_text.py`, CLIP/OCR/detector scores in `semvision.py` / `imagepatch.py` — the solver currently thresholds these and **throws the margin away**; just return it | trivial | scale differs per helper → normalize by quantiles over the corpus (LOTUS does exactly this for small-LLM logprobs) |
| D | **Self-generated row assertions** (per row) | the agent also emits invariants/preconditions; score = fraction satisfied on that row (CodeT's execution agreement, transposed to the data axis) | one extra agent call | assertions may be as wrong as the program |
| E | **Learned head on A–D** | logistic regression over (branch id, margins, agreement, length) fitted on an oracle-labeled slice — exactly what docetl does for embedding proxies | uses labels | fitting and testing on the same labels needs a split or an LTT correction |
| F | **`ABSTAIN` by construction** | program returns `ABSTAIN` when no rule fires | cheapest | degenerate 2-level score (one usable threshold), but already converts "silent wrong default" into "escalate" |

Recommended default: **A + C, normalized to corpus quantiles**, with F as the floor; add E once the
labeling harness exists. Then use BARGAIN_R / BARGAIN_P to pick the cut and route the rest to the
oracle: "the code isn't good enough" becomes a **cost dial** — any program, however mediocre, yields
a certified result at target `T`, just with more oracle calls — and the synthesis loop's objective
becomes *oracle cost at fixed certified quality* instead of raw accuracy.

**If we refuse to build a score at all**, BARGAIN degenerates: all `M` prefixes collapse to the
full corpus, and the only testable hypothesis is "this program's accuracy over the whole corpus
≥ `T`" — a single-shot Clopper–Pearson / betting test. Still worth having as certification (§ Phase 2),
but there is no partial-coverage dial and a failing test yields nothing usable.

### 4.1b Why the codegen-agent literature does not already solve this

Codegen papers *do* use statistics and confidence scores — but on the **program axis**, not the
data axis:

- **CodeT** (dual execution agreement): run `K` candidate programs against `K'` generated tests;
  solutions passing the same test set form a consensus set scored by `|solutions| × |tests|`. Ranks
  *programs*. **AlphaCode** clusters programs by behavior on generated inputs — CodeT reports it is
  consistently worse because trivial programs (always return `None`) cluster together.
- **Conformal / PAC line**: conformal language modeling, ConU, selective generation, PAC prediction
  sets for code built from ASTs/partial programs, and LTT-based code UQ (arXiv:2605.12201). Their
  guarantee is *"the prediction **set of programs** contains a correct program w.p. ≥ 1−α"* or
  *"abstain to hold a target risk on this generation"*.

Neither shape is what a proxy cascade needs: **for one fixed program, which rows can I trust?**
The bridge is mechanical — execution agreement is a statistic over a (program × input) matrix;
CodeT/AlphaCode aggregate along the *input* axis to rank programs, and we aggregate along the
*program* axis to score each row (signal A). Cheap, and as far as this survey found, unclaimed.

### 4.2 PPI / prediction-powered inference for aggregate answers
For `count` / `avg` / `group-by-agg` queries: run the program on **all** rows, buy oracle
labels on a random subsample, and use the PPI rectifier to get an unbiased estimate with a
valid CI. The guarantee is on **the query answer**, not on the program — it stays valid even
if the program is badly biased, and it degrades gracefully to the classical CI when the
program is useless. This is the right tool for our AQP-flavored queries and complements the
per-row cascade.

### 4.3 Noisy oracle — the actual open slot
Both papers *define* the oracle as truth. We cannot. Options, roughly by increasing rigor:
1. **`k`-vote self-consistency** as the oracle, plus a small **human-labeled anchor set** to
   *estimate* the oracle's own error rate `η`.
2. **Bound propagation**: if oracle error ≤ `η`, a certified proxy-vs-oracle agreement of `T`
   implies true accuracy ≥ `T − η` (worst-case coupling). Cheap, honest, slightly loose.
3. **DSL / PPI with two label tiers** (LLM = imperfect surrogate, human or strong-ensemble =
   gold): doubly-robust, valid even with *non-random* surrogate error. Egami et al. show that
   ignoring surrogate noise destroys coverage — a 90%-accurate surrogate gives a nominal 95%
   CI only ~40% real coverage. That number alone justifies this direction.
4. Report the **guarantee-degradation curve** as `η` grows — the differentiating theory plot.

### 4.4 Oracle-free signals — spend labels only where they buy information
These cannot certify, but they decide *which rows to label* and prune candidates for free:
- **differential testing** between the `k` synthesized programs → label only disagreement rows
  (maximum information per label);
- **metamorphic / invariant checks** — permutation invariance, paraphrase stability, join-key
  consistency, idempotence, dedup, monotonicity of a threshold predicate;
- **schema/type/coverage checks**, crash/empty-output, distribution sanity (predicted positive
  rate vs. prior).

### 4.5 The loop as best-arm identification
Reframe refinement as **sequential experimental design**: candidates = arms, oracle labels =
expensive noisy rewards. Use successive halving / LUCB with a shared label budget instead of
fully evaluating every candidate. Gives a budget-optimality statement *and* a
near-best-selection guarantee — and it composes with 4.4 (which rows) and 3(c) (when to stop).

### 4.6 Stratify on *program structure* (only possible when the proxy is a program)
LOTUS importance-samples on the proxy score. We can stratify on the branch/rule the program
took, then Neyman-allocate labels across strata. Tighter CI per label, and it localizes the
failure ("the default branch is where 80% of the error lives") — which is directly actionable
feedback for the next iteration. This is a genuinely novel twist of the proxy-guarantee idea.

### 4.7 Composition across a plan, and reuse across a workload
- **Composition**: GenDB compiles a query into extract + relational plan; per-operator
  guarantees must compose (filter∘join∘agg) under one δ budget. LOTUS gives per-operator
  guarantees; end-to-end composition is largely open and is *our* setting.
- **Reuse**: mmqa q3a–q3g share one corpus and one extracted attribute. Labels bought for one
  query certify the shared attribute for the others → amortized certification across a
  workload, with a δ budget spent per query. This is GenDB's existing amortization story,
  extended to guarantees.

---

## 5. Implementation plan

**Phase 0 — position against docetl (done for MOAR, §3.1/3.3).** Remaining: read BARGAIN §3
(the three algorithms + the WSR test), LOTUS §4 cascade, and `docetl/docs/design/model-cascade.md`
(its "Risks" section documents the failure modes they already hit, incl. targets that are
statistically impossible at a given sample size). Then `pip install` the BARGAIN library and read
its `Proxy`/`Oracle` interface — it is the reference implementation of the statistics we need, do
not re-derive them.

**Phase 0.5 — the one-week prototype that tests the whole thesis.** Implement
`proxy_predict(row) -> (label, confidence)` for a GenDB-synthesized program (confidence from
`k`-program vote margin + branch provenance), call `BARGAIN_R` with `target=0.95, δ=0.05`, and
measure oracle calls saved on mmqa q3a–g. If a synthesized program's confidence is too flat to
beat a small-LLM proxy, the whole direction is in trouble and we learn it cheaply.

**Phase 1 — oracle harness.** `oracle_label.py`: strong-LLM row labeler; `k`-vote
self-consistency; on-disk cache keyed by `(query, row_id, prompt_hash)` so experiments are
reproducible and cost is paid once; explicit cost/latency accounting. Everything downstream
reads labels from this cache.

**Phase 2 — certification, kept *out* of the loop.** `certify.py`: given the `K` programs an
existing run produced plus a label budget, return either "`p_i` certified: `P(precision ≥ T) ≥
1−δ`" or "nothing certified". Finite-sample bounds only (Clopper–Pearson / EB-betting), LTT
Bonferroni over `K`. **This is the smallest change that makes the current loop honest**, and it
does not require touching the refinement code.

**Phase 3 — label allocation.** Replace "one random sample" with: disagreement-driven
sampling (4.4), stratification by program branch (4.6), racing over candidates (4.5). Metric:
**labels needed to certify at (T, δ)**.

**Phase 4 — abstention + cascade (4.1).** Prompt contract: programs emit `(label, conf)` or
`ABSTAIN`. BARGAIN PT/RT picks the routing threshold. Report the cost/accuracy Pareto front:
oracle calls saved at guaranteed `T`, vs. LOTUS cascade, vs. BARGAIN, vs. MOAR, vs. all-LLM.

**Phase 5 — noisy-oracle correction (4.3).** Anchor set + `η` estimation + bound propagation,
then DSL/PPI two-tier estimation. Deliverable: the guarantee-degradation curve.

**Phase 6 — evaluation on SemBench.**
- *Use the SemBench ground truth only as the meta-evaluator*, never inside the loop. This is
  the methodological trick that makes the study possible: we can simulate an expensive noisy
  oracle (sample LLM labels, inject known noise `η`) and then **check whether the guarantee
  actually holds** on the full GT — the check SUPG failed >75% of the time.
- Headline plots: (i) labels vs. certified quality; (ii) oracle calls saved at target `T`;
  (iii) **empirical violation rate** over ≥100 trials per (T, δ); (iv) the **selection-bias
  money plot** — validation score of best-of-K vs. its true full-corpus score, naive reuse vs.
  LTT. Plot (iv) is producible today from existing runs, before any new labeling.
- Workloads: mmqa q3a–g (shared text corpus, small gold), movie (large gold, id-keyed), cars /
  medical (multi-modal, scale suffixes), ecomm (hardest: ARI/ranking metrics where membership
  guarantees do not apply — say so explicitly rather than forcing it).

---

## 6. Risks and honest caveats

- **Tiny gold sets.** mmqa gold is 1–13 items/query. Certify per-row precision/recall, not
  set-F1; several queries will be uncertifiable at any reasonable δ — report that honestly.
- **Class imbalance.** Positives are ~6% (13/200). Recall bounds need stratified/importance
  sampling or the whole budget lands on negatives.
- **Joins.** The labeling unit is a *pair* (M×N). Needs blocking first; LOTUS treats joins with
  a separate algorithm, and our guarantee must follow suit.
- **Oracle cost multiplier.** `k`-vote self-consistency multiplies oracle cost by `k`; budget
  accordingly, and report cost in *oracle calls*, not wall-clock.
- **Non-membership metrics** (aggregation, ranking, ARI) do not fit precision/recall targets —
  use PPI (4.2) there, or scope them out.

---

## 7. References

- BARGAIN — *Cut Costs, Not Accuracy: LLM-Powered Data Processing with Guarantees*,
  Zeighami, Shankar, Parameswaran. arXiv:2509.02896 · PACMMOD 10.1145/3769776 ·
  code: github.com/ucbepic/BARGAIN
- LOTUS — *Semantic Operators: A Declarative Model for Rich, AI-based Data Processing* /
  *Semantic Operators and Their Optimization*, Patel et al. arXiv:2407.11418 · VLDB vol18 p4171
- MOAR — *Multi-Objective Agentic Rewrites for Unstructured Data Processing*, Wei, Shankar,
  Zeighami, Chung, Ozcan, Parameswaran. arXiv:2512.02289v4 (1 Apr 2026). Code:
  github.com/ucbepic/docetl (`docetl/moar/`, `docetl/reasoning_optimizer/directives/`)
- DocETL model cascades — `docetl/operations/utils/cascade.py` + `docs/design/model-cascade.md`
  (PR #491, 2026-06-07): BARGAIN vendored as the statistical core for filter/resolve/equijoin
- SUPG — Kang et al., *Approximate Selection with Guarantees using Proxies*, VLDB 2020
- LTT — Angelopoulos et al., *Learn then Test: Calibrating Predictive Algorithms to Achieve Risk
  Control*, Annals of Applied Statistics 19(2), 2025
- PPI — Angelopoulos et al., *Prediction-Powered Inference*, 2023 + follow-ups (stratified,
  active, cost-optimal budgets)
- DSL — Egami, Hinck, Stewart, Wei, *Using Imperfect Surrogates for Downstream Inference*,
  NeurIPS 2023
- WSR — Waudby-Smith & Ramdas, *Estimating means of bounded random variables by betting*,
  JRSS-B 2024 (the anytime-valid test BARGAIN builds on)
