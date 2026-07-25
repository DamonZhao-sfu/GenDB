# VIS_OPERATOR_PLAN — semdb VADAR 视觉算子库补齐计划

> 对标论文：**CADENZA: Compiling Natural-Language Intent into Task-Specific Operator DAGs
> for Semantic Query Processing**（Jaehyun Ha, Yongjoo Park, Wook-Shin Han；SIGMOD '27；
> arXiv 2606.29151）。本文只覆盖 **image modality**，text 算子另议。
>
> 状态：**P0 与 P1.1–P1.4 已实现**（分支 `vis-operator`，实施计划见
> [VIS_OPERATOR_IMPL_PLAN.md](VIS_OPERATOR_IMPL_PLAN.md)）。P1.5 / P2 / P3 / P4 未开始。
> 决策前提（已确认）：(1) VLM 采「受控放开 + 计费」；(2) 补齐范围 = **Score 化优先 + 新算子并行**。

---

## 0. 结论先行

我们和论文的差距**不主要在算子个数**（11 vs 11，看起来接近），而在三件结构性的事：

1. **输出 schema 丢了 `Score`。** 论文里每个 image 算子的输出 schema 都带 `Score`
   （`R_Cls(Label, Score)`、`R_Obj(BBox, Label, Score)`、`R_OCR(Text, Score)`…）。
   我们 `vadar/predefined.py` 的 `classify` / `detect` / `read_text` **底层拿到了 score 却在
   包装层扔掉了**。没有 score ⇒ 没有 proxy cascade、没有 router、没有阈值 BO 调优 ——
   论文 §5 的整套物理层能力全部无法落地。这是**唯一的卡脖子项**。
2. **结构性算子退化成确定性切分。** 论文 `OpImgRegion` 输出 `(RegIdx, BBox, Mask)`，是学习到的
   区域提案/分割；我们只有 `regions_grid` / `regions_center` / `crop` 三个 model-free 切法。
   论文 E-Commerce Plan 2 的「region 对齐 + 逐区域抽取」在我们这里做不出来。
3. **每个逻辑算子只有一个物理实现。** 论文的 `(directive, operator) → implementation` 目录里
   一个算子有 symbolic / specialized / general-purpose / composite 四族实现 + 一个数据感知
   router。我们 `semvision._run_attr` 的 tier 契约（`cv/clip/dino/detector/domain/distilled/vlm`）
   在**形式上已经是这个东西**，但 `dino` / `distilled` / `vlm` 三档是 stub，直接返回 `("none", 0.0)`。

其中 **第 1 条是第 3 条和本分支上 `VALSET_GUARANTEE_IMPL.md` / `FEEDBACK_LOOP_IMPL.md` 的共同前置**：
验证集要度量 quality、反馈环要调阈值，都需要算子吐出可比较的 confidence。

---

## 1. 论文侧：image 算子目录（Table 1 原文）

| Category | Operator | Output Schema | Description |
|---|---|---|---|
| Structural | `OpImgRegion` | `R_Reg(RegIdx, BBox, Mask)` | Segments the image into indexed regions (boxes or masks). |
| Structural | `OpImgOCR` | `R_OCR(Text, Score)` | Extracts text content with spatial coordinates. |
| Attributive | `OpImgCls` | `R_Cls(Label, Score)` | Classifies the entire image into categories. |
| Attributive | `OpImgObj` | `R_Obj(BBox, Label, Score)` | Detects objects with bounding boxes and class labels. |
| Attributive | `OpImgKeypt` | `R_Kpt(KptIdx, Point, Label)` | Identifies semantic keypoints (e.g., joints, landmarks). |
| Attributive | `OpImgVQA` | `R_VQA(Answer, Score)` | Answers natural language questions about the image. |
| Associative | `OpImgSceneRel` | `R_SGG(Subject, Predicate, Object, Score)` | Extracts visual relationships (scene graph triples). |
| Associative | `OpImgPairScore` | `R_Sim(Score)` | Quantifies visual similarity between an image pair. |
| Generative | `OpImgCap` | `R_Cap(Caption)` | Generates a descriptive caption for the image. |
| Generative | `OpImgEdit` | `R_Edit(Image)` | Generates a modified image based on constraints. |
| Latent | `OpImgEmbed` | `R_Vec(Vector)` | Encodes the image into a dense vector representation. |

### 支撑机制（比算子表本身更重要）

- **多实现 + 数据感知 router**（§5.1）。实现目录按 `(directive d, operator l)` 索引，
  `d ∈ {symbolic, specialized, general-purpose, composite}`。`SynthesizeRouter` 用**便宜的
  非推理特征**（论文原文：*image resolution/aspect ratio, and simple pixel statistics*）算一个
  difficulty score，再用 **percentile bucketing**（rank 百分位 + softmax 参数化的 N−1 个切点）
  把行分派到 N 个实现。切点由 Bayesian optimization 在验证集上调。
- **Template 1 — Partitioned Extraction**：`Apply_t(r) ⇝ Norm(⋃_{s∈P(r)} Apply_t(s))`。
  只有当 task 是 partition-local 且 normalizer 保住跨分区标识符时才等价，否则算近似替代方案，
  靠验证集排名择优。
- **Template 2 — Cross-Modal Proxying**：`Apply_t^img(r) ⇝ Apply_t^txt(Apply_c(r))`。
  典型例：OCR → 文本过滤，代替重量级 vision model；**安全网是把 OCR 空/低置信的行路由到更强的
  vision backend**。
- **Proxy Cascade**：便宜 proxy 打分 → 只有分数落在不确定带的行才升级到 LLM/VLM。
- **实测用法**（论文 §7）：Wildlife 大量用 `OpImgCls` + specialized backend；E-Commerce Q6 用
  `OpImgCls` + cropping-based decomposition；CarDamage Q8 是「coarse `OpImgCls` scan → 只对
  positives 跑 LLM verifier」，F1 0.08–0.35 → 0.56。E-Commerce Plan 3 是
  `OpImgRegion → OpImgCls(brand-like) → OpImgVQA("Is this region brand X?")` 的**闭式验证**，
  把开放式抽取降级成 yes/no。

---

## 2. 我们侧：现状清单

### 2.1 `vadar/predefined.py` — 暴露给 agent 的 11 个 primitive

```
classify(image, options, template)          -> str        # CLIP zero-shot，只回 label
best_ocr_match(image, options)              -> str        # OCR + difflib 模糊匹配
dominant_colors(image, min_frac, center_frac) -> list[str]  # HSV CV
verify_property(image, prop)                -> bool       # CLIP vs 否定
score(image, text)                          -> float      # CLIP 图-文相似度
read_text(image)                            -> str        # 裸 OCR 文本
detect(image, object_prompt)                -> list[patch]  # YOLO，只回子图
crop(image, l, t, r, b)                     -> patch
regions_grid(image, rows, cols, overlap)    -> list[patch]
regions_center(image, frac)                 -> patch
pair_score(image, other)                    -> float      # CLIP 图-图余弦
```

### 2.2 底座（`imagepatch.py` / `semvision.py`）已有但**没暴露**的能力

| 能力 | 位置 | 现状 |
|---|---|---|
| `read_text_boxes(min_conf)` → `[{text, box(绝对坐标), score}]` | `imagepatch.py:132` | 已实现，**不在 `MODULES_SIGNATURES` 里**，agent 看不见 |
| `clip_classify` 返回 `(label, score)` | `semvision.py:151` | score 在 `predefined.classify` 被丢弃 |
| `detect_boxes` 返回 `(label, conf, box)` | `semvision.py:334` | label/conf/box 在 `predefined.detect` 被丢弃 |
| `clip_multilabel(labels, thresh)` | `semvision.py:165` | 已实现，`ImagePatch` 和 `predefined` 都没包 |
| `embed_corpus` / `save_embeddings` / `load_embeddings` | `semvision.py:197-213` | 只在离线物化路径用，**没有 per-image `embed()` primitive**，也没有向量检索 |
| `domain_classify` + `XrayClassifier`（torchxrayvision） | `semvision.py:382-418` | 只能通过 extractor-spec `tier="domain"` 走，agent 的 program 里够不着 |

### 2.3 VLM 侧

- `semvqa.py` — **已经是标准的 `OpImgVQA`**：guided decoding 约束到 value space，score 从
  token logprobs 推导（`_score_from_logprobs`），还有 `img_vqa_batch` 和 `escalate(theta=...)`
  —— 即一个现成的 proxy cascade。
- `semcaption.py` — **已经是 `OpImgCap`**，把 caption 物化成一列。
- `semruntime.py` — residual 层的 `vlm_answer` / `vlm_judge`，带 `METER`（judge_calls /
  skipped / scores）。
- **但**：`orchestrator.mjs:210` 的 `VADAR_RUNTIME_FORBIDDEN` 把 `semvqa|semcaption|img_vqa`
  正则拉黑，生成的 VADAR 程序**一行都调不到**。所以论文的 CarDamage Q8 pattern
  （coarse scan → selective verifier）和 E-Commerce Plan 3（闭式验证）在我们这里**结构上不可表达**。

### 2.4 extractor-spec tier 契约（`semvision.validate_extractor_spec`）

允许 `cv | clip | dino | detector | domain | distilled | vlm`，但 `_run_attr` 里
**`dino` / `distilled` / `vlm` 三档没有实现**，落到最后一行 `return ("none", 0.0)`（residual miss）。
即：论文那张 implementation catalog 我们有**表**没有**货**。

---

## 3. Gap Matrix

| 论文算子 | 我们的对应物 | 覆盖度 | 缺什么 |
|---|---|---|---|
| `OpImgRegion (RegIdx, BBox, Mask)` | `regions_grid` / `regions_center` / `crop` | 🟡 部分 | 只有 model-free 网格切分。无区域**提案**（selective search / SAM / DINO），无 `Mask`，无 `RegIdx` 稳定编号，patch 不回吐自己的 box |
| `OpImgOCR (Text, Score)` | `read_text` (+ 未暴露的 `read_text_boxes`) | 🟡 部分 | agent 拿不到 spatial coords 和 score；`read_text` 只回拼接字符串 |
| `OpImgCls (Label, Score)` | `classify` / `verify_property` | 🟡 部分 | **丢 Score**；无 multilabel；无 domain-model（X-ray）实现；只有 CLIP 一个 backend |
| `OpImgObj (BBox, Label, Score)` | `detect` | 🟡 部分 | **丢 BBox / Label / Score**，只回子图列表；YOLOv8n **闭词表 COCO-80**，OOV 直接返回 `[]`；无 open-vocab 检测器 |
| `OpImgKeypt (KptIdx, Point, Label)` | — | 🔴 缺失 | 完全没有 |
| `OpImgVQA (Answer, Score)` | `semvqa.py` + `semruntime.vlm_answer` | 🟠 有实现但**被 guard 隔离** | 不是 agent 可组合的算子；program 内无法做 cascade / 闭式验证 |
| `OpImgSceneRel (S,P,O,Score)` | — | 🔴 缺失 | 完全没有（成本高，SemBench 需求弱） |
| `OpImgPairScore (Score)` | `pair_score` | 🟢 有 | 缺批量/向量化 join 路径；只有 CLIP 一个 backend（无 DINOv2） |
| `OpImgCap (Caption)` | `semcaption.py` | 🟠 有实现但**被 guard 隔离** | 物化列在 orchestrator 层，program 里读不到 |
| `OpImgEdit (Image)` | — | ⚪ 非目标 | 查询处理用不上，明确不做 |
| `OpImgEmbed (Vector)` | `semvision.embed_corpus` | 🟡 部分 | 无 per-image `embed()` primitive，无向量索引 / kNN / top-k 检索算子 |

**机制层：**

| 论文机制 | 我们 | 缺口 |
|---|---|---|
| 每算子多实现目录 | tier 契约存在，3/7 档是 stub | 需要把 `dino` / `distilled` / `vlm` 填上，并让**同一个逻辑算子**可绑定多实现 |
| 数据感知 router（percentile bucketing + BO） | 无 | 需要 cheap feature 提取 + 分位分桶；BO 可复用本分支的 valset |
| Proxy cascade | `semvqa.escalate` 有，但只在 residual 层 | 需要下沉成 program 内可写的模式 |
| Template 1 分区抽取 | `regions_grid` 是 partitioner | 缺 `Norm(⋃ …)` 合并算子（去重/投票/坐标归一） |
| Template 2 跨模态代理 | orchestrator 层的 caption 物化 | 缺 program 内的 `caption → text` 路径 + **低置信回退到 vision** 的安全网 |

---

## 4. 补齐计划

### P0 — Score 化：把 `(Value, Score)` 输出契约补回来 ⭐ 最高优先

**为什么最优先**：router / cascade / 阈值 BO / 验证集 quality 度量全部依赖它；而且底座**已经算出了
score**，只是包装层丢了 —— 改动最小、收益最大。

**做法**：保留现有签名不动（生成过的 program 不能碎），并列新增 `*_detail` 变体。

```python
# vadar/predefined.py 新增
def classify_detail(image, options, template="a photo of {}"):
    """OpImgCls: -> (label, score). score 是 CLIP softmax 后的置信度，跨行可比。"""

def verify_detail(image, prop):
    """-> (bool, score)。"""

def detect_detail(image, object_prompt, min_conf=0.25):
    """OpImgObj: -> [{"image": patch, "label": str, "box": (l,t,r,b) 绝对像素, "score": float}]"""

def ocr_detail(image, min_conf=0.0):
    """OpImgOCR: -> [{"text": str, "box": (l,t,r,b) 绝对像素, "score": float}]
    （直接暴露已有的 ImagePatch.read_text_boxes）"""

def best_ocr_match_detail(image, options, cutoff=0.6):
    """-> (value, score)，score = 模糊匹配相似度，"none" 时为 0.0。"""
```

同时：`ImagePatch` 增加 `.box` 的只读访问器，让 `regions_grid` / `detect` 产出的子图能报出自己的
`RegIdx` + `BBox`（论文 `OpImgRegion` 的输出 schema 要求）。

**落点**：`vadar/predefined.py`（主要）、`imagepatch.py`（少量 getter）、
`vadar/predefined.py::MODULES_SIGNATURES`（文档块必须同步，否则 agent 看不见）。

**验收**：
- 每个 image 算子调用能写进 `trace_<q>.json`，带 `score`；
- 对任一 attribute 能跑 theta 扫描，画出 score–F1 曲线；
- `tests/` 增加：`classify_detail` 的 score 与 `semvision.clip_classify` 第二返回值一致。

**风险**：`MODULES_SIGNATURES` 变长会稀释 agent 注意力 —— 用「同名 detail 后缀 + 一句话说明何时用
detail」控制篇幅，不要把两套 API 平铺成 22 条。

---

### P1 — 新算子（非 VLM 部分），与 P0 并行

按「SemBench 能用上 × 实现成本」排序：

#### P1.1 `OpImgEmbed` + 向量检索（最高价值）
```python
def embed(image) -> list[float]          # CLIP 图像向量（复用 ClipEncoder，带 patch-level 缓存）
def embed_text(text) -> list[float]      # 文本向量（论文 OpTxtImgSim 落地成 OpImgEmbed→OpTxtEmbed）
def topk_similar(query, candidates, k=5) -> list[(idx, score)]   # 向量 top-k
```
解锁：image–image join / dedup / rank 的**向量化**执行（现在 `pair_score` 是 O(N·M) 逐对 CLIP 调用），
以及 E-Commerce Plan 2 的 region 对齐。底座 `semvision.embed_corpus` 已存在。

#### P1.2 `OpImgCls` 多标签 + domain backend
```python
def classify_multi(image, options, thresh=0.5) -> list[(label, score)]   # 包 clip_multilabel（已存在）
def domain_classify(image, model_id, labels, threshold=0.5) -> (str, float)  # 包 XrayClassifier（已存在）
```
纯暴露工作，无新模型。Healthcare 场景（x_ray / skin）直接受益。

#### P1.3 `OpImgRegion` 升级为真正的 Structural 算子
分两步，**先做零新依赖的那步**：
```python
def regions_propose(image, method="contour", max_regions=8) -> list[patch]
    # method="contour": 纯 CV（阈值 + 连通域 / saliency），零新模型，先补 RegIdx + BBox
    # method="sam" | "dino": 可选依赖，装了才可用，补 Mask 与开放词表提案
```
E-Commerce「一张图多件衣服」和 CarDamage「局部损伤」都需要这个。Mask 后置到 P2。

#### P1.4 `OpImgObj` 开放词表
现在 `find()` 遇到 COCO-80 外的类名直接返回 `[]` 并 warn —— 对 SemBench 的
wildlife species / car parts 是硬伤。补一个 open-vocab 检测器（OWL-ViT 或 GroundingDINO，
后者也正好把 tier `dino` 的 stub 填上）作为 `detect` 的第二实现，YOLO 命中词表时仍走 YOLO。

#### P1.5 `OpImgKeypt`（低优先）
YOLOv8-pose 一行接入，但 SemBench 现有 image query 无明确需求。**排在 P1 末尾，可延后**。

#### P1.6 `OpImgSceneRel`（暂缓）／`OpImgEdit`（非目标）
SceneRel 需要 SGG 模型或 VLM 结构化输出，成本高、需求弱 —— 记录为已知缺口，本轮不做。
Edit 属于生成任务，与查询处理无关，**明确非目标**。

---

### P2 — VLM 受控放开：`OpImgVQA` / `OpImgCap` 算子化 + 计费

**目标**：让论文的 CarDamage Q8（coarse scan → selective verifier）和 E-Commerce Plan 3
（闭式验证）在生成的 program 里可表达，同时**不让 agent 无节制烧 VLM**。

```python
def vqa(image, question, choices=None) -> (answer, score)
    """OpImgVQA。内部走 semruntime.vlm_answer → semvqa（guided decoding + logprob score）。
    无 endpoint / 超预算时返回 ("none", 0.0) 并记 METER.skipped —— 语义与现有 residual 一致。
    强烈建议传 choices：闭式回答比开放式抽取稳得多（论文 Plan 3 的核心观察）。"""

def caption(image) -> str
    """OpImgCap。只读 orchestrator 预物化的 caption 列，program 内不触发 VLM（零成本）。
    缺列时返回 ""。这是 Template 2 跨模态代理的入口。"""
```

**guard 改造**（`orchestrator.mjs`）：
从「**禁止 import `semvqa|semcaption`**」改为「**禁止裸调 endpoint**」：
- ✅ 允许：`semruntime` 的 `vlm_answer` / `vlm_judge` / `predefined.vqa` / `predefined.caption`
- ❌ 仍禁：`openai` / `requests` / `httpx` / `urllib` / `socket` / `subprocess` / 裸 `--endpoint` 参数
- 即 guard 的语义从「零 VLM」变成「**所有 VLM 调用必须经过计量层**」

**计费**（新增，必须与放开同时落地，否则等于没有约束）：
- `semruntime.METER` 增加 per-query 预算：`SEMDB_VLM_BUDGET`（默认按语料行数的一个比例，如 20%）；
- 超预算后 `vqa()` 直接返回 `("none", 0.0)` 并计 `skipped`，**不抛异常**（program 语义不碎）；
- `trace_<q>.json` 记录 `vlm_calls / vlm_budget / skipped / mean_score`，喂给
  `FEEDBACK_LOOP_IMPL.md` 的 cost 维度反馈。

**验收**：
- 复现论文 Q8 pattern：`classify_detail` 粗筛 → 只对 positives 跑 `vqa` 验证，
  VLM 调用量 ≤ 全量的 30%，F1 优于纯 `classify_detail`；
- 关掉 endpoint 时全部退化为 skip，跑通不报错（回归现有离线行为）。

---

### P3 — 多实现 + 数据感知 router（对齐论文 §5.1）

前置：P0（score）+ P1（多个可选 backend）+ P2（VLM 作为最强实现）。

1. **填 stub**：`semvision._run_attr` 的 `dino`（P1.4 的 open-vocab）/ `distilled`（小模型）/
   `vlm`（P2 的 semvqa）三档接上，让 tier 契约名副其实。
2. **算子 → 实现列表**：给每个逻辑算子登记有序实现集
   `K = [symbolic, specialized, general-purpose]`（成本递增）。
3. **router**：论文的 cheap non-inferential feature（分辨率 / 宽高比 / 简单像素统计，
   我们可再加 OCR 文本长度、CLIP margin）→ difficulty score → **percentile bucketing**
   （rank 百分位 + softmax 参数化切点，避免绝对阈值对分布漂移敏感）→ 分派到第 k 个实现。
4. **调参**：切点 + 各实现阈值交给本分支已有的验证集跑 BO
   （直接对接 `VALSET_GUARANTEE_IMPL.md` 的采样+LLM标注验证集，以及
   `FEEDBACK_LOOP_IMPL.md` 的 quality/cost/latency 反馈）。

这一期是把「算子库」变成「**可调优的物理层**」，也是论文相对 LOTUS/Palimpzest 拿到
165.7× latency / 310.3× cost 的真正来源。

---

### P4 — 两个 alternative-generation template（可选，收尾）

- **Template 1 合并算子**：`normalize_union(results, key=...)` —— 分区抽取后的去重/投票/坐标归一。
  `regions_grid` 已经把 box 报成绝对坐标，跨分区标识符是保住的，补一个合并函数即可。
- **Template 2 安全网**：`ocr_detail` 空或低分 → 自动回退到 `vqa`/`classify_detail`
  （论文明确点名的 safeguard）。

---

## 5. 优先级汇总

| 期 | 内容 | 依赖 | 相对成本 | 解锁的论文能力 |
|---|---|---|---|---|
| ✅ **P0** | 全算子 `(Value, Score)` 化 + BBox/RegIdx 暴露 | 无 | 小（底座已有） | cascade / router / BO 的**前置** |
| ✅ **P1.1** | `OpImgEmbed` + top-k 检索 | P0 | 小 | 向量化 join/rank，Plan 2 region 对齐 |
| ✅ **P1.2** | multilabel + domain backend 暴露 | P0 | 极小（纯暴露） | Healthcare 场景 |
| ✅ **P1.3** | `OpImgRegion` 区域提案（contour 先行） | P0 | 中 | E-Commerce / CarDamage 局部抽取 |
| ✅ **P1.4** | `OpImgObj` 开放词表 | — | 中（新模型） | Wildlife species / car parts |
| **P1.5** | `OpImgKeypt` | — | 小 | 无强需求，可延后 |
| **P2** | `OpImgVQA`/`OpImgCap` 算子化 + guard 改造 + 计费 | P0 | 中 | Q8 selective verifier、Plan 3 闭式验证 |
| **P3** | 多实现 + percentile router + BO | P0/P1/P2 | 大 | 论文的核心性能来源 |
| **P4** | Template 1/2 的合并与安全网 | P3 | 小 | 分区抽取 / 跨模态代理 |
| — | `OpImgSceneRel` | — | 大 | **暂缓**，记为已知缺口 |
| — | `OpImgEdit` | — | — | **非目标** |

---

## 6. 风险与注意

1. **API 面积膨胀 vs agent 注意力**。`MODULES_SIGNATURES` 是直接喂给 Signature/API/Program
   三个 agent 的 prompt。从 11 条涨到 20+ 条会显著稀释注意力。对策：detail 变体用统一后缀 +
   一句话「何时用 detail」的规则，而不是逐条铺开文档；必要时按 query 的模态/意图**裁剪**注入的算子子集
   （论文本身也是把 catalog 放进 prompt 做 in-context learning，同样面临这个问题）。
2. **向后兼容**。已经生成并跑通的 `solve_<q>.py` / `compiled_<q>.py` 调的是旧签名，
   P0 必须**只增不改**。
3. **guard 放开的安全边界**。P2 一旦允许 `semruntime` 进入生成代码，"离线可复现" 这条性质就没了。
   计费 + trace 记录必须**与放开同一次落地**，否则 cost 维度的实验数据不可信。
4. **新模型依赖**（OWL-ViT / GroundingDINO / SAM / YOLO-pose）都要走**可选依赖 + lazy import**，
   缺失时降级并 warn，不能让没装的环境直接跑不起来（`ImagePatch._ocr` 的 lazy easyocr 是现成范式）。
5. **score 的可比性**。CLIP softmax 分数、YOLO 置信度、OCR 置信度、VQA logprob 分数**量纲不同**，
   跨算子直接比会出错。P3 的 percentile bucketing 正好回避了这点（论文选百分位而非绝对阈值就是这个原因）
   —— 在 P0 的文档里就要写清「score 只在同一算子内跨行可比」。

## 7. 开放问题

- P1.3 的区域提案先用纯 CV contour 还是直接上 SAM？（成本 vs 质量；建议先 contour 打通接口，
  SAM 作为 `method="sam"` 的可选后端）
- P2 的 VLM 预算按「语料比例」还是「绝对调用数」？需要先在一个 scenario 上量一下 Q8 pattern 的实际
  升级率再定。
- `OpImgSceneRel` 是否真的可以长期不做 —— 取决于后续是否引入需要空间关系的 query
  （"X 在 Y 左边"）。目前 SemBench image query 里没有。
