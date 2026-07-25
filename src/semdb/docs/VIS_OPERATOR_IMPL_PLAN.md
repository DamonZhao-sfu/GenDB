# VIS Operator P0 + P1.1–P1.4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 semdb VADAR 视觉算子库补齐到论文 CADENZA 的 image 算子契约 —— 所有算子输出
`(Value, Score)`，并补上 `OpImgEmbed` / multilabel+domain 后端 / `OpImgRegion` 区域提案 /
`OpImgObj` 开放词表。

**Architecture:** 三层不变 —— `semvision.py`（无状态后端函数，吃 path 或 PIL region）→
`imagepatch.py`（`ImagePatch` 方法，带 box 偏移与 encoder 缓存）→ `vadar/predefined.py`
（暴露给 agent 的自由函数 + `MODULES_SIGNATURES` 文档块）。新增能力一律**只增不改**已有签名；
所有 `*_detail` 变体返回 score，旧签名改为在其上取第一个返回值。

**Tech Stack:** Python 3.12（conda env `gendb`）、numpy、PIL、transformers（CLIP / OWLv2）、
ultralytics（YOLO）、pytest。

## Global Constraints

- **只增不改**：已生成的 `solve_<q>.py` / `compiled_<q>.py` 调用旧签名，任何已有函数的
  参数顺序、默认值、返回类型都不得改变。
- **零硬依赖新增**：OWLv2 走 lazy import + try/except 降级；contour 区域提案纯 numpy 实现，
  不引入 scipy/opencv。
- **score 语义**：score 只在**同一算子内跨行可比**。CLIP softmax、YOLO 置信度、OCR 置信度、
  difflib ratio 量纲不同，禁止跨算子直接比较 —— 这条必须写进 `MODULES_SIGNATURES`。
- **离线约束不变**：本计划不碰 `orchestrator.mjs` 的 offline guard（那是 P2）。新增函数
  一律不得引入 endpoint / 网络 / VLM。`tests/test_vadar_offline_guard.mjs` 必须保持绿。
- **API 面积单一真源**：新增函数必须同时出现在 `predefined.PREDEFINED_API`（自动派生）和
  `MODULES_SIGNATURES`，由 Task 1 的不变式测试强制。
- 测试命令统一：`python3 -m pytest tests/<file> -v`（当前 shell 的 python3 已是 gendb env）。
- 提交信息用 conventional commits（`feat:` / `test:` / `refactor:`）。

---

## File Structure

| 文件 | 职责 | 本计划的改动 |
|---|---|---|
| `semvision.py` | 无状态视觉后端（CV / CLIP / 检测器 / domain） | +`embed_image` +`embed_text` +`topk_similar` +`propose_region_boxes` +`_label_4c` +`OwlDetector` +`get_open_detector` +`detect_open_boxes` |
| `imagepatch.py` | `ImagePatch`：region 语义 + ctx 缓存 | +`bbox` +`classify_detail` +`verify_detail` +`find_detail` +`best_ocr_match_detail` +`embed` +`topk_similar` +`topk_text` +`classify_multi` +`domain_classify` +`propose_regions` +`find_open`；`classify`/`verify_property`/`find`/`best_ocr_match` 改为 detail 的薄包装 |
| `vadar/predefined.py` | agent 可见的自由函数 + 文档块 | +12 个包装函数 +`PREDEFINED_API` 自动派生表 + `MODULES_SIGNATURES` 同步 |
| `vadar/run.py` | 参考跑通器 | `_build_extract` 的硬编码名单换成 `PREDEFINED_API` |
| `agents/vadar-{program,solver,api}/user-prompt.md`、`imagepatch_prompt.md` | agent prompt 的 API 广告 | 同步新函数（Task 8） |
| `tests/test_predefined_api_surface.py` | **新** 不变式：广告的 API 必须真实存在 | 新建 |
| `tests/test_imagepatch_detail.py` | **新** P0 score 化 | 新建 |
| `tests/test_imagepatch_embed_topk.py` | **新** P1.1 | 新建 |
| `tests/test_imagepatch_multi_domain.py` | **新** P1.2 | 新建 |
| `tests/test_semvision_regions_propose.py` | **新** P1.3 | 新建 |
| `tests/test_imagepatch_open_vocab.py` | **新** P1.4 | 新建 |

---

## Task 1: 单一真源的 API 面积 + 不变式测试

先做这个。历史上 `predefined.detect` 被广告给 Program agent 但 `ImagePatch.find` 不存在，
每次调用 `AttributeError` 被 per-row guard 吞成静默的 `none` 列
（见 `tests/test_imagepatch_find_ocr.py` 顶部注释）。本任务把这类 bug 封死，之后才敢加 12 个函数。

**Files:**
- Modify: `vadar/predefined.py`（文件末尾追加 `PREDEFINED_API`）
- Modify: `vadar/run.py:73-77`（`_build_extract` 的硬编码名单）
- Test: `tests/test_predefined_api_surface.py`（新建）

**Interfaces:**
- Produces: `predefined.PREDEFINED_API: dict[str, callable]` —— 后续所有任务新增的函数会
  自动进入这张表，`vadar/run.py` 与不变式测试都读它。

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_predefined_api_surface.py`：

```python
"""不变式：广告给 agent 的每个 API 名字都必须真实存在且可调用。

历史 bug：`predefined.detect` 出现在 MODULES_SIGNATURES 里，但它依赖的
`ImagePatch.find` 根本没实现 —— 每次调用抛 AttributeError，被引擎的 per-row
guard 吞成静默的 'none' 列。这里把广告面与实现面锁在一起。
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imagepatch  # noqa: E402
from vadar import predefined  # noqa: E402

SEMDB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEF_RE = re.compile(r"^def\s+(\w+)\s*\(", re.M)
IMPORT_RE = re.compile(r"from vadar\.predefined import\s*(?:\(([^)]*)\)|([^\n]*))")
PATCH_CALL_RE = re.compile(r"\bpatch\.(\w+)\s*\(")

PROMPT_FILES = ["agents/vadar-program/user-prompt.md",
                "agents/vadar-solver/user-prompt.md",
                "agents/vadar-api/user-prompt.md"]


def _read(rel):
    return open(os.path.join(SEMDB, rel), encoding="utf-8").read()


def _imported_names(text):
    out = set()
    for m in IMPORT_RE.finditer(text):
        body = m.group(1) if m.group(1) is not None else m.group(2)
        out |= {t.strip() for t in body.replace("\n", " ").split(",") if t.strip()}
    return out


def test_documented_signatures_match_the_api_table_exactly():
    documented = set(DEF_RE.findall(predefined.MODULES_SIGNATURES))
    assert documented == set(predefined.PREDEFINED_API), (
        f"only documented: {documented - set(predefined.PREDEFINED_API)}; "
        f"only exported: {set(predefined.PREDEFINED_API) - documented}")


def test_every_exported_api_entry_is_callable():
    for name, fn in predefined.PREDEFINED_API.items():
        assert callable(fn), name


def test_runner_namespace_is_built_from_the_api_table():
    """vadar/run.py 不得再硬编码名字列表 —— 否则新函数对生成程序不可见。"""
    src = _read("vadar/run.py")
    assert "PREDEFINED_API" in src


def test_agent_prompt_imports_reference_only_real_functions():
    for rel in PROMPT_FILES:
        for name in _imported_names(_read(rel)):
            assert name in predefined.PREDEFINED_API, f"{rel} advertises missing {name!r}"


def test_imagepatch_prompt_advertises_only_real_methods():
    for name in set(PATCH_CALL_RE.findall(_read("imagepatch_prompt.md"))):
        assert hasattr(imagepatch.ImagePatch, name), f"imagepatch_prompt.md advertises missing {name!r}"
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `python3 -m pytest tests/test_predefined_api_surface.py -v`
Expected: FAIL — `AttributeError: module 'vadar.predefined' has no attribute 'PREDEFINED_API'`

- [ ] **Step 3: 实现 `PREDEFINED_API`**

在 `vadar/predefined.py` 的**文件最末尾**（`MODULES_SIGNATURES` 字符串之后）追加：

```python


def _public_api():
    """The agent-visible API surface, derived from this module rather than hand-listed —
    a new predefined function is exposed to the generated program the moment it is defined
    here, and `tests/test_predefined_api_surface.py` fails if it is not also documented in
    `MODULES_SIGNATURES`."""
    import inspect
    import sys as _sys
    mod = _sys.modules[__name__]
    return {n: o for n, o in vars(mod).items()
            if inspect.isfunction(o) and not n.startswith("_") and o.__module__ == __name__}


PREDEFINED_API = _public_api()
```

- [ ] **Step 4: 把 `vadar/run.py` 换成读 `PREDEFINED_API`**

`vadar/run.py:73-77`，把

```python
    ns = {name: getattr(predefined, name) for name in
          ("classify", "best_ocr_match", "dominant_colors", "verify_property", "score", "read_text",
           "detect", "crop", "regions_grid", "regions_center", "pair_score")}
```

替换为

```python
    ns = dict(predefined.PREDEFINED_API)      # single-sourced; see predefined._public_api
```

- [ ] **Step 5: 跑测试，确认通过**

Run: `python3 -m pytest tests/test_predefined_api_surface.py -v`
Expected: 5 passed

- [ ] **Step 6: 跑全量回归，确认没打破别的**

Run: `python3 -m pytest tests/ -q`
Expected: 与本任务前同样的 pass/skip 结果，无新增 failure

- [ ] **Step 7: 提交**

```bash
git add tests/test_predefined_api_surface.py vadar/predefined.py vadar/run.py
git commit -m "test(semdb): lock the predefined API surface to its implementation

vadar/run.py hardcoded the 11 predefined names, and three agent prompt files
each restate the import list, so an advertised-but-missing function silently
became a 'none' column. PREDEFINED_API is now derived from the module and the
invariant test asserts docs, exports and prompts agree."
```

---

## Task 2: P0 — `ImagePatch` 的 `(Value, Score)` 方法

**Files:**
- Modify: `imagepatch.py:87-112`（CLIP 方法）、`imagepatch.py:162-184`（`best_ocr_match`）、
  `imagepatch.py:186-207`（`find`）、`imagepatch.py:61-68`（新增 `bbox`）
- Test: `tests/test_imagepatch_detail.py`（新建）

**Interfaces:**
- Produces:
  - `ImagePatch.bbox -> tuple[int,int,int,int]`（绝对像素，整图时为 `(0,0,w,h)`）
  - `ImagePatch.classify_detail(options, template="a photo of {}") -> (str, float)`
  - `ImagePatch.verify_detail(prop, template="a photo of {}") -> (bool, float)`
  - `ImagePatch.find_detail(object_prompt, min_conf=0.25) -> list[dict]`，
    每项 `{"image": ImagePatch, "label": str, "box": (l,t,r,b) 绝对像素, "score": float}`
  - `ImagePatch.best_ocr_match_detail(options, cutoff=0.6, min_len=3) -> (str, float)`
- Consumes: `semvision.clip_classify`（已返回 `(label, prob)`）、`semvision.detect_boxes`
  （已返回 `[(label, conf, box)]`）、`ImagePatch.read_text_boxes`（已返回 text+box+score）

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_imagepatch_detail.py`：

```python
"""P0 — 每个 image 算子吐出论文 Table 1 的 (Value, Score) 输出 schema。

底座本来就算出了 score（clip_classify 返回 (label, prob)，detect_boxes 返回
(label, conf, box)），只是包装层扔掉了。没有 score 就没有 cascade / router /
阈值调优，所以这是整条线的前置。
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imagepatch  # noqa: E402


class FakeEncoder:
    """把每个 label 映射到一个固定向量，让 argmax 与 score 都是可预测的。"""

    def __init__(self, image_vec, label_vecs):
        self.iv = np.asarray(image_vec, np.float32)
        self.iv /= np.linalg.norm(self.iv)
        self.lv = {k: np.asarray(v, np.float32) / np.linalg.norm(v)
                   for k, v in label_vecs.items()}

    def encode_image(self, src, key=None):
        return self.iv

    def encode_text(self, labels, template="a photo of {}"):
        return np.stack([self.lv[str(l)] for l in labels])


class MockYolo:
    class _M:
        names = {0: "zebra", 1: "person"}

    def __init__(self, dets):
        self.model = self._M()
        self._d = dets

    def detect(self, img_path, min_conf=0.25):
        return [d for d in self._d if d[1] >= min_conf]


class MockOcr:
    def __init__(self, rows):
        self._rows = rows

    def readtext(self, _src, detail=1):
        return self._rows


def _img(tmp_path, size=(200, 100)):
    p = os.path.join(str(tmp_path), "x.png")
    Image.new("RGB", size, (255, 255, 255)).save(p)
    return p


def _ctx(**kw):
    base = {"encoder": None, "palette": None}
    base.update(kw)
    return base


# --- bbox (OpImgRegion 的 BBox 输出) ----------------------------------------

def test_bbox_is_the_full_frame_for_a_whole_image(tmp_path):
    p = imagepatch.ImagePatch(_img(tmp_path, (200, 100)), _ctx())
    assert p.bbox == (0, 0, 200, 100)


def test_bbox_of_a_crop_is_absolute(tmp_path):
    p = imagepatch.ImagePatch(_img(tmp_path, (200, 100)), _ctx())
    assert p.crop(0.5, 0.0, 1.0, 1.0).bbox == (100, 0, 200, 100)


# --- OpImgCls (Label, Score) ------------------------------------------------

def test_classify_detail_returns_label_and_score(tmp_path):
    enc = FakeEncoder([1, 0], {"cat": [1, 0], "dog": [0, 1]})
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(encoder=enc))
    label, score = p.classify_detail(["cat", "dog"])
    assert label == "cat"
    assert 0.0 <= score <= 1.0 and score > 0.5


def test_classify_still_returns_only_the_label(tmp_path):
    """向后兼容：已生成的程序调的是这个签名。"""
    enc = FakeEncoder([1, 0], {"cat": [1, 0], "dog": [0, 1]})
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(encoder=enc))
    assert p.classify(["cat", "dog"]) == "cat"


def test_verify_detail_returns_bool_and_score(tmp_path):
    enc = FakeEncoder([1, 0], {"a cat": [1, 0], "not a cat": [0, 1]})
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(encoder=enc))
    ok, score = p.verify_detail("a cat")
    assert ok is True and score > 0.5
    assert p.verify_property("a cat") is True


# --- OpImgObj (BBox, Label, Score) ------------------------------------------

def test_find_detail_returns_label_box_and_score(tmp_path):
    det = MockYolo([("zebra", 0.9, (10, 20, 60, 80))])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(detector=det))
    d = p.find_detail("zebra")[0]
    assert d["label"] == "zebra" and d["score"] == 0.9
    assert d["box"] == (10, 20, 60, 80)
    assert isinstance(d["image"], imagepatch.ImagePatch)
    assert d["image"].box == (10, 20, 60, 80)


def test_find_detail_boxes_are_absolute_inside_a_crop(tmp_path):
    det = MockYolo([("zebra", 0.9, (0, 0, 20, 20))])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(detector=det))
    d = p.crop(100, 50, 200, 100).find_detail("zebra")[0]
    assert d["box"] == (100, 50, 120, 70)


def test_find_detail_is_ordered_by_score(tmp_path):
    det = MockYolo([("zebra", 0.4, (0, 0, 10, 10)), ("zebra", 0.95, (20, 20, 30, 30))])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(detector=det))
    assert [d["score"] for d in p.find_detail("zebra")] == [0.95, 0.4]


def test_find_is_unchanged_and_still_returns_patches(tmp_path):
    det = MockYolo([("zebra", 0.9, (10, 20, 60, 80))])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(detector=det))
    hits = p.find("zebra")
    assert len(hits) == 1 and hits[0].box == (10, 20, 60, 80)


def test_find_detail_out_of_vocabulary_is_empty(tmp_path):
    det = MockYolo([("zebra", 0.9, (0, 0, 10, 10))])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(detector=det))
    imagepatch.ImagePatch._OOV_WARNED.discard("impala")
    assert p.find_detail("impala") == []


# --- OpImgOCR (Text, Score) -------------------------------------------------

def test_best_ocr_match_detail_scores_a_fuzzy_hit(tmp_path):
    ocr = MockOcr([([[0, 0], [1, 0], [1, 1], [0, 1]], "DELTA AIRLINES", 0.9)])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(ocr=ocr))
    val, score = p.best_ocr_match_detail(["Delta Air Lines", "United"])
    assert val == "Delta Air Lines" and score > 0.9      # difflib ratio, near-exact
    assert p.best_ocr_match(["Delta Air Lines", "United"]) == "Delta Air Lines"


def test_best_ocr_match_detail_scores_the_distinctive_token_path(tmp_path):
    """长文本模糊匹配必然失败，只有 distinctive-token 命中能救回来 —— 它的 score
    是命中率，与模糊匹配的 ratio 不是一个量纲，但对同一算子跨行仍可比。"""
    text = "welcome aboard delta lines flight 42 departing from gate b7"
    ocr = MockOcr([([[0, 0], [1, 0], [1, 1], [0, 1]], text, 0.9)])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(ocr=ocr))
    val, score = p.best_ocr_match_detail(["Delta Air Lines", "United"])
    assert val == "Delta Air Lines" and score == 1.0     # 'delta' + 'lines' both read


def test_best_ocr_match_detail_scores_a_miss_as_zero(tmp_path):
    ocr = MockOcr([([[0, 0], [1, 0], [1, 1], [0, 1]], "xyzzy qwerty", 0.9)])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(ocr=ocr))
    assert p.best_ocr_match_detail(["Delta Air Lines", "United"]) == ("none", 0.0)


def test_best_ocr_match_detail_on_empty_text_is_none(tmp_path):
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(ocr=MockOcr([])))
    assert p.best_ocr_match_detail(["Delta Air Lines"]) == ("none", 0.0)
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `python3 -m pytest tests/test_imagepatch_detail.py -v`
Expected: FAIL — `AttributeError: 'ImagePatch' object has no attribute 'bbox'` 等

- [ ] **Step 3: 实现**

3a. `imagepatch.py`，在 `size` property（第 61-68 行）之后插入 `bbox`：

```python
    @property
    def bbox(self):
        """This patch's box in ABSOLUTE image pixels — `(0, 0, w, h)` for a whole image.
        `OpImgRegion`'s BBox output; a region's RegIdx is its index in the producing list."""
        if self.box is not None:
            return self.box
        w, h = self._image_size()
        return (0, 0, w, h)
```

3b. 把 CLIP 块的**前半**（第 87-101 行：区块注释 + `classify` / `best_text_match` /
`verify_property`）替换为 detail-first 版本。**`score`（第 103 行）和 `pair_score`
（第 108 行）本来就返回 float，保持原样不动**：

```python
    # --- CLIP: (VALUE, Score) — the paper's R_Cls(Label, Score) output schema -----
    def classify_detail(self, options, template="a photo of {}"):
        """OpImgCls: (best-matching VALUE, confidence in [0,1]). The score is comparable
        ACROSS ROWS for this operator — not against another operator's score."""
        label, s = semvision.clip_classify(self._src(), list(options),
                                           self.ctx["encoder"], template, key=self._key)
        return label, float(s)

    def classify(self, options, template="a photo of {}"):
        """Best-matching option from a closed value space (enum or DB column values)."""
        return self.classify_detail(options, template)[0]

    def best_text_match(self, options, template="{}"):
        return self.classify(options, template)

    def verify_detail(self, prop, template="a photo of {}"):
        """(True iff the patch matches `prop` better than its negation, confidence of the
        WINNING side)."""
        label, s = semvision.clip_classify(self._src(), [prop, f"not {prop}"],
                                           self.ctx["encoder"], template, key=self._key)
        return label == prop, float(s)

    def verify_property(self, prop, template="a photo of {}"):
        """True iff the patch matches `prop` better than its negation."""
        return self.verify_detail(prop, template)[0]
```

（`score` 和 `pair_score` 本来就返回 float，不动。）

3c. 把 `best_ocr_match`（第 162-184 行）替换为 detail-first 版本：

```python
    def best_ocr_match_detail(self, options, cutoff=0.6, min_len=3):
        """Read the image text (OCR) and match it to the closest VALUE in a value space —
        returns (the field value, match strength in [0,1]); ("none", 0.0) when nothing
        matches. Wins over `classify` for wordmark logos (airline names). Strict, to avoid
        false positives on non-logo images: requires a strong fuzzy match OR that the
        option's DISTINCTIVE (non-generic) tokens actually appear in the OCR text."""
        import difflib
        text = self.read_text().lower().strip()
        if len(text) < min_len:
            return "none", 0.0
        by_lower = {o.lower(): o for o in options}
        m = difflib.get_close_matches(text, list(by_lower), n=1, cutoff=cutoff)
        if m:
            return by_lower[m[0]], float(difflib.SequenceMatcher(None, text, m[0]).ratio())
        toks = set(text.split())
        best, best_s = "none", 0.0
        for o in options:
            distinctive = [w for w in o.lower().split() if w not in self._GENERIC_TOKENS]
            if not distinctive:
                continue
            hit = sum(1 for w in distinctive if w in toks) / len(distinctive)
            if hit >= 0.6 and hit > best_s:            # its distinctive name must be read
                best, best_s = o, hit
        return best, float(best_s)

    def best_ocr_match(self, options, cutoff=0.6, min_len=3):
        """The matched VALUE only — see `best_ocr_match_detail` for the match strength."""
        return self.best_ocr_match_detail(options, cutoff, min_len)[0]
```

3d. 把 `find`（第 189-207 行）拆成共享的行提取 + 两个视图：

```python
    def _detect_rows(self, object_prompt, min_conf=0.25):
        """[(label, conf, box in THIS patch's frame)], best first — [] for a class outside
        the detector's closed vocabulary (it warns rather than pretending)."""
        det = self.ctx.get("detector")
        if det is None:
            det = self.ctx["detector"] = semvision.get_detector()
        want = str(object_prompt).strip().lower()
        vocab = {c.lower(): c for c in semvision.detector_classes(det)}
        if vocab and want not in vocab:
            if want not in ImagePatch._OOV_WARNED:
                ImagePatch._OOV_WARNED.add(want)
                print(f"[imagepatch] find({object_prompt!r}): outside the detector's "
                      f"vocabulary ({len(vocab)} classes) — use classify/verify_property, "
                      f"or find_open for an open-vocabulary detector")
            return []
        cls = [vocab[want]] if vocab else [object_prompt]
        return semvision.detect_boxes(self._src(), cls, det, min_conf=min_conf)

    def find_detail(self, object_prompt, min_conf=0.25):
        """OpImgObj proper: [{"image": patch, "label", "box" (ABSOLUTE px), "score"}],
        most confident first — the paper's R_Obj(BBox, Label, Score)."""
        out = []
        for label, conf, box in self._detect_rows(object_prompt, min_conf):
            child = self._child(box)
            out.append({"image": child, "label": label, "box": child.box, "score": float(conf)})
        return out

    def find(self, object_prompt, min_conf=0.25):
        """Detected instances of `object_prompt` as SUB-PATCHES (YOLO), highest confidence
        first. `len(...)` counts, truthiness tests presence, and each element can be
        further classified / read. Returns [] for a class outside the detector's closed
        vocabulary (see `semvision.detector_classes`) — it warns rather than pretending."""
        return [d["image"] for d in self.find_detail(object_prompt, min_conf)]
```

（`_OOV_WARNED` 的类属性定义留在原位，不动。）

- [ ] **Step 4: 跑测试，确认通过**

Run: `python3 -m pytest tests/test_imagepatch_detail.py -v`
Expected: 14 passed

- [ ] **Step 5: 跑既有 imagepatch 回归**

Run: `python3 -m pytest tests/test_imagepatch_find_ocr.py tests/test_imagepatch_regions.py tests/test_imagepatch_compose.py -v`
Expected: 全绿（`test_imagepatch_compose.py` 若因缺数据 skip 属正常）

- [ ] **Step 6: 提交**

```bash
git add imagepatch.py tests/test_imagepatch_detail.py
git commit -m "feat(semdb): ImagePatch emits (Value, Score) for cls/obj/ocr

The backends already computed a confidence — clip_classify returns (label, prob)
and detect_boxes returns (label, conf, box) — but the wrappers dropped it, so no
threshold, cascade or router could be built on top. classify/verify_property/
find/best_ocr_match are now thin views over *_detail, keeping their signatures."
```

---

## Task 3: P0 — `predefined` 包装 + `MODULES_SIGNATURES` 同步

**Files:**
- Modify: `vadar/predefined.py`（函数区 + `MODULES_SIGNATURES`）
- Test: `tests/test_predefined_api_surface.py`（Task 1 的不变式测试自动覆盖，无需改）

**Interfaces:**
- Consumes: Task 2 的 `ImagePatch.{bbox, classify_detail, verify_detail, find_detail, best_ocr_match_detail}`
- Produces（自由函数，agent 可见）：
  - `classify_detail(image, options, template="a photo of {}") -> (str, float)`
  - `verify_detail(image, prop) -> (bool, float)`
  - `detect_detail(image, object_prompt, min_conf=0.25) -> list[dict]`
  - `ocr_detail(image, min_conf=0.0) -> list[dict]`
  - `best_ocr_match_detail(image, options) -> (str, float)`
  - `bbox(image) -> tuple[float, float, float, float]`

- [ ] **Step 1: 跑不变式测试，确认它会挡住不同步**

先只加函数不加文档，验证护栏有效。在 `vadar/predefined.py` 的 `pair_score` 之后加：

```python
def classify_detail(image, options, template="a photo of {}"):
    """OpImgCls with the paper's (Label, Score) output schema."""
    return image.classify_detail(options, template)
```

Run: `python3 -m pytest tests/test_predefined_api_surface.py::test_documented_signatures_match_the_api_table_exactly -v`
Expected: FAIL — `only exported: {'classify_detail'}`

- [ ] **Step 2: 补齐全部 6 个函数**

`vadar/predefined.py`，在 `pair_score` 之后（`MODULES_SIGNATURES` 之前）：

```python
# --- P0: the (Value, Score) views — the paper's Table 1 output schemas ----------

def classify_detail(image, options, template="a photo of {}"):
    """OpImgCls: (best-matching VALUE, confidence)."""
    return image.classify_detail(options, template)


def verify_detail(image, prop):
    """(True iff the image matches `prop`, confidence of the winning side)."""
    return image.verify_detail(prop)


def detect_detail(image, object_prompt, min_conf=0.25):
    """OpImgObj: one dict per instance — {"image", "label", "box", "score"}."""
    return image.find_detail(object_prompt, min_conf)


def ocr_detail(image, min_conf=0.0):
    """OpImgOCR: one dict per text box — {"text", "box", "score"}."""
    return image.read_text_boxes(min_conf)


def best_ocr_match_detail(image, options):
    """(the OCR-matched VALUE or "none", match strength)."""
    return image.best_ocr_match_detail(options)


def bbox(image):
    """This image's own box in ABSOLUTE pixels — (0, 0, w, h) for a whole image."""
    return image.bbox
```

- [ ] **Step 3: 同步 `MODULES_SIGNATURES`**

在 `MODULES_SIGNATURES` 字符串**开头**（`'''` 之后、第一个 `"""` 之前）插入 score 语义说明：

```
SCORES. Every `*_detail` variant returns the operator's confidence alongside its value.
A score is comparable ACROSS ROWS for the SAME primitive (so a threshold on it is
meaningful, and a cheap pass can hand only its low-score rows to a heavier one), but
NOT across different primitives — CLIP probabilities, detector confidences and OCR
match strengths are on different scales. Prefer the plain variant when you only need
the value; reach for `_detail` when you need to gate, rank or cascade.
```

在字符串**末尾**（结尾 `'''` 之前）追加 6 个文档块：

```
"""
OpImgCls with a score: same as `classify`, but returns (VALUE, confidence in [0,1]).
Args:
    image (image): the image.
    options (list): the value space.
    template (string): prompt template with one "{}", default "a photo of {}".
Returns:
    tuple: (best-matching option value, confidence).
"""
def classify_detail(image, options, template="a photo of {}"):

"""
Same as `verify_property`, but also returns the confidence of the winning side.
Args:
    image (image): the image.
    prop (string): the property, e.g. "a damaged car".
Returns:
    tuple: (bool, confidence in [0,1]).
"""
def verify_detail(image, prop):

"""
OpImgObj proper: one entry PER DETECTED INSTANCE, most confident first, each a dict
{"image": the sub-image, "label": the class name, "box": (left, top, right, bottom) in
ABSOLUTE image pixels, "score": the detector confidence}. Use over `detect` when you need
the box or the score (to rank instances, or to keep only confident ones).
Args:
    image (image): the image.
    object_prompt (string): simple object name from the detector's closed vocabulary.
    min_conf (float): drop detections below this confidence (default 0.25).
Returns:
    list: dicts as above (empty if none).
"""
def detect_detail(image, object_prompt, min_conf=0.25):

"""
OpImgOCR proper: one entry per detected text box — {"text", "box": (left, top, right,
bottom) in ABSOLUTE image pixels, "score"}. Use over `read_text` when WHERE the text sits
matters (a label in a corner, a caption strip) or to drop unreliable reads.
Args:
    image (image): the image.
    min_conf (float): drop boxes below this OCR confidence (default 0.0 = keep all).
Returns:
    list: dicts as above.
"""
def ocr_detail(image, min_conf=0.0):

"""
Same as `best_ocr_match`, but also returns how strongly the OCR text matched. A miss is
("none", 0.0), so the score doubles as a gate for non-logo images.
Args:
    image (image): the image.
    options (list): the value space.
Returns:
    tuple: (matched value or "none", match strength in [0,1]).
"""
def best_ocr_match_detail(image, options):

"""
This image's own box in ABSOLUTE pixels, (0, 0, width, height) for a whole image and the
region's box for a sub-image from crop / regions_grid / detect / regions_propose. Pair it
with the region's index in the list it came from to identify a region.
Args:
    image (image): the image.
Returns:
    tuple: (left, top, right, bottom).
"""
def bbox(image):
```

- [ ] **Step 4: 跑测试，确认通过**

Run: `python3 -m pytest tests/test_predefined_api_surface.py -v`
Expected: 5 passed

- [ ] **Step 5: 确认离线守卫仍绿**

Run: `node --test tests/test_vadar_offline_guard.mjs`
Expected: 全绿（新函数不含 endpoint / 网络字样）

- [ ] **Step 6: 提交**

```bash
git add vadar/predefined.py
git commit -m "feat(semdb): expose the (Value, Score) operators to the VADAR agents

Adds classify_detail / verify_detail / detect_detail / ocr_detail /
best_ocr_match_detail / bbox, and documents in MODULES_SIGNATURES that a score
is comparable across rows for one primitive but never across primitives."
```

---

## Task 4: P1.1 — `OpImgEmbed` + 向量 top-k

现在 image–image join 走 `pair_score` 逐对调 CLIP，是 O(N·M) 次编码。本任务给出向量化路径，
同时补上论文 `OpTxtImgSim → OpImgEmbed → OpTxtEmbed` 的落地形式。

**Files:**
- Modify: `semvision.py:184-201`（Latent 区块）
- Modify: `imagepatch.py`（新增 3 个方法）、`imagepatch.py:23`（顶部加 `import numpy as np`）
- Modify: `vadar/predefined.py`
- Test: `tests/test_imagepatch_embed_topk.py`（新建）

**Interfaces:**
- Produces:
  - `semvision.embed_image(src, encoder, key=None) -> np.ndarray[D]`（单位范数）
  - `semvision.embed_text(text, encoder, template="{}") -> np.ndarray[D]`
  - `semvision.topk_similar(query_vec, matrix, k=5) -> list[(int, float)]`
  - `ImagePatch.embed() -> list[float]`
  - `ImagePatch.topk_similar(others, k=5) -> list[(int, float)]`
  - `ImagePatch.topk_text(texts, k=5, template="a photo of {}") -> list[(str, float)]`
  - `predefined.embed(image)` / `predefined.topk_similar(image, others, k=5)` /
    `predefined.topk_text(image, texts, k=5)`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_imagepatch_embed_topk.py`：

```python
"""P1.1 — OpImgEmbed（向量）+ 向量化 top-k。

pair_score 是逐对 CLIP 调用；一个 image-image join / rank 需要的是「一次编码、
一次矩阵乘」。topk_text 则把 classify 的 argmax 放宽成带分数的候选集，
这正是 cascade 的候选来源。
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imagepatch  # noqa: E402
import semvision  # noqa: E402


class VecEncoder:
    """按 (path, box) 的 box 是否为 None 分别给向量，够测 region 语义。"""

    def __init__(self, vec, label_vecs=None):
        self.v = np.asarray(vec, np.float32) / np.linalg.norm(vec)
        self.lv = {k: np.asarray(x, np.float32) / np.linalg.norm(x)
                   for k, x in (label_vecs or {}).items()}

    def encode_image(self, src, key=None):
        return self.v

    def encode_text(self, labels, template="a photo of {}"):
        return np.stack([self.lv[str(l)] for l in labels])


def _img(tmp_path, name="x.png"):
    p = os.path.join(str(tmp_path), name)
    Image.new("RGB", (40, 40), (255, 255, 255)).save(p)
    return p


# --- semvision 层 -----------------------------------------------------------

def test_topk_similar_ranks_and_rescales_to_unit_interval():
    q = np.asarray([1.0, 0.0], np.float32)
    m = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], np.float32)
    rows = semvision.topk_similar(q, m, k=2)
    assert [i for i, _ in rows] == [0, 1]
    assert rows[0][1] == 1.0
    assert abs(rows[1][1] - 0.5) < 1e-6


def test_topk_similar_on_an_empty_corpus_is_empty():
    assert semvision.topk_similar(np.asarray([1.0, 0.0], np.float32),
                                  np.zeros((0, 2), np.float32)) == []


def test_topk_similar_clamps_k_to_the_corpus_size():
    q = np.asarray([1.0, 0.0], np.float32)
    m = np.asarray([[1.0, 0.0]], np.float32)
    assert len(semvision.topk_similar(q, m, k=10)) == 1


def test_embed_image_and_embed_text_are_unit_norm():
    enc = VecEncoder([3.0, 4.0], {"cat": [1.0, 0.0]})
    assert abs(np.linalg.norm(semvision.embed_image("a.png", enc)) - 1.0) < 1e-6
    assert abs(np.linalg.norm(semvision.embed_text("cat", enc)) - 1.0) < 1e-6


# --- ImagePatch 层 ----------------------------------------------------------

def test_embed_returns_a_plain_float_list(tmp_path):
    p = imagepatch.ImagePatch(_img(tmp_path), {"encoder": VecEncoder([1.0, 0.0])})
    v = p.embed()
    assert isinstance(v, list) and all(isinstance(x, float) for x in v)
    assert abs(sum(x * x for x in v) - 1.0) < 1e-6


def test_topk_similar_ranks_other_patches_by_index(tmp_path):
    enc = VecEncoder([1.0, 0.0])
    q = imagepatch.ImagePatch(_img(tmp_path, "q.png"), {"encoder": enc})
    others = [imagepatch.ImagePatch(_img(tmp_path, f"o{i}.png"), {"encoder": enc})
              for i in range(3)]
    rows = q.topk_similar(others, k=2)
    assert len(rows) == 2
    assert all(0 <= i < 3 for i, _ in rows)
    assert all(s == 1.0 for _, s in rows)      # 同一编码器 -> 全同向


def test_topk_similar_on_no_candidates_is_empty(tmp_path):
    q = imagepatch.ImagePatch(_img(tmp_path), {"encoder": VecEncoder([1.0, 0.0])})
    assert q.topk_similar([], k=3) == []


def test_topk_text_returns_texts_with_scores_best_first(tmp_path):
    enc = VecEncoder([1.0, 0.0], {"cat": [1.0, 0.0], "dog": [0.0, 1.0],
                                  "bird": [-1.0, 0.0]})
    p = imagepatch.ImagePatch(_img(tmp_path), {"encoder": enc})
    rows = p.topk_text(["dog", "cat", "bird"], k=2)
    assert [t for t, _ in rows] == ["cat", "dog"]
    assert rows[0][1] > rows[1][1]
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `python3 -m pytest tests/test_imagepatch_embed_topk.py -v`
Expected: FAIL — `AttributeError: module 'semvision' has no attribute 'topk_similar'`

- [ ] **Step 3: 实现 semvision 侧**

`semvision.py`，在 `img_pair_score`（第 188-194 行）之后插入：

```python
def embed_image(src, encoder, key=None):
    """OpImgEmbed: the unit-norm image vector for one image or region. `pair_score` and
    `topk_similar` both consume this space, so a corpus can be encoded once and reused."""
    return np.asarray(_encode_image(encoder, src, key), np.float32)


def embed_text(text, encoder, template="{}"):
    """OpTxtEmbed on the image side: the unit-norm text vector in the SAME space as
    `embed_image`. This is how the paper grounds OpTxtImgSim into OpImgEmbed→OpTxtEmbed."""
    return np.asarray(encoder.encode_text([str(text)], template)[0], np.float32)


def topk_similar(query_vec, matrix, k=5):
    """Top-k rows of an [N, D] unit-norm `matrix` by cosine against `query_vec`, rescaled
    to [0,1] on the same scale as `img_pair_score`/`clip_match`. Returns [(row index,
    score)], best first — one matmul instead of N pairwise encoder calls."""
    q = np.asarray(query_vec, np.float32)
    m = np.asarray(matrix, np.float32)
    if m.size == 0:
        return []
    sims = np.clip((m @ q + 1.0) / 2.0, 0.0, 1.0)
    k = max(0, min(int(k), int(sims.shape[0])))
    order = np.argsort(-sims, kind="stable")[:k]
    return [(int(i), float(sims[i])) for i in order]
```

- [ ] **Step 4: 实现 ImagePatch 侧**

4a. `imagepatch.py` 顶部，把 `import semvision`（第 23 行）改为：

```python
import numpy as np

import semvision
```

（numpy 已是 semvision 的硬依赖，这里不引入新依赖。）

4b. 在 `pair_score`（第 108-112 行）之后插入：

```python
    # --- Latent: OpImgEmbed + vectorized top-k ------------------------------
    def embed(self):
        """OpImgEmbed: this patch's unit-norm vector as a plain float list (JSON-safe, so
        it can be materialized into a column)."""
        return [float(v) for v in
                semvision.embed_image(self._src(), self.ctx["encoder"], self._key)]

    def topk_similar(self, others, k=5):
        """Rank other patches against this one by IMAGE-IMAGE similarity, returning
        [(index into `others`, score)] best first. The vectorized form of `pair_score` —
        encodes each candidate once, then a single matmul. Use for an image-to-image
        join, dedup, or top-k."""
        others = list(others)
        q = semvision.embed_image(self._src(), self.ctx["encoder"], self._key)
        if not others:
            return []
        mat = np.stack([semvision.embed_image(o._src(), self.ctx["encoder"], o._key)
                        for o in others])
        return semvision.topk_similar(q, mat, k)

    def topk_text(self, texts, k=5, template="a photo of {}"):
        """Rank candidate TEXTS against this patch, returning [(text, score)] best first.
        `classify` keeps only the argmax; this keeps the top-k WITH scores — the candidate
        set a cascade needs before a heavier verifier runs on a few options."""
        texts = [str(t) for t in texts]
        if not texts:
            return []
        enc = self.ctx["encoder"]
        q = semvision.embed_image(self._src(), enc, self._key)
        rows = semvision.topk_similar(q, enc.encode_text(texts, template), k)
        return [(texts[i], s) for i, s in rows]
```

- [ ] **Step 5: 跑测试，确认通过**

Run: `python3 -m pytest tests/test_imagepatch_embed_topk.py -v`
Expected: 8 passed

- [ ] **Step 6: 加 predefined 包装 + 文档块**

`vadar/predefined.py`，在 P0 区块之后追加：

```python
# --- P1.1: Latent — OpImgEmbed and vectorized ranking --------------------------

def embed(image):
    """OpImgEmbed: the image's dense vector."""
    return image.embed()


def topk_similar(image, others, k=5):
    """Vectorized OpImgPairScore: rank `others` against `image`."""
    return image.topk_similar(others, k)


def topk_text(image, texts, k=5):
    """Rank candidate texts against the image; the top-k with scores."""
    return image.topk_text(texts, k)
```

`MODULES_SIGNATURES` 末尾追加：

```
"""
OpImgEmbed: encodes the image into a dense vector (a plain list of floats, unit-norm).
Two vectors are compared by `pair_score`-style cosine; use this when you want to encode
once and compare many times, or to materialize a vector column.
Args:
    image (image): the image.
Returns:
    list: the vector.
"""
def embed(image):

"""
Ranks OTHER images against this one by image-to-image similarity — the vectorized form of
`pair_score` (one encode per candidate, then one matmul, instead of a pairwise call per
comparison). Use for an image-to-image join, dedup or top-k.
Args:
    image (image): the query image.
    others (list): candidate images.
    k (int): how many to keep (default 5).
Returns:
    list: (index into `others`, score in [0,1]) tuples, best first.
"""
def topk_similar(image, others, k=5):

"""
Ranks candidate TEXTS against the image and keeps the top-k WITH scores. `classify` keeps
only the single best option; use this when you want a short candidate list out of a large
value space (e.g. narrow 135 airline names down to 5, then verify those 5 more carefully).
Args:
    image (image): the image.
    texts (list): candidate strings.
    k (int): how many to keep (default 5).
Returns:
    list: (text, score in [0,1]) tuples, best first.
"""
def topk_text(image, texts, k=5):
```

- [ ] **Step 7: 跑不变式 + 全量**

Run: `python3 -m pytest tests/test_predefined_api_surface.py tests/test_imagepatch_embed_topk.py tests/test_semvision_embed.py -v`
Expected: 全绿

- [ ] **Step 8: 提交**

```bash
git add semvision.py imagepatch.py vadar/predefined.py tests/test_imagepatch_embed_topk.py
git commit -m "feat(semdb): OpImgEmbed + vectorized top-k over images and texts

pair_score encodes both sides on every comparison, so an image-image join cost
O(N*M) encoder calls. embed/topk_similar encode once and rank with one matmul;
topk_text turns classify's argmax into a scored candidate set for cascades."
```

---

## Task 5: P1.2 — CLIP multilabel + domain 专家模型

两者的后端 `semvision.clip_multilabel` 和 `semvision.domain_classify` **都已实现**，只是
`ImagePatch` 和 `predefined` 没有包 —— agent 够不着。本任务是纯暴露，无新模型。

**Files:**
- Modify: `imagepatch.py`（2 个方法）
- Modify: `vadar/predefined.py`
- Test: `tests/test_imagepatch_multi_domain.py`（新建）

**Interfaces:**
- Consumes: `semvision.clip_multilabel(img, labels, encoder, thresh, template, key)`（返回
  `(list[label], conf)`）、`semvision.domain_classify(img, model, positive_labels, threshold)`
  （返回 `("yes"|"no", score)`）、`semvision.get_domain_model(model_id)`
- Produces:
  - `ImagePatch.classify_multi(options, thresh=0.5, template="a photo of {}") -> (list[str], float)`
  - `ImagePatch.domain_classify(model_id, labels, threshold=0.5) -> (str, float)`
  - `predefined.classify_multi(image, options, thresh=0.5)` /
    `predefined.domain_classify(image, model_id, labels, threshold=0.5)`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_imagepatch_multi_domain.py`：

```python
"""P1.2 — 多标签 CLIP 与 domain 专家模型的暴露。

两个后端在 semvision 里早就实现了（clip_multilabel / domain_classify + XrayClassifier），
但只有 extractor-spec 的 tier 路径够得着；VADAR 生成的程序完全用不上。
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imagepatch  # noqa: E402


class MultiEncoder:
    def __init__(self, image_vec, label_vecs):
        self.iv = np.asarray(image_vec, np.float32)
        self.iv /= np.linalg.norm(self.iv)
        self.lv = {k: np.asarray(v, np.float32) / np.linalg.norm(v)
                   for k, v in label_vecs.items()}

    def encode_image(self, src, key=None):
        return self.iv

    def encode_text(self, labels, template="a photo of {}"):
        return np.stack([self.lv[str(l)] for l in labels])


class FakeDomainModel:
    def __init__(self, probs):
        self._p = probs
        self.calls = 0

    def probs(self, _src):
        self.calls += 1
        return self._p


def _img(tmp_path):
    p = os.path.join(str(tmp_path), "x.png")
    Image.new("RGB", (40, 40), (255, 255, 255)).save(p)
    return p


# --- multilabel -------------------------------------------------------------

def test_classify_multi_keeps_every_label_over_the_threshold(tmp_path):
    enc = MultiEncoder([1.0, 0.0], {"stripes": [1.0, 0.0], "spots": [0.99, 0.14],
                                    "plain": [-1.0, 0.0]})
    p = imagepatch.ImagePatch(_img(tmp_path), {"encoder": enc, "palette": None})
    labels, conf = p.classify_multi(["stripes", "spots", "plain"], thresh=0.5)
    assert set(labels) == {"stripes", "spots"}
    assert 0.0 <= conf <= 1.0


def test_classify_multi_can_return_nothing_but_still_scores(tmp_path):
    enc = MultiEncoder([1.0, 0.0], {"plain": [-1.0, 0.0]})
    p = imagepatch.ImagePatch(_img(tmp_path), {"encoder": enc, "palette": None})
    labels, conf = p.classify_multi(["plain"], thresh=0.9)
    assert labels == []
    assert 0.0 <= conf <= 1.0


# --- domain -----------------------------------------------------------------

def test_domain_classify_returns_yes_no_and_the_max_positive_prob(tmp_path):
    model = FakeDomainModel({"Pneumonia": 0.81, "Effusion": 0.10})
    ctx = {"encoder": None, "palette": None, "domain": {"fake:m": model}}
    p = imagepatch.ImagePatch(_img(tmp_path), ctx)
    assert p.domain_classify("fake:m", ["Pneumonia"], threshold=0.5) == ("yes", 0.81)
    assert p.domain_classify("fake:m", ["Effusion"], threshold=0.5) == ("no", 0.10)


def test_domain_model_is_cached_in_ctx_across_calls(tmp_path):
    model = FakeDomainModel({"Pneumonia": 0.9})
    ctx = {"encoder": None, "palette": None, "domain": {"fake:m": model}}
    p = imagepatch.ImagePatch(_img(tmp_path), ctx)
    p.domain_classify("fake:m", ["Pneumonia"])
    p.domain_classify("fake:m", ["Pneumonia"])
    assert model.calls == 2                     # 每次都推理
    assert ctx["domain"]["fake:m"] is model     # 但模型只加载一次


def test_domain_classify_creates_the_domain_cache_when_absent(tmp_path):
    """ctx 没有 'domain' 键时不能 KeyError —— 引擎并不总会预建它。"""
    ctx = {"encoder": None, "palette": None}
    p = imagepatch.ImagePatch(_img(tmp_path), ctx)
    loaded = {}

    def fake_get(model_id):
        loaded[model_id] = FakeDomainModel({"Pneumonia": 0.7})
        return loaded[model_id]

    import semvision
    real = semvision.get_domain_model
    semvision.get_domain_model = fake_get
    try:
        assert p.domain_classify("fake:m", ["Pneumonia"], threshold=0.5) == ("yes", 0.7)
    finally:
        semvision.get_domain_model = real
    assert "fake:m" in ctx["domain"]
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `python3 -m pytest tests/test_imagepatch_multi_domain.py -v`
Expected: FAIL — `AttributeError: 'ImagePatch' object has no attribute 'classify_multi'`

- [ ] **Step 3: 实现**

`imagepatch.py`，在 `verify_property` 之后（`score` 之前）插入：

```python
    def classify_multi(self, options, thresh=0.5, template="a photo of {}"):
        """OpImgCls, multilabel: (every option scoring over `thresh`, confidence). Use when
        the field is a SET (several attributes true at once), not a single enum value."""
        return semvision.clip_multilabel(self._src(), list(options), self.ctx["encoder"],
                                         thresh, template, key=self._key)
```

在 `dominant_colors`（CV 区块）之后插入：

```python
    # --- domain specialists (a task-tuned model, not a zero-shot one) --------
    def domain_classify(self, model_id, labels, threshold=0.5):
        """A domain-specialist classifier — e.g.
        `domain_classify("torchxrayvision:densenet121-res224-all", ["Pneumonia"])`.
        Returns ("yes"|"no", the max probability over `labels`). The model is loaded once
        and cached in `ctx`, so a corpus scan pays for it a single time."""
        cache = self.ctx.setdefault("domain", {})
        if model_id not in cache:
            cache[model_id] = semvision.get_domain_model(model_id)
        return semvision.domain_classify(self._src(), cache[model_id], list(labels), threshold)
```

- [ ] **Step 4: 跑测试，确认通过**

Run: `python3 -m pytest tests/test_imagepatch_multi_domain.py -v`
Expected: 5 passed

- [ ] **Step 5: 加 predefined 包装 + 文档块**

`vadar/predefined.py` 追加：

```python
# --- P1.2: more OpImgCls backends — multilabel and domain specialists ----------

def classify_multi(image, options, thresh=0.5):
    """OpImgCls, multilabel: every option over the threshold, with a confidence."""
    return image.classify_multi(options, thresh)


def domain_classify(image, model_id, labels, threshold=0.5):
    """A domain-specialist classifier (e.g. chest X-ray pathologies)."""
    return image.domain_classify(model_id, labels, threshold)
```

`MODULES_SIGNATURES` 追加：

```
"""
OpImgCls, multilabel: keeps EVERY option whose match clears `thresh`, not just the best
one. Use when the field is a SET (several attributes hold at once, e.g. damage types on
one car) rather than a single enum value — `classify` would force one winner.
Args:
    image (image): the image.
    options (list): the value space.
    thresh (float): keep options scoring at or above this (default 0.5).
Returns:
    tuple: (list of matching values, confidence in [0,1]).
"""
def classify_multi(image, options, thresh=0.5):

"""
Runs a DOMAIN-SPECIALIST classifier — a model trained for this exact domain, far more
accurate there than zero-shot CLIP. Currently available: chest X-ray pathologies via
model_id "torchxrayvision:densenet121-res224-all" (labels are pathology names such as
"Pneumonia", "Effusion", "Cardiomegaly"). Returns "yes" when the strongest listed label
clears `threshold`.
Args:
    image (image): the image.
    model_id (string): the specialist model, e.g. "torchxrayvision:densenet121-res224-all".
    labels (list): the positive labels to test for.
    threshold (float): decision threshold (default 0.5).
Returns:
    tuple: ("yes" or "no", the max probability over `labels`).
"""
def domain_classify(image, model_id, labels, threshold=0.5):
```

- [ ] **Step 6: 跑不变式，确认通过**

Run: `python3 -m pytest tests/test_predefined_api_surface.py tests/test_imagepatch_multi_domain.py -v`
Expected: 全绿

- [ ] **Step 7: 提交**

```bash
git add imagepatch.py vadar/predefined.py tests/test_imagepatch_multi_domain.py
git commit -m "feat(semdb): expose CLIP multilabel and domain specialists to the agents

clip_multilabel and domain_classify (torchxrayvision) were reachable only through
the extractor-spec tier path; a VADAR-generated program could not use either, so
set-valued fields collapsed to one enum value and X-ray queries fell back to CLIP."
```

---

## Task 6: P1.3 — `OpImgRegion` 区域提案（contour，零新依赖）

论文的 `OpImgRegion` 输出 `(RegIdx, BBox, Mask)`；我们只有网格切分。本任务补上**基于前景连通域的
区域提案** —— 纯 numpy，确定性，不引入 scipy/opencv/SAM。Mask 留到后续（`method="sam"` 现在
明确降级并告警，不假装支持）。

**Files:**
- Modify: `semvision.py`（新增区块，放在 `cv_dominant_colors` 之后）
- Modify: `imagepatch.py`（`regions_center` 之后新增 `propose_regions`）
- Modify: `vadar/predefined.py`
- Test: `tests/test_semvision_regions_propose.py`（新建）

**Interfaces:**
- Produces:
  - `semvision._label_4c(mask: np.ndarray[bool]) -> np.ndarray[int32]`（0 为背景）
  - `semvision.propose_region_boxes(src, max_regions=8, min_area_frac=0.01, size=128, tol=0.12) -> list[tuple[float,float,float,float]]`
    （**图像分数** `(l, t, r, b)`，各值在 `[0,1]`，按面积降序）
  - `ImagePatch.propose_regions(max_regions=8, min_area_frac=0.01, method="contour") -> list[ImagePatch]`
  - `predefined.regions_propose(image, max_regions=8, min_area_frac=0.01)`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_semvision_regions_propose.py`：

```python
"""P1.3 — OpImgRegion：前景连通域区域提案（确定性、model-free）。

论文的 OpImgRegion 输出 (RegIdx, BBox, Mask)；我们此前只有 regions_grid 这种
盲切。E-Commerce 一张图多件衣服、CarDamage 局部损伤，都需要按内容切而不是按网格切。
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imagepatch  # noqa: E402
import semvision  # noqa: E402


def _two_blobs(tmp_path, size=(200, 100)):
    """白底 + 左右两个分离的黑块。"""
    im = Image.new("RGB", size, (255, 255, 255))
    px = im.load()
    for x in range(10, 60):
        for y in range(20, 80):
            px[x, y] = (0, 0, 0)
    for x in range(130, 190):
        for y in range(20, 80):
            px[x, y] = (0, 0, 0)
    p = os.path.join(str(tmp_path), "blobs.png")
    im.save(p)
    return p


def _blank(tmp_path):
    p = os.path.join(str(tmp_path), "blank.png")
    Image.new("RGB", (100, 100), (255, 255, 255)).save(p)
    return p


# --- connected components ---------------------------------------------------

def test_label_4c_separates_disconnected_blobs():
    m = np.zeros((5, 7), bool)
    m[1:3, 1:3] = True
    m[1:3, 5:7] = True
    lab = semvision._label_4c(m)
    assert lab[0, 0] == 0                       # 背景是 0
    assert lab[1, 1] != lab[1, 5]               # 两个块不同标签
    assert len(set(lab[m].tolist())) == 2


def test_label_4c_joins_an_l_shape_into_one_component():
    m = np.zeros((4, 4), bool)
    m[1, 1:4] = True
    m[1:4, 1] = True
    lab = semvision._label_4c(m)
    assert len(set(lab[m].tolist())) == 1


def test_label_4c_on_an_empty_mask_is_all_background():
    assert semvision._label_4c(np.zeros((3, 3), bool)).max() == 0


# --- region proposal --------------------------------------------------------

def test_propose_region_boxes_finds_both_blobs(tmp_path):
    boxes = semvision.propose_region_boxes(_two_blobs(tmp_path), max_regions=8)
    assert len(boxes) == 2
    for l, t, r, b in boxes:
        assert 0.0 <= l < r <= 1.0 and 0.0 <= t < b <= 1.0
    lefts = sorted(l for l, _, _, _ in boxes)
    assert lefts[0] < 0.4 and lefts[1] > 0.5    # 一左一右


def test_propose_region_boxes_respects_max_regions(tmp_path):
    assert len(semvision.propose_region_boxes(_two_blobs(tmp_path), max_regions=1)) == 1


def test_propose_region_boxes_drops_specks_below_min_area(tmp_path):
    assert semvision.propose_region_boxes(_two_blobs(tmp_path), min_area_frac=0.9) == []


def test_propose_region_boxes_on_a_uniform_image_is_empty(tmp_path):
    assert semvision.propose_region_boxes(_blank(tmp_path)) == []


def test_propose_region_boxes_is_deterministic(tmp_path):
    p = _two_blobs(tmp_path)
    assert semvision.propose_region_boxes(p) == semvision.propose_region_boxes(p)


# --- ImagePatch 层 ----------------------------------------------------------

def test_propose_regions_returns_sub_patches_in_absolute_pixels(tmp_path):
    p = imagepatch.ImagePatch(_two_blobs(tmp_path), {"encoder": None, "palette": None})
    regions = p.propose_regions()
    assert len(regions) == 2
    for r in regions:
        assert isinstance(r, imagepatch.ImagePatch)
        l, t, rr, b = r.bbox
        assert 0 <= l < rr <= 200 and 0 <= t < b <= 100


def test_propose_regions_inside_a_crop_offsets_to_absolute(tmp_path):
    p = imagepatch.ImagePatch(_two_blobs(tmp_path), {"encoder": None, "palette": None})
    sub = p.crop(100, 0, 200, 100)              # 只含右边那块
    regions = sub.propose_regions()
    assert len(regions) == 1
    assert regions[0].bbox[0] >= 100            # 已偏移回整图坐标


def test_propose_regions_warns_and_falls_back_for_an_unavailable_method(tmp_path, capsys):
    p = imagepatch.ImagePatch(_two_blobs(tmp_path), {"encoder": None, "palette": None})
    regions = p.propose_regions(method="sam")
    assert len(regions) == 2                    # 降级到 contour，而不是假装支持
    assert "contour" in capsys.readouterr().out
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `python3 -m pytest tests/test_semvision_regions_propose.py -v`
Expected: FAIL — `AttributeError: module 'semvision' has no attribute '_label_4c'`

- [ ] **Step 3: 实现 semvision 侧**

`semvision.py`，在 `cv_dominant_colors`（第 117 行结束）之后、CLIP 区块之前插入：

```python
# ---------------------------------------------------------------------------
# ① Structural — OpImgRegion by foreground connected components (model-free)
# ---------------------------------------------------------------------------

def _label_4c(mask):
    """4-connected component labels for a boolean mask (two-pass union-find). Returns an
    int32 array of the same shape where 0 is background and each component has its own id.
    Deterministic and dependency-free — deliberately not scipy.ndimage.label, so region
    proposal works in any environment that can already run the CLIP path."""
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32)
    parent = [0]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    nxt = 1
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            up = int(lab[y - 1, x]) if y else 0
            left = int(lab[y, x - 1]) if x else 0
            if up and left:
                lab[y, x] = min(up, left)
                union(up, left)
            elif up or left:
                lab[y, x] = up or left
            else:
                lab[y, x] = nxt
                parent.append(nxt)
                nxt += 1
    if nxt > 1:
        root = np.array([find(i) for i in range(nxt)], np.int32)
        lab = root[lab]
    return lab


def propose_region_boxes(src, max_regions=8, min_area_frac=0.01, size=128, tol=0.12):
    """OpImgRegion (BBox only, no Mask): content-driven region proposals as boxes.

    Estimates the background color from the border pixels, marks every pixel further than
    `tol` from it as foreground, labels the 4-connected components, and returns their
    bounding boxes as FRACTIONS of the image — (left, top, right, bottom) each in [0,1],
    largest component first. Unlike `regions_grid` this cuts along content, which is what
    a product photo with several items or a localized defect needs. Model-free, so it is
    the cheap first backend for OpImgRegion; a learned proposer (SAM/DINO) would be a
    second one under the same signature.
    """
    im = _open(src).resize((size, size))
    arr = np.asarray(im, np.float32) / 255.0
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    bg = np.median(border, axis=0)
    fg = np.abs(arr - bg).max(-1) > tol
    if not fg.any():
        return []
    lab = _label_4c(fg)
    ids, counts = np.unique(lab[lab > 0], return_counts=True)
    keep = [(int(i), int(c)) for i, c in zip(ids, counts)
            if c / float(size * size) >= min_area_frac]
    keep.sort(key=lambda ic: (-ic[1], ic[0]))          # largest first, id breaks ties
    out = []
    for cid, _c in keep[:max_regions]:
        ys, xs = np.nonzero(lab == cid)
        out.append((float(xs.min()) / size, float(ys.min()) / size,
                    float(xs.max() + 1) / size, float(ys.max() + 1) / size))
    return out
```

- [ ] **Step 4: 实现 ImagePatch 侧**

`imagepatch.py`，在 `regions_center` 之后追加：

```python
    def propose_regions(self, max_regions=8, min_area_frac=0.01, method="contour"):
        """OpImgRegion: content-driven regions as SUB-PATCHES, largest first (a region's
        RegIdx is its index here, its BBox is `.bbox`). `method="contour"` splits on
        foreground connected components — deterministic and model-free, unlike
        `regions_grid`'s blind cut. Any other method warns and falls back to contour
        rather than pretending a learned proposer is installed."""
        if method != "contour":
            print(f"[imagepatch] propose_regions: method {method!r} is not available — "
                  f"using 'contour'")
        boxes = semvision.propose_region_boxes(self._src(), max_regions, min_area_frac)
        return [self._child(b) for b in boxes]
```

- [ ] **Step 5: 跑测试，确认通过**

Run: `python3 -m pytest tests/test_semvision_regions_propose.py -v`
Expected: 12 passed

- [ ] **Step 6: 加 predefined 包装 + 文档块**

`vadar/predefined.py` 追加：

```python
# --- P1.3: Structural — OpImgRegion by content, not by grid --------------------

def regions_propose(image, max_regions=8, min_area_frac=0.01):
    """OpImgRegion: foreground regions as sub-images, largest first."""
    return image.propose_regions(max_regions, min_area_frac)
```

`MODULES_SIGNATURES` 追加：

```
"""
Splits the image into regions along its CONTENT — it estimates the background from the
border, then returns each foreground blob as a sub-image, largest first. Prefer over
`regions_grid` when the image holds several distinct objects (a product photo with two
garments, a localized defect on a car): a grid cuts blindly and can slice one object
across cells, while this cuts around each object. Deterministic and model-free.
A region's index in the returned list identifies it; its box is `bbox(region)`.
Args:
    image (image): the image.
    max_regions (int): keep at most this many, largest first (default 8).
    min_area_frac (float): ignore blobs smaller than this fraction of the image
        (default 0.01).
Returns:
    list: sub-images (empty for a uniform image).
"""
def regions_propose(image, max_regions=8, min_area_frac=0.01):
```

- [ ] **Step 7: 跑不变式 + region 回归**

Run: `python3 -m pytest tests/test_predefined_api_surface.py tests/test_semvision_regions_propose.py tests/test_imagepatch_regions.py -v`
Expected: 全绿

- [ ] **Step 8: 提交**

```bash
git add semvision.py imagepatch.py vadar/predefined.py tests/test_semvision_regions_propose.py
git commit -m "feat(semdb): OpImgRegion proposals from foreground connected components

regions_grid cuts blindly and can slice one object across cells. propose_regions
estimates the background from the border and returns each foreground blob, so a
product photo with several items decomposes along its content. Pure numpy union-find,
no scipy/opencv/SAM dependency; unavailable methods warn and fall back."
```

---

## Task 7: P1.4 — `OpImgObj` 开放词表检测

`find()` 遇到 COCO-80 之外的类名直接返回 `[]` —— 对 Wildlife 的物种和 CarDamage 的部件是硬伤。
本任务加 OWLv2 作为第二实现（同时把 extractor-spec 的 `dino` tier 从 stub 变成有货）。

**Files:**
- Modify: `semvision.py`（检测器区块末尾）
- Modify: `imagepatch.py`（`find` 之后新增 `find_open`）
- Modify: `vadar/predefined.py`
- Test: `tests/test_imagepatch_open_vocab.py`（新建）

**Interfaces:**
- Produces:
  - `semvision.OwlDetector(model_id)`，方法 `detect_prompts(src, prompts, min_conf=0.1) -> [(label, conf, box)]`
  - `semvision.get_open_detector(model_id="google/owlv2-base-patch16-ensemble") -> OwlDetector`
  - `semvision.detect_open_boxes(src, prompts, detector, min_conf=0.1) -> [(label, conf, box)]`
  - `ImagePatch.find_open(object_prompt, min_conf=0.1) -> list[ImagePatch]`
  - `ImagePatch.find_open_detail(object_prompt, min_conf=0.1) -> list[dict]`（与 `find_detail` 同形）
  - `predefined.detect_open(image, object_prompt, min_conf=0.1)`
- **不变**：`find()` 的行为不变 —— OOV 仍然警告并返回 `[]`，只是警告文案指向 `find_open`
  （已在 Task 2 改好）。不做静默自动升级：OWLv2 是个数百 MB 的模型，隐式加载会把一次
  "detect 打错字" 变成一次几十秒的停顿。

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_imagepatch_open_vocab.py`：

```python
"""P1.4 — OpImgObj 的开放词表实现。

YOLOv8n 的词表是闭的（COCO-80）：find("impala") / find("bumper") 永远返回 []。
Wildlife 的物种和 CarDamage 的部件都在词表外，所以需要一个 prompt 即词表的检测器。
真实 OWLv2 权重不一定在环境里，所以这里全部对着 mock 测接口契约。
"""
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imagepatch  # noqa: E402
import semvision  # noqa: E402


class MockOwl:
    """形状同 semvision.OwlDetector：detect_prompts(src, prompts, min_conf)。"""

    def __init__(self, rows):
        self._rows = rows
        self.seen = []

    def detect_prompts(self, src, prompts, min_conf=0.1):
        self.seen.append((tuple(prompts), min_conf))
        return sorted([r for r in self._rows if r[0] in prompts and r[1] >= min_conf],
                      key=lambda r: -r[1])


def _img(tmp_path, size=(200, 100)):
    p = os.path.join(str(tmp_path), "x.png")
    Image.new("RGB", size, (255, 255, 255)).save(p)
    return p


def _ctx(**kw):
    base = {"encoder": None, "palette": None}
    base.update(kw)
    return base


def test_detect_open_boxes_sorts_by_confidence(tmp_path):
    owl = MockOwl([("impala", 0.4, (0, 0, 10, 10)), ("impala", 0.8, (20, 20, 40, 40))])
    rows = semvision.detect_open_boxes(_img(tmp_path), ["impala"], owl)
    assert [r[1] for r in rows] == [0.8, 0.4]


def test_find_open_detects_a_name_yolo_can_never_see(tmp_path):
    owl = MockOwl([("impala", 0.7, (10, 20, 60, 80))])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(open_detector=owl))
    hits = p.find_open("impala")
    assert len(hits) == 1 and hits[0].box == (10, 20, 60, 80)
    assert owl.seen[0][0] == ("impala",)


def test_find_open_detail_carries_label_box_and_score(tmp_path):
    owl = MockOwl([("bumper", 0.66, (10, 20, 60, 80))])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(open_detector=owl))
    d = p.find_open_detail("bumper")[0]
    assert d["label"] == "bumper" and d["score"] == 0.66 and d["box"] == (10, 20, 60, 80)


def test_find_open_inside_a_crop_offsets_to_absolute(tmp_path):
    owl = MockOwl([("impala", 0.7, (0, 0, 20, 20))])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(open_detector=owl))
    hit = p.crop(100, 50, 200, 100).find_open("impala")[0]
    assert hit.box == (100, 50, 120, 70)


def test_find_open_honors_min_conf(tmp_path):
    owl = MockOwl([("impala", 0.2, (0, 0, 10, 10))])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(open_detector=owl))
    assert p.find_open("impala", min_conf=0.5) == []


def test_find_open_degrades_to_empty_when_no_backend_is_installed(tmp_path, capsys):
    """OWLv2 是可选依赖：装不上时告警并返回 []，不能让整条查询崩掉。"""
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx())
    imagepatch.ImagePatch._OPEN_WARNED.clear()
    real = semvision.get_open_detector

    def boom(*a, **kw):
        raise ImportError("no transformers owlv2")

    semvision.get_open_detector = boom
    try:
        assert p.find_open("impala") == []
    finally:
        semvision.get_open_detector = real
    assert "open-vocabulary" in capsys.readouterr().out


def test_closed_find_still_warns_and_points_at_find_open(tmp_path, capsys):
    class MockYolo:
        class _M:
            names = {0: "zebra"}

        model = _M()

        def detect(self, _p, min_conf=0.25):
            return []

    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(detector=MockYolo()))
    imagepatch.ImagePatch._OOV_WARNED.discard("impala")
    assert p.find("impala") == []
    assert "find_open" in capsys.readouterr().out
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `python3 -m pytest tests/test_imagepatch_open_vocab.py -v`
Expected: FAIL — `AttributeError: module 'semvision' has no attribute 'detect_open_boxes'`

- [ ] **Step 3: 实现 semvision 侧**

`semvision.py`，在 `YoloDetector` 类之后、domain 区块之前插入：

```python
# ---------------------------------------------------------------------------
# ③ Open-vocabulary detector (OWLv2) — the prompt IS the vocabulary
# ---------------------------------------------------------------------------

def detect_open_boxes(src, prompts, detector, min_conf=0.1):
    """OpImgObj with an OPEN vocabulary: [(label, conf, (x1,y1,x2,y2))], most confident
    first. Where `detect_boxes` can only answer for the detector's fixed class list,
    here the prompts ARE the class list, so a species or a car part is detectable."""
    return detector.detect_prompts(src, list(prompts), min_conf)


_OPEN_DETECTOR_CACHE = {}


def get_open_detector(model_id="google/owlv2-base-patch16-ensemble"):
    if model_id not in _OPEN_DETECTOR_CACHE:
        _OPEN_DETECTOR_CACHE[model_id] = OwlDetector(model_id)
    return _OPEN_DETECTOR_CACHE[model_id]


class OwlDetector:
    """OWLv2 open-vocabulary detection. Optional: `transformers` must be able to load the
    weights, and callers degrade to [] with a warning when it cannot."""

    def __init__(self, model_id="google/owlv2-base-patch16-ensemble"):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self.proc = Owlv2Processor.from_pretrained(model_id)

    def detect_prompts(self, src, prompts, min_conf=0.1):
        im = _open(src)
        queries = [[f"a photo of a {p}" for p in prompts]]
        inp = self.proc(text=queries, images=im, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inp)
        sizes = self.torch.tensor([[im.size[1], im.size[0]]]).to(self.device)
        # The post-process helper was renamed across transformers releases.
        post = (getattr(self.proc, "post_process_grounded_object_detection", None)
                or self.proc.post_process_object_detection)
        res = post(outputs=out, target_sizes=sizes, threshold=min_conf)[0]
        rows = [(prompts[int(l)], float(s), tuple(float(v) for v in b))
                for s, l, b in zip(res["scores"], res["labels"], res["boxes"])]
        return sorted(rows, key=lambda r: -r[1])
```

- [ ] **Step 4: 实现 ImagePatch 侧**

`imagepatch.py`，在 `find` 之后插入（`_OPEN_WARNED` 与 `_OOV_WARNED` 并列声明为类属性）：

```python
    _OPEN_WARNED = set()

    def _open_detector(self):
        """The open-vocabulary backend, loaded once per run. Returns None (warning once)
        when the optional weights are unavailable — a missing extra must degrade the
        query, not crash it."""
        det = self.ctx.get("open_detector")
        if det is not None:
            return det
        try:
            det = self.ctx["open_detector"] = semvision.get_open_detector()
            return det
        except Exception as e:                # noqa: BLE001 — optional heavy dependency
            if "load" not in ImagePatch._OPEN_WARNED:
                ImagePatch._OPEN_WARNED.add("load")
                print(f"[imagepatch] no open-vocabulary detector available ({e}) — "
                      f"find_open returns []; use classify/verify_property instead")
            return None

    def find_open_detail(self, object_prompt, min_conf=0.1):
        """OpImgObj over an OPEN vocabulary: [{"image", "label", "box" (ABSOLUTE px),
        "score"}]. The prompt IS the class, so names outside the closed detector's
        COCO-80 (a species, a car part) are detectable here."""
        det = self._open_detector()
        if det is None:
            return []
        out = []
        for label, conf, box in semvision.detect_open_boxes(
                self._src(), [str(object_prompt)], det, min_conf):
            child = self._child(box)
            out.append({"image": child, "label": label, "box": child.box, "score": float(conf)})
        return out

    def find_open(self, object_prompt, min_conf=0.1):
        """Instances of ANY object name as SUB-PATCHES, most confident first. Slower than
        `find` but not limited to a closed vocabulary."""
        return [d["image"] for d in self.find_open_detail(object_prompt, min_conf)]
```

- [ ] **Step 5: 跑测试，确认通过**

Run: `python3 -m pytest tests/test_imagepatch_open_vocab.py -v`
Expected: 7 passed

- [ ] **Step 6: 加 predefined 包装 + 文档块**

`vadar/predefined.py` 追加：

```python
# --- P1.4: OpImgObj over an open vocabulary -----------------------------------

def detect_open(image, object_prompt, min_conf=0.1):
    """OpImgObj, open vocabulary: instances of ANY object name, as sub-images."""
    return image.find_open(object_prompt, min_conf)
```

`MODULES_SIGNATURES` 追加：

```
"""
Detects instances of ANY object name and returns them as SUB-IMAGES, most confident first
— the prompt IS the vocabulary. Use when `detect` reports the name is outside its closed
COCO-80 list (a species like "impala", a part like "bumper"). Slower and less precise than
`detect`, so prefer `detect` whenever the name is in its vocabulary. Returns [] with a
warning when the open-vocabulary backend is not installed.
Args:
    image (image): the image.
    object_prompt (string): any object name.
    min_conf (float): drop detections below this confidence (default 0.1).
Returns:
    list: detected instances as images (empty if none).
"""
def detect_open(image, object_prompt, min_conf=0.1):
```

- [ ] **Step 7: 跑不变式 + 检测器回归**

Run: `python3 -m pytest tests/test_predefined_api_surface.py tests/test_imagepatch_open_vocab.py tests/test_imagepatch_find_ocr.py -v`
Expected: 全绿

- [ ] **Step 8: 提交**

```bash
git add semvision.py imagepatch.py vadar/predefined.py tests/test_imagepatch_open_vocab.py
git commit -m "feat(semdb): open-vocabulary OpImgObj via OWLv2

YOLOv8n's vocabulary is closed (COCO-80), so find('impala') and find('bumper')
could never return anything — exactly the classes Wildlife and CarDamage need.
find_open takes the prompt as the class. find() is deliberately NOT auto-upgraded:
loading OWLv2 implicitly would turn a typo into a multi-second stall."
```

---

## Task 8: agent prompt 同步 + 全量验证 + 文档收尾

新函数已经进了 `PREDEFINED_API`，但 agent 的 prompt 文件是散文，不受不变式测试的「必须全覆盖」
约束（只受「不得广告不存在的名字」约束）。本任务让 agent 真的看得见新算子。

**Files:**
- Modify: `agents/vadar-program/user-prompt.md:25-26`、`agents/vadar-solver/user-prompt.md:33`、
  `agents/vadar-api/user-prompt.md:11`（import 行）
- Modify: `agents/vadar-solver/prompt.md`、`agents/vadar-program/prompt.md`（选择规则）
- Modify: `imagepatch_prompt.md`（`patch.*` 方法表）
- Modify: `docs/VIS_OPERATOR_PLAN.md`（勾掉 P0/P1.1-1.4）

- [ ] **Step 1: 更新三个 user-prompt 的 import 行**

三处都改成显式列出全部 24 个名字。`agents/vadar-program/user-prompt.md:25-26` 与
`agents/vadar-solver/user-prompt.md:33`、`agents/vadar-api/user-prompt.md:11` 统一为：

```python
from vadar.predefined import (classify, classify_detail, classify_multi, best_ocr_match,
                              best_ocr_match_detail, dominant_colors, domain_classify,
                              verify_property, verify_detail, score, read_text, ocr_detail,
                              detect, detect_detail, detect_open, crop, bbox, regions_grid,
                              regions_center, regions_propose, pair_score, embed,
                              topk_similar, topk_text)
```

- [ ] **Step 2: 在 solver/program 的选择规则里加三条**

`agents/vadar-solver/prompt.md` 与 `agents/vadar-program/prompt.md` 的规则段追加：

```
- Need a confidence to gate or rank on (keep only sure rows, or hand the unsure ones to a
  second pass)? Use the `*_detail` variant — a score is comparable across rows for that
  one primitive, never across different primitives.
- Several distinct objects in one photo -> `regions_propose` (cuts along content), not
  `regions_grid` (cuts blindly). A name `detect` says is out of vocabulary -> `detect_open`.
- Narrowing a LARGE value space -> `topk_text` for a scored short list, then verify those.
  Ranking images against each other -> `topk_similar`, not repeated `pair_score`.
```

- [ ] **Step 3: 更新 `imagepatch_prompt.md` 的方法表**

在现有 `patch.*` 列表后追加（保持同样的对齐风格）：

```
patch.classify_detail(options, template) -> (str, float)   # OpImgCls with a confidence
patch.classify_multi(options, thresh=0.5) -> (list, float) # set-valued fields
patch.verify_detail(prop) -> (bool, float)
patch.find_detail(name) -> list[dict]        # {"image","label","box","score"}
patch.find_open(name) -> list[ImagePatch]    # open vocabulary (slower)
patch.read_text_boxes(min_conf=0.0) -> list[dict]   # {"text","box","score"}
patch.best_ocr_match_detail(options) -> (str, float)
patch.domain_classify(model_id, labels) -> (str, float)    # e.g. chest X-ray
patch.propose_regions(max_regions=8) -> list[ImagePatch]   # cuts along content
patch.embed() -> list[float]
patch.topk_similar(others, k=5) -> list[(int, float)]
patch.topk_text(texts, k=5) -> list[(str, float)]
patch.bbox -> (l, t, r, b)                   # absolute pixels
```

- [ ] **Step 4: 跑全量测试套件**

Run: `python3 -m pytest tests/ -q`
Expected: 无 failure（缺数据的 skip 属正常）

Run: `node --test tests/test_vadar_offline_guard.mjs tests/test_refine_pure.mjs tests/test_val_feedback.mjs`
Expected: 全绿

- [ ] **Step 5: 跑一次真实的 VADAR 参考程序，确认 namespace 换了以后没坏**

Run: `python3 -m vadar.run q7`
Expected: 正常输出 q7 的 P/R/F1（若 SemBench 数据不在本机则跳过，记录跳过原因）

- [ ] **Step 6: 更新 `docs/VIS_OPERATOR_PLAN.md`**

在「5. 优先级汇总」表格的 P0 / P1.1 / P1.2 / P1.3 / P1.4 行首加 ✅，并在文档顶部
「状态」一行改为：

```
> 状态：P0 与 P1.1–P1.4 已实现（分支 `vis-operator`，见 docs/VIS_OPERATOR_IMPL_PLAN.md）；
> P1.5 / P2 / P3 / P4 未开始。
```

- [ ] **Step 7: 提交**

```bash
git add agents/ imagepatch_prompt.md docs/VIS_OPERATOR_PLAN.md
git commit -m "docs(semdb): advertise the new image operators to the VADAR agents

The three user-prompt import lines and imagepatch_prompt.md are the only way an
agent learns the API exists; PREDEFINED_API now covers 24 functions but a prompt
that lists 11 leaves the other 13 unreachable in practice."
```

---

## 验收标准（整体）

1. `python3 -m pytest tests/ -q` 无 failure。
2. `tests/test_predefined_api_surface.py` 通过 —— `MODULES_SIGNATURES` 与 `PREDEFINED_API`
   精确相等，agent prompt 不广告任何不存在的名字。
3. `node --test tests/test_vadar_offline_guard.mjs` 通过 —— 新增算子无一引入网络/VLM。
4. `predefined.PREDEFINED_API` 含 24 个函数（原 11 + 新 13）。
5. 论文 Table 1 的 image 算子覆盖度从「11 中 1 个完整」提升到：
   `OpImgCls` / `OpImgObj` / `OpImgOCR` / `OpImgPairScore` / `OpImgEmbed` / `OpImgRegion`
   六个完整（带 Score/BBox），`OpImgVQA`、`OpImgCap` 待 P2，`OpImgKeypt`、`OpImgSceneRel`、
   `OpImgEdit` 未做。

## 本计划明确不做

- **P2**：VLM 受控放开（`vqa` / `caption` 算子化 + guard 改造 + 计费）。
- **P3**：多实现 router + BO 调优。
- **P1.5**：`OpImgKeypt`。
- `OpImgSceneRel`、`OpImgEdit`、`OpImgRegion` 的 `Mask` 输出、SAM/DINO 区域提案后端。
- 不改 `semvision._run_attr` 的 tier 分派（`dino`/`distilled`/`vlm` 仍是 stub）—— 那是 P3 的事，
  尽管 Task 7 已经把 `dino` 需要的 open-vocab 后端备好了。
