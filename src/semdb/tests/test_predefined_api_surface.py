"""不变式：广告给 agent 的每个 API 名字都必须真实存在且可调用。

历史 bug：`predefined.detect` 出现在 MODULES_SIGNATURES 里，但它依赖的
`ImagePatch.find` 根本没实现 —— 每次调用抛 AttributeError，被引擎的 per-row
guard 吞成静默的 'none' 列。这里把广告面与实现面锁在一起。

vision 与 text 算子现在同住 `vadar/predefined.py`，所以这一条不变式覆盖两族。
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vadar import imagepatch  # noqa: E402
from vadar import predefined  # noqa: E402

SEMDB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEF_RE = re.compile(r"^def\s+(\w+)\s*\(", re.M)
IMPORT_RE = re.compile(r"from vadar\.predefined import\s*(?:\(([^)]*)\)|([^\n]*))")
PATCH_CALL_RE = re.compile(r"\bpatch\.(\w+)\s*\(")

PROMPT_FILES = ["agents/vadar-solver/user-prompt.md",
                "agents/vadar-solver/user-prompt-text.md",
                "agents/vadar-api/user-prompt.md",
                "agents/vadar-api/user-prompt-text.md"]


def _read(rel):
    return open(os.path.join(SEMDB, rel), encoding="utf-8").read()


def _imported_names(text):
    """The names an agent prompt tells the model to import. Prompts wrap the import in
    markdown backticks, so strip those before matching."""
    out = set()
    for m in IMPORT_RE.finditer(text):
        body = m.group(1) if m.group(1) is not None else m.group(2)
        out |= {t.strip().strip("`") for t in body.replace("\n", " ").split(",") if t.strip()}
    return out - {""}


def test_documented_signatures_match_the_api_table_exactly():
    documented = set(DEF_RE.findall(predefined.MODULES_SIGNATURES))
    assert documented == set(predefined.PREDEFINED_API), (
        f"only documented: {documented - set(predefined.PREDEFINED_API)}; "
        f"only exported: {set(predefined.PREDEFINED_API) - documented}")


def test_every_exported_api_entry_is_callable():
    for name, fn in predefined.PREDEFINED_API.items():
        assert callable(fn), name


def test_both_operator_families_are_exported_from_one_module():
    """vision 与 text 算子合并后，生成代码只有一个 import 根。"""
    api = set(predefined.PREDEFINED_API)
    assert {"classify", "detect", "dominant_colors"} <= api               # vision
    assert {"normalize", "contains_any", "text_classify_detail"} <= api   # text


def test_no_hardcoded_value_spaces_leak_into_the_library():
    """算子库不得内置任何具体 taxonomy —— value space 必须运行时由 query 传入。

    历史问题：`_MOVIE_GENRE_ALIASES` / `_EUROPE_LOCATIONS` / `_GERMANY_LOCATIONS`
    把 SemBench 的具体值域焊死在库里，和「不许硬编码 value space」的 prompt 规则
    直接冲突，且对任何新语料都是错的。"""
    src = _read("vadar/predefined.py")
    for banned in ("_MOVIE_GENRE_ALIASES", "_EUROPE_LOCATIONS", "_GERMANY_LOCATIONS",
                   "classify_movie_genres", "has_movie_genres", "destination_in_region"):
        assert banned not in src, f"{banned} is a hardcoded value space"
    assert not hasattr(predefined, "classify_movie_genres")


def test_vadar_package_exports_the_whole_generated_code_surface():
    """生成代码只写 `from vadar import ...` —— 这几个名字必须都在包顶层。"""
    import vadar
    for name in ("ImagePatch", "get_encoder", "resolve_image_path"):
        assert hasattr(vadar, name), name


def test_agent_prompt_imports_reference_only_real_functions():
    for rel in PROMPT_FILES:
        for name in _imported_names(_read(rel)):
            assert name in predefined.PREDEFINED_API, f"{rel} advertises missing {name!r}"


def test_imagepatch_api_doc_advertises_only_real_methods():
    for name in set(PATCH_CALL_RE.findall(_read("vadar/API.md"))):
        assert hasattr(imagepatch.ImagePatch, name), f"vadar/API.md advertises missing {name!r}"


def test_classify_never_abstains_but_classify_or_none_does():
    """mmqa q7 shipped `if classify_detail(...)[0] != "none"` — a guard that is always
    true, so all 200 images were labelled. The abstaining variant is the sanctioned fix."""
    class FakePatch:
        def __init__(self, score): self._score = score
        def classify_detail(self, options, template="a photo of {}"):
            return options[0], self._score

    confident, unsure = FakePatch(0.9), FakePatch(0.1)
    options = ["British Airways", "Delta Air Lines"]

    # argmax always returns a member of `options` — "none" is not reachable
    assert predefined.classify_detail(unsure, options)[0] in options
    assert predefined.classify_detail(unsure, options)[0] != "none"

    assert predefined.classify_or_none(confident, options, min_conf=0.5) == "British Airways"
    assert predefined.classify_or_none(unsure, options, min_conf=0.5) == "none"
    # the threshold is inclusive, so a plan may pin it exactly at the observed score
    assert predefined.classify_or_none(FakePatch(0.5), options, min_conf=0.5) != "none"


def test_the_argmax_trap_is_stated_where_the_agent_reads_it():
    doc = predefined.MODULES_SIGNATURES
    assert "ARGMAX NEVER ABSTAINS" in doc
    assert "classify_or_none" in doc
