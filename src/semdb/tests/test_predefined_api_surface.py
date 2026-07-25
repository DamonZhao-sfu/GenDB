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
