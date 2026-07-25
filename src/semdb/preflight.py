"""Compile-time gate for agent-generated solver code.

Runs BEFORE the solver is executed so a syntax error or an undefined name costs
one cheap static pass instead of a full corpus run, and so the feedback handed
back to the code-generating agent carries an exact location plus source context
rather than a truncated stderr tail.

Two stages, in order; the first failure wins and later stages are skipped:

  1. syntax  -- ``compile(src, path, "exec")``. Catches SyntaxError /
     IndentationError / TabError with an exact (line, col).
  2. static  -- pyflakes, when importable. ``UndefinedName`` / ``UndefinedLocal``
     are errors (they are the static shadow of the NameError the run would have
     raised); everything else pyflakes reports is a warning and does not fail
     the gate. Without pyflakes the stage reports ``unavailable`` and passes.

Deliberately NOT done here:

  * **No import smoke test.** Importing ``solve_<q>.py`` would execute its
    module-level code, which is not guaranteed to sit behind ``if __name__``;
    a "check" that writes files or reads the corpus is not a check. The
    UndefinedName stage covers the same error class with zero side effects.
  * **No offline-compliance check.** That regex list lives in
    ``orchestrator.mjs`` (``VADAR_RUNTIME_FORBIDDEN``) and is applied by the
    caller. Duplicating it here in Python would let the two copies drift.

CLI::

    python3 preflight.py solve_q3a.py [_vadar_helpers_q3a.py ...] --out preflight.json

Exit code is 0 when the gate passes and 1 when it fails, so a shell caller can
branch on it; the JSON report is written either way.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

CONTEXT_RADIUS = 3

# pyflakes message classes that mean "this would have raised NameError at run
# time". Everything else it reports (unused import, redefinition, f-string
# without placeholders, ...) is style noise for generated code and only warns.
FATAL_PYFLAKES = frozenset({"UndefinedName", "UndefinedLocal", "UndefinedExport"})


@dataclass(frozen=True)
class Finding:
    """One diagnostic. ``line``/``col`` are 1-based; ``col`` is 0 when unknown."""
    file: str
    line: int
    col: int
    error_class: str
    message: str
    context: tuple[str, ...] = field(default=())


def _context_lines(source: str, line: int, radius: int = CONTEXT_RADIUS) -> tuple[str, ...]:
    """Numbered source window around ``line``, with the offending line marked.

    Returns () when the line number is outside the file (a SyntaxError at EOF
    can report a line past the last one).
    """
    lines = source.splitlines()
    if not lines or line < 1:
        return ()
    lo, hi = max(1, line - radius), min(len(lines), line + radius)
    if lo > len(lines):
        return ()
    width = len(str(hi))
    return tuple(
        f"{'>>' if i == line else '  '} {str(i).rjust(width)} | {lines[i - 1]}"
        for i in range(lo, hi + 1)
    )


def check_syntax(path: str, source: str) -> Finding | None:
    """Compile ``source`` to bytecode. Returns a Finding on failure, else None."""
    try:
        compile(source, path, "exec")
    except SyntaxError as exc:                     # covers Indentation/TabError
        line = exc.lineno or 1
        return Finding(
            file=path, line=line, col=exc.offset or 0,
            error_class=type(exc).__name__,
            message=exc.msg or str(exc),
            context=_context_lines(source, line),
        )
    except ValueError as exc:                      # e.g. source with NUL bytes
        return Finding(file=path, line=1, col=0, error_class="ValueError",
                       message=str(exc), context=())
    return None


def check_static(path: str, source: str) -> tuple[list[Finding], list[Finding], str]:
    """Run pyflakes over ``source``.

    Returns ``(errors, warnings, status)`` where status is "pyflakes" when the
    checker ran and "unavailable" when it is not installed (in which case both
    lists are empty and the gate passes -- a missing optional linter must never
    block code generation).
    """
    try:
        from pyflakes import api as pyflakes_api  # type: ignore[import-untyped]
    except ImportError:
        return [], [], "unavailable"

    collected: list[Any] = []

    class _Collector:
        """pyflakes Reporter protocol: the three methods its api calls."""

        def unexpectedError(self, filename: str, msg: str) -> None:
            collected.append(Finding(file=filename, line=1, col=0,
                                     error_class="PyflakesInternalError",
                                     message=str(msg)))

        def syntaxError(self, filename: str, msg: str, lineno: int, offset: int, text: str) -> None:
            # check_syntax already ran and passed, so reaching here means the two
            # parsers disagree. Record it rather than silently dropping it.
            collected.append(Finding(file=filename, line=lineno or 1, col=offset or 0,
                                     error_class="SyntaxError", message=str(msg)))

        def flake(self, message: Any) -> None:
            collected.append(message)

    pyflakes_api.check(source, path, _Collector())

    errors: list[Finding] = []
    warnings: list[Finding] = []
    for item in collected:
        if isinstance(item, Finding):                       # reporter-level problem
            errors.append(item)
            continue
        cls = type(item).__name__
        line = getattr(item, "lineno", 1) or 1
        finding = Finding(
            file=path, line=line, col=getattr(item, "col", 0) or 0,
            error_class=cls,
            message=(item.message % item.message_args) if hasattr(item, "message_args") else str(item),
            context=_context_lines(source, line) if cls in FATAL_PYFLAKES else (),
        )
        (errors if cls in FATAL_PYFLAKES else warnings).append(finding)
    return errors, warnings, "pyflakes"


def preflight(paths: list[str], *, names_are_fatal: bool = True) -> dict[str, Any]:
    """Gate every file in ``paths``. First failing stage wins.

    ``names_are_fatal=False`` downgrades undefined-name findings to warnings, for
    generated code that legitimately injects names into globals().
    """
    report: dict[str, Any] = {
        "ok": True, "stage": None, "files": list(paths),
        "errors": [], "warnings": [], "static_checker": None,
    }

    sources: dict[str, str] = {}
    for path in paths:
        try:
            with open(path, encoding="utf-8") as handle:
                sources[path] = handle.read()
        except OSError as exc:
            report.update(ok=False, stage="read",
                          errors=[asdict(Finding(file=path, line=0, col=0,
                                                 error_class="OSError", message=str(exc)))])
            return report

    # Stage 1 -- syntax. Report every file's syntax error, not just the first:
    # the agent can fix them all in one edit.
    syntax_errors = [f for path, src in sources.items() if (f := check_syntax(path, src))]
    if syntax_errors:
        report.update(ok=False, stage="syntax",
                      errors=[asdict(f) for f in syntax_errors])
        return report

    # Stage 2 -- static names.
    errors: list[Finding] = []
    warnings: list[Finding] = []
    status = "unavailable"
    for path, src in sources.items():
        errs, warns, status = check_static(path, src)
        if names_are_fatal:
            errors.extend(errs)
        else:
            warnings.extend(errs)
        warnings.extend(warns)

    report["static_checker"] = status
    report["warnings"] = [asdict(w) for w in warnings]
    if errors:
        report.update(ok=False, stage="static", errors=[asdict(e) for e in errors])
    return report


def render_text(report: dict[str, Any]) -> str:
    """Human/agent-readable rendering of a report -- the block the feedback uses."""
    if report["ok"]:
        checker = report.get("static_checker") or "unavailable"
        note = "" if checker == "pyflakes" else "  (static name check skipped: pyflakes not installed)"
        return f"COMPILE OK{note}"
    lines = [f"COMPILE FAILED at stage `{report['stage']}`"]
    for err in report["errors"]:
        where = f"{err['file']}:{err['line']}" + (f":{err['col']}" if err["col"] else "")
        lines.append(f"{err['error_class']} at {where} -- {err['message']}")
        lines.extend(err.get("context") or [])
    extra = report.get("warnings") or []
    if extra:
        lines.append(f"Also flagged (non-fatal, {len(extra)}):")
        lines.extend(f"  {w['error_class']} at {w['file']}:{w['line']} -- {w['message']}"
                     for w in extra[:5])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Static compile gate for generated solver code.")
    ap.add_argument("paths", nargs="+", help="Python files to check (solver first).")
    ap.add_argument("--out", help="Write the JSON report here.")
    ap.add_argument("--warn-only-names", action="store_true",
                    help="Downgrade undefined-name findings from error to warning.")
    ap.add_argument("--quiet", action="store_true", help="Do not print the text rendering.")
    args = ap.parse_args(argv)

    report = preflight(args.paths, names_are_fatal=not args.warn_only_names)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    if not args.quiet:
        print(render_text(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
