#!/usr/bin/env python3
"""vadar/agents.py — the strict VADAR 3-agent pipeline (SignatureAgent -> APIAgent ->
ProgramAgent), ported from VADAR/agents/agents.py, over OUR predefined API. Each agent
calls the code-LLM (generator.Generator). The Program agent's output is `extract(image)`,
which the engine execs over the corpus."""
import re

from . import predefined
from .prompts import SIGNATURE_PROMPT, API_PROMPT, PROGRAM_PROMPT


def _tag(text, tag):
    return [m.strip() for m in re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)]


def _strip_fences(s):
    return s.replace("```python", "").replace("```", "").strip()


class SignatureAgent:
    """Proposes NEW helper signatures (docstring+signature) over the predefined API."""

    def __init__(self, generator):
        self.gen = generator
        self.headers = []                       # [{docstring, signature}]

    def propose(self, questions):
        out = self.gen.generate(SIGNATURE_PROMPT.format(
            signatures=predefined.MODULES_SIGNATURES, question="\n\n".join(questions)))
        docs, sigs = _tag(out, "docstring"), _tag(out, "signature")
        self.headers = [{"docstring": d, "signature": s} for d, s in zip(docs, sigs)]
        return self.headers


class APIAgent:
    """Implements each proposed helper by composing predefined + already-generated helpers."""

    def __init__(self, generator, signature_agent):
        self.gen = generator
        self.sig = signature_agent
        self.api = []                           # [{docstring, signature, implementation}]

    def implement(self):
        gen_sigs = ""
        for h in self.sig.headers:
            out = self.gen.generate(API_PROMPT.format(
                predef_signatures=predefined.MODULES_SIGNATURES,
                generated_signatures=gen_sigs, docstring=h["docstring"], signature=h["signature"]))
            impl = _tag(out, "implementation")
            body = _strip_fences(impl[0]) if impl else "return None"
            self.api.append({**h, "implementation": body})
            gen_sigs += h["docstring"] + "\n" + h["signature"] + "\n\n"
        return self.api


class ProgramAgent:
    """Writes `extract(image)` using the full API (predefined + generated)."""

    def __init__(self, generator, api_agent):
        self.gen = generator
        self.api_agent = api_agent

    def program(self, question, value_spaces_doc):
        api_str = predefined.MODULES_SIGNATURES + "\n" + "\n".join(
            a["docstring"] + "\n" + a["signature"] for a in self.api_agent.api)
        out = self.gen.generate(PROGRAM_PROMPT.format(
            api=api_str, value_spaces=value_spaces_doc, question=question))
        prog = _tag(out, "program")
        return _strip_fences(prog[0]) if prog else _strip_fences(out)


def synthesize(generator, questions, value_spaces_doc):
    """Run the full VADAR pipeline for a query-set: signature -> api -> program.
    Returns (helpers_source, program_source, api)."""
    sig = SignatureAgent(generator); sig.propose(questions)
    api = APIAgent(generator, sig); api.implement()
    prog = ProgramAgent(generator, api).program(questions[0], value_spaces_doc)
    # assemble the generated helper functions into source
    def _fn(h):
        body = "\n".join("    " + ln for ln in h["implementation"].splitlines())
        return h["signature"] + "\n" + body
    helpers_src = "\n\n".join(_fn(h) for h in api.api)
    return helpers_src, prog, api.api
