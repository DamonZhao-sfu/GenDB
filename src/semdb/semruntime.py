"""
semruntime.py — runtime helpers that compiled SemDB queries import for the
RESIDUAL path (the rows the cheap relational plan was unsure about).

`vlm_judge` sends the ORIGINAL semantic predicate to a vLLM / OpenAI-compatible
endpoint and returns a boolean. It works for both modalities: pass image_path for
a VLM judge (q2a/q7 logo), omit it for a text-only judge (q3/q5/q6). Every call is
counted in METER so the compiled query can report residual_calls.

Example (inside a generated compiled_<q>.py):
    from semruntime import vlm_judge, METER
    if vlm_judge(
            f"Determine if the image shows the logo of {airline}. Answer YES or NO.",
            endpoint=args.endpoint, model=args.model, api_key=args.api_key,
            image_path=image_file):
        pairs.append((airline, uri))
    ...
    residual_calls = METER.judge_calls
"""

import base64
import json
import os
import urllib.request


class _Meter:
    def __init__(self):
        self.judge_calls = 0


METER = _Meter()


def _data_url(path):
    ext = (os.path.splitext(path)[1].lstrip(".") or "png").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    with open(path, "rb") as f:
        return f"data:image/{mime};base64," + base64.b64encode(f.read()).decode()


def vlm_judge(prompt, endpoint, model, api_key="EMPTY", image_path=None, timeout=120):
    """Send the semantic yes/no predicate to the endpoint. Returns True/False.

    `prompt` is the original AI.IF question (e.g. "does this image show the logo
    of Southwest Airlines?"). image_path adds the image for a VLM judge; omit it
    for a text predicate. Requires an OpenAI-compatible server (vLLM)."""
    if not endpoint:
        raise RuntimeError(
            "vlm_judge needs an --endpoint (vLLM/OpenAI server). Without one the "
            "residual path cannot re-check unsure rows — pass --endpoint to the "
            "compiled query (the orchestrator forwards it).")
    METER.judge_calls += 1
    content = [{"type": "text", "text": prompt.rstrip() + "\nAnswer strictly YES or NO."}]
    if image_path:
        content.append({"type": "image_url", "image_url": {"url": _data_url(image_path)}})
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 4,
        "temperature": 0,
    }
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read())
    text = out["choices"][0]["message"]["content"].strip().lower()
    return text.startswith("y") or "yes" in text[:6]


# Text-only alias for readability in generated code.
def llm_judge(prompt, endpoint, model, api_key="EMPTY", timeout=120):
    return vlm_judge(prompt, endpoint, model, api_key=api_key, image_path=None, timeout=timeout)
