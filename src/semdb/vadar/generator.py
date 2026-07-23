#!/usr/bin/env python3
"""vadar/generator.py — the code-LLM backend (VADAR's engine_utils.Generator analog).
OpenAI-compatible, so it points at a vLLM/OpenAI endpoint or codex via base_url/model.
Configure with env: VADAR_BASE_URL, VADAR_MODEL, VADAR_API_KEY."""
import os


class Generator:
    def __init__(self, model=None, base_url=None, api_key=None, temperature=0.2):
        from openai import OpenAI
        base_url = base_url or os.environ.get("VADAR_BASE_URL")          # e.g. http://localhost:8000/v1
        api_key = api_key or os.environ.get("VADAR_API_KEY", "EMPTY")
        self.model = model or os.environ.get("VADAR_MODEL", "gpt-4o")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.temperature = temperature

    def generate(self, prompt):
        r = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content
