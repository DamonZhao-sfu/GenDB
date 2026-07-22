#!/usr/bin/env bash
# Replays the compiled q3 family over the small-model schema and validates it.
# (Phases A and B — the Schema Designer agent and the Haiku extractor — already
#  produced schema.json and movie_attrs.json; this reruns Phase C + verify.)
set -euo pipefail
cd "$(dirname "$0")"
echo "=== Phase C: compiled SQL simulation (0 model calls) ==="
python3 compiled_q3.py
echo
echo "=== Verify: small-model schema vs oracle labels ==="
python3 validate.py
