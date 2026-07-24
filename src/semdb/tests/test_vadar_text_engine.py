import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import vadar_text_engine
from vadar.predefined_text import best_lexical_match


class _Driver:
    def map_columns(self, header):
        return {"id": "id", "text": "description", "context": ["title"]}

    def extract(self, text):
        return {"genre": best_lexical_match(text, ["comedy", "horror"])}


def test_offline_engine_writes_zero_model_call_metadata(tmp_path):
    table = tmp_path / "movies.csv"
    with table.open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["id", "title", "description"])
        writer.writerow(["1", "Funny Film", "A warm comedy"])
        writer.writerow(["2", "Unknown", "A documentary"])

    schema = {"attributes": [{"name": "genre", "type": "string"}]}
    output = tmp_path / "attrs.json"
    meta = vadar_text_engine.run(_Driver(), schema, str(table), str(output))

    records = json.loads(output.read_text())
    assert [record["genre"] for record in records] == ["comedy", "none"]
    assert [record["id"] for record in records] == ["1", "2"]
    assert meta["engine"] == "vadar-text-offline"
    assert meta["llm_calls"] == 0
    assert json.loads((tmp_path / "attrs.json.meta.json").read_text())["llm_calls"] == 0
