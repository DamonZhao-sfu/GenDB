# src/semdb/tests/test_semtext_run_extraction.py
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import semtext


def _fake_gen(monkeypatch):
    import semextract
    def fake(cfg, json_schema, prompt, modality, image_path=None, text=None):
        # classify → return the label that appears in the text; else first option.
        return json.dumps({"value": "positive" if "great" in (text or "") else "negative"})
    monkeypatch.setattr(semextract, "gen_endpoint", fake)


class _Driver:
    def map_columns(self, header):
        return {"id": "id", "text": "review", "context": []}
    def extract(self, patch):
        return {"sentiment": patch.classify(["positive", "negative"])}


def test_run_extraction_writes_attrs_and_meta(tmp_path, monkeypatch):
    _fake_gen(monkeypatch)
    semtext.METER.reset()
    table = tmp_path / "reviews.csv"
    table.write_text("id,review\n1,a great film\n2,a dull film\n")
    schema = {"attributes": [{"name": "sentiment", "type": "string"}]}
    out = tmp_path / "attrs.json"
    meta = semtext.run_extraction(_Driver(), schema, str(table), str(out),
                                  model="m", endpoint="http://x/v1", concurrency=2)
    recs = json.load(open(out))
    assert {r["id"]: r["sentiment"] for r in recs} == {"1": "positive", "2": "negative"}
    assert all("conf" in r for r in recs)
    m = json.load(open(str(out) + ".meta.json"))
    assert m["rows"] == 2 and m["llm_calls"] == meta["llm_calls"] == 2
