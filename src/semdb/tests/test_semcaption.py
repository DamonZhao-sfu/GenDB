"""OpImgCap — the caption pass produces a corpus-level, joinable column."""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import semcaption  # noqa: E402
import semextract  # noqa: E402


def _corpus(tmp_path, n=3):
    from PIL import Image
    d = str(tmp_path)
    rows = []
    for i in range(n):
        name = f"img{i}.png"
        Image.new("RGB", (8, 8)).save(os.path.join(d, name))
        rows.append({"uri": name})
    table = os.path.join(d, "images.csv")
    with open(table, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["uri"])
        w.writeheader()
        w.writerows(rows)
    return table, d


def test_caption_schema_is_a_guided_json_string_field():
    js = semextract.build_json_schema(semcaption.CAPTION_SCHEMA)
    assert js["properties"]["caption"]["type"] == "string"
    assert "conf" in js["required"]


def test_run_writes_the_attribute_table_contract(tmp_path, monkeypatch):
    table, d = _corpus(tmp_path)
    monkeypatch.setattr(semextract, "gen_endpoint",
                        lambda *a, **k: json.dumps({"caption": "a red shoe", "conf": 0.9}))
    out = os.path.join(d, "captions.json")
    meta = semcaption.run(table, out, id_col="uri", image_col="uri", image_dir=d,
                          model="vlm", endpoint="http://x/v1", concurrency=2)
    rows = json.load(open(out))
    assert meta["rows"] == 3
    assert {"uri", "caption", "conf"} <= set(rows[0])
    assert rows[0]["caption"] == "a red shoe"


def test_load_returns_an_id_to_caption_map(tmp_path):
    p = os.path.join(str(tmp_path), "c.json")
    json.dump([{"uri": "a.png", "caption": "a cat", "conf": 0.8},
               {"uri": "b.png", "caption": "a dog", "conf": 0.7}], open(p, "w"))
    assert semcaption.load(p) == {"a.png": "a cat", "b.png": "a dog"}


def test_caption_prompt_asks_for_literal_visible_content():
    p = semcaption.CAPTION_PROMPT.lower()
    assert "strict json" in p and "caption" in p
    assert "visible" in p
