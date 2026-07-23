import os, sys, csv, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import semvision, semextract

DATA = "/localhome/hza214/SemBench/files/ecomm/data/sf_250"
MANIFEST = "/local-scratch/localhome/hza214/GenDB/src/semdb/runs/_materialized/ecomm_sf250/IMAGES.csv"
GT = "/localhome/hza214/SemBench/files/ecomm/raw_results/ground_truth/Q2.csv"


class ImgDriver:
    def __init__(self, image_dir): self.image_dir = image_dir
    def map_columns(self, header): return {"id": "id", "image": "filename", "text": None, "context": []}
    def preprocess(self, row, cols):
        return {"image_path": semextract.resolve_image_path(row["filename"], self.image_dir)}


def test_ecomm_q2_hits_yellow_silver_sports_shoes(tmp_path):
    if not os.path.exists(MANIFEST):
        import pytest; pytest.skip("materialize ecomm first")
    schema = {"attributes": [
        {"name": "product_type", "type": "enum",
         "extractor": {"tier": "clip", "method": "classify",
                       "labels": ["sports_shoes", "sandal", "boot", "other_footwear", "not_footwear"]}},
        {"name": "colors", "type": "list[enum]",
         "extractor": {"tier": "cv", "method": "dominant_colors", "params": {"min_frac": 0.03}}},
    ]}
    out = tmp_path / "attrs.json"
    semvision.run(ImgDriver(os.path.join(DATA, "images")), schema, MANIFEST, str(out),
                  image_dir=os.path.join(DATA, "images"))
    recs = {r["id"]: r for r in json.load(open(out))}
    gt = {r["id"] for r in csv.DictReader(open(GT))}
    hit = [i for i, r in recs.items()
           if r.get("product_type") == "sports_shoes"
           and {"yellow", "silver"}.issubset(set(r.get("colors", [])))]
    # diagnostic: show the 5 GT shoes' extraction
    for g in gt:
        print("GT", g, recs.get(g, {}).get("product_type"), recs.get(g, {}).get("colors"))
    assert set(hit) & gt, f"no GT shoe recovered; hits={hit[:5]} gt={list(gt)[:5]}"
