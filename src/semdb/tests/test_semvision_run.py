import os, sys, csv, json
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import semvision


class CVDriver:
    def __init__(self, image_dir): self.image_dir = image_dir
    def map_columns(self, header): return {"id": "id", "image": "filename", "text": None, "context": []}
    def preprocess(self, row, cols):
        import semextract
        return {"image_path": semextract.resolve_image_path(row["filename"], self.image_dir)}


def test_run_writes_attrs_and_meta(tmp_path):
    imgdir = tmp_path / "images"; imgdir.mkdir()
    for i, rgb in [(1, (240, 220, 40)), (2, (0, 0, 0))]:
        Image.fromarray(np.full((64, 64, 3), rgb, np.uint8)).save(imgdir / f"{i}.png")
    table = tmp_path / "IMAGES.csv"
    with open(table, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "filename"]); w.writerow([1, "1.png"]); w.writerow([2, "2.png"])
    schema = {"attributes": [{"name": "colors", "type": "list[enum]",
              "extractor": {"tier": "cv", "method": "dominant_colors"}}]}
    out = tmp_path / "attrs.json"
    meta = semvision.run(CVDriver(str(imgdir)), schema, str(table), str(out), image_dir=str(imgdir))
    recs = json.load(open(out))
    assert meta["rows"] == 2 and len(recs) == 2
    by = {r["id"]: r for r in recs}
    assert "yellow" in by["1"]["colors"] and "black" in by["2"]["colors"]
    assert os.path.exists(str(out) + ".meta.json")
