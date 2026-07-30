import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vadar import backend as semvision

CSV = "/localhome/hza214/SemBench/files/animals/data/sf_200/image_data.csv"


def _local(p): return p.replace("/home/jiale/SemBench", "/localhome/hza214/SemBench")


def test_yolo_detects_zebra_in_camera_trap(tmp_path):
    if not os.path.exists(CSV):
        import pytest; pytest.skip("animals data absent")
    zeb = []
    for r in csv.DictReader(open(CSV)):
        if "zebra" in str(r.get("Species", "")).lower():
            p = _local(r["ImagePath"])
            if os.path.exists(p):
                zeb.append(p)
        if len(zeb) >= 4:
            break
    if not zeb:
        import pytest; pytest.skip("no local zebra images")
    det = semvision.get_detector("yolov8n.pt")
    hits = sum(1 for p in zeb if semvision.detect(p, ["zebra"], det, min_conf=0.2)[0])
    # YOLO should recover the majority of clear zebra camera-trap shots
    assert hits >= max(1, len(zeb) // 2), f"only {hits}/{len(zeb)} zebras detected"
