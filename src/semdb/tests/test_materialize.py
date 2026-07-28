"""EComm parquet materialization and normalized product view."""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import materialize as M  # noqa: E402


def _inputs(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    details = pd.DataFrame([
        {
            "id": 1, "price": 100.0, "baseColour": "Black", "colour1": "",
            "colour2": "", "brandName": "Acme", "productDisplayName": "Black shoe",
            "styleImages": {"default": {"imageURL": "https://x/1.jpg"}},
            "productDescriptors": {"description": {"value": "A running shoe"}},
            "masterCategory": {"typeName": "Footwear"},
        },
        {
            "id": 2, "price": 700.0, "baseColour": "Red", "colour1": "NA",
            "colour2": "", "brandName": "Beta", "productDisplayName": "Red shirt",
            "styleImages": {"default": {"imageURL": "https://x/2.jpg"}},
            "productDescriptors": {"description": {"value": "A shirt"}},
            "masterCategory": {"typeName": "Apparel"},
        },
    ])
    details.to_parquet(data / "styles_details.parquet", index=False)
    pd.DataFrame({"id": [1, 2], "productDisplayName": ["Black shoe", "Red shirt"]}).to_parquet(
        data / "styles.parquet", index=False)
    pd.DataFrame({
        "id": ["1", "2"], "filename": ["1.jpg", "2.jpg"],
        "link": ["https://x/1.jpg", "https://x/2.jpg"],
    }).to_parquet(data / "image_mapping.parquet", index=False)
    return data


def test_normalized_product_view_flattens_nested_fields(tmp_path):
    data, out = _inputs(tmp_path), tmp_path / "out"
    M.materialize_ecomm(str(data), str(out))
    rows = pd.read_csv(out / "ecomm_products.csv", keep_default_na=False)
    assert list(rows["filename"]) == ["1.jpg", "2.jpg"]
    assert rows.loc[0, "description"] == "A running shoe"
    assert rows.loc[0, "masterCategoryName"] == "Footwear"
    assert rows.loc[0, "semantic_text"] == "Black shoe - A running shoe"
    assert rows.loc[1, "colour1"] == "NA", "literal NA must not become SQL NULL"


def test_source_provenance_controls_cache_reuse(tmp_path):
    data, out = _inputs(tmp_path), tmp_path / "out"
    M.materialize_ecomm(str(data), str(out))
    product = out / "ecomm_products.csv"
    first_mtime = product.stat().st_mtime_ns
    M.materialize_ecomm(str(data), str(out))
    assert product.stat().st_mtime_ns == first_mtime
    meta = json.load(open(out / "materialize_meta.json"))
    assert meta["rows"]["ecomm_products.csv"] == 2


def test_duplicate_physical_ids_are_rejected(tmp_path):
    data = _inputs(tmp_path)
    mapping = pd.read_parquet(data / "image_mapping.parquet")
    mapping.loc[1, "id"] = mapping.loc[0, "id"]
    mapping.to_parquet(data / "image_mapping.parquet", index=False)
    try:
        M.materialize_ecomm(str(data), str(tmp_path / "out"))
    except SystemExit as exc:
        assert "not unique" in str(exc)
    else:
        raise AssertionError("duplicate ids must fail materialization")
