"""Deterministic EComm CTE execution before semantic self-join sampling."""
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import frame_builder as FB  # noqa: E402


TAIL = "connection_id => 'x'"


def _products(tmp_path):
    path = tmp_path / "ecomm_products.csv"
    pd.DataFrame([
        {"id": "1", "filename": "1.jpg", "price": 100, "baseColour": "Black",
         "colour1": "", "colour2": "", "semantic_text": "black shoe"},
        {"id": "2", "filename": "2.jpg", "price": 700, "baseColour": "Red",
         "colour1": "", "colour2": "", "semantic_text": "red shirt"},
        {"id": "3", "filename": "3.jpg", "price": 100, "baseColour": "Blue",
         "colour1": "NA", "colour2": "", "semantic_text": "blue shoe"},
        {"id": "4", "filename": "4.jpg", "price": 100, "baseColour": "Purple",
         "colour1": "", "colour2": "", "semantic_text": "purple bag"},
    ]).to_csv(path, index=False)
    return path


def _write(tmp_path, name, sql):
    path = tmp_path / name
    path.write_text(sql)
    return path


def test_q7_style_price_prefix_and_diagonal_domain(tmp_path):
    sql = f"""WITH product_selection AS (
      SELECT * FROM fashion_product_images.STYLES_DETAILS styles_details
      WHERE true AND price <= 500
    )
    SELECT p1.id, p2.id FROM product_selection p1 JOIN product_selection p2
      ON AI.IF(('same?', p1.semantic_text, p2.semantic_text), {TAIL})"""
    source = _write(tmp_path, "q7.sql", sql)
    out = tmp_path / "rows.csv"
    meta = FB.build_ecomm_frame(str(_products(tmp_path)), str(source), str(out),
                                query="q7")
    assert set(pd.read_csv(out)["id"].astype(str)) == {"1", "3", "4"}
    assert meta["candidate"]["ordered"] is True
    assert meta["candidate"]["include_diagonal"] is True


def test_q9_empty_colour_is_not_confused_with_literal_na(tmp_path):
    sql = f"""WITH product_selection AS (
      SELECT images.* FROM fashion_product_images.STYLES_DETAILS styles_details
      JOIN fashion_product_images.IMAGE_MAPPING mapping ON styles_details.id = mapping.id
      JOIN fashion_product_images.IMAGES images ON images.id = mapping.id
      WHERE true
        AND styles_details.baseColour IN ('Black', 'Blue', 'Red')
        AND styles_details.colour1 = ''
        AND styles_details.colour2 = ''
        AND price < 800
    )
    SELECT p1.id, p2.id FROM product_selection p1 JOIN product_selection p2
      ON p1.id != p2.id
      AND AI.IF(('same?', p1.filename, p2.filename), {TAIL})"""
    source = _write(tmp_path, "q9.sql", sql)
    out = tmp_path / "rows.csv"
    meta = FB.build_ecomm_frame(str(_products(tmp_path)), str(source), str(out),
                                query="q9")
    assert set(pd.read_csv(out)["id"].astype(str)) == {"1", "2"}
    assert meta["candidate"]["include_diagonal"] is False
    written = json.load(open(str(out) + ".meta.json"))
    assert written["output_rows"] == 2


def test_q8_filters_the_structured_side_before_cross_pairing(tmp_path):
    products = tmp_path / "ecomm_products.csv"
    pd.DataFrame([
        {"id": "1", "productDisplayName": "short shoe",
         "description": "x" * 2999},
        {"id": "2", "productDisplayName": "long shirt",
         "description": "y" * 3000},
        {"id": "3", "productDisplayName": "long bag",
         "description": "z" * 3100},
    ]).to_csv(products, index=False)
    sql = f"""WITH product_selection AS (
      SELECT * FROM fashion_product_images.STYLES_DETAILS styles_details
      WHERE true
        AND CHAR_LENGTH(
          styles_details.productDescriptors.description.value) >= 3000
    )
    SELECT styles_details.id
    FROM product_selection AS styles_details
    JOIN EXTERNAL_OBJECT_TRANSFORM(
      TABLE `fashion_product_images.IMAGES`, ['SIGNED_URL']) AS images
      ON AI.IF(('fits?', images.ref, styles_details.productDisplayName, ' ',
                styles_details.productDescriptors.description.value), {TAIL})"""
    source = _write(tmp_path, "q8.sql", sql)
    out = tmp_path / "left.csv"
    meta = FB.build_ecomm_cross_left_frame(
        str(products), str(source), str(out), query="q8")
    rows = pd.read_csv(out, keep_default_na=False)

    assert set(rows["id"].astype(str)) == {"2", "3"}
    assert rows.set_index("id").loc[2, "predicate_text"].startswith("long shirt ")
    assert meta["output_rows"] == 2
    assert "productDescriptors.description.value" in meta["source_where"]
    assert "styles_details.description" in meta["where"]


def test_filtered_cross_left_builder_is_fail_closed_outside_q8(tmp_path):
    sql = f"""WITH product_selection AS (
      SELECT * FROM fashion_product_images.STYLES_DETAILS styles_details
      WHERE true
    )
    SELECT * FROM product_selection styles_details
    JOIN fashion_product_images.IMAGES images
      ON AI.IF(('fits?', images.ref, styles_details.productDisplayName), {TAIL})"""
    with pytest.raises(ValueError, match="not implemented"):
        FB.deterministic_cross_left_spec(
            sql, benchmark="ecomm", query="q14")


def test_ai_inside_deterministic_prefix_is_refused():
    sql = f"""WITH product_selection AS (
      SELECT * FROM products styles_details
      WHERE AI.IF(('cheap?', styles_details.semantic_text), {TAIL})
    )
    SELECT * FROM product_selection p1 JOIN product_selection p2
      ON AI.IF(('same?', p1.semantic_text, p2.semantic_text), {TAIL})"""
    try:
        FB.deterministic_spec(sql, benchmark="ecomm", query="q")
    except ValueError as exc:
        assert "contains an AI call" in str(exc)
    else:
        raise AssertionError("semantic prefixes must fail closed")
