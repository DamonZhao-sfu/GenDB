"""Source-aware candidate-domain planning for validation."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import validation_plan as VP  # noqa: E402


TAIL = "connection_id => 'x'"


def test_self_join_is_ordered_and_includes_diagonal_without_exclusion():
    sql = f"""SELECT a.id || '-' || b.id
    FROM products a JOIN products b
      ON AI.IF(('same?', a.text, b.text), {TAIL})"""
    plan = VP.build_plan(sql, benchmark="ecomm", query="q")
    assert plan["candidate"] == {
        "unit": "pair", "site_id": "s0", "aliases": ["a", "b"],
        "base": "products", "ordered": True, "include_diagonal": True,
    }


def test_cross_alias_inequality_excludes_diagonal_in_either_direction():
    for condition in ("a.id != b.id", "b.id <> a.id"):
        sql = f"""SELECT a.id, b.id FROM products a JOIN products b
        ON {condition} AND AI.IF(('same?', a.text, b.text), {TAIL})"""
        assert VP.build_plan(sql)["candidate"]["include_diagonal"] is False


def test_prompt_text_cannot_fake_a_relational_inequality():
    sql = f"""SELECT a.id, b.id FROM products a JOIN products b
    ON AI.IF(('example: a.id != b.id', a.text, b.text), {TAIL})"""
    assert VP.build_plan(sql)["candidate"]["include_diagonal"] is True


def test_sites_have_source_spans_and_stable_fingerprints():
    sql = f"""SELECT * FROM products a
    WHERE AI.IF(('first', a.text), {TAIL})
      AND AI.IF(('second', a.text), {TAIL})"""
    plan1 = VP.build_plan(sql)
    plan2 = VP.build_plan(sql)
    assert [s["site_id"] for s in plan1["sites"]] == ["s0", "s1"]
    assert all(sql[s["start"]:s["end"]].lower().startswith("ai.if")
               for s in plan1["sites"])
    assert [s["fingerprint"] for s in plan1["sites"]] == [
        s["fingerprint"] for s in plan2["sites"]]
    assert plan1["candidate"]["unit"] == "multi_site"


def test_boolean_multi_site_query_uses_one_ordered_root_tuple():
    sql = f"""WITH images AS (SELECT * FROM products p WHERE price < 1000)
    SELECT * FROM images a, images b, images c
    WHERE AI.IF(('shoe?', a.ref), {TAIL})
      AND AI.IF(('shirt?', b.ref), {TAIL})
      AND AI.IF(('same?', a.ref, b.ref), {TAIL})
      AND AI.IF(('same?', b.ref, c.ref), {TAIL})"""
    candidate = VP.build_plan(sql)["candidate"]
    assert candidate["unit"] == "tuple"
    assert candidate["aliases"] == ["a", "b", "c"]
    assert candidate["base"] == "images"
    assert candidate["arity"] == 3
    assert candidate["composition"]["kind"] == "boolean_and"


def test_filter_then_extract_uses_one_typed_pair_label():
    sql = f"""WITH matched AS (
      SELECT t.id, i.uri, i.ref FROM tracks t, images i
      WHERE AI.IF(('logo?', t.name, i.uri), {TAIL})
    )
    SELECT id, uri, AI.GENERATE(('color?', m.ref), {TAIL}) FROM matched m"""
    candidate = VP.build_plan(sql)["candidate"]
    assert candidate["unit"] == "pair"
    assert candidate["aliases"] == ["t", "i"]
    assert candidate["composition"] == {
        "kind": "filter_then_extract",
        "gate_site": "s0",
        "value_site": "s1",
        "negative_value": "no_match",
        "positive_prefix": "match:",
    }
