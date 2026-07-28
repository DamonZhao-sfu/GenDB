"""AI call-site extraction and per-row/pairwise classification.

The classification decides whether a per-row validation set is even DEFINED for a
call site, so a silent misclassification produces a val set that measures the wrong
thing while looking healthy. Every shape assertion here is anchored on a real
SemBench ecomm query, quoted down to the clause that makes the answer what it is.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import predicate as P  # noqa: E402

ECOMM = "/localhome/hza214/SemBench/files/ecomm/queries/dialects/bigquery"
has_ecomm = pytest.mark.skipif(not os.path.isdir(ECOMM), reason="SemBench ecomm not present")

AI_TAIL = "connection_id => '<<connection>>', model_params => JSON '{\"a\":1}'"


def sites(sql):
    return P.parse_sql(sql)


# --- shape: single alias ---------------------------------------------------

def test_single_alias_is_per_row():
    sql = f"SELECT id FROM images WHERE AI.IF(('is it a shoe?', images.ref), {AI_TAIL})"
    (s,) = sites(sql)
    assert s.shape == "per_row"
    assert s.kind == "if"
    assert s.prompt == "is it a shoe?"
    assert s.aliases == ("images",)


# --- shape: self-join ------------------------------------------------------

def test_self_join_aliases_are_pairwise():
    """FROM t AS a, t AS b -- the call compares a row to another row of one relation."""
    sql = f"""
    SELECT a.id FROM images as a, images as b
    WHERE AI.IF(('same colour?', a.ref, b.ref), {AI_TAIL})"""
    (s,) = sites(sql)
    assert s.shape == "pairwise"
    assert "self-join" in s.reason
    assert s.aliases == ("a", "b")


def test_self_join_through_a_cte_is_pairwise():
    sql = f"""
    WITH sel AS (SELECT * FROM styles WHERE price < 800)
    SELECT p1.id FROM sel p1 LEFT OUTER JOIN sel p2 ON p1.uri != p2.uri
      AND AI.IF(('same category?', p1.ref, p2.ref), {AI_TAIL})"""
    (s,) = sites(sql)
    assert s.shape == "pairwise"
    assert "sel" in s.reason


# --- shape: join connectivity ---------------------------------------------

def test_two_aliases_tied_by_equi_joins_are_per_row():
    """The discriminator is connectivity, not alias count: these denote one product."""
    sql = f"""
    SELECT s.id FROM styles s
    JOIN mapping m ON m.link = s.imageURL
    JOIN images i ON i.filename = m.filename
    WHERE AI.IF(('matches?', i.ref, s.title), {AI_TAIL})"""
    (s,) = sites(sql)
    assert s.shape == "per_row"
    assert "equi-joins" in s.reason


def test_two_aliases_joined_only_by_the_ai_predicate_are_pairwise():
    """If the AI call IS the join condition, its operands were never paired 1:1."""
    sql = f"""
    SELECT s.id FROM styles as s
    JOIN images as i ON AI.IF(('image fits description?', i.ref, s.title), {AI_TAIL})"""
    (s,) = sites(sql)
    assert s.shape == "pairwise"
    assert "not connected" in s.reason


def test_an_ai_join_condition_cannot_supply_its_own_connectivity():
    """Regression: blanking AI spans before building the graph. If the AI predicate
    contributed an edge it would declare its own operands joined -> per_row."""
    sql = f"""
    SELECT s.id FROM styles as s
    JOIN images as i ON AI.IF(('x', i.ref, s.title), {AI_TAIL})
    WHERE AI.IF(('y', i.ref), {AI_TAIL})"""
    join_site, row_site = sites(sql)
    assert join_site.shape == "pairwise"
    assert row_site.shape == "per_row"


def test_a_call_outside_the_join_clause_is_still_pairwise_when_unjoined():
    """q14's AI.SCORE lives in ORDER BY, not in a JOIN ... ON, yet ranges over the
    same unjoined cross product. A 'is it inside ON?' rule would get this wrong."""
    sql = f"""
    SELECT s.id, ARRAY_AGG(i.uri ORDER BY AI.SCORE(('fits?', i.ref, s.title), {AI_TAIL}))
    FROM styles as s JOIN images as i ON AI.IF(('fits?', i.ref, s.title), {AI_TAIL})"""
    score_site = next(s for s in sites(sql) if s.kind == "score")
    assert score_site.shape == "pairwise"


def test_named_arguments_do_not_create_join_edges():
    """`=>` must not be read as `=`; otherwise every AI call's own named args would
    wire unrelated aliases together."""
    sql = f"""
    SELECT s.id FROM styles as s JOIN images as i ON AI.IF(('x', i.ref, s.title), {AI_TAIL})"""
    assert P.join_components(sql) == [] or all(
        len(c) < 2 for c in P.join_components(sql))


@pytest.mark.parametrize("op", [">=", "<=", "!="])
def test_inequalities_do_not_create_join_edges(op):
    sql = f"""
    SELECT s.id FROM styles as s JOIN images as i ON i.price {op} s.price
    WHERE AI.IF(('x', i.ref, s.title), {AI_TAIL})"""
    (s,) = sites(sql)
    assert s.shape == "pairwise", f"{op} is not an equi-join and must not tie aliases"


# --- prompt / literal handling --------------------------------------------

def test_triple_quoted_prompt_is_extracted_whole():
    sql = f"""SELECT id FROM images WHERE AI.IF(('''line one
line two, with a comma and a ) paren''', images.ref), {AI_TAIL})"""
    (s,) = sites(sql)
    assert "line one" in s.prompt and "line two" in s.prompt
    assert ")" in s.prompt


def test_interleaved_literals_concatenate_in_order():
    sql = (f"SELECT id FROM images WHERE AI.IF(('The image ', images.ref, "
           f"' fits: ', images.title), {AI_TAIL})")
    (s,) = sites(sql)
    assert s.prompt == "The image  fits:"
    assert s.columns == ("images.ref", "images.title")


def test_if_has_a_boolean_value_space_for_oracle_guidance():
    sql = f"SELECT id FROM images WHERE AI.IF(('is it a shoe?', images.ref), {AI_TAIL})"
    (s,) = sites(sql)
    assert s.choices == ("true", "false")


def test_concatenated_prompt_keeps_column_placeholders():
    sql = f"""SELECT Airlines FROM airport
    WHERE AI.IF("Given destinations '" || Destinations || "' of " || Airlines ||
                ", the airline has flights to Europe.", {AI_TAIL})"""
    (s,) = sites(sql)
    assert s.prompt == (
        "Given destinations '{Destinations}' of {Airlines}, "
        "the airline has flights to Europe."
    )


def test_unqualified_predicate_columns_are_recorded():
    """mmqa q6 names columns without aliases; losing them makes validation read the
    unrelated last CSV column (`Airport`) instead of Airlines + Destinations."""
    sql = f"""SELECT Airlines FROM mmqa.tampa_international_airport
    WHERE AI.IF('Destinations: ' || Destinations || ', airline: ' || Airlines,
                {AI_TAIL})"""
    (s,) = sites(sql)
    assert s.columns == ("Destinations", "Airlines")
    assert s.aliases == ()


def test_a_column_name_inside_the_prompt_text_is_not_a_column_ref():
    sql = (f"SELECT id FROM images WHERE AI.IF(('mentions styles.title verbatim', "
           f"images.ref), {AI_TAIL})")
    (s,) = sites(sql)
    assert s.aliases == ("images",)


def test_nested_parens_in_the_tuple_do_not_end_the_call():
    sql = (f"SELECT id FROM images WHERE AI.IF(('a (pair of) shoe(s)', images.ref), "
           f"{AI_TAIL})")
    (s,) = sites(sql)
    assert s.prompt == "a (pair of) shoe(s)"


# --- kinds -----------------------------------------------------------------

def test_classify_categories_become_choices():
    sql = f"""SELECT AI.CLASSIFY(('classify it: ', images.ref),
        categories => [('Dress', 'a one-piece garment'), ('Socks', 'worn on feet')],
        {AI_TAIL}) FROM images"""
    (s,) = sites(sql)
    assert s.kind == "classify"
    assert s.choices == ("Dress", "Socks")


def test_generate_is_recognised_and_has_no_choices():
    sql = f"SELECT AI.GENERATE(('extract the colour: ', images.ref), {AI_TAIL}) FROM images"
    (s,) = sites(sql)
    assert s.kind == "generate" and s.choices == ()


def test_multiple_call_sites_are_returned_in_source_order():
    sql = f"""SELECT id FROM images
    WHERE AI.IF(('first', images.ref), {AI_TAIL})
      AND AI.IF(('second', images.ref), {AI_TAIL})"""
    assert [s.prompt for s in sites(sql)] == ["first", "second"]


def test_a_query_with_no_ai_call_yields_nothing():
    assert sites("SELECT id FROM images WHERE price < 10") == []


# --- pick_site -------------------------------------------------------------

def _two_sites():
    return sites(f"""SELECT id FROM images
    WHERE AI.IF(('first', images.ref), {AI_TAIL})
      AND AI.IF(('second', images.ref), {AI_TAIL})""")


def test_pick_site_refuses_to_guess_between_several():
    """Silently taking the first would build a val set for a different question
    than the caller believes it asked for."""
    with pytest.raises(ValueError, match="--call-site"):
        P.pick_site(_two_sites(), None, None)


def test_pick_site_error_lists_the_indices_to_choose_from():
    with pytest.raises(ValueError) as exc:
        P.pick_site(_two_sites(), None, None)
    assert "[0]" in str(exc.value) and "[1]" in str(exc.value)


def test_pick_site_by_index():
    assert P.pick_site(_two_sites(), 1, None).prompt == "second"


def test_pick_site_rejects_an_out_of_range_index():
    with pytest.raises(ValueError, match="out of range"):
        P.pick_site(_two_sites(), 5, None)


def test_pick_site_by_shape_when_it_is_unambiguous():
    sql = f"""SELECT a.id FROM images as a, images as b
    WHERE AI.IF(('is shoe', a.ref), {AI_TAIL})
      AND AI.IF(('same colour', a.ref, b.ref), {AI_TAIL})"""
    all_sites = sites(sql)
    assert P.pick_site(all_sites, None, "per_row").prompt == "is shoe"
    assert P.pick_site(all_sites, None, "pairwise").prompt == "same colour"


def test_pick_site_by_shape_reports_when_that_shape_is_absent():
    sql = f"SELECT id FROM images WHERE AI.IF(('x', images.ref), {AI_TAIL})"
    with pytest.raises(ValueError, match="no pairwise call site"):
        P.pick_site(sites(sql), None, "pairwise")


def test_identical_call_sites_report_distinct_indices():
    """Frozen dataclasses compare by value, so an index derived with list.index()
    would report [0] twice for two equal sites."""
    sql = f"""SELECT a.id FROM images as a, images as b, images as c
    WHERE AI.IF(('same', a.ref, b.ref), {AI_TAIL})
      AND AI.IF(('same', a.ref, b.ref), {AI_TAIL})"""
    with pytest.raises(ValueError) as exc:
        P.pick_site(sites(sql), None, None)
    assert "[0]" in str(exc.value) and "[1]" in str(exc.value)


# --- the real corpus -------------------------------------------------------

EXPECTED = {
    # query: (n_per_row, n_pairwise)
    "q2":  (1, 0),   # single AI.IF over images.ref
    "q4":  (1, 0),   # AI.GENERATE colour extraction
    "q6":  (1, 0),   # AI.CLASSIFY with 5 categories
    "q8":  (0, 1),   # JOIN ... ON AI.IF
    "q9":  (0, 1),   # self-join p1/p2
    "q10": (3, 2),   # 3 base filters + 2 same-colour-brand pair predicates
    "q11": (4, 3),
    "q12": (2, 0),
    "q13": (1, 0),   # 2 aliases, but chained equi-joins
    "q14": (1, 2),   # 'white socks' per-row; AI.SCORE + AI.IF pairwise
}


@has_ecomm
@pytest.mark.parametrize("query,counts", sorted(EXPECTED.items()))
def test_real_ecomm_queries_classify_as_expected(query, counts):
    found = P.parse_file(os.path.join(ECOMM, f"{query}.sql"))
    per_row = sum(1 for s in found if s.shape == "per_row")
    pairwise = sum(1 for s in found if s.shape == "pairwise")
    assert (per_row, pairwise) == counts


@has_ecomm
def test_every_real_call_site_has_a_nonempty_prompt():
    """An empty prompt means the literal scanner lost the question -- the oracle
    would then be asked nothing at all."""
    for query in EXPECTED:
        for site in P.parse_file(os.path.join(ECOMM, f"{query}.sql")):
            assert site.prompt.strip(), f"{query} line {site.line} has an empty prompt"


@has_ecomm
def test_q6_choices_come_from_the_sql():
    (site,) = P.parse_file(os.path.join(ECOMM, "q6.sql"))
    assert set(site.choices) == {"Dress", "Bottomwear", "Socks", "Topwear", "Innerwear"}
