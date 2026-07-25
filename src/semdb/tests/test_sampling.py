"""Sampling designs: reproducibility, without-replacement, and the weight-sum
identity that makes the downstream estimator unbiased."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sampling as S  # noqa: E402


def _ids(n):
    return [f"m{i}" for i in range(n)]


# --- uniform ---------------------------------------------------------------

def test_uniform_is_reproducible_and_without_replacement():
    a = S.sample_uniform(_ids(200), 60, seed=7)
    b = S.sample_uniform(_ids(200), 60, seed=7)
    assert a.ids == b.ids
    assert len(set(a.ids)) == 60
    assert S.sample_uniform(_ids(200), 60, seed=8).ids != a.ids


def test_uniform_weights_sum_to_the_population():
    s = S.sample_uniform(_ids(200), 60, seed=7)
    s.check()
    assert s.weight_sum() == pytest.approx(200.0)
    assert all(p == pytest.approx(60 / 200) for p in s.pi.values())


def test_uniform_rejects_impossible_sizes():
    with pytest.raises(ValueError):
        S.sample_uniform(_ids(10), 11, seed=1)
    with pytest.raises(ValueError):
        S.sample_uniform(_ids(10), 0, seed=1)


# --- allocation ------------------------------------------------------------

def test_allocation_sums_exactly_and_respects_bounds():
    sizes = {"a": 800, "b": 150, "c": 50}
    alloc = S.allocate(sizes, 120, min_per_stratum=2)
    assert sum(alloc.values()) == 120
    assert all(2 <= alloc[h] <= sizes[h] for h in sizes)
    # proportional: the big stratum gets the most
    assert alloc["a"] > alloc["b"] > alloc["c"]


def test_allocation_handles_a_stratum_smaller_than_its_share():
    # 'c' would get ~40 proportionally but only has 3 rows; the rest must absorb it.
    alloc = S.allocate({"a": 100, "b": 100, "c": 3}, 60, min_per_stratum=1)
    assert sum(alloc.values()) == 60
    assert alloc["c"] <= 3


def test_neyman_needs_sigma_and_shifts_mass_to_the_noisy_stratum():
    sizes = {"a": 100, "b": 100}
    with pytest.raises(ValueError):
        S.allocate(sizes, 40, allocation="neyman")
    alloc = S.allocate(sizes, 40, allocation="neyman", sigma={"a": 0.5, "b": 0.1})
    assert sum(alloc.values()) == 40
    assert alloc["a"] > alloc["b"]


def test_allocation_rejects_an_n_below_the_per_stratum_floor():
    with pytest.raises(ValueError):
        S.allocate({"a": 10, "b": 10, "c": 10}, 4, min_per_stratum=2)


# --- stratified ------------------------------------------------------------

def _strata(ids):
    return {i: f"c{hash(i) % 4}" for i in ids}


def test_stratified_weights_sum_to_the_population():
    ids = _ids(500)
    s = S.sample_stratified(ids, _strata(ids), 100, seed=3, min_per_stratum=2)
    s.check()
    assert s.n == 100
    assert s.weight_sum() == pytest.approx(500.0)


def test_stratified_allocation_is_recorded_and_consistent():
    ids = _ids(500)
    s = S.sample_stratified(ids, _strata(ids), 100, seed=3, min_per_stratum=2)
    table = s.meta["strata"]
    assert sum(v["n_h"] for v in table.values()) == 100
    for h, v in table.items():
        drawn = [i for i in s.ids if s.stratum[i] == h]
        assert len(drawn) == v["n_h"]
        assert all(s.pi[i] == pytest.approx(v["n_h"] / v["N_h"]) for i in drawn)


def test_stratified_rejects_a_zero_allocation_floor():
    """n_h = 0 gives those units pi = 0, which no unbiased estimator can handle."""
    ids = _ids(100)
    with pytest.raises(ValueError, match="zero inclusion probability"):
        S.sample_stratified(ids, _strata(ids), 20, seed=1, min_per_stratum=0)


def test_stratified_refuses_more_strata_than_the_sample_size():
    ids = _ids(100)
    one_each = {i: i for i in ids}
    with pytest.raises(ValueError, match="stratify more coarsely"):
        S.sample_stratified(ids, one_each, 20, seed=1)


# --- stratum collapsing ----------------------------------------------------

def test_collapse_merges_only_the_small_strata():
    stratum_of = {**{f"m{i}": "big" for i in range(50)},
                  **{f"t{i}": f"tiny{i}" for i in range(5)}}
    merged, report = S.collapse_small_strata(stratum_of, min_size=2)
    assert merged["m0"] == "big"
    assert len({merged[f"t{i}"] for i in range(5)}) == 1     # all tinies now share one
    assert report["merged"] == sorted(f"tiny{i}" for i in range(5))
    assert report["pooled_size"] == 5


def test_collapse_is_a_no_op_when_every_stratum_is_big_enough():
    stratum_of = {f"m{i}": f"c{i % 4}" for i in range(100)}
    merged, report = S.collapse_small_strata(stratum_of, min_size=2)
    assert merged == stratum_of and report["merged"] == []


def test_collapse_never_drops_a_row():
    stratum_of = {f"m{i}": f"c{i}" for i in range(30)}       # all size 1
    merged, _ = S.collapse_small_strata(stratum_of, min_size=2)
    assert set(merged) == set(stratum_of)
    assert len(set(merged.values())) == 1                    # everything pooled


def test_collapse_folds_an_undersized_pool_into_a_real_stratum():
    """When the pooled remainder is itself too small it must join a survivor,
    not stand alone as a stratum that cannot be split."""
    stratum_of = {**{f"m{i}": "big" for i in range(50)},
                  **{f"s{i}": "mid" for i in range(6)},
                  "lonely": "tiny"}
    merged, report = S.collapse_small_strata(stratum_of, min_size=3)
    assert merged["lonely"] == "mid"                          # smallest survivor
    assert report["pooled_into"] == "mid"
    assert S.POOLED not in set(merged.values())


def test_stratified_rejects_ids_without_a_stratum():
    ids = _ids(20)
    partial = {i: "a" for i in ids[:10]}
    with pytest.raises(ValueError, match="no stratum"):
        S.sample_stratified(ids, partial, 5, seed=1)


# --- importance / pps ------------------------------------------------------

def test_mixed_proposal_is_a_distribution_with_a_floor():
    q = S.mixed_proposal([10.0, 1.0, 0.0, 0.0], epsilon=0.2)
    assert q.sum() == pytest.approx(1.0)
    assert (q >= 0.2 / 4 - 1e-12).all()          # uniform floor holds even for score 0
    assert q[0] > q[1] > q[2]


def test_mixed_proposal_survives_degenerate_scores():
    q = S.mixed_proposal([0.0, 0.0, 0.0], epsilon=0.2)
    assert q == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    q2 = S.mixed_proposal([np.nan, -5.0, 2.0], epsilon=0.5)
    assert np.isfinite(q2).all() and q2.sum() == pytest.approx(1.0)


def test_pps_draws_the_requested_size_without_replacement():
    ids = _ids(200)
    scores = [float(i) for i in range(200)]
    s = S.sample_pareto_pps(ids, scores, 60, seed=5)
    assert s.n == 60 and len(set(s.ids)) == 60
    s.check()


def test_pps_oversamples_high_score_rows():
    """The whole point of the design: with 6% positives, uniform would put ~4 of
    60 labels on them; pps must do materially better."""
    n_pop, n_pos = 500, 30
    ids = _ids(n_pop)
    positives = set(ids[:n_pos])
    scores = [5.0 if i in positives else 1.0 for i in ids]
    caught = [len(set(S.sample_pareto_pps(ids, scores, 60, seed=s).ids) & positives)
              for s in range(20)]
    uniform = [len(set(S.sample_uniform(ids, 60, seed=s).ids) & positives)
               for s in range(20)]
    assert np.mean(caught) > 1.5 * np.mean(uniform)


def test_pps_weights_are_bounded_by_the_epsilon_floor():
    ids = _ids(300)
    scores = [1000.0 if i == "m0" else 0.001 for i in ids]
    eps = 0.3
    s = S.sample_pareto_pps(ids, scores, 50, seed=2, epsilon=eps)
    assert max(s.weights.values()) <= s.meta["w_max"] + 1e-6
    assert s.meta["w_max"] == pytest.approx(300 / (50 * eps))


def test_pps_weight_sum_is_unbiased_for_the_population_size():
    """sum 1/pi is N only in expectation for pps -- average over draws."""
    ids = _ids(400)
    scores = list(np.linspace(0.1, 3.0, 400))
    sums = [S.sample_pareto_pps(ids, scores, 80, seed=s).weight_sum() for s in range(40)]
    assert np.mean(sums) == pytest.approx(400.0, rel=0.06)


def test_pps_handles_certainty_units():
    """A score so dominant that n*q_i >= 1 must be taken with probability 1, not
    given an inclusion probability above 1."""
    ids = _ids(50)
    scores = [100.0] * 5 + [0.001] * 45
    s = S.sample_pareto_pps(ids, scores, 20, seed=1, epsilon=0.01)
    assert all(0 < p <= 1.0 for p in s.pi.values())
    assert s.meta["n_certainty"] >= 1
    for i in ids[:5]:
        assert i in s.ids


def test_pps_is_reproducible():
    ids, scores = _ids(100), list(np.linspace(0, 1, 100))
    assert S.sample_pareto_pps(ids, scores, 30, seed=4).ids == \
           S.sample_pareto_pps(ids, scores, 30, seed=4).ids


# --- split -----------------------------------------------------------------

@pytest.mark.parametrize("method", ["uniform", "stratified", "pareto_pps"])
def test_split_is_disjoint_and_covers_the_parent(method):
    ids = _ids(400)
    if method == "uniform":
        parent = S.sample_uniform(ids, 100, seed=9)
    elif method == "stratified":
        parent = S.sample_stratified(ids, _strata(ids), 100, seed=9, min_per_stratum=2)
    else:
        parent = S.sample_pareto_pps(ids, list(np.linspace(0.1, 2, 400)), 100, seed=9)
    select, cert = S.split_sample(parent, 40, seed=10)
    assert select.n == 60 and cert.n == 40
    assert not (set(select.ids) & set(cert.ids))
    assert set(select.ids) | set(cert.ids) == set(parent.ids)


def test_split_preserves_the_weight_sum_in_each_half():
    ids = _ids(400)
    parent = S.sample_uniform(ids, 100, seed=9)
    select, cert = S.split_sample(parent, 40, seed=10)
    select.check()
    cert.check()
    assert select.weight_sum() == pytest.approx(400.0)
    assert cert.weight_sum() == pytest.approx(400.0)


def test_split_keeps_strata_balanced_in_both_halves():
    ids = _ids(400)
    parent = S.sample_stratified(ids, _strata(ids), 120, seed=9, min_per_stratum=2)
    select, cert = S.split_sample(parent, 40, seed=10)
    for h in {v for v in parent.stratum.values()}:
        in_parent = sum(1 for i in parent.ids if parent.stratum[i] == h)
        in_cert = sum(1 for i in cert.ids if cert.stratum[i] == h)
        # cert takes ~1/3 of every stratum, not 100% of one and 0% of another
        assert abs(in_cert / in_parent - 40 / 120) < 0.2


def test_split_with_zero_cert_returns_the_whole_sample_as_select():
    parent = S.sample_uniform(_ids(100), 30, seed=1)
    select, cert = S.split_sample(parent, 0, seed=2)
    assert cert.n == 0 and select.n == 30
    select.check()


def test_split_rejects_a_cert_size_it_cannot_honour():
    parent = S.sample_uniform(_ids(100), 30, seed=1)
    with pytest.raises(ValueError):
        S.split_sample(parent, 30, seed=2)
