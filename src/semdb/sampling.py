"""Probability sampling designs for validation-set construction.

Pure functions: no I/O, no model calls, no global state. Every design draws
WITHOUT replacement from a finite population of row ids and returns a `Sample`
carrying, for each drawn id, its first-order inclusion probability `pi_i` under
the design. The Horvitz-Thompson weight is `w_i = 1 / pi_i`, and the invariant
that makes downstream estimates unbiased is

    sum_{i in sample} w_i == N        (the population size)

`Sample.check()` asserts it. It holds EXACTLY for `uniform` and `stratified`,
and only IN EXPECTATION for `pareto_pps` -- see `sample_pareto_pps`.

Three designs:

  uniform     SRSWOR. pi_i = n/N for every unit.
  stratified  SRSWOR inside each stratum. pi_i = n_h/N_h. Cuts variance when the
              strata correlate with the quantity being estimated; the strata must
              come from signals available BEFORE any program is generated, or the
              design is defined in terms of the thing it is meant to measure.
  pareto_pps  Unequal-probability WoR sampling with fixed size n and target
              inclusion probabilities proportional to a supplied score (Rosen
              1997). This is the "importance" design: it puts labels where the
              information is, which matters when positives are ~6% of the corpus
              and a uniform sample would spend nearly every label on negatives.

Splitting into SELECT / CERT
----------------------------
`split_sample` draws ONE sample of size n_select + n_cert and randomly partitions
it, rather than drawing twice. Two draws would either overlap (invalidating the
sealed certification set) or force the second draw to come from the population
minus the first, which would certify performance on a subpopulation rather than
on the corpus. A random partition of a valid sample is itself a valid sample of
the same design on the same population, with inclusion probabilities scaled by
the partition fraction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

# Floor on any unit's mixed proposal mass, as a multiple of the uniform mass 1/N.
# Keeps pareto weights finite: w_i <= N / (n * epsilon).
MIN_EPSILON = 1e-6


@dataclass(frozen=True)
class Sample:
    """A drawn sample plus everything needed to estimate from it.

    ids      -- drawn row ids, in draw order (deterministic given the seed).
    pi       -- id -> first-order inclusion probability under the design.
    stratum  -- id -> stratum label ("" when the design is unstratified).
    N        -- population size the sample was drawn from.
    meta     -- design-specific record (allocation table, epsilon, ...), for
                provenance and for the estimator to dispatch on.
    """
    ids: tuple[str, ...]
    pi: Mapping[str, float]
    stratum: Mapping[str, str]
    method: str
    seed: int
    N: int
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def weights(self) -> dict[str, float]:
        """Horvitz-Thompson weights w_i = 1/pi_i."""
        return {i: 1.0 / self.pi[i] for i in self.ids}

    def weight_sum(self) -> float:
        return sum(self.weights.values())

    def check(self, *, rel_tol: float = 1e-9) -> None:
        """Assert the design's invariants. Raises AssertionError on violation.

        The weight-sum identity is exact for equal-probability designs and only
        an expectation for pps, so pps is checked loosely -- a hard equality
        assert there would fail on perfectly valid samples.
        """
        assert len(set(self.ids)) == len(self.ids), "sampled ids repeat (not without-replacement)"
        assert all(0.0 < self.pi[i] <= 1.0 for i in self.ids), "inclusion probability outside (0, 1]"
        total = self.weight_sum()
        if self.method == "pareto_pps":
            assert 0.5 * self.N <= total <= 2.0 * self.N, (
                f"pps weight sum {total:.1f} implausible for N={self.N}")
        else:
            assert math.isclose(total, self.N, rel_tol=max(rel_tol, 1e-9)), (
                f"weights sum to {total} but population is {self.N}")


def _rng(seed: int) -> np.random.Generator:
    """One place to construct the generator, so seeding is uniform across designs."""
    return np.random.default_rng(seed)


def _draw_wor(rng: np.random.Generator, ids: Sequence[str], n: int) -> list[str]:
    """SRSWOR of size n from ids, in draw order."""
    idx = rng.permutation(len(ids))[:n]
    return [ids[i] for i in idx]


# --------------------------------------------------------------------------
# uniform
# --------------------------------------------------------------------------

def sample_uniform(ids: Sequence[str], n: int, seed: int) -> Sample:
    """Simple random sample without replacement. pi_i = n/N for every unit."""
    ids = [str(i) for i in ids]
    N = len(ids)
    if n > N:
        raise ValueError(f"cannot draw n={n} from a population of {N}")
    if n <= 0:
        raise ValueError("n must be positive")
    drawn = _draw_wor(_rng(seed), ids, n)
    pi = n / N
    return Sample(ids=tuple(drawn), pi={i: pi for i in drawn},
                  stratum={i: "" for i in drawn}, method="uniform",
                  seed=seed, N=N, meta={"n": n})


# --------------------------------------------------------------------------
# stratified
# --------------------------------------------------------------------------

def allocate(sizes: Mapping[str, int], n: int, *, allocation: str = "proportional",
             sigma: Mapping[str, float] | None = None, min_per_stratum: int = 1) -> dict[str, int]:
    """Split a total sample size n across strata.

    proportional -- n_h proportional to N_h. Needs no labels, so it is the only
                    allocation available before anything has been labeled.
    neyman       -- n_h proportional to N_h * sigma_h, the variance-minimizing
                    allocation. Requires per-stratum standard deviations from a
                    pilot; passing `sigma` is mandatory for it.

    Guarantees: sum(n_h) == n exactly (largest-remainder rounding), and
    floor_h <= n_h <= N_h for every stratum with N_h > 0, where the effective
    floor is ``min(min_per_stratum, N_h)`` -- a stratum can never be asked for
    more rows than it has, so the floor yields to the stratum size.
    """
    strata = [h for h, size in sizes.items() if size > 0]
    if not strata:
        raise ValueError("no non-empty strata")
    N = sum(sizes[h] for h in strata)
    if n > N:
        raise ValueError(f"cannot draw n={n} from a population of {N}")
    floor = {h: min(min_per_stratum, sizes[h]) for h in strata}
    if sum(floor.values()) > n:
        raise ValueError(
            f"n={n} cannot give {min_per_stratum} to each of {len(strata)} strata; "
            f"raise n to >= {sum(floor.values())} or use fewer strata")

    if allocation == "proportional":
        share = {h: sizes[h] / N for h in strata}
    elif allocation == "neyman":
        if not sigma:
            raise ValueError("neyman allocation requires per-stratum sigma (from a pilot)")
        mass = {h: sizes[h] * float(sigma.get(h, 0.0)) for h in strata}
        total = sum(mass.values())
        if total <= 0:                       # zero variance everywhere -> nothing to optimize
            share = {h: sizes[h] / N for h in strata}
        else:
            share = {h: mass[h] / total for h in strata}
    else:
        raise ValueError(f"unknown allocation {allocation!r}")

    # Largest remainder, then repair the min/max bounds and re-balance so the
    # total still lands exactly on n.
    exact = {h: n * share[h] for h in strata}
    alloc = {h: int(math.floor(exact[h])) for h in strata}
    for h in sorted(strata, key=lambda h: exact[h] - alloc[h], reverse=True):
        if sum(alloc.values()) >= n:
            break
        alloc[h] += 1
    for h in strata:
        alloc[h] = max(floor[h], min(alloc[h], sizes[h]))

    # Bound repair may have moved the total off n; push it back, respecting bounds.
    def _rebalance(alloc: dict[str, int]) -> dict[str, int]:
        guard = 0
        while (delta := sum(alloc.values()) - n) != 0:
            guard += 1
            if guard > 10 * len(strata) + 100:
                raise ValueError(f"cannot allocate n={n} within stratum bounds {dict(sizes)}")
            if delta > 0:                     # over -- take from the largest that can spare
                donors = [h for h in strata if alloc[h] > floor[h]]
                if not donors:
                    raise ValueError(
                        f"n={n} is below the {min_per_stratum}-per-stratum floor for "
                        f"{len(strata)} strata")
                alloc[max(donors, key=lambda h: alloc[h])] -= 1
            else:                             # under -- give to the emptiest that has room
                takers = [h for h in strata if alloc[h] < sizes[h]]
                if not takers:
                    raise ValueError(f"n={n} exceeds the population {N}")
                alloc[min(takers, key=lambda h: alloc[h] / sizes[h])] += 1
        return alloc

    return _rebalance(alloc)


POOLED = "_pooled"


def collapse_small_strata(stratum_of: Mapping[str, str], min_size: int,
                          *, pooled_label: str = POOLED) -> tuple[dict[str, str], dict[str, Any]]:
    """Merge strata smaller than `min_size` into one pooled stratum.

    Unsupervised stratifiers (k-means especially) routinely emit a cluster with a
    handful of members, and such a stratum cannot support an allocation that has
    to survive a SELECT/CERT split. Collapsing is the standard survey fix and is
    always safe: coarser strata give up some variance reduction but introduce no
    bias, whereas dropping the rows would change the population and forcing an
    allocation the stratum cannot fill would produce pi > 1.

    Returns the rewritten mapping and a report of what was merged.
    """
    counts: dict[str, int] = {}
    for h in stratum_of.values():
        counts[str(h)] = counts.get(str(h), 0) + 1
    small = {h for h, c in counts.items() if c < min_size}
    if not small:
        return {i: str(h) for i, h in stratum_of.items()}, {"merged": [], "pooled_size": 0}

    out = {i: (pooled_label if str(h) in small else str(h)) for i, h in stratum_of.items()}
    pooled_size = sum(counts[h] for h in small)

    # If the pooled stratum is itself too small, fold it into the smallest
    # surviving stratum rather than leaving a stratum that cannot be split.
    survivors = {h: c for h, c in counts.items() if h not in small}
    if pooled_size < min_size and survivors:
        target = min(survivors, key=lambda h: survivors[h])
        out = {i: (target if h == pooled_label else h) for i, h in out.items()}
        return out, {"merged": sorted(small), "pooled_size": pooled_size,
                     "pooled_into": target}
    return out, {"merged": sorted(small), "pooled_size": pooled_size,
                 "pooled_into": pooled_label}


def sample_stratified(ids: Sequence[str], stratum_of: Mapping[str, str], n: int, seed: int,
                      *, allocation: str = "proportional",
                      sigma: Mapping[str, float] | None = None,
                      min_per_stratum: int = 1) -> Sample:
    """SRSWOR within each stratum. pi_i = n_h / N_h for i in stratum h.

    Every stratum MUST receive at least one row. A stratum with n_h = 0 gives its
    units pi_i = 0, and no unbiased estimator exists for a population containing
    units the design can never draw -- the sample would silently be a sample of a
    subpopulation. So `min_per_stratum` is floored at 1 and a stratification
    finer than the sample size is rejected rather than quietly degraded.
    """
    ids = [str(i) for i in ids]
    N = len(ids)
    missing = [i for i in ids if i not in stratum_of]
    if missing:
        raise ValueError(f"{len(missing)} ids have no stratum (first: {missing[0]!r})")
    if min_per_stratum < 1:
        raise ValueError("min_per_stratum must be >= 1: a stratum allocated 0 rows gives "
                         "its units zero inclusion probability, which has no estimator")

    members: dict[str, list[str]] = {}
    for i in ids:
        members.setdefault(str(stratum_of[i]), []).append(i)
    sizes = {h: len(v) for h, v in members.items()}
    if len(members) > n:
        raise ValueError(
            f"{len(members)} strata but n={n}: every stratum needs at least one row. "
            f"Raise n to >= {len(members)}, or stratify more coarsely "
            f"(fewer --strata-k, or a lower-cardinality column).")
    alloc = allocate(sizes, n, allocation=allocation, sigma=sigma,
                     min_per_stratum=min_per_stratum)

    rng = _rng(seed)
    drawn: list[str] = []
    pi: dict[str, float] = {}
    stratum: dict[str, str] = {}
    for h in sorted(members):                      # sorted -> reproducible across dict orders
        take = alloc[h]
        picked = _draw_wor(rng, members[h], take)
        drawn.extend(picked)
        for i in picked:
            pi[i] = take / sizes[h]
            stratum[i] = h
    return Sample(ids=tuple(drawn), pi=pi, stratum=stratum, method="stratified",
                  seed=seed, N=N,
                  meta={"n": n, "allocation": allocation,
                        "strata": {h: {"N_h": sizes[h], "n_h": alloc[h]} for h in sorted(members)}})


# --------------------------------------------------------------------------
# importance / pareto pps
# --------------------------------------------------------------------------

def mixed_proposal(scores: Sequence[float], epsilon: float) -> np.ndarray:
    """Normalize scores into a proposal, mixed with the uniform distribution.

    q_i = (1 - epsilon) * s_i / sum(s) + epsilon / N

    The uniform component is what bounds the weights: q_i >= epsilon/N implies
    w_i <= N / (n * epsilon). Post-hoc weight truncation would do the same job
    but biases the estimator; mixing is a design choice with exact weights.
    Non-finite or negative scores are treated as 0.
    """
    if not MIN_EPSILON <= epsilon <= 1.0:
        raise ValueError(f"epsilon must be in [{MIN_EPSILON}, 1], got {epsilon}")
    s = np.asarray(scores, dtype=float)
    s = np.where(np.isfinite(s) & (s > 0), s, 0.0)
    N = len(s)
    total = s.sum()
    base = (s / total) if total > 0 else np.full(N, 1.0 / N)
    return (1.0 - epsilon) * base + epsilon / N


def sample_pareto_pps(ids: Sequence[str], scores: Sequence[float], n: int, seed: int,
                      *, epsilon: float = 0.2) -> Sample:
    """Fixed-size WoR sampling with pi_i approximately proportional to `scores`.

    Pareto order sampling (Rosen 1997): with target inclusion probabilities
    lambda_i = n * q_i, draw u_i ~ U(0,1) and keep the n units with the smallest
    ranking variable

        xi_i = (u_i / (1 - u_i)) * ((1 - lambda_i) / lambda_i)

    Units whose lambda_i would exceed 1 are taken with certainty and the
    remaining size is re-solved against the rest, which is the standard fix and
    is what keeps every pi_i a genuine probability.

    Honest caveat: Rosen's pi_i = lambda_i is an asymptotic identity, not an
    exact one, so `sum 1/pi_i == N` holds in expectation rather than per draw.
    `Sample.check()` therefore only sanity-bounds pps weight sums. Any interval
    built on this design must be valid for weighted bounded variables (a betting
    / WSR bound); Clopper-Pearson does NOT apply.
    """
    ids = [str(i) for i in ids]
    N = len(ids)
    if len(scores) != N:
        raise ValueError(f"got {len(scores)} scores for {N} ids")
    if n > N:
        raise ValueError(f"cannot draw n={n} from a population of {N}")
    if n <= 0:
        raise ValueError("n must be positive")

    q = mixed_proposal(scores, epsilon)

    # Solve for inclusion probabilities: certainty units absorb their own mass.
    lam = np.minimum(n * q, 1.0)
    certain = lam >= 1.0
    guard = 0
    while True:
        guard += 1
        if guard > N + 1:                                  # cannot happen; fail loudly if it does
            raise RuntimeError("pps inclusion-probability solve did not converge")
        n_free = n - int(certain.sum())
        if n_free <= 0:
            break
        free = ~certain
        q_free_total = q[free].sum()
        if q_free_total <= 0:
            break
        lam_free = np.minimum(n_free * q[free] / q_free_total, 1.0)
        newly = np.zeros(N, dtype=bool)
        newly[np.flatnonzero(free)[lam_free >= 1.0]] = True
        if not newly.any():
            lam = np.where(certain, 1.0, 0.0)
            lam[free] = lam_free
            break
        certain |= newly

    rng = _rng(seed)
    order = np.flatnonzero(certain).tolist()
    n_free = n - len(order)
    if n_free > 0:
        free_idx = np.flatnonzero(~certain)
        u = rng.uniform(size=len(free_idx))
        lam_f = lam[free_idx]
        xi = (u / (1.0 - u)) * ((1.0 - lam_f) / lam_f)
        order.extend(free_idx[np.argsort(xi)[:n_free]].tolist())

    drawn = [ids[i] for i in order]
    return Sample(
        ids=tuple(drawn),
        pi={ids[i]: float(lam[i]) for i in order},
        stratum={ids[i]: "" for i in order},
        method="pareto_pps", seed=seed, N=N,
        meta={"n": n, "epsilon": epsilon,
              "n_certainty": int(certain.sum()),
              "w_max": float(N / (n * epsilon)),
              "score_range": [float(np.min(scores)) if N else 0.0,
                              float(np.max(scores)) if N else 0.0]},
    )


# --------------------------------------------------------------------------
# SELECT / CERT split
# --------------------------------------------------------------------------

def split_sample(sample: Sample, n_cert: int, seed: int) -> tuple[Sample, Sample]:
    """Randomly partition one sample into (select, cert), disjoint by construction.

    Partitioning happens WITHIN each stratum, so a stratified sample splits into
    two stratified samples rather than into two samples with lopsided strata.
    Inclusion probabilities are scaled by the realized partition fraction of the
    unit's stratum, which keeps `sum 1/pi == N` in each half.
    """
    if not 0 <= n_cert < sample.n:
        raise ValueError(f"n_cert must be in [0, {sample.n}), got {n_cert}")

    groups: dict[str, list[str]] = {}
    for i in sample.ids:
        groups.setdefault(sample.stratum[i], []).append(i)
    sizes = {h: len(v) for h, v in groups.items()}
    stratified = any(sample.stratum.values())

    if n_cert == 0:
        cert_alloc = {h: 0 for h in groups}
    elif not stratified:
        cert_alloc = allocate(sizes, n_cert, allocation="proportional", min_per_stratum=0)
    else:
        # Both halves must remain valid stratified samples of the SAME population,
        # so every stratum has to appear on both sides -- which needs n_h >= 2.
        thin = {h: size for h, size in sizes.items() if size < 2}
        if thin:
            raise ValueError(
                f"cannot split a stratified sample when {len(thin)} stratum/strata have "
                f"fewer than 2 rows ({dict(list(thin.items())[:3])}...): each stratum must "
                f"appear in both SELECT and CERT. Raise --n/--cert-n or --min-per-stratum "
                f"to >= 2, or stratify more coarsely.")
        cert_alloc = allocate(sizes, n_cert, allocation="proportional", min_per_stratum=1)
        starved = {h: sizes[h] for h in groups if sizes[h] - cert_alloc[h] < 1}
        if starved:
            raise ValueError(
                f"cert split would empty stratum/strata {list(starved)} out of SELECT; "
                f"lower --cert-n or stratify more coarsely")

    rng = _rng(seed)
    select_ids: list[str] = []
    cert_ids: list[str] = []
    pi_sel: dict[str, float] = {}
    pi_cert: dict[str, float] = {}
    for h in sorted(groups):
        member = groups[h]
        perm = rng.permutation(len(member))
        take = cert_alloc[h]
        picked_cert = [member[j] for j in perm[:take]]
        picked_sel = [member[j] for j in perm[take:]]
        f_cert = take / len(member)
        f_sel = 1.0 - f_cert
        for i in picked_cert:
            pi_cert[i] = sample.pi[i] * f_cert
        for i in picked_sel:
            pi_sel[i] = sample.pi[i] * f_sel
        cert_ids.extend(picked_cert)
        select_ids.extend(picked_sel)

    def _part(part_ids: list[str], pi: dict[str, float], role: str) -> Sample:
        return Sample(
            ids=tuple(part_ids), pi=pi,
            stratum={i: sample.stratum[i] for i in part_ids},
            method=sample.method, seed=sample.seed, N=sample.N,
            meta={**dict(sample.meta), "split": role, "split_seed": seed,
                  "n": len(part_ids), "parent_n": sample.n},
        )

    return _part(select_ids, pi_sel, "select"), _part(cert_ids, pi_cert, "cert")
