"""
Test suite for the anchored TPP reference implementation.

Run with:   python -m pytest tests -q       (or:  python tests/test_atpp.py)

The regression tests at the bottom pin the published figures of the paper.  If
one of them fails after you change the generator or the engine, the paper's
numbers no longer match the code -- fix the code or regenerate every figure.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atpp import (AnchoredTPP, Standardiser, generate_core, review_list,
                  best_sign_review, selftest_invariance, ELEMENTS, IDX,
                  VISIBLE_MUD, HIDDEN_MUD)
from atpp.compositional import clr, multiplicative_replacement
from atpp.baselines import single_element_anchor_selected


# ----------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def bench():
    X, truth = generate_core(seed=42)
    return X, truth


@pytest.fixture(scope="module")
def fitted(bench):
    X, truth = bench
    anchors = np.where(truth == VISIBLE_MUD)[0][:20]
    eng = AnchoredTPP(missing="zero").fit_data(X, feature_names=ELEMENTS)
    eng.fit(anchors, seed=0)
    return eng, truth, anchors


# ----------------------------------------------------------------- algebra
def test_orthonormal(fitted):
    eng, _, _ = fitted
    assert np.allclose(eng.W_.T @ eng.W_, np.eye(2), atol=1e-10)


def test_proposition1_omega_bg_is_inert(fitted):
    """Rows of T outside the anchor set are zero, so omega_bg cancels exactly."""
    eng, _, anchors = fitted
    W0 = eng.W_
    for wb in [1e-6, 1e-3, 0.05, 0.5, 1.0, 10.0]:
        e = AnchoredTPP(missing="zero").fit_data(eng.Xs_ * 0 + eng.Xs_)
        e.standardise = False
        e.Xs_ = eng.Xs_
        e.fit(anchors, background_weight=wb, seed=0)
        assert np.abs(e.W_ - W0).max() < 1e-12, f"omega_bg={wb} changed the fit"


def test_proposition1_pull_scale_is_inert(fitted):
    eng, _, anchors = fitted
    W0 = eng.W_
    for c in [0.1, 1.0, 10.0, 1000.0]:
        e = AnchoredTPP(standardise=False, missing="zero")
        e.Xs_ = eng.Xs_
        e.n_, e.p_ = eng.Xs_.shape
        e.feature_names_ = eng.feature_names_
        e.fit(anchors, pull=3.0 * c, jitter=0.3 * c, seed=0)
        assert np.abs(e.W_ - W0).max() < 1e-10, f"pull scale {c} changed the fit"


def test_proposition2_fit_uses_only_anchor_rows(fitted):
    """Perturbing every non-anchor row must leave the projection unchanged."""
    eng, _, anchors = fitted
    Xs = eng.Xs_.copy()
    mask = np.ones(len(Xs), bool)
    mask[anchors] = False
    Xs[mask] += 10.0 * np.random.default_rng(1).normal(size=Xs[mask].shape)
    e = AnchoredTPP(standardise=False, missing="zero").fit_data(Xs).fit(anchors, seed=0)
    assert np.abs(e.W_ - eng.W_).max() < 1e-12


def test_first_axis_tracks_anchor_centroid(fitted):
    eng, _, _ = fitted
    assert eng.centroid_alignment() > 0.99


def test_selftest_passes():
    out = selftest_invariance(verbose=False)
    assert out["max_dev_omega_bg"] < 1e-10
    assert out["max_dev_rho"] < 1e-10
    assert out["max_dev_background_perturbation"] < 1e-10


# ----------------------------------------------------------------- deployment safety
def test_standardiser_reuses_parameters():
    """The v2.0 bug: applying a fitted projection to separately-standardised
    data. Standardiser must reuse the fitted centre and scale."""
    rng = np.random.default_rng(0)
    A = rng.normal(0, 1, (200, 5))
    B = rng.normal(4, 3, (50, 5))       # different location and scale
    sc = Standardiser().fit(A)
    Bs = sc.transform(B)
    assert not np.allclose(Bs.mean(0), 0, atol=0.5), \
        "transform must NOT re-centre on B's own moments"
    assert np.allclose(sc.transform(A).mean(0), 0, atol=1e-10)


def test_transform_new_well_is_consistent(fitted):
    eng, truth, _ = fitted
    X_new, _ = generate_core(seed=7)
    Y = eng.transform(X_new)
    assert Y.shape == (len(X_new), 2)
    assert np.isfinite(Y).all()


def test_transform_rejects_wrong_width(fitted):
    eng, _, _ = fitted
    with pytest.raises(ValueError):
        eng.transform(np.zeros((10, 3)))


def test_missing_values_raise_by_default():
    X, truth = generate_core(seed=42)
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        AnchoredTPP().fit_data(X)


def test_missing_values_imputed_when_asked():
    X, truth = generate_core(seed=42)
    X[0, 0] = np.nan
    eng = AnchoredTPP(missing="mean").fit_data(X)
    assert np.isfinite(eng.Xs_).all()


def test_bad_anchor_indices_rejected(bench):
    X, _ = bench
    eng = AnchoredTPP(missing="zero").fit_data(X)
    with pytest.raises(ValueError):
        eng.fit([5, 99999])
    with pytest.raises(ValueError):
        eng.fit([])


def test_single_anchor_warns_but_works(bench):
    X, truth = bench
    eng = AnchoredTPP(missing="zero").fit_data(X)
    with pytest.warns(UserWarning):
        eng.fit([int(np.where(truth == VISIBLE_MUD)[0][0])])
    assert np.isfinite(eng.scores()).all()


def test_fit_before_fit_data_raises():
    with pytest.raises(RuntimeError):
        AnchoredTPP().fit([1, 2, 3])


def test_anchor_order_preserved(bench):
    """Anchor order must survive de-duplication: the axis-2 jitter is assigned
    sequentially, so sorting would silently change the projection."""
    X, truth = bench
    a = np.where(truth == VISIBLE_MUD)[0][:10]
    eng = AnchoredTPP(missing="zero").fit_data(X)
    W1 = eng.fit(a, seed=0).W_.copy()
    W2 = eng.fit(np.concatenate([a, a]), seed=0).W_.copy()  # duplicates only
    assert np.abs(W1 - W2).max() < 1e-12


# ----------------------------------------------------------------- compositional
def test_multiplicative_replacement_preserves_ratios():
    A = np.array([[10.0, 20.0, 0.0, 70.0]])
    B = multiplicative_replacement(A)
    assert (B > 0).all()
    assert abs(B[0, 1] / B[0, 0] - 2.0) < 1e-9
    assert abs(B.sum() - A.sum()) < 1e-9


def test_clr_rows_sum_to_zero():
    rng = np.random.default_rng(0)
    A = np.abs(rng.normal(10, 3, (20, 6)))
    L = clr(A)
    assert np.allclose(L.sum(axis=1), 0, atol=1e-10)


def test_clr_rejects_negatives():
    with pytest.raises(ValueError):
        clr(np.array([[1.0, -1.0, 2.0]]))


# ----------------------------------------------------------------- protocol
def test_review_list_excludes_visible(fitted):
    eng, truth, _ = fitted
    r = review_list(eng.scores(), truth, 15)
    assert not (truth[r["top_idx"]] == VISIBLE_MUD).any()
    assert r["total_hidden"] == 15


def test_best_sign_review_is_at_least_as_good(fitted):
    eng, truth, _ = fitted
    s = eng.scores()
    a = review_list(s, truth, 15)["found_hidden_mud"]
    b = best_sign_review(s, truth, 15)["found_hidden_mud"]
    assert b >= a


def test_review_list_rejects_oversized_k(fitted):
    eng, truth, _ = fitted
    with pytest.raises(ValueError):
        review_list(eng.scores(), truth, 100000)


# ----------------------------------------------------------------- regression
# These pin the published numbers. Do not "fix" a failure by editing the target.
def test_published_loadings(fitted):
    eng, _, _ = fitted
    load = dict(eng.loadings())
    assert load["Cl"] == pytest.approx(0.856, abs=5e-3)
    assert load["Ba"] == pytest.approx(0.411, abs=5e-3)
    assert load["K"] == pytest.approx(0.216, abs=5e-3)
    assert load["Sr"] == pytest.approx(0.123, abs=5e-3)
    assert load["S"] == pytest.approx(-0.092, abs=5e-3), \
        "the negative sulphur weight is the paper's headline diagnostic result"


def test_published_headline_recovery(fitted):
    eng, truth, _ = fitted
    assert review_list(eng.scores(), truth, 15)["found_hidden_mud"] == 11


def test_published_baselines_recover_zero(fitted):
    """Both baselines fail under EITHER sign -- the headline comparison is not
    an artefact of principal-component sign ambiguity."""
    eng, truth, _ = fitted
    from atpp.baselines import pca_scores
    assert best_sign_review(eng.Xs_[:, IDX["Ba"]], truth, 15)["found_hidden_mud"] == 0
    assert best_sign_review(pca_scores(eng.Xs_)[:, 0], truth, 15)["found_hidden_mud"] == 0


def test_known_benchmark_limitation(fitted):
    """DOCUMENTED WEAKNESS, asserted so it cannot be forgotten.

    On the published configuration a label-free anchor-selected single-element
    rule beats anchored TPP, because the generator gives Cl no natural variance
    and so leaves it a pure marker. See exp_single_element.py and the CHANGELOG.
    """
    eng, truth, anchors = fitted
    one_el = review_list(single_element_anchor_selected(eng.Xs_, anchors),
                         truth, 15)["found_hidden_mud"]
    tpp = review_list(eng.scores(), truth, 15)["found_hidden_mud"]
    assert one_el >= tpp, (
        "The single-element shortcut appears to have been closed -- if you "
        "changed the generator, update the paper's Section 5 discussion too.")


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
