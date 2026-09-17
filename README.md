# Anchored Targeted Projection Pursuit — reference implementation v3.2

Companion code for Li, Shahrani and Yuan, *"Anchored Targeted Projection Pursuit:
Hypothesis-Driven Quality Control of Multivariate Geochemical Data from Sparse
Analyst Anchors"*, submitted to the *Journal of Geo-Energy and Environment* (ICCK).

Licence: MIT for the code, CC-BY-4.0 for figures and result files. See
`LICENSE`.

---

## Read this first

Five things change how the results should be quoted. None of them break the
method; four of them narrow what it can claim.

1. **There are no hyperparameters.** The background weight `omega_bg` and the
   pull magnitude `rho` have *exactly zero* effect on the fitted projection.
   Rows of the target outside the anchor set are identically zero, so
   `omega_bg` multiplies zero and cancels. Verify with `selftest_invariance()`.

2. **The closed form solves the alignment objective, not least squares.**
   Earlier docstrings said `W = UVᵀ` minimises `‖Ω^½(XW − T)‖²_F`. For a
   rectangular `W` that is false — the dispersion term `tr(WᵀXᵀΩXW)` is
   constant only when `W` is square. `W = UVᵀ` exactly maximises `tr(WᵀXᵀΩT)`,
   the classical orthogonal Procrustes solution (Schönemann 1966).
   `fit_least_squares()` provides the iterative alternative for comparison.

3. **The method is not label-free.** It uses sparse *positive* anchors and no
   negative labels — the same information one-class and PU learning consume.
   What distinguishes anchored TPP is interpretable loadings, absence of
   tuning, well-posedness at `p ≫ n` and interactive speed — not superior
   recall.

4. **Anchor economy is real but not a plateau.** Earlier drafts said recovery
   was flat beyond about five anchors. It is not. On the synthetic benchmark,
   averaged over 30 anchor draws, recovery climbs monotonically across the
   whole range tested (`exp_anchor_economy.py`):

   | m | 1 | 2 | 3 | 5 | 8 | 10 | 15 | 20 | 30 | 40 |
   |---|---|---|---|---|---|---|---|---|---|---|
   | of 15 | 3.90 | 4.67 | 5.93 | 6.03 | 7.17 | 8.30 | 9.33 | 9.67 | 11.63 | 12.00 |

   Useful performance is available from a handful of anchors; anchor count is a
   genuine lever, not an inert parameter.

5. **The fully synthetic benchmark has a shortcut. The real-background
   experiment does not, and is the primary evidence.** In the generator, `Cl`
   has no natural geological loading, so the planted signal makes it a ~4σ pure
   marker. A rule that simply picks the element the anchors are most enriched
   in recovers **12.6/15 against anchored TPP's 8.9** there. Run
   `experiments/exp_single_element.py` before quoting any synthetic number.

   On real background (`exp_real_semisup.py`, 60 realisations, of 12 hidden)
   that ordering reverses:

   | method | uses anchors | recovered |
   |---|---|---|
   | blind PCA | no | 0.77 ± 0.09 |
   | single-element K₂O (a priori) | no | 1.97 ± 0.14 |
   | anchored TPP, raw wt% | yes | 3.08 ± 0.24 |
   | single-element, anchor-selected | yes | 3.93 ± 0.18 |
   | one-class SVM | yes | 4.95 ± 0.20 |
   | **anchored TPP, clr** | yes | **7.03 ± 0.26** |
   | PU logistic | yes | **7.58 ± 0.15** |

   Anchored TPP leads every method that does not use anchors, and every
   anchor-using method except PU logistic, which it trails by 0.55 ± 0.24
   (p = 0.035). The clr transform is worth a factor of **2.3** — a larger
   effect than the spread between competing algorithms.

---

## Install

```bash
pip install -r requirements.txt
```

Core engine needs only NumPy. Figures need Matplotlib; baselines need
scikit-learn; the semi-synthetic experiments need pandas and SciPy.

## Verify

```bash
python -m pytest tests -q          # 25 tests, ~1 s
python -c "from atpp import selftest_invariance; selftest_invariance()"
```

## Reproduce

```bash
python experiments/run_all.py                        # everything not needing the CSV
python experiments/run_all.py --csv path/to/XRF.csv  # including semi-synthetic
```

### Getting the real dataset

The semi-synthetic experiments need the published cuttings XRF dataset of
**Hall, B. (2018), *CSEG Recorder***, "Geochemical facies analysis using
unsupervised machine learning". That dataset is third-party and is **not
redistributed in this archive**; obtain it from its original source under
whatever terms that source sets, then pass it with `--csv`.

The scripts expect the columns listed in `experiments/semisynth.py`: nine
normative mineralogical components, twelve major-element oxides, and Zr in ppm,
plus `Well Name` and `Depth`.

To check the pipeline executes before you have it:

```bash
python experiments/_make_demo_csv.py --out /tmp/demo.csv
python experiments/exp_real_compare.py --csv /tmp/demo.csv --n-real 4
```

The stand-in reproduces the column schema and rough covariance of the real
data. It is not the real data, method orderings on it may invert, and numbers
from it must never be quoted. Note that running it overwrites
`results_real_compare.json`; restore that file from the archive afterwards, or
regenerate it with the real CSV.

---

## Figure map

Every figure in the manuscript has a generator here. Figures are referred to by
filename rather than number, because paper numbering has changed during
revision and may change again in proof.

| Paper | File | Generated by | Needs CSV |
|---|---|---|---|
| Fig. 1 | `fig_workflow.pdf` | `make_fig_workflow.py` | no |
| Fig. 2 | `fig_sensitivity.pdf` | `exp_sensitivity.py` | no |
| Fig. 3 | `fig_compare.pdf` | `make_figs.py` | no |
| Fig. 4 | `fig_loadings.pdf` | `make_figs.py` | no |
| Fig. 5 | `fig_robust.pdf` | `make_figs.py` | no |
| Fig. 6 | `fig_spatial.pdf` | `exp_spatial.py` | no |
| Fig. 7 | `fig_real_compare.pdf` | `exp_real_compare.py` | **yes** |
| Fig. 8 | `fig_real_strength.pdf` | `exp_real_compare.py` | **yes** |
| Fig. 9 | `fig_real_semisup.pdf` | `exp_real_semisup.py` | **yes** |

Two scripts produce numbers quoted in the text but no paper figure:
`exp_anchor_economy.py` (anchor economy on the synthetic benchmark) and
`exp_single_element.py` (the benchmark-shortcut diagnostic; its figure
`fig_single_element_check.pdf` is a diagnostic, not a paper figure).

### Scope

This package contains only what the manuscript reports. Experiments whose
sections were cut during revision (the dimensionality and missing-value
ablation, and the synthetic semi-supervised comparison), together with the toy
example and the interactive notebook, were removed in v3.1.

Figures are sized for a two-column journal layout with lettering that survives
typesetting at 8–9 pt. Multi-panel comparisons use a 2×2 layout with a
dedicated legend cell rather than a wide 1×3 strip. Fonts are embedded as
TrueType (`pdf.fonttype: 42`); Type 3 fonts are rejected in production.

---

## Package layout

```
atpp/                 core engine (NumPy only)
  core.py             AnchoredTPP: fit, scores, loadings, least-squares variant
  baselines.py        PCA, single-element, MCD, isolation forest, OC-SVM, PU
  benchmark.py        fully synthetic core generator
  compositional.py    clr, multiplicative replacement
  protocol.py         review-list evaluation
experiments/          one script per manuscript result (see figure map)
tests/                25 tests covering invariance, locality, protocol
figures/              generated PDFs, as used in the manuscript
```

## Results files

| File | Written by | Contents |
|---|---|---|
| `results_v30.json` | `run_all.py` | every number the synthetic sections quote |
| `results_anchor_economy.json` | `exp_anchor_economy.py` | synthetic anchor sweep |
| `results_semisynth_v23.json` | `exp_real_semisup.py` | semi-synthetic recovery table and anchor sweep |
| `results_real_compare.json` | `exp_real_compare.py` | panel counts and strength sweep |

If a number in the manuscript disagrees with these files, the manuscript is
wrong. Regenerate; do not edit the JSON.

### Changes in v3.2

- Added `exp_anchor_economy.py`. The anchor-economy figures quoted in the text
  previously had no generator, and the values in circulation were taken from a
  single anchor draw.
- The spatial-autocorrelation seed base changed from `1000 + s` to `100 + s` in
  v3.0 so that `run_all.py` and `exp_spatial.py` agree. Means shifted by about
  0.5; the manuscript now quotes the current values
  (8.88 / 7.46 / 5.94 / 5.19 for dispersed anchors at φ = 0 / 0.7 / 0.85 / 0.95).
- `exp_spatial.py` reports **standard errors**, not standard deviations. The
  figure caption in the manuscript was corrected to match.
- The real-data anchor sweep now includes m = 1 and m = 12, 18.
- `XRF_dataset.csv` removed; see "Getting the real dataset".
- Added `LICENSE`, `CITATION.cff` and `.zenodo.json`.

---

## Citation

Li, Y., Shahrani, S. S., & Yuan, Z. Anchored Targeted Projection Pursuit: Hypothesis-Driven
Quality Control of Multivariate Geochemical Data from Sparse Analyst Anchors.
Submitted to the *Journal of Geo-Energy and Environment*.

Software: see `CITATION.cff`; Zenodo DOI 10.5281/zenodo.22668675.

Dataset for the semi-synthetic experiments: Hall, B. (2018), *CSEG Recorder*.
Not redistributed here.
