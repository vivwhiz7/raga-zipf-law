# raga-zipf-law

Code for *Zipf–Mandelbrot scaling in Hindustani raga notation* (Vivaan Gupta, 2026).

The analysis fits Zipf–Mandelbrot rank–frequency laws to 116 notated compositions in ragas Bhairav, Todi and Poorvi from the SANGEET dataset, compares three units of analysis (pitch, duration, pitch × duration), checks where each raga's prescribed vadi and samvadi rank, and tests whether the prescribed pakad is over-represented against a within-composition shuffle null.

## Requirements

Python 3 with NumPy and Matplotlib. Results in the paper were produced with Python 3.9.6, NumPy 2.0.2 and Matplotlib 3.9.4.

## Data

The compositions are not included here. Clone the SANGEET dataset (Misra & Chattopadhyay, 2023) so that it sits **next to** this folder:

```
parent/
├── raga-zipf-law/      (this repository)
└── Sangeet/            (git clone https://github.com/cmisra/Sangeet)
    └── Bhatkhande Dataset/
```

The scripts read `../Sangeet/Bhatkhande Dataset/{Bhairav,Todi,Poorvi}-*.xml`.

## Run

```bash
python3 run_final.py        # fits, bootstrap, permutation tests (about 2 minutes); writes output/results.json
python3 make_figs_final.py  # figures, 300 dpi
```

Both write to `output/`. The numbers reported in the paper are in `output/results.json`.

## Files

| File | Role |
|---|---|
| `zm_core.py` | XML parsing, taal-aware durations in matras, token schemes, rank–frequency |
| `zm_fit2.py` | Normalised Zipf–Mandelbrot fitting with the identifiability test |
| `zm_ngram.py` | Svara n-grams (within compositions) and pakad extraction |
| `zm_stats.py` | Composition-level bootstrap and pakad permutation test |
| `run_final.py` | Runs the full analysis and prints a log |
| `make_figs_final.py` | Figures 1–4 |

## Method notes

**Fitting.** The model is the normalised Zipf–Mandelbrot law, f(r) = (r + q)^−s / Σ(r + q)^−s. It is fitted by a grid search over q (0 to 400) and s (steps of 0.05), keeping the pair with the highest R² in log–log space, then refined on a finer local grid.

**Identifiability.** Each fit is run with s ceilings of 6, 12 and 25. If s still sits at the largest ceiling, the exponent is unbounded and no fit is reported. All three pitch-only distributions fail this test.

**Bootstrap.** Compositions, not notes, are resampled (200 replicates, seed 42), because notes within a composition are autocorrelated. The resampling loop uses a coarser s grid (step 0.10) than the point fit, so a few point estimates sit just outside their interval.

**Permutation test.** Svaras are shuffled within each composition (300 permutations, seed 7), keeping each composition's notes and destroying their order. The statistic is the median rank of the pakad's n-grams; pakad n-grams that do not occur are ranked one below the last observed rank. The one-sided p-value is (k + 1)/(300 + 1), where k is the number of shuffles that rank the pakad at least as well as the real data.

**Parser limitation.** Grace notes (`<sup>`), sustains (`-`) and the gamak marker (`@`) are handled. The markers `` ` ``, `!`, `#` and `(m)` are not, so the notes they mark are counted as ordinary svaras.

## Licence

MIT. See `LICENSE`.
