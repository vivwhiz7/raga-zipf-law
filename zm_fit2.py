"""Identifiability-aware normalised ZM fitting."""
import numpy as np

Q_GRID = np.concatenate([np.linspace(0.0, 5.0, 101),
                         np.linspace(5.0, 50.0, 91)[1:],
                         np.linspace(50.0, 400.0, 71)[1:]])

def _fit_ceiling(ranks, freqs, s_max, step=0.05):
    y = np.log(freqs)
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst <= 0 or len(ranks) < 4:
        return None
    ss = np.arange(0.05, s_max + 1e-9, step)
    best = (-np.inf, np.nan, np.nan)
    for q in Q_GRID:
        x = np.log(ranks + q)
        P = -np.outer(ss, x)
        P -= np.log(np.exp(P).sum(axis=1))[:, None]
        r2 = 1.0 - ((y[None, :] - P) ** 2).sum(axis=1) / sst
        k = int(np.argmax(r2))
        if r2[k] > best[0]:
            best = (float(r2[k]), float(q), float(ss[k]))
    return best   # (R2, q, s)

def fit_zm(ranks, freqs, ceilings=(6.0, 12.0, 25.0)):
    """
    Fit at several s ceilings. If s tracks the ceiling, the exponent is not
    identifiable -- the rank-frequency curve is not Zipf-Mandelbrot in shape and
    no finite (q,s) describes it. Report that rather than quoting a bounded number.
    """
    runs = {}
    for c in ceilings:
        r = _fit_ceiling(ranks, freqs, c)
        if r is None:
            return dict(ok=False, identifiable=False, reason='too few points')
        runs[c] = dict(R2=r[0], q=r[1], s=r[2], at_ceiling=bool(r[2] >= 0.95 * c))
    n_at = sum(v['at_ceiling'] for v in runs.values())
    top_c = max(ceilings)
    top = runs[top_c]
    # Non-identifiable iff s is still tracking the LARGEST ceiling. Exceeding a
    # small ceiling only means s is large, not that it is unbounded.
    identifiable = not top['at_ceiling']
    ref = dict(top)
    # refine the identifiable case on a fine local grid
    if identifiable:
        q0, s0, r20 = ref['q'], ref['s'], ref['R2']
        y = np.log(freqs); sst = float(np.sum((y - y.mean()) ** 2))
        for _ in range(3):
            for q in np.linspace(max(0.0, q0 * 0.9), q0 * 1.1, 25):
                x = np.log(ranks + q)
                for s in np.linspace(max(0.01, s0 - 0.06), s0 + 0.06, 25):
                    P = -s * x; P = P - np.log(np.exp(P).sum())
                    v = 1.0 - float(((y - P) ** 2).sum()) / sst
                    if v > r20:
                        r20, q0, s0 = v, q, s
        ref = dict(R2=r20, q=q0, s=s0, at_ceiling=False)
    ridge = bool(identifiable and (ref['q'] > 100.0 or ref['s'] > 8.0))
    return dict(ok=True, identifiable=identifiable, ridge=ridge,
                ridge_note=('q>100 or s>8: fit sits on the ZM q-s ridge where the '
                            'curve approaches an exponential, not a power law'
                            if ridge else ''),
                q=ref['q'], s=ref['s'], R2=ref['R2'],
                ceiling_runs={str(c): v for c, v in runs.items()},
                n_ceilings_hit=n_at,
                reason=('' if identifiable else
                        's still at the largest ceiling (%.0f) -> exponent unbounded, '
                        'curve is not Zipf-Mandelbrot in shape' % max(ceilings)))

def fit_fast(ranks, freqs, s_max=25.0):
    r = _fit_ceiling(ranks, freqs, s_max, step=0.10)
    if r is None:
        return None
    return dict(R2=r[0], q=r[1], s=r[2], at_ceiling=bool(r[2] >= 0.95 * s_max))
