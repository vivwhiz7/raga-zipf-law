"""
zm_core.py — parsing + Zipf-Mandelbrot fitting for the SANGEET corpus.

Corrections relative to the previous analysis.py:
  C1  Todi encoding: the CONTENT notes were already correct. SANGEET writes
      vikrit (komal/tivra) svaras in UPPERCASE and shuddha in lowercase --
      confirmed by AROHANA (s-R-G-M-p-D-n-su) and PAKAD in the Todi files
      themselves, and by Bhairav as a control. The previous D->d, R->r, G->g
      content remap destroyed that distinction. The corrupted field is the
      <VADI>/<SAMVADI> metadata pair, which contradicts AROHANA/PAKAD in the
      same file. We correct the METADATA and leave all content untouched.
  C2  Durations in matras, not cell-relative. A CONTENT cell is one bibhaga,
      whose width in matras is BEAT_PATTERN[col_index]. Previously every cell
      was treated as 1 unit, so a token '0.50' meant 2 matras in a Tritaal
      composition and 1 matra in an Ektaal one, pooled into the same bin.
  C3  ZM fitting by exact profile least squares instead of curve_fit with
      bounds. For fixed q the log-space model is linear in (log A, s), so the
      inner solve is exact. No optimiser bounds -> no s pinned at 20, no
      A ~ 1e29. A properly normalised 2-parameter ZM (sum f = 1) is also fitted.
"""
import re, math, glob
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET
import numpy as np

RAGAS = ['Bhairav', 'Todi', 'Poorvi']
SCHEMES = ['pitch', 'duration', 'pitch_duration']

SVARA_RE = re.compile(r'[srRgGmMpdDnN][ul]?')
VALID_PITCHES = set('s r R g G m M p d D n N'.split())

# C1: metadata correction, justified from AROHANA/PAKAD within the same files.
META_CORRECTION = {'Todi': dict(vadi='D', samvadi='G')}
META_CORRECTION_NOTE = (
    "Todi <VADI>/<SAMVADI> read 'd'/'g' but AROHANA (s-R-G-M-p-D-n-su) and "
    "PAKAD (Dl-nl-s-R-G-R-s-M-G-R-G-R-s) in the same files write komal Dha as "
    "'D' and komal Ga as 'G'. Metadata corrected to D/G; content untouched."
)

def _strip_octave(tok):
    return re.sub(r'[ul]+$', '', tok)

def _clean_cell_text(raw):
    text = raw or ''
    text = re.sub(r'<sup>(.*?)</sup>', r' \1 ', text)   # grace note -> own slot
    text = text.replace('@', ' ')                        # gamak marker
    return text

def parse_file(fp, duration_mode='matra'):
    """
    duration_mode:
      'matra' -> slot duration = bibhaga_width_in_matras / n_slots   (C2, primary)
      'cell'  -> slot duration = 1 / n_slots                          (legacy behaviour)
    Returns (notes, meta). notes = list of (svara, duration_float).
    """
    root = ET.parse(fp).getroot()
    rg, tl = root.find('RAAG'), root.find('TAAL')

    beat = [int(x) for x in (tl.findtext('BEAT_PATTERN') or '').split('-') if x.isdigit()]
    maatra = int(tl.findtext('MAATRA') or 0)
    bibhaga = int(tl.findtext('BIBHAGA') or 0) or (len(beat) or 1)
    mean_w = (maatra / bibhaga) if (maatra and bibhaga) else 1.0

    meta = dict(
        raga=(rg.findtext('RAAG_NAME') or '?').strip(),
        thaat=(rg.findtext('THAAT') or '?').strip(),
        vadi=(rg.findtext('VADI') or '?').strip(),
        samvadi=(rg.findtext('SAMVADI') or '?').strip(),
        taal=(tl.findtext('TAAL_NAME') or '?').strip(),
        maatra=maatra, bibhaga=bibhaga, beat_pattern=beat,
        even_taal=(len(set(beat)) <= 1 and len(beat) > 0),
    )

    notes = []
    n_overflow = 0
    for col in root.findall('.//COL'):
        ce = col.find('CONTENT')
        if ce is None:
            continue
        slots = _clean_cell_text(''.join(ce.itertext())).split()
        if not slots:
            continue

        if duration_mode == 'matra':
            try:
                ci = int(col.get('INDEX'))
            except (TypeError, ValueError):
                ci = -1
            if 0 <= ci < len(beat):
                cell_w = float(beat[ci])
            else:
                cell_w = mean_w          # taan overflow cell: use mean bibhaga width
                n_overflow += 1
        else:
            cell_w = 1.0

        n = len(slots)
        base = cell_w / n
        i = 0
        while i < n:
            s = slots[i]
            if s == '-':
                if notes:
                    sv, d = notes[-1]
                    notes[-1] = (sv, d + base)
                i += 1
                continue
            pcs = [_strip_octave(m) for m in SVARA_RE.findall(s)
                   if _strip_octave(m) in VALID_PITCHES]
            if not pcs:
                i += 1
                continue
            j = i + 1
            while j < n and slots[j] == '-':
                j += 1
            per = base * (j - i) / len(pcs)
            for pc in pcs:
                notes.append((pc, per))
            i = j

    meta['n_overflow_cells'] = n_overflow
    return notes, meta


def load_corpus(corpus_dir, raga, duration_mode='matra', even_only=False):
    corpus_dir = Path(corpus_dir)
    files = sorted(corpus_dir.glob(f'{raga}-*.xml'))
    all_notes, metas, used = [], [], 0
    for f in files:
        notes, meta = parse_file(f, duration_mode=duration_mode)
        if even_only and not meta['even_taal']:
            continue
        all_notes.extend(notes)
        metas.append(meta)
        used += 1
    base = dict(metas[0]) if metas else {}
    base.update(META_CORRECTION.get(raga, {}))
    base['n_files'] = used
    base['n_files_total'] = len(files)
    base['n_overflow_cells'] = sum(m['n_overflow_cells'] for m in metas)
    base['taal_mix'] = Counter(m['taal'] for m in metas)
    return all_notes, base


def make_tokens(notes, scheme, dur_dp=2):
    if scheme == 'pitch':
        return [sv for sv, _ in notes]
    if scheme == 'duration':
        return [f'{round(d, dur_dp):.2f}' for _, d in notes]
    if scheme == 'pitch_duration':
        return [f'{sv}:{round(d, dur_dp):.2f}' for sv, d in notes]
    raise ValueError(scheme)


def rank_frequency(tokens):
    c = Counter(tokens)
    total = sum(c.values())
    items = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
    toks = [t for t, _ in items]
    ranks = np.arange(1, len(items) + 1, dtype=float)
    freqs = np.array([n / total for _, n in items], dtype=float)
    return toks, ranks, freqs


# ---------------------------------------------------------------- ZM fitting
def _r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def _q_grid(coarse=True):
    if coarse:
        return np.concatenate([np.linspace(0.0, 5.0, 101),
                               np.linspace(5.0, 50.0, 91)[1:],
                               np.linspace(50.0, 300.0, 51)[1:]])
    return np.concatenate([np.linspace(0.0, 5.0, 501),
                           np.linspace(5.0, 50.0, 451)[1:],
                           np.linspace(50.0, 300.0, 251)[1:]])


def fit_zm_raw(ranks, freqs, coarse=False):
    """f(r) = A/(r+q)^s, A free. Exact linear inner solve for (logA, s)."""
    y = np.log(freqs)
    best = None
    for q in _q_grid(coarse):
        x = np.log(ranks + q)
        X = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = _r2(y, X @ coef)
        if best is None or r2 > best[0]:
            best = (r2, q, float(coef[0]), float(-coef[1]))
    r2, q, logA, s = best
    QMAX = _q_grid(coarse)[-1]
    degen = (abs(logA) > 25.0) or (q >= QMAX * 0.999) or (s > 12.0)
    A = math.exp(logA) if abs(logA) < 700 else float('inf')
    return dict(model='raw', A=A, logA=float(logA), q=q, s=s, R2=r2, n_par=3,
                degenerate=bool(degen),
                degen_reason=('|logA|>25' if abs(logA) > 25 else
                              'q at grid bound' if q >= QMAX * 0.999 else
                              's>12' if s > 12 else ''))


def fit_zm_a1(ranks, freqs, coarse=False):
    """f(r) = 1/(r+q)^s, A fixed to 1 (the previous script's 'norm')."""
    y = np.log(freqs)
    best = None
    for q in _q_grid(coarse):
        x = np.log(ranks + q)
        s = -float(np.sum(x * y) / np.sum(x * x))
        r2 = _r2(y, -s * x)
        if best is None or r2 > best[0]:
            best = (r2, q, s)
    r2, q, s = best
    QMAX = _q_grid(coarse)[-1]
    return dict(model='A1', A=1.0, q=q, s=s, R2=r2, n_par=2,
                degenerate=bool(q >= QMAX * 0.999 or s > 12.0),
                degen_reason=('q at grid bound' if q >= QMAX * 0.999 else
                              's>12' if s > 12 else ''))


def fit_zm_norm(ranks, freqs, coarse=False):
    """
    Proper normalised ZM: f(r) = (r+q)^-s / SUM_j (r_j+q)^-s, so SUM f = 1.
    Two free parameters. 2-D grid then local refinement.
    """
    y = np.log(freqs)

    def score(q, s):
        x = np.log(ranks + q)
        lz = np.log(np.sum(np.exp(-s * x)))
        return _r2(y, -s * x - lz)

    qs = _q_grid(coarse=True)
    ss = np.arange(0.10, 6.01, 0.02)
    best = (-np.inf, None, None)
    for q in qs:
        x = np.log(ranks + q)
        M = np.exp(-np.outer(ss, x))                       # (ns, n)
        lz = np.log(M.sum(axis=1))
        pred = -np.outer(ss, x) - lz[:, None]
        ss_res = ((y[None, :] - pred) ** 2).sum(axis=1)
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2v = 1.0 - ss_res / ss_tot
        k = int(np.argmax(r2v))
        if r2v[k] > best[0]:
            best = (float(r2v[k]), float(q), float(ss[k]))
    # local refinement
    r2, q0, s0 = best
    for _ in range(3):
        dq = max(q0 * 0.05, 0.02)
        ds = 0.01
        for q in np.linspace(max(0.0, q0 - dq), q0 + dq, 21):
            for s in np.linspace(max(0.01, s0 - ds), s0 + ds, 21):
                v = score(q, s)
                if v > r2:
                    r2, q0, s0 = v, q, s
    QMAX = _q_grid(coarse=True)[-1]
    return dict(model='norm', A=float('nan'), q=q0, s=s0, R2=r2, n_par=2,
                degenerate=bool(q0 >= QMAX * 0.999 or s0 >= 5.99),
                degen_reason=('q at grid bound' if q0 >= QMAX * 0.999 else
                              's at grid bound' if s0 >= 5.99 else ''))


def bootstrap_zm(notes, scheme, n_boot=200, seed=0, fitter=fit_zm_norm):
    """Resample the note sequence with replacement; refit; return percentile CIs."""
    rng = np.random.default_rng(seed)
    arr = np.array(notes, dtype=object)
    n = len(arr)
    qs, ss, r2s = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        toks = make_tokens([tuple(arr[i]) for i in idx], scheme)
        _, r, f = rank_frequency(toks)
        if len(r) < 4:
            continue
        fit = fitter(r, f, coarse=True)
        qs.append(fit['q']); ss.append(fit['s']); r2s.append(fit['R2'])
    def ci(v):
        v = np.array(v, dtype=float)
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return dict(n_boot=len(ss), q_ci=ci(qs), s_ci=ci(ss), R2_ci=ci(r2s),
                s_sd=float(np.std(ss, ddof=1)), q_sd=float(np.std(qs, ddof=1)))


def fit_zm_norm_fast(ranks, freqs, coarse=True):
    """Coarse normalised-ZM fit for resampling loops (no local refinement)."""
    y = np.log(freqs)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return dict(model='norm', q=float('nan'), s=float('nan'), R2=float('nan'))
    qs = np.concatenate([np.linspace(0.0, 5.0, 26),
                         np.linspace(5.5, 50.0, 46),
                         np.linspace(55.0, 300.0, 25)])
    ss = np.arange(0.10, 6.01, 0.05)
    best = (-np.inf, np.nan, np.nan)
    for q in qs:
        x = np.log(ranks + q)
        P = -np.outer(ss, x)
        P -= np.log(np.exp(P).sum(axis=1))[:, None]
        r2v = 1.0 - ((y[None, :] - P) ** 2).sum(axis=1) / ss_tot
        k = int(np.argmax(r2v))
        if r2v[k] > best[0]:
            best = (float(r2v[k]), float(q), float(ss[k]))
    return dict(model='norm', q=best[1], s=best[2], R2=best[0])
