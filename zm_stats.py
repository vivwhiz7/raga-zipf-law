"""Composition-level bootstrap + pakad permutation test."""
import numpy as np
import zm_core as z
import zm_ngram as zn
import zm_fit2 as f2

def bootstrap_by_composition(per_file, scheme, n_boot=400, seed=42, unit_ngram=None):
    """
    Resample COMPOSITIONS with replacement -- the actual sampling unit.
    Resampling individual notes is wrong twice over: notes within a composition
    are strongly autocorrelated (understates variance), and note-level resampling
    drops ~37% of singleton types (biases the vocabulary and hence the fit).
    """
    rng = np.random.default_rng(seed)
    k = len(per_file)
    qs, ss, r2s, Ns = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, k, k)
        if unit_ngram:
            sel = [per_file[i] for i in idx]
            toks = zn.ngram_tokens(sel, n=unit_ngram, unit='pitch')
        else:
            notes = []
            for i in idx:
                notes.extend(per_file[i][1])
            toks = z.make_tokens(notes, scheme)
        tk, r, f = z.rank_frequency(toks)
        if len(r) < 4:
            continue
        fit = f2.fit_fast(r, f)                      # s ceiling 25, not 6
        if fit is None or fit['at_ceiling']:
            continue                                  # drop non-identifiable replicates
        qs.append(fit['q']); ss.append(fit['s']); r2s.append(fit['R2']); Ns.append(len(tk))
    def ci(v):
        v = np.asarray(v, float)
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    if len(ss) < 20:
        return dict(n_boot=len(ss), insufficient=True, q_ci=[None, None],
                    s_ci=[None, None], R2_ci=[None, None], s_sd=None, q_sd=None,
                    N_median=None, n_dropped=n_boot - len(ss))
    return dict(n_boot=len(ss), insufficient=False, q_ci=ci(qs), s_ci=ci(ss),
                R2_ci=ci(r2s), s_sd=float(np.std(ss, ddof=1)),
                q_sd=float(np.std(qs, ddof=1)), N_median=float(np.median(Ns)),
                n_dropped=n_boot - len(ss))


def pakad_permutation(corpus_dir, raga, per_file, n=3, n_perm=500, seed=7):
    """
    H0: the pakad's n-grams are no more frequent than chance given the raga's
    unigram distribution. Shuffle svaras within each composition (destroying
    sequence, preserving each composition's note inventory exactly), recompute
    the pakad n-grams' median rank, repeat.
    """
    rng = np.random.default_rng(seed)
    pak, svaras, grams = zn.pakad_ngrams(corpus_dir, raga, n=n)

    def median_rank(seqs):
        toks = []
        for s in seqs:
            for i in range(len(s) - n + 1):
                toks.append('-'.join(s[i:i + n]))
        tk, r, f = z.rank_frequency(toks)
        rank = {t: i + 1 for i, t in enumerate(tk)}
        found = [rank[g] for g in grams if g in rank]
        # unfound grams are worse than every observed rank
        allr = found + [len(tk) + 1] * (len(grams) - len(found))
        return float(np.median(allr)), len(found), len(tk)

    obs_seqs = [[sv for sv, _ in notes] for _, notes, _ in per_file]
    obs, n_found, vocab = median_rank(obs_seqs)

    null = []
    for _ in range(n_perm):
        sh = []
        for s in obs_seqs:
            a = np.array(s, dtype=object)
            rng.shuffle(a)
            sh.append(list(a))
        null.append(median_rank(sh)[0])
    null = np.asarray(null, float)
    p = float((np.sum(null <= obs) + 1) / (n_perm + 1))   # one-sided: obs ranks better
    return dict(pakad=pak, n=n, n_grams=len(grams), n_found=n_found, vocab=vocab,
                obs_median_rank=obs, null_median=float(np.median(null)),
                null_ci=[float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
                p_value=p, n_perm=n_perm)
