import json, pickle, warnings, math
from pathlib import Path
from collections import Counter
import numpy as np
import zm_core as z, zm_ngram as zn, zm_fit2 as f2, zm_stats as zs
warnings.filterwarnings('ignore')

from pathlib import Path as _P
_HERE = _P(__file__).resolve().parent
CORPUS = str(_HERE.parent / 'Sangeet' / 'Bhatkhande Dataset')
OUT = _HERE / 'output'; OUT.mkdir(parents=True, exist_ok=True)


NB=200; log=print

corp={r: z.load_corpus(CORPUS,r) for r in z.RAGAS}
pf={r: zn.per_file_notes(CORPUS,r) for r in z.RAGAS}

log('='*78); log('1  CORPUS'); log('='*78)
for r in z.RAGAS:
    n,m=corp[r]
    log(f'  {r:<9} files={m["n_files"]:>3}  notes={len(n):>5}  vadi={m["vadi"]} samvadi={m["samvadi"]}  '
        f'taan-overflow cells={m["n_overflow_cells"]}  taals={dict(m["taal_mix"])}')

log(''); log('='*78); log('2  ZM FITS  (normalised, identifiability-tested)'); log('='*78)
fits={}
for r in z.RAGAS:
    n,_=corp[r]; fits[r]={}
    for sch in z.SCHEMES:
        t,rk,fr=z.rank_frequency(z.make_tokens(n,sch))
        d=f2.fit_zm(rk,fr); d['N']=len(t); d['L']=len(n)
        if d['identifiable']:
            b=zs.bootstrap_by_composition(pf[r],sch,n_boot=NB)
            d['boot']=b
            log(f'  {r:<9}{sch:<15}N={len(t):>4}  q={d["q"]:7.3f} s={d["s"]:5.3f} R2={d["R2"]:.4f}   '
                f's 95%CI[{b["s_ci"][0]:.2f},{b["s_ci"][1]:.2f}]  q 95%CI[{b["q_ci"][0]:.1f},{b["q_ci"][1]:.1f}]')
        else:
            d['boot']=None
            log(f'  {r:<9}{sch:<15}N={len(t):>4}  NOT IDENTIFIABLE — {d["reason"]}')
        fits[r][sch]=d

log(''); log('='*78); log('3  VADI / SAMVADI  (pitch ranks, corrected metadata)'); log('='*78)
vadi={}
for r in z.RAGAS:
    n,m=corp[r]; t,rk,fr=z.rank_frequency(z.make_tokens(n,'pitch'))
    rank={x:i+1 for i,x in enumerate(t)}; freq=dict(zip(t,fr))
    vadi[r]=dict(vadi=m['vadi'],vadi_rank=rank.get(m['vadi']),vadi_pct=100*freq.get(m['vadi'],0),
                 samvadi=m['samvadi'],samvadi_rank=rank.get(m['samvadi']),
                 samvadi_pct=100*freq.get(m['samvadi'],0),rank1=t[0],rank1_pct=100*fr[0],
                 sa_rank=rank.get('s'),full=[(x,100*v) for x,v in zip(t,fr)])
    d=vadi[r]
    log(f'  {r:<9}rank1={d["rank1"]}({d["rank1_pct"]:.2f}%)  vadi {d["vadi"]}->rank {d["vadi_rank"]} '
        f'({d["vadi_pct"]:.2f}%)  samvadi {d["samvadi"]}->rank {d["samvadi_rank"]} ({d["samvadi_pct"]:.2f}%)')

log(''); log('='*78); log('4  SVARA N-GRAMS  (within composition)'); log('='*78)
ngr={}
for r in z.RAGAS:
    ngr[r]={}
    for n_ in (1,2,3):
        toks=zn.ngram_tokens(pf[r],n=n_,unit='pitch')
        t,rk,fr=z.rank_frequency(toks)
        d=f2.fit_zm(rk,fr); d['N']=len(t); d['L']=len(toks)
        d['top']=[(x,100*v) for x,v in zip(t[:5],fr[:5])]
        if d['identifiable']:
            b=zs.bootstrap_by_composition(pf[r],None,n_boot=NB,unit_ngram=n_)
            d['boot']=b
            log(f'  {r:<9}n={n_}  N={len(t):>5} L={len(toks):>6}  q={d["q"]:7.3f} s={d["s"]:5.3f} '
                f'R2={d["R2"]:.4f}  s 95%CI[{b["s_ci"][0]:.2f},{b["s_ci"][1]:.2f}]')
        else:
            d['boot']=None
            log(f'  {r:<9}n={n_}  N={len(t):>5} L={len(toks):>6}  NOT IDENTIFIABLE')
        ngr[r][n_]=d

log(''); log('='*78); log('5  SENSITIVITY  (pitch_duration)'); log('='*78)
sens={}
for lab,kw in [('matra (primary)',dict(duration_mode='matra')),
               ('cell-relative (legacy)',dict(duration_mode='cell')),
               ('matra, even-bibhaga taals only',dict(duration_mode='matra',even_only=True))]:
    sens[lab]={}
    for r in z.RAGAS:
        n,m=z.load_corpus(CORPUS,r,**kw)
        t,rk,fr=z.rank_frequency(z.make_tokens(n,'pitch_duration'))
        d=f2.fit_zm(rk,fr)
        sens[lab][r]=dict(files=m['n_files'],L=len(n),N=len(t),q=d.get('q'),s=d.get('s'),
                          R2=d.get('R2'),identifiable=d['identifiable'])
        log(f'  {lab:<32}{r:<9}files={m["n_files"]:>3} N={len(t):>4} '
            f'q={d.get("q",float("nan")):7.3f} s={d.get("s",float("nan")):5.3f} R2={d.get("R2",float("nan")):.4f}')

log(''); log('='*78); log('6  PAKAD PERMUTATION TEST'); log('='*78)
perm={}
for r in z.RAGAS:
    perm[r]={}
    for n_ in (2,3):
        p=zs.pakad_permutation(CORPUS,r,pf[r],n=n_,n_perm=300)
        perm[r][n_]=p
        log(f'  {r:<9}n={n_}  obs median rank {p["obs_median_rank"]:6.1f}/{p["vocab"]:<5} '
            f'null {p["null_median"]:6.1f} [{p["null_ci"][0]:.0f},{p["null_ci"][1]:.0f}]  '
            f'p={p["p_value"]:.4f}  ({p["n_found"]}/{p["n_grams"]} present)')

pickle.dump(dict(corp=corp,fits=fits,vadi=vadi,ngram=ngr,sens=sens,perm=perm,pf=pf),
            open(OUT/'final.pkl','wb'))
def clean(o):
    if isinstance(o,dict): return {str(k):clean(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [clean(x) for x in o]
    if isinstance(o,(np.floating,float)):
        v=float(o); return None if math.isnan(v) else v
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,Counter): return dict(o)
    return o
json.dump(clean(dict(fits=fits,vadi=vadi,ngram=ngr,sens=sens,perm=perm)),
          open(OUT/'results.json','w'),indent=1,default=str)
log('\nSaved final.pkl + results.json')
