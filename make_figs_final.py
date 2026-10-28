import pickle, warnings
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import zm_core as z, zm_ngram as zn
warnings.filterwarnings('ignore')

from pathlib import Path as _P
_HERE = _P(__file__).resolve().parent
CORPUS = str(_HERE.parent / 'Sangeet' / 'Bhatkhande Dataset')
OUT = _HERE / 'output'; OUT.mkdir(parents=True, exist_ok=True)


rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8.5,
                 'axes.titlesize':9,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
                 'legend.fontsize':7,'axes.linewidth':.8,'xtick.major.width':.8,
                 'ytick.major.width':.8,'savefig.bbox':'tight','savefig.pad_inches':.02})

st=pickle.load(open(OUT/'final.pkl','rb'))
corp,fits,ngr,perm,pf = st['corp'],st['fits'],st['ngram'],st['perm'],st['pf']
RAGAS=z.RAGAS
RC={'Bhairav':'#1f4e79','Todi':'#a11d33','Poorvi':'#1b7837'}
# scheme: colour, marker, greyscale-safe
SCH={'pitch':('#8c8c8c','o','pitch only'),
     'duration':('#d95f0e','^','duration only'),
     'pitch_duration':('#1f4e79','s','pitch x duration')}
NG ={1:('#8c8c8c','o','n = 1'),2:('#d95f0e','^','n = 2'),3:('#1f4e79','s','n = 3')}
def zm(r,q,s):
    v=(r+q)**(-s); return v/v.sum()
def panel(ax,letter):
    ax.text(-0.16,1.04,letter,transform=ax.transAxes,fontsize=10,fontweight='bold',va='bottom')

# ---------------- Figure 1 : unit of analysis
fig,axes=plt.subplots(1,3,figsize=(7.2,2.5),sharey=True)
for k,(ax,raga) in enumerate(zip(axes,RAGAS)):
    notes,_=corp[raga]
    for sch in z.SCHEMES:
        c,m,lab=SCH[sch]
        t,r,f=z.rank_frequency(z.make_tokens(notes,sch))
        ax.scatter(r,f,s=8,alpha=.55,color=c,marker=m,linewidths=0,label=f'{lab} (N={len(t)})')
        d=fits[raga][sch]
        if d['identifiable']:
            xs=np.arange(1,len(r)+1,dtype=float)
            ax.plot(xs,zm(xs,d['q'],d['s']),'-',color=c,lw=1.2)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('rank $r$'); ax.set_title(raga,color=RC[raga],fontweight='bold')
    ax.set_ylim(1e-4,.6)
    if k==0: ax.set_ylabel('relative frequency $f(r)$')
    ax.legend(loc='lower left',frameon=False,handletextpad=.3,borderpad=.2)
    panel(ax,'ABC'[k])
fig.tight_layout(w_pad=.6); fig.savefig(OUT/'fig1_unit_of_analysis.png',dpi=300); plt.close(fig)

# ---------------- Figure 2 : n-gram order
fig,axes=plt.subplots(1,3,figsize=(7.2,2.5),sharey=True)
for k,(ax,raga) in enumerate(zip(axes,RAGAS)):
    for n_ in (1,2,3):
        c,m,lab=NG[n_]
        toks=zn.ngram_tokens(pf[raga],n=n_,unit='pitch')
        t,r,f=z.rank_frequency(toks)
        ax.scatter(r,f,s=8,alpha=.55,color=c,marker=m,linewidths=0,label=f'{lab} (N={len(t)})')
        d=ngr[raga][n_]
        if d['identifiable']:
            xs=np.arange(1,len(r)+1,dtype=float)
            ax.plot(xs,zm(xs,d['q'],d['s']),'-',color=c,lw=1.2)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('rank $r$'); ax.set_title(raga,color=RC[raga],fontweight='bold')
    ax.set_ylim(8e-5,.6)
    if k==0: ax.set_ylabel('relative frequency $f(r)$')
    ax.legend(loc='lower left',frameon=False,handletextpad=.3,borderpad=.2)
    panel(ax,'ABC'[k])
fig.tight_layout(w_pad=.6); fig.savefig(OUT/'fig2_ngram_order.png',dpi=300); plt.close(fig)

# ---------------- Figure 3 : pakad permutation (trigram)
fig,axes=plt.subplots(1,3,figsize=(7.2,2.15))
for k,(ax,raga) in enumerate(zip(axes,RAGAS)):
    p=perm[raga][3]
    xhi=max(p['vocab'],p['null_ci'][1],p['null_median'])*1.04
    ax.axvspan(p['null_ci'][0],p['null_ci'][1],color='#cfcfcf',alpha=.9,zorder=1,
               label='null 95% interval')
    ax.axvline(p['null_median'],color='#4d4d4d',ls='--',lw=1.1,zorder=2,
               label=f'null median = {p["null_median"]:.0f}')
    ax.axvline(p['obs_median_rank'],color=RC[raga],lw=2.2,zorder=3,
               label=f'observed = {p["obs_median_rank"]:.0f}')
    ax.set_xlim(0,xhi); ax.set_ylim(0,1); ax.set_yticks([])
    ax.set_xlabel('median rank')
    ax.set_title(f'{raga}  ($p$ = {p["p_value"]:.3f})',color=RC[raga],fontweight='bold')
    ax.legend(loc='upper center',frameon=False,handletextpad=.5,borderpad=.15,
              labelspacing=.25,fontsize=6.8)
    panel(ax,'ABC'[k])
fig.tight_layout(w_pad=.7); fig.savefig(OUT/'fig3_pakad_permutation.png',dpi=300); plt.close(fig)

# ---------------- Figure 4 : exponents with CIs  (A pitch x duration, B trigram)
fig,axes=plt.subplots(1,2,figsize=(5.4,2.4),sharey=True)
for k,(ax,(src,key,lab)) in enumerate(zip(axes,[(fits,'pitch_duration','pitch x duration'),
                                                (ngr,3,'svara trigram')])):
    for i,raga in enumerate(RAGAS):
        d=src[raga][key]
        if not d['identifiable'] or d.get('boot') is None: continue
        b=d['boot']
        ax.errorbar(d['s'],i,xerr=[[max(0,d['s']-b['s_ci'][0])],[max(0,b['s_ci'][1]-d['s'])]],
                    fmt='o',color=RC[raga],capsize=3,ms=5,lw=1.2)
    ax.set_yticks(range(3)); ax.set_yticklabels(RAGAS)
    ax.set_xlabel('ZM exponent $s$'); ax.set_title(lab,fontweight='bold')
    ax.grid(axis='x',alpha=.25,lw=.6); ax.set_ylim(-.6,2.6)
    panel(ax,'AB'[k])
fig.tight_layout(w_pad=.8); fig.savefig(OUT/'fig4_exponents_ci.png',dpi=300); plt.close(fig)
print('done:',sorted(x.name for x in OUT.glob('fig*.png')))
