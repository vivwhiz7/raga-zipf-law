"""N-gram + pakad extension. Bigrams are built within compositions only."""
import re
from collections import Counter
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
import zm_core as z

def per_file_notes(corpus_dir, raga, duration_mode='matra'):
    out = []
    for f in sorted(Path(corpus_dir).glob(f'{raga}-*.xml')):
        notes, meta = z.parse_file(f, duration_mode=duration_mode)
        out.append((f.name, notes, meta))
    return out

def ngram_tokens(per_file, n=2, unit='pitch'):
    """N-grams of svaras, never spanning a composition boundary."""
    toks = []
    for _, notes, _ in per_file:
        seq = [sv for sv, _ in notes] if unit == 'pitch' else \
              [f'{sv}:{round(d,2):.2f}' for sv, d in notes]
        for i in range(len(seq) - n + 1):
            toks.append('-'.join(seq[i:i + n]))
    return toks

def pakad_ngrams(corpus_dir, raga, n=2):
    """Extract the raga's PAKAD phrase and return its svara n-grams."""
    f = sorted(Path(corpus_dir).glob(f'{raga}-*.xml'))[0]
    rg = ET.parse(f).getroot().find('RAAG')
    pak = (rg.findtext('PAKAD') or '').strip()
    parts = [p for p in re.split(r'[-\s]+', pak) if p]
    svaras = []
    for p in parts:
        for m in z.SVARA_RE.findall(p):
            b = z._strip_octave(m)
            if b in z.VALID_PITCHES:
                svaras.append(b)
    grams = ['-'.join(svaras[i:i + n]) for i in range(len(svaras) - n + 1)]
    return pak, svaras, grams
