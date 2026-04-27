"""
Extended Model v3b: Citation-Sandhi Mapping with Perceptual Encoding
====================================================================
Both recording and retrieval of citation-sandhi mappings pass through
tau-controlled perceptual encoding.

Low tau: tones get confused during encoding -> noisy frequency table -> poor production
High tau: tones perceived accurately -> clean table -> good production
"""

import numpy as np
from typing import List, Tuple, Dict
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csl_tone_model import *

ALL_TONE_CODES = [11, 13, 22, 31, 33, 44, 53, 55]

def perceive_tone(true_tone, tau, rng=None):
    """Simulate noisy tone perception controlled by tau."""
    if rng is None: rng = np.random
    sims = np.array([tone_similarity(true_tone, t, tau) for t in ALL_TONE_CODES])
    probs = sims / sims.sum()
    return ALL_TONE_CODES[rng.choice(len(ALL_TONE_CODES), p=probs)]


class ToneCSLModelV3b(ToneCSLModel):
    def __init__(self, alpha=1.0, chi=0.1, lam=0.02, tau=2.0, noise=11.0):
        super().__init__(alpha=alpha, chi=chi, lam=lam, tau=tau, noise=noise)
        self.citation_to_sandhi = {}
        self._rng = np.random.RandomState()

    def train(self, trials, seed=None):
        if seed is not None: self._rng = np.random.RandomState(seed)
        for t in trials: self._register(t)
        na = len(self.audio_forms); nv = len(self.visual_refs)
        self.M = np.zeros((na, nv)); self.M_type = np.zeros((na, 2))
        for trial in trials:
            self._process_monosyllabic(trial)
            self._update(trial)

    def _process_monosyllabic(self, trial):
        for audio_key, type_key in [('a1','t1'), ('a2','t2')]:
            dis = trial[audio_key]; con = trial[type_key]
            seg1, st1, seg2, st2 = dis
            if con == 'VN':
                nc = self._lookup_cit(seg2)
                if nc is not None:
                    self._record(perceive_tone(nc, self.tau, self._rng),
                                 perceive_tone(st2, self.tau, self._rng), 'sigma2', con)
                    vc = self._lookup_cit(seg1)
                    if vc is not None:
                        self._record(perceive_tone(vc, self.tau, self._rng),
                                     perceive_tone(st1, self.tau, self._rng), 'sigma1', con)
            elif con == 'NC':
                nc = self._lookup_cit(seg1)
                if nc is not None:
                    self._record(perceive_tone(nc, self.tau, self._rng),
                                 perceive_tone(st1, self.tau, self._rng), 'sigma1', con)
                    cc = self._lookup_cit(seg2)
                    if cc is not None:
                        self._record(perceive_tone(cc, self.tau, self._rng),
                                     perceive_tone(st2, self.tau, self._rng), 'sigma2', con)

    def _lookup_cit(self, segment):
        for wd in [NOUNS, VERBS, CONTAINERS]:
            for k, (s, t) in wd.items():
                if s == segment: return t
        return None

    def _record(self, pcit, psan, position, construction):
        key = (pcit, position, construction)
        if key not in self.citation_to_sandhi: self.citation_to_sandhi[key] = {}
        if psan not in self.citation_to_sandhi[key]: self.citation_to_sandhi[key][psan] = 0
        self.citation_to_sandhi[key][psan] += 1

    def produce_sandhi(self, citation_tone, position, construction):
        perceived_q = perceive_tone(citation_tone, self.tau, self._rng)
        votes = {}
        for (oc, op, ot), sc in self.citation_to_sandhi.items():
            if op != position or ot != construction: continue
            sim = tone_similarity(perceived_q, oc, self.tau)
            for st, cnt in sc.items():
                if st not in votes: votes[st] = 0.0
                votes[st] += sim * cnt
        if not votes:
            return ALL_TONE_CODES[self._rng.randint(len(ALL_TONE_CODES))], 0.0
        for st in votes: votes[st] += self._rng.normal(0, self.noise * 0.05)
        best = max(votes, key=votes.get)
        total = sum(max(0,v) for v in votes.values())
        return best, votes[best]/total if total > 0 else 0

    def test_productive_sandhi(self, verbose=False):
        c_both=0; c_s1=0; c_s2=0; total=0; results=[]
        for v in ['V1','V2','V3','V4']:
            for n in ['N1','N2','N3','N4']:
                s1,ct1=ALL_WORDS[v]; s2,ct2=ALL_WORDS[n]
                act=apply_sandhi(s1,ct1,s2,ct2,'VN')
                p1,_=self.produce_sandhi(ct1,'sigma1','VN')
                p2,_=self.produce_sandhi(ct2,'sigma2','VN')
                ok1=(p1==act[1]); ok2=(p2==act[3])
                if ok1: c_s1+=1
                if ok2: c_s2+=1
                if ok1 and ok2: c_both+=1
                total+=1
                tr=(v,n) in VN_PAIRS_EXPOSURE
                results.append({'type':'VN','s1':ok1,'s2':ok2,'both':ok1 and ok2,
                    'trained':tr,'citation':(ct1,ct2),'actual':(act[1],act[3]),
                    'predicted':(p1,p2)})
                if verbose:
                    m='V' if ok1 and ok2 else ('H' if ok1 or ok2 else 'X')
                    print(f"  {m} VN {s1}{ct1}-{s2}{ct2} -> ({p1},{p2}) actual ({act[1]},{act[3]})")
        for n in ['N1','N2','N3','N4']:
            for c in ['C1','C2','C3','C4']:
                s1,ct1=ALL_WORDS[n]; s2,ct2=ALL_WORDS[c]
                act=apply_sandhi(s1,ct1,s2,ct2,'NC')
                p1,_=self.produce_sandhi(ct1,'sigma1','NC')
                p2,_=self.produce_sandhi(ct2,'sigma2','NC')
                ok1=(p1==act[1]); ok2=(p2==act[3])
                if ok1: c_s1+=1
                if ok2: c_s2+=1
                if ok1 and ok2: c_both+=1
                total+=1
                tr=(n,c) in NC_PAIRS_EXPOSURE
                results.append({'type':'NC','s1':ok1,'s2':ok2,'both':ok1 and ok2,
                    'trained':tr,'citation':(ct1,ct2),'actual':(act[1],act[3]),
                    'predicted':(p1,p2)})
                if verbose:
                    m='V' if ok1 and ok2 else ('H' if ok1 or ok2 else 'X')
                    print(f"  {m} NC {s1}{ct1}-{s2}{ct2} -> ({p1},{p2}) actual ({act[1]},{act[3]})")
        return {'both':c_both/total,'sigma1':c_s1/total,'sigma2':c_s2/total,'total':total,'results':results}


if __name__ == '__main__':
    print("=" * 65)
    print("V3b: Productive Sandhi with Perceptual Encoding")
    print("=" * 65)

    # Tau sweep
    print("\n--- tau sweep (N=200 per point) ---\n")
    print(f"{'tau':>5} | {'Memory':>7} | {'Classif':>7} | {'Prod(both)':>10} | {'s1':>6} | {'s2':>6}")
    print("-" * 58)

    for tau in [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]:
        ml,cl,pb,p1,p2 = [],[],[],[],[]
        for si in range(200):
            np.random.seed(si)
            md = ToneCSLModelV3b(tau=tau, noise=11.0)
            md.train(build_exposure_trials(), seed=si)
            ml.append(md.test_word_memory())
            cl.append(md.test_generalisation())
            r = md.test_productive_sandhi()
            pb.append(r['both']); p1.append(r['sigma1']); p2.append(r['sigma2'])
        print(f"{tau:5.1f} | {np.mean(ml):7.3f} | {np.mean(cl):7.3f} | "
              f"{np.mean(pb):10.3f} | {np.mean(p1):6.3f} | {np.mean(p2):6.3f}")

    print("\n--- Done ---")
