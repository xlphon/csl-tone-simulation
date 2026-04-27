"""
Computational Simulation of Cross-Situational Tone Learning (v2)
================================================================
Extended Kachergis Associative Model with Tone Encoding

Key fix in v2: 
- Generalisation test uses construction-type-level associations
  (aggregate over all VN or NC referents), not specific visual referents
- Added configurable noise to match human-level accuracy
- Better parameter sweep with finer grid
"""

import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ============================================================
# 1. STIMULUS DEFINITIONS (same as v1)
# ============================================================

NOUNS = {'N1': ('du', 13), 'N2': ('ta', 53), 'N3': ('pu', 53), 'N4': ('ga', 13)}
VERBS = {'V1': ('pa', 53), 'V2': ('da', 13), 'V3': ('bu', 13), 'V4': ('ku', 53)}
CONTAINERS = {'C1': ('ka', 53), 'C2': ('ba', 13), 'C3': ('gu', 13), 'C4': ('tu', 53)}
ALL_WORDS = {**NOUNS, **VERBS, **CONTAINERS}

SANDHI_RULES = {
    'R1': {'input': (53, 13), 'output': (44, 13), 'type': 'VN'},
    'R2': {'input': (53, 53), 'output': (44, 53), 'type': 'VN'},
    'R3': {'input': (13, 53), 'output': (22, 53), 'type': 'VN'},
    'R4': {'input': (13, 13), 'output': (22, 13), 'type': 'VN'},
    'R5': {'input': (53, 13), 'output': (55, 22), 'type': 'NC'},
    'R6': {'input': (53, 53), 'output': (55, 31), 'type': 'NC'},
    'R7': {'input': (13, 13), 'output': (11, 33), 'type': 'NC'},
    'R8': {'input': (13, 53), 'output': (11, 13), 'type': 'NC'},
}

VN_PAIRS_EXPOSURE = [
    ('V1','N1'), ('V2','N1'), ('V2','N2'), ('V4','N2'),
    ('V1','N3'), ('V3','N3'), ('V3','N4'), ('V4','N4'),
]
NC_PAIRS_EXPOSURE = [
    ('N1','C1'), ('N1','C2'), ('N2','C2'), ('N2','C4'),
    ('N3','C1'), ('N3','C3'), ('N4','C2'), ('N4','C4'),
]

TONE_LETTER_TO_PITCH = {1: 89.9, 2: 100.9, 3: 120.0, 4: 142.0, 5: 160.0}


def get_sandhi_rule(tone1, tone2, construction):
    for rn, r in SANDHI_RULES.items():
        if r['input'] == (tone1, tone2) and r['type'] == construction:
            return rn
    raise ValueError(f"No rule for ({tone1}, {tone2}) in {construction}")


def apply_sandhi(seg1, tone1, seg2, tone2, construction):
    rule = SANDHI_RULES[get_sandhi_rule(tone1, tone2, construction)]
    return (seg1, rule['output'][0], seg2, rule['output'][1])


def make_disyllabic(w1_key, w2_key, construction):
    s1, t1 = ALL_WORDS[w1_key]
    s2, t2 = ALL_WORDS[w2_key]
    return apply_sandhi(s1, t1, s2, t2, construction)


# ============================================================
# 2. TONE SIMILARITY
# ============================================================

def tone_vec(tc):
    return np.array([TONE_LETTER_TO_PITCH[tc // 10], TONE_LETTER_TO_PITCH[tc % 10]])

# Pre-compute max distance for normalisation
_MAX_DIST_SQ = np.sum((tone_vec(11) - tone_vec(55)) ** 2)


def tone_similarity(ta, tb, tau):
    if ta == tb:
        return 1.0
    dist_sq = np.sum((tone_vec(ta) - tone_vec(tb)) ** 2)
    return np.exp(-tau * dist_sq / _MAX_DIST_SQ)


def audio_similarity(fa, fb, tau):
    """Similarity between two disyllabic forms (seg1,t1,seg2,t2)."""
    if fa[0] != fb[0] or fa[2] != fb[2]:  # segments must match
        return 0.0
    return tone_similarity(fa[1], fb[1], tau) * tone_similarity(fa[3], fb[3], tau)


# ============================================================
# 3. EXPOSURE TRIALS
# ============================================================

def build_exposure_trials():
    """Build 192 exposure trials (32 items x 2 orders x 3 blocks)."""
    specs = []

    # Training Set 1: same-type pairs
    for n, vn_pairs, nc_pairs in [
        ('N1', [('V1','N1'),('V2','N1')], [('N1','C1'),('N1','C2')]),
        ('N2', [('V2','N2'),('V4','N2')], [('N2','C2'),('N2','C4')]),
        ('N3', [('V1','N3'),('V3','N3')], [('N3','C1'),('N3','C3')]),
        ('N4', [('V3','N4'),('V4','N4')], [('N4','C2'),('N4','C4')]),
    ]:
        # VN pair
        specs.append((vn_pairs[0], 'VN', vn_pairs[1], 'VN'))
        specs.append((vn_pairs[1], 'VN', vn_pairs[0], 'VN'))
        # NC pair
        specs.append((nc_pairs[0], 'NC', nc_pairs[1], 'NC'))
        specs.append((nc_pairs[1], 'NC', nc_pairs[0], 'NC'))

    # Training Set 2: mixed pairs
    mixed = [
        (('V1','N1'),'VN', ('N1','C2'),'NC'),
        (('N1','C2'),'NC', ('V1','N1'),'VN'),
        (('V2','N1'),'VN', ('N1','C1'),'NC'),
        (('N1','C1'),'NC', ('V2','N1'),'VN'),
        (('V2','N2'),'VN', ('N2','C4'),'NC'),
        (('N2','C4'),'NC', ('V2','N2'),'VN'),
        (('V4','N2'),'VN', ('N2','C2'),'NC'),
        (('N2','C2'),'NC', ('V4','N2'),'VN'),
        (('V1','N3'),'VN', ('N3','C3'),'NC'),
        (('N3','C3'),'NC', ('V1','N3'),'VN'),
        (('V3','N3'),'VN', ('N3','C1'),'NC'),
        (('N3','C1'),'NC', ('V3','N3'),'VN'),
        (('V4','N4'),'VN', ('N4','C2'),'NC'),
        (('N4','C2'),'NC', ('V4','N4'),'VN'),
        (('V3','N4'),'VN', ('N4','C4'),'NC'),
        (('N4','C4'),'NC', ('V3','N4'),'VN'),
    ]
    specs.extend(mixed)

    assert len(specs) == 32, f"Expected 32, got {len(specs)}"

    # Convert to trial dicts
    items_32 = []
    for s in specs:
        pair1_keys, type1, pair2_keys, type2 = s
        audio1 = make_disyllabic(pair1_keys[0], pair1_keys[1], type1)
        audio2 = make_disyllabic(pair2_keys[0], pair2_keys[1], type2)
        vis1 = f"{pair1_keys[0]}{pair1_keys[1]}"
        vis2 = f"{pair2_keys[0]}{pair2_keys[1]}"
        items_32.append({
            'a1': audio1, 'a2': audio2,
            'v1': vis1, 'v2': vis2,
            't1': type1, 't2': type2,
        })

    # 32 items + reversed = 64
    items_64 = []
    for it in items_32:
        items_64.append(it)
        items_64.append({
            'a1': it['a2'], 'a2': it['a1'],
            'v1': it['v2'], 'v2': it['v1'],
            't1': it['t2'], 't2': it['t1'],
        })

    # 3 blocks x 64 = 192
    all_trials = []
    for _ in range(3):
        block = items_64.copy()
        np.random.shuffle(block)
        all_trials.extend(block)

    return all_trials


# ============================================================
# 4. MODEL
# ============================================================

class ToneCSLModel:
    """
    Extended Kachergis associative model with tone-precision parameter.
    
    Key change from v1: We track associations at TWO levels:
    1. M_item[audio, visual]: item-level association (for word memory)
    2. M_type[audio, construction_type]: type-level association (for generalisation)
       This aggregates evidence about which tone patterns go with VN vs NC
    """

    def __init__(self, alpha=1.0, chi=0.1, lam=0.02, tau=2.0, noise=0.3):
        self.alpha = alpha
        self.chi = chi
        self.lam = lam
        self.tau = tau
        self.noise = noise  # decision noise

        self.audio_forms = []
        self.audio_idx = {}
        self.visual_refs = []
        self.visual_idx = {}

        self.M = None  # item-level association matrix
        # Construction type associations: for each audio form, 
        # how strongly is it associated with VN vs NC?
        self.M_type = None  # shape: (n_audio, 2) where 0=VN, 1=NC

    def _register(self, trial):
        for a in [trial['a1'], trial['a2']]:
            if a not in self.audio_idx:
                self.audio_idx[a] = len(self.audio_forms)
                self.audio_forms.append(a)
        for v in [trial['v1'], trial['v2']]:
            if v not in self.visual_idx:
                self.visual_idx[v] = len(self.visual_refs)
                self.visual_refs.append(v)

    def train(self, trials):
        # Register all stimuli
        for t in trials:
            self._register(t)

        na = len(self.audio_forms)
        nv = len(self.visual_refs)
        self.M = np.zeros((na, nv))
        self.M_type = np.zeros((na, 2))  # 0=VN, 1=NC

        for trial in trials:
            self._update(trial)

    def _update(self, trial):
        a1 = self.audio_idx[trial['a1']]
        a2 = self.audio_idx[trial['a2']]
        v1 = self.visual_idx[trial['v1']]
        v2 = self.visual_idx[trial['v2']]
        t1_idx = 0 if trial['t1'] == 'VN' else 1
        t2_idx = 0 if trial['t2'] == 'VN' else 1

        audios = [a1, a2]
        visuals = [v1, v2]
        types = [t1_idx, t2_idx]

        # Compute attention (Kachergis 2012)
        attn = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                fam = self.M[audios[i], visuals[j]]
                unc_a = self._entropy(self.M[audios[i], :])
                unc_v = self._entropy(self.M[:, visuals[j]])
                attn[i, j] = self.alpha * fam + self.chi * unc_a * unc_v

        attn_sum = attn.sum()
        if attn_sum > 0:
            attn /= attn_sum
        else:
            attn = np.ones((2, 2)) / 4.0

        # Update item-level associations
        for i in range(2):
            for j in range(2):
                self.M[audios[i], visuals[j]] += attn[i, j] - self.lam * self.M[audios[i], visuals[j]]

        # Update construction-type associations
        # Key insight: the model learns that certain tone patterns
        # are associated with VN (dynamic) or NC (static) constructions
        for i in range(2):
            # Strengthen association between audio[i] and its correct type
            self.M_type[audios[i], types[i]] += attn[i, i] * 0.5
            # Decay
            self.M_type[audios[i], :] *= (1 - self.lam * 0.5)

        # Tone similarity spreading within M_type
        sim = audio_similarity(trial['a1'], trial['a2'], self.tau)
        if 0 < sim < 1.0:
            spread = 0.05 * sim
            for t_idx in range(2):
                avg = (self.M_type[a1, t_idx] + self.M_type[a2, t_idx]) / 2
                self.M_type[a1, t_idx] += spread * (avg - self.M_type[a1, t_idx])
                self.M_type[a2, t_idx] += spread * (avg - self.M_type[a2, t_idx])

    @staticmethod
    def _entropy(dist):
        d = np.maximum(dist, 1e-10)
        s = d.sum()
        if s <= 0:
            return np.log(len(d))
        d = d / s
        return -np.sum(d * np.log(d + 1e-10))

    # --- TEST: Word Memory ---
    def test_word_memory(self):
        """
        2AFC: hear a disyllabic form from exposure, choose between
        correct visual referent and a distractor.
        """
        test_pairs = []
        for v, n in VN_PAIRS_EXPOSURE:
            test_pairs.append((make_disyllabic(v, n, 'VN'), f"{v}{n}"))
        for n, c in NC_PAIRS_EXPOSURE:
            test_pairs.append((make_disyllabic(n, c, 'NC'), f"{n}{c}"))

        correct = 0
        total = 0
        for audio, vis_correct in test_pairs:
            a_idx = self.audio_idx.get(audio)
            vc_idx = self.visual_idx.get(vis_correct)
            if a_idx is None or vc_idx is None:
                continue

            # Random distractor
            distractors = [j for j in range(len(self.visual_refs)) if j != vc_idx]
            vd_idx = np.random.choice(distractors)

            sc = self.M[a_idx, vc_idx] + np.random.normal(0, self.noise)
            sd = self.M[a_idx, vd_idx] + np.random.normal(0, self.noise)

            if sc > sd:
                correct += 1
            elif sc == sd:
                correct += 0.5
            total += 1

        return correct / total if total > 0 else 0.5

    # --- TEST: Generalisation ---
    def test_generalisation(self):
        """
        2AFC: see a visual referent, hear two forms with same segments
        but different tone patterns. Choose the one with correct sandhi.
        
        The model uses construction-type associations: for VN referents,
        the correct form should have stronger VN-type association than
        the incorrect form (which has NC-type tones or no sandhi).
        """
        gen_trials = self._build_gen_trials()

        correct = 0
        total = 0

        for construction, correct_audio, incorrect_audio in gen_trials:
            type_idx = 0 if construction == 'VN' else 1

            sc = self._type_score(correct_audio, type_idx)
            si = self._type_score(incorrect_audio, type_idx)

            sc += np.random.normal(0, self.noise * 0.5)
            si += np.random.normal(0, self.noise * 0.5)

            if sc > si:
                correct += 1
            elif sc == si:
                correct += 0.5
            total += 1

        return correct / total if total > 0 else 0.5

    def _type_score(self, audio_form, type_idx):
        """
        Score an audio form for a given construction type.
        
        Key insight: generalisation is about TONE PATTERNS, not specific
        segmental combinations. So we compare the tone pattern of the 
        test form to the tone patterns of all known forms, regardless
        of whether segments match.
        
        tone_pattern_similarity compares (tone1, tone2) tuples using
        the tau-controlled Gaussian kernel.
        """
        test_tones = (audio_form[1], audio_form[3])  # (tone_σ1, tone_σ2)
        
        total_score = 0.0
        total_weight = 0.0

        for known_form, a_idx in self.audio_idx.items():
            known_tones = (known_form[1], known_form[3])
            # Similarity based on tone pattern only
            sim = tone_similarity(test_tones[0], known_tones[0], self.tau) * \
                  tone_similarity(test_tones[1], known_tones[1], self.tau)
            if sim > 0:
                total_score += sim * self.M_type[a_idx, type_idx]
                total_weight += sim

        if total_weight > 0:
            return total_score / total_weight
        return 0.0

    def _build_gen_trials(self):
        """Build generalisation trials: (construction_type, correct_audio, incorrect_audio)"""
        trials = []

        # Novel VN combinations
        vn_novel = [('V1','N2'),('V1','N4'),('V2','N3'),('V2','N4'),
                    ('V3','N1'),('V3','N2'),('V4','N1'),('V4','N3')]
        for v, n in vn_novel:
            correct = make_disyllabic(v, n, 'VN')
            # Wrong rule: NC sandhi applied
            incorrect_wr = make_disyllabic(v, n, 'NC')
            # No rule: citation tones
            s1, t1 = ALL_WORDS[v]; s2, t2 = ALL_WORDS[n]
            incorrect_nr = (s1, t1, s2, t2)
            trials.append(('VN', correct, incorrect_wr))
            trials.append(('VN', correct, incorrect_nr))

        # Novel NC combinations
        nc_novel = [('N1','C3'),('N1','C4'),('N2','C1'),('N2','C3'),
                    ('N3','C2'),('N3','C4'),('N4','C1'),('N4','C3')]
        for n, c in nc_novel:
            correct = make_disyllabic(n, c, 'NC')
            incorrect_wr = make_disyllabic(n, c, 'VN')
            s1, t1 = ALL_WORDS[n]; s2, t2 = ALL_WORDS[c]
            incorrect_nr = (s1, t1, s2, t2)
            trials.append(('NC', correct, incorrect_wr))
            trials.append(('NC', correct, incorrect_nr))

        return trials


# ============================================================
# 5. SIMULATION ENGINE
# ============================================================

def run_sim(n_sims=200, alpha=1.0, chi=0.1, lam=0.02, tau=2.0, noise=0.3, seed=42):
    np.random.seed(seed)
    mem_list, gen_list = [], []
    for i in range(n_sims):
        m = ToneCSLModel(alpha=alpha, chi=chi, lam=lam, tau=tau, noise=noise)
        m.train(build_exposure_trials())
        mem_list.append(m.test_word_memory())
        gen_list.append(m.test_generalisation())
    return np.array(mem_list), np.array(gen_list)


def param_sweep_tau(tau_values, n_sims=100, **kwargs):
    results = {'tau': tau_values, 'mem_m': [], 'mem_s': [], 'gen_m': [], 'gen_s': []}
    for tau in tau_values:
        mem, gen = run_sim(n_sims=n_sims, tau=tau, **kwargs)
        results['mem_m'].append(mem.mean())
        results['mem_s'].append(mem.std())
        results['gen_m'].append(gen.mean())
        results['gen_s'].append(gen.std())
        print(f"  tau={tau:.1f}: mem={mem.mean():.3f}({mem.std():.3f}), gen={gen.mean():.3f}({gen.std():.3f})")
    return results


def grid_search(n_sims=50):
    """Search over noise and tau to find params matching human data."""
    best = {'error': 999, 'params': {}}
    
    for noise in [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
        for tau in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]:
            mem, gen = run_sim(n_sims=n_sims, noise=noise, tau=tau)
            mm, gm = mem.mean(), gen.mean()
            err = (mm - 0.67)**2 + (gm - 0.58)**2
            if err < best['error']:
                best = {'error': err, 'params': {'noise': noise, 'tau': tau},
                        'mem': mm, 'gen': gm, 'mem_sd': mem.std(), 'gen_sd': gen.std()}
            print(f"  noise={noise:.1f}, tau={tau:.1f}: mem={mm:.3f}, gen={gm:.3f}, err={err:.4f}")
    
    return best


# ============================================================
# 6. PLOTTING
# ============================================================

def plot_sweep(results, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    tau = np.array(results['tau'])
    mm = np.array(results['mem_m']); ms = np.array(results['mem_s'])
    gm = np.array(results['gen_m']); gs = np.array(results['gen_s'])

    ax.plot(tau, mm, 'o-', color='#2166AC', lw=2, label='Word Memory (model)')
    ax.fill_between(tau, mm-ms, mm+ms, alpha=0.15, color='#2166AC')
    ax.plot(tau, gm, 's-', color='#B2182B', lw=2, label='Generalisation (model)')
    ax.fill_between(tau, gm-gs, gm+gs, alpha=0.15, color='#B2182B')

    ax.axhline(0.67, color='#2166AC', ls='--', alpha=0.5, label='Human mem (M=.67)')
    ax.axhline(0.58, color='#B2182B', ls='--', alpha=0.5, label='Human gen (M=.58)')
    ax.axhline(0.50, color='grey', ls=':', alpha=0.4, label='Chance')

    ax.set_xlabel('Tone Precision (τ)', fontsize=13)
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.set_title('Model Performance vs. Tone Precision', fontsize=14)
    ax.legend(fontsize=10, loc='best')
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_dist(mem, gen, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, human, color, title in [
        (axes[0], mem, 0.67, '#2166AC', 'Word Memory'),
        (axes[1], gen, 0.58, '#B2182B', 'Generalisation'),
    ]:
        ax.hist(data, bins=20, color=color, alpha=0.7, edgecolor='white')
        ax.axvline(human, color='red', ls='--', lw=2, label=f'Human M={human}')
        ax.axvline(data.mean(), color='black', ls='-', lw=2, label=f'Model M={data.mean():.2f}')
        ax.set_xlabel('Accuracy'); ax.set_ylabel('Count')
        ax.set_title(title); ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("CSL Tone Learning Simulation v2")
    print("=" * 60)

    # Step 1: Grid search for best parameters
    print("\n--- Grid Search ---")
    best = grid_search(n_sims=50)
    print(f"\nBest params: {best['params']}")
    print(f"  mem={best['mem']:.3f} (SD={best['mem_sd']:.3f}), gen={best['gen']:.3f} (SD={best['gen_sd']:.3f})")
    print(f"  error={best['error']:.4f}")

    # Step 2: Tau sweep at best noise
    print(f"\n--- Tau Sweep (noise={best['params']['noise']}) ---")
    tau_vals = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0])
    sweep = param_sweep_tau(tau_vals, n_sims=100, noise=best['params']['noise'])
    plot_sweep(sweep, 'fig_tau_sweep_v2.png')

    # Step 3: Detailed sim at best params
    print(f"\n--- Detailed Simulation (N=500) ---")
    mem, gen = run_sim(n_sims=500, **best['params'])
    print(f"  Word Memory:    M={mem.mean():.3f}, SD={mem.std():.3f}")
    print(f"  Generalisation: M={gen.mean():.3f}, SD={gen.std():.3f}")
    print(f"  Human:          Mem=.670(SD=.17), Gen=.580(SD=.08)")
    plot_dist(mem, gen, 'fig_dist_v2.png')

    print("\nDone!")
