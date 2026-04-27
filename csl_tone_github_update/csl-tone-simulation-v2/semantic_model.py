"""Distributed semantic representation diagnostic.

This variant replaces atomic visual labels with sparse distributed semantic
vectors. The goal is not to build a full semantic model, but to check whether the
core dissociation between item-level word memory and tone-pattern generalisation
survives when visual referents are represented as feature vectors.

Vector design:
    - Structural dimensions: VN referents share dynamic/agent features;
      NC referents share static/containment features.
    - Noun-identity dimensions: VN and NC referents involving the same noun share
      a small subset of dimensions.
    - Item-specific dimensions: each referent has several random dimensions.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csl_tone_model import (  # noqa: E402
    ALL_WORDS,
    NC_PAIRS_EXPOSURE,
    VN_PAIRS_EXPOSURE,
    ToneCSLModel,
    build_exposure_trials,
    make_disyllabic,
    tone_similarity,
)


def build_semantic_vectors(seed: int = 42, n_dims: int = 50) -> Tuple[Dict[str, np.ndarray], int]:
    """Build sparse semantic vectors for all trained visual referents."""
    rng = np.random.RandomState(seed)
    vectors: Dict[str, np.ndarray] = {}

    noun_feature_pool = list(range(5, 10))
    rng.shuffle(noun_feature_pool)
    noun_features = {
        noun: [noun_feature_pool[i % len(noun_feature_pool)], noun_feature_pool[(i + 2) % len(noun_feature_pool)]]
        for i, noun in enumerate(["N1", "N2", "N3", "N4"])
    }

    def add_item(ref: str, construction: str, noun: str) -> None:
        vec = np.zeros(n_dims)
        if construction == "VN":
            vec[0] = 1  # dynamic/event-like
            vec[2] = 1  # agent/action feature
        else:
            vec[1] = 1  # static/object-like
            vec[3] = 1  # containment/location feature
        vec[4] = 1      # shared referential object feature
        for dim in noun_features[noun]:
            vec[dim] = 1
        for dim in rng.choice(np.arange(10, n_dims), size=3, replace=False):
            vec[dim] = 1
        vectors[ref] = vec

    for verb, noun in VN_PAIRS_EXPOSURE:
        add_item(f"{verb}{noun}", "VN", noun)
    for noun, container in NC_PAIRS_EXPOSURE:
        add_item(f"{noun}{container}", "NC", noun)

    return vectors, n_dims


class ToneCSLModelSemantic:
    """Associative learner mapping auditory forms to distributed semantics."""

    def __init__(
        self,
        alpha: float = 1.0,
        chi: float = 0.1,
        lam: float = 0.02,
        tau: float = 7.0,
        noise: float = 11.0,
        semantic_vectors: Dict[str, np.ndarray] | None = None,
        n_semantic_dims: int = 50,
    ) -> None:
        self.alpha = alpha
        self.chi = chi
        self.lam = lam
        self.tau = tau
        self.noise = noise
        self.semantic_vectors = semantic_vectors or {}
        self.n_sem = n_semantic_dims
        self.audio_forms: List[Tuple[str, int, str, int]] = []
        self.audio_idx: Dict[Tuple[str, int, str, int], int] = {}
        self.M_sem: np.ndarray | None = None

        self.vn_proto = np.zeros(n_semantic_dims)
        self.vn_proto[[0, 2]] = 1
        self.vn_proto /= np.linalg.norm(self.vn_proto)

        self.nc_proto = np.zeros(n_semantic_dims)
        self.nc_proto[[1, 3]] = 1
        self.nc_proto /= np.linalg.norm(self.nc_proto)

    def _register_audio(self, form: Tuple[str, int, str, int]) -> None:
        if form not in self.audio_idx:
            self.audio_idx[form] = len(self.audio_forms)
            self.audio_forms.append(form)

    def train(self, trials: List[Dict[str, object]]) -> None:
        for trial in trials:
            self._register_audio(trial["a1"])
            self._register_audio(trial["a2"])
        self.M_sem = np.zeros((len(self.audio_forms), self.n_sem))
        for trial in trials:
            self._update(trial)

    def _update(self, trial: Dict[str, object]) -> None:
        if self.M_sem is None:
            raise RuntimeError("Model must be initialised before update.")

        audio_indices = [self.audio_idx[trial["a1"]], self.audio_idx[trial["a2"]]]
        sem_vectors = [
            self.semantic_vectors.get(trial["v1"], np.zeros(self.n_sem)),
            self.semantic_vectors.get(trial["v2"], np.zeros(self.n_sem)),
        ]

        attention = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                familiarity = float(np.dot(self.M_sem[audio_indices[i]], sem_vectors[j]))
                audio_uncertainty = self._entropy(self.M_sem[audio_indices[i]])
                semantic_uncertainty = self._entropy(sem_vectors[j])
                attention[i, j] = self.alpha * familiarity + self.chi * audio_uncertainty * semantic_uncertainty

        total_attention = attention.sum()
        attention = attention / total_attention if total_attention > 0 else np.ones((2, 2)) / 4

        for i in range(2):
            for j in range(2):
                self.M_sem[audio_indices[i]] += attention[i, j] * sem_vectors[j]
                self.M_sem[audio_indices[i]] -= self.lam * self.M_sem[audio_indices[i]]

    @staticmethod
    def _entropy(vector: np.ndarray) -> float:
        values = np.abs(vector)
        total = values.sum()
        if total <= 0:
            return np.log(max(len(vector), 1))
        probs = np.maximum(values / total, 1e-10)
        return float(-np.sum(probs * np.log(probs)))

    def test_word_memory(self) -> float:
        """2AFC form-referent memory using semantic-vector dot products."""
        if self.M_sem is None:
            raise RuntimeError("Train the model before testing.")
        pairs = []
        for verb, noun in VN_PAIRS_EXPOSURE:
            pairs.append((make_disyllabic(verb, noun, "VN"), f"{verb}{noun}"))
        for noun, container in NC_PAIRS_EXPOSURE:
            pairs.append((make_disyllabic(noun, container, "NC"), f"{noun}{container}"))

        correct = 0
        total = 0
        for audio, correct_ref in pairs:
            audio_idx = self.audio_idx.get(audio)
            correct_vec = self.semantic_vectors.get(correct_ref)
            if audio_idx is None or correct_vec is None:
                continue
            distractor_ref = np.random.choice([r for r in self.semantic_vectors if r != correct_ref])
            distractor_vec = self.semantic_vectors[distractor_ref]
            correct_score = float(np.dot(self.M_sem[audio_idx], correct_vec)) + np.random.normal(0, self.noise)
            distractor_score = float(np.dot(self.M_sem[audio_idx], distractor_vec)) + np.random.normal(0, self.noise)
            if correct_score > distractor_score:
                correct += 1
            elif correct_score == distractor_score:
                correct += 0.5
            total += 1
        return correct / total if total > 0 else 0.5

    def test_generalisation(self) -> float:
        """2AFC tone-pattern generalisation using semantic construction prototypes."""
        if self.M_sem is None:
            raise RuntimeError("Train the model before testing.")
        correct = 0
        total = 0
        for construction, correct_audio, incorrect_audio in self._build_generalisation_trials():
            proto = self.vn_proto if construction == "VN" else self.nc_proto
            correct_score = self._semantic_type_score(correct_audio, proto) + np.random.normal(0, self.noise * 0.5)
            incorrect_score = self._semantic_type_score(incorrect_audio, proto) + np.random.normal(0, self.noise * 0.5)
            if correct_score > incorrect_score:
                correct += 1
            elif correct_score == incorrect_score:
                correct += 0.5
            total += 1
        return correct / total if total > 0 else 0.5

    def _semantic_type_score(self, audio_form: Tuple[str, int, str, int], prototype: np.ndarray) -> float:
        if self.M_sem is None:
            raise RuntimeError("Train the model before testing.")
        test_tones = (audio_form[1], audio_form[3])
        total_score = 0.0
        total_weight = 0.0
        for known_form, audio_idx in self.audio_idx.items():
            known_tones = (known_form[1], known_form[3])
            sim = tone_similarity(test_tones[0], known_tones[0], self.tau) * tone_similarity(
                test_tones[1], known_tones[1], self.tau
            )
            if sim > 0:
                total_score += sim * float(np.dot(self.M_sem[audio_idx], prototype))
                total_weight += sim
        return total_score / total_weight if total_weight > 0 else 0.0

    @staticmethod
    def _build_generalisation_trials() -> List[Tuple[str, Tuple[str, int, str, int], Tuple[str, int, str, int]]]:
        trials = []
        novel_vn = [
            ("V1", "N2"), ("V1", "N4"), ("V2", "N3"), ("V2", "N4"),
            ("V3", "N1"), ("V3", "N2"), ("V4", "N1"), ("V4", "N3"),
        ]
        for verb, noun in novel_vn:
            correct = make_disyllabic(verb, noun, "VN")
            wrong_rule = make_disyllabic(verb, noun, "NC")
            seg1, tone1 = ALL_WORDS[verb]
            seg2, tone2 = ALL_WORDS[noun]
            no_rule = (seg1, tone1, seg2, tone2)
            trials.append(("VN", correct, wrong_rule))
            trials.append(("VN", correct, no_rule))

        novel_nc = [
            ("N1", "C3"), ("N1", "C4"), ("N2", "C1"), ("N2", "C3"),
            ("N3", "C2"), ("N3", "C4"), ("N4", "C1"), ("N4", "C3"),
        ]
        for noun, container in novel_nc:
            correct = make_disyllabic(noun, container, "NC")
            wrong_rule = make_disyllabic(noun, container, "VN")
            seg1, tone1 = ALL_WORDS[noun]
            seg2, tone2 = ALL_WORDS[container]
            no_rule = (seg1, tone1, seg2, tone2)
            trials.append(("NC", correct, wrong_rule))
            trials.append(("NC", correct, no_rule))
        return trials


def run_semantic_comparison(tau: float = 7.0, n_sims: int = 100, seed: int = 42) -> Dict[str, float]:
    """Compare the semantic variant against the atomic-label base model."""
    semantic_vectors, n_dims = build_semantic_vectors(seed=seed)
    sem_memory, sem_generalisation = [], []
    base_memory, base_generalisation = [], []
    for sim_seed in range(seed, seed + n_sims):
        np.random.seed(sim_seed)
        semantic_model = ToneCSLModelSemantic(
            tau=tau,
            noise=11.0,
            semantic_vectors=semantic_vectors,
            n_semantic_dims=n_dims,
        )
        semantic_model.train(build_exposure_trials())
        sem_memory.append(semantic_model.test_word_memory())
        sem_generalisation.append(semantic_model.test_generalisation())

        np.random.seed(sim_seed)
        base_model = ToneCSLModel(tau=tau, noise=11.0)
        base_model.train(build_exposure_trials())
        base_memory.append(base_model.test_word_memory())
        base_generalisation.append(base_model.test_generalisation())

    return {
        "semantic_memory": float(np.mean(sem_memory)),
        "semantic_generalisation": float(np.mean(sem_generalisation)),
        "base_memory": float(np.mean(base_memory)),
        "base_generalisation": float(np.mean(base_generalisation)),
    }


if __name__ == "__main__":
    print("=" * 72)
    print("Distributed semantic representation diagnostic")
    print("=" * 72)
    print(f"{'tau':>5} | {'semantic memory':>15} | {'semantic gen':>12} | {'base memory':>11} | {'base gen':>8}")
    print("-" * 72)
    for tau_value in [0.5, 2.0, 5.0, 7.0, 10.0, 20.0]:
        out = run_semantic_comparison(tau=tau_value, n_sims=10, seed=42)
        print(
            f"{tau_value:5.1f} | {out['semantic_memory']:15.3f} | "
            f"{out['semantic_generalisation']:12.3f} | {out['base_memory']:11.3f} | "
            f"{out['base_generalisation']:8.3f}"
        )
