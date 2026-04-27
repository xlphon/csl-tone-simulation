"""Productive tone-sandhi diagnostic without citation exposure.

This script asks a deliberately stronger question than the 2AFC
classificatory-generalisation test:

    citation tones + construction type -> sandhi output tones

The base CSL model has learned associations between *observed sandhi surface
patterns* and construction types, but it has not learned a citation-to-sandhi
mapping table. Therefore this diagnostic should be interpreted as an upper-limit
exemplar-selection baseline, not as a psychologically complete production model.
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
    apply_sandhi,
    build_exposure_trials,
    tone_similarity,
)

TonePattern = Tuple[int, int]


def build_production_trials() -> List[Dict[str, object]]:
    """Return all VN and NC productive-sandhi trials."""
    trials: List[Dict[str, object]] = []

    for verb in ["V1", "V2", "V3", "V4"]:
        for noun in ["N1", "N2", "N3", "N4"]:
            seg1, tone1 = ALL_WORDS[verb]
            seg2, tone2 = ALL_WORDS[noun]
            target = apply_sandhi(seg1, tone1, seg2, tone2, "VN")
            trials.append(
                {
                    "construction": "VN",
                    "word1": verb,
                    "word2": noun,
                    "citation": (tone1, tone2),
                    "target": (target[1], target[3]),
                    "trained": (verb, noun) in VN_PAIRS_EXPOSURE,
                }
            )

    for noun in ["N1", "N2", "N3", "N4"]:
        for container in ["C1", "C2", "C3", "C4"]:
            seg1, tone1 = ALL_WORDS[noun]
            seg2, tone2 = ALL_WORDS[container]
            target = apply_sandhi(seg1, tone1, seg2, tone2, "NC")
            trials.append(
                {
                    "construction": "NC",
                    "word1": noun,
                    "word2": container,
                    "citation": (tone1, tone2),
                    "target": (target[1], target[3]),
                    "trained": (noun, container) in NC_PAIRS_EXPOSURE,
                }
            )
    return trials


class ToneCSLModelNoCitationProduction(ToneCSLModel):
    """Base model plus a no-citation productive-sandhi probe."""

    def known_tone_patterns(self) -> List[TonePattern]:
        """Unique tone patterns observed in the exposure corpus."""
        return sorted({(form[1], form[3]) for form in self.audio_forms})

    def pattern_type_score(self, tone_pattern: TonePattern, construction: str) -> float:
        """Similarity-weighted type-association score for a candidate tone pattern."""
        type_idx = 0 if construction == "VN" else 1
        total_score = 0.0
        total_weight = 0.0
        for known_form, audio_idx in self.audio_idx.items():
            known_pattern = (known_form[1], known_form[3])
            sim = tone_similarity(tone_pattern[0], known_pattern[0], self.tau) * tone_similarity(
                tone_pattern[1], known_pattern[1], self.tau
            )
            if sim > 0:
                total_score += sim * self.M_type[audio_idx, type_idx]
                total_weight += sim
        return total_score / total_weight if total_weight > 0 else 0.0

    def produce_from_exemplars(self, citation: TonePattern, construction: str) -> Tuple[TonePattern, float]:
        """Select a sandhi tone pattern from the stored exemplar pool.

        The citation-similarity term is only a weak prior. Because the model has
        not observed citation-to-sandhi mappings, this is not true derivation.
        """
        best_pattern: TonePattern | None = None
        best_score = -np.inf
        for pattern in self.known_tone_patterns():
            type_score = self.pattern_type_score(pattern, construction)
            citation_prior = tone_similarity(pattern[0], citation[0], 1.0) * tone_similarity(pattern[1], citation[1], 1.0)
            score = type_score * (0.3 + 0.7 * citation_prior)
            score += np.random.normal(0, self.noise * 0.1)
            if score > best_score:
                best_score = score
                best_pattern = pattern
        if best_pattern is None:
            return citation, 0.0
        return best_pattern, float(best_score)

    def test_productive_sandhi(self, verbose: bool = False) -> Dict[str, object]:
        """Evaluate productive output over all VN and NC combinations."""
        results = []
        for trial in build_production_trials():
            predicted, score = self.produce_from_exemplars(trial["citation"], trial["construction"])
            target = trial["target"]
            sigma1 = predicted[0] == target[0]
            sigma2 = predicted[1] == target[1]
            row = {
                **trial,
                "predicted": predicted,
                "correct_sigma1": sigma1,
                "correct_sigma2": sigma2,
                "correct_both": sigma1 and sigma2,
                "score": score,
            }
            results.append(row)
            if verbose:
                mark = "✓" if row["correct_both"] else "✗"
                print(
                    f"{mark} {trial['construction']} {trial['word1']}-{trial['word2']} "
                    f"citation={trial['citation']} predicted={predicted} target={target}"
                )

        return {
            "both": float(np.mean([r["correct_both"] for r in results])),
            "sigma1": float(np.mean([r["correct_sigma1"] for r in results])),
            "sigma2": float(np.mean([r["correct_sigma2"] for r in results])),
            "total": len(results),
            "results": results,
        }


if __name__ == "__main__":
    print("=" * 72)
    print("Productive sandhi without citation exposure")
    print("=" * 72)
    print("This is an exemplar-selection baseline, not true citation-to-sandhi production.\n")

    for tau in [0.5, 2.0, 5.0, 7.0, 10.0, 20.0]:
        both, s1, s2 = [], [], []
        for seed in range(30):
            np.random.seed(seed)
            model = ToneCSLModelNoCitationProduction(tau=tau, noise=11.0)
            model.train(build_exposure_trials())
            result = model.test_productive_sandhi()
            both.append(result["both"])
            s1.append(result["sigma1"])
            s2.append(result["sigma2"])
        print(
            f"tau={tau:>4}: both={np.mean(both):.3f}, "
            f"sigma1={np.mean(s1):.3f}, sigma2={np.mean(s2):.3f}"
        )
