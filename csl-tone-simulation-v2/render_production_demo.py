"""Render productive tone-sandhi model outputs as tables, F0 plots, audio,
and segment-level diagnostics.

Place this file inside `csl-tone-simulation-v2/` and run, for example:

    python render_production_demo.py --tau 7.0 --noise 11.0 --seed 42
    python render_production_demo.py --strict-noun-only --tau 7.0 --seed 42

The productive model outputs discrete tone codes, e.g. (44, 13). This script
turns those tone choices into:

    demo_outputs/production_outputs.csv
    demo_outputs/segment_memory_outputs.csv
    demo_outputs/f0_plots/*.png
    demo_outputs/audio/*_target.wav
    demo_outputs/audio/*_predicted.wav

Important interpretation:
    - In the productive-sandhi probe, segmental identity is clamped by the query:
      the model is given the two lexical items/segments and predicts their tones.
      Therefore segment correctness in `production_outputs.csv` is a design check,
      not evidence that the production model learned segments.
    - `segment_memory_outputs.csv` is the meaningful segment-level diagnostic: it
      asks the learned item-level CSL matrix which trained auditory form best maps
      to each trained visual referent, then evaluates segments and tones separately.

The audio is a simple sonification of F0 contours, not natural speech synthesis.
"""

from __future__ import annotations

import argparse
import csv
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This demo requires matplotlib. Install it with: pip install matplotlib") from exc

try:
    from csl_tone_model import (
        ALL_WORDS,
        NOUNS,
        VN_PAIRS_EXPOSURE,
        NC_PAIRS_EXPOSURE,
        TONE_LETTER_TO_PITCH,
        apply_sandhi,
        build_exposure_trials,
        make_disyllabic,
    )
    from productive_sandhi_with_citation import ToneCSLModelV3b, perceive_tone
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import the simulation modules. Put this script in the "
        "`csl-tone-simulation-v2/` directory and run it from there."
    ) from exc

TonePair = Tuple[int, int]
AudioForm = Tuple[str, int, str, int]


@dataclass(frozen=True)
class ProductionTrial:
    construction: str
    word1: str
    word2: str
    seg1: str
    seg2: str
    citation: TonePair
    target: TonePair
    trained: bool


class ToneCSLModelV3bNounOnly(ToneCSLModelV3b):
    """Stricter citation model: only the overt noun citation cue is recorded.

    The existing V3b model records citation-to-sandhi correspondences for all
    syllables for which a lexical citation tone can be looked up. That is useful
    as an idealised productive model. This stricter subclass records only the
    noun that is overtly available as the monosyllabic citation cue in the
    behavioural exposure phase.
    """

    def _process_monosyllabic(self, trial):  # noqa: D401 - matches parent API
        noun_seg, noun_citation_tone = self._shared_noun_in_trial(trial)
        if noun_seg is None:
            return

        for audio_key, type_key in [("a1", "t1"), ("a2", "t2")]:
            disyllable = trial[audio_key]
            construction = trial[type_key]
            seg1, sandhi_t1, seg2, sandhi_t2 = disyllable

            if construction == "VN" and seg2 == noun_seg:
                self._record(
                    perceive_tone(noun_citation_tone, self.tau, self._rng),
                    perceive_tone(sandhi_t2, self.tau, self._rng),
                    "sigma2",
                    construction,
                )
            elif construction == "NC" and seg1 == noun_seg:
                self._record(
                    perceive_tone(noun_citation_tone, self.tau, self._rng),
                    perceive_tone(sandhi_t1, self.tau, self._rng),
                    "sigma1",
                    construction,
                )

    @staticmethod
    def _shared_noun_in_trial(trial) -> Tuple[str | None, int | None]:
        segments_a1 = {trial["a1"][0], trial["a1"][2]}
        segments_a2 = {trial["a2"][0], trial["a2"][2]}
        shared_segments = segments_a1.intersection(segments_a2)

        for _noun_key, (seg, citation_tone) in NOUNS.items():
            if seg in shared_segments:
                return seg, citation_tone
        return None, None


def build_production_trials() -> List[ProductionTrial]:
    """Build all VN and NC production trials, including trained and novel pairs."""
    trials: List[ProductionTrial] = []

    for verb in ["V1", "V2", "V3", "V4"]:
        for noun in ["N1", "N2", "N3", "N4"]:
            seg1, cit1 = ALL_WORDS[verb]
            seg2, cit2 = ALL_WORDS[noun]
            target_form = apply_sandhi(seg1, cit1, seg2, cit2, "VN")
            trials.append(
                ProductionTrial(
                    construction="VN",
                    word1=verb,
                    word2=noun,
                    seg1=seg1,
                    seg2=seg2,
                    citation=(cit1, cit2),
                    target=(target_form[1], target_form[3]),
                    trained=(verb, noun) in VN_PAIRS_EXPOSURE,
                )
            )

    for noun in ["N1", "N2", "N3", "N4"]:
        for container in ["C1", "C2", "C3", "C4"]:
            seg1, cit1 = ALL_WORDS[noun]
            seg2, cit2 = ALL_WORDS[container]
            target_form = apply_sandhi(seg1, cit1, seg2, cit2, "NC")
            trials.append(
                ProductionTrial(
                    construction="NC",
                    word1=noun,
                    word2=container,
                    seg1=seg1,
                    seg2=seg2,
                    citation=(cit1, cit2),
                    target=(target_form[1], target_form[3]),
                    trained=(noun, container) in NC_PAIRS_EXPOSURE,
                )
            )

    return trials


def has_mapping(model: ToneCSLModelV3b, citation_tone: int, position: str, construction: str) -> bool:
    """Return whether the model has any relevant mapping for this query."""
    return any(pos == position and con == construction for (_tone, pos, con) in model.citation_to_sandhi)


def train_productive_model(
    tau: float,
    noise: float,
    seed: int,
    strict_noun_only: bool,
) -> ToneCSLModelV3b:
    """Train and return the productive model used by all probes."""
    np.random.seed(seed)
    model_cls = ToneCSLModelV3bNounOnly if strict_noun_only else ToneCSLModelV3b
    model = model_cls(tau=tau, noise=noise)
    model.train(build_exposure_trials(), seed=seed)
    return model


def run_productive_probe(model: ToneCSLModelV3b) -> List[Dict[str, object]]:
    """Return item-level tone-production outputs plus clamped segment diagnostics."""
    rows: List[Dict[str, object]] = []
    for trial in build_production_trials():
        pred1, confidence1 = model.produce_sandhi(trial.citation[0], "sigma1", trial.construction)
        pred2, confidence2 = model.produce_sandhi(trial.citation[1], "sigma2", trial.construction)
        predicted_tones = (int(pred1), int(pred2))

        # Productive tone-sandhi probe is asked to transform the tones of a given
        # segmental pair. It does not independently generate segmental strings.
        predicted_seg1 = trial.seg1
        predicted_seg2 = trial.seg2

        correct_sigma1 = predicted_tones[0] == trial.target[0]
        correct_sigma2 = predicted_tones[1] == trial.target[1]
        correct_tones_both = correct_sigma1 and correct_sigma2
        correct_segment1 = predicted_seg1 == trial.seg1
        correct_segment2 = predicted_seg2 == trial.seg2
        correct_segments_both = correct_segment1 and correct_segment2
        correct_whole_form = correct_segments_both and correct_tones_both

        rows.append(
            {
                "probe": "productive_tone_given_segments",
                "construction": trial.construction,
                "word1": trial.word1,
                "word2": trial.word2,
                "target_seg1": trial.seg1,
                "target_seg2": trial.seg2,
                "predicted_seg1": predicted_seg1,
                "predicted_seg2": predicted_seg2,
                "target_segments": f"{trial.seg1}-{trial.seg2}",
                "predicted_segments": f"{predicted_seg1}-{predicted_seg2}",
                "citation": tone_pair_to_string(trial.citation),
                "target_tones": tone_pair_to_string(trial.target),
                "predicted_tones": tone_pair_to_string(predicted_tones),
                "target_sigma1": trial.target[0],
                "target_sigma2": trial.target[1],
                "predicted_sigma1": predicted_tones[0],
                "predicted_sigma2": predicted_tones[1],
                "correct_segment1": correct_segment1,
                "correct_segment2": correct_segment2,
                "correct_segments_both": correct_segments_both,
                "correct_sigma1": correct_sigma1,
                "correct_sigma2": correct_sigma2,
                "correct_tones_both": correct_tones_both,
                "correct_whole_form": correct_whole_form,
                "trained": trial.trained,
                "confidence_sigma1": round(float(confidence1), 4),
                "confidence_sigma2": round(float(confidence2), 4),
                "mapping_available_sigma1": has_mapping(model, trial.citation[0], "sigma1", trial.construction),
                "mapping_available_sigma2": has_mapping(model, trial.citation[1], "sigma2", trial.construction),
                "segment_eval_note": "segments_are_clamped_by_query_not_independently_generated",
            }
        )

    return rows


def trained_target_items() -> List[Tuple[str, str, str, AudioForm]]:
    """Return trained visual refs and their target auditory forms.

    Each tuple is (construction, word1, word2, target_audio_form).
    """
    items: List[Tuple[str, str, str, AudioForm]] = []
    for v, n in VN_PAIRS_EXPOSURE:
        items.append(("VN", v, n, make_disyllabic(v, n, "VN")))
    for n, c in NC_PAIRS_EXPOSURE:
        items.append(("NC", n, c, make_disyllabic(n, c, "NC")))
    return items


def score_audio_for_visual(model: ToneCSLModelV3b, audio: AudioForm, visual_ref: str, noise_sd: float, rng: np.random.RandomState) -> float:
    """Return the learned M[audio, visual] association score, optionally noised."""
    a_idx = model.audio_idx.get(audio)
    v_idx = model.visual_idx.get(visual_ref)
    if a_idx is None or v_idx is None:
        return float("-inf")
    score = float(model.M[a_idx, v_idx])
    if noise_sd > 0:
        score += float(rng.normal(0.0, noise_sd))
    return score


def run_segment_memory_probe(
    model: ToneCSLModelV3b,
    segment_memory_noise: float = 0.0,
    seed: int = 42,
) -> List[Dict[str, object]]:
    """Evaluate whether the item-level CSL matrix retrieves correct segments.

    This is the meaningful segment-level diagnostic. It is restricted to trained
    visual referents because novel combinations such as V1N2 were never registered
    as visual references in the item-level CSL matrix. For each trained referent,
    the model chooses the associated auditory form with the highest M score.
    """
    rng = np.random.RandomState(seed + 100_000)
    rows: List[Dict[str, object]] = []

    for construction, word1, word2, target_audio in trained_target_items():
        visual_ref = f"{word1}{word2}"
        candidates = list(model.audio_forms)
        scored = [
            (score_audio_for_visual(model, cand, visual_ref, segment_memory_noise, rng), cand)
            for cand in candidates
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        predicted_score, predicted_audio = scored[0]
        target_score = score_audio_for_visual(model, target_audio, visual_ref, 0.0, rng)
        target_rank = 1 + [cand for _score, cand in scored].index(target_audio) if target_audio in [cand for _score, cand in scored] else None

        target_seg1, target_tone1, target_seg2, target_tone2 = target_audio
        predicted_seg1, predicted_tone1, predicted_seg2, predicted_tone2 = predicted_audio

        correct_segment1 = predicted_seg1 == target_seg1
        correct_segment2 = predicted_seg2 == target_seg2
        correct_segments_both = correct_segment1 and correct_segment2
        correct_tone1 = predicted_tone1 == target_tone1
        correct_tone2 = predicted_tone2 == target_tone2
        correct_tones_both = correct_tone1 and correct_tone2
        correct_whole_form = predicted_audio == target_audio

        rows.append(
            {
                "probe": "segment_memory_trained_items",
                "construction": construction,
                "word1": word1,
                "word2": word2,
                "visual_ref": visual_ref,
                "target_seg1": target_seg1,
                "target_seg2": target_seg2,
                "predicted_seg1": predicted_seg1,
                "predicted_seg2": predicted_seg2,
                "target_segments": f"{target_seg1}-{target_seg2}",
                "predicted_segments": f"{predicted_seg1}-{predicted_seg2}",
                "target_tones": tone_pair_to_string((target_tone1, target_tone2)),
                "predicted_tones": tone_pair_to_string((predicted_tone1, predicted_tone2)),
                "target_form": audio_form_to_string(target_audio),
                "predicted_form": audio_form_to_string(predicted_audio),
                "correct_segment1": correct_segment1,
                "correct_segment2": correct_segment2,
                "correct_segments_both": correct_segments_both,
                "correct_tone1": correct_tone1,
                "correct_tone2": correct_tone2,
                "correct_tones_both": correct_tones_both,
                "correct_whole_form": correct_whole_form,
                "target_score": round(float(target_score), 6),
                "predicted_score": round(float(predicted_score), 6),
                "target_rank": target_rank,
                "candidate_count": len(candidates),
                "segment_eval_note": "learned_item_level_M_retrieval_trained_refs_only",
            }
        )
    return rows


def tone_pair_to_string(pair: TonePair) -> str:
    return f"{pair[0]}-{pair[1]}"


def audio_form_to_string(audio: AudioForm) -> str:
    return f"{audio[0]}{audio[1]}-{audio[2]}{audio[3]}"


def tone_code_to_f0(tone_code: int, n_samples: int) -> np.ndarray:
    """Convert a Chao two-digit tone code into a linear F0 contour."""
    start_digit = tone_code // 10
    end_digit = tone_code % 10
    try:
        f0_start = float(TONE_LETTER_TO_PITCH[start_digit])
        f0_end = float(TONE_LETTER_TO_PITCH[end_digit])
    except KeyError as exc:
        raise ValueError(f"Tone code {tone_code} is not a valid two-digit Chao tone code") from exc
    return np.linspace(f0_start, f0_end, n_samples)


def pair_to_f0(
    tone_pair: TonePair,
    sample_rate: int = 22050,
    syllable_dur: float = 0.45,
    gap_dur: float = 0.06,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return time and F0 arrays for a disyllabic tone pair."""
    n_syl = int(round(sample_rate * syllable_dur))
    n_gap = int(round(sample_rate * gap_dur))

    f0_1 = tone_code_to_f0(tone_pair[0], n_syl)
    f0_gap = np.full(n_gap, np.nan)
    f0_2 = tone_code_to_f0(tone_pair[1], n_syl)
    f0 = np.concatenate([f0_1, f0_gap, f0_2])
    time = np.arange(f0.size) / sample_rate
    return time, f0


def synthesize_tone_pair(
    tone_pair: TonePair,
    sample_rate: int = 22050,
    syllable_dur: float = 0.45,
    gap_dur: float = 0.06,
) -> np.ndarray:
    """Create a simple harmonic sonification for a disyllabic tone pair."""
    n_syl = int(round(sample_rate * syllable_dur))
    n_gap = int(round(sample_rate * gap_dur))

    syllables = []
    for tone_code in tone_pair:
        f0 = tone_code_to_f0(tone_code, n_syl)
        phase = 2.0 * math.pi * np.cumsum(f0) / sample_rate
        signal = 0.60 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.12 * np.sin(3 * phase)

        attack = max(1, int(0.04 * sample_rate))
        release = max(1, int(0.05 * sample_rate))
        envelope = np.ones(n_syl)
        envelope[:attack] = np.linspace(0.0, 1.0, attack)
        envelope[-release:] = np.linspace(1.0, 0.0, release)
        syllables.append(signal * envelope)

    gap = np.zeros(n_gap)
    audio = np.concatenate([syllables[0], gap, syllables[1]])
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = 0.95 * audio / peak
    return audio.astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = 22050) -> None:
    """Write mono float audio as 16-bit PCM WAV without external dependencies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def plot_f0_pair(row: Dict[str, object], path: Path, sample_rate: int = 22050) -> None:
    """Plot target and predicted F0 tracks for one production item."""
    target = (int(row["target_sigma1"]), int(row["target_sigma2"]))
    predicted = (int(row["predicted_sigma1"]), int(row["predicted_sigma2"]))

    t_target, f0_target = pair_to_f0(target, sample_rate=sample_rate)
    t_pred, f0_pred = pair_to_f0(predicted, sample_rate=sample_rate)

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 4.2))
    plt.plot(t_target, f0_target, label=f"target {tone_pair_to_string(target)}", linewidth=2)
    plt.plot(t_pred, f0_pred, label=f"predicted {tone_pair_to_string(predicted)}", linewidth=2, linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("F0 (Hz)")
    plt.title(
        f"{row['construction']} {row['word1']}-{row['word2']} "
        f"({row['target_segments']}) | tone_correct={row['correct_tones_both']}"
    )
    plt.ylim(75, 175)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def safe_stem(row: Dict[str, object], index: int) -> str:
    status = "OK" if row["correct_tones_both"] else "ERR"
    return f"{index:02d}_{row['construction']}_{row['word1']}_{row['word2']}_{status}"


def write_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_demo_rows(rows: Sequence[Dict[str, object]], n_examples: int) -> List[Dict[str, object]]:
    """Select a useful mix: errors first, then correct outputs."""
    errors = [r for r in rows if not r["correct_tones_both"]]
    correct = [r for r in rows if r["correct_tones_both"]]
    return list(errors[:n_examples]) if len(errors) >= n_examples else list(errors + correct[: n_examples - len(errors)])


def render_outputs(
    production_rows: Sequence[Dict[str, object]],
    segment_rows: Sequence[Dict[str, object]],
    output_dir: Path,
    n_examples: int,
    sample_rate: int,
) -> None:
    write_csv(production_rows, output_dir / "production_outputs.csv")
    write_csv(segment_rows, output_dir / "segment_memory_outputs.csv")

    selected = select_demo_rows(production_rows, n_examples)
    for i, row in enumerate(selected, start=1):
        stem = safe_stem(row, i)
        target = (int(row["target_sigma1"]), int(row["target_sigma2"]))
        predicted = (int(row["predicted_sigma1"]), int(row["predicted_sigma2"]))

        plot_f0_pair(row, output_dir / "f0_plots" / f"{stem}.png", sample_rate=sample_rate)
        write_wav(output_dir / "audio" / f"{stem}_target.wav", synthesize_tone_pair(target, sample_rate=sample_rate), sample_rate)
        write_wav(output_dir / "audio" / f"{stem}_predicted.wav", synthesize_tone_pair(predicted, sample_rate=sample_rate), sample_rate)


def mean_bool(rows: Sequence[Dict[str, object]], key: str) -> float:
    return float(np.mean([bool(r[key]) for r in rows])) if rows else float("nan")


def print_summary(
    production_rows: Sequence[Dict[str, object]],
    segment_rows: Sequence[Dict[str, object]],
    output_dir: Path,
    strict_noun_only: bool,
) -> None:
    trained = [r for r in production_rows if r["trained"]]
    novel = [r for r in production_rows if not r["trained"]]

    print("\nProduction demo finished")
    print("=" * 34)
    print(f"Mode: {'strict noun-only citation cue' if strict_noun_only else 'citation-augmented / all lexical citation lookup'}")

    print("\nProductive tone probe: given lexical segments, predict tones")
    print(f"Items:                 {len(production_rows)}")
    print(f"Tone: both correct:    {mean_bool(production_rows, 'correct_tones_both'):.3f}")
    print(f"Tone: sigma1 correct:  {mean_bool(production_rows, 'correct_sigma1'):.3f}")
    print(f"Tone: sigma2 correct:  {mean_bool(production_rows, 'correct_sigma2'):.3f}")
    print(f"Whole form correct:    {mean_bool(production_rows, 'correct_whole_form'):.3f}")
    print(f"Trained items:         {mean_bool(trained, 'correct_tones_both'):.3f}")
    print(f"Novel items:           {mean_bool(novel, 'correct_tones_both'):.3f}")
    print("Segment correctness:   1.000 by design because segments are clamped in this probe")

    print("\nSegment memory probe: learned item-level M, trained visual refs only")
    print(f"Items:                 {len(segment_rows)}")
    print(f"Segments both correct: {mean_bool(segment_rows, 'correct_segments_both'):.3f}")
    print(f"Tones both correct:    {mean_bool(segment_rows, 'correct_tones_both'):.3f}")
    print(f"Whole form correct:    {mean_bool(segment_rows, 'correct_whole_form'):.3f}")

    print(f"\nSaved outputs to: {output_dir.resolve()}")
    print(f"- Productive CSV:      {output_dir / 'production_outputs.csv'}")
    print(f"- Segment-memory CSV:  {output_dir / 'segment_memory_outputs.csv'}")
    print(f"- F0 plots:            {output_dir / 'f0_plots'}")
    print(f"- Audio:               {output_dir / 'audio'}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau", type=float, default=7.0, help="Tone precision parameter used by the model.")
    parser.add_argument("--noise", type=float, default=11.0, help="Decision noise parameter used by the model.")
    parser.add_argument("--seed", type=int, default=4, help="Random seed for reproducible model outputs.")
    parser.add_argument("--output-dir", type=Path, default=Path("demo_outputs"), help="Directory for CSV, plots, and audio.")
    parser.add_argument("--n-examples", type=int, default=8, help="Number of examples to render as plots/audio.")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Sample rate for audio rendering.")
    parser.add_argument(
        "--strict-noun-only",
        action="store_true",
        help=(
            "Record only the overt noun citation cue. Without this flag, the demo follows "
            "the current citation-augmented productive model and records all lexical citation lookups."
        ),
    )
    parser.add_argument(
        "--segment-memory-noise",
        type=float,
        default=0.0,
        help=(
            "Optional noise added when the segment-memory probe retrieves the best auditory form "
            "from the item-level M matrix. Default 0 gives a deterministic diagnostic."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    model = train_productive_model(
        tau=args.tau,
        noise=args.noise,
        seed=args.seed,
        strict_noun_only=args.strict_noun_only,
    )
    production_rows = run_productive_probe(model)
    segment_rows = run_segment_memory_probe(model, segment_memory_noise=args.segment_memory_noise, seed=args.seed)
    render_outputs(production_rows, segment_rows, args.output_dir, args.n_examples, args.sample_rate)
    print_summary(production_rows, segment_rows, args.output_dir, args.strict_noun_only)


if __name__ == "__main__":
    main()
