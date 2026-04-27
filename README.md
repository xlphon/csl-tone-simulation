# Cross-Situational Tone Learning: Computational Simulation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xlphon/csl-tone-simulation/blob/main/csl-tone-simulation-v2/csl_tone_simulation_colab.ipynb)

Computational simulations of cross-situational word learning with structured auditory representations and tone sandhi.

This repository contains the simulation code for:

> Xinbing Luo. *Computational Simulation of Cross-Situational Tone Learning.* PhD thesis chapter, University of Cambridge, 2026.

The simulations extend the associative cross-situational learning model of Kachergis, Yu, and Shiffrin (2012) to ask how learners can acquire both word–referent mappings and systematic tonal alternations from ambiguous input.

## What this repository demonstrates

The models distinguish three increasingly demanding levels of tone-sandhi knowledge:

| Level | Question | Main diagnostic |
|---|---|---|
| **Word memory** | Can the learner map heard disyllabic surface forms to visual referents? | `M`, the item-level form–referent association matrix |
| **Classificatory generalisation** | Can the learner identify whether a novel sandhi surface pattern is compatible with a construction type? | `M_type`, the type-level tone-pattern association matrix |
| **Productive sandhi** | Can the learner generate a sandhi output from citation tones and a construction type? | citation-to-sandhi mapping tables |

The productive-sandhi diagnostic is intentionally stronger than the 2AFC generalisation task. The model's productive output is a pair of discrete Chao-tone codes, such as `44-13`; it is not a speech waveform. The demo script provides an **auditory sonification** of those selected tone patterns.

## Run without downloading: Colab

Click the badge at the top of this README, or open:

```text
https://colab.research.google.com/github/xlphon/csl-tone-simulation/blob/main/csl-tone-simulation-v2/csl_tone_simulation_colab.ipynb
```

The notebook clones the repository inside Colab, installs the small dependency set, runs the main simulations, and displays the production-demo tables, F0 plots, and audio renderings.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── LICENSE
├── csl-tone-simulation/                  # Older version, retained for record if present
└── csl-tone-simulation-v2/
    ├── README.md
    ├── requirements.txt
    ├── csl_tone_model.py                 # Base associative model with tone similarity
    ├── productive_sandhi_no_citation.py  # Productive diagnostic without citation mappings
    ├── productive_sandhi_with_citation.py# Citation-augmented productive model
    ├── semantic_model.py                 # Distributed semantic representation diagnostic
    ├── render_production_demo.py         # CSV + F0 plots + audio sonification demo
    └── csl_tone_simulation_colab.ipynb   # Browser-run notebook
```

## Local quick start

```bash
git clone https://github.com/xlphon/csl-tone-simulation.git
cd csl-tone-simulation/csl-tone-simulation-v2
pip install -r requirements.txt

python csl_tone_model.py
python productive_sandhi_no_citation.py
python productive_sandhi_with_citation.py
python semantic_model.py
```

## Core model: `csl_tone_model.py`

The base model extends the Kachergis et al. associative learner with structured auditory forms:

```text
(seg1, tone1, seg2, tone2)
```

It contains two association matrices:

- **Item-level matrix `M`**: supports word memory.
- **Type-level matrix `M_type`**: supports classification of tone patterns by construction type.

Tone similarity is computed over Chao-tone pitch vectors using a Gaussian kernel controlled by `τ`:

- Low `τ`: tone categories are perceptually diffuse; similar tone patterns spread activation broadly.
- High `τ`: tone categories are more sharply encoded.

In this architecture, `τ` mainly affects classificatory generalisation, whereas decision noise `σ` mainly affects item-level word memory.

## Productive-sandhi models

### `productive_sandhi_no_citation.py`

This script asks whether the base model can produce a sandhi output from citation tones and construction type **without** ever receiving citation-to-sandhi evidence.

The answer should be interpreted conservatively: without citation exposure, the model can classify observed surface patterns, but it cannot derive productive citation-to-sandhi mappings. This script is therefore an exemplar-selection baseline rather than a full production model.

### `productive_sandhi_with_citation.py`

This model adds citation-to-sandhi mapping tables. On exposure trials, it records correspondences between citation tones and observed sandhi tones. Both recording and retrieval pass through `τ`-controlled perceptual encoding.

Interpretive note: the current citation-augmented implementation records citation-to-sandhi correspondences by looking up lexical citation tones. This is useful for testing what becomes possible when citation information is available, but it is more idealised than a strictly noun-only exposure interpretation. For that stricter diagnostic, use the `--strict-noun-only` option in `render_production_demo.py`.

## Production demo: tone, segment, F0, and audio outputs

Run:

```bash
python render_production_demo.py --tau 7.0 --noise 11.0 --seed 4 --n-examples 8
```

Stricter noun-only citation version:

```bash
python render_production_demo.py --strict-noun-only --tau 7.0 --noise 11.0 --seed 4 --n-examples 8
```

The demo creates:

```text
demo_outputs/
├── production_outputs.csv
├── segment_memory_outputs.csv
├── f0_plots/
└── audio/
```

### `production_outputs.csv`

This file evaluates the productive tone-sandhi probe:

```text
given lexical segments + citation tones + construction type -> predicted sandhi tones
```

Columns include:

- `construction`: `VN` or `NC`.
- `word1`, `word2`: lexical items in the test form.
- `target_segments`, `predicted_segments`: segmental frame.
- `citation`: citation-tone input.
- `target_tones`, `predicted_tones`: correct and model-selected sandhi tone patterns.
- `correct_sigma1`, `correct_sigma2`, `correct_tones_both`: tone accuracy.
- `correct_segments_both`: always `True` in this probe because segments are clamped by the query.
- `correct_whole_form`: both segments and tones correct.
- `mapping_available_sigma1`, `mapping_available_sigma2`: whether a relevant citation-to-sandhi table exists.

### `segment_memory_outputs.csv`

This is the meaningful segment-level diagnostic. It asks the item-level association matrix `M`:

```text
given visual referent -> which trained auditory form is most strongly associated?
```

This makes it possible to separate:

- segmental identity accuracy;
- tonal accuracy;
- whole-form accuracy.

This distinction matters because the productive tone-sandhi probe clamps segmental identity, while the word-memory probe tests whether the model learned form–referent mappings at the segmental and tonal levels.

### Audio interpretation

The `.wav` files are **not natural speech synthesis** and are **not TTS**. They are simple sonifications of model-selected F0 contours:

1. The model predicts a discrete tone pair, such as `44-13`.
2. Each Chao tone digit is mapped to an F0 value.
3. A linear F0 contour is generated for each syllable.
4. The F0 contour is rendered with a simple harmonic carrier.

Suggested thesis wording:

> The productive model returns discrete tonal outputs rather than acoustic waveforms. To make these outputs interpretable, I created an auditory rendering in which the model-selected Chao-tone contours were imposed on a simple harmonic carrier. The resulting files should therefore be interpreted as sonifications of the model's tonal choices, not as natural speech generated by the model.

## Distributed semantic diagnostic: `semantic_model.py`

This script replaces atomic visual labels with sparse distributed semantic vectors inspired by EARSHOT-style representations. It checks whether the word-memory/generalisation dissociation survives when referents are encoded as feature vectors rather than as indivisible labels.

The implementation uses:

- structural dimensions shared by VN or NC referents;
- noun-identity dimensions shared across VN and NC items involving the same noun;
- item-specific random dimensions.

## Recommended citation statement

If using this repository in the thesis or related materials, describe the demo as:

> an auditory sonification of model-selected tone patterns

rather than:

> speech generated by the model

This avoids overclaiming: the model generates tonal categories, and the demo renders those categories as audible F0 contours.

## Key references

- Bybee, J. (2001). *Phonology and Language Use*. Cambridge University Press.
- Kachergis, G., Yu, C., & Shiffrin, R. M. (2012). An associative model of adaptive inference for learning word–referent mappings. *Psychonomic Bulletin & Review, 19*(2), 317–324.
- Magnuson, J. S., You, H., Luthra, S., Li, M., Nam, H., Escabí, M., ... & Rueckl, J. G. (2020). EARSHOT: A minimal neural network model of incremental human speech recognition. *Cognitive Science, 44*(4), e12823.
- Nosofsky, R. M. (1986). Attention, similarity, and the identification–categorization relationship. *Journal of Experimental Psychology: General, 115*(1), 39–57.
- Thiessen, E. D., Kronstein, A. T., & Hufnagle, D. G. (2013). The extraction and integration framework. *Psychological Bulletin, 139*(4), 792–814.

## License

MIT License.
