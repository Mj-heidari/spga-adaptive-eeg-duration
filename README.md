# Adaptive observation length for pediatric resting-state EEG

A stable-prefix stopping policy that decides, per subject, how much EEG to
record. A frozen graph-attention encoder emits a calibrated probability after
every second four-second epoch, and acquisition halts at the first checkpoint
where the predicted class, the confidence and the inter-checkpoint drift are
all stable. Easy cases stop early; ambiguous ones continue to the full budget.

Requiring two consecutive stable checkpoints is what separates the policy from
a plain confidence threshold: a single excursion above the threshold on a noisy
prefix does not terminate acquisition.

## Design

Evaluation uses subject-disjoint five-fold validation with four mutually
exclusive roles inside every training fold — encoder fitting, temperature
calibration, stopping-threshold selection, and test — so the threshold is never
chosen on the data it is scored on.

Two endpoints are run. The primary one is a developmental age band, chosen
because it is decodable from resting EEG and therefore gives the accuracy axis
of the trade-off a non-degenerate scale. A questionnaire-derived attention
score is carried as a declared negative control: a stopping policy must not
appear to save time on a label that carries no signal, and a single-endpoint
study cannot distinguish the two cases.

## Findings

18 of 129 electrodes, 48 s per subject.

**Primary endpoint — age band, 318 subjects:**

| Policy | BA [95 % CI] | AUC | mean s | early % |
|---|---|---|---|---|
| Full | 0.715 [0.666, 0.764] | 0.787 | 48.0 | 0.0 |
| Stable prefix | 0.712 [0.662, 0.760] | 0.778 | 35.9 | 43.1 |
| Confidence only | 0.709 [0.659, 0.755] | 0.763 | 27.5 | 56.0 |
| First checkpoint | 0.700 [0.651, 0.748] | 0.757 | 8.0 | 100.0 |
| Spectral, full budget | 0.674 [0.621, 0.724] | 0.768 | 48.0 | 0.0 |

The policy reduces mean acquisition by 25 % for a paired change of −0.003
balanced accuracy, interval [−0.011, 0.000].

**Negative control — 312 subjects:** every policy sits at chance (balanced
accuracy 0.484–0.522, all intervals covering 0.5). The policy stops early for
0.6 % of subjects here against 43.1 % on the primary endpoint. The mechanism is
visible in the calibration: the fitted temperature reached the upper bound of
its search in three of five folds, against 0.98–1.46 on the primary endpoint.
With no signal the calibrator flattens the probabilities, the confidence
condition never fires, and acquisition runs to the full budget. **The duration
saving is conditional on signal rather than a property of the rule.**

A fixed eight-second protocol nonetheless reaches 0.700, only 0.015 below the
full budget at one sixth of the recording time. Most of the decodable signal is
present in the first two epochs, and this is reported as the principal
limitation of the approach.

## Labels

The distribution used here provides age, sex, handedness and four bifactor
questionnaire scores, and contains no clinical diagnosis. No diagnostic label
is constructed. `--target` selects among what is available: `age`, `attention`
or `sex`.

## Montage

The vertex electrode is the recording reference in this net and is identically
zero in every window of every subject. Carried as a graph node it contributes a
permanently invalid node for the whole cohort, which per-node validity masking
absorbs silently — no error and no warning. It is excluded, leaving 18
positions. Subjects with zero-amplitude segments in the retained interval are
excluded by an amplitude audit before any model is fitted; a further ten
subjects have one or two electrodes flat throughout, which is ordinary
electrode failure and is what the validity mask exists to handle.

## Requirements

Python 3.11 or newer with NumPy, SciPy, scikit-learn and MNE.

```bash
pip install -e ".[io]"
python -m unittest discover -s tests -v
```

## Usage

Set `root` in `configs/hbn.json` to a local copy of the dataset, then:

```bash
python build_hbn_cohort.py --config configs/hbn.json --target age --age-cut 10.0
# review local/manifest.proposed.csv and local/labels.proposed.csv, then copy both into place
python -m eegstudy.cli prepare --config configs/hbn.json
python check_flat.py --config configs/hbn.json
python -m eegstudy.cli run --config configs/hbn.json --out runs/hbn_age
```

For the negative control, rebuild with `--target attention` and copy both the
manifest and the labels, since that cohort is smaller. The analysis is light:
roughly 3,900 windows and seconds of compute.

## Data availability

Recordings come from the Healthy Brain Network EEG distribution (Release 4,
OpenNeuro accession ds005508) and are not redistributed here. Generated caches,
runs, checkpoints and predictions are excluded from version control.
