# RFD-Bench: Robot Failure Diagnosis

[![tests](https://github.com/Punyajain-cmd/rfd-bench/actions/workflows/tests.yml/badge.svg)](https://github.com/Punyajain-cmd/rfd-bench/actions/workflows/tests.yml)

RFD-Bench asks a model to read the record of a failed robot episode and say **why** it failed, **when** it first went wrong,
or that the evidence is **insufficient** to tell. Every episode has one injected cause, and counterfactual twins differ only in that cause.

- **Dataset (v0, public):** https://huggingface.co/datasets/punyajain/rfd-bench-v0 (CC BY 4.0, 216 public rows)
- **This repo:** the v0 scorers, validator, shortcut probes and baselines, plus the v1 physics-simulation code (MuJoCo).

> **Status: experimental pilot.** v0 is synthetic text from a scripted kinematic simulator, not real robot logs.
> It measures careful reading of a short structured trace, not real-world robot competence. Both versions are easy for a
> hand-written rule (see Limitations).

## What's here

| Path | What it is |
|---|---|
| `v0/data/` | The 216 public v0 rows (train 117 / val 43 / test_public 56), same rows as on Hugging Face |
| `v0/common.py` | Loader and scorer (`acc`, Omni-style score, `cause_and_step`) |
| `v0/validate.py` | Schema, pairing and leakage checks |
| `v0/heuristic.py` | Rule-based reference reader, random and always-abstain baselines |
| `v0/probes.py` | Shortcut probes (option-only, length, instruction-only, bag-of-words) |
| `v0/llm_raw.py`, `v0/llm_baseline.py`, `v0/rescore.py` | Local-LLM baseline runner and strict/lenient answer parsers |
| `v0/aggregate_results/`, `v0/results/` | Aggregate pilot scores (no per-row predictions) |
| `v0/heldout_proof/` | sha256 manifest proving the private test split exists |
| `v1/sim.py` | MuJoCo planar-push simulator with physical fault injection |
| `v1/evalall.py`, `v1/abstain_check.py`, `v1/tune.py` | v1 validation, probes, baselines, abstain check and tuning (reference code) |
| `v1/report_v1_pilot.txt` | Aggregate v1 pilot report |

## Quick start (v0)

```bash
git clone https://github.com/Punyajain-cmd/rfd-bench && cd rfd-bench
pip install -r requirements.txt
cd v0
python validate.py     # schema and leakage checks, prints ERRORS 0
python heuristic.py    # rule reader and trivial baselines on test_public
python probes.py       # shortcut probes
```

Or load the data from Hugging Face:

```python
from datasets import load_dataset
ds = load_dataset("punyajain/rfd-bench-v0")
```

The scripts report a `test_private` column with `n=0`: that split is withheld (see below).

## Task and scoring (v0)

Input: `episode_text` (instruction, explicit success contract, planner line, 16-step sensor/action table) and 5 options,
one of which is always `insufficient_evidence`. Output: an option letter and the earliest failing step, as `X, t=N`.

Labels: `perception_miss`, `grasp_slip`, `wrong_object`, `collision`, `joint_limit`, `instruction_misread`,
`sim_contact_artifact`, `timeout`, `insufficient_evidence`.

- **acc**: predicted label equals gold (a correct abstain counts as correct).
- **Omni**: +1 correct, -1 wrong non-abstain answer, 0 for abstaining on an answerable row. Always-abstain is therefore not 0.
- **cause_and_step**: on answerable rows, both label and step exact.

## Results

**v0, public test (n=56), reproduced by `heuristic.py` and `probes.py` in this repo:**

| Method | acc | Omni | cause+step |
|---|---|---|---|
| Rule reader (structured `steps`) | 0.911 | 0.821 | 0.896 |
| Random | 0.214 | -0.393 | 0.000 |
| Always abstain | 0.143 | 0.143 | 0.000 |

**v0, Qwen2.5 pilot (combined 128-row public+private test, from `aggregate_results/rescore_output.txt`):**

| Model | Evidence | Strict acc / Omni (unparsed) | Lenient acc / Omni |
|---|---|---|---|
| 0.5B | full | 0.000 / -1.000 (128/128) | 0.180 / -0.641 |
| 0.5B | blank log | 0.000 / -1.000 (128/128) | 0.156 / -0.648 |
| 1.5B | full | 0.078 / -0.695 (66/128) | 0.203 / -0.414 |
| 1.5B | blank log | 0.094 / -0.258 (39/128) | 0.117 / +0.062 |

These are tiny local models. Near-chance scores say nothing about whether larger models use the evidence. The lenient
parser maps unparseable replies to abstain, which can inflate Omni.

**v1 pilot (144 MuJoCo episodes, from `v1/report_v1_pilot.txt`):** telemetry-feature gradient boosting reaches 0.969
label accuracy, and a conservative rule reaches acc 0.922 / Omni 0.859. Option-only probes sit at chance (0.125, chance 1/7).

## v1: physics version (code only)

v1 replaces the scripted trace with a MuJoCo planar push: a force-limited paddle pushes a box to a goal. Causes are physical
interventions on a shared base scene (`low_friction`, `heavy_payload`, `weak_actuator`, `hidden_obstacle`, `encoder_bias`,
`no_failure`, `insufficient_evidence`), and 75% of episodes carry a sub-threshold distractor fault. Every episode is
re-simulated to check that the nominal scene succeeds, the faulted run fails, repairing only the primary fault succeeds,
and removing only the distractor still fails (144/144 passed).

The v1 dataset is **not released yet**, so `evalall.py` and `abstain_check.py` are published as reference code and
expect the dataset files. `sim.py` runs on its own (`pip install mujoco numpy`).

## What is withheld, and why

- **v0 private test (72 rows)** and **v1 private test (32 episodes)**, with their labels, telemetry and frames: held out
  so the benchmark can still be scored without contamination. Only a sha256 manifest is published.
- **The episode generators** (`gen.py` for v0 and v1): the generators are deterministic, so publishing them would let
  anyone regenerate the private test rows exactly. They will be released if the private split is retired.
- **Per-row model predictions** and `true_hidden_cause` on abstain rows (stripped, as on Hugging Face).

## Limitations

- v0 is too easy for explicit thresholds (`contact_N > 40`, `q_max_ratio == 1.0`); the rule reader gets about 0.91.
- v0 has an abstention length leak (a length-only model separates abstain rows at about 0.91 vs 0.875 majority) and a
  dropout confound (early dropout correlates with abstain).
- v1 heavy-vs-weak abstain twins are not perfectly indistinguishable (a leave-group-out LR recovers the hidden cause 72%,
  chance 50%). Encoder-bias onset is always t=0. Frames reveal `hidden_obstacle`.
- Single primary cause, one task type per version, small n (216 public v0 rows, 144 v1 episodes).
- Only tiny local LLMs have been scored; no capability claims are made.

## License

Code: MIT. v0 data: CC BY 4.0. © 2026 Punya Malaiya Jain.
