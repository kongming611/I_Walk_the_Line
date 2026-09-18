# Gate-3A.1 invariant-risk confirmation

This directory is an independent confirmation experiment. It does not modify or rerun Gate-0, Gate-1, Gate-2, or Gate-3A artifacts.

The primary model, secondary model, baselines, independent test families/seeds, evaluation metrics, and decision gate were frozen in `frozen_protocol.json` before any confirmatory true-risk result was evaluated.

Run the fast checks without loading CDFM:

```powershell
python gate3a1_invariant_risk/test_gate3a1.py
```

Run the experiment stages:

```powershell
python gate3a1_invariant_risk/run_experiment.py --stage prepare
python gate3a1_invariant_risk/run_experiment.py --stage infer
python gate3a1_invariant_risk/run_experiment.py --stage analyze
```

`prepare` audits the frozen protocol, verifies the old Gate-3A source hashes and 60 old caches, builds the post-hoc development audit from those caches, and generates 50 new datasets. `infer` runs exactly one original CDFM inference plus two bootstrap inferences for each new dataset (and DirectLiNGAM only for the old-candidate baseline), caching every result. `analyze` fits all preprocessing and models on the old 60 tasks only and writes the confirmatory outputs.
