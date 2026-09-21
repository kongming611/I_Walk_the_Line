# Gate-5B: Independent-Repair Witnesses

This directory is an isolated follow-up to `gate5a_minimum_repair/`. Gate-5A
is read-only and is snapshotted before any Gate-5B artifact is written.

`RR_CI` is the paired Gate-5A local-Markov/RCIT baseline. `RR_ANM` is a
separate additive independent-noise mechanism certificate. They must not be
combined by taking their maximum.

## Run order

From the repository root:

```powershell
python gate5b_independent_repair_witnesses/run_experiment.py --stage prepare-dev
python gate5b_independent_repair_witnesses/run_experiment.py --stage run-dev
python gate5b_independent_repair_witnesses/run_experiment.py --stage dev-screen
```

`dev-screen` is a hard gate. If it does not print `GATE5B_DEV_GO`, stop and
retain the DEV decision. It does not launch FINAL automatically.

Only after a passing DEV screen:

```powershell
python gate5b_independent_repair_witnesses/run_experiment.py --stage freeze
python gate5b_independent_repair_witnesses/run_experiment.py --stage prepare-final
python gate5b_independent_repair_witnesses/run_experiment.py --stage infer
python gate5b_independent_repair_witnesses/run_experiment.py --stage build-ci
python gate5b_independent_repair_witnesses/run_experiment.py --stage build-anm
python gate5b_independent_repair_witnesses/run_experiment.py --stage build-certificates
python gate5b_independent_repair_witnesses/run_experiment.py --stage open-final-truth
python gate5b_independent_repair_witnesses/run_experiment.py --stage report
python gate5b_independent_repair_witnesses/run_experiment.py --stage verify
```

Inference can be sharded with `--shard-index` and `--shard-count`; all shards
must finish before certificate construction.

## Gate interpretation

The primary FINAL certificate is `RR_ANM`. GO requires validity at least 95%,
bad-graph nonzero coverage at least 40%, severe-graph `RR_ANM >= 2` at least
50%, and at least a 10 percentage-point severe-graph gain over `RR_CI`.
Technical failure above 10% is `INCONCLUSIVE`; validity failure is `STOP`;
validity with insufficient efficacy is `BORDERLINE`.
