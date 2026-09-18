# Gate-3A: CDFM risk predictability

This directory is an independent, minimal feasibility gate. It does not modify
Gate-0/1/2 or CDFM. The deployment-time features use only observed data, the
CDFM output, DirectLiNGAM output, and optional CDFM bootstrap stability.

The preregistered primary candidate is Ridge regression using all available
ground-truth-free diagnostics. A fixed RandomForestRegressor is reported only
as a secondary sensitivity analysis. The main split is four-fold
leave-one-mechanism-family-out; random train/test splitting is not used.

Run from the repository root with the already working Python environment:

```powershell
python gate3a_risk_predictability/test_gate3a.py
python gate3a_risk_predictability/run_experiment.py --smoke --stage all --bootstrap 0
python gate3a_risk_predictability/run_experiment.py --stage all --n-seeds 15 --bootstrap 2
```

Inference is cached under `cache/raw_predictions/`. Re-running skips complete
cache entries unless `--rerun` is supplied. The full scientific outputs are
written under `results/`; smoke outputs are isolated under `results/smoke/` and
`cache/smoke/`.

