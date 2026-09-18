# Gate-3C final topic decision

This directory is an independent final decision gate. It does not modify or reuse
the caches/results of `gate0_cdfm_defer/`, `gate1_cdfm_defer/`,
`gate2_reliable_routing/`, `gate3a_risk_predictability/`, or
`gate3a1_invariant_risk/`.

The pipeline is staged so that final-test truth is opened only after the frozen
protocol, DEV model-selection receipt, final-model receipt, and pre-truth hashes
have been written:

```powershell
python run_experiment.py --stage prepare
python -m pytest -q test_gate3c.py
python run_experiment.py --stage infer
python run_experiment.py --stage analyze
```

`--stage all` runs the same sequence. The final artifacts are under `results/`.
The two new generator mechanisms are implemented locally in `run_experiment.py`;
the historical Gate-2 generator file is imported but not edited.
