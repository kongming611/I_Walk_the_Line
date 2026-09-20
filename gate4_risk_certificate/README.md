# Gate-4: ground-truth-free graph-risk certificate

This directory contains the isolated Gate-4 experiment for
`Ground-truth-free graph-risk certification for causal foundation models under
mechanism shift`.

The pipeline is staged so that full-graph and leave-one-variable CDFM outputs
are built first, TRAIN/CALIBRATION/DEV truth is used only for fitting and
calibration, and FINAL truth is opened only after the frozen protocol,
certificate predictions, and pre-truth hashes exist.

```text
python gate4_risk_certificate/run_experiment.py --stage prepare
python gate4_risk_certificate/run_experiment.py --stage infer-subsets
python gate4_risk_certificate/run_experiment.py --stage build-features
python gate4_risk_certificate/run_experiment.py --stage fit-freeze
python gate4_risk_certificate/run_experiment.py --stage open-final-truth
python gate4_risk_certificate/run_experiment.py --stage report
python gate4_risk_certificate/run_experiment.py --stage verify
```

`--stage all` is available for a resumed end-to-end run. Historical Gate
directories and caches are read-only inputs and are never rewritten.
