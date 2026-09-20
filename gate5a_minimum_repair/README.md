# Gate-5A: ground-truth-free minimum-repair/refutation certification

This is an isolated experiment for the frozen question:

> Can held-out causal contradictions produce a non-vacuous ground-truth-free
> lower bound on CDFM graph error?

The experiment uses six synthetic domains with 20 tasks per domain, `D=10`
and `N=1000`. Every task is split into two fixed halves:

```text
X_discovery  -> CDFM raw graph -> deterministic DAG projection G_hat
X_validation -> local-Markov CI tests -> Holm-controlled refutation witnesses
```

The true DAG is opened only after the certificate predictions and pre-truth
hashes have been written. `RR_lower_bound` is the smallest searched SHD radius
at which a DAG can avoid all rejected witness statements. The search is exact
for `k=0,1,2,3`; if no repair is found by `k=3`, the result is recorded as
`RR_lower_bound=4` with `rr_exact=false`, meaning only `RR >= 4` is claimed.

The CDFM output is not assumed to be acyclic. The candidate graph is defined
before validation and truth are used: retain CDFM-selected orientations,
resolve reciprocal pairs by the larger discovery-only edge probability, then
add edges in descending probability order while rejecting cycle-forming edges.

## Run stages

```powershell
python gate5a_minimum_repair/run_experiment.py --stage prepare
python gate5a_minimum_repair/run_experiment.py --stage infer
python gate5a_minimum_repair/run_experiment.py --stage build-witnesses
python gate5a_minimum_repair/run_experiment.py --stage build-certificates
python gate5a_minimum_repair/run_experiment.py --stage open-final-truth
python gate5a_minimum_repair/run_experiment.py --stage report
python gate5a_minimum_repair/run_experiment.py --stage verify
```

`--stage all` is available after a successful smoke test. `infer` also accepts
`--shard-index` and `--shard-count` for resumable local execution.

The experiment does not modify Gate-0 through Gate-4 directories or their
caches.
