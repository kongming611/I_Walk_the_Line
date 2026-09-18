TOPIC DECISION: **TOPIC_GO**
METHOD DECISION: **METHOD_BORDERLINE**

Selected proposed model: `M3_RandomForest`. DEV selection status: `PASS`.

## Final domain results

The primary evidence is the six-domain macro, not a pooled metric. `risk@50` retains the 50% of tasks with the lowest predicted risk.

| FINAL domain | selected-model rho | rho 95% CI | risk@50 | full risk | relative reduction@50 |
|---|---:|---:|---:|---:|---:|
| piecewise_exponential | 0.508 | [-0.043, 0.872] | 0.203 | 0.244 | 16.8% |
| piecewise_gaussian | 0.620 | [0.217, 0.846] | 0.318 | 0.340 | 6.4% |
| piecewise_student_t3 | 0.380 | [-0.143, 0.858] | 0.091 | 0.141 | 35.0% |
| quadratic_exponential | 0.543 | [0.098, 0.837] | 0.271 | 0.295 | 8.4% |
| quadratic_gaussian | 0.747 | [0.344, 0.947] | 0.236 | 0.306 | 22.8% |
| quadratic_student_t3 | 0.735 | [0.389, 0.906] | 0.207 | 0.302 | 31.5% |

**FINAL macro:** rho=0.589, 95% CI=[0.398, 0.724], relative risk reduction@50=20.1%.
**FINAL pooled:** rho=0.676, risk@50=0.204, full risk=0.271, relative reduction@50=24.7%.

## DEV model selection

| model | DEV macro Spearman | domains rho>0.20 | domains rho<0 |
|---|---:|---:|---:|
| M1_Ridge | 0.373 | 5/8 | 2/8 |
| M2_AdditiveSpline | 0.429 | 6/8 | 2/8 |
| M3_RandomForest | 0.470 | 6/8 | 1/8 |
| M4_HistGradientBoosting | 0.488 | 6/8 | 1/8 |

Selection rule: highest macro Spearman among candidates with at least 6/8 DEV domains above 0.20 and at most 1/8 below zero; ties within 0.02 use the simpler model. Final selected model: `M3_RandomForest`.

## Baselines and method increment

Final threshold-aware Ridge macro rho=0.618; selected minus threshold-aware increment=-0.029 rho.
Final threshold-aware Ridge macro relative reduction@50=20.6%; selected increment=-0.5 percentage points.

| model | final macro rho | final macro relative reduction@50 |
|---|---:|---:|
| B0_constant | 0.000 | 0.0% |
| B1_metadata_ridge | 0.112 | 0.7% |
| B2_ood_only | 0.027 | 0.9% |
| B3_confidence_ridge | 0.516 | 15.6% |
| B4_threshold_aware_ridge | 0.618 | 20.6% |
| B5_proposed_ridge | 0.604 | 21.7% |
| B6_selected_proposed | 0.589 | 20.1% |
| M2_AdditiveSpline | 0.514 | 19.2% |
| M3_RandomForest | 0.589 | 20.1% |
| M4_HistGradientBoosting | 0.599 | 21.5% |

## Does model capacity matter?

DEV macro Spearman comparison is in `model_capacity_dev_comparison.png`. The preregistered capacity interpretation is: **model capacity matters materially**.
Best complex DEV model: `M4_HistGradientBoosting`, gain over Ridge=0.115; best complex FINAL diagnostic model: `M4_HistGradientBoosting`, gain over Ridge=-0.004.
Non-selected models are diagnostic only; the FINAL model was selected before FINAL TEST truth.

## Historical boundary

- Gate-3A primary all-feature Ridge: mean Spearman approximately 0.200, BORDERLINE.
- Gate-3A post-hoc stable 5-feature Ridge: old four-mechanism exploratory mean rho approximately 0.55; it is not independent evidence.
- Gate-3A.1 independent confirmation primary stable_ridge_v1: softsign rho approximately 0.665, sine rho approximately 0.231, mean rho approximately 0.448, mean risk reduction@50 approximately 21.6%; sine missed preregistered 0.30, so BORDERLINE.
- Gate-3A.1 pre-frozen threshold-aware Ridge: softsign rho approximately 0.522, sine rho approximately 0.537; adaptive-threshold distance had promise, but this was not a GO conversion.

This round does not reinterpret those results as GO. Quadratic and piecewise are reported as unseen mechanism families for the risk predictor; no claim is made about what CDFM pretraining did or did not contain.

## Answers to the final topic questions

1. Cross unseen mechanism: yes; the final test mechanisms are quadratic and piecewise.
2. Cross unseen graph instances: yes by design; all 360 split tasks use unique graph seeds, with final seeds 600000 onward and no overlap with prior Gate seeds.
3. With unseen exponential noise: quadratic_exponential rho=0.543, piecewise_exponential rho=0.508; this is supported by the joint-shift domains.
4. OOD score alone: final macro rho=0.027; it is not sufficient as the main explanation under this gate.
5. Threshold-aware information: final macro rho=0.618; it is a frozen baseline and is stable across final domains.
6. Probability/resampling stability increment: selected minus threshold-aware rho=-0.029, reduction increment=-0.5 percentage points; method decision is **METHOD_BORDERLINE**.
7. More complex models: model capacity matters materially.
8. Main bottleneck: FINAL evidence does not support model capacity as the main bottleneck—threshold-aware Ridge (macro rho=0.618) exceeds selected RF (0.589), and proposed Ridge (0.604) also exceeds selected RF; the new stability features add no measured increment. The remaining limitation is therefore the incremental representation/training-coverage problem under shift, with inherent unpredictability still not separable without a dedicated ablation.
9. Formal paper main line recommendation: yes, subject to method development.

## Freeze and truth-isolation audit

FINAL TEST truth was not used for feature construction, model selection, preprocessing, hyperparameter choice, or Gate threshold modification.

After FINAL TEST truth was opened, were any feature/model/hyperparameter/Gate modifications made? **No.**

The required pre-truth files were written before `load_final_truth_after_freeze()` ran: `frozen_protocol.json`, `model_selection_receipt.json`, `final_model_receipt.json`, and `pre_truth_hashes.json`.
