# Gate-4 result: **GATE4_STOP**

Primary certificate: 90% one-sided upper bound for whole-graph risk `1 - directed F1`; bad graph threshold `risk > 0.30`.

## FINAL macro metrics

| method                   |   exceedance_rate |   exceedance_upper_95 |   absolute_calibration_error |   macro_aurc |   macro_safe_coverage |   macro_false_safe_rate |   bad_auroc |
|:-------------------------|------------------:|----------------------:|-----------------------------:|-------------:|----------------------:|------------------------:|------------:|
| constant                 |          0.127778 |              0.1752   |                    0.0277778 |     0.305877 |             0         |                0        |    0.5      |
| entropy                  |          0.183333 |              0.236104 |                    0.0833333 |     0.255224 |             0.05      |                0.111111 |    0.802169 |
| ood                      |          0.311111 |              0.370748 |                    0.211111  |     0.321306 |             0.0333333 |                0.233333 |    0.533098 |
| self_compatibility       |          0.183333 |              0.236104 |                    0.0833333 |     0.253238 |             0         |                0        |    0.802799 |
| lovo                     |          0.45     |              0.511379 |                    0.35      |     0.300474 |             0.411111  |                0.480556 |    0.531837 |
| ordinary_risk_regression |          0.283333 |              0.341975 |                    0.183333  |     0.262816 |             0.244444  |                0.125631 |    0.807086 |
| threshold_aware          |          0.122222 |              0.168994 |                    0.0222222 |     0.248523 |             0         |                0        |    0.834825 |
| proposed_blpc            |          0.127778 |              0.1752   |                    0.0277778 |     0.250848 |             0         |                0        |    0.833186 |

## FINAL macro metrics (95% secondary certificate)

| method                   |   exceedance_rate |   exceedance_upper_95 |   absolute_calibration_error |   macro_aurc |   macro_safe_coverage |   macro_false_safe_rate |   bad_auroc |
|:-------------------------|------------------:|----------------------:|-----------------------------:|-------------:|----------------------:|------------------------:|------------:|
| constant                 |          0.116667 |              0.162763 |                    0.0666667 |     0.305877 |             0         |               0         |    0.5      |
| entropy                  |          0.144444 |              0.193678 |                    0.0944444 |     0.255224 |             0.0222222 |               0.166667  |    0.802169 |
| ood                      |          0.288889 |              0.347749 |                    0.238889  |     0.321306 |             0.0222222 |               0.0416667 |    0.533098 |
| self_compatibility       |          0.105556 |              0.150218 |                    0.0555556 |     0.253238 |             0         |               0         |    0.802799 |
| lovo                     |          0.327778 |              0.387899 |                    0.277778  |     0.300474 |             0.3       |               0.239855  |    0.531837 |
| ordinary_risk_regression |          0.222222 |              0.277754 |                    0.172222  |     0.262816 |             0.2       |               0.10039   |    0.807086 |
| threshold_aware          |          0.111111 |              0.156505 |                    0.0611111 |     0.248523 |             0         |               0         |    0.834825 |
| proposed_blpc            |          0.111111 |              0.156505 |                    0.0611111 |     0.250848 |             0         |               0         |    0.833186 |

## Gate checks

Reliability: `{"pooled_exceedance": false, "macro_exceedance": false, "pooled_upper_95": false, "five_domains_exceedance": true, "no_domain_over_25": false, "false_safe": true}`
Increment: `{"aurc_relative_improvement": -0.009354661317452322, "safe_coverage_gain": 0.0, "bad_auroc_gain": -0.0016391375614677184, "exceedance_loss_vs_threshold": 0.005555555555555536, "beats_dev_selected_baseline_aurc": false}`

Risk–coverage points for pooled and every FINAL domain are in `final_risk_coverage.csv`; the pooled 90% plot is `risk_coverage_curve.png`.
Final truth was opened only after protocol, feature, certificate, prediction, and pre-truth hash receipts existed.
LOVO abstentions remain explicit in `lovo_pair_coverage`; technical failures remain in the denominator.
