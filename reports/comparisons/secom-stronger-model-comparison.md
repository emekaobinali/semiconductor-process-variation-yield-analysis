# SECOM: fixed stronger-model comparison

Scope: development data only, existing five folds, 1,253 records including 83 failures. Threshold is fixed at 0.50 in all configurations. Thirty new fits were performed (three models x two indicator variants x five folds). Existing reference, logistic C=1, and shallow-tree results are reused, not refitted. No search, threshold optimization, early stopping, advanced feature selection, or full-development predictive refit. The reserved test set was not parsed or evaluated.

## Settings and rationale

- Logistic C=0.1: tenfold stronger L2 coefficient penalty than the existing C=1 logistic baseline; lbfgs, balanced class weights, max_iter=2000. Standardize numeric inputs only. A linear, directly inspectable baseline with stronger protection against fitting noise.
- Small forest: 100 trees, max_depth=5, min_samples_leaf=10, max_features=sqrt, balanced class weights, seed 42. Aggregates constrained nonlinear trees. Less directly interpretable than a single tree; explanations would require separate checks.
- Shallow boosting: histogram gradient boosting, 100 rounds, learning_rate=0.1, max_depth=3, max_leaf_nodes=7, min_samples_leaf=20, l2_regularization=1, balanced class weights, no early stopping, seed 42. Justified as a second constrained nonlinear approach whose sequential corrections differ from the forest's averaging; explainable but not intrinsically transparent.

Each fit uses the approved training-only constant filtering, median imputation, and optional missingness flags. Numeric scaling applies only to logistic regression. Labels inform class weights only within training. No class resampling. Timestamps and row IDs are not predictive inputs. The loader skips non-development lines before parsing; reserved manifest access is blocked. Retained flags are binary and unscaled.

## Validation comparison

AP is the unweighted mean ± sample standard deviation across validation folds. Remaining rates and counts pool out-of-fold decisions, so each record contributes once per configuration. SD is variability, not a confidence interval. FP means passing records flagged; FN means failures missed. TP means failures detected; TN means passes correctly unflagged. Undefined precision means nothing was flagged.

| Model | Mean AP ± SD | Recall | Precision | False alarms | Missed failures | Specificity | Balanced accuracy | TP | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reference without flags | 0.0662 ± 0.0020 | 0.0% | undefined | 0 | 83 | 100.0% | 50.0% | 0 | 1170 |
| reference + flags | 0.0662 ± 0.0020 | 0.0% | undefined | 0 | 83 | 100.0% | 50.0% | 0 | 1170 |
| logistic without flags | 0.1294 ± 0.0264 | 21.7% | 12.7% | 124 | 65 | 89.4% | 55.5% | 18 | 1046 |
| logistic + flags | 0.1281 ± 0.0274 | 22.9% | 14.0% | 117 | 64 | 90.0% | 56.4% | 19 | 1053 |
| tree without flags | 0.1072 ± 0.0206 | 57.8% | 11.8% | 359 | 35 | 69.3% | 63.6% | 48 | 811 |
| tree + flags | 0.1071 ± 0.0205 | 57.8% | 11.8% | 359 | 35 | 69.3% | 63.6% | 48 | 811 |
| logistic_C0.1 without flags | 0.1290 ± 0.0209 | 25.3% | 12.3% | 150 | 62 | 87.2% | 56.2% | 21 | 1020 |
| logistic_C0.1 + flags | 0.1257 ± 0.0231 | 24.1% | 12.9% | 135 | 63 | 88.5% | 56.3% | 20 | 1035 |
| small_forest without flags | 0.1987 ± 0.0602 | 19.3% | 17.6% | 75 | 67 | 93.6% | 56.4% | 16 | 1095 |
| small_forest + flags | 0.1919 ± 0.0391 | 27.7% | 25.3% | 68 | 60 | 94.2% | 60.9% | 23 | 1102 |
| shallow_boosting without flags | 0.2216 ± 0.0467 | 14.5% | 30.0% | 28 | 71 | 97.6% | 56.0% | 12 | 1142 |
| shallow_boosting + flags | 0.2216 ± 0.0467 | 14.5% | 30.0% | 28 | 71 | 97.6% | 56.0% | 12 | 1142 |

## New-model fold results

| Model | Fold | AP | Recall | Precision | FP | FN | Specificity | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_C0.1 without flags | 1 | 0.1246 | 17.6% | 7.9% | 35 | 14 | 85.0% | 51.3% |
| logistic_C0.1 + flags | 1 | 0.1320 | 23.5% | 11.4% | 31 | 13 | 86.8% | 55.1% |
| small_forest without flags | 1 | 0.2066 | 29.4% | 26.3% | 14 | 12 | 94.0% | 61.7% |
| small_forest + flags | 1 | 0.1794 | 35.3% | 27.3% | 16 | 11 | 93.2% | 64.2% |
| shallow_boosting without flags | 1 | 0.2086 | 17.6% | 23.1% | 10 | 14 | 95.7% | 56.7% |
| shallow_boosting + flags | 1 | 0.2086 | 17.6% | 23.1% | 10 | 14 | 95.7% | 56.7% |
| logistic_C0.1 without flags | 2 | 0.1535 | 35.3% | 19.4% | 25 | 11 | 89.3% | 62.3% |
| logistic_C0.1 + flags | 2 | 0.1563 | 29.4% | 18.5% | 22 | 12 | 90.6% | 60.0% |
| small_forest without flags | 2 | 0.1977 | 23.5% | 19.0% | 17 | 13 | 92.7% | 58.1% |
| small_forest + flags | 2 | 0.2513 | 41.2% | 33.3% | 14 | 10 | 94.0% | 67.6% |
| shallow_boosting without flags | 2 | 0.2723 | 17.6% | 60.0% | 2 | 14 | 99.1% | 58.4% |
| shallow_boosting + flags | 2 | 0.2723 | 17.6% | 60.0% | 2 | 14 | 99.1% | 58.4% |
| logistic_C0.1 without flags | 3 | 0.1259 | 17.6% | 9.4% | 29 | 14 | 87.6% | 52.6% |
| logistic_C0.1 + flags | 3 | 0.1118 | 17.6% | 10.3% | 26 | 14 | 88.9% | 53.3% |
| small_forest without flags | 3 | 0.1139 | 11.8% | 7.4% | 25 | 15 | 89.3% | 50.5% |
| small_forest + flags | 3 | 0.1497 | 17.6% | 14.3% | 18 | 14 | 92.3% | 55.0% |
| shallow_boosting without flags | 3 | 0.1605 | 5.9% | 14.3% | 6 | 16 | 97.4% | 51.7% |
| shallow_boosting + flags | 3 | 0.1605 | 5.9% | 14.3% | 6 | 16 | 97.4% | 51.7% |
| logistic_C0.1 without flags | 4 | 0.1428 | 37.5% | 15.8% | 32 | 10 | 86.3% | 61.9% |
| logistic_C0.1 + flags | 4 | 0.1328 | 31.2% | 15.2% | 28 | 11 | 88.0% | 59.6% |
| small_forest without flags | 4 | 0.2835 | 25.0% | 30.8% | 9 | 12 | 96.2% | 60.6% |
| small_forest + flags | 4 | 0.2073 | 31.2% | 35.7% | 9 | 11 | 96.2% | 63.7% |
| shallow_boosting without flags | 4 | 0.2646 | 18.8% | 42.9% | 4 | 13 | 98.3% | 58.5% |
| shallow_boosting + flags | 4 | 0.2646 | 18.8% | 42.9% | 4 | 13 | 98.3% | 58.5% |
| logistic_C0.1 without flags | 5 | 0.0985 | 18.8% | 9.4% | 29 | 13 | 87.6% | 53.2% |
| logistic_C0.1 + flags | 5 | 0.0955 | 18.8% | 9.7% | 28 | 13 | 88.0% | 53.4% |
| small_forest without flags | 5 | 0.1920 | 6.2% | 9.1% | 10 | 15 | 95.7% | 51.0% |
| small_forest + flags | 5 | 0.1717 | 12.5% | 15.4% | 11 | 14 | 95.3% | 53.9% |
| shallow_boosting without flags | 5 | 0.2019 | 12.5% | 25.0% | 6 | 14 | 97.4% | 55.0% |
| shallow_boosting + flags | 5 | 0.2019 | 12.5% | 25.0% | 6 | 14 | 97.4% | 55.0% |

## Boundaries

The new ensembles improved ranking but did not improve failure recall over the shallow tree at threshold 0.50. The tree detects 48 of 83 failures with 359 false alarms. The forest with indicators detects 23 with 68 false alarms: 25 fewer detections and 291 fewer false alarms. Boosting detects 12 with 28 false alarms: 36 fewer detections and 331 fewer false alarms. These are different operating tradeoffs, not a demonstrated replacement that catches more failures at the same or lower false-alarm burden.

Mean AP rose from about 0.107 for the tree to 0.192–0.199 for the forest and 0.222 for boosting. Stronger logistic regularization remained near the original logistic AP of 0.128–0.129 and did not establish a meaningful improvement. Fold SD is not a confidence interval, and this development evidence does not establish statistical or deployment superiority.

Indicators had mixed effects: the forest detected seven more failures and produced seven fewer false alarms with indicators at 0.50, despite slightly lower mean AP. Stronger logistic regression had lower mean AP with indicators. Boosting produced identical reported metrics with and without indicators. No indicator configuration is selected here.

All 30 planned fits completed without warnings. An independent recount of saved out-of-fold decisions verified every new confusion matrix. The reporting-check quoting error was repaired without refitting. Existing baseline fits were not repeated, and the reserved test set remained untouched. No final model or threshold decision has been made.

All results are development comparisons after previous development exploration, not independent final performance estimates. An operating false-alarm budget has not been agreed, so no numerical increase can be declared operationally reasonable without that decision. Scores from weighted models are not established as calibrated probabilities; identical thresholds do not impose identical review budgets. No causal interpretation or physical feature meanings are inferred. Entity dependence and measurement timing remain unresolved.

Verification: each configuration has exactly one saved validation prediction per development record; fold confusion counts sum to pooled counts; all scores are finite and raw development values are unchanged. Exact settings, predictions, and warnings are saved locally for audit.
