# SECOM baseline comparison — development only

1,253 development records (83 failures, 1,170 passes), using the existing saved five-fold assignments. All preprocessing was fitted on each fold's training records. Failure is the positive class. The fixed decision threshold is 0.5. No reserved test records were parsed or evaluated. The loader streams past excluded raw lines without tokenizing them. No search, threshold tuning, calibration, or final predictive model refit was performed.

## Fixed models

- No-skill reference: assigns each record its training fold's failure prevalence; predicts pass at 0.5.
- Logistic regression: C=1, L2 regularization under the installed defaults, lbfgs solver, balanced class weights, maximum 2,000 iterations. Numeric measurements standardized using training data; missingness flags unscaled. Regularization constrains coefficients to limit overfitting; coefficients are not physical causes.
- Shallow decision tree: maximum depth 3, minimum 20 training records per leaf, balanced class weights, seed 42. No scaling. Depth and leaf constraints limit complexity but do not eliminate instability.

Each model was evaluated with and without varying training missingness flags. Numeric inputs and train-only medians are identical within each paired comparison. No resampling occurred. Class weights use training outcomes only. Weighted scores are not established as calibrated failure probabilities.

## Aggregate validation results

AP is the unweighted mean ± sample standard deviation across the five validation folds. Other metrics below use summed out-of-fold confusion counts; each development record contributes once per configuration. AP measures ranking quality; threshold metrics measure the fixed 0.5 decision rule. SD describes fold variation, not a confidence interval: folds share training records. No independent-test or population significance claim is made. Precision is undefined when no records are flagged, not artificially reported as zero.

| Configuration | Mean AP ± fold SD | Recall | Precision | Specificity | Balanced accuracy | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reference without indicators | 0.0662 ± 0.0020 | 0.00% | undefined | 100.00% | 50.00% | 1170 | 0 | 83 | 0 |
| reference + indicators | 0.0662 ± 0.0020 | 0.00% | undefined | 100.00% | 50.00% | 1170 | 0 | 83 | 0 |
| logistic without indicators | 0.1294 ± 0.0264 | 21.69% | 12.68% | 89.40% | 55.54% | 1046 | 124 | 65 | 18 |
| logistic + indicators | 0.1281 ± 0.0274 | 22.89% | 13.97% | 90.00% | 56.45% | 1053 | 117 | 64 | 19 |
| tree without indicators | 0.1072 ± 0.0206 | 57.83% | 11.79% | 69.32% | 63.57% | 811 | 359 | 35 | 48 |
| tree + indicators | 0.1071 ± 0.0205 | 57.83% | 11.79% | 69.32% | 63.57% | 811 | 359 | 35 | 48 |

## Individual validation folds

Each fold contains 16 or 17 failures, so recall is sensitive to individual cases. TN: passing records correctly passed. FP: passing records flagged. FN: failures missed. TP: failures detected.

| Configuration | Fold | AP | Recall | Precision | Specificity | Balanced accuracy | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reference without indicators | 1 | 0.0677 | 0.00% | undefined | 100.00% | 50.00% | 234 | 0 | 17 | 0 |
| reference + indicators | 1 | 0.0677 | 0.00% | undefined | 100.00% | 50.00% | 234 | 0 | 17 | 0 |
| logistic without indicators | 1 | 0.1182 | 17.65% | 10.00% | 88.46% | 53.05% | 207 | 27 | 14 | 3 |
| logistic + indicators | 1 | 0.1298 | 23.53% | 15.38% | 90.60% | 57.06% | 212 | 22 | 13 | 4 |
| tree without indicators | 1 | 0.1056 | 58.82% | 13.89% | 73.50% | 66.16% | 172 | 62 | 7 | 10 |
| tree + indicators | 1 | 0.1056 | 58.82% | 13.89% | 73.50% | 66.16% | 172 | 62 | 7 | 10 |
| reference without indicators | 2 | 0.0677 | 0.00% | undefined | 100.00% | 50.00% | 234 | 0 | 17 | 0 |
| reference + indicators | 2 | 0.0677 | 0.00% | undefined | 100.00% | 50.00% | 234 | 0 | 17 | 0 |
| logistic without indicators | 2 | 0.1491 | 17.65% | 11.11% | 89.74% | 53.70% | 210 | 24 | 14 | 3 |
| logistic + indicators | 2 | 0.1517 | 29.41% | 18.52% | 90.60% | 60.01% | 212 | 22 | 12 | 5 |
| tree without indicators | 2 | 0.1023 | 58.82% | 11.63% | 67.52% | 63.17% | 158 | 76 | 7 | 10 |
| tree + indicators | 2 | 0.1023 | 58.82% | 11.63% | 67.52% | 63.17% | 158 | 76 | 7 | 10 |
| reference without indicators | 3 | 0.0677 | 0.00% | undefined | 100.00% | 50.00% | 234 | 0 | 17 | 0 |
| reference + indicators | 3 | 0.0677 | 0.00% | undefined | 100.00% | 50.00% | 234 | 0 | 17 | 0 |
| logistic without indicators | 3 | 0.1280 | 23.53% | 14.29% | 89.74% | 56.64% | 210 | 24 | 13 | 4 |
| logistic + indicators | 3 | 0.1056 | 17.65% | 9.38% | 87.61% | 52.63% | 205 | 29 | 14 | 3 |
| tree without indicators | 3 | 0.0940 | 47.06% | 12.90% | 76.92% | 61.99% | 180 | 54 | 9 | 8 |
| tree + indicators | 3 | 0.0940 | 47.06% | 12.90% | 76.92% | 61.99% | 180 | 54 | 9 | 8 |
| reference without indicators | 4 | 0.0640 | 0.00% | undefined | 100.00% | 50.00% | 234 | 0 | 16 | 0 |
| reference + indicators | 4 | 0.0640 | 0.00% | undefined | 100.00% | 50.00% | 234 | 0 | 16 | 0 |
| logistic without indicators | 4 | 0.1594 | 31.25% | 16.13% | 88.89% | 60.07% | 208 | 26 | 11 | 5 |
| logistic + indicators | 4 | 0.1580 | 25.00% | 15.38% | 90.60% | 57.80% | 212 | 22 | 12 | 4 |
| tree without indicators | 4 | 0.0914 | 50.00% | 8.08% | 61.11% | 55.56% | 143 | 91 | 8 | 8 |
| tree + indicators | 4 | 0.0914 | 50.00% | 8.08% | 61.11% | 55.56% | 143 | 91 | 8 | 8 |
| reference without indicators | 5 | 0.0640 | 0.00% | undefined | 100.00% | 50.00% | 234 | 0 | 16 | 0 |
| reference + indicators | 5 | 0.0640 | 0.00% | undefined | 100.00% | 50.00% | 234 | 0 | 16 | 0 |
| logistic without indicators | 5 | 0.0923 | 18.75% | 11.54% | 90.17% | 54.46% | 211 | 23 | 13 | 3 |
| logistic + indicators | 5 | 0.0957 | 18.75% | 12.00% | 90.60% | 54.67% | 212 | 22 | 13 | 3 |
| tree without indicators | 5 | 0.1424 | 75.00% | 13.64% | 67.52% | 71.26% | 158 | 76 | 4 | 12 |
| tree + indicators | 5 | 0.1423 | 75.00% | 13.64% | 67.52% | 71.26% | 158 | 76 | 4 | 12 |

## Interpretation boundaries

Missingness contributions are model-dependent paired comparisons, not proof of physical mechanisms or generalization beyond this dataset. Primary ranking comparisons use mean within-fold AP, not cross-fold pooled score rankings. Reported performance is exploratory development validation; subsequent model choices would make it selection evidence rather than an unbiased final estimate. Unknown entity dependence and measurement acquisition timing remain limitations. This run stops at baseline comparison.

## Findings

Logistic regression had the highest observed mean average precision (0.1294 without indicators), versus 0.0662 for the reference and about 0.1072 for the shallow tree. This suggests some ranking signal in these development folds, not reliable operational screening or statistically established superiority.

Indicators did not demonstrate a clear benefit on the primary metric. Logistic mean AP decreased by 0.00125, improving in three folds and declining in two. At threshold 0.5, indicators increased detected failures from 18 to 19 and reduced false alarms from 124 to 117. These small decision-rule improvements coexist with slightly lower ranking performance. Tree mean AP changed by only -0.000019, with identical confusion counts. No fitted tree selected an indicator for a decision split; the tiny ranking change should not be attributed to missingness signal. Adding columns can alter tie resolution between competing numeric splits.

The tree detected 48 of 83 failures but flagged 359 passing records. Logistic regression with indicators detected 19 failures and flagged 117 passing records. The tree creates many false alarms; logistic regression misses most failures at the fixed threshold. Neither is ready for operational use on this evidence. Scores from class-weighted models are not established as calibrated probabilities.

All 30 planned fits finished without warnings; logistic regression converged within 117 iterations. Independent recounting of saved validation predictions matched all six confusion matrices. No model-setting changes or exploratory reruns occurred after observing results. The initial dependency access failure happened before fitting; a fresh installation of the same pinned versions resolved it.

## Single next recommended step

Agree on an acceptable false-alarm versus missed-failure tradeoff for a clearly labeled portfolio screening scenario, before any development-only threshold assessment. This is an engineering decision, not a value to invent from test outcomes. Preserve indicators and keep the reserved test set untouched.
