# Technical engineering summary

## Objective and dataset
This project assessed whether anonymous semiconductor process measurements could screen for failed quality tests, emphasizing process variation and the tradeoff between catching failures and generating review workload. Inspection confirmed 1,567 SECOM records and 590 measurement columns, resolving the source feature-count discrepancy directly. There were 104 failures (6.6%) and 1,463 passes. About 4.54% of measurement cells were missing; every record had missing values. Anonymous features have no supported physical names or units. Quality-test timestamps were preserved but excluded from primary modeling; a chronological sensitivity analysis was not completed.

## Data handling and validation
A fixed shuffled stratified split reserved 314 records (21 failures), leaving 1,253 development records (83 failures). Saved row manifests and five stratified fold assignments defined all comparisons. Training-only median imputation and constant/all-missing filtering were fitted separately within each fold. Logistic regression additionally used training-only numeric scaling. Missingness indicators were compared explicitly. Incomplete records were not dropped; validation/test class proportions were preserved. Row IDs and timestamps were excluded as predictors. The test was reserved until the method was frozen.

## Baselines and stronger models
The prevalence reference, logistic regression and shallow decision tree were followed by stronger-regularized logistic regression, a constrained Random Forest and shallow histogram gradient boosting. Fixed settings and the same folds were used without broad hyperparameter searches. Mean fold average precision without indicators was 0.0662 for the reference, 0.1294 for logistic regression, 0.1072 for the shallow tree, 0.1290 for stronger-regularized logistic regression, 0.1987 for Random Forest and 0.2216 for boosting. Boosting ranked failures best in this comparison but did not provide the selected detection/workload tradeoff at evaluated thresholds. Fold variability is not a confidence interval.

## Threshold choice and frozen method
Threshold comparisons reused saved out-of-fold predictions. Random Forest without indicators at 0.35 caught 65 of 83 development failures, missed 18 and generated 387 false alarms. The shallow tree at 0.50 caught 48 with 359 false alarms. Thus the chosen forest caught 17 more failures for 28 more false alarms. Compared with forest at 0.40, the 0.35 threshold caught 17 additional failures at the cost of 139 additional false alarms. At 0.35, adding indicators caught the same 65 failures but increased false alarms to 418. The user approved the higher-detection operating point; it was not established as a universal operational optimum.

The frozen forest used 100 trees, maximum depth 5, minimum 10 records per leaf, square-root feature sampling, balanced class weights and seed 42. Full-development preprocessing retained 468 numeric inputs with median imputation, no scaling and no indicator inputs. The method was frozen before one full-development fit and one reserved-test evaluation.

## Final reserved-test result
| Metric | Development validation | Reserved test |
|---|---:|---:|
| Failure recall | 78.3% | 66.7% |
| Precision | 14.4% | 12.1% |
| Specificity | 66.9% | 65.2% |
| Balanced accuracy | 72.6% | 65.9% |
| Average precision | 0.1987 | 0.1872 |

The test caught 14 failures, missed 7, generated 102 false alarms and correctly passed 191 records. Of 116 alerts, only 14 were failures. Performance degraded relative to development validation, especially recall and balanced accuracy. Development AP is the mean of five fold scores, whereas test AP is computed across the reserved set. Development estimates informed selection and are not an independent final estimate. No tuning followed the test result.

## Interpretation and limitations
Measurements 511, 131 and 104 ranked highly under both built-in and permutation importance. Measurement 511 was first by permutation importance and second by built-in importance; its observed median was higher for failures, with substantial overlap. Only six features overlapped across the top-20 lists. Measurement 060 ranked first by built-in importance but shuffling it slightly improved training AP. Correlation, missingness, tails and shuffle variability complicated interpretation.

No fitted fold estimators were saved, so permutation importance used the final model's development training data. It is explicitly an in-sample reliance diagnostic, not independently validated importance. Anonymous-feature importance is predictive association only, not a physical root cause.

The model demonstrated useful screening signal but is not production-ready because of false alarms and missed failures. Only 21 test failures, unresolved production dependence and timing, untested drift, uncalibrated scores and unknown review costs constrain deployment claims. The project demonstrates reproducible validation, leakage prevention and explicit engineering tradeoffs; it does not establish improved manufacturing yield or causal failure mechanisms.
