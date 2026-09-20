# SECOM ensemble threshold comparison

Only saved out-of-fold development predictions were used. No model training, raw-data inspection, or reserved-test access occurred. Thresholds 0.20, 0.50, and 0.80 were fixed before examining the results. Both missingness variants remain in scope. Scores at or above a threshold trigger a flag.

All configurations cover the same 1,253 development records, including 83 failures and 1,170 passes, using the same five folds. Each record contributes once to pooled counts. Rates below are computed from pooled counts; per-fold metrics are provided separately. These are exploratory development comparisons, not independent test performance or evidence of a statistically established advantage. Weighted-model scores are not calibrated probabilities and the same numeric threshold does not imply the same review workload across models.

| Model | Indicators | Threshold | Recall | Precision | False alarms | Missed failures | Specificity | Balanced accuracy | Detected failures | Total flagged |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tree | False | 0.20 | 62.7% | 10.5% | 443 | 31 | 62.1% | 62.4% | 52 | 495 |
| tree | False | 0.50 | 57.8% | 11.8% | 359 | 35 | 69.3% | 63.6% | 48 | 407 |
| tree | False | 0.80 | 20.5% | 15.6% | 92 | 66 | 92.1% | 56.3% | 17 | 109 |
| tree | True | 0.20 | 62.7% | 10.4% | 446 | 31 | 61.9% | 62.3% | 52 | 498 |
| tree | True | 0.50 | 57.8% | 11.8% | 359 | 35 | 69.3% | 63.6% | 48 | 407 |
| tree | True | 0.80 | 20.5% | 15.6% | 92 | 66 | 92.1% | 56.3% | 17 | 109 |
| small_forest | False | 0.20 | 97.6% | 7.4% | 1014 | 2 | 13.3% | 55.5% | 81 | 1095 |
| small_forest | False | 0.50 | 19.3% | 17.6% | 75 | 67 | 93.6% | 56.4% | 16 | 91 |
| small_forest | False | 0.80 | 0.0% | undefined | 0 | 83 | 100.0% | 50.0% | 0 | 0 |
| small_forest | True | 0.20 | 97.6% | 7.1% | 1065 | 2 | 9.0% | 53.3% | 81 | 1146 |
| small_forest | True | 0.50 | 27.7% | 25.3% | 68 | 60 | 94.2% | 60.9% | 23 | 91 |
| small_forest | True | 0.80 | 0.0% | undefined | 0 | 83 | 100.0% | 50.0% | 0 | 0 |
| shallow_boosting | False | 0.20 | 39.8% | 16.9% | 162 | 50 | 86.2% | 63.0% | 33 | 195 |
| shallow_boosting | False | 0.50 | 14.5% | 30.0% | 28 | 71 | 97.6% | 56.0% | 12 | 40 |
| shallow_boosting | False | 0.80 | 0.0% | 0.0% | 3 | 83 | 99.7% | 49.9% | 0 | 3 |
| shallow_boosting | True | 0.20 | 39.8% | 16.9% | 162 | 50 | 86.2% | 63.0% | 33 | 195 |
| shallow_boosting | True | 0.50 | 14.5% | 30.0% | 28 | 71 | 97.6% | 56.0% | 12 | 40 |
| shallow_boosting | True | 0.80 | 0.0% | 0.0% | 3 | 83 | 99.7% | 49.9% | 0 | 3 |

Within each model, 0.80 is the conservative candidate, 0.50 the middle candidate, and 0.20 the aggressive candidate. These names describe the relative alert rates of this fixed candidate set, not an optimized economic balance. Flags represent records proposed for further review, not confirmed individual wafers or instructions to scrap product. No final model or threshold is selected.

## What the operating points mean

Random forest: at 0.80, no records are flagged and all 83 failures are missed; this conservative extreme has no failure-screening value in these predictions. At 0.50, 91 records are flagged, catching 16 failures without indicators or 23 with indicators. At 0.20, 81 failures are caught but 1,095–1,146 of all 1,253 records are flagged (87.4–91.5%). That catches nearly all failures by sending almost everything for review; it is not evidence of efficient screening.

Gradient boosting: at 0.80, three passing records are flagged and no failures are caught. At 0.50, 40 records are flagged, catching 12 failures and generating 28 false alarms. At 0.20, 195 records are flagged, catching 33 failures and generating 162 false alarms. Both indicator variants have the same results at every candidate threshold.

Shallow tree: at 0.80, 109 records are flagged, catching 17 failures and generating 92 false alarms. At 0.50, 407 records are flagged, catching 48 failures and generating 359 false alarms. At 0.20, 495–498 records are flagged, catching 52 failures and generating 443–446 false alarms.

## Comparison with the current shallow tree at 0.50

Boosting at 0.20 is a useful reduced-workload option for review: 197 fewer false alarms, but 15 fewer failures caught. Balanced accuracy is 63.0%, versus 63.6% for the tree, which illustrates how similar balanced accuracy can hide different screening outcomes. This is not a proven best practical tradeoff; review capacity and the cost of missed failures have not been agreed.

Forest at 0.20 catches 33 more failures than the tree at 0.50, but adds 655–706 false alarms. Forest at 0.50 produces 284–291 fewer false alarms but catches 25–32 fewer failures. Neither is an unqualified improvement over the current tree operating point.

Across other candidates, forest with indicators at 0.50 catches 23 failures with 68 false alarms, compared with 17 failures and 92 false alarms for the conservative tree at 0.80. It therefore improves both observed counts over that conservative tree point, though not over the current tree at 0.50. This is development evidence only, not statistical proof of general superiority.

No tested ensemble point simultaneously matches or exceeds the current tree's 48 detections while keeping false alarms at or below 359. The three thresholds are illustrative and coarse; results do not establish whether untested intermediate thresholds could do so. No extra thresholds were searched and no final operating point is selected.

Validation: all configurations have identical record, fold, and outcome memberships; the stored 0.50 predictions are reproduced; fold confusion counts reconcile to pooled totals; lowering thresholds never reduces detections or false alarms. Source prediction files are unchanged. No threshold search was conducted beyond these three candidates.
