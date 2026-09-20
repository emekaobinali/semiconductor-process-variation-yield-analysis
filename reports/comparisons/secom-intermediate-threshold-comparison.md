# SECOM intermediate threshold comparison

Only saved out-of-fold development scores were used. No models were loaded or trained; no raw measurements or reserved test files were accessed. Fixed candidates: 0.25, 0.30, 0.35, 0.40, 0.45. Both missingness variants are included, so no indicator choice is implicit.

Population: 1,253 development records, 83 failures and 1,170 passes, with identical saved five-fold memberships. Each record contributes one held-out prediction per model configuration. Rates below pool out-of-fold counts. Flag a record if its score is at least the threshold. Class-weighted scores are not calibrated probabilities.

Reference: shallow tree at 0.50 detects 48 failures, misses 35, and generates 359 false alarms (811 true negatives). Recall 57.8%, precision 11.8%, specificity 69.3%, balanced accuracy 63.6%; total flagged 407. Its indicator variants have identical decisions at this threshold.

| Model | Indicators | Threshold | Recall | Precision | False alarms | Missed failures | Specificity | Balanced accuracy | Detected failures | Total flagged |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small_forest | False | 0.25 | 91.6% | 8.5% | 815 | 7 | 30.3% | 61.0% | 76 | 891 |
| small_forest | False | 0.30 | 89.2% | 11.0% | 596 | 9 | 49.1% | 69.1% | 74 | 670 |
| small_forest | False | 0.35 | 78.3% | 14.4% | 387 | 18 | 66.9% | 72.6% | 65 | 452 |
| small_forest | False | 0.40 | 57.8% | 16.2% | 248 | 35 | 78.8% | 68.3% | 48 | 296 |
| small_forest | False | 0.45 | 34.9% | 17.2% | 140 | 54 | 88.0% | 61.5% | 29 | 169 |
| small_forest | True | 0.25 | 91.6% | 8.0% | 873 | 7 | 25.4% | 58.5% | 76 | 949 |
| small_forest | True | 0.30 | 86.7% | 9.9% | 656 | 11 | 43.9% | 65.3% | 72 | 728 |
| small_forest | True | 0.35 | 78.3% | 13.5% | 418 | 18 | 64.3% | 71.3% | 65 | 483 |
| small_forest | True | 0.40 | 54.2% | 14.5% | 266 | 38 | 77.3% | 65.7% | 45 | 311 |
| small_forest | True | 0.45 | 38.6% | 17.9% | 147 | 51 | 87.4% | 63.0% | 32 | 179 |
| shallow_boosting | False | 0.25 | 37.3% | 21.1% | 116 | 52 | 90.1% | 63.7% | 31 | 147 |
| shallow_boosting | False | 0.30 | 32.5% | 23.3% | 89 | 56 | 92.4% | 62.5% | 27 | 116 |
| shallow_boosting | False | 0.35 | 26.5% | 24.2% | 69 | 61 | 94.1% | 60.3% | 22 | 91 |
| shallow_boosting | False | 0.40 | 18.1% | 22.4% | 52 | 68 | 95.6% | 56.8% | 15 | 67 |
| shallow_boosting | False | 0.45 | 15.7% | 26.5% | 36 | 70 | 96.9% | 56.3% | 13 | 49 |
| shallow_boosting | True | 0.25 | 37.3% | 21.1% | 116 | 52 | 90.1% | 63.7% | 31 | 147 |
| shallow_boosting | True | 0.30 | 32.5% | 23.3% | 89 | 56 | 92.4% | 62.5% | 27 | 116 |
| shallow_boosting | True | 0.35 | 26.5% | 24.2% | 69 | 61 | 94.1% | 60.3% | 22 | 91 |
| shallow_boosting | True | 0.40 | 18.1% | 22.4% | 52 | 68 | 95.6% | 56.8% | 15 | 67 |
| shallow_boosting | True | 0.45 | 15.7% | 26.5% | 36 | 70 | 96.9% | 56.3% | 13 | 49 |

Validation: record/fold/outcome membership matches the baseline; no duplicate or missing records; summed per-fold confusion counts match the totals; scores are finite in [0,1]; lower thresholds do not reduce detection or false-alarm counts; the baseline reproduces the prior totals; source prediction files are unchanged.

These are exploratory development operating points evaluated after earlier model and threshold comparisons, not independent confirmation or population performance estimates. No confidence claim or operational optimum is established. Review capacity, missed-failure costs, dependence between production records, and measurement availability remain unresolved. Flags mean additional review of a production record, not automatic rejection or an identified physical defect. No final model or threshold is selected.

## Promising candidates for review

Random forest without missingness indicators at 0.40 matches the shallow tree's 48 detected failures and 35 missed failures, while reducing false alarms from 359 to 248: 111 fewer, or 30.9% fewer false alarms. Total reviews fall from 407 to 296. This is the clearest aggregate improvement at equal detection count among the requested candidates. Equal counts do not establish that the same individual failures are detected. It is an observed development advantage, not independent confirmation.

Random forest without indicators at 0.35 detects 65 failures, 17 more than the tree, while producing 387 false alarms, 28 more. It flags 452 records versus the tree's 407. This is a higher-detection candidate if the extra review workload is acceptable; balanced accuracy is 72.6%. Compared with forest at 0.40, it catches 17 additional failures at the cost of 139 additional false alarms, so the baseline used for comparison matters.

Gradient boosting at 0.25 detects 31 failures with 116 false alarms, flagging 147 records. Compared with the shallow tree, that is 17 fewer detections and 243 fewer false alarms. It offers a lower-workload option, not equal detection performance. Both indicator configurations give identical results at all requested thresholds.

Random forest without indicators outperforms its indicator variant on both aggregate detection and false-alarm counts at 0.30 and 0.40, and achieves equal detections with fewer false alarms at 0.25 and 0.35. At 0.45, indicators catch three additional failures but create seven additional false alarms. Indicator usefulness therefore remains conditional on the chosen operating point.

No final model, indicator configuration, or threshold is chosen. These five fixed candidates were calculated once from existing predictions; model parameters and preprocessing were unchanged.
