# SECOM threshold tradeoffs — development validation only

The fixed candidate thresholds were 0.20, 0.50, and 0.80, selected before reviewing results. A record is flagged when its saved failure score is at least the threshold. These are three illustrative operating points, not optimized thresholds. Both missingness variants are included so no feature decision is silently made.

The only input was the existing development out-of-fold prediction file. Each of 1,253 records (83 failures and 1,170 passes) was previously scored by a model that did not train on it. No raw measurements, timestamps, reserved test files, model objects, or training routines were opened or used. No new models were fitted. The input file remained unchanged.

Rates below use pooled counts across the existing five folds. Recall is the share of failures caught; precision is the share of flagged records that failed; specificity is the share of passes left unflagged; balanced accuracy averages recall and specificity. False alarms and missed failures are record counts. Full per-fold results are supplied separately. Both models used balanced training class weights; these scores are not established as calibrated probabilities, so 0.20 does not mean a known 20% failure risk.

## Logistic — without missingness indicators

| Threshold | Recall | Precision | False alarms | Missed failures | Specificity | Balanced accuracy | Total flagged |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 28.9% | 11.9% | 177 | 59 | 84.9% | 56.9% | 201 |
| 0.50 | 21.7% | 12.7% | 124 | 65 | 89.4% | 55.5% | 142 |
| 0.80 | 18.1% | 14.6% | 88 | 68 | 92.5% | 55.3% | 103 |

**Threshold 0.20:** flag 201 records for additional review: 24 actually failed and 177 passed. Leave 59 actual failures unflagged. These are retrospective screening analogies, not instructions to scrap product or proof of real deployment performance.

**Threshold 0.50:** flag 142 records for additional review: 18 actually failed and 124 passed. Leave 65 actual failures unflagged. These are retrospective screening analogies, not instructions to scrap product or proof of real deployment performance.

**Threshold 0.80:** flag 103 records for additional review: 15 actually failed and 88 passed. Leave 68 actual failures unflagged. These are retrospective screening analogies, not instructions to scrap product or proof of real deployment performance.

## Logistic — with missingness indicators

| Threshold | Recall | Precision | False alarms | Missed failures | Specificity | Balanced accuracy | Total flagged |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 27.7% | 11.7% | 173 | 60 | 85.2% | 56.5% | 196 |
| 0.50 | 22.9% | 14.0% | 117 | 64 | 90.0% | 56.4% | 136 |
| 0.80 | 15.7% | 14.6% | 76 | 70 | 93.5% | 54.6% | 89 |

**Threshold 0.20:** flag 196 records for additional review: 23 actually failed and 173 passed. Leave 60 actual failures unflagged. These are retrospective screening analogies, not instructions to scrap product or proof of real deployment performance.

**Threshold 0.50:** flag 136 records for additional review: 19 actually failed and 117 passed. Leave 64 actual failures unflagged. These are retrospective screening analogies, not instructions to scrap product or proof of real deployment performance.

**Threshold 0.80:** flag 89 records for additional review: 13 actually failed and 76 passed. Leave 70 actual failures unflagged. These are retrospective screening analogies, not instructions to scrap product or proof of real deployment performance.

## Tree — without missingness indicators

| Threshold | Recall | Precision | False alarms | Missed failures | Specificity | Balanced accuracy | Total flagged |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 62.7% | 10.5% | 443 | 31 | 62.1% | 62.4% | 495 |
| 0.50 | 57.8% | 11.8% | 359 | 35 | 69.3% | 63.6% | 407 |
| 0.80 | 20.5% | 15.6% | 92 | 66 | 92.1% | 56.3% | 109 |

**Threshold 0.20:** flag 495 records for additional review: 52 actually failed and 443 passed. Leave 31 actual failures unflagged. These are retrospective screening analogies, not instructions to scrap product or proof of real deployment performance.

**Threshold 0.50:** flag 407 records for additional review: 48 actually failed and 359 passed. Leave 35 actual failures unflagged. These are retrospective screening analogies, not instructions to scrap product or proof of real deployment performance.

**Threshold 0.80:** flag 109 records for additional review: 17 actually failed and 92 passed. Leave 66 actual failures unflagged. These are retrospective screening analogies, not instructions to scrap product or proof of real deployment performance.

## Tree — with missingness indicators

| Threshold | Recall | Precision | False alarms | Missed failures | Specificity | Balanced accuracy | Total flagged |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 62.7% | 10.4% | 446 | 31 | 61.9% | 62.3% | 498 |
| 0.50 | 57.8% | 11.8% | 359 | 35 | 69.3% | 63.6% | 407 |
| 0.80 | 20.5% | 15.6% | 92 | 66 | 92.1% | 56.3% | 109 |

**Threshold 0.20:** flag 498 records for additional review: 52 actually failed and 446 passed. Leave 31 actual failures unflagged. These are retrospective screening analogies, not instructions to scrap product or proof of real deployment performance.

**Threshold 0.50:** flag 407 records for additional review: 48 actually failed and 359 passed. Leave 35 actual failures unflagged. These are retrospective screening analogies, not instructions to scrap product or proof of real deployment performance.

**Threshold 0.80:** flag 109 records for additional review: 17 actually failed and 92 passed. Leave 66 actual failures unflagged. These are retrospective screening analogies, not instructions to scrap product or proof of real deployment performance.

## Options for review

- Conservative candidate: 0.80, emphasizing fewer false alarms among these candidates, at the cost of missed failures.
- Middle candidate: 0.50, the original reference decision rule. This is a compromise candidate, not a claim that manufacturing costs are balanced or that balanced accuracy is maximized.
- Aggressive candidate: 0.20, intended to prioritize failure detection and accept more screening work. A shallow tree has only a few distinct leaf scores, so different thresholds may produce identical decisions; assess the actual counts rather than the label.

Choosing an operating point requires an agreed cost or review-capacity tradeoff. These development comparisons are exploratory and do not establish an optimal threshold, model, or indicator choice. No final decision has been made. The screening analogy concerns production records, not confirmed individual wafers or physical root causes. Unknown entity dependence and acquisition timing remain limitations. The reserved test set remains untouched.

## Verification

Every model/indicator group has exactly one saved validation prediction per development record. Summed fold confusion counts match pooled counts at each threshold. The 0.50 decisions exactly reproduce the baseline prediction file. Lowering thresholds never reduced detections or false alarms. No threshold search, repeat fitting, or additional experiments were run.
