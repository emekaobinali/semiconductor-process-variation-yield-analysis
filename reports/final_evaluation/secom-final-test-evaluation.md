# SECOM frozen final test evaluation

Approved final method: Random Forest without missingness indicators; threshold 0.35. The complete method was saved before reserved-test access, then fitted once on all 1,253 development records. The original preprocessing code, saved forest parameters, and pinned packages were reused unchanged. The test set was scored once after the fitted method was persisted. No tuning or revisions followed the result.

## Frozen method

100 trees; maximum depth 5; minimum 10 records per leaf; sqrt feature sampling; balanced class weights; bootstrap enabled; seed 42. Complete remaining settings are in the canonical config/frozen_method.json record. Original 590-column order; development-derived constant/all-missing filtering; development medians; no scaling; no indicator columns supplied to the classifier. The development fit retained 468 numeric inputs. All test transformations used frozen development parameters. No records removed.

## Final results compared with development validation

| Measure | Development out-of-fold | Reserved test |
|---|---:|---:|
| Records | 1,253 | 314 |
| Actual failures | 83 | 21 |
| Actual passes | 1,170 | 293 |
| Failures caught (TP) | 65 | 14 |
| Failures missed (FN) | 18 | 7 |
| False alarms (FP) | 387 | 102 |
| Correctly identified passes (TN) | 783 | 191 |
| Total flagged | 452 | 116 |
| Failure recall | 78.3% | 66.7% |
| Precision | 14.4% | 12.1% |
| Specificity | 66.9% | 65.2% |
| Balanced accuracy | 72.6% | 65.9% |
| Average precision | 0.1987 | 0.1872 |

Development AP is the mean of five validation-fold AP values (fold SD 0.0602); test AP is calculated once across the reserved set. Other development rates pool held-out validation decisions. Development results were used for selection, whereas this test was reserved. Different sample sizes make raw count comparisons inappropriate as evidence of improvement; compare rates. The final model is trained on more records than each validation-fold model.

## Confusion matrix

Performance degraded on the reserved set: recall declined from 78.3% to 66.7% (11.6 percentage points), precision from 14.4% to 12.1% (2.3 points), specificity from 66.9% to 65.2% (1.7 points), and balanced accuracy from 72.6% to 65.9% (6.7 points). Average precision decreased from the development-fold mean of 0.1987 to 0.1872. Ranking performance was relatively similar, but failure detection at the frozen threshold held up less well.

In screening terms, 116 of 314 records were flagged for additional review. Fourteen flags corresponded to actual failures and 102 were false alarms. Seven actual failures remained unflagged. The observed false-alarm rate among passes was 34.8%, compared with 33.1% in development. The model detected two-thirds of test failures, but most alerts were passing records. This supports a limited portfolio demonstration of screening tradeoffs, not a claim of deployment-ready manufacturing control or causal diagnosis.

The direction of change is clear in the observed estimates, but 21 failures are too few to treat the size of that change as precisely established. Development estimates also benefited from model and threshold selection. No adjustment was made in response to the test outcome.

Rows are actual outcomes; columns are frozen model decisions.

| Actual outcome | Predicted pass | Flagged failure |
|---|---:|---:|
| Pass | 191 | 102 |
| Failure | 7 | 14 |

## Uncertainty and integrity

There are only 21 test failures; one detection changes recall by 4.76 percentage points. The approximate 95% Wilson interval for test recall is 45.4% to 82.8%, assuming independent records. Unknown production dependence may affect coverage. This interval does not include training or model-selection uncertainty. Scores are not established as calibrated probabilities; associations do not establish physical root causes or deployment readiness.

Original file and split hashes matched; partition IDs were unique, disjoint and complete; all prepared values were finite; classifier/preprocessing state was unchanged by test transformation and scoring; test scores were persisted before reporting. Model training warnings: []. No final method settings were changed. The reserved test is now used and must not become a tuning set. No GitHub, app, deployment, or publishing work was started.
