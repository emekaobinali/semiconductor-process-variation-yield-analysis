# SECOM: approved protocol and split record

Status: split created; no preprocessing, association analysis, feature selection, modeling, or test evaluation performed.

## Completed split

| Partition | Records | Passes | Failures |
|---|---:|---:|---:|
| Development | 1,253 | 1,170 | 83 |
| Reserved test | 314 | 293 | 21 |

Counts above were checked at split creation only. Every original record belongs to exactly one partition, with no overlap. All 590 original measurements, NaN markers, and quality-test timestamps remain in the unchanged raw files. The partition manifests contain original row identifiers, not copied or processed measurements. They define the datasets by reference.

The split uses seed 42, fixed in advance, and shuffled stratified allocation. The total test count is rounded upward from 20%; class counts use proportional largest-remainder allocation. Python's built-in random generator was used because scikit-learn is not installed. Saved manifests define the exact split; it is not claimed to reproduce scikit-learn's splitting algorithm. Repeating the same algorithm reproduced the same identifiers. Original file hashes match the completed inspection.

## Approved handling and validation rules

1. Preserve original files and traceable working copies. Failure is the positive outcome. Row identifiers and timestamps are metadata, excluded from primary predictive inputs.
2. Keep the reserved test partition untouched after creation. Further exploration and all preprocessing, filtering, feature selection, scaling, and model tuning use development data only. Test evaluation requires a frozen method and explicit user authorization.
3. For later modeling, use training-only median imputation and preserve missingness flags. Do not discard incomplete records or assume zero is missing. Observed-value summaries must report available counts and never present imputed values as measured data.
4. Within each training subset only, exclude all-missing or constant-observed measurement inputs. Keep a varying missingness flag even when its measurement input is excluded. Preserve all original columns and log exclusions. No arbitrary near-constant or missingness cutoff.
5. Fit the entire preprocessing and selection pipeline inside each training fold. Never fit on validation or test records. Unknown acquisition timing and entity relationships remain leakage and dependence limitations.
6. Use five-fold stratified cross-validation within development data, with shared folds for comparisons. Tuning scores are development results. Freeze the method before the reserved test is evaluated.
7. Primary ranking metric: average precision. Also report failure recall, precision, specificity, balanced accuracy, false alarms, missed failures, and confusion-matrix counts. Accuracy and ROC-AUC are supplementary. Report uncertainty with assumptions and acknowledge the small failure count.
8. Preserve natural class proportions in validation and test data. Initially avoid synthetic oversampling and deleting passing records. Any later weighting and alert-threshold choice must be development-only; the user decides the false-alarm versus missed-failure tradeoff.
9. Keep associations exploratory; account for multiple comparisons if many features are tested. Do not assign unsupported physical meaning, equate unusual values with defects, or infer causation from association.
10. Preserve timestamps for a later, separate chronological sensitivity analysis with earlier-only fitting and tied timestamps kept together. It will not drive the primary analysis or select the primary model. Differences would not by themselves prove drift.
11. Retain missingness indicators for now; later evaluate their incremental predictive contribution through a controlled development-only comparison with and without them. Do not assume they are useful. Any final choice is frozen before test evaluation.

## Next proposed milestone

Implement and check the development-only preprocessing pipeline. No such implementation or fitting has started in this step.

Source: McCann, M. and Johnston, A. (2008), [SECOM, UCI Machine Learning Repository](https://doi.org/10.24432/C54305), CC BY 4.0. Inspection and split performed September 6, 2026.
