# SECOM preprocessing verification

Implemented and verified on 1,253 development records only. No predictive models trained; no reserved test records parsed or evaluated.

## Preparation behavior

All 590 raw measurement columns, the original missing-value mask, and timestamps are preserved. Timestamps and row identifiers remain metadata. Numeric inputs are excluded only when they have zero or one distinct observed value in the fitted training subset. Missingness flags that vary in training are retained, including flags for excluded constant measurements. Retained numeric columns use training-only median imputation. Optional numeric-only standard scaling was verified; it is off by default. No row deletion, missingness threshold, nearly constant filtering, or feature ranking was performed.

## Full-development structural check

- Retained numeric measurements: **468**.
- Retained missingness indicators: **506**.
- Total prepared columns: **974**.
- Excluded constant-observed numeric inputs: **122**.
- Excluded all-missing numeric inputs: **0**.
- Nonfinite prepared values: **0**.

These are preprocessing counts, not model results. The full-development fitted object is not saved for cross-validation reuse.

## Five-fold isolation checks

Each fold used a new preprocessing pipeline, fitted only on its training records. Validation records were transformed with frozen training parameters. Both scaling-off and scaling-on paths were checked.

| Fold | Training records | Validation records | Numeric inputs | Missingness flags | Total inputs |
|---|---:|---:|---:|---:|---:|
| 1 | 1002 | 251 | 468 | 502 | 970 |
| 2 | 1002 | 251 | 468 | 506 | 974 |
| 3 | 1002 | 251 | 468 | 506 | 974 |
| 4 | 1003 | 250 | 468 | 450 | 918 |
| 5 | 1003 | 250 | 468 | 506 | 974 |

## Verification and limits

All **8 automated tests passed**. They cover medians, missingness preservation, constant/all-missing filtering, zero handling, unchanged raw values, frozen transformation state, numeric-only scaling, independent fold fits, invalid inputs, and ignoring excluded records. All five development validation folds produced finite outputs; each development record appeared in exactly one validation fold. No measurement associations, model metrics, or test results were calculated.

The loader streams the original mixed-partition text files but skips non-development lines before parsing. Reserved manifest access was blocked during verification. No test subset was materialized. The full raw missingness mask remains available even where constant flags are excluded from model inputs.

Missingness flags have **not** been shown to improve prediction; that remains a later development-only comparison. Unknown measurement timing, entity dependence, and missingness mechanisms remain unresolved. Anonymous column order must be preserved. Optional scaling is an implementation capability, not a model-selection decision.

Source: [UCI SECOM](https://doi.org/10.24432/C54305). Reusable implementation, tests, pinned dependencies, and detailed verification counts are in the companion source package. This step stops before predictive modeling.
