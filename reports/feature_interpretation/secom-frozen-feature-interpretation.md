# Frozen Random Forest: descriptive feature interpretation

The frozen model, preprocessing, and threshold 0.35 are unchanged. No models were fitted, and no reserved-test records or results were inspected. Model-file and in-memory state checks passed.

**Evaluation limitation:** the final model was trained on all 1,253 development records (1,170 passes, 83 failures). Saved fold predictions cannot support permutation importance without their fitted estimators; no fitted fold estimators were available. The permutation results below therefore describe **in-sample model reliance**, not independent validation or generalizable feature importance.

All 468 retained numeric measurements were evaluated. Feature names use the original 1-based SECOM column number. The other 122 columns were excluded by the already frozen constant-feature rule.

Built-in importance is normalized weighted Gini reduction. Permutation importance is the decrease in average precision (AP) after shuffling one imputed numeric column across development records, averaged over five fixed shuffles (seed 42). The same five permutations were used for every feature. AP is threshold-independent; threshold 0.35 was not changed or optimized. SD measures shuffle variation only, not population uncertainty.

Training-data reference AP: 0.9938; this is not a new validation result. Top-10 overlap: 4 of 10. Top-20 overlap: 6 of 20.

## Top 20 by development permutation importance

Built-in percentages are shares of total impurity importance, not failure probabilities or percentage of failures explained. AP losses are absolute AP units.

| Rank | Anonymous feature | Built-in rank | Built-in % | AP loss ± shuffle SD | Missing % | Shuffle rank range |
|---:|---|---:|---:|---:|---:|---:|
| 1 | measurement_511 | 2 | 2.86 | 0.0148 ± 0.0029 | 0.16 | 1–1 |
| 2 | measurement_131 | 3 | 2.71 | 0.0071 ± 0.0035 | 0.64 | 2–4 |
| 3 | measurement_104 | 4 | 2.35 | 0.0049 ± 0.0015 | 0.16 | 2–7 |
| 4 | measurement_126 | 12 | 1.00 | 0.0045 ± 0.0023 | 0.64 | 2–19 |
| 5 | measurement_472 | 46 | 0.47 | 0.0035 ± 0.0004 | 0.32 | 4–9 |
| 6 | measurement_034 | 7 | 1.43 | 0.0033 ± 0.0021 | 0.00 | 3–76 |
| 7 | measurement_548 | 27 | 0.65 | 0.0029 ± 0.0013 | 16.84 | 3–53 |
| 8 | measurement_417 | 85 | 0.31 | 0.0027 ± 0.0005 | 0.64 | 8–17 |
| 9 | measurement_082 | 91 | 0.30 | 0.0025 ± 0.0007 | 1.76 | 10–26 |
| 10 | measurement_133 | 62 | 0.40 | 0.0025 ± 0.0009 | 0.56 | 8–54 |
| 11 | measurement_127 | 23 | 0.73 | 0.0025 ± 0.0018 | 0.64 | 6–146 |
| 12 | measurement_200 | 81 | 0.32 | 0.0023 ± 0.0005 | 0.32 | 10–28 |
| 13 | measurement_003 | 26 | 0.66 | 0.0023 ± 0.0007 | 0.88 | 9–51 |
| 14 | measurement_564 | 20 | 0.76 | 0.0022 ± 0.0022 | 17.56 | 5–118 |
| 15 | measurement_341 | 89 | 0.30 | 0.0021 ± 0.0008 | 0.32 | 9–64 |
| 16 | measurement_540 | 61 | 0.41 | 0.0020 ± 0.0003 | 0.56 | 16–34 |
| 17 | measurement_296 | 44 | 0.48 | 0.0020 ± 0.0007 | 0.16 | 8–41 |
| 18 | measurement_418 | 49 | 0.47 | 0.0020 ± 0.0014 | 0.16 | 3–203 |
| 19 | measurement_456 | 24 | 0.69 | 0.0019 ± 0.0017 | 0.08 | 6–210 |
| 20 | measurement_204 | 29 | 0.63 | 0.0019 ± 0.0011 | 0.32 | 5–170 |

Built-in top 20, in order: measurement_060, measurement_511, measurement_131, measurement_104, measurement_248, measurement_478, measurement_034, measurement_065, measurement_574, measurement_057, measurement_154, measurement_126, measurement_520, measurement_342, measurement_206, measurement_112, measurement_427, measurement_066, measurement_273, measurement_564.

## Observed development distributions

Values are reported exactly on the anonymous dataset scale; no physical unit is inferred. Medians and [Q1, Q3] exclude missing observations. Missing percentages use all 1,170 passes or 83 failures as their denominators. These are descriptive comparisons, not adjusted effects or significance tests.

| Feature | Pass observed n | Pass median [Q1, Q3] | Fail observed n | Fail median [Q1, Q3] | Missing pass / fail % | Outlier % |
|---|---:|---|---:|---|---:|---:|
| measurement_511 | 1168 | 46.182 [35.227, 63.338] | 83 | 60.841 [46.598, 89.702] | 0.17 / 0.00 | 5.76 |
| measurement_131 | 1162 | 0.7536 [0.6848, 0.8121] | 83 | 0.7692 [0.7171, 0.821] | 0.68 / 0.00 | 0.24 |
| measurement_104 | 1168 | -0.0101 [-0.0119, -0.0084] | 83 | -0.008 [-0.0105, -0.0057] | 0.17 / 0.00 | 1.68 |
| measurement_126 | 1162 | 1.154 [0.9791, 1.3558] | 83 | 1.097 [0.94125, 1.239] | 0.68 / 0.00 | 2.25 |
| measurement_472 | 1166 | 7.4027 [5.7748, 9.6808] | 83 | 7.5116 [6.0115, 9.8481] | 0.34 / 0.00 | 11.37 |
| measurement_034 | 1170 | 8.77 [8.5788, 9.043] | 83 | 8.8356 [8.659, 9.1855] | 0.00 / 0.00 | 3.83 |
| measurement_548 | 975 | 403.14 [400.72, 407.48] | 67 | 405.53 [400.81, 407.46] | 16.67 / 19.28 | 1.15 |
| measurement_417 | 1162 | 3.2358 [2.6741, 4.0224] | 83 | 3.0361 [2.564, 3.8668] | 0.68 / 0.00 | 1.29 |
| measurement_082 | 1148 | -0.0196 [-0.027425, -0.012075] | 83 | -0.0193 [-0.02575, -0.00735] | 1.88 / 0.00 | 9.99 |
| measurement_133 | 1163 | 2.3099 [2.2765, 2.3524] | 83 | 2.3167 [2.2772, 2.3619] | 0.60 / 0.00 | 0.16 |
| measurement_127 | 1162 | 2.734 [2.563, 2.868] | 83 | 2.768 [2.622, 2.938] | 0.68 / 0.00 | 2.89 |
| measurement_200 | 1166 | 8.58 [6.72, 11.468] | 83 | 8.62 [7.12, 11.235] | 0.34 / 0.00 | 10.81 |
| measurement_003 | 1159 | 2201.6 [2183.2, 2218.6] | 83 | 2196.1 [2173.5, 2214.3] | 0.94 / 0.00 | 2.01 |
| measurement_564 | 970 | 0.6505 [0.5671, 0.77395] | 63 | 0.6575 [0.5671, 0.72885] | 17.09 / 24.10 | 0.48 |
| measurement_341 | 1166 | 0.0456 [0.0344, 0.0646] | 83 | 0.0544 [0.03605, 0.0757] | 0.34 / 0.00 | 8.73 |
| measurement_540 | 1163 | 3.0869 [1.8937, 3.9418] | 83 | 3.0097 [1.6126, 3.9272] | 0.60 / 0.00 | 0.00 |
| measurement_296 | 1169 | 196.88 [129.28, 272.97] | 82 | 231.52 [162.85, 298.08] | 0.09 / 1.20 | 4.72 |
| measurement_418 | 1168 | 7.4102 [5.7669, 9.2058] | 83 | 6.7334 [5.2537, 8.2919] | 0.17 / 0.00 | 7.51 |
| measurement_456 | 1169 | 3.7545 [2.9106, 4.3665] | 83 | 4.1434 [3.1045, 4.4559] | 0.09 / 0.00 | 0.24 |
| measurement_204 | 1166 | 30.173 [24.95, 33.353] | 83 | 31.699 [24.488, 35.275] | 0.34 / 0.00 | 1.12 |

Outliers are observed values outside pooled development Q1 − 1.5×IQR or Q3 + 1.5×IQR. This is a descriptive tail flag, not evidence of a bad measurement; no values were removed.

## Correlation cautions

For each selected feature, the strongest absolute Spearman correlation with another retained measurement is shown. Computed on pairwise observed development values only, with a minimum of 30 pairs.

| Feature | Most correlated measurement | Spearman rho | Observed pairs |
|---|---|---:|---:|
| measurement_511 | measurement_239 | 0.698 | 1251 |
| measurement_131 | measurement_123 | -0.816 | 1245 |
| measurement_104 | measurement_511 | 0.635 | 1251 |
| measurement_126 | measurement_123 | 0.646 | 1245 |
| measurement_472 | measurement_200 | 0.987 | 1249 |
| measurement_034 | measurement_038 | 0.401 | 1253 |
| measurement_548 | measurement_550 | 0.386 | 1042 |
| measurement_417 | measurement_144 | 0.998 | 1245 |
| measurement_082 | measurement_158 | -0.257 | 110 |
| measurement_133 | measurement_123 | -0.462 | 1245 |
| measurement_127 | measurement_129 | 0.704 | 1245 |
| measurement_200 | measurement_472 | 0.987 | 1249 |
| measurement_003 | measurement_008 | -0.465 | 1239 |
| measurement_564 | measurement_566 | 0.564 | 1033 |
| measurement_341 | measurement_205 | 0.978 | 1249 |
| measurement_540 | measurement_268 | 0.998 | 1246 |
| measurement_296 | measurement_161 | 0.993 | 1251 |
| measurement_418 | measurement_145 | 0.989 | 1251 |
| measurement_456 | measurement_184 | 0.997 | 1252 |
| measurement_204 | measurement_476 | 0.996 | 1249 |

## Interpretation of the leading measurements

Measurement 511 is the clearest shared signal: permutation rank 1 in all five shuffles and built-in rank 2. Observed failure values have median 60.84 [46.60, 89.70], versus 46.18 [35.23, 63.34] for passes. Missingness is only 0.16% overall. This supports a descriptive association with higher values, with substantial distribution overlap; it does not establish a standalone cutoff or causal mechanism.
Measurements 131 and 104 also rank in the top four under both methods. Their observed failure medians are higher, but their pass/fail distributions overlap. Measurement 131 is strongly correlated with measurement 123 (rho −0.816), so its apparent contribution is not independent.
The four shared top-10 features are 511, 131, 104 and 034. Six features are shared in the top 20: those four plus 126 and 564. Agreement is therefore partial, not broad confirmation of a stable feature ranking.
The largest disagreement is measurement 060: built-in rank 1 (4.96% of total impurity importance), but permutation rank 468, with mean AP loss −0.00231 ± 0.00136. Shuffling it slightly improves in-sample AP. Its observed failure median is 5.8091 [1.6696, 15.9841], versus 0.7355 [−1.9827, 3.7555] for passes (83 and 1,165 observed records). Missingness is 0.40% overall, 0.43% for passes and 0% for failures; 13.14% of observed values fall outside pooled Tukey fences. A marginal distribution difference and frequent tree use do not guarantee a positive contribution to AP in this fitted ensemble. These findings do not establish whether tails or redundancy explain the discrepancy.
Missingness warrants particular caution for measurements 548 and 564: 16.84% and 17.56% missing overall. For 564 the rate is 24.10% among failures versus 17.09% among passes, leaving only 63 observed failures. Imputation could therefore contribute to its apparent importance; this analysis does not isolate that effect.
Examples of near-redundant inputs include 472/200 (rho 0.987), 417/144 (0.998), and 540/268 (0.998). Measurement 472 also has 11.37% of observed values outside pooled Tukey fences; measurement 082 has 9.99%. These are caution flags, not proof that outliers drive predictions.
Rank stability is limited beyond the leaders: 034 ranges from rank 3 to 76, 548 from 3 to 53, and 564 from 5 to 118 across just five shuffles. The near-perfect training reference AP (0.9938) further limits interpretation: many single-feature shuffles barely disturb the fitted training ranking. These small AP losses must not be interpreted as independent validation gains or reasons to alter the frozen method.

## Main limitations

- Both importance methods describe this fitted model. Training-data permutation cannot establish independent predictive importance and may reflect overfitting. No test-set-driven choices were made.
- Five shuffles assess random permutation variability only. They do not assess sensitivity to a different development sample, fold, or fitted forest. Rank differences smaller than shuffle variability should not be overinterpreted.
- Median imputation remains part of the frozen model. Excluding explicit missingness indicators does not remove all missingness information: repeated imputed medians can still influence splits. Permuting the imputed feature moves both observed values and imputed values; their contributions are not separated here.
- Correlated inputs can share or substitute for information, reducing individual permutation importance. Shuffling also creates combinations of measurements that may be uncommon in real records.
- Built-in importance can favor variables offering more possible split points. It measures accumulated splitting benefit, whereas AP permutation measures disruption of failure ranking; their magnitudes are not directly comparable.
- Distribution shifts and outlier flags do not demonstrate why the model relies on a feature. No outlier-removal or missingness ablation was performed. Overlapping distributions are expected; a univariate median difference is not a decision rule.
- There are only 83 development failures, sometimes fewer observed values per feature. Anonymous measurements have no supported physical names or units. Importance and class differences are associations, not evidence of a physical root cause.

Method reference: [scikit-learn permutation importance documentation](https://scikit-learn.org/stable/modules/permutation_importance.html).

Frozen model SHA-256: `0ff4125f395f430ea87c1e728a284d09cd0369e6780627d45a413102ef30ad71`.