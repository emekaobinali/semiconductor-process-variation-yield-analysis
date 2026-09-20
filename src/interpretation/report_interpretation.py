from src.paths import ROOT, resource, report, require_file, protect_completed_evidence
from pathlib import Path
import json,csv,shutil

def main():
    protect_completed_evidence()
    """Render cached interpretation only; no estimator or data loading."""
    root = ROOT
    src=resource('feature_interpretation')
    out=ROOT / 'artifacts/reproduction'
    out.mkdir(parents=True, exist_ok=True)
    a=json.loads((resource('feature_interpretation/audit.json')).read_text())
    rows=list(csv.DictReader((resource('feature_interpretation/top20_features.csv')).open()))
    corr=list(csv.DictReader((resource('feature_interpretation/selected_correlations.csv')).open()))
    f=lambda r,k:float(r[k])
    lines=['# Frozen Random Forest: descriptive feature interpretation','',
    'The frozen model, preprocessing, and threshold 0.35 are unchanged. No models were fitted, and no reserved-test records or results were inspected. Model-file and in-memory state checks passed.','',
    '**Evaluation limitation:** the final model was trained on all 1,253 development records (1,170 passes, 83 failures). Saved fold predictions cannot support permutation importance without their fitted estimators; no fitted fold estimators were available. The permutation results below therefore describe **in-sample model reliance**, not independent validation or generalizable feature importance.','',
    'All 468 retained numeric measurements were evaluated. Feature names use the original 1-based SECOM column number. The other 122 columns were excluded by the already frozen constant-feature rule.','',
    'Built-in importance is normalized weighted Gini reduction. Permutation importance is the decrease in average precision (AP) after shuffling one imputed numeric column across development records, averaged over five fixed shuffles (seed 42). The same five permutations were used for every feature. AP is threshold-independent; threshold 0.35 was not changed or optimized. SD measures shuffle variation only, not population uncertainty.','',
    f'Training-data reference AP: {a["training_average_precision_reference"]:.4f}; this is not a new validation result. Top-10 overlap: {len(a["top10_overlap"])} of 10. Top-20 overlap: {len(a["top20_overlap"])} of 20.','',
    '## Top 20 by development permutation importance','',
    'Built-in percentages are shares of total impurity importance, not failure probabilities or percentage of failures explained. AP losses are absolute AP units.','',
    '| Rank | Anonymous feature | Built-in rank | Built-in % | AP loss ± shuffle SD | Missing % | Shuffle rank range |',
    '|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
     lines.append(f'| {r["permutation_rank"]} | {r["feature"]} | {r["builtin_rank"]} | {100*f(r,"builtin_importance"):.2f} | {f(r,"permutation_AP_loss"):.4f} ± {f(r,"permutation_shuffle_SD"):.4f} | {f(r,"missing_pct"):.2f} | {r["shuffle_rank_min"]}–{r["shuffle_rank_max"]} |')
    lines+=['','Built-in top 20, in order: '+', '.join(a['builtin_top20'])+'.','',
    '## Observed development distributions','',
    'Values are reported exactly on the anonymous dataset scale; no physical unit is inferred. Medians and [Q1, Q3] exclude missing observations. Missing percentages use all 1,170 passes or 83 failures as their denominators. These are descriptive comparisons, not adjusted effects or significance tests.','',
    '| Feature | Pass observed n | Pass median [Q1, Q3] | Fail observed n | Fail median [Q1, Q3] | Missing pass / fail % | Outlier % |',
    '|---|---:|---|---:|---|---:|---:|']
    for r in rows:
     fmt=lambda label:f'{f(r,label+"_median"):.5g} [{f(r,label+"_q1"):.5g}, {f(r,label+"_q3"):.5g}]'
     lines.append(f'| {r["feature"]} | {int(f(r,"pass_observed_n"))} | {fmt("pass")} | {int(f(r,"fail_observed_n"))} | {fmt("fail")} | {f(r,"pass_missing_pct"):.2f} / {f(r,"fail_missing_pct"):.2f} | {f(r,"observed_tukey_outlier_pct"):.2f} |')
    lines+=['','Outliers are observed values outside pooled development Q1 − 1.5×IQR or Q3 + 1.5×IQR. This is a descriptive tail flag, not evidence of a bad measurement; no values were removed.','',
    '## Correlation cautions','',
    'For each selected feature, the strongest absolute Spearman correlation with another retained measurement is shown. Computed on pairwise observed development values only, with a minimum of 30 pairs.','',
    '| Feature | Most correlated measurement | Spearman rho | Observed pairs |','|---|---|---:|---:|']
    for r in corr:lines.append(f'| {r["feature"]} | {r["most_correlated_feature"]} | {f(r,"spearman_rho"):.3f} | {r["pairwise_observed_n"]} |')
    lines+=['','## Interpretation of the leading measurements','',
    'Measurement 511 is the clearest shared signal: permutation rank 1 in all five shuffles and built-in rank 2. Observed failure values have median 60.84 [46.60, 89.70], versus 46.18 [35.23, 63.34] for passes. Missingness is only 0.16% overall. This supports a descriptive association with higher values, with substantial distribution overlap; it does not establish a standalone cutoff or causal mechanism.',
    'Measurements 131 and 104 also rank in the top four under both methods. Their observed failure medians are higher, but their pass/fail distributions overlap. Measurement 131 is strongly correlated with measurement 123 (rho −0.816), so its apparent contribution is not independent.',
    'The four shared top-10 features are 511, 131, 104 and 034. Six features are shared in the top 20: those four plus 126 and 564. Agreement is therefore partial, not broad confirmation of a stable feature ranking.',
    'The largest disagreement is measurement 060: built-in rank 1 (4.96% of total impurity importance), but permutation rank 468, with mean AP loss −0.00231 ± 0.00136. Shuffling it slightly improves in-sample AP. Its observed failure median is 5.8091 [1.6696, 15.9841], versus 0.7355 [−1.9827, 3.7555] for passes (83 and 1,165 observed records). Missingness is 0.40% overall, 0.43% for passes and 0% for failures; 13.14% of observed values fall outside pooled Tukey fences. A marginal distribution difference and frequent tree use do not guarantee a positive contribution to AP in this fitted ensemble. These findings do not establish whether tails or redundancy explain the discrepancy.',
    'Missingness warrants particular caution for measurements 548 and 564: 16.84% and 17.56% missing overall. For 564 the rate is 24.10% among failures versus 17.09% among passes, leaving only 63 observed failures. Imputation could therefore contribute to its apparent importance; this analysis does not isolate that effect.',
    'Examples of near-redundant inputs include 472/200 (rho 0.987), 417/144 (0.998), and 540/268 (0.998). Measurement 472 also has 11.37% of observed values outside pooled Tukey fences; measurement 082 has 9.99%. These are caution flags, not proof that outliers drive predictions.',
    'Rank stability is limited beyond the leaders: 034 ranges from rank 3 to 76, 548 from 3 to 53, and 564 from 5 to 118 across just five shuffles. The near-perfect training reference AP (0.9938) further limits interpretation: many single-feature shuffles barely disturb the fitted training ranking. These small AP losses must not be interpreted as independent validation gains or reasons to alter the frozen method.','',
    '## Main limitations','',
    '- Both importance methods describe this fitted model. Training-data permutation cannot establish independent predictive importance and may reflect overfitting. No test-set-driven choices were made.',
    '- Five shuffles assess random permutation variability only. They do not assess sensitivity to a different development sample, fold, or fitted forest. Rank differences smaller than shuffle variability should not be overinterpreted.',
    '- Median imputation remains part of the frozen model. Excluding explicit missingness indicators does not remove all missingness information: repeated imputed medians can still influence splits. Permuting the imputed feature moves both observed values and imputed values; their contributions are not separated here.',
    '- Correlated inputs can share or substitute for information, reducing individual permutation importance. Shuffling also creates combinations of measurements that may be uncommon in real records.',
    '- Built-in importance can favor variables offering more possible split points. It measures accumulated splitting benefit, whereas AP permutation measures disruption of failure ranking; their magnitudes are not directly comparable.',
    '- Distribution shifts and outlier flags do not demonstrate why the model relies on a feature. No outlier-removal or missingness ablation was performed. Overlapping distributions are expected; a univariate median difference is not a decision rule.',
    '- There are only 83 development failures, sometimes fewer observed values per feature. Anonymous measurements have no supported physical names or units. Importance and class differences are associations, not evidence of a physical root cause.','',
    'Method reference: [scikit-learn permutation importance documentation](https://scikit-learn.org/stable/modules/permutation_importance.html).','',
    f'Frozen model SHA-256: `{a["model_sha256"]}`.']
    (out/'secom-frozen-feature-interpretation.md').write_text('\n'.join(lines),encoding='utf-8')
    for name in ['all_feature_importances.csv','top20_features.csv','selected_correlations.csv','audit.json']:
     shutil.copyfile(resource('feature_interpretation/'+name),out/('secom-feature-'+name))
    print(out/'secom-frozen-feature-interpretation.md')


if __name__ == '__main__':
    main()
