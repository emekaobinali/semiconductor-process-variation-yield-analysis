from src.paths import ROOT, resource, report, require_file, protect_completed_evidence
from pathlib import Path
import hashlib, json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from src.preprocessing import load_development
from src.integrity import verify_preprocessing_provenance

def main():
    protect_completed_evidence()
    require_file(resource('final_evaluation/frozen_fitted_model.joblib'))
    """Descriptive frozen-model interpretation; no fitting or reserved-set reads."""

    root = ROOT
    out = ROOT / 'artifacts/reproduction/feature_interpretation'
    out.mkdir(parents=True, exist_ok=True)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    model_path = resource('final_evaluation/frozen_fitted_model.joblib')
    before = sha(model_path)
    bundle = joblib.load(model_path)
    state_before = joblib.hash(bundle)
    prep, model = bundle['preprocessing'], bundle['classifier']
    freeze = bundle['freeze']
    verify_preprocessing_provenance(freeze['preprocessing_source_sha256'])
    assert sha(resource('splits/development_rows.csv')) == freeze['development_manifest_sha256']
    assert model.get_params() == freeze['model_parameters']
    assert bundle['threshold'] == .35
    x, y, meta = load_development(root)
    n = bundle['numeric_output_count']
    z = prep.transform(x)[:, :n]
    names = prep.get_feature_names_out()[:n]
    assert x.shape == (1253,590) and y.sum() == 83 and n == 468
    assert list(names) == [f'measurement_{j+1:03d}' for j in prep.numeric_columns_]
    assert np.allclose(z, np.where(np.isnan(x[:,prep.numeric_columns_]), prep.medians_, x[:,prep.numeric_columns_]))
    assert np.isclose(model.feature_importances_.sum(),1)
    cache = resource('feature_interpretation/permutation_raw.npz')
    positive = list(model.classes_).index(1)
    with threadpool_limits(limits=1):
        if cache.exists():
            saved = np.load(cache)
            assert str(saved['model_sha256']) == before
            losses, baseline = saved['losses'], float(saved['baseline'])
        else:
            baseline = average_precision_score(y, model.predict_proba(z)[:,positive])
            losses = np.zeros((n,5))
            # Same five permutations for every feature: reproducible paired shuffle noise.
            rng = np.random.default_rng(42)
            permutations = [rng.permutation(len(y)) for _ in range(5)]
            modified = z.copy()
            for j in range(n):
                for r, order in enumerate(permutations):
                    modified[:,j] = z[order,j]
                    losses[j,r] = baseline - average_precision_score(y,model.predict_proba(modified)[:,positive])
                modified[:,j] = z[:,j]
                if (j+1)%50 == 0:
                    print(f'Completed {j+1}/{n} measurements',flush=True)
            np.savez(cache,losses=losses,baseline=baseline,model_sha256=before)

    table = pd.DataFrame({'feature':names,'source_column_1based':prep.numeric_columns_+1,
        'builtin_importance':model.feature_importances_, 'permutation_AP_loss':losses.mean(axis=1),
        'permutation_shuffle_SD':losses.std(axis=1,ddof=1)})
    table['builtin_rank'] = table.builtin_importance.rank(method='min',ascending=False).astype(int)
    table['permutation_rank'] = table.permutation_AP_loss.rank(method='min',ascending=False).astype(int)
    ranks = np.argsort(np.argsort(-losses,axis=0),axis=0)+1
    table['shuffle_rank_min'] = ranks.min(axis=1)
    table['shuffle_rank_max'] = ranks.max(axis=1)
    table['top20_shuffles'] = (ranks<=20).sum(axis=1)
    for i,j in enumerate(prep.numeric_columns_):
        v = x[:,j]
        table.loc[i,'missing_pct'] = 100*np.isnan(v).mean()
        observed = v[~np.isnan(v)]
        q1,q3 = np.quantile(observed,[.25,.75])
        low,high = q1-1.5*(q3-q1),q3+1.5*(q3-q1)
        table.loc[i,'observed_tukey_outlier_pct'] = 100*((observed<low)|(observed>high)).mean()
        table.loc[i,'distinct_observed_values'] = len(np.unique(observed))
        for label, code in [('pass',0),('fail',1)]:
            group = v[y==code]
            obs = group[~np.isnan(group)]
            table.loc[i,label+'_missing_pct'] = 100*np.isnan(group).mean()
            table.loc[i,label+'_observed_n'] = len(obs)
            for name,value in zip(['q1','median','q3'],np.quantile(obs,[.25,.5,.75])):
                table.loc[i,label+'_'+name] = value
    table = table.sort_values('permutation_rank')
    table.to_csv(out/'all_feature_importances.csv',index=False)
    selected = table.head(20).copy()
    selected.to_csv(out/'top20_features.csv',index=False)
    # Pairwise observed values, across all retained inputs; no median artifacts.
    corr = pd.DataFrame(x[:,prep.numeric_columns_],columns=names).corr(method='spearman',min_periods=30)
    pairs=[]
    for name in selected.feature:
        partner = corr[name].drop(name).abs().idxmax()
        a,b = list(names).index(name),list(names).index(partner)
        pairs.append({'feature':name,'most_correlated_feature':partner,'spearman_rho':float(corr.loc[name,partner]),
            'pairwise_observed_n':int((~np.isnan(x[:,prep.numeric_columns_[a]]) & ~np.isnan(x[:,prep.numeric_columns_[b]])).sum())})
    pd.DataFrame(pairs).to_csv(out/'selected_correlations.csv',index=False)
    assert sha(model_path)==before and joblib.hash(bundle)==state_before
    summary={'model_sha256':before,'development_n':len(y),'development_failures':int(y.sum()),
     'evaluation':'IN-SAMPLE development reliance only; final estimator trained on these records; no held-out importance available without refitting',
     'training_average_precision_reference':baseline,'permutation_repeats':5,'seed':42,
     'top10_overlap':sorted(set(table.nsmallest(10,'builtin_rank').feature)&set(table.head(10).feature)),
     'top20_overlap':sorted(set(table.nsmallest(20,'builtin_rank').feature)&set(table.head(20).feature)),
     'builtin_top20':list(table.nsmallest(20,'builtin_rank').feature),
     'model_unchanged':True,'reserved_test_accessed':False,
     'notes':'Only development manifest rows parsed from shared raw files. No fit calls. Threshold remains .35. Shuffle SD is not a confidence interval.'}
    (out/'audit.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    print(selected.to_string(index=False))
    print(pd.DataFrame(pairs).to_string(index=False))


if __name__ == '__main__':
    main()
