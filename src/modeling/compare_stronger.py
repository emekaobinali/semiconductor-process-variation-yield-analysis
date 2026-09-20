from src.paths import ROOT, resource, report, require_file, protect_completed_evidence
from pathlib import Path
import csv,json,warnings,sys,hashlib,zipfile
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score,confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from src.preprocessing import MeasurementPreparation,load_development

def main():
    protect_completed_evidence()
    """One fixed comparison on saved development folds; no reserved test access."""

    project = ROOT
    audit=resource('stronger_comparison')
    output=ROOT / 'artifacts/reproduction'
    if audit.exists(): raise SystemExit('Refusing an unplanned repeat of this comparison.')
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            path=str(args[0]).replace('\\','/').lower()
            if any(path.endswith('/'+n) for n in ['reserved_test_rows.csv','split_record.json','summary.json']):
                raise RuntimeError('Reserved or mixed-partition artifact access blocked.')
    sys.addaudithook(guard)
    def models(name):
        if name=='logistic_C0.1': return LogisticRegression(C=0.1,solver='lbfgs',class_weight='balanced',max_iter=2000)
        if name=='small_forest': return RandomForestClassifier(n_estimators=100,max_depth=5,min_samples_leaf=10,max_features='sqrt',class_weight='balanced',random_state=42,n_jobs=1)
        return HistGradientBoostingClassifier(max_iter=100,learning_rate=0.1,max_leaf_nodes=7,max_depth=3,min_samples_leaf=20,l2_regularization=1,class_weight='balanced',early_stopping=False,random_state=42)
    names=['logistic_C0.1','small_forest','shallow_boosting']
    audit.mkdir()
    (resource('stronger_comparison/fixed_settings.json')).write_text(json.dumps({name:models(name).get_params() for name in names},indent=2),encoding='utf-8')
    x,y,meta=load_development(project)
    original=x.copy()
    with (resource('verification/development_validation_folds.csv')).open() as f:
        rows=list(csv.DictReader(f))
    mapping={int(r['source_row_1based']):int(r['validation_fold']) for r in rows}
    assert len(rows)==len(mapping)==len(x)==1253
    assert set(mapping)==set(meta['source_rows'])
    folds=np.array([mapping[r] for r in meta['source_rows']])
    assert set(folds)==set(range(1,6)) and y.sum()==83
    def metrics(actual,scores):
        tn,fp,fn,tp=map(int,confusion_matrix(actual,np.asarray(scores)>=0.5,labels=[0,1]).ravel())
        recall=tp/(tp+fn); specificity=tn/(tn+fp)
        return dict(AP=float(average_precision_score(actual,scores)),recall=recall,precision=tp/(tp+fp) if tp+fp else None,specificity=specificity,balanced_accuracy=(recall+specificity)/2,TN=tn,FP=fp,FN=fn,TP=tp)
    records=[]; predictions=[]; diagnostics=[]
    with threadpool_limits(limits=1):
        for fold in range(1,6):
            train=np.flatnonzero(folds!=fold); val=np.flatnonzero(folds==fold)
            for name in names:
                prep=MeasurementPreparation(scale_numeric=name=='logistic_C0.1')
                a=prep.fit_transform(x[train]);b=prep.transform(x[val]);n=len(prep.numeric_columns_)
                for flags in [False,True]:
                    model=models(name)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        model.fit(a if flags else a[:,:n],y[train])
                    if any(issubclass(w.category,ConvergenceWarning) for w in caught):
                        raise RuntimeError('Convergence failed. Stop before adjusting settings.')
                    scores=model.predict_proba(b if flags else b[:,:n])[:,list(model.classes_).index(1)]
                    assert np.isfinite(scores).all()
                    records.append(dict(model=name,indicators=flags,fold=fold,**metrics(y[val],scores)))
                    diagnostics.append(dict(model=name,indicators=flags,fold=fold,warnings=[str(w.message) for w in caught]))
                    predictions.extend(dict(model=name,indicators=flags,fold=fold,source_row_1based=meta['source_rows'][i],failure=int(y[i]),score=float(s),prediction=int(s>=0.5)) for i,s in zip(val,scores))
            print(f'Completed fixed fold {fold}/5',flush=True)
    np.testing.assert_array_equal(x,original)
    aggregate=[]
    for name in names:
        for flags in [False,True]:
            r=[z for z in records if z['model']==name and z['indicators']==flags]
            p=[z for z in predictions if z['model']==name and z['indicators']==flags]
            assert len(p)==1253 and len({z['source_row_1based'] for z in p})==1253
            m=metrics(np.array([z['failure'] for z in p]),np.array([z['score'] for z in p]));m.pop('AP')
            for key in ['TN','FP','FN','TP']:assert sum(z[key] for z in r)==m[key]
            aggregate.append(dict(model=name,indicators=flags,mean_AP=float(np.mean([z['AP'] for z in r])),AP_fold_sd=float(np.std([z['AP'] for z in r],ddof=1)),**m))
    baseline=json.loads((resource('baseline_comparison/aggregate.json')).read_text())
    combined=[dict(model=r['model'],indicators=r['indicators'],mean_AP=r['mean']['AP'],AP_fold_sd=r['fold_sd']['AP'],**r['pooled']) for r in baseline]+aggregate
    def save_csv(path,data):
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    save_csv(resource('stronger_comparison/development_oof_predictions.csv'),predictions)
    save_csv(output/'secom-stronger-model-fold-metrics.csv',records)
    save_csv(output/'secom-model-comparison-summary.csv',combined)
    (audit/'aggregate.json').write_text(json.dumps(aggregate,indent=2))
    (audit/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    print(json.dumps(aggregate,indent=2))
    def pct(v):return 'undefined' if v is None else f'{100*v:.1f}%'
    def name(r):return r['model']+(' + flags' if r['indicators'] else ' without flags')
    table='\n'.join(f"| {name(r)} | {r['mean_AP']:.4f} ± {r['AP_fold_sd']:.4f} | {pct(r['recall'])} | {pct(r['precision'])} | {r['FP']} | {r['FN']} | {pct(r['specificity'])} | {pct(r['balanced_accuracy'])} | {r['TP']} | {r['TN']} |" for r in combined)
    foldtable='\n'.join(f"| {name(r)} | {r['fold']} | {r['AP']:.4f} | {pct(r['recall'])} | {pct(r['precision'])} | {r['FP']} | {r['FN']} | {pct(r['specificity'])} | {pct(r['balanced_accuracy'])} |" for r in records)
    report=f'''# SECOM: fixed stronger-model comparison

    Scope: development data only, existing five folds, 1,253 records including 83 failures. Threshold is fixed at 0.50 in all configurations. Thirty new fits were performed (three models x two indicator variants x five folds). Existing reference, logistic C=1, and shallow-tree results are reused, not refitted. No search, threshold optimization, early stopping, advanced feature selection, or full-development predictive refit. The reserved test set was not parsed or evaluated.

    ## Settings and rationale

    - Logistic C=0.1: tenfold stronger L2 coefficient penalty than the existing C=1 logistic baseline; lbfgs, balanced class weights, max_iter=2000. Standardize numeric inputs only. A linear, directly inspectable baseline with stronger protection against fitting noise.
    - Small forest: 100 trees, max_depth=5, min_samples_leaf=10, max_features=sqrt, balanced class weights, seed 42. Aggregates constrained nonlinear trees. Less directly interpretable than a single tree; explanations would require separate checks.
    - Shallow boosting: histogram gradient boosting, 100 rounds, learning_rate=0.1, max_depth=3, max_leaf_nodes=7, min_samples_leaf=20, l2_regularization=1, balanced class weights, no early stopping, seed 42. Justified as a second constrained nonlinear approach whose sequential corrections differ from the forest's averaging; explainable but not intrinsically transparent.

    Each fit uses the approved training-only constant filtering, median imputation, and optional missingness flags. Numeric scaling applies only to logistic regression. Labels inform class weights only within training. No class resampling. Timestamps and row IDs are not predictive inputs. The loader skips non-development lines before parsing; reserved manifest access is blocked. Retained flags are binary and unscaled.

    ## Validation comparison

    AP is the unweighted mean ± sample standard deviation across validation folds. Remaining rates and counts pool out-of-fold decisions, so each record contributes once per configuration. SD is variability, not a confidence interval. FP means passing records flagged; FN means failures missed. TP means failures detected; TN means passes correctly unflagged. Undefined precision means nothing was flagged.

    | Model | Mean AP ± SD | Recall | Precision | False alarms | Missed failures | Specificity | Balanced accuracy | TP | TN |
    |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
    {table}

    ## New-model fold results

    | Model | Fold | AP | Recall | Precision | FP | FN | Specificity | Balanced accuracy |
    |---|---:|---:|---:|---:|---:|---:|---:|---:|
    {foldtable}

    ## Boundaries

    All results are development comparisons after previous development exploration, not independent final performance estimates. An operating false-alarm budget has not been agreed, so no numerical increase can be declared operationally reasonable without that decision. Scores from weighted models are not established as calibrated probabilities; identical thresholds do not impose identical review budgets. No causal interpretation or physical feature meanings are inferred. Entity dependence and measurement timing remain unresolved.

    Verification: each configuration has exactly one saved validation prediction per development record; fold confusion counts sum to pooled counts; all scores are finite and raw development values are unchanged. Exact settings, predictions, and warnings are saved locally for audit.
    '''
    (output/'secom-stronger-model-comparison.md').write_text(report,encoding='utf-8')
    with zipfile.ZipFile(output/'secom-stronger-model-source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for file in ['compare_stronger.py','preprocessing.py','requirements.txt']:z.write(resource(file),file)
        for file in ['aggregate.json','fixed_settings.json','diagnostics.json']:z.write(audit/file,'results/'+file)


if __name__ == '__main__':
    main()
