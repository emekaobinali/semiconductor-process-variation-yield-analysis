from src.paths import ROOT, resource, report, require_file, protect_completed_evidence
from pathlib import Path
import csv, json, sys, warnings, zipfile
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from src.preprocessing import load_development, MeasurementPreparation

def main():
    protect_completed_evidence()
    """Single fixed development-only baseline comparison; no search or refitting."""

    project = ROOT
    output=ROOT / 'artifacts/reproduction'
    audit=resource('baseline_comparison')
    if audit.exists():
        raise SystemExit('Comparison already exists. Refusing an unplanned repeat.')
    audit.mkdir()
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            p=str(args[0]).replace('\\','/').lower()
            if any(p.endswith('/'+n) for n in ['reserved_test_rows.csv','split_record.json','summary.json']):
                raise RuntimeError('Reserved/mixed-partition artifact access prohibited.')
    sys.addaudithook(guard)
    settings={
        'reference':'DummyClassifier(strategy=prior); same fold prevalence score for every record',
        'logistic':'LogisticRegression(C=1, solver=lbfgs, class_weight=balanced, max_iter=2000); numeric standard scaling; unscaled binary indicators',
        'tree':'DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, class_weight=balanced, random_state=42); no scaling',
        'threshold':0.5,'folds':'existing saved five-fold development assignments',
        'selection':'No tuning, searches, calibration, feature ranking, threshold optimization, or full-development predictive refit.',
        'indicator_comparison':'Same fitted training-only preprocessing; omit only appended indicator columns in without condition.',
    }
    (resource('baseline_comparison/fixed_settings.json')).write_text(json.dumps(settings,indent=2))
    x,y,meta=load_development(project)
    original=x.copy()
    with (resource('verification/development_validation_folds.csv')).open() as f:
        assignments=list(csv.DictReader(f))
    assert len(assignments)==len(x)
    mapping={int(r['source_row_1based']):int(r['validation_fold']) for r in assignments}
    assert set(mapping)==set(meta['source_rows']) and set(mapping.values())==set(range(1,6))
    fold_ids=np.array([mapping[i] for i in meta['source_rows']])
    def metrics(labels,scores):
        pred=(scores>=0.5).astype(int)
        tn,fp,fn,tp=map(int,confusion_matrix(labels,pred,labels=[0,1]).ravel())
        recall=tp/(tp+fn); specificity=tn/(tn+fp)
        return dict(AP=float(average_precision_score(labels,scores)),recall=recall,
            precision=tp/(tp+fp) if tp+fp else None,specificity=specificity,
            balanced_accuracy=(recall+specificity)/2,TN=tn,FP=fp,FN=fn,TP=tp)
    results=[]; predictions=[]; diagnostics=[]
    with threadpool_limits(limits=1):
        for fold in range(1,6):
            train=np.flatnonzero(fold_ids!=fold); val=np.flatnonzero(fold_ids==fold)
            for name in ['reference','logistic','tree']:
                prep=MeasurementPreparation(scale_numeric=name=='logistic')
                a=prep.fit_transform(x[train]); b=prep.transform(x[val])
                n=len(prep.numeric_columns_)
                for flags in [False,True]:
                    a_use=a if flags else a[:,:n]; b_use=b if flags else b[:,:n]
                    if name=='reference': model=DummyClassifier(strategy='prior')
                    elif name=='logistic': model=LogisticRegression(C=1,solver='lbfgs',class_weight='balanced',max_iter=2000)
                    else: model=DecisionTreeClassifier(max_depth=3,min_samples_leaf=20,class_weight='balanced',random_state=42)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        model.fit(a_use,y[train])
                    if any(issubclass(w.category,ConvergenceWarning) for w in caught):
                        raise RuntimeError('Logistic convergence failed; report failure before any adjustment.')
                    scores=model.predict_proba(b_use)[:,list(model.classes_).index(1)]
                    assert np.isfinite(scores).all()
                    result=dict(model=name,indicators=flags,fold=fold,validation_records=len(val),**metrics(y[val],scores))
                    results.append(result)
                    diagnostics.append(dict(model=name,indicators=flags,fold=fold,columns=a_use.shape[1],warnings=[str(w.message) for w in caught],iterations=getattr(model,'n_iter_',np.array([])).tolist(),indicator_split_nodes=int(np.sum(model.tree_.feature>=n)) if name=='tree' else None))
                    predictions.extend(dict(model=name,indicators=flags,fold=fold,source_row_1based=meta['source_rows'][i],failure=int(y[i]),score=float(score),prediction=int(score>=0.5)) for i,score in zip(val,scores))
            print(f'Completed saved fold {fold}/5',flush=True)
    np.testing.assert_array_equal(x,original)
    aggregate=[]
    for name in ['reference','logistic','tree']:
        for flags in [False,True]:
            records=[r for r in results if r['model']==name and r['indicators']==flags]
            p=[r for r in predictions if r['model']==name and r['indicators']==flags]
            assert len(p)==1253 and len({r['source_row_1based'] for r in p})==1253
            pooled=metrics(np.array([r['failure'] for r in p]),np.array([r['score'] for r in p]))
            # Pool confusion counts, but summarize AP within folds because score scales
            # can differ between separately fitted models.
            pooled.pop('AP')
            means={key:float(np.mean([r[key] for r in records])) if all(r[key] is not None for r in records) else None for key in ['AP','recall','precision','specificity','balanced_accuracy']}
            sds={key:float(np.std([r[key] for r in records],ddof=1)) if means[key] is not None else None for key in means}
            aggregate.append(dict(model=name,indicators=flags,mean=means,fold_sd=sds,pooled=pooled))
    for name in ['reference','logistic','tree']:
        paired=[next(r['AP'] for r in results if r['model']==name and r['fold']==f and r['indicators'])-next(r['AP'] for r in results if r['model']==name and r['fold']==f and not r['indicators']) for f in range(1,6)]
        print(name,'paired AP changes',paired,flush=True)
    def save_csv(path,records):
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    save_csv(audit/'fold_metrics.csv',results)
    save_csv(resource('baseline_comparison/development_oof_predictions.csv'),predictions)
    (audit/'aggregate.json').write_text(json.dumps(aggregate,indent=2))
    (audit/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    print(json.dumps(aggregate,indent=2))
    def percent(v): return 'undefined' if v is None else f'{v*100:.2f}%'
    def label(r):return r['model']+(' + indicators' if r['indicators'] else ' without indicators')
    summary_table='\n'.join(f"| {label(r)} | {r['mean']['AP']:.4f} ± {r['fold_sd']['AP']:.4f} | {percent(r['pooled']['recall'])} | {percent(r['pooled']['precision'])} | {percent(r['pooled']['specificity'])} | {percent(r['pooled']['balanced_accuracy'])} | {r['pooled']['TN']} | {r['pooled']['FP']} | {r['pooled']['FN']} | {r['pooled']['TP']} |" for r in aggregate)
    fold_table='\n'.join(f"| {label(r)} | {r['fold']} | {r['AP']:.4f} | {percent(r['recall'])} | {percent(r['precision'])} | {percent(r['specificity'])} | {percent(r['balanced_accuracy'])} | {r['TN']} | {r['FP']} | {r['FN']} | {r['TP']} |" for r in results)
    report=f'''# SECOM baseline comparison — development only

    1,253 development records (83 failures, 1,170 passes), using the existing saved five-fold assignments. All preprocessing was fitted on each fold's training records. Failure is the positive class. The fixed decision threshold is 0.5. No reserved test records were parsed or evaluated. The loader streams past excluded raw lines without tokenizing them. No search, threshold tuning, calibration, or final predictive model refit was performed.

    ## Fixed models

    - No-skill reference: assigns each record its training fold's failure prevalence; predicts pass at 0.5.
    - Logistic regression: C=1, L2 regularization under the installed defaults, lbfgs solver, balanced class weights, maximum 2,000 iterations. Numeric measurements standardized using training data; missingness flags unscaled. Regularization constrains coefficients to limit overfitting; coefficients are not physical causes.
    - Shallow decision tree: maximum depth 3, minimum 20 training records per leaf, balanced class weights, seed 42. No scaling. Depth and leaf constraints limit complexity but do not eliminate instability.

    Each model was evaluated with and without varying training missingness flags. Numeric inputs and train-only medians are identical within each paired comparison. No resampling occurred. Class weights use training outcomes only. Weighted scores are not established as calibrated failure probabilities.

    ## Aggregate validation results

    AP is the unweighted mean ± sample standard deviation across the five validation folds. Other metrics below use summed out-of-fold confusion counts; each development record contributes once per configuration. AP measures ranking quality; threshold metrics measure the fixed 0.5 decision rule. SD describes fold variation, not a confidence interval: folds share training records. No independent-test or population significance claim is made. Precision is undefined when no records are flagged, not artificially reported as zero.

    | Configuration | Mean AP ± fold SD | Recall | Precision | Specificity | Balanced accuracy | TN | FP | FN | TP |
    |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
    {summary_table}

    ## Individual validation folds

    Each fold contains 16 or 17 failures, so recall is sensitive to individual cases. TN: passing records correctly passed. FP: passing records flagged. FN: failures missed. TP: failures detected.

    | Configuration | Fold | AP | Recall | Precision | Specificity | Balanced accuracy | TN | FP | FN | TP |
    |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
    {fold_table}

    ## Interpretation boundaries

    Missingness contributions are model-dependent paired comparisons, not proof of physical mechanisms or generalization beyond this dataset. Primary ranking comparisons use mean within-fold AP, not cross-fold pooled score rankings. Reported performance is exploratory development validation; subsequent model choices would make it selection evidence rather than an unbiased final estimate. Unknown entity dependence and measurement acquisition timing remain limitations. This run stops at baseline comparison.
    '''
    (output/'secom-baseline-comparison.md').write_text(report)
    save_csv(output/'secom-baseline-fold-metrics.csv',results)
    with zipfile.ZipFile(output/'secom-baseline-reproducibility.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['compare_baselines.py','preprocessing.py','requirements.txt']:
            z.write(resource(name),name)
        for name in ['fixed_settings.json','fold_metrics.csv','aggregate.json','diagnostics.json']:
            z.write(audit/name,'results/'+name)


if __name__ == '__main__':
    main()
