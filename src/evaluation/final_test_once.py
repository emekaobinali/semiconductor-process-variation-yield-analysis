from src.paths import ROOT, resource, report, require_file, protect_completed_evidence
from pathlib import Path
from datetime import datetime,timezone
import csv,json,hashlib,platform,pickle,warnings,math
from importlib.metadata import version
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score,confusion_matrix
from threadpoolctl import threadpool_limits
from src.preprocessing import MeasurementPreparation,load_development

def main():
    protect_completed_evidence()
    raise SystemExit('Final test already completed. Replay is disabled; use the saved final report.')
    """User-approved frozen final fit and single reserved-test evaluation.

    Refuses to run if a final evaluation directory exists. Recovery after an error
    must inspect the saved stage record, never blindly refit or rescore.
    """

    project = ROOT
    out=ROOT / 'artifacts/reproduction'
    final=resource('final_evaluation')
    if final.exists(): raise SystemExit('Final evaluation already initiated. Stop; do not rerun.')
    def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    def now():return datetime.now(timezone.utc).isoformat()
    def write_json(p,value):
        with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
    params=json.loads((resource('stronger_comparison/fixed_settings.json')).read_text())['small_forest']
    split=json.loads((resource('splits/split_record.json')).read_text())
    versions={}
    for requirement in (resource('requirements.txt')).read_text().splitlines():
        package,expected=requirement.split('==')
        versions[package]=version(package)
        if versions[package]!=expected:raise RuntimeError(f'Pinned version mismatch: {package}, {versions[package]} versus {expected}')
    assert params['random_state']==42 and params['n_estimators']==100 and params['max_depth']==5 and params['min_samples_leaf']==10
    assert params['class_weight']=='balanced' and params['max_features']=='sqrt'
    dev_manifest=resource('splits/development_rows.csv')
    assert digest(dev_manifest)==split['development']['manifest_sha256']
    final.mkdir()
    freeze={
        'frozen_utc':now(),'authorization':'User approved Random Forest, no missingness indicators, threshold 0.35, full-development fit and one reserved-test evaluation.',
        'classifier':'RandomForestClassifier','model_parameters':params,
        'threshold':0.35,'decision_rule':'failure score >= 0.35',
        'failure_class':1,'source_labels':{'-1':'pass','1':'failure'},
        'preprocessing':'Original 590-column order. Fit MeasurementPreparation(scale_numeric=False) on development only; retain columns with >1 distinct observed development value; median-impute from development; pass only numeric columns to classifier. All computed missingness indicators are excluded by the same output slice used in validation.',
        'scaling':False,'missingness_indicators_in_classifier':False,
        'feature_ranking_or_selection':'Only approved development-derived constant/all-missing filtering; no additional selection.',
        'class_weighting':'balanced, computed from full-development outcomes by the estimator',
        'development_manifest_sha256':split['development']['manifest_sha256'],
        'reserved_manifest_expected_sha256':split['reserved_test']['manifest_sha256'],
        'expected_source_hashes':split['raw_files'],
        'preprocessing_source_sha256':digest(resource('preprocessing.py')),
        'evaluation_script_sha256':digest(Path(__file__)),
        'versions':versions,'python':platform.python_version(),
        'evaluation':'One reserved-set predict_proba call after fitting and persisting model; no calibration, tuning, threshold revision, or subsequent fit.',
    }
    write_json(final/'frozen_method.json',freeze)
    write_json(out/'secom-frozen-method.json',freeze)
    print('Method frozen and saved before reserved-test access.',flush=True)
    stage_path=final/'events.jsonl'
    def event(stage):
        with stage_path.open('a',encoding='utf-8') as f:f.write(json.dumps({'utc':now(),'stage':stage})+'\n')
    event('method_frozen')
    x,y,meta=load_development(project)
    assert x.shape==(1253,590) and int(y.sum())==83
    prep=MeasurementPreparation(scale_numeric=False)
    train_prepared=prep.fit_transform(x)
    n_numeric=len(prep.numeric_columns_)
    train_numeric=train_prepared[:,:n_numeric]
    assert np.isfinite(train_numeric).all() and n_numeric==468
    model=RandomForestClassifier(**params)
    event('development_fit_started')
    with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(train_numeric,y)
    bundle={'preprocessing':prep,'numeric_output_count':n_numeric,'classifier':model,'threshold':0.35,'freeze':freeze}
    joblib.dump(bundle,final/'frozen_fitted_model.joblib')
    state_before=hashlib.sha256(pickle.dumps(bundle)).hexdigest()
    write_json(final/'fitted_state.json',{'utc':now(),'development_records':len(x),'numeric_inputs':n_numeric,'constant_numeric_inputs_excluded':len(prep.constant_columns_),'all_missing_numeric_inputs_excluded':len(prep.all_missing_columns_),'retained_measurements':prep.get_feature_names_out()[:n_numeric].tolist(),'training_medians':prep.medians_.tolist(),'warnings':[str(w.message) for w in caught],'model_file_sha256':digest(final/'frozen_fitted_model.joblib')})
    event('development_fit_persisted')
    print('Frozen pipeline fitted once on 1,253 development records and saved.',flush=True)

    # Reserved records are first parsed only after the method and fitted model exist.
    test_manifest=resource('splits/reserved_test_rows.csv')
    assert digest(test_manifest)==split['reserved_test']['manifest_sha256']
    with test_manifest.open(newline='') as f:ids=[int(r['source_row_1based']) for r in csv.DictReader(f)]
    assert len(ids)==len(set(ids))==314
    assert not set(ids)&set(meta['source_rows'])
    assert set(ids)|set(meta['source_rows'])==set(range(1,1568))
    raw=resource('inspection/raw')
    for name,entry in split['raw_files'].items():assert digest(raw/name)==entry['sha256']
    event('reserved_records_opened')
    wanted=set(ids);values={};labels={}
    with (raw/'secom.data').open() as f:
        for row_id,line in enumerate(f,1):
            if row_id in wanted:
                row=[float(v) for v in line.split()]
                assert len(row)==590
                values[row_id]=row
    with (raw/'secom_labels.data').open() as f:
        for row_id,line in enumerate(f,1):
            if row_id in wanted:
                label=int(line.split(maxsplit=1)[0]);assert label in (-1,1)
                labels[row_id]=int(label==1)
    test_x=np.asarray([values[i] for i in ids]);test_y=np.asarray([labels[i] for i in ids])
    assert test_x.shape==(314,590) and int(test_y.sum())==21
    test_numeric=prep.transform(test_x)[:,:n_numeric]
    assert np.isfinite(test_numeric).all()
    event('single_test_scoring_started')
    with threadpool_limits(limits=1):
        scores=model.predict_proba(test_numeric)[:,list(model.classes_).index(1)]
    pred=(scores>=freeze['threshold']).astype(int)
    with (final/'reserved_test_predictions.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['source_row_1based','actual_failure','failure_score','predicted_failure'])
        w.writerows((i,int(actual),float(score),int(decision)) for i,actual,score,decision in zip(ids,test_y,scores,pred))
    event('single_test_predictions_persisted')
    assert hashlib.sha256(pickle.dumps(bundle)).hexdigest()==state_before
    assert digest(resource('preprocessing.py'))==freeze['preprocessing_source_sha256']
    tn,fp,fn,tp=map(int,confusion_matrix(test_y,pred,labels=[0,1]).ravel())
    recall=tp/(tp+fn);precision=tp/(tp+fp) if tp+fp else None;specificity=tn/(tn+fp)
    def wilson(k,n):
        z=1.959963984540054;p=k/n;den=1+z*z/n
        mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        return [mid-half,mid+half]
    metrics={'records':314,'failures':21,'passes':293,'TP':tp,'FN':fn,'FP':fp,'TN':tn,'flagged':tp+fp,'recall':recall,'precision':precision,'specificity':specificity,'balanced_accuracy':(recall+specificity)/2,'average_precision':float(average_precision_score(test_y,scores)),'recall_95pct_wilson_interval':wilson(tp,tp+fn),'recall_interval_assumption':'Independent Bernoulli outcomes; unknown production dependencies may invalidate nominal coverage. Conditional on this fitted model; does not include training/selection uncertainty.'}
    write_json(final/'reserved_test_metrics.json',metrics)
    write_json(out/'secom-final-test-metrics.json',metrics)
    event('evaluation_complete_method_unchanged')

    # Compare to the previously saved development summaries, not training-set scores.
    with (report('secom-intermediate-thresholds.csv')).open() as f:
        dev=next(r for r in csv.DictReader(f) if r['model']=='small_forest' and r['indicators']=='False' and float(r['threshold'])==0.35)
    dev_ap=next(r for r in json.loads((resource('stronger_comparison/aggregate.json')).read_text()) if r['model']=='small_forest' and not r['indicators'])
    def pct(v):return 'undefined' if v is None else f'{v*100:.1f}%'
    report=f'''# SECOM frozen final test evaluation

    Approved final method: Random Forest without missingness indicators; threshold 0.35. The complete method was saved before reserved-test access, then fitted once on all 1,253 development records. The original preprocessing code, saved forest parameters, and pinned packages were reused unchanged. The test set was scored once after the fitted method was persisted. No tuning or revisions followed the result.

    ## Frozen method

    100 trees; maximum depth 5; minimum 10 records per leaf; sqrt feature sampling; balanced class weights; bootstrap enabled; seed 42. Complete remaining settings are in the canonical config/frozen_method.json record. Original 590-column order; development-derived constant/all-missing filtering; development medians; no scaling; no indicator columns supplied to the classifier. The development fit retained {n_numeric} numeric inputs. All test transformations used frozen development parameters. No records removed.

    ## Final results compared with development validation

    | Measure | Development out-of-fold | Reserved test |
    |---|---:|---:|
    | Records | 1,253 | 314 |
    | Actual failures | 83 | 21 |
    | Actual passes | 1,170 | 293 |
    | Failures caught (TP) | {dev['TP']} | {tp} |
    | Failures missed (FN) | {dev['FN']} | {fn} |
    | False alarms (FP) | {dev['FP']} | {fp} |
    | Correctly identified passes (TN) | {dev['TN']} | {tn} |
    | Total flagged | {dev['flagged']} | {tp+fp} |
    | Failure recall | {pct(float(dev['recall']))} | {pct(recall)} |
    | Precision | {pct(float(dev['precision']))} | {pct(precision)} |
    | Specificity | {pct(float(dev['specificity']))} | {pct(specificity)} |
    | Balanced accuracy | {pct(float(dev['balanced_accuracy']))} | {pct(metrics['balanced_accuracy'])} |
    | Average precision | {dev_ap['mean_AP']:.4f} | {metrics['average_precision']:.4f} |

    Development AP is the mean of five validation-fold AP values (fold SD {dev_ap['AP_fold_sd']:.4f}); test AP is calculated once across the reserved set. Other development rates pool held-out validation decisions. Development results were used for selection, whereas this test was reserved. Different sample sizes make raw count comparisons inappropriate as evidence of improvement; compare rates. The final model is trained on more records than each validation-fold model.

    ## Confusion matrix

    Rows are actual outcomes; columns are frozen model decisions.

    | Actual outcome | Predicted pass | Flagged failure |
    |---|---:|---:|
    | Pass | {tn} | {fp} |
    | Failure | {fn} | {tp} |

    ## Uncertainty and integrity

    There are only 21 test failures; one detection changes recall by 4.76 percentage points. The approximate 95% Wilson interval for test recall is {pct(metrics['recall_95pct_wilson_interval'][0])} to {pct(metrics['recall_95pct_wilson_interval'][1])}, assuming independent records. Unknown production dependence may affect coverage. This interval does not include training or model-selection uncertainty. Scores are not established as calibrated probabilities; associations do not establish physical root causes or deployment readiness.

    Original file and split hashes matched; partition IDs were unique, disjoint and complete; all prepared values were finite; classifier/preprocessing state was unchanged by test transformation and scoring; test scores were persisted before reporting. Model training warnings: {[str(w.message) for w in caught]}. No final method settings were changed. The reserved test is now used and must not become a tuning set. No GitHub, app, deployment, or publishing work was started.
    '''
    (out/'secom-final-test-evaluation.md').write_text(report,encoding='utf-8')
    print(json.dumps(metrics,indent=2),flush=True)


if __name__ == '__main__':
    main()
