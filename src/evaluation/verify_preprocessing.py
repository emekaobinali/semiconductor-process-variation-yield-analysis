from src.paths import ROOT, resource, report, require_file, protect_completed_evidence
from pathlib import Path
import csv
import hashlib
import io
import json
import pickle
import platform
import unittest
import zipfile
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from src.preprocessing import load_development, make_preprocessing_pipeline
import sys
import scipy,joblib,threadpoolctl

def main():
    protect_completed_evidence()
    if resource("verification").exists():
        raise SystemExit("Preprocessing verification is already archived. Run synthetic tests instead.")
    """Verify preprocessing only, using the fixed development membership."""

    project = ROOT
    workspace = ROOT
    audit_dir = resource('verification')
    audit_dir.mkdir(exist_ok=True)

    # Guard all Python-level file opens: neither the reserved manifest nor the mixed
    # inspection summary is permitted in this run. The raw text loader must skip
    # excluded source lines before tokenization, as separately verified by unit tests.
    forbidden = ('reserved_test_rows.csv', 'split_record.json', 'summary.json')
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = str(args[0]).replace('\\','/').lower()
            if any(path.endswith('/'+name) for name in forbidden):
                raise RuntimeError('Attempted access to a reserved or mixed-partition artifact.')
    sys.addaudithook(guard)

    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_preprocessing.py', top_level_dir=str(ROOT))
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    print(log.getvalue())
    if not result.wasSuccessful():
        raise SystemExit('Preprocessing tests failed; no verification report produced.')

    x, y, metadata = load_development(project)
    assert x.shape == (1253,590)
    assert len(y) == 1253
    original_x = x.copy()
    mask = metadata['missing_mask'].copy()
    prototype = make_preprocessing_pipeline(scale_numeric=False)
    rows = []
    fold_assignments = []

    def audit_fit(pipeline, train_x, validation_x):
        pipeline.fit(train_x)
        fitted = pipeline['measurements']
        before = pickle.dumps(pipeline)
        train_z = pipeline.transform(train_x)
        val_z = pipeline.transform(validation_x)
        assert pickle.dumps(pipeline) == before
        assert np.isfinite(train_z).all() and np.isfinite(val_z).all()
        assert len(fitted.get_feature_names_out()) == train_z.shape[1] == val_z.shape[1]
        assert len(set(fitted.get_feature_names_out())) == train_z.shape[1]
        expected_medians = np.nanmedian(train_x[:,fitted.numeric_columns_],axis=0)
        np.testing.assert_array_equal(fitted.medians_,expected_medians)
        n_numeric = len(fitted.numeric_columns_)
        np.testing.assert_array_equal(val_z[:,n_numeric:],np.isnan(validation_x[:,fitted.indicator_columns_]).astype(float))
        # Every excluded numeric column has <=1 distinct observed training value.
        for j in np.r_[fitted.constant_columns_,fitted.all_missing_columns_]:
            assert len(np.unique(train_x[:,j][~np.isnan(train_x[:,j])])) <= 1
        return fitted, {
            'training_records':len(train_x), 'validation_records':len(validation_x),
            'numeric_measurements':n_numeric,
            'missingness_indicators':len(fitted.indicator_columns_),
            'output_columns':train_z.shape[1],
            'constant_observed_measurements':len(fitted.constant_columns_),
            'all_missing_measurements':len(fitted.all_missing_columns_),
            'validation_transform_changed_fitted_state':False,
            'nonfinite_output_values':0,
        }

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (training, validation) in enumerate(folds.split(x,y),1):
        assert not set(training) & set(validation)
        p = clone(prototype)
        fitted, info = audit_fit(p,x[training],x[validation])
        info.update(fold=fold, training_failures=int(y[training].sum()), validation_failures=int(y[validation].sum()))
        rows.append(info)
        fold_assignments.extend({'source_row_1based':metadata['source_rows'][int(i)],'validation_fold':fold} for i in validation)
        # Also exercise optional scaling on actual development folds, independently.
        scaled = make_preprocessing_pipeline(scale_numeric=True)
        audit_fit(scaled,x[training],x[validation])

    assert len(fold_assignments) == len(x)
    assert len({r['source_row_1based'] for r in fold_assignments}) == len(x)

    # Full-development fit is a structural audit, not a replacement for fold fitting.
    full, summary = audit_fit(clone(prototype),x,x)
    summary.pop('validation_records')
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(metadata['missing_mask'],mask)
    feature_audit = []
    numeric_set,indicator_set = set(full.numeric_columns_),set(full.indicator_columns_)
    allmissing_set = set(full.all_missing_columns_)
    for j in range(590):
        feature_audit.append({
            'measurement':f'measurement_{j+1:03d}',
            'observed_development_records':int((~mask[:,j]).sum()),
            'missing_development_records':int(mask[:,j].sum()),
            'numeric_input_retained':j in numeric_set,
            'numeric_exclusion_reason':'' if j in numeric_set else ('all_missing_in_training' if j in allmissing_set else 'constant_observed_in_training'),
            'missingness_flag_retained':j in indicator_set,
        })
    for filename, records in [('feature_audit.csv',feature_audit),('development_validation_folds.csv',fold_assignments)]:
        with (audit_dir/filename).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    summary.update({
        'unit_tests_passed':result.testsRun,
        'original_development_values_unchanged':True,
        'full_missing_mask_preserved':True,
        'timestamps_preserved_as_metadata':True,
        'scaling_default':'off; optional numeric-only StandardScaler verified',
        'predictive_models_trained':0,
        'reserved_test_records_parsed':0,
        'reserved_artifact_access':'blocked by runtime guard',
        'fold_checks':rows,
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'scikit_learn':sklearn.__version__},
        'source_code_sha256':hashlib.sha256((resource('preprocessing.py')).read_bytes()).hexdigest(),
    })
    (audit_dir/'verification.json').write_text(json.dumps(summary,indent=2))
    (audit_dir/'unit_tests.txt').write_text(log.getvalue())

    versions={'numpy':np.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__,'joblib':joblib.__version__,'threadpoolctl':threadpoolctl.__version__}
    (resource('requirements.txt')).write_text(''.join(f'{key}=={value}\n' for key,value in versions.items()))

    readme = '''# SECOM development-only preprocessing

    No predictive model is included. The loader uses only the saved development row membership. It never opens the reserved manifest. Because the original raw files contain both partitions, it streams past non-development lines without tokenizing, recording, or evaluating their values. A synthetic fixture with malformed excluded lines tests this behavior. Timestamps and row identifiers remain metadata outside model inputs.

    Install requirements.txt in a project environment. Use data/raw for downloaded source files and data/splits for saved manifests. Run synthetic tests from the repository root with `python -m unittest discover -s tests -t . -v`. The evidence package omits source data, record memberships, dependencies, and reserved artifacts. Restore only the approved development manifest for a local rerun. No refitting on the test set is authorized.

    Use make_preprocessing_pipeline(scale_numeric=False). Each cross-validation fold requires a fresh clone fitted on that fold's training rows. Do not preprocess all development data before cross-validation. The full-development fit in verification is a structural check only; no fitted object or preprocessed full-development table is supplied for cross-validation reuse.

    Fit retains measurements with at least two distinct observed training values; medians are computed only for these measurements. It retains missingness indicators that vary within training, including indicators of excluded constant-observed measurements. All-missing and always-observed flags are constant and omitted from model inputs, but the complete original missingness mask is returned separately for audit. Later unexpected missingness is imputed using the frozen training median without adding a new output column. The usefulness of flags has not been evaluated; later compare with/without flags using development data only.

    Scaling is optional and off by default until model selection warrants it. If enabled, StandardScaler fits on imputed training numeric measurements only; binary flags remain unscaled. For scientific distributions and variation summaries use original observed measurements, never imputed/scaled outputs as experimental observations. No row deletion, near-constant cutoff, high-missingness cutoff, feature ranking, outcome associations, synthetic oversampling, or threshold tuning is implemented.

    Input contract: numeric arrays must retain the original 590-column order. The loader enforces width and builds that order; the transformer validates dimensionality, fitted width, and infinite values. An arbitrary external array with permuted columns of the same width cannot be detected automatically. Feature names are positional identifiers, not physical descriptions.

    Every unit-test fixture is synthetic software test data. It is not experimental or simulated semiconductor evidence. Public source: McCann and Johnston (2008), SECOM, UCI Machine Learning Repository, https://doi.org/10.24432/C54305 (CC BY 4.0).
    '''
    (resource('README.md')).write_text(readme)
    output=ROOT / 'artifacts/reproduction'; output.mkdir(exist_ok=True)
    fold_text='\n'.join(f"| {r['fold']} | {r['training_records']} | {r['validation_records']} | {r['numeric_measurements']} | {r['missingness_indicators']} | {r['output_columns']} |" for r in rows)
    report=f'''# SECOM preprocessing verification

    Implemented and verified on 1,253 development records only. No predictive models trained; no reserved test records parsed or evaluated.

    ## Preparation behavior

    All 590 raw measurement columns, the original missing-value mask, and timestamps are preserved. Timestamps and row identifiers remain metadata. Numeric inputs are excluded only when they have zero or one distinct observed value in the fitted training subset. Missingness flags that vary in training are retained, including flags for excluded constant measurements. Retained numeric columns use training-only median imputation. Optional numeric-only standard scaling was verified; it is off by default. No row deletion, missingness threshold, nearly constant filtering, or feature ranking was performed.

    ## Full-development structural check

    - Retained numeric measurements: **{summary['numeric_measurements']}**.
    - Retained missingness indicators: **{summary['missingness_indicators']}**.
    - Total prepared columns: **{summary['output_columns']}**.
    - Excluded constant-observed numeric inputs: **{summary['constant_observed_measurements']}**.
    - Excluded all-missing numeric inputs: **{summary['all_missing_measurements']}**.
    - Nonfinite prepared values: **0**.

    These are preprocessing counts, not model results. The full-development fitted object is not saved for cross-validation reuse.

    ## Five-fold isolation checks

    Each fold used a new preprocessing pipeline, fitted only on its training records. Validation records were transformed with frozen training parameters. Both scaling-off and scaling-on paths were checked.

    | Fold | Training records | Validation records | Numeric inputs | Missingness flags | Total inputs |
    |---|---:|---:|---:|---:|---:|
    {fold_text}

    ## Verification and limits

    All **{result.testsRun} automated tests passed**. They cover medians, missingness preservation, constant/all-missing filtering, zero handling, unchanged raw values, frozen transformation state, numeric-only scaling, independent fold fits, invalid inputs, and ignoring excluded records. All five development validation folds produced finite outputs; each development record appeared in exactly one validation fold. No measurement associations, model metrics, or test results were calculated.

    The loader streams the original mixed-partition text files but skips non-development lines before parsing. Reserved manifest access was blocked during verification. No test subset was materialized. The full raw missingness mask remains available even where constant flags are excluded from model inputs.

    Missingness flags have **not** been shown to improve prediction; that remains a later development-only comparison. Unknown measurement timing, entity dependence, and missingness mechanisms remain unresolved. Anonymous column order must be preserved. Optional scaling is an implementation capability, not a model-selection decision.

    Source: [UCI SECOM](https://doi.org/10.24432/C54305). Reusable implementation, tests, pinned dependencies, and detailed verification counts are in the companion source package. This step stops before predictive modeling.
    '''
    (output/'secom-preprocessing-verification.md').write_text(report)
    with zipfile.ZipFile(output/'secom-preprocessing-source.zip','w',zipfile.ZIP_DEFLATED) as bundle:
        for name in ['preprocessing.py','test_preprocessing.py','verify_preprocessing.py','requirements.txt','README.md']:
            bundle.write(resource(name),name)
        for name in ['verification.json','feature_audit.csv','unit_tests.txt']:
            bundle.write(audit_dir/name,'verification/'+name)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
