from src.paths import ROOT, resource, report, require_file, protect_completed_evidence
from pathlib import Path
from collections import Counter
import csv
import hashlib
import io
import json
import math
import platform
import random
from datetime import datetime, timezone

def main():
    protect_completed_evidence()
    """Create the approved split once, without loading the measurement matrix."""

    base = resource('inspection')
    destination = resource('splits')
    if destination.exists():
        raise SystemExit('Split directory already exists. Refusing to replace an established split.')
    original = json.loads((resource('inspection/summary.json')).read_text())
    raw = resource('inspection/raw')
    for name, metadata in original['files'].items():
        assert hashlib.sha256((raw / name).read_bytes()).hexdigest() == metadata['sha256'], name

    # Labels are used only to allocate the approved stratified split.
    # A source row is a traceability identifier, never a model input.
    labels = [int(line.split(maxsplit=1)[0]) for line in (raw / 'secom_labels.data').read_text().splitlines()]
    assert Counter(labels) == Counter({-1: 1463, 1: 104})
    seed = 42  # Fixed before any split is generated or evaluated; never searched.
    test_fraction = 0.20
    total_test = math.ceil(len(labels) * test_fraction)
    counts = Counter(labels)
    quotas = {label: total_test * count / len(labels) for label, count in counts.items()}
    allocation = {label: math.floor(quota) for label, quota in quotas.items()}
    remaining = total_test - sum(allocation.values())
    for label in sorted(counts, key=lambda label: (-(quotas[label] - allocation[label]), label))[:remaining]:
        allocation[label] += 1

    def split_ids():
        rng = random.Random(seed)
        development, test = [], []
        for label in sorted(counts):
            ids = [i + 1 for i, value in enumerate(labels) if value == label]
            rng.shuffle(ids)
            test.extend(ids[:allocation[label]])
            development.extend(ids[allocation[label]:])
        # Keep randomized order so the saved row order does not imply temporal analysis.
        rng.shuffle(development)
        rng.shuffle(test)
        return development, test

    development, test = split_ids()
    assert (development, test) == split_ids()
    assert len(development) == 1253 and len(test) == 314
    assert not set(development) & set(test)
    assert set(development) | set(test) == set(range(1, len(labels) + 1))
    assert len(set(development)) == len(development)
    assert len(set(test)) == len(test)
    assert Counter(labels[i - 1] for i in development) == Counter({-1: 1170, 1: 83})
    assert Counter(labels[i - 1] for i in test) == Counter({-1: 293, 1: 21})

    def csv_bytes(ids):
        stream = io.StringIO(newline='')
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['source_row_1based'])
        writer.writerows((i,) for i in ids)
        return stream.getvalue().encode('utf-8')

    dev_bytes, test_bytes = csv_bytes(development), csv_bytes(test)
    record = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'seed': seed, 'python_version': platform.python_version(),
        'algorithm': 'Python random.Random(42); ascending labels; shuffle each class; allocate ceil(0.2*N) test rows using proportional largest remainders; shuffle each final partition. Not sklearn train_test_split. Saved manifests are authoritative.',
        'raw_relative_to_split_directory': '../../secom-inspection/raw',
        'raw_files': original['files'],
        'development': {'records':1253, 'passes':1170, 'failures':83, 'manifest_sha256':hashlib.sha256(dev_bytes).hexdigest()},
        'reserved_test': {'records':314, 'passes':293, 'failures':21, 'manifest_sha256':hashlib.sha256(test_bytes).hexdigest()},
        'checks': {'complete_coverage':True, 'no_overlap':True, 'no_duplicate_ids':True, 'deterministic_reproduction':True, 'original_hashes_match':True},
        'test_policy': 'After this creation-time allocation check, do not inspect, transform, fit on, select using, tune using, or evaluate reserved test records until the full method is frozen and the user authorizes evaluation.',
        'measurement_matrix_loaded':False,
        'preprocessing_started':False,
        'chronological_split_created':False,
    }
    destination.mkdir(parents=True)
    for filename, payload in [('development_rows.csv',dev_bytes),('reserved_test_rows.csv',test_bytes),('split_record.json',json.dumps(record,indent=2).encode())]:
        with (destination / filename).open('xb') as handle:
            handle.write(payload)

    outputs = ROOT / 'artifacts/reproduction'
    outputs.mkdir(exist_ok=True)
    protocol = '''# SECOM: approved protocol and split record

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
    '''
    with (outputs / 'secom-approved-protocol-and-split.md').open('x',encoding='utf-8') as handle:
        handle.write(protocol)
    print(json.dumps({k:record[k] for k in ['seed','development','reserved_test','checks','measurement_matrix_loaded','preprocessing_started']},indent=2))


if __name__ == '__main__':
    main()
