"""Explicit public aggregate allowlist; no scientific workflow imports."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'models': 'reports/results/comparisons/secom-model-comparison-summary.csv',
    'thresholds': 'reports/results/comparisons/secom-ensemble-threshold-summary.csv',
    'intermediate': 'reports/results/comparisons/secom-intermediate-thresholds.csv',
    'baseline_folds': 'reports/results/baseline_comparison/fold_metrics.csv',
    'stronger_folds': 'reports/results/comparisons/secom-stronger-model-fold-metrics.csv',
    'test': 'reports/final_evaluation/secom-final-test-metrics.json',
    'frozen': 'config/frozen_method.json',
    'inspection': 'reports/results/inspection/secom-structure-results.json',
    'features': 'reports/feature_interpretation/secom-feature-all_feature_importances.csv',
    'correlations': 'reports/feature_interpretation/secom-feature-selected_correlations.csv',
    'technical': 'reports/summaries/technical-engineering-summary.md',
    'recruiter': 'reports/summaries/recruiter-summary.md',
    'protocol': 'reports/methodology/secom-approved-protocol-and-split.md',
    'interpretation': 'reports/feature_interpretation/secom-frozen-feature-interpretation.md',
    'reproducibility': 'reports/methodology/reproducibility.md',
    'attribution': 'data/README.md',
    'workflow_figure': 'figures/secom-project-workflow.png',
    'tradeoff_figure': 'figures/secom-development-screening-tradeoffs.png',
    'matrix_figure': 'figures/secom-reserved-test-confusion-matrix.png',
    'feature_figure': 'figures/secom-anonymous-feature-importance.png',
}
MODELS = {'reference': 'Prevalence reference', 'logistic': 'Logistic regression',
          'tree': 'Shallow decision tree', 'logistic_C0.1': 'Regularized logistic regression (C=0.1)',
          'small_forest': 'Random Forest', 'shallow_boosting': 'Gradient Boosting'}

def source_path(key):
    return ROOT / SOURCES[key]

def load(key):
    path = source_path(key)
    if path.suffix == '.csv':
        return pd.read_csv(path)
    if path.suffix == '.json':
        return json.loads(path.read_text(encoding='utf-8-sig'))
    return path.read_text(encoding='utf-8-sig')

def threshold_table():
    cols = ['model', 'indicators', 'threshold', 'recall', 'precision', 'FP', 'FN',
            'specificity', 'balanced_accuracy', 'TP', 'TN', 'flagged']
    data = pd.concat([load('thresholds')[cols], load('intermediate')[cols]], ignore_index=True)
    if data.duplicated(['model', 'indicators', 'threshold']).any():
        raise ValueError('Duplicate saved operating points: review source files.')
    return data.sort_values(['model', 'indicators', 'threshold']).reset_index(drop=True)

def operating_point(model, indicators, threshold):
    data = threshold_table()
    rows = data[(data.model == model) & (data.indicators == indicators) & (data.threshold == threshold)]
    if len(rows) != 1:
        raise ValueError('Only saved development operating points are supported.')
    return rows.iloc[0]

def selected_development():
    return operating_point('small_forest', False, 0.35)
