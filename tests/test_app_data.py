"""Presentation checks only: never load raw records or fitted scientific artifacts.

Run in the app environment: python -m unittest tests.test_app_data -v
"""
import ast
import builtins
import hashlib
import io
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest
from app.data_loader import ROOT, SOURCES, load, threshold_table, operating_point
from app.components import NOTICE

PAGES = ['Home / Overview', 'Dataset & Workflow', 'Model Comparison', 'Threshold Tradeoffs',
         'Final Test Results', 'Confusion Matrix', 'Feature Interpretation', 'Limitations']

class AppDataTests(unittest.TestCase):
    def test_allowed_sources_present(self):
        for relative in SOURCES.values():
            self.assertTrue((ROOT / relative).is_file(), relative)
            self.assertTrue(relative.startswith(('reports/', 'config/', 'figures/', 'data/README')))

    def test_saved_confusion_metrics(self):
        records = load('models').to_dict('records') + threshold_table().to_dict('records') + [load('test')]
        for row in records:
            with self.subTest(row=row):
                total = row['TP'] + row['FN'] + row['FP'] + row['TN']
                self.assertIn(total, (1253, 314))
                self.assertAlmostEqual(row['recall'], row['TP'] / (row['TP'] + row['FN']))
                self.assertAlmostEqual(row['specificity'], row['TN'] / (row['TN'] + row['FP']))
                self.assertAlmostEqual(row['balanced_accuracy'], (row['recall'] + row['specificity']) / 2)
                if row['TP'] + row['FP']:
                    self.assertAlmostEqual(row['precision'], row['TP'] / (row['TP'] + row['FP']))
                else:
                    self.assertTrue(math.isnan(row['precision']))

    def test_ap_and_counts_match_folds(self):
        folds = pd.concat([load('baseline_folds'), load('stronger_folds')])
        for row in load('models').itertuples():
            subset = folds[(folds.model == row.model) & (folds.indicators == row.indicators)]
            self.assertEqual(len(subset), 5)
            self.assertAlmostEqual(row.mean_AP, subset.AP.mean())
            self.assertAlmostEqual(row.AP_fold_sd, subset.AP.std())
            for key in ['TP','FN','FP','TN']:
                self.assertEqual(getattr(row, key), subset[key].sum())

    def test_thresholds_are_exact_source_rows(self):
        combined = threshold_table()
        self.assertEqual(len(combined), 38)
        for key in ['thresholds', 'intermediate']:
            for source in load(key).to_dict('records'):
                actual = operating_point(source['model'], source['indicators'], source['threshold'])
                for field in combined.columns:
                    if pd.isna(source[field]):
                        self.assertTrue(pd.isna(actual[field]))
                    else:
                        self.assertEqual(actual[field], source[field])
        with self.assertRaises(ValueError):
            operating_point('small_forest', False, 0.36)

    def test_final_result_and_method(self):
        result = load('test')
        self.assertEqual([result[k] for k in ['TP','FN','FP','TN']], [14,7,102,191])
        self.assertEqual([f'{result[k]:.1%}' for k in ['recall','precision','specificity','balanced_accuracy']],
                         ['66.7%','12.1%','65.2%','65.9%'])
        self.assertEqual(f'{result["average_precision"]:.4f}', '0.1872')
        frozen = load('frozen')
        self.assertEqual(frozen['threshold'], 0.35)
        self.assertFalse(frozen['missingness_indicators_in_classifier'])

    def test_no_scientific_imports(self):
        for path in [ROOT/'streamlit_app.py', * (ROOT/'app').rglob('*.py')]:
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [n.name for n in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or '']
                self.assertFalse(any(n.split('.')[0] in {'src','sklearn','joblib','preprocessing'} for n in names), path)

    def test_all_pages_and_controls_without_scientific_access(self):
        public_files = [p for folder in ['reports','config','figures','data','src','notebooks']
                        for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        # data contains public split manifests only; hash bytes, never parse records.
        public_files = [p for p in public_files if 'raw' not in p.parts]
        hashes = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in public_files}
        original_open, original_io_open = builtins.open, io.open

        def guard(original):
            def checked(file, mode='r', *args, **kwargs):
                if isinstance(file, (str, Path)):
                    path = Path(file).resolve()
                    if path.is_relative_to(ROOT):
                        relative = path.relative_to(ROOT)
                        forbidden = relative.parts[0] == 'artifacts' or relative.parts[:2] in [('data','raw'), ('data','splits')]
                        if forbidden or path.suffix in {'.joblib','.pkl','.pickle','.npz'}:
                            raise AssertionError(f'Forbidden scientific data access: {relative}')
                return original(file, mode, *args, **kwargs)
            return checked

        with patch('builtins.open', guard(original_open)), patch('io.open', guard(original_io_open)):
            app = AppTest.from_file(str(ROOT/'streamlit_app.py'), default_timeout=20).run()
            for page in PAGES:
                app.sidebar.radio[0].set_value(page).run()
                self.assertFalse(app.exception, page)
                self.assertFalse(app.error, page)
                self.assertEqual(app.warning[0].value, NOTICE)
            app.sidebar.radio[0].set_value('Model Comparison').run()
            for mode in ['Without','With','Both']:
                app.main.radio[0].set_value(mode).run()
                expected = load('models')
                if mode != 'Both':
                    expected = expected[expected.indicators == (mode == 'With')]
                self.assertEqual(len(app.dataframe[0].value), len(expected))
                self.assertFalse(app.exception)
            app.multiselect[0].set_value([]).run()
            self.assertFalse(app.exception)
            app.sidebar.radio[0].set_value('Threshold Tradeoffs').run()
            for model in ['small_forest','shallow_boosting','tree']:
                app.selectbox[0].set_value(model).run()
                for mode in ['Without','With']:
                    app.main.radio[0].set_value(mode).run()
                    expected = threshold_table()
                    expected = expected[(expected.model == model) & (expected.indicators == (mode == 'With'))]
                    self.assertEqual(app.selectbox[1].options, [f'{t:.2f}' for t in expected.threshold])
                    for row in expected.itertuples():
                        app.selectbox[1].set_value(row.threshold).run()
                        self.assertFalse(app.exception)
                        displayed = app.dataframe[0].value.iloc[0]
                        self.assertEqual(displayed['Failures caught'], row.TP)
                        self.assertEqual(displayed['False alarms'], row.FP)
                        self.assertEqual(displayed['Failure recall'], f'{row.recall:.1%}')
            app.sidebar.radio[0].set_value('Final Test Results').run()
            self.assertEqual([m.value for m in app.metric], ['66.7%','12.1%','65.2%','65.9%','0.1872'])
            app.sidebar.radio[0].set_value('Feature Interpretation').run()
            for method in ['Permutation importance','Built-in importance']:
                app.main.radio[0].set_value(method).run()
                app.selectbox[0].set_value(20).run()
                rank = 'permutation_rank' if method.startswith('Permutation') else 'builtin_rank'
                expected = load('features').sort_values(rank).head(20)
                self.assertEqual(app.dataframe[0].value.feature.tolist(), expected.feature.tolist())
                for feature in expected.feature:
                    app.selectbox[1].set_value(feature).run()
                    self.assertFalse(app.exception, feature)
        for path, digest in hashes.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest, str(path))

if __name__ == '__main__':
    unittest.main()
