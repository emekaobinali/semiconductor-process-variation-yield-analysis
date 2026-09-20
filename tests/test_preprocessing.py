"""Synthetic examples below test software behavior, not semiconductor science."""
import pickle
import tempfile
import unittest
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from src.preprocessing import MeasurementPreparation, load_development, make_preprocessing_pipeline


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        # Variable, constant-with-missing, all-missing, complete variable, zero constant.
        self.x = np.array([[1, 7, np.nan, 10, 0], [3, np.nan, np.nan, 20, 0], [np.nan, 7, np.nan, 30, 0]], float)

    def test_training_median_and_flags(self):
        p = MeasurementPreparation().fit(self.x)
        assert_array_equal(p.numeric_columns_, [0, 3])
        assert_array_equal(p.constant_columns_, [1, 4])
        assert_array_equal(p.all_missing_columns_, [2])
        assert_array_equal(p.indicator_columns_, [0, 1])
        assert_allclose(p.transform(self.x), [[1,10,0,0],[3,20,0,1],[2,30,1,0]])

    def test_transform_does_not_learn_from_validation(self):
        p = MeasurementPreparation(scale_numeric=True).fit(self.x)
        before = pickle.dumps(p)
        changed = np.array([[np.nan, 999, 123, 1e12, 500], [1e9, np.nan, 456, np.nan, 600]])
        transformed = p.transform(changed)
        self.assertEqual(before, pickle.dumps(p))
        assert_allclose(transformed[0,0], 0)  # Training median remains 2.
        assert_allclose(transformed[1,1], 0)  # Training median remains 20.
        assert_array_equal(transformed[:,-2:], [[1,0],[0,1]])

    def test_raw_values_are_unchanged(self):
        original = self.x.copy()
        p = MeasurementPreparation().fit(self.x)
        p.transform(self.x)
        assert_array_equal(self.x, original)

    def test_schema_and_nonfinite_rejection(self):
        p = MeasurementPreparation().fit(self.x)
        with self.assertRaises(ValueError): p.transform(self.x[:,:4])
        with self.assertRaises(ValueError): p.transform(np.full((1,5), np.inf))
        with self.assertRaises(ValueError): MeasurementPreparation().fit(np.full((1,5), np.inf))

    def test_training_only_filter_and_new_missingness(self):
        p = MeasurementPreparation().fit([[0,1],[0,2]])
        assert_array_equal(p.numeric_columns_, [1])
        assert_allclose(p.transform([[100,np.nan]]), [[1.5]])
        # An always-observed training feature has no fitted missingness flag.
        # Unexpected validation missingness is imputed without changing the schema.
        self.assertEqual(len(p.indicator_columns_), 0)

    def test_optional_scaling_leaves_flags_binary(self):
        p = MeasurementPreparation(scale_numeric=True).fit(self.x)
        result = p.transform(self.x)
        assert_allclose(result[:,:2].mean(axis=0), 0, atol=1e-12)
        assert_allclose(result[:,:2].std(axis=0), 1, atol=1e-12)
        assert_array_equal(result[:,2:], [[0,0],[0,1],[1,0]])

    def test_fold_clones_are_independent_and_y_is_ignored(self):
        prototype = make_preprocessing_pipeline()
        a, b = clone(prototype), clone(prototype)
        a.fit(self.x, [0,1,0])
        shifted = self.x.copy(); shifted[:,0] += 100
        b.fit(shifted, [1,0,1])
        assert_allclose(a['measurements'].medians_, [2,20])
        assert_allclose(b['measurements'].medians_, [102,20])
        c = clone(prototype).fit(self.x, [1,0,1])
        assert_array_equal(a.transform(self.x), c.transform(self.x))
        self.assertFalse(hasattr(prototype['measurements'], 'medians_'))

    def test_loader_ignores_excluded_records_and_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project = root; (project/'data/splits').mkdir(parents=True)
            raw = project/'data/raw'; raw.mkdir(parents=True)
            (project/'data/splits/development_rows.csv').write_text('source_row_1based\n3\n1\n')
            # No reserved manifest exists; excluded rows are deliberately unparseable.
            (raw/'secom.data').write_text(' '.join(['1']*590)+'\nEXCLUDED MUST NOT PARSE\n'+' '.join(['NaN']*590)+'\n')
            (raw/'secom_labels.data').write_text('-1 "19/07/2008 11:55:00"\nEXCLUDED MUST NOT PARSE\n1 "20/07/2008 12:00:00"\n')
            x,y,meta = load_development(project)
            assert_array_equal(y, [1,0])
            self.assertEqual(meta['source_rows'], [3,1])
            self.assertTrue(meta['missing_mask'][0].all())
            self.assertEqual(meta['test_timestamps'][0], '20/07/2008 12:00:00')


if __name__ == '__main__':
    unittest.main(verbosity=2)
