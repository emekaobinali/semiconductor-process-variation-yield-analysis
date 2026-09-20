"""Training-only SECOM preparation. No predictors or test-set access."""
import csv
from pathlib import Path
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


def load_development(project_dir):
    """Only parse rows listed in the development manifest; ignore all others."""
    project_dir = Path(project_dir)
    with (project_dir / 'data/splits/development_rows.csv').open(newline='') as stream:
        ids = [int(row['source_row_1based']) for row in csv.DictReader(stream)]
    if not ids or len(set(ids)) != len(ids) or min(ids) < 1:
        raise ValueError('Development identifiers must be unique positive row numbers.')
    wanted = set(ids)
    raw_dir = project_dir / 'data/raw'
    measurements, labels, timestamps = {}, {}, {}
    with (raw_dir / 'secom.data').open() as stream:
        for row_id, line in enumerate(stream, 1):
            if row_id not in wanted:
                continue  # Do not tokenize or inspect excluded records.
            values = [float(token) for token in line.split()]
            if len(values) != 590:
                raise ValueError(f'Wrong measurement width at development row {row_id}.')
            measurements[row_id] = values
    with (raw_dir / 'secom_labels.data').open() as stream:
        for row_id, line in enumerate(stream, 1):
            if row_id not in wanted:
                continue
            label, timestamp = line.split(maxsplit=1)
            if int(label) not in (-1, 1):
                raise ValueError('Unexpected development label.')
            labels[row_id] = int(int(label) == 1)
            timestamps[row_id] = timestamp.strip().strip('"')
    if set(measurements) != wanted or set(labels) != wanted:
        raise ValueError('Missing development rows in source files.')
    x = np.asarray([measurements[i] for i in ids], dtype=float)
    return x, np.asarray([labels[i] for i in ids]), {
        'source_rows': ids,
        'test_timestamps': [timestamps[i] for i in ids],
        'missing_mask': np.isnan(x),
    }


class MeasurementPreparation(TransformerMixin, BaseEstimator):
    """Fit medians, column retention and optional scaling on training X only.

    Input columns always correspond to the original anonymous measurements.
    Output: retained numeric measurements, followed by varying missingness flags.
    All raw columns and the full missingness mask remain separate from this object.
    """
    def __init__(self, scale_numeric=False):
        self.scale_numeric = scale_numeric

    def _array(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError('Expected a nonempty two-dimensional measurement array.')
        if np.isinf(x).any():
            raise ValueError('Infinite values are not supported.')
        return x

    def fit(self, X, y=None):
        x = self._array(X)
        self.n_features_in_ = x.shape[1]
        missing = np.isnan(x)
        unique_counts = np.asarray([len(np.unique(col[~np.isnan(col)])) for col in x.T])
        self.numeric_columns_ = np.flatnonzero(unique_counts > 1)
        self.all_missing_columns_ = np.flatnonzero(unique_counts == 0)
        self.constant_columns_ = np.flatnonzero(unique_counts == 1)
        self.indicator_columns_ = np.flatnonzero(missing.any(axis=0) & ~missing.all(axis=0))
        self.medians_ = np.nanmedian(x[:, self.numeric_columns_], axis=0) if len(self.numeric_columns_) else np.empty(0)
        self.scaler_ = None
        if self.scale_numeric and len(self.numeric_columns_):
            numeric = x[:, self.numeric_columns_].copy()
            numeric = np.where(np.isnan(numeric), self.medians_, numeric)
            self.scaler_ = StandardScaler().fit(numeric)
        self.feature_names_out_ = np.asarray(
            [f'measurement_{i + 1:03d}' for i in self.numeric_columns_] +
            [f'measurement_{i + 1:03d}__missing' for i in self.indicator_columns_], dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, 'medians_')
        x = self._array(X)
        if x.shape[1] != self.n_features_in_:
            raise ValueError('Measurement columns must match the original training schema and order.')
        numeric = x[:, self.numeric_columns_].copy()
        numeric = np.where(np.isnan(numeric), self.medians_, numeric)
        if self.scaler_ is not None:
            numeric = self.scaler_.transform(numeric)
        flags = np.isnan(x[:, self.indicator_columns_]).astype(float)
        return np.hstack([numeric, flags])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'feature_names_out_')
        return self.feature_names_out_.copy()


def make_preprocessing_pipeline(scale_numeric=False):
    return Pipeline([('measurements', MeasurementPreparation(scale_numeric=scale_numeric))])
