from src.paths import ROOT, resource, report, require_file, protect_completed_evidence
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math

def main():
    protect_completed_evidence()

    raw = resource('inspection/raw')
    data_lines = (raw / 'secom.data').read_text().splitlines()
    label_lines = (raw / 'secom_labels.data').read_text().splitlines()
    tokens = [line.split() for line in data_lines]
    widths = Counter(map(len, tokens))
    assert len(widths) == 1, widths
    width = len(tokens[0])
    bad_tokens = []
    rows = []
    for i, row in enumerate(tokens):
        values = []
        for j, token in enumerate(row):
            try:
                values.append(float(token))
            except ValueError:
                bad_tokens.append((i + 1, j + 1, token))
        rows.append(values)
    assert not bad_tokens, bad_tokens
    labels, dates = [], []
    for line in label_lines:
        label, timestamp = line.split(maxsplit=1)
        labels.append(int(label))
        dates.append(datetime.strptime(timestamp.strip('"'), '%d/%m/%Y %H:%M:%S'))
    assert len(rows) == len(labels)
    missing_cols = [sum(math.isnan(row[j]) for row in rows) for j in range(width)]
    missing_rows = [sum(math.isnan(value) for value in row) for row in rows]
    unique_cols = [len({row[j] for row in rows if not math.isnan(row[j])}) for j in range(width)]
    date_counts = Counter(dates)
    canonical_rows = [tuple(None if math.isnan(v) else v for v in row) for row in rows]
    row_counts = Counter(canonical_rows)
    full_counts = Counter((r, y, t) for r, y, t in zip(canonical_rows, labels, dates))
    summary = {
        'source_url': 'https://archive.ics.uci.edu/static/public/179/secom.zip',
        'inspection_utc': datetime.now(timezone.utc).isoformat(),
        'files': {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(raw.iterdir())},
        'data_rows': len(rows), 'measurement_columns': width,
        'row_width_counts': dict(widths),
        'blank_data_lines': sum(not line.strip() for line in data_lines),
        'blank_label_lines': sum(not line.strip() for line in label_lines),
        'label_rows': len(labels), 'label_counts': dict(Counter(labels)),
        'fail_fraction': labels.count(1) / len(labels),
        'total_measurement_cells': len(rows) * width,
        'missing_cells': sum(missing_cols),
        'missing_cell_fraction': sum(missing_cols) / (len(rows) * width),
        'missing_tokens': dict(Counter(token for row in tokens for token in row if math.isnan(float(token)))),
        'infinite_values': sum(math.isinf(v) for row in rows for v in row),
        'columns_with_missing': sum(v > 0 for v in missing_cols),
        'columns_without_missing': sum(v == 0 for v in missing_cols),
        'all_missing_columns_1based': [j + 1 for j, count in enumerate(missing_cols) if count == len(rows)],
        'constant_observed_columns': sum(n == 1 for n in unique_cols),
        'constant_fully_populated_columns': sum(n == 1 and m == 0 for n, m in zip(unique_cols, missing_cols)),
        'constant_with_missing_columns': sum(n == 1 and m > 0 for n, m in zip(unique_cols, missing_cols)),
        'columns_with_multiple_observed_values': sum(n > 1 for n in unique_cols),
        'rows_with_missing': sum(n > 0 for n in missing_rows),
        'all_missing_rows': sum(n == width for n in missing_rows),
        'missing_per_row_min_max': [min(missing_rows), max(missing_rows)],
        'date_min': min(dates).isoformat(' '), 'date_max': max(dates).isoformat(' '),
        'first_date': dates[0].isoformat(' '), 'last_date': dates[-1].isoformat(' '),
        'timestamps_nondecreasing': all(a <= b for a, b in zip(dates, dates[1:])),
        'adjacent_timestamp_reversals': sum(a > b for a, b in zip(dates, dates[1:])),
        'unique_timestamps': len(date_counts),
        'timestamp_groups_with_ties': sum(n > 1 for n in date_counts.values()),
        'rows_in_timestamp_ties': sum(n for n in date_counts.values() if n > 1),
        'timestamp_duplicate_excess': sum(n - 1 for n in date_counts.values()),
        'duplicate_measurement_row_excess': sum(n - 1 for n in row_counts.values()),
        'duplicate_full_record_excess': sum(n - 1 for n in full_counts.values()),
        'first_row_preview': tokens[0][:5], 'first_label_line': label_lines[0],
    }
    resource('inspection/summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print('\n--- SOURCE NOTES ---\n' + (raw / 'secom.names').read_text())


if __name__ == '__main__':
    main()
