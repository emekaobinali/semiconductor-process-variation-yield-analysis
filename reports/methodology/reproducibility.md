# Portability and reproducibility verification

## Scope

Only this staging repository was modified. No model selection, threshold comparison, feature selection, model fitting, notebook execution or reserved-test scoring was performed. Existing scientific reports, settings, split/fold manifests, raw copies, predictions and model artifacts remain byte-identical: 59 copied non-code/non-notebook evidence files matched the staging manifest at portability verification; subsequent duplicate consolidation is recorded in public-cleanup.json. The employer-facing root README and four PNG/SVG figures are complete.

## Portable execution layout

- Repository root is derived from `src/paths.py`, not a username, drive, current working directory or previous workspace location.
- `config/resource_paths.json` explicitly maps scientific resources to `data/`, `config/`, `reports/` and local `artifacts/` locations. Keys identify the historical artifacts; values are portable repository-relative paths.
- `src` and its subdirectories are importable packages. All executable workflows use `main()` and a main guard. Importing a source module cannot run its analysis.
- Preprocessing reads `data/splits/development_rows.csv` and only parses that manifest's rows from `data/raw/secom.data` and `data/raw/secom_labels.data`. Tests exercise the same layout with synthetic fixtures containing deliberately unparseable excluded rows.
- Use module entry points from the repository root, rather than executing nested script filenames directly. Example safe metadata check: `python -B -m src.inspection.save_evidence`. This now checks the cleaned notebook structure without executing its cells.
- Archived model comparison scripts retain their existing-result guards. The final-test entry point explicitly refuses replay before loading data or a model. A write guard prevents analysis scripts from overwriting completed evidence; permitted newly generated outputs are confined to `artifacts/reproduction/`.
- The historical preprocessing verification command refuses to repeat archived checks; use the synthetic unit tests below instead. This avoids regenerating saved fold assignments or environment files.
- A root `preprocessing.py` compatibility import supports the historical module name embedded in the unchanged joblib artifact. The artifact was not loaded for this verification. `config/preprocessing_provenance.json` connects the original source hash with the unchanged transformer implementation; historical frozen hashes were not rewritten to match migrated code.

## Verified tests

Run from the repository root:

```text
python -B -m unittest discover -s tests -t . -v
```

Result: **13 tests passed** in the working Python 3.12.14 environment. Eight synthetic preprocessing tests cover imputation, training-only filtering/scaling, unchanged input values and fitted state, independent fold clones, schema checks and development-only loading. Five portability tests cover source imports, contained path resolution, final-test replay refusal, notebook sanitation and unchanged preprocessing transformer logic.

All **16 source modules** imported successfully. A fresh subprocess import check rejected any attempt to read raw data, split manifests or serialized scientific artifacts during import. No such access occurred.

## Dependencies and clean installation

`requirements.txt` contains the observed runtime versions of NumPy, pandas, SciPy, scikit-learn, joblib, threadpoolctl, narwhals, python-dateutil, six and Windows-only tzdata. Narwhals is a required dependency in the installed scikit-learn metadata. Cloudpickle was removed because it was only included in historical environment bookkeeping and is not needed by the project implementation. Standard-library unittest is used; pytest is not required. Notebook inspection code uses the standard library; an optional notebook front end is not included in runtime requirements.

An isolated `.venv-portability` was successfully created inside staging. This command attempted a safe offline installation:

```text
python -m pip install --no-index --disable-pip-version-check -r requirements.txt
```

It stopped with:

```text
ERROR: Could not find a version that satisfies the requirement numpy==2.5.3 (from versions: none)
ERROR: No matching distribution found for numpy==2.5.3
```

No local package source was available to that offline install. This does **not** establish whether the pinned versions are available from an online index. Online installation, cross-platform installation, imports/tests inside the new environment, and end-to-end clean-clone scientific reproduction remain unverified. No package versions were substituted and the existing environment was not changed.

## Data and omitted local artifacts

`data/README.md` is the authoritative download/location guide. A clone needs the UCI files in `data/raw/` for data-dependent work; synthetic tests and imports do not need them. Saved split/fold manifests are included and must not be regenerated.

The approved public exclusions intentionally omit the fitted joblib model, record-level predictions and permutation cache. A public clone can inspect saved reports/aggregate evidence and test preprocessing, but cannot directly replay artifact-dependent interpretation or prediction checks without those exact local artifacts. Restore them to the paths recorded in `config/resource_paths.json`; do not retrain automatically when they are absent. This is a documented reproducibility boundary, not a claim of full clean-clone reproduction.

## Notebook and public-file review

The inspection notebook now locates `data/raw` from a kernel started at the repository root or `notebooks/`. Code cells were parsed without execution. Recorded outputs and execution counts were cleared; cell metadata was removed; only generic Python kernel metadata remains. Scientific inspection results remain in the unchanged JSON/report artifacts. The notebook was removed from `.gitignore` and is suitable to include based on this review.

The intended public text files were scanned for personal Windows/macOS/Linux home paths, the local username, private-key headers, common token patterns and credential assignments. No matches were found. This is a targeted source/notebook review, not an exhaustive external secret-scanner certification. Historical relative provenance paths are not personal machine paths and remain unchanged.

## Remaining blockers

1. Demonstrate installation and tests in an independent environment with the exact pins, using an available package index or approved wheel source.
2. If full artifact-dependent reproduction from a public clone is required, separately approve a distribution strategy for the currently excluded artifacts. No publication or release decision was made here.

The original reports remain the scientific record. No Git repository was initialized and nothing was published.

### Optional figure rendering

The four PNG/SVG figure pairs are already included; viewing them needs no Python packages. To regenerate them from saved aggregate results, install the separate optional dependencies and run:

```text
python -m pip install -r requirements-figures.txt
python figures/render_figures.py
```

ReportLab 4.4.9 and pypdfium2 5.13.0 were the rendering versions used. Pillow is required for PNG export and is pinned in the optional file. These packages are not part of the core scientific/runtime requirements. Figure rendering does not retrain models or evaluate test records. Fresh installation of either dependency file remains unverified.
