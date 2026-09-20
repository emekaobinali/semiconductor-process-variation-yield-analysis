# SECOM data provenance

Source: McCann and Johnston (2008), SECOM, UCI Machine Learning Repository.
Dataset DOI: https://doi.org/10.24432/C54305
Original archive: https://archive.ics.uci.edu/static/public/179/secom.zip
The completed project records identify the dataset license as CC BY 4.0; retain source attribution.

Download the archive manually and extract secom.data, secom_labels.data and secom.names into data/raw/. Compare checksums against reports/results/inspection/secom-structure-results.json. Existing local copies are preserved and excluded from Git.

The source has 1,567 rows and 590 anonymous measurements. Labels encode pass as -1 and failure as 1. Do not infer physical names or units. Quality-test timestamps are metadata, not established measurement acquisition times.

Portable code resolves these locations from the repository root. Imports and synthetic tests require no raw download. Start an optional notebook kernel at the repository root or notebooks/; the inspection notebook resolves data/raw from either location. Do not execute the notebook to repeat the already completed inspection during portability verification.

Saved record-level predictions and the fitted model are intentionally local-only. Their exact restore locations are listed in config/resource_paths.json. Missing artifacts must not trigger automatic retraining or another reserved-test evaluation. See reports/methodology/reproducibility.md for the verified scope and remaining clean-install limitations.

data/splits contains the authoritative saved partitions and development folds. Do not regenerate them. The reserved test has already been evaluated once and must not be reused for tuning. No download, data inspection or model execution is triggered by this document.

## License separation

The MIT license at the repository root applies to project code. SECOM remains separately licensed under CC BY 4.0; preserve the dataset citation and attribution above. The project-code license does not relicense the SECOM dataset.
