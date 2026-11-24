# Experimental results

This directory contains the 90 released run-level `sotif_search_results.csv` files and `run_configs/`, with one corresponding configuration snapshot per result file. The configuration copies preserve scenario/method parameters and seeds; only machine-specific filesystem paths are replaced with portable paths. `data_manifest.csv` records relative CSV paths, scenario/method/run identifiers, byte counts, seeds, and SHA-256 hashes.

The folder names retain the original experiment identifiers. The full-method directory suffix `ImprovedDGGA` corresponds to RMD-ES in the paper. The four `XDGGA_NO_*` names identify component-removal ablations. Some runs have different valid CSV record counts; the configured search budget and recorded rows are not interchangeable.

Field descriptions and derived metric rules are documented in `../docs/DATA_DICTIONARY.md`.
