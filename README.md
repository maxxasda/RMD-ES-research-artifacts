# RMD-ES research code and experimental data

This repository contains the search implementation, experiment configurations, and run-level results for the RMD-ES study on SOTIF-oriented dangerous-scenario search in autonomous driving.

> **Reproduction scope.** The released CSV analysis can be reproduced from the included data. Re-running the original simulator experiments requires a separately installed OASIS/CARLA environment, compatible API/server versions, and the scenario/map assets registered in that environment. OASIS itself and the simulator maps are not included. The code uses OASIS's ground-truth-driven `DefaultDriver` with `PerfectControl`; it does not evaluate camera/radar/lidar perception or detailed vehicle dynamics.

> **CARLA API note.** The original dependency file lists `carla==0.10.0`, while a legacy comment in the launcher refers to a CARLA 0.9.14 Python egg. Match the Python API to the OASIS server installed at your site before running searches; do not assume those two references are interchangeable.

## What is included

| Path | Contents |
|---|---|
| `scenario/search_lib/` | Search algorithms, scenario parsers, simulation interface, evaluation and utility code. The full proposed method is named `ImprovedDGGAAlgorithm` in the source; `XDGGA_FULL` selects its full mode, referred to as RMD-ES in the paper. |
| `scenario/sotif_search_main.py` and `run_search.sh` | Search entry point and a launcher for one configured scenario–algorithm pair. |
| `scenario/config/search_config.yaml` | Portable one-scenario/one-algorithm example. Set the host, ports, paths, scenario, method, and seed for your installation. |
| `scenario_xosc/` | The three OpenSCENARIO templates used in the released experiments: highway cut-in, CCRH, and CBNAO. OASIS maps and deployment files are not included. |
| `results/` | 90 original `sotif_search_results.csv` files, three scenarios × ten methods × three runs, plus one path-redacted configuration snapshot per run and a SHA-256 data manifest. |
| `scripts/` | Script to regenerate run-level statistics and mean convergence data; optional PDF/PNG plotting script. |
| `requirements-search.txt` | Pinned Python package versions recorded for the search implementation. |
| `requirements-analysis.txt` | Minimal pinned packages for the analysis and plotting scripts. |
| `docs/` | Experiment environment, data dictionary, run guide, and licensing notes. |

The legacy result-directory name `ImprovedDGGA` maps to the paper-facing name **RMD-ES**. `XDGGA_NO_RISK`, `XDGGA_NO_ARCHIVE`, `XDGGA_NO_STAGNATION`, and `XDGGA_NO_DIVERSITY` are ablation variants. Baselines are Random, PSO, NSGA-II, GCE3S, and DGGA.

## Recorded experiment environment

The following is the machine configuration recorded for the experiments, not a minimum hardware requirement:

| Item | Recorded configuration |
|---|---|
| Operating system | Ubuntu 20.04 |
| CPU | Intel Core i7-12700K |
| GPU | NVIDIA GeForce RTX 3080 |
| NVIDIA driver | 535.230.02 |
| CUDA | 12.2 |
| Memory | 64 GB RAM |
| Available disk space | 80 GB |
| Python | 3.8 |
| Simulator interfaces | CARLA port 2000; OASIS TCS port 8880 |

The older project README includes a PyTorch/CUDA example for a flow-model workflow. That workflow is outside the RMD-ES paper scope and is not part of this release.

## Reproduce the released analysis

Create a Python 3.8 environment, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-analysis.txt
python scripts/reproduce_analysis.py --plot
```

By default, the script reads `results/` and writes tables, JSON, a SHA-256 manifest, and (with `--plot`) a convergence figure to `reproduced_outputs/`. It calculates each run's metrics first, then takes arithmetic means across the three independent runs. It does not report standard deviations or significance tests. See [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) for metric definitions and convergence handling.

## Run a new search with OASIS/CARLA

1. Set up OASIS and its compatible CARLA client/server outside this repository. Ensure the OASIS simulation and TCS endpoints are reachable; the recorded ports are 2000 and 8880.
2. Import the appropriate `.xosc` template and its required map into your OASIS installation. For highway cut-in, the parser expects the imported scenario file to be named `lane_change_scenario.xosc`; the repository template is `lanechange.xosc`.
3. Edit `scenario/config/search_config.yaml`. Configure **one scenario and one algorithm per run**, set a fixed seed for reproducibility, and update the OASIS/CARLA host and paths. Method identifiers accepted by the source include `RANDOM`, `PSO`, `NSGA2`, `GCE3S`, `DGGA`, `XDGGA_FULL`, and the four `XDGGA_NO_*` ablation names.
4. Create and activate the Python 3.8 environment. Install the pinned search dependencies in `requirements-search.txt`; then install the CARLA client version that matches your OASIS installation if it differs from the listed package.
5. Launch the configured experiment:

```bash
bash run_search.sh
```

The search writes a new run folder under `results/`. The committed result CSVs are the released study data; do not overwrite them when running new experiments.

For full setup notes and limitations, see [`docs/EXPERIMENT_ENVIRONMENT.md`](docs/EXPERIMENT_ENVIRONMENT.md) and [`docs/REPRODUCING_SEARCH.md`](docs/REPRODUCING_SEARCH.md).

For creating and pushing the public GitHub repository, see [`docs/GITHUB_RELEASE.md`](docs/GITHUB_RELEASE.md). TVT's review model is summarized in [`docs/TVT_REPOSITORY_NOTE.md`](docs/TVT_REPOSITORY_NOTE.md).

## License and public access

This repository is prepared for public viewing, but no open-source license has been assigned. Without a license, the code and data remain under their applicable copyright; public visibility alone does not grant reuse or redistribution rights. Review [`LICENSE_STATUS.md`](LICENSE_STATUS.md) before choosing a license or publishing on behalf of all authors and affiliated organizations.

