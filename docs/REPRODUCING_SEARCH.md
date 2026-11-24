# Reproducing a search run

## Prerequisites

Install Python 3.8, the packages in `requirements-search.txt`, OASIS SIM, and a CARLA server/client version compatible with that OASIS deployment. Import the correct scenario files and maps into OASIS. The simulator stack and map assets are external dependencies and are not bundled in this repository.

The source project's CARLA version references are inconsistent: its requirements list CARLA Python API 0.10.0, but a legacy launcher comment names a 0.9.14 egg. Resolve that version against the installed service before launching.

## Configure one experiment

Edit `scenario/config/search_config.yaml`:

- Use one entry under `scenario_classes` and one under `search_algorithms`.
- Set the random `seed` explicitly.
- Set `simulation_scenario_path` and `sim_task.carla_parames[].scenario_path` to the OASIS scenario directory.
- Set `host`, `carla_port`, and `tcs_port` to the reachable endpoints.

Accepted search labels in `SearchFactory.py` include `RANDOM`, `PSO`, `NSGA2`, `DGGA`, `GCE3S`, `XDGGA_FULL`, `XDGGA_NO_RISK`, `XDGGA_NO_ARCHIVE`, `XDGGA_NO_STAGNATION`, and `XDGGA_NO_DIVERSITY`. The project treats each experiment as one scenario plus one algorithm per run.

For highway cut-in, the parser expects the scenario file to be named `lane_change_scenario.xosc` in the configured OASIS directory. The supplied template is `scenario_xosc/lanechange.xosc`; import/copy it using the expected local name. For CCRH and CBNAO the corresponding templates are `CCRH.xosc` and `CBNAO.xosc`.

## Launch

```bash
conda create -n rmd-es python=3.8
conda activate rmd-es
python -m pip install -r requirements-search.txt
bash run_search.sh
```

CARLA bindings are site-specific; if `carla==0.10.0` is incompatible with your OASIS server, replace it with the server-matched Python API and record that change in your experiment metadata. Do not overwrite the released CSVs. New runs write to a new timestamped directory under `results/`.

## Reproduce summary tables and convergence output

Use `requirements-analysis.txt` and run `python scripts/reproduce_analysis.py --plot`. The analysis reads the released result CSVs; it does not start OASIS or recreate the simulator observations. See `DATA_DICTIONARY.md` for data fields and analysis rules.
