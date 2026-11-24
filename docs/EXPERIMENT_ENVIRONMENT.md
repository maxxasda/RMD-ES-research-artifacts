# Experiment environment

## Recorded host configuration

The experiment setup table supplied with the project records this host configuration:

| Component | Recorded setup |
|---|---|
| Operating system | Ubuntu 20.04 |
| CPU | Intel Core i7-12700K |
| GPU | NVIDIA GeForce RTX 3080 |
| NVIDIA driver | 535.230.02 |
| CUDA | 12.2 |
| RAM | 64 GB |
| Free disk space | 80 GB |
| Python | 3.8 |
| CARLA service port | 2000 |
| OASIS TCS port | 8880 |

These values describe the recorded machine, not minimum requirements. GPU acceleration and CUDA behavior are governed by the separately installed OASIS/CARLA stack.

## Simulator and system-under-test scope

OASIS SIM executes parameterized OpenSCENARIO files. The ego vehicle uses the built-in `DefaultDriver`; its decisions consume simulation ground truth. The vehicle uses `PerfectControl`, an idealized kinematic execution mode that does not model detailed vehicle dynamics. RMD-ES changes scenario parameters and does not modify the internal `DefaultDriver` configuration.

The repository does not include the OASIS service, CARLA server, simulator maps, or a full OASIS deployment. Those must be provisioned independently. The project notes use CARLA port 2000 and OASIS TCS port 8880; update the configuration to match your deployment.

## Version records and known mismatch

The project README recommends Python 3.8. `requirements-search.txt` selects direct package versions from the original pinned dependency list. The original list says `carla==0.10.0`, while a legacy comment in `run_search.sh` mentions the CARLA 0.9.14 Python egg. This release records that discrepancy rather than asserting compatibility. Confirm the correct client binding with the installed OASIS/CARLA server before running experiments.

The old README's PyTorch 1.13/CUDA 11.6 example was used for an out-of-scope flow-model path. It is not included in this RMD-ES search release. CUDA 12.2 and the NVIDIA driver above come from the supplied experiment machine table.
