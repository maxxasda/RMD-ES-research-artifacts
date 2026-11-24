# RMD-ES 代码与实验数据

本仓库包含 RMD-ES 自动驾驶 SOTIF 危险场景搜索研究的核心搜索源码、实验配置与逐次运行结果。

> **复现范围：**仓库中的 CSV 后处理分析可基于所含数据复现。重新运行原始仿真实验，需要另外安装并配置 OASIS/CARLA、匹配的 Python API 和服务端版本，以及在 OASIS 中登记的场景和地图。本仓库不包含 OASIS 服务和仿真地图。实验使用 OASIS 内置的 `DefaultDriver` 与 `PerfectControl`，其输入为仿真真值；不评价摄像头、雷达或激光雷达感知能力，也不模拟详细车辆动力学。

> **CARLA API 版本说明：**原项目依赖清单记录 `carla==0.10.0`，但启动脚本中留有 CARLA 0.9.14 Python egg 的旧注释。运行前要依据你安装的 OASIS 服务端确认兼容 API 版本，不能假定两者可以互换。

## 仓库内容

| 路径 | 内容 |
|---|---|
| `scenario/search_lib/` | 搜索算法、场景解析、仿真接口、评价和工具代码。源码中完整方法类名为 `ImprovedDGGAAlgorithm`；`XDGGA_FULL` 代表完整模式，论文中称为 RMD-ES。 |
| `scenario/sotif_search_main.py`、`run_search.sh` | 搜索入口和单个场景—算法组合的启动脚本。 |
| `scenario/config/search_config.yaml` | 单场景、单算法的可移植配置示例。需按本地环境设置主机、端口、路径、场景、算法和随机种子。 |
| `scenario_xosc/` | 高速邻车切入、CCRH、CBNAO 三个 OpenSCENARIO 模板；不包含 OASIS 地图和部署文件。 |
| `results/` | 90 份原始 `sotif_search_results.csv`（3 个场景 × 10 种方法 × 每种 3 次运行），每次运行的脱敏路径配置副本，以及 SHA-256 数据清单。 |
| `scripts/` | 重新计算逐次运行统计和平均收敛数据的脚本，以及可选的 PDF/PNG 绘图脚本。 |
| `requirements-search.txt` | 搜索源码依赖版本记录。 |
| `requirements-analysis.txt` | 数据分析和绘图脚本的固定依赖版本。 |
| `docs/` | 实验环境、数据字典、运行指南和许可说明。 |

历史结果目录名 `ImprovedDGGA` 在论文中对应 **RMD-ES**。`XDGGA_NO_RISK`、`XDGGA_NO_ARCHIVE`、`XDGGA_NO_STAGNATION`、`XDGGA_NO_DIVERSITY` 是消融变体。对比算法为 Random、PSO、NSGA-II、GCE3S 和 DGGA。

## 实验环境记录

以下是实验时记录的本机环境，不代表运行的最低配置：Ubuntu 20.04、Intel Core i7-12700K、NVIDIA GeForce RTX 3080、驱动 535.230.02、CUDA 12.2、64 GB 内存、80 GB 可用磁盘空间、Python 3.8。CARLA 端口为 2000，OASIS TCS 端口为 8880。

旧版项目说明中的 PyTorch/CUDA 示例用于流模型流程，不属于当前 RMD-ES 论文的实验范围，本发布包未包含该流程。

## 复现已发布数据的分析

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-analysis.txt
python scripts/reproduce_analysis.py --plot
```

脚本默认读取 `results/`，把表格、JSON、SHA-256 清单和收敛图写入 `reproduced_outputs/`。脚本先计算每次运行的指标，再对每组 3 次独立运行取算术平均；不计算标准差或显著性检验。指标口径见 [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md)。

## 运行新的 OASIS/CARLA 搜索

在仓库外安装并配置与 CARLA 兼容的 OASIS 服务端，导入相应 `.xosc` 场景及地图，修改 `scenario/config/search_config.yaml`，确保每次仅配置一个场景和一种算法，并设置固定随机种子。高速切入场景在 OASIS 中需要使用 `lane_change_scenario.xosc` 文件名；仓库模板文件名为 `lanechange.xosc`。完成 Python 3.8 环境和依赖安装后运行：

```bash
bash run_search.sh
```

新实验会写入 `results/` 新目录。请勿覆盖仓库中已发布的 CSV 结果。

完整说明见 [`docs/EXPERIMENT_ENVIRONMENT.md`](docs/EXPERIMENT_ENVIRONMENT.md) 和 [`docs/REPRODUCING_SEARCH.md`](docs/REPRODUCING_SEARCH.md)。

## 公开与许可

本仓库按公开浏览准备，但尚未指定开源许可证。未指定许可证时，代码和数据仍受相应版权保护；公开可见不等于允许再利用或再分发。设置许可或代表全部作者及单位公开前，请先阅读 [`LICENSE_STATUS.md`](LICENSE_STATUS.md)。
