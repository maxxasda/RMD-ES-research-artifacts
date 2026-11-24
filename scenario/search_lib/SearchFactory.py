from SearchAlgorithm import NSGA2Algorithm, PSOAlgorithm, RandomAlgorithm, DGGAAlgorithm, GCE3SAlgorithm, ImprovedDGGAAlgorithm
from parse_xosc import HighWayCutInScenarioParse, CCRH, CBNAO
from SearchTools import SotifProblem, SotifMinimize


# class ConfigFactory:
#     @staticmethod
#     def create_scenario_config_instance(search_algorithm):
#         if search_algorithm == "PSO":
#             return PSOAlgorithm(config)
#         elif search_algorithm == "NSGA2":
#             return NSGA2Algorithm(config)
#         elif search_algorithm == "RANDOM":
#             return RandomAlgorithm(config)
#         else:
#             raise ValueError(f"未知的搜索类型：{search_algorithm}")

class ParseFactory:
    @staticmethod
    def create_scenario_parse_instance(class_name):
        # if class_name == 'CutInScenarioParse':
        #     return CutInScenarioParse()
        # elif class_name == 'IntersectionScenarioParse':
        #     return IntersectionScenarioParse()
        # elif class_name == 'LaneChangeScenarioParse':
        #     return LaneChangeScenarioParse()
        if class_name == 'HighWayCutInScenarioParse':
            return HighWayCutInScenarioParse()
        elif class_name == 'CCRH':
            return CCRH()
        elif class_name == 'CBNAO':
            return CBNAO()
        else:
            raise ValueError(f"未知的解析类型：{class_name}")


class SearchFactory:
    XDGGA_ABLATION_MODES = {
        "X-DGGA": "full",
        "XDGGA": "full",
        "XDGGA_FULL": "full",
        "XDGGA_NO_RISK": "no_risk_weight",
        "XDGGA_NO_ARCHIVE": "no_archive",
        "XDGGA_NO_STAGNATION": "no_stagnation",
        "XDGGA_NO_DIVERSITY": "no_diversity",
    }

    @staticmethod
    def create_searcher_instance(search_type, config):
        if search_type == "PSO":
            return PSOAlgorithm(config)
        elif search_type == "NSGA2":
            return NSGA2Algorithm(config)
        elif search_type == "RANDOM":
            return RandomAlgorithm(config)
        elif search_type == "DGGA":  # 新增DGGA分支
            return DGGAAlgorithm(config)
        elif search_type == "GCE3S":  # 新增 GCE3S 算法分支
            return GCE3SAlgorithm(config)
        elif search_type == "ImprovedDGGA":
            config.ablation_mode = getattr(config, "ablation_mode", "full")
            return ImprovedDGGAAlgorithm(config)
        elif search_type in SearchFactory.XDGGA_ABLATION_MODES:
            config.ablation_mode = SearchFactory.XDGGA_ABLATION_MODES[search_type]
            return ImprovedDGGAAlgorithm(config)
        else:
            raise ValueError(f"未知的搜索类型：{search_type}")


class ProblemFactory:
    @staticmethod
    def create_problem_instance(problem_type, config, parse_scenario):
        if problem_type == "customize":
            return SotifProblem(config, parse_scenario)
        else:
            raise ValueError(f"未知的问题类型：{problem_type}")


class MinimizeFactory:
    @staticmethod
    def create_minimize_instance(Minimize_type, problem, algorithm, config):
        if Minimize_type == "customize":
            return SotifMinimize(problem, algorithm, config)
        else:
            raise ValueError(f"未知的策略类型：{Minimize_type}")
