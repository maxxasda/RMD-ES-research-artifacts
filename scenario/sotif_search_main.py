import os
import sys
sys.path.append( os.path.join(os.path.dirname(__file__)) )
import pandas as pd
import argparse
from search_lib.utils import *
import numpy as np
import datetime

from search_lib.SearchFactory import ParseFactory,ProblemFactory,MinimizeFactory,SearchFactory
import logger
log = logger.setup_app_level_logger()



if __name__ == '__main__':
    # 解析命令行参数，默认加载 ./config/search_config.yaml 配置
    log_parser = argparse.ArgumentParser()  # 创建参数解析器
    log_parser.add_argument(
        "--config",
        type=str,
        default='./config/search_config.yaml',  # 默认配置文件路径
    )
    config_args = log_parser.parse_args()  # 解析命令行参数（如运行时指定--config自定义路径）
    search_config = read_yaml(config_args.config) #  读取YAML配置文件内容 （调用search_lib.utils中的方法）
    # print(1, search_config)



    args = Args()  # 创建参数对象（Args为自定义类，用于统一管理参数）

    # 设置项目根路径：优先从环境变量SOTIF_PROJECT_PATH获取，否则用配置文件中的默认值
    args.project_path = os.environ.get("SOTIF_PROJECT_PATH", search_config['project_path'])
    # 设置仿真场景文件存储路径（从配置文件读取）
    args.simulation_scenario_path = search_config['simulation_scenario_path']
    # 设置数据类别（当前仅用search，从配置文件第一个元素取）
    args.data_class = search_config['data_class'][0]

    # 遍历所有场景类型
    for scenario_class in search_config['scenario_classes']:
        # 遍历所有搜索算法
        for index in range(len(search_config['search_algorithms'])):
            # 设置当前循环的场景类型（如FrontBrake）
            args.scenario_class = scenario_class
            # 设置当前循环的搜索算法（如PSO）
            args.search_algorithm = search_config['search_algorithms'][index]


            # 创建场景解析器实例（根据场景类型动态选择解析逻辑）
            parse_scenario = ParseFactory.create_scenario_parse_instance(args.scenario_class)

            # 生成该场景的专属配置文件和项目存储路径
            # 参数：基础参数、随机种子（从配置文件取，当前为null）、备注信息
            seed_value = search_config['seed'][index] if index < len(search_config['seed']) else None
            config_yaml_path, search_project = parse_scenario.create_config_yaml(
                args,
                seed_value,   # print的结果为None
                note="搜索算法数据仿真" + scenario_class
            )
            # print(2, config_yaml_path)
            # print(3, search_project)
            # 读取生成的场景专属配置文件
            yaml_config = read_yaml(config_yaml_path)
            # print(4, yaml_config)

            # 创建搜索配置对象，合并场景配置与仿真任务配置
            config = SearchConfig(yaml_config, search_config['sim_task'])
            # print(search_config['sim_task'])


            # 设置搜索结果保存路径：项目路径 + 场景文件名（不含后缀）
            config.search_save_path = os.path.join(search_project, config.scenario_file.split('.')[0])

            # 设置关键场景保存路径（在结果路径下创建critical_scenario子目录）
            config.critical_scenario_path = os.path.join(config.search_save_path, "critical_scenario")
            # 确保目录存在（不存在则创建，exist_ok=True避免重复创建报错   ）
            os.makedirs(config.critical_scenario_path, exist_ok=True)

            # 将当前场景的配置文件备份到结果目录（方便后续追溯参数）
            save_yaml(
                os.path.join(config.search_save_path, 'config.yaml'),  # 备份路径
                yaml_config  # 要备份的配置内容
            )

            # 创建问题实例（定义搜索目标和约束，如"寻找导致车辆制动失效的场景参数"）
            problem = ProblemFactory.create_problem_instance('customize', config, parse_scenario)

            # 创建搜索算法实例（根据配置的算法类型，当前为PSO粒子群优化）
            algorithm = SearchFactory.create_searcher_instance(config.search_type, config)

            # 创建优化器实例（整合问题和算法，负责执行搜索流程）
            minimize = MinimizeFactory.create_minimize_instance('customize', problem, algorithm, config)



            # 运行搜索（核心步骤：算法根据问题定义在参数空间中搜索关键场景） 执行搜索算法
            minimize.run()

            # 获取搜索结果：X是决策变量（如车辆速度、距离等场景参数），F是目标函数值（如危险程度）
            X = minimize.res.X  # 最优决策变量（如[80, 5, 0.2] → 车速 80km/h、跟车距离 5m、制动延迟 0.2s）
            F = minimize.res.F  # 最优目标函数值



            # 定义结果表格的列名：约束条件（ans）+ 目标函数（objs）
            columns = []
            for i in config.ans + config.objs:
                columns.append(i)

            # 处理结果格式（兼容单条结果和多条结果的情况）
            if isinstance(X[0], float):  # 若只有一条结果（X是一维数组）
                # 扩展维度为二维数组，与F拼接后转为DataFrame
                results = pd.DataFrame(
                    np.append(np.expand_dims(X, axis=0), np.expand_dims(F, axis=0), axis=1).tolist(),
                    columns=columns
                )

            else:  # 若有多条结果（X是二维数组）
                results = pd.DataFrame(
                    np.append(X, F.tolist(), axis=1).tolist(),  # X和F按列拼接
                    columns=columns
                )

            # 保存搜索结果到当前运行目录
            results.to_csv(os.path.join(config.search_save_path, "sotif_fit_results.csv"))

            # 打印日志，提示任务完成
            log.info("任务完成！！！！！！！！！！！！！！")

            current_time = datetime.datetime.now()

            # 打印当前时间
            print("当前时间：", current_time)

