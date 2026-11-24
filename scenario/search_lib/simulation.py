import multiprocessing
import os

from tqdm import tqdm
import random
import time
import numpy as np

from utils import *
from collision_check import collision_detection, plot_vehicles
from oasis_command import OasisCommand
from Criterion import carla_simulation_evaluation
import pandas as pd
import carla

import threading
from functools import partial

import logger

log = logger.get_logger(__name__)


def _eval_worker(ans, config, carla_parames, result_queue, progress_total=None, progress_offset=0):
    try:
        data = carla_simulation_evaluation(ans, config, carla_parames)
        result_queue.put({"ok": True, "data": data, "error": ""})
    except Exception as exc:
        result_queue.put({"ok": False, "data": {}, "error": repr(exc)})


def get_local_location(vehicle, location) -> carla.Location:
    """将全局坐标系下的坐标转到局部坐标系下

    Args:
        location (Location): 待变换的全局坐标系坐标
    """
    res = np.array(vehicle.get_transform().get_inverse_matrix()).dot(
        np.array([location.x, location.y, location.z, 1])
    )

    return carla.Location(x=res[0], y=res[1], z=res[2])

    # # log.info("config:",config)
    # for loop in range(10):
    #     # log.info("********************************")
    #     start_oasis(config["scenario_file"])
    #     time.sleep(config["sim_sleep_time"])
    #     stop_oasis()
    #     time.sleep(0.5)
    #     if loop > 5 and not os.path.exists(os.path.join(config["search_save_path"], "pid_fit.csv")):
    #         log.info("重启 oasis  carla 和 simulattion 服务!!!!!!")
    #         restart_oasis_server(config["oasis_path"])
    #         time.sleep(3)
    #         continue
    #
    #     if os.path.exists(os.path.join(config["search_save_path"], "pid_fit.csv")):
    #         break
    #     else:
    #         log.info("仿真执行器执行失败！！！！！！，restart carla and simulation")
    #         restart_oasis_carla_simulation()
    #         # stop_oasis()
    #         time.sleep(5)
    #         continue
    # log.info("!!!!!!!!!!!!!!!!!!!!!!!!!")


class SIMULATION:
    def __init__(self, config, parse_scenario, carla_parames):
        # 步骤1：保存核心配置与场景解析器
        self.config = config  # 搜索配置对象（包含n_var、xl、xu、场景路径等）
        self.config.add_parmes = [
            item for item in getattr(self.config, "add_parmes", [])
            if item != "similarity"
        ]
        self.parse_scenario = parse_scenario  # 场景解析器（如FrontBrake）

        # 步骤2：保存CARLA仿真参数（注释的connect_carla是预留的CARLA连接方法）
        # self.connect_carla(carla_parames['host'],carla_parames['carla_port'])  # 预留：连接CARLA服务器
        self.carla_parames = carla_parames  # CARLA参数（host/port/scenario_path等）
        self.simulation_scenario_path = carla_parames['scenario_path']  # 仿真场景文件路径

        # 步骤3：初始化OasisCommand（CARLA仿真指令客户端）
        self.oasis_command = OasisCommand(
            carla_parames['host'],        # CARLA主机地址（如localhost）
            tcs_port=carla_parames['tcs_port']  # 仿真控制端口
        )

    def connect_carla(self, host='localhost', port=2000):
        client = carla.Client(host, port)
        client.set_timeout(10.0)
        self.world = client.get_world()

    def oasis_scenario_exec(self, sim_config):

        self.oasis_command.start_oasis(sim_config['scenario_file'])
        time.sleep(sim_config['sim_sleep_time'])
        self.oasis_command.stop_oasis()

    def save_sim_process(self, x, fit, add_parmes_tmp, step=0):
        # 1. 初始化DataFrame，列名包含step+决策变量+目标指标+附加指标
        pd_config = {}
        for i in ["step"] + self.config.ans + self.config.objs + self.config.add_parmes:
            pd_config[i] = []
        self.search_collector = pd.DataFrame(pd_config)


        # 2. 遍历每一组参数，拼接step+参数+指标数据
        for i in range(np.size(x, 0)):
            obss = []    # 决策变量（如触发距离、速度）
            for obs in x[i]:
                obss.append(obs)
            tmp_fit = []
            # 拼接目标指标（ans_fit）和附加指标（collision/min_TTC等）
            for objs in self.config.objs + self.config.add_parmes:
                if objs in fit.keys():
                    tmp_fit.append(fit[objs][i])
                if objs in add_parmes_tmp.keys():
                    tmp_fit.append(add_parmes_tmp[objs][i])

            tmp = [step] + obss + tmp_fit
            self.search_collector.loc[len(self.search_collector.index)] = tmp

        # 3. 追加写入CSV文件（sotif_search_results.csv）
        search_result_path = os.path.join(self.config.search_save_path, "sotif_search_results.csv")
        if os.path.exists(search_result_path):
            self.search_collector.to_csv(search_result_path, mode="a", header=False, index=False)
        else:
            # 文件不存在，创建文件并写入（写入列名）
            self.search_collector.to_csv(search_result_path,
                                         mode="w", header=True, index=False)
        # self.save_search_history()
        # 4. 清空DataFrame，释放内存
        self.search_collector.drop(self.search_collector.index, inplace=True)

    def sim_exec(self, x, step=0, save_flag=True, progress_total=None, progress_offset=0):
        # 步骤1：初始化指标存储容器
        fit = dict()  # 存储目标指标（ans_fit）
        for obj in self.config.objs:
            fit[obj] = []
        pop_index = 0  # 粒子索引（每代的参数组序号）
        target_speed = 0
        add_params = {}  # 存储附加指标（collision/min_TTC等）
        for add in self.config.add_parmes:
            add_params[add] = []

        # 步骤2：遍历PSO生成的每一组参数（x是粒子群，每组ans对应一组场景参数）
        loop_total = int(progress_total) if progress_total is not None else int(np.size(x, 0))
        loop_initial = int(progress_offset) if progress_total is not None else 0
        for ans in tqdm(x, total=loop_total, initial=loop_initial, leave=False, dynamic_ncols=True):
            # ans = [50,10,10]
            # log.info(f">>>>>>>>>>>>step:{self.step}--->index:{pop_index}")
            pop_index += 1
            if isinstance(ans, list):
                pass
            else:
                ans = [round(num, 4) for num in ans.tolist()]
            log.info('\n')
            log.info(f"ans:{ans}")  # 打印当前仿真的参数（如[23.5662,7.6201,5.59]）
            log.info('\n')

            pd_config = {}
            for i in self.config.ans:
                pd_config[i] = []
            ans_parm = pd.DataFrame(pd_config)
            ans_parm.loc[len(ans_parm.index)] = ans
            # scenario_file = os.path.join(self.config.simulation_scenario_path, self.config.scenario_file)

            # 步骤3：修改场景文件（xosc）的参数
            scenario_file = os.path.join(self.simulation_scenario_path, self.config.scenario_file)
            self.parse_scenario.extract_and_save_positions(scenario_file, ans)  # 核心：用ans更新xosc的触发距离/速度等

            # 步骤4：启动Oasissim仿真（异步线程执行，避免阻塞）
            sim_config = dict()
            sim_config['scenario_file'] = self.config.scenario_file
            sim_config['search_save_path'] = self.config.search_save_path
            # sim_config['oasis_path'] = self.config.oasis_path
            sim_config['sim_sleep_time'] = self.config.sim_sleep_time
            # sim_config['sim_sleep_time'] = 50

            # threading.Thread：Python 内置的线程类，用于创建一个新的 “子线程”（区别于执行sim_exec的 “主线程”）
            thread1 = threading.Thread(target=partial(self.oasis_scenario_exec, sim_config))    # 创建异步执行线程

            # thread1 = threading.Thread(target=self.oasis_scenario_exec, args=(sim_config ))
            thread1.start()     # 启动子线程执行任务
            # self.oasis_scenario_exec(sim_config)
            # sim_process = multiprocessing.Process(target=self.oasis_scenario_exec, args=(sim_config,))
            # sim_process.start()
            time.sleep(1)   # 等待仿真启动

            # collision ,collision_fit= self.cal_fit(ans,save_flag)
            # 步骤5：提取仿真结果（碰撞、TTC、轨迹等）
            eval_timeout_s = float(getattr(self.config, "single_scene_timeout_s", 480.0))
            data_collection = {}

            # Pool worker is daemonic, it can not spawn child processes.
            # In that case, evaluate inline to avoid BrokenPipe/empty queue issues.
            if multiprocessing.current_process().daemon:
                try:
                    data_collection = carla_simulation_evaluation(ans, self.config, self.carla_parames)
                except Exception as exc:
                    log.warning(f"[watchdog] inline scenario eval failed, skip sample: {repr(exc)}")
                    data_collection = {"scene_error": repr(exc)}
            else:
                result_queue = multiprocessing.Queue(maxsize=1)
                eval_process = multiprocessing.Process(
                    target=_eval_worker,
                    args=(ans, self.config, self.carla_parames, result_queue),
                )
                eval_process.start()
                eval_process.join(timeout=eval_timeout_s)

                if eval_process.is_alive():
                    eval_process.terminate()
                    eval_process.join(timeout=5)
                    log.warning(f"[watchdog] scenario timeout (> {eval_timeout_s}s), skip sample ans={ans}")
                    data_collection = {"scene_timeout": f"single_scene_timeout>{eval_timeout_s}s"}
                else:
                    try:
                        if not result_queue.empty():
                            payload = result_queue.get_nowait()
                            if payload.get("ok", False):
                                data_collection = payload.get("data", {}) or {}
                            else:
                                err_msg = payload.get("error", "")
                                log.warning(f"[watchdog] scenario eval failed, skip sample: {err_msg}")
                                data_collection = {"scene_error": err_msg}
                        else:
                            log.warning("[watchdog] scenario eval returned empty payload, skip sample.")
                            data_collection = {"scene_error": "empty_result_queue"}
                    except Exception as exc:
                        log.warning(f"[watchdog] read eval payload failed: {repr(exc)}")
                        data_collection = {"scene_error": repr(exc)}
                    finally:
                        try:
                            result_queue.close()
                            result_queue.join_thread()
                        except Exception:
                            pass
            self.oasis_command.stop_oasis()  # 强制停止仿真，避免资源占用

            # 步骤6：计算核心风险指标
            # 6.1 碰撞标记与碰撞拟合值
            collision = "collision" in data_collection.keys()

            if collision:
                collision_fit = data_collection['collision']['collision_fit']
            else:
                collision_fit = 0
            # print(collision_fit, collision, 1111111111111111222222222222222734767863786419864893621949238648963128964896389289164)
            # False 0
            # True 要么-20 要么10，根据relative_location['x'] > 3.5来决定

            # 6.2 最小TTC（裁剪到[0,100]）
            min_ttc = np.clip(data_collection.get("min_TTC", 100), 0, 100)
            min_distance = np.clip(data_collection.get("min_distance", 100), 0, 100)

            # 6.3 目标函数值（ans_fit）：核心危险度指标
            ans_fit = min_ttc + collision_fit
            log.info(f"ans_fit:{ans_fit},min_TTC:{min_ttc},collision_fit:{collision_fit}")

            # 步骤7：存储指标到容器
            if "collision" in add_params:
                add_params["collision"].append(collision)
            if "min_TTC" in add_params:
                add_params["min_TTC"].append(min_ttc)
            if "min_distance" in add_params:
                add_params["min_distance"].append(min_distance)

            fit[self.config.objs[0]].append(ans_fit)

            # 步骤8：保存关键场景（碰撞场景/轨迹）
            # 8.1 构建场景文件名（迭代代次+参数+指标）
            file_name = self.config.search_type + '_' + str(step) + '_'
            for ele_ans in ans:
                file_name += "_" + str(ele_ans)
            file_name += "_ans_fit:" + str(ans_fit)

            os.makedirs(self.config.critical_scenario_path, exist_ok=True)
            collision_file = os.path.join(self.config.critical_scenario_path, file_name)

            # 8.2 保存碰撞结果（若碰撞）
            if collision:
                # fit[self.config.objs[1]].append(data_collection['collision']['collision_fit'])
                collision_fit = data_collection['collision']['collision_fit']
                save_collision_results(collision_file, collision, ans,
                                       data_collection['collision']['ego'],
                                       data_collection['collision']['vec'], True)

            # 8.3 保存轨迹数据（若有）
            if 'Trajectory' in data_collection.keys():
                save_trajectory(collision_file, data_collection['Trajectory'])
            # if 'TTC_BUFFER' in data_collection.keys():
            #     save_trajectory(critical_file_name, data_collection['TTC_BUFFER'])
            # else:
            #     fit[self.config.objs[1]].append(0)

        # 步骤9：保存当前代的所有结果到CSV
        if save_flag:
            self.save_sim_process(x, fit, add_params, step)

        # 步骤10：返回目标函数值，供PSO算法迭代优化
        return fit
