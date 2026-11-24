import os
import docker
import time
import logger

import requests

log = logger.get_logger(__name__)


class OasisCommand():
    def __init__(self, host='127.0.0.1', tcs_port=8880):
        # docker_client = docker.from_env()
        # carla_container = 'oasis-carla_oasis-carla_1'
        # simulation_container = 'oasis-simulation'

        # 配置TCS服务（仿真控制服务）的IP和端口
        self.tcs_ip = host
        self.tcs_port = tcs_port
        # OASIS command latency can be high after updates/startup.
        self.request_timeout_s = 480.0
        # Keep retries conservative to avoid extremely long stalls.
        self.request_retry = 0
        self.TCS_API_PATH = f"http://{self.tcs_ip}:{self.tcs_port}/tcs"
        print(self.TCS_API_PATH)

    def _post_with_retry(self, data, action_name):
        last_exc = None
        for attempt in range(self.request_retry + 1):
            try:
                return requests.post(self.TCS_API_PATH, json=data, timeout=self.request_timeout_s)
            except Exception as exc:
                last_exc = exc
                if attempt < self.request_retry:
                    time.sleep(0.3 * (attempt + 1))
        log.warning(f"{action_name} request failed: {repr(last_exc)}")
        return None

    def start_oasis(self, scenario_file):
        scenario_path = os.path.join(scenario_file)
        data = {
            "command": "LoadScenario",
            "params": {
                "filename": scenario_path,
            },
        }
        # result = await post_request_httpx(self.TCS_API_PATH, data)
        # log.info(f"LoadScenario send data {data}")
        response = self._post_with_retry(data, "LoadScenario")
        if response is None:
            return

        # if response.status_code == 200 and response.json().get("code") == 200:
        #     log.info("load scenario success\n")
        # else:
        #     log.info(
        #         f"load scenario failed, response is {response.status_code} {response.json()}\n"
        #     )
        #     return

        # 指令2：启动仿真（Start）
        data = {
            "command": "Start",
            "params": {
            },
        }
        # 轻量级校验：仅判断状态码，无强拦截（避免偶发网络波动导致仿真终止）
        response = self._post_with_retry(data, "Start")
        if response is None:
            return
        if response.status_code == 200 and response.json().get("code") == 200:
            # log.info(f"start scenario success\n")
            pass    # 启动成功，无日志输出（减少冗余）
        else:
            # log.info(
            #     f"start scenario failed, response is {response.status_code} {response.json()}\n"
            # )
            return  # 启动失败，直接返回（由上层逻辑重试）

    def stop_oasis(self):
        data = {
            "command": "Stop",
            "params": {}
        }
        response = self._post_with_retry(data, "Stop")
        time.sleep(1)   # 等待1秒：让CARLA释放仿真资源（避免下一次仿真端口占用）
        # if response.status_code == 200 and response.json().get("code") == 200:
        #     log.info(f"stop scenario successfully\n")
        # else:
        #     log.info(f"stop scenario failed, response is {response.status_code} {response.json()} \n")

        # 单步执行
        # step_data = {
        #     "command": "Step",
        #     "params": {
        #       "size":100
        #     },
        # }
        # response = requests.post(self.TCS_API_PATH, json=step_data)
        # if response.status_code == 200 and response.json().get("code") == 200:
        #     # log.info(f"Step scenario success")
        # else:
        #     # log.info(
        #         f"Step scenario failed, response is {response.status_code} {response.json()}"
        #     )
        #     return

        # 暂停
        # pause_data = {
        #     "command": "Pause",
        #     "params": {
        #     },
        # }
        # response = requests.post(self.TCS_API_PATH, json=pause_data)
        # if response.status_code == 200 and response.json().get("code") == 200:
        #     # log.info(f"Pause scenario success")
        # else:
        #     # log.info(
        #         f"Pause scenario failed, response is {response.status_code} {response.json()}"
        #     )
        #     return

        # 终止指令
        # stop_command = {
        #     "command": "Stop",
        #     "params": {
        #     },
        # }
        # response = requests.post(self.TCS_API_PATH, json=stop_command)
        # if response.status_code == 200 and response.json().get("code") == 200:
        #     # log.info(f"Stop scenario success")
        # else:
        #     # log.info(
        #         f"Stop scenario failed, response is {response.status_code} {response.json()}"
        #     )
        #     return

    # def restart_oasis_carla_simulation():
    #     carla = docker_client.containers.get(carla_container)
    #     simulation = docker_client.containers.get(simulation_container)
    #     carla.restart()
    #     simulation.restart()
    #     time.sleep(40)
    #     return
    #
    #
    # def restart_oasis_server(oasis_path):
    #     os.system( os.path.join(oasis_path,'stop.sh'))
    #     os.system( os.path.join(oasis_path,'start.sh'))
    #     time.sleep(3)
    #
    # def oasis_scenario_exec(config):
    #
    #     log.info("start ",config["scenario_path"])
    #     start_oasis(config["scenario_path"])
    #     time.sleep(config['sim_sleep_time'])
    #     stop_oasis()
    #


# start_oasis('zj_right_cut_in.xosc')
# time.sleep(15)
# stop_oasis()
# log.info("1111111111")


if __name__ == '__main__':
    import carla

    host = "172.16.19.205"
    carla_port = 2000
    tcs_port = 8880
    oasis = OasisCommand(host, tcs_port)
    oasis.start_oasis('front_brake.xosc')
    # time.sleep(15)
    # oasis.stop_oasis()
    client = carla.Client(host, carla_port)
    client.set_timeout(12.0)
    world = client.get_world()
    while 1:
        frame = world.get_snapshot().frame
        # new_status = py_trees.common.Status.RUNNING
        vehicles = world.get_actors().filter("*vehicle*")
        print("frame:", frame)
        print("vehicles:", vehicles)