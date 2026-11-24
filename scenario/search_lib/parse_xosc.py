import xml.etree.ElementTree as ET
from utils import *
import numpy as np

import logger

log = logger.get_logger(__name__)


class ScenarioParse():
    def __init__(self):
        pass

    def extract_and_save_positions(self, xosc_file, ans=None):
        pass

    def create_config_yaml(self, args):
        pass


class HighWayCutInScenarioParse(ScenarioParse):
    def __init__(self):
        super().__init__()
        pass

    def extract_and_save_positions(self, xosc_file, ans=None):
        try:
            # 瑙ｆ瀽 XML 鏂囦欢
            # ans  adv: x,y,v,d
            if ans[1] < 0:
                re_lo = ans[0] + 1
            else:
                re_lo = ans[0] - 1
            npc_v = ans[1] + ans[2]
            npc_v = np.clip(npc_v, 1, 100)
            ans = {'ego_v': ans[2], 'npc_v': npc_v, 'x': re_lo, 'y': 3.5, 'z': 0, 'h': 0, 'dis': ans[0]}
            ans1 = {'ego_v': ans['ego_v'] * 3.6, 'npc_v': npc_v * 3.6, 'x': re_lo, 'y': 3.5, 'z': 0, 'h': 0,
                    'dis': ans['dis']}
            print("ans:", ans1)
            tree = ET.parse(xosc_file)
            root = tree.getroot()
            output_file = xosc_file
            # 瀹氫箟涓€涓懡鍚嶇┖闂达紙濡傛灉鏈夊懡鍚嶇┖闂撮渶瑕佹牴鎹牴鏍囩澶勭悊锛?
            namespace = {}

            # 鎻愬彇 Storyboard -> Init -> Actions -> Private -> TeleportAction -> Position
            positions = {}
            init_speed = {}
            for private_action in root.findall(".//Storyboard/Init/Actions/Private", namespace):
                entity_ref = private_action.get("entityRef")
                WorldPosition = private_action.find("./PrivateAction/TeleportAction/Position/WorldPosition", namespace)
                if WorldPosition is not None:
                    x = WorldPosition.get("x")
                    y = WorldPosition.get("y")
                    z = WorldPosition.get("z")
                    h = WorldPosition.get("h")
                    positions[entity_ref] = {"x": float(x), "y": float(y), "z": float(z), "h": float(h)}
                SpeedActionTarget = private_action.find(
                    "./PrivateAction/LongitudinalAction/SpeedAction/SpeedActionTarget/AbsoluteTargetSpeed", namespace)
                if SpeedActionTarget is not None:
                    if entity_ref == 'Ego_001':
                        # init_speed[entity_ref] = { "init_speed:":SpeedActionTarget.get('value') }
                        SpeedActionTarget.set('value', str(ans['ego_v']))
                        init_speed[entity_ref] = ans['ego_v']
                    if entity_ref == 'vehicle_001':
                        SpeedActionTarget.set('value', str(ans['npc_v']))
                        init_speed[entity_ref] = ans['npc_v']
                        # init_speed[entity_ref] = { "init_speed:":SpeedActionTarget.get('value') }
            vehicle_absolute = local_to_global(positions['Ego_001'], {'x': ans['x'], 'y': ans['y'], 'z': 0.0, 'h': 0.0})
            # trigger_absolute = local_to_global(positions['Ego_001'], {'x':ans['x'] + 15, 'y': ans['y'], 'z': 0.0, 'h': 0.0})
            if "vehicle_001" in positions:
                for private_action in root.findall(".//Storyboard/Init/Actions/Private", namespace):
                    if private_action.get("entityRef") == "vehicle_001":
                        teleport_action = private_action.find("./PrivateAction/TeleportAction/Position/WorldPosition",
                                                              namespace)
                        if teleport_action is not None:
                            teleport_action.set("x", vehicle_absolute['x'])
                            teleport_action.set("y", vehicle_absolute['y'])
                            teleport_action.set("h", vehicle_absolute['h'])
                        # SpeedActionTarget = private_action.find(
                        #     "./PrivateAction/LongitudinalAction/SpeedAction/SpeedActionTarget/AbsoluteTargetSpeed",
                        #     namespace)
                        # if SpeedActionTarget is not None:
                        #     SpeedActionTarget.set('value',str(ans['v']))
            trigger_positions = {}
            for trigger in root.findall(".//Storyboard/Story/Act/ManeuverGroup", namespace):
                entity_ref = trigger.find("./Actors/EntityRef", namespace).get("entityRef")
                story_trigger_rel_distance = trigger.find(
                    "./Maneuver/Event/StartTrigger/ConditionGroup/Condition/ByEntityCondition/EntityCondition/RelativeDistanceCondition",
                    namespace)
                if story_trigger_rel_distance is not None:
                    story_trigger_rel_distance.set('value', str(ans['dis']))
            # 鎵撳嵃鎻愬彇鐨勪綅缃俊鎭?            # log.info("Extracted Positions:")
            # for entity, pos in positions.items():
            #     log.info(f"{entity}: {pos}")
            # log.info("Extracted speed:")
            # for entity, pos in init_speed.items():
            #     log.info(f"{entity}: {pos}")
            # log.info("Extracted trigger:")
            # for entity, pos in trigger_positions.items():
            #     log.info(f"{entity}: {pos}")
            # 淇敼锛氬皢 npc vehicle 鐨勭浉瀵逛富杞︿綅缃洿璇ヤ负鍏ㄥ眬鍧愭爣浣嶇疆
            # 淇濆瓨淇敼鍚庣殑鏂囦欢
            tree.write(output_file, encoding="UTF-8", xml_declaration=True)
            log.info(f"Modified xosc file saved as: {output_file}")

        except ET.ParseError as e:
            log.info(f"Error parsing the file: {e}")
        except FileNotFoundError:
            log.info(f"File not found: {xosc_file}")
        except Exception as E:
            log.info(f"parse xosc error:{E}")

    def create_config_yaml(self, args, seed=None, note="搜索算法仿真"):
        path_items = [args.project_path, "results", args.scenario_class, args.data_class]
        search_save_path = create_time_fold(os.path.join(*path_items), "_" + args.search_algorithm + "_sotif_search")
        if seed is None:
            seed = int(np.random.random() * 100)
            np.random.seed(seed)

        config_yaml = {
            # 1. 搜索算法基础配置
            "search_type": args.search_algorithm,
            "name": "highway cut-in search for sotif",
            "scenario_class": "HighWayCutInScenarioParse",

            # 2. 搜索变量和目标函数配置
            "n_var": 3,
            "n_obj": 1,
            "ans": ["dis_to_trigger", "npc_relative_v", "ego_absolute_speed"],
            "objs": ["target_fit"],
            "add_parmes": ['collision', 'min_TTC'],

            # 3. 搜索变量范围，与 ans 顺序一一对应
            # dis_to_trigger: 触发距离，单位 m
            # npc_relative_v: NPC 相对自车速度，单位 m/s
            # ego_absolute_speed: 自车绝对速度，单位 m/s
            "xl": [5, -15, 3],
            "xu": [80, 15, 30],

            # 4. 进化搜索超参数
            "pop_size": 50,
            "n_offsprings": 50,
            "n_gen": 50,
            "seed": seed,
            "crossoverProb": 0.8,
            "crossoverEta": 20,
            "mutationEta": 25,

            # 5. 场景筛选与路径配置
            "speed_level": [4],
            "delete_speed_level": [],
            "search_save_path": search_save_path,
            "search_project_path": args.project_path,
            "simulation_scenario_path": args.simulation_scenario_path,
            "scenario_file": "lane_change_scenario.xosc",

            # 6. 仿真控制参数与备注
            "sim_sleep_time": 15,
            "note information": note,
        }
        save_yaml(os.path.join(search_save_path, 'config.yaml'), config_yaml)
        return os.path.join(search_save_path, 'config.yaml'), search_save_path


class CBNAO(ScenarioParse):
    EGO_NAME = "Ego_001"
    MOTOR_NAME = "vehicle_362"

    def __init__(self):
        super().__init__()

    @staticmethod
    def _find_private(root, entity_ref, namespace):
        for private_action in root.findall(".//Storyboard/Init/Actions/Private", namespace):
            if private_action.get("entityRef") == entity_ref:
                return private_action
        return None

    @staticmethod
    def _set_init_speed(root, entity_ref, speed_mps, namespace):
        private_action = CBNAO._find_private(root, entity_ref, namespace)
        if private_action is None:
            return

        speed_target = private_action.find(
            "./PrivateAction/LongitudinalAction/SpeedAction/SpeedActionTarget/AbsoluteTargetSpeed",
            namespace
        )
        if speed_target is not None:
            speed_target.set("value", str(speed_mps))

    def extract_and_save_positions(self, xosc_file, ans=None):
        try:
            # ans:
            # 0 ego_init_speed_kmh
            # 1 motor_trigger_distance_m
            # 2 motor_target_speed_kmh
            ego_speed = float(ans[0]) / 3.6
            motor_trigger_distance = float(ans[1])
            motor_target_speed = float(ans[2]) / 3.6

            tree = ET.parse(xosc_file)
            root = tree.getroot()
            namespace = {}

            self._set_init_speed(root, self.EGO_NAME, ego_speed, namespace)
            self._set_init_speed(root, self.MOTOR_NAME, 0.0, namespace)

            for group in root.findall(".//Storyboard/Story/Act/ManeuverGroup", namespace):
                entity_ref_node = group.find("./Actors/EntityRef", namespace)
                if entity_ref_node is None or entity_ref_node.get("entityRef") != self.MOTOR_NAME:
                    continue

                for event in group.findall("./Maneuver/Event", namespace):
                    speed_target = event.find(
                        "./Action/PrivateAction/LongitudinalAction/SpeedAction/SpeedActionTarget/AbsoluteTargetSpeed",
                        namespace
                    )
                    if speed_target is not None:
                        speed_target.set("value", str(motor_target_speed))

                    speed_dynamics = event.find(
                        "./Action/PrivateAction/LongitudinalAction/SpeedAction/SpeedActionDynamics",
                        namespace
                    )
                    if speed_dynamics is not None:
                        speed_dynamics.set("value", "2")

                    rel_distance = event.find(
                        "./StartTrigger/ConditionGroup/Condition/ByEntityCondition/EntityCondition/RelativeDistanceCondition",
                        namespace
                    )
                    if rel_distance is not None:
                        rel_distance.set("entityRef", self.EGO_NAME)
                        rel_distance.set("value", str(motor_trigger_distance))

            tree.write(xosc_file, encoding="UTF-8", xml_declaration=True)
            log.info(f"Modified CBNAO xosc file saved as: {xosc_file}")

        except ET.ParseError as e:
            log.info(f"Error parsing the file: {e}")
        except FileNotFoundError:
            log.info(f"File not found: {xosc_file}")
        except Exception as E:
            log.info(f"parse CBNAO xosc error:{E}")

    def create_config_yaml(self, args, seed=None, note="search algorithm simulation"):
        path_items = [args.project_path, "results", args.scenario_class, args.data_class]
        search_save_path = create_time_fold(os.path.join(*path_items), "_" + args.search_algorithm + "_sotif_search")
        if seed is None:
            seed = int(np.random.random() * 100)
            np.random.seed(seed)

        config_yaml = {
            "search_type": args.search_algorithm,
            "name": "cbnao search for sotif",
            "scenario_class": "CBNAO",
            "n_var": 3,
            "n_obj": 1,
            "ans": [
                "ego_init_speed_kmh",
                "motor_trigger_distance_m",
                "motor_target_speed_kmh",
            ],
            "objs": ["target_fit"],
            "add_parmes": ['collision', 'min_TTC', 'min_distance'],
            "xl": [20, 20, 18],
            "xu": [70, 80, 42],
            "pop_size": 50,
            "n_offsprings": 50,
            "n_gen": 50,
            "seed": seed,
            "crossoverProb": 0.8,
            "crossoverEta": 20,
            "mutationEta": 25,
            "speed_level": [4],
            "delete_speed_level": [],
            "search_project_path": args.project_path,
            "search_save_path": search_save_path,
            "simulation_scenario_path": args.simulation_scenario_path,
            "scenario_file": "CBNAO.xosc",
            "sim_sleep_time": 15,
            "note information": note
        }
        save_yaml(os.path.join(search_save_path, 'config.yaml'), config_yaml)
        return os.path.join(search_save_path, 'config.yaml'), search_save_path


class CCRH(ScenarioParse):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _find_private(root, entity_ref, namespace):
        for private_action in root.findall(".//Storyboard/Init/Actions/Private", namespace):
            if private_action.get("entityRef") == entity_ref:
                return private_action
        return None

    @staticmethod
    def _set_init_speed(root, entity_ref, speed_mps, namespace):
        private_action = CCRH._find_private(root, entity_ref, namespace)
        if private_action is None:
            return

        speed_target = private_action.find(
            "./PrivateAction/LongitudinalAction/SpeedAction/SpeedActionTarget/AbsoluteTargetSpeed",
            namespace
        )
        if speed_target is not None:
            speed_target.set("value", str(speed_mps))

    @staticmethod
    def _get_lane_position(root, entity_ref, namespace):
        private_action = CCRH._find_private(root, entity_ref, namespace)
        if private_action is None:
            return None
        return private_action.find("./PrivateAction/TeleportAction/Position/LanePosition", namespace)

    def extract_and_save_positions(self, xosc_file, ans=None):
        try:
            # ans:
            # 0 ego_init_speed_kmh
            # 1 ego_to_cutout_distance_m
            # 2 cutout_init_speed_kmh
            # 3 cutout_trigger_distance_m
            # 4 cutout_lanechange_time_s
            # 5 cutout_target_speed_kmh
            ego_speed = float(ans[0]) / 3.6
            ego_to_cutout_distance = float(ans[1])
            cutout_init_speed = float(ans[2]) / 3.6
            cutout_trigger_distance = float(ans[3])
            lanechange_time = float(ans[4])
            cutout_target_speed = float(ans[5]) / 3.6

            tree = ET.parse(xosc_file)
            root = tree.getroot()
            output_file = xosc_file
            namespace = {}

            self._set_init_speed(root, "Ego_001", ego_speed, namespace)
            self._set_init_speed(root, "vehicle_998", cutout_init_speed, namespace)
            self._set_init_speed(root, "vehicle_939", 0.0, namespace)

            ego_lane_position = self._get_lane_position(root, "Ego_001", namespace)
            cutout_lane_position = self._get_lane_position(root, "vehicle_998", namespace)
            if ego_lane_position is not None and cutout_lane_position is not None:
                ego_s = float(ego_lane_position.get("s"))
                base_distance = ego_s - float(cutout_lane_position.get("s"))
                direction = 1 if base_distance >= 0 else -1
                cutout_lane_position.set("s", str(ego_s - direction * ego_to_cutout_distance))

            for event in root.findall(".//Storyboard/Story/Act/ManeuverGroup/Maneuver/Event", namespace):
                speed_target = event.find(
                    "./Action/PrivateAction/LongitudinalAction/SpeedAction/SpeedActionTarget/AbsoluteTargetSpeed",
                    namespace
                )
                if speed_target is not None:
                    speed_target.set("value", str(cutout_target_speed))

                speed_dynamics = event.find(
                    "./Action/PrivateAction/LongitudinalAction/SpeedAction/SpeedActionDynamics",
                    namespace
                )
                if speed_dynamics is not None:
                    speed_dynamics.set("value", str(lanechange_time))

                lane_change_dynamics = event.find(
                    "./Action/PrivateAction/LateralAction/LaneChangeAction/LaneChangeActionDynamics",
                    namespace
                )
                if lane_change_dynamics is not None:
                    lane_change_dynamics.set("value", str(lanechange_time))

                rel_distance = event.find(
                    "./StartTrigger/ConditionGroup/Condition/ByEntityCondition/EntityCondition/RelativeDistanceCondition",
                    namespace
                )
                if rel_distance is not None:
                    rel_distance.set("value", str(cutout_trigger_distance))

            tree.write(output_file, encoding="UTF-8", xml_declaration=True)
            log.info(f"Modified CCRH xosc file saved as: {output_file}")

        except ET.ParseError as e:
            log.info(f"Error parsing the file: {e}")
        except FileNotFoundError:
            log.info(f"File not found: {xosc_file}")
        except Exception as E:
            log.info(f"parse CCRH xosc error:{E}")

    def create_config_yaml(self, args, seed=None, note="鎼滅储绠楁硶浠跨湡"):
        path_items = [args.project_path, "results", args.scenario_class, args.data_class]
        search_save_path = create_time_fold(os.path.join(*path_items), "_" + args.search_algorithm + "_sotif_search")
        if seed is None:
            seed = int(np.random.random() * 100)
            np.random.seed(seed)

        config_yaml = {
            "search_type": args.search_algorithm,
            "name": "ccrh search for sotif",
            "scenario_class": "CCRH",
            "n_var": 6,
            "n_obj": 1,
            "ans": [
                "ego_init_speed_kmh",
                "ego_to_cutout_distance_m",
                "cutout_init_speed_kmh",
                "cutout_trigger_distance_m",
                "cutout_lanechange_time_s",
                "cutout_target_speed_kmh",
            ],
            "objs": ["target_fit"],
            "add_parmes": ['collision', 'min_TTC', 'min_distance'],
            "xl": [60, 20, 60, 10, 1, 60],
            "xu": [120, 90, 120, 80, 5, 120],
            "pop_size": 50,
            "n_offsprings": 50,
            "n_gen": 50,
            "seed": seed,
            "crossoverProb": 0.8,
            "crossoverEta": 20,
            "mutationEta": 25,
            "speed_level": [4],
            "delete_speed_level": [],
            "search_project_path": args.project_path,
            "search_save_path": search_save_path,
            "simulation_scenario_path": args.simulation_scenario_path,
            "scenario_file": "CCRH.xosc",
            "sim_sleep_time": 15,
            "note information": note
        }
        save_yaml(os.path.join(search_save_path, 'config.yaml'), config_yaml)
        return os.path.join(search_save_path, 'config.yaml'), search_save_path
