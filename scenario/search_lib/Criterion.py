import carla
import py_trees
from collision_check import collision_detection
import time
import numpy as np

import copy
from utils import global_to_local, global_to_local_velocity, sqrt_sum_squares, calc_closest_distance, save_trajectory, \
    save_collision_results, copy_file

from data_struct import Location, Transform, Actor, Rotation, BoundingBox, Vector3D

import logger

log = logger.get_logger(__name__)


def _build_motion_signature(shared_data):
    if not shared_data.ego_vehicle or not shared_data.ego_vehicle.is_alive:
        return None

    ego_loc = shared_data.ego_vehicle.transform.location
    npc_locs = []
    for key in sorted(shared_data.vec_dict.keys()):
        if key.find("Ego") >= 0 or key.find("ego") >= 0:
            continue
        actor = shared_data.vec_dict[key]
        npc_locs.append((float(actor.transform.location.x), float(actor.transform.location.y)))

    if len(npc_locs) == 0:
        return None

    return {
        "ego": (float(ego_loc.x), float(ego_loc.y)),
        "npcs": tuple(npc_locs),
    }


def _motion_changed(last_sig, cur_sig, eps=0.05):
    if last_sig is None or cur_sig is None:
        return True
    if len(last_sig["npcs"]) != len(cur_sig["npcs"]):
        return True

    ego_dx = abs(cur_sig["ego"][0] - last_sig["ego"][0])
    ego_dy = abs(cur_sig["ego"][1] - last_sig["ego"][1])
    if ego_dx > eps or ego_dy > eps:
        return True

    for (cx, cy), (lx, ly) in zip(cur_sig["npcs"], last_sig["npcs"]):
        if abs(cx - lx) > eps or abs(cy - ly) > eps:
            return True

    return False


class Criterion(py_trees.behaviour.Behaviour):  # 琛屼负鏍戞鏋?鎵€鏈夎瘎浼拌妭鐐圭殑鐖剁被

    """
    Base class for all criteria used to evaluate a scenario for success/failure

    Important parameters (PUBLIC):
    - name: Name of the criterion
    - actor: Actor of the criterion
    - optional: Indicates if a criterion is optional (not used for overall analysis)
    - terminate on failure: Whether or not the criteria stops on failure

    - test_status: Used to access the result of the criterion
    - success_value: Result in case of success (e.g. max_speed, zero collisions, ...)
    - acceptable_value: Result that does not mean a failure,  but is not good enough for a success
    - actual_value: Actual result after running the scenario
    - units: units of the 'actual_value'. This is a string and is used by the result writter
    """

    def __init__(self, name):
        super().__init__(name)
        # super(Criterion, self).__init__(name)

    def update(self):
        pass  # 瀛愮被瀹炵幇鍏蜂綋閫昏緫


class ShareData():
    def __init__(self, name="ShareData", carla_parames=None):
        client = carla.Client(carla_parames['host'], carla_parames['carla_port'])
        client.set_timeout(12.0)
        self.world = client.get_world()
        self.data_collector = dict()
        self.vec_dict = dict()
        self.frame = 0
        self.ego_vehicle = Actor()
        self.running = True
        self.running_flag = False
        self.ans = []
        self.same_speed_sum = 0
        self.args = None

    def clearn(self):
        self.data_collector = dict()
        self.vec_dict = dict()
        self.frame = 0
        self.running = True
        self.running_flag = False
        self.ego_vehicle = Actor()
        self.ans = []
        self.same_speed_sum = 0


class DataAdaptor(py_trees.behaviour.Behaviour):
    def __init__(self, name="DataAdaptor", shared_data=None):
        super().__init__(name)
        self.shared_data = shared_data
        self.new_status = py_trees.common.Status.SUCCESS

    def update(self):
        self.shared_data.frame = self.shared_data.world.get_snapshot().frame
        vehicles = self.shared_data.world.get_actors().filter("*vehicle*")

        if len(vehicles) < 2:
            if self.shared_data.running_flag:
                self.shared_data.running = False
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.FAILURE

        self.shared_data.running_flag = True
        self.shared_data.vec_dict = dict()

        for vec in vehicles:
            transform = vec.get_transform()
            bbox = vec.bounding_box.extent
            velocity = vec.get_velocity()
            role_name = vec.attributes.get("role_name", "")

            location = Location(x=transform.location.x, y=-transform.location.y, z=transform.location.z)
            rotation = Rotation(yaw=(-transform.rotation.yaw / 180) * np.pi, pitch=0, roll=0)
            actor_transform = Transform(location=location, rotation=rotation)
            actor_bbox = BoundingBox(extent=Vector3D(bbox.x * 2, bbox.y * 2, bbox.z * 2))
            actor_velocity = Vector3D(x=velocity.x, y=velocity.y, z=velocity.z)

            if role_name in ["Ego_001", "Ego"] or role_name.find('Ego') >= 0 or role_name.find('ego') >= 0:
                self.shared_data.ego_vehicle.name = role_name
                self.shared_data.ego_vehicle.transform = actor_transform
                self.shared_data.ego_vehicle.bounding_box = actor_bbox
                self.shared_data.ego_vehicle.velocity = actor_velocity
                self.shared_data.ego_vehicle.timestamp = time.time()
                self.shared_data.ego_vehicle.is_alive = True
            else:
                npc_vehicle = Actor(
                    name=role_name,
                    transform=actor_transform,
                    bounding_box=actor_bbox,
                    velocity=actor_velocity,
                    timestamp=time.time(),
                    is_alive=True
                )
                self.shared_data.vec_dict[role_name] = npc_vehicle

        return self.new_status


class CollisionDetection(Criterion):
    def __init__(self, name="collision_check", shared_data=None):
        super().__init__(name)
        self.new_status = py_trees.common.Status.RUNNING
        self.collision_flag = True
        self.shared_data = shared_data

    def update(self):
        if self.new_status != py_trees.common.Status.RUNNING:
            return self.new_status

        if self.shared_data.ego_vehicle.is_alive:
            ego_vehicle = self.shared_data.ego_vehicle
            vec_dict = self.shared_data.vec_dict

            for key in vec_dict:
                if key.find('Ego') >= 0 or key.find('ego') >= 0:
                    continue

                collision = collision_detection(ego_vehicle, vec_dict[key])
                if self.collision_flag and collision:
                    self.collision_flag = False
                    relative_location = global_to_local(ego_vehicle, vec_dict[key])
                    scenario_class = getattr(getattr(self.shared_data, "args", None), "scenario_class", "")
                    if scenario_class == "CBNAO":
                        if relative_location['x'] > 1.5 and abs(relative_location['y']) < 4.0:
                            collision_fit = -20
                        else:
                            collision_fit = 10
                    elif relative_location['x'] > 3.5:
                        collision_fit = -20
                    else:
                        collision_fit = 10

                    self.shared_data.data_collector['collision'] = {
                        "ego": ego_vehicle.get_state(),
                        "vec": vec_dict[key].get_state(),
                        "ans": self.shared_data.ans,
                        "collision_fit": collision_fit
                    }
                    self.new_status = py_trees.common.Status.SUCCESS
                    self.shared_data.running = False
        return self.new_status


class FrontBrakeCalcProcess(Criterion):
    def __init__(self, name="front_brake_calc_process", shared_data=None):
        super().__init__(name)
        self.new_status = py_trees.common.Status.RUNNING
        self.shared_data = shared_data

    def update(self):
        return self.new_status

class CutInCalcProcess(Criterion):
    def __init__(self, name="collision_check", shared_data=None):
        super().__init__(name)
        self.new_status = py_trees.common.Status.RUNNING
        self.shared_data = shared_data

    def update(self):
        if self.new_status != py_trees.common.Status.RUNNING:
            return self.new_status
        # log.info("-----------------------------------",self.shared_data.frame)

        if self.shared_data.ego_vehicle:
            ego_vehicle = self.shared_data.ego_vehicle
            vec_dict = self.shared_data.vec_dict
            for key in vec_dict:
                if key.find('Ego') >= 0 or key.find('ego') >= 0:
                    continue

                # 灏唙ehicle_001 浣嶅Э鎶曞奖鍒颁富杞﹀潗鏍囩郴
                relative_location = global_to_local(ego_vehicle, vec_dict[key])

                # log.info("relative_location:",relative_location)
                if relative_location['x'] < -4 or relative_location['x'] > self.shared_data.ans[0] * 3:
                    self.shared_data.running = False
                ttc = float('inf')
                # log.info("relative_location:",relative_location)
                # TTC
                # if relative_location['x'] > 5 and abs(relative_location['y']) < 5:
                relative_distance = sqrt_sum_squares(relative_location['x'], relative_location['y'])
                # relative_distance = calc_closest_distance(ego_vehicle,vec_dict[key])
                relative_location_velocity = global_to_local_velocity(ego_vehicle, vec_dict[key])
                relative_speed = sqrt_sum_squares(relative_location_velocity['vx'], relative_location_velocity['vy'])
                # log.info("relative_speed:", relative_location_velocity['vx'],relative_speed)
                # log.info("relative_location:",relative_location['x'], relative_location['y'])
                # log.info("speed_sum:", self.shared_data.same_speed_sum)

                # 鐩稿悓杞﹂亾浼氳繘琛岀疮鍔犺绠楋紝鐢ㄤ簬涓嶅悓杞﹂亾鍦烘櫙
                # if relative_location_velocity['vx'] > 0 and relative_speed > 0:
                #     self.shared_data.same_speed_sum += 1
                # 涓嶅悓杞﹂亾浼氬仛妯悜鐩稿璺濈鍒ゆ柇
                if abs(relative_speed) * 3.6 < 5e-1 and abs(relative_location['y']) < 5:
                    self.shared_data.same_speed_sum += 1
                if self.shared_data.same_speed_sum >= 150:
                    self.shared_data.running = False
                # log.info("self.shared_data.same_speed_sum:",self.shared_data.same_speed_sum)

                if relative_location['x'] < self.shared_data.ans[0] - 0.1 and relative_location['x'] > 4 and vec_dict[
                    key].velocity.x > 1e-1:
                    # 杞﹀ご鏈濆悜閫熷害涓鸿礋锛岃瘉鏄庤溅杈嗘鍦ㄦ帴杩慹go杞︼紝璁＄畻ttc
                    if relative_location_velocity['vx'] < 0:
                        # ttc = relative_distance / np.clip(relative_speed,1e-3,1e10)
                        ttc = relative_distance / -relative_location_velocity['vx']
                        # log.info("ttc:",ttc)
                        self.shared_data.data_collector['min_TTC'] = self.shared_data.data_collector.get('min_TTC', ttc)
                        # log.info(self.shared_data.data_collector['min_TTC'],ttc)
                        if self.shared_data.data_collector['min_TTC'] > ttc:
                            self.shared_data.data_collector['min_TTC'] = ttc
                    else:
                        ttc = float('inf')
                        self.shared_data.data_collector['min_TTC'] = self.shared_data.data_collector.get('min_TTC', ttc)
                    # self.shared_data.data_collector['TTC_BUFFER'] = self.shared_data.data_collector.get('TTC_BUFFER', list())
                    # self.shared_data.data_collector['TTC_BUFFER'].append(ttc)
                # TRAJECTORY 閬嶅巻杞﹁締, 璁板綍杞﹁締杞ㄨ抗淇℃伅
                if vec_dict[key].velocity.x > 1e-1:
                    # flag = trajectory.get(key,False)
                    self.shared_data.data_collector['Trajectory'] = self.shared_data.data_collector.get('Trajectory',
                                                                                                        dict())
                    vec_status_dict = [self.shared_data.ego_vehicle.get_state()]

                    vec_status_dict[0]['ttc'] = ttc
                    if 'ego' in self.shared_data.data_collector['Trajectory'].keys():
                        self.shared_data.data_collector['Trajectory']['ego'] += vec_status_dict
                    else:
                        self.shared_data.data_collector['Trajectory']['ego'] = vec_status_dict

                    if key in self.shared_data.data_collector['Trajectory'].keys():
                        self.shared_data.data_collector['Trajectory'][key] += [vec_dict[key].get_state()]
                    else:
                        self.shared_data.data_collector['Trajectory'][key] = [vec_dict[key].get_state()]

        return self.new_status


class CCRHCalcProcess(Criterion):
    def __init__(self, name="ccrh_calc_process", shared_data=None):
        super().__init__(name)
        self.new_status = py_trees.common.Status.RUNNING
        self.shared_data = shared_data

    def update(self):
        if self.new_status != py_trees.common.Status.RUNNING:
            return self.new_status

        if self.shared_data.ego_vehicle:
            ego_vehicle = self.shared_data.ego_vehicle
            vec_dict = self.shared_data.vec_dict
            target_key = "vehicle_939" if "vehicle_939" in vec_dict else None
            if target_key is None:
                for key in vec_dict:
                    if key.find('Ego') < 0 and key.find('ego') < 0:
                        target_key = key
                        break

            if target_key is None:
                return self.new_status

            target_vehicle = vec_dict[target_key]
            relative_location = global_to_local(ego_vehicle, target_vehicle)
            relative_distance = sqrt_sum_squares(relative_location['x'], relative_location['y'])
            relative_location_velocity = global_to_local_velocity(ego_vehicle, target_vehicle)

            self.shared_data.data_collector['min_distance'] = self.shared_data.data_collector.get(
                'min_distance', relative_distance
            )
            if self.shared_data.data_collector['min_distance'] > relative_distance:
                self.shared_data.data_collector['min_distance'] = relative_distance

            ttc = float('inf')
            if relative_location['x'] > 0 and relative_location_velocity['vx'] < -1e-3:
                ttc = relative_distance / -relative_location_velocity['vx']
                self.shared_data.data_collector['min_TTC'] = self.shared_data.data_collector.get('min_TTC', ttc)
                if self.shared_data.data_collector['min_TTC'] > ttc:
                    self.shared_data.data_collector['min_TTC'] = ttc
            else:
                self.shared_data.data_collector['min_TTC'] = self.shared_data.data_collector.get('min_TTC', ttc)

            if relative_location['x'] < -10:
                self.shared_data.running = False

            self.shared_data.data_collector['Trajectory'] = self.shared_data.data_collector.get('Trajectory', dict())
            vec_status_dict = [ego_vehicle.get_state()]
            vec_status_dict[0]['ttc'] = ttc
            vec_status_dict[0]['min_distance'] = self.shared_data.data_collector['min_distance']
            if 'ego' in self.shared_data.data_collector['Trajectory'].keys():
                self.shared_data.data_collector['Trajectory']['ego'] += vec_status_dict
            else:
                self.shared_data.data_collector['Trajectory']['ego'] = vec_status_dict

            if target_key in self.shared_data.data_collector['Trajectory'].keys():
                self.shared_data.data_collector['Trajectory'][target_key] += [target_vehicle.get_state()]
            else:
                self.shared_data.data_collector['Trajectory'][target_key] = [target_vehicle.get_state()]

        return self.new_status


class CBNAOCalcProcess(Criterion):
    def __init__(self, name="cbnao_calc_process", shared_data=None):
        super().__init__(name)
        self.new_status = py_trees.common.Status.RUNNING
        self.shared_data = shared_data

    def update(self):
        if self.new_status != py_trees.common.Status.RUNNING:
            return self.new_status

        if not self.shared_data.ego_vehicle:
            return self.new_status

        ego_vehicle = self.shared_data.ego_vehicle
        vec_dict = self.shared_data.vec_dict
        target_key = "vehicle_362" if "vehicle_362" in vec_dict else None
        if target_key is None:
            for key in vec_dict:
                if key.find('Ego') < 0 and key.find('ego') < 0:
                    target_key = key
                    break

        if target_key is None:
            return self.new_status

        target_vehicle = vec_dict[target_key]
        relative_location = global_to_local(ego_vehicle, target_vehicle)
        relative_distance = sqrt_sum_squares(relative_location['x'], relative_location['y'])
        relative_location_velocity = global_to_local_velocity(ego_vehicle, target_vehicle)

        self.shared_data.data_collector['min_distance'] = self.shared_data.data_collector.get(
            'min_distance', relative_distance
        )
        if self.shared_data.data_collector['min_distance'] > relative_distance:
            self.shared_data.data_collector['min_distance'] = relative_distance

        ttc = float('inf')
        if relative_location['x'] > 0 and relative_location_velocity['vx'] < -1e-3:
            ttc = relative_distance / -relative_location_velocity['vx']
            self.shared_data.data_collector['min_TTC'] = self.shared_data.data_collector.get('min_TTC', ttc)
            if self.shared_data.data_collector['min_TTC'] > ttc:
                self.shared_data.data_collector['min_TTC'] = ttc
        else:
            self.shared_data.data_collector['min_TTC'] = self.shared_data.data_collector.get('min_TTC', ttc)

        if relative_location['x'] < -10:
            self.shared_data.running = False

        self.shared_data.data_collector['Trajectory'] = self.shared_data.data_collector.get('Trajectory', dict())
        ego_status = [ego_vehicle.get_state()]
        ego_status[0]['ttc'] = ttc
        ego_status[0]['min_distance'] = self.shared_data.data_collector['min_distance']
        if 'ego' in self.shared_data.data_collector['Trajectory'].keys():
            self.shared_data.data_collector['Trajectory']['ego'] += ego_status
        else:
            self.shared_data.data_collector['Trajectory']['ego'] = ego_status

        if target_key in self.shared_data.data_collector['Trajectory'].keys():
            self.shared_data.data_collector['Trajectory'][target_key] += [target_vehicle.get_state()]
        else:
            self.shared_data.data_collector['Trajectory'][target_key] = [target_vehicle.get_state()]

        return self.new_status


def create_root(scenario_class, shared_data):
    da = DataAdaptor(shared_data=shared_data)
    parallel_node = py_trees.composites.Parallel(
        name="Parallel",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll()
    )
    collision_detection = CollisionDetection(shared_data=shared_data)

    if scenario_class == "FrontBrake":
        calc_process = FrontBrakeCalcProcess(shared_data=shared_data)
    elif scenario_class == "CCRH":
        calc_process = CCRHCalcProcess(shared_data=shared_data)
    elif scenario_class == "CBNAO":
        calc_process = CBNAOCalcProcess(shared_data=shared_data)
    elif scenario_class in ["HighWayCutInScenarioParse", "HighWayCutInScenarioParse1"]:
        calc_process = CutInCalcProcess(shared_data=shared_data)
    else:
        calc_process = CutInCalcProcess(shared_data=shared_data)

    parallel_node.add_children([collision_detection, calc_process])
    behavior_node = py_trees.composites.Sequence(name="Root", memory=False)
    behavior_node.add_children([da, parallel_node])
    return behavior_node, shared_data

def carla_simulation_evaluation(ans=None, args=None, carla_parames=None):
    last_frame = -1
    shared_data = ShareData(carla_parames=carla_parames)
    root, shared_data = create_root(args.scenario_class, shared_data)
    shared_data.ans = ans
    shared_data.args = args
    tree = py_trees.trees.BehaviourTree(root)

    while shared_data.running:
        current_frame = shared_data.world.get_snapshot().frame
        if current_frame != last_frame:
            tree.tick()
            last_frame = current_frame
            time.sleep(0.03)

    data_collector = copy.deepcopy(shared_data.data_collector)
    shared_data.clearn()
    del shared_data
    return data_collector

if __name__ == "__main__":
    collector = carla_simulation_evaluation([1, 2, 3, 4])
    # log.info(collector)
