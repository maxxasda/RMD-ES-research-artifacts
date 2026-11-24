import os
import math
import yaml
import json
from datetime import datetime
from collision_check import plot_vehicles
from shutil import copy
# 使用shapely的nearest_points函数获取最近点
from shapely.ops import nearest_points
from shapely.geometry import Polygon, Point
from data_struct import Actor


# import logger

# logger.logger.log()
# log = logger.get_logger(__name__)

class SearchConfig():
    def __init__(self, yaml_config, sim_task):
        self.scenario_class = yaml_config['scenario_class']
        self.search_type = yaml_config['search_type']
        self.n_var = yaml_config['n_var']
        self.n_obj = yaml_config['n_obj']
        self.objs = yaml_config['objs']
        self.ans = yaml_config['ans']
        self.add_parmes = yaml_config['add_parmes']
        self.xl = yaml_config['xl']
        self.xu = yaml_config['xu']
        self.pop_size = yaml_config['pop_size']
        self.n_offsprings = yaml_config['n_offsprings']
        self.n_gen = yaml_config['n_gen']
        self.seed = yaml_config['seed']
        self.crossoverProb = yaml_config['crossoverProb']  # 变异概率
        self.crossoverEta = yaml_config['crossoverEta']
        self.mutationEta = yaml_config['mutationEta']
        self.verbose = True

        # sotif sotif project 目录
        self.search_project_path = yaml_config['search_project_path']
        # sotif 数据存储路径
        self.search_save_path = yaml_config['search_save_path']

        # oasis的安装目录级目录 搜索场景文件会放在该目录下 /oasis_data/temp
        self.simulation_scenario_path = yaml_config['simulation_scenario_path']
        # 执行的场景文件名
        self.scenario_file = yaml_config['scenario_file']
        # 关键场景路径
        self.critical_scenario_path = None
        self.sim_sleep_time = yaml_config['sim_sleep_time']
        # task config
        self.sim_task = sim_task


def create_time_fold(base_fold, suffix):
    # 在base_fold（如 oasis-data） 目录下创建 2025_02_10_18_59_sotif_search 文件,基于当前时间创建对应的目录
    now = datetime.now()
    # 格式化为 "年月日时分" 格式
    formatted_time = now.strftime("%Y_%m_%d_%H_%M")

    # search_save_path = os.path.join(os.path.dirname(__file__) , formatted_time + "_nsga2_search" )
    search_save_path = os.path.join(base_fold, formatted_time + suffix)
    # # log.info("search_save_path:::",search_save_path)
    os.makedirs(search_save_path, exist_ok=True)
    return search_save_path


# 接收 YAML 文件路径，读取文件内容并解析为Python 字典 / 列表（与 YAML 的层级结构一一对应），返回给调用方使用。
def read_yaml(config_yaml_path):
    with open(config_yaml_path, 'r') as file:
        yaml_config = yaml.load(file, Loader=yaml.FullLoader)
    return yaml_config


# 保存 Python 数据为 YAML 文件
def save_yaml(search_save_path, config_yaml):
    with open(search_save_path, 'w', encoding='utf-8') as file:
        yaml.dump(config_yaml, file, allow_unicode=True)


def find_file(file_name, start_path):
    """
    在指定路径寻找文件，如果不存在则在上一级目录寻找，最多返回一次上级目录。
    :param file_name: 要查找的文件名
    :param start_path: 开始查找的路径
    :return: 文件的绝对路径，如果未找到则返回 None
    """
    # 检查当前目录
    current_path = os.path.abspath(start_path)
    file_path = os.path.join(current_path, file_name)
    if os.path.isfile(file_path):
        return file_path

    # 检查上一级目录
    parent_path = os.path.dirname(current_path)
    file_path = os.path.join(parent_path, file_name)
    if os.path.isfile(file_path):
        return file_path

    # 未找到文件
    return None


def global_to_local_velocity(Ego, vehicle):
    """
    将 vehicle 速度 的全局坐标投影到 Ego_001 的局部坐标系中。

    :param Ego_001: dict, ego 车辆的位置和朝向 {'vx': float, 'vy': float, 'z': float,}
    :param vehicle: dict, 被投影车辆的位置和朝向 {'vx': float, 'vy': float, 'z': float}
    :return: dict, vehicle 在 Ego_001 局部坐标系中的位置和朝向
    """
    if isinstance(Ego, Actor):
        ex, ey, eh = Ego.velocity.x, Ego.velocity.y, Ego.transform.rotation.yaw
        vx, vy, vh = vehicle.velocity.x, vehicle.velocity.y, vehicle.transform.rotation.yaw

    else:
        # ex, ey ,eh = Ego.velocity.x, Ego.velocity.y, Ego.transform.rotation.yaw
        # vx, vy ,vh = vehicle.velocity.x, vehicle.velocity.y, vehicle.transform.rotation.yaw

        ex, ey, eh = Ego['vx'], Ego['vy'], Ego['h']
        vx, vy, vh = vehicle['vx'], vehicle['vy'], vehicle['h']

    # 计算相对速度（平移）
    dx = vx - ex
    dy = vy - ey
    dh = eh
    # 将相对位置旋转到局部坐标系（逆旋转）
    local_vx = dx * math.cos(-dh) - dy * math.sin(-dh)
    local_vy = dx * math.sin(-dh) + dy * math.cos(-dh)
    return {"vx": local_vx, "vy": local_vy}


def global_to_local(vehicle1, vehicle2):
    """
    将 vehicle 的全局坐标投影到 vehicle1 的局部坐标系中。

    :param vehicle1: dict, ego 车辆的位置和朝向 {'x': float, 'y': float, 'z': float, 'h': float}
    :param vehicle: dict, 被投影车辆的位置和朝向 {'x': float, 'y': float, 'z': float, 'h': float}
    :return: dict, vehicle 在 vehicle1 局部坐标系中的位置和朝向
    """
    # 提取 vehicle1 和 vehicle 的信息
    if isinstance(vehicle1, Actor):
        ex, ey, ez, eh = float(vehicle1.transform.location.x), float(vehicle1.transform.location.y), float(
            vehicle1.transform.location.z), float(vehicle1.transform.rotation.yaw)
        vx, vy, vz, vh = float(vehicle2.transform.location.x), float(vehicle2.transform.location.y), float(
            vehicle2.transform.location.z), float(vehicle2.transform.rotation.yaw)
    else:
        ex, ey, ez, eh = vehicle1['x'], vehicle1['y'], vehicle1['z'], vehicle1['h']
        vx, vy, vz, vh = vehicle2['x'], vehicle2['y'], vehicle2['z'], vehicle2['h']
    # 计算相对位置（平移）
    dx = vx - ex
    dy = vy - ey
    dz = vz - ez
    # 将相对位置旋转到局部坐标系（逆旋转）
    local_x = dx * math.cos(-eh) - dy * math.sin(-eh)
    local_y = dx * math.sin(-eh) + dy * math.cos(-eh)
    local_z = dz  # z 轴不受旋转影响

    # 计算相对航向角
    local_h = vh - eh

    # 规范化航向角到 [-π, π]
    while local_h > math.pi:
        local_h -= 2 * math.pi
    while local_h < -math.pi:
        local_h += 2 * math.pi

    return {"x": local_x, "y": local_y, "z": local_z, "h": local_h}


def local_to_global(Ego_001, local_vehicle):
    """
    将相对坐标和航向转换为全局绝对坐标系。
    :param Ego_001: dict, ego 车辆的绝对坐标和朝向 {'x': float, 'y': float, 'z': float, 'h': float}
    :param local_vehicle: dict, vehicle_001 的相对坐标和朝向 {'x': float, 'y': float, 'z': float, 'h': float}
    :return: dict, vehicle_001 的全局绝对坐标和朝向
    """
    # 提取 Ego_001 和 local_vehicle 的信息
    if isinstance(Ego_001, Actor):
        ex, ey, ez, eh = float(Ego_001.transform.location.x), float(Ego_001.transform.location.y), float(
            Ego_001.transform.location.z), float(Ego_001.transform.rotation.yaw)
        lx, ly, lz, lh = float(local_vehicle.transform.location.x), float(local_vehicle.transform.location.y), float(
            local_vehicle.transform.location.z), float(local_vehicle.transform.rotation.yaw)
    else:
        ex, ey, ez, eh = Ego_001['x'], Ego_001['y'], Ego_001['z'], Ego_001['h']
        lx, ly, lz, lh = local_vehicle['x'], local_vehicle['y'], local_vehicle['z'], local_vehicle['h']
    # 将相对坐标旋转回全局坐标系
    global_x = lx * math.cos(eh) - ly * math.sin(eh) + ex
    global_y = lx * math.sin(eh) + ly * math.cos(eh) + ey
    global_z = lz + ez  # z 轴直接加偏移

    # 计算全局航向角
    global_h = lh + eh

    # 规范化航向角到 [-π, π]
    while global_h > math.pi:
        global_h -= 2 * math.pi
    while global_h < -math.pi:
        global_h += 2 * math.pi

    return {"x": str(global_x), "y": str(global_y), "z": str(global_z), "h": str(global_h)}


def save_collision_results(folder_path, collision, ans, ego_vehicle, vec_dict, save_result=False):
    if collision:
        collision_flag = "collision_ans"
    else:
        collision_flag = "not_collision_ans"
    # file_name = ''
    # for ele_ans in ans:
    #     file_name += "_" + str(ele_ans)
    # os.makedirs(folder_path, exist_ok=True)
    # collision_file = os.path.join(folder_path, collision_flag )
    collision_file = folder_path + collision_flag
    plot_vehicles(ego_vehicle, vec_dict, collision, collision_file + '.png', save=save_result)
    # return collision_file


def convert_json(critical_file_name, trajectory):
    # {'name': 'vehicle_001', 'type': '',
    #  'transform': {'location': {'x': 240.2030792236328, 'y': -8.381999969482422, 'z': -0.0008935928344726562},
    #                'rotation': {'pitch': 0, 'yaw': -0.0, 'roll': 0}},
    #  'velocity': {'x': -0.000602781365159899, 'y': -1.7830353726822068e-07, 'z': -0.022034969180822372},
    #  'bounding_box': {'extent': {'x': 1.852684736251831, 'y': 0.8943392634391785, 'z': 0.7745251059532166}},
    #  'is_alive': True, 'timestamp': 1739520381.9131176}
    data = {
        role_name: [
            {
                "timestamp": point["timestamp"],
                "motion": "MS_MOVING",
                "type": "car",
                "rotation": [
                    point['transform']['rotation']['yaw'],
                    0.0,
                    0.0
                ],
                "location": [
                    point['transform']['location']['x'],
                    point['transform']['location']['y'],
                    point['transform']['location']['z']
                ],
                "velocity": [
                    point['velocity']['x'],
                    point['velocity']['y']
                ],
                "ttc": [point.get("ttc", float('inf'))]
            }
            for point in trajectory[role_name]  # 创建4个元素
        ] for role_name in trajectory.keys()
    }

    # 文件路径

    file_path = critical_file_name + '.json'
    # 将数据保存为 JSON 文件
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

    # log.info(f"JSON 文件已生成：{file_path}")


def save_trajectory(critical_file_name, trajectory):
    convert_json(critical_file_name, trajectory)


def copy_file(from_path, to_path):
    copy(from_path, to_path)


def sqrt_sum_squares(x, y):
    return math.sqrt(x ** 2 + y ** 2)


def rotate_point(x, y, angle):
    """根据给定的角度旋转点 (x, y)"""
    rad = math.radians(angle)
    cos_angle = math.cos(rad)
    sin_angle = math.sin(rad)
    return (cos_angle * x - sin_angle * y, sin_angle * x + cos_angle * y)


def get_vehicle_corners(vehicle):
    """根据车辆的中心点、朝向、长宽计算四个角点"""
    x, y, angle, length, width = vehicle.transform.location.x, vehicle.transform.location.y, vehicle.transform.rotation.yaw, vehicle.bounding_box.extent.x, vehicle.bounding_box.extent.y

    # 车辆的四个角点相对于中心的偏移
    half_length = length / 2
    half_width = width / 2

    # 车辆四个角点相对于中心的偏移坐标
    corners = [
        (-half_length, -half_width),  # 左下
        (half_length, -half_width),  # 右下
        (half_length, half_width),  # 右上
        (-half_length, half_width)  # 左上
    ]

    # 旋转每个角点
    rotated_corners = []
    for corner in corners:
        rotated_corners.append((
            x + rotate_point(corner[0], corner[1], angle)[0],
            y + rotate_point(corner[0], corner[1], angle)[1]
        ))

    return rotated_corners


def closest_points_shapely(vehicle_a, vehicle_b):
    """返回车辆a和车辆b之间的最近点"""
    poly_a = get_vehicle_polygon(vehicle_a)
    poly_b = get_vehicle_polygon(vehicle_b)

    closest_point_a, closest_point_b = nearest_points(poly_a, poly_b)
    return closest_point_a, closest_point_b


def get_vehicle_polygon(vehicle):
    """根据车辆的中心点、朝向、长宽生成车辆的Polygon对象"""
    # 获取车辆a和车辆b的角点
    corners = get_vehicle_corners(vehicle)
    return Polygon(corners)


def calc_closest_points(vec1, vec2):
    # 获取车辆a和车辆b的多边形
    poly_a = get_vehicle_polygon(vec1)
    poly_b = get_vehicle_polygon(vec2)
    closest_point_a, closest_point_b = nearest_points(poly_a, poly_b)
    return closest_point_a, closest_point_b


def calc_closest_distance(vec1, vec2, origin=False):
    closest_point_a, closest_point_b = calc_closest_points(vec1, vec2)
    distance = sqrt_sum_squares(abs(closest_point_a.x - closest_point_b.x), abs(closest_point_a.y - closest_point_b.y))
    return distance


class Args():
    def __init__(self):
        self.scenario_class = None  # 场景类型
        self.search_algorithm = None  # 搜索算法
        self.simulation_scenario_path = None  # 模拟场景路径
        self.project_path = None
        self.data_class = None

#
# #
# v = global_to_local_velocity({"vx":0,"vy":10,"h":math.pi/2},{"vx":0,"vy":20,"h":math.pi})
# log.info(v)
# vec1 = {'x': 645.8973388671875, 'y': -8.224997520446777, 'z': 0.0, 'h': -1.7485461672997008e-07, 'vx': 33.319664001464844, 'vy': 1.559280644869432e-05, 'boundingbox': {'length': 4.892748832702637, 'width': 2.149045467376709}, 'timestamp': 1739450514.223616}
# vec2 = {'x': 630.8973388671875, 'y': -8.224997520446777, 'z': 0.0, 'h': -1.7485461672997008e-07, 'vx': 33.319664001464844, 'vy': 1.559280644869432e-05, 'boundingbox': {'length': 4.892748832702637, 'width': 2.149045467376709}, 'timestamp': 1739450514.223616}
#
# distance = calc_closest_distance(vec1,vec2)
# log.info(distance)


# data = {'ego':{'name': 'vehicle_001', 'type': '',
#          'transform': {'location': {'x': 240.2030792236328, 'y': -8.381999969482422, 'z': -0.0008935928344726562},
#                        'rotation': {'pitch': 0, 'yaw': -0.0, 'roll': 0}},
#          'velocity': {'x': -0.000602781365159899, 'y': -1.7830353726822068e-07, 'z': -0.022034969180822372},
#          'bounding_box': {'extent': {'x': 1.852684736251831, 'y': 0.8943392634391785, 'z': 0.7745251059532166}},
#          'is_alive': True, 'timestamp': 1739520381.9131176}}
# convert_json("",data)
