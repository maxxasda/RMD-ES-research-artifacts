import numpy as np
import matplotlib.pyplot as plt
from data_struct import Actor
def get_rotated_bbox(vehicle):
    """
    获取车辆的旋转边界框顶点。
    :param vehicle: dict, 包含 x, y, h, boundingbox
    :return: numpy array, 4x2 矩阵表示顶点的全局坐标
    """
    if isinstance(vehicle,Actor):
        cx, cy ,h= vehicle.transform.location.x, vehicle.transform.location.y, vehicle.transform.rotation.yaw
        length, width = vehicle.bounding_box.extent.x, vehicle.bounding_box.extent.y
    else:
        cx, cy ,h= vehicle['transform']['location']['x'], vehicle['transform']['location']['y'],vehicle['transform']['rotation']['yaw']
        length, width = vehicle['bounding_box']['extent']['x'], vehicle['bounding_box']['extent']['y']

    # 车辆局部坐标的顶点
    half_l, half_w = length / 2, width / 2
    local_vertices = np.array([
        [-half_l, -half_w],  # 左下
        [-half_l, half_w],   # 左上
        [half_l, half_w],    # 右上
        [half_l, -half_w]    # 右下
    ])

    # 构造旋转矩阵
    rotation_matrix = np.array([
        [np.cos(h), -np.sin(h)],
        [np.sin(h), np.cos(h)]
    ])

    # 计算全局顶点
    global_vertices = local_vertices @ rotation_matrix.T + np.array([cx, cy])

    return global_vertices

def collision_detection(vehicle1, vehicle2):
    """
    使用分离轴定理 (SAT) 检测两个车辆是否碰撞。
    :param vehicle1: dict, 包含 x, y, h, boundingbox
    :param vehicle2: dict, 包含 x, y, h, boundingbox
    :return: bool, True 如果碰撞，False 如果未碰撞
    """
    # 获取两个车辆的顶点
    bbox1 = get_rotated_bbox(vehicle1)
    bbox2 = get_rotated_bbox(vehicle2)

    # 获取所有边作为轴
    edges = np.vstack([
        bbox1[1:] - bbox1[:-1],
        bbox1[0] - bbox1[-1],  # bbox1 的最后一条边
        bbox2[1:] - bbox2[:-1],
        bbox2[0] - bbox2[-1]   # bbox2 的最后一条边
    ])

    # 计算每条边的法线（分离轴）
    axes = np.array([-edges[:, 1], edges[:, 0]]).T

    # 投影顶点到分离轴并检查重叠
    for axis in axes:
        axis = axis / np.linalg.norm(axis)  # 单位化
        proj1 = bbox1 @ axis
        proj2 = bbox2 @ axis

        if proj1.max() < proj2.min() or proj2.max() < proj1.min():
            return False  # 如果在某个轴上不重叠，则未碰撞

    return True  # 所有轴都重叠，则碰撞

def plot_vehicles(vehicle1, vehicle2, collision,save_path_file,save = False):
    if not save:
        return
    """
    可视化两个车辆的边界框。
    :param vehicle1: dict, 包含车辆1的信息
    :param vehicle2: dict, 包含车辆2的信息
    :param collision: bool, True 如果发生碰撞
    """
    if isinstance(vehicle1,Actor):
        vehicle1 = vehicle1.get_state()
    if isinstance(vehicle2,Actor):
        vehicle2 = vehicle2.get_state()

    bbox1 = get_rotated_bbox(vehicle1)
    bbox2 = get_rotated_bbox(vehicle2)

    plt.figure(figsize=(8, 8))

    # 绘制车辆1
    plt.fill(bbox1[:, 0], bbox1[:, 1], color='blue', alpha=0.5, label="Ego_001")
    plt.plot(bbox1[[0, 1, 2, 3, 0], 0], bbox1[[0, 1, 2, 3, 0], 1], 'b-', lw=2)
    # 绘制车辆1的方向箭头
    # vehicle['bounding_box']['extent']
    # vehicle['transform']['location']['x'], vehicle['transform']['location']['y'],vehicle['transform']['rotation']['yaw']
    arrow_length = vehicle1["bounding_box"]["extent"]['x'] / 5
    plt.arrow(
        vehicle1['transform']['location']['x'], vehicle1['transform']['location']['y'],
        arrow_length * np.cos(vehicle1['transform']['rotation']['yaw']),
        arrow_length * np.sin(vehicle1['transform']['rotation']['yaw']),
        color='blue', head_width=0.5, head_length=0.5, linewidth=1
    )
    # 绘制车辆2
    plt.fill(bbox2[:, 0], bbox2[:, 1], color='red', alpha=0.5, label="vehicle_001")
    plt.plot(bbox2[[0, 1, 2, 3, 0], 0], bbox2[[0, 1, 2, 3, 0], 1], 'r-', lw=2)
    # 绘制车辆2的方向箭头
    arrow_length = vehicle2["bounding_box"]["extent"]['x'] / 5
    plt.arrow(
        vehicle2['transform']['location']['x'], vehicle2['transform']['location']['y'],
        arrow_length * np.cos(vehicle2['transform']['rotation']['yaw']),
        arrow_length * np.sin(vehicle2['transform']['rotation']['yaw']),
        color='red', head_width=0.5, head_length=1, linewidth=2
    )
    # 添加信息
    plt.legend()
    plt.title("Collision: {}".format("Yes" if collision else "No"), fontsize=16)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.axis('equal')
    # plt.show()
    plt.savefig(save_path_file,dpi=300, bbox_inches='tight')



# vehicle_001 = {'x': 114.31352233886719, 'y': -131.3505401611328, 'z': 0.0, 'h': (0.28955093026161194/ 180) * np.pi, 'boundingbox': {'length': 5.91519021987915, 'width': 1.9884328842163086}}
# Ego_001 =  {'x': 114.52430725097656, 'y': -135.0790557861328, 'z': 0.0, 'h': (-0.687896728515625)/ 180 *np.pi, 'boundingbox': {'length': 4.892748832702637, 'width': 2.149045467376709}}
# # 示例输入
# Ego_001 = {
#     'x': 0,
#     'y': 0,
#     'z': 0.0,
#     'h': -1.8,
#     'boundingbox': {'length': 5, 'width': 3}
# }
# vehicle_001 = {
#     'x': 5,
#     'y': -1,
#     'z': 0.0,
#     'h': -6,
#     'boundingbox': {'length': 5, 'width': 3}
# }
#
# 检测碰撞
# collision = check_collision_sat(Ego_001, vehicle_001)
#
# # 可视化
# plot_vehicles(Ego_001, vehicle_001, collision)
