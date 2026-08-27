# -*- coding: utf-8 -*-
"""
3D Vascular Skeleton Extraction and SWC Conversion
兼容版本的血管骨架提取器
Author: Medical Image Processing Tool
Date: 2024
"""
import time
import numpy as np
import tifffile
from pathlib import Path
import networkx as nx
from scipy.spatial import KDTree
import os
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class VascularSkeletonExtractor:
    """
    从3D TIFF图像中提取血管骨架并保存为SWCFormat（兼容版本）
    """

    def __init__(self,
                 vessel_scale: float = 1.0,
                 threshold: float = 0.1,
                 min_vessel_length: int = 20,
                 hole_fill_size: int = 2,
                 use_hessian: bool = True):
        """
        初始化参数

        Args:
            vessel_scale: 血管尺度参数
            threshold: 二值化阈值
            min_vessel_length: 最小血管长度
            hole_fill_size: 空洞填充大小
            use_hessian: 是否使用Hessian矩阵方法
        """
        self.vessel_scale = vessel_scale
        self.threshold = threshold
        self.min_vessel_length = min_vessel_length
        self.hole_fill_size = hole_fill_size
        self.use_hessian = use_hessian
        self.spacing = (1.0, 1.0, 1.0)

    def connected_components_to_swc(self, G: nx.Graph) -> List[Dict]:
        """
        把每个连通分量分别转成 SWC，返回一个“大列表”，
        用负 parent=-id 隔开不同分量，方便后续拆开或一次性保存。
        """
        swc_all = []
        offset = 0  # 全局 ID 偏移
        for comp_id, nodes in enumerate(nx.connected_components(G)):
            if len(nodes) < 3:  # 太小直接跳过
                continue
            subG = G.subgraph(nodes).copy()
            # 在每个分量里再按 Z 最大挑根
            root = max(nodes, key=lambda n: G.nodes[n]['z'])
            # 转 SWC
            swc_part = self.convert_to_swc(subG, root)
            # 全局 ID 重编号
            for node in swc_part:
                node['id'] += offset
                if node['parent_id'] != -1:
                    node['parent_id'] += offset
                else:
                    node['parent_id'] = -(comp_id + 1)  # 用负 parent 标记新根
            swc_all.extend(swc_part)
            offset += len(swc_part)
            # print(f"{comp_id + 1}：{len(swc_part)}")
        return swc_all

    def build_skeleton_graph(self, skeleton: np.ndarray) -> nx.Graph:
        """
        构建骨架图结构
        """
        # print("构建骨架图...")
        # 获取骨架点坐标
        skeleton_points = np.argwhere(skeleton > 0)
        if len(skeleton_points) == 0:
            # print("Error: No skeleton points")
            return nx.Graph()
        # print(f"找到 {len(skeleton_points)} 个骨架点")
        # 构建KDTree
        kdtree = KDTree(skeleton_points)

        # 创建图
        G = nx.Graph()

        # 添加节点
        for i, point in enumerate(skeleton_points):
            z, y, x = point
            radius = 0.0

            G.add_node(i,
                       pos=point,
                       x=float(x),
                       y=float(y),
                       z=float(z),
                       radius=float(radius))

        # 连接邻居节点
        # print("连接邻居节点...")
        max_distance = 1.8  # 最大连接距离

        for i, point in enumerate(skeleton_points):
            # 查找半径内的邻居
            indices = kdtree.query_ball_point(point, max_distance)
            for j in indices:
                if i != j:
                    # 计算距离
                    dist = np.linalg.norm(skeleton_points[i] - skeleton_points[j])

                    # 只连接近距离的点
                    if 0.5 < dist < 1.8:
                        G.add_edge(i, j, weight=dist)
        # print(f"图构建完成: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
        return G

    def find_root_node(self, G: nx.Graph) -> int:
        """
        找到根节点
        """
        if G.number_of_nodes() == 0:
            return 0

        # 找到Z值最大的点（假设底部是根部）
        max_z = -np.inf
        root_id = 0

        for node_id, data in G.nodes(data=True):
            z = data['z']
            if z > max_z:
                max_z = z
                root_id = node_id

        return root_id

    def convert_to_swc(self, G: nx.Graph, root_id: int) -> List[Dict]:
        """
        Convert ToSWCFormat
        """
        # print("Convert ToSWCFormat...")

        # 创建有向树
        try:
            bfs_tree = nx.bfs_tree(G, root_id)
        except:
            # 如果图不连通，使用最小生成树
            # print("使用最小生成树...")
            mst = nx.minimum_spanning_tree(G)
            bfs_tree = nx.bfs_tree(mst, root_id)

        # 构建SWC节点列表
        swc_nodes = []
        id_mapping = {}

        # 添加根节点
        root_data = G.nodes[root_id]
        swc_node = {
            'id': 1,
            'type': 2,  # 2=轴突
            'x': float(root_data['x']),
            'y': float(root_data['y']),
            'z': float(root_data['z']),
            'radius': float(root_data['radius']),
            'parent_id': -1
        }
        swc_nodes.append(swc_node)
        id_mapping[root_id] = 1

        # 遍历其他节点
        swc_id = 2

        for node in nx.dfs_preorder_nodes(bfs_tree, root_id):
            if node == root_id:
                continue

            # 查找父节点
            predecessors = list(bfs_tree.predecessors(node))
            if not predecessors:
                continue

            parent_id = predecessors[0]
            if parent_id not in id_mapping:
                continue

            node_data = G.nodes[node]

            swc_node = {
                'id': swc_id,
                'type': 2,
                'x': float(node_data['x']),
                'y': float(node_data['y']),
                'z': float(node_data['z']),
                'radius': float(node_data['radius']),
                'parent_id': id_mapping[parent_id]
            }

            swc_nodes.append(swc_node)
            id_mapping[node] = swc_id
            swc_id += 1

        # print(f"生成 {len(swc_nodes)} 个SWC节点")
        return swc_nodes

    def save_swc(self, swc_nodes: List[Dict], output_path: str, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """
        SaveSWCFile
        """
        # print(f"SaveSWCFile: {output_path}")
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            # 头部信息
            f.write("# SWC format for vascular skeleton\n")
            f.write("# Created by VascularSkeletonExtractor\n")
            f.write("# Format: ID Type X Y Z Radius ParentID\n")
            f.write(f"# Total nodes: {len(swc_nodes)}\n")
            # 节点数据
            for node in swc_nodes:
                line = (f"{node['id']} {node['type']} "
                        f"{node['x']:.3f} {node['y']:.3f} {node['z']:.3f} "
                        f"{node['radius']:.3f} {node['parent_id'] if node['parent_id'] > 0 else -1}\n")
                f.write(line)
        # print(f"SWC文件保存成功")


if __name__ == "__main__":
    start_time = time.time()
    # root = r"D:\BaiduNetdiskDownload\res\bigSize\FilterResults"
    # root = r"D:\SY\xueguan\lunwen_picture"
    # root = r"D:\SY\xueguan\lunwen_picture\Data512\Data_50\treat_one\cut_0_5_6_0_0_6"
    # root = r"D:\SY\xueguan\lunwen_picture\Data512\Data_50\treat_one\cut_0_5_7_0_0_10"
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_54"
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_54_add"
    # root = r"D:\SY\xueguan\lunwen_picture\Data512\Data_50\predict_100\MakeResults\DivideResults\testData"
    # root = r"D:\SY\xueguan\lunwen_picture\Data512"
    # root = r"D:\SY\xueguan\lunwen_picture\t3"
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_108"
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_108\signal_data_contrast"
    # root = r"D:\LuoJi\20251110\train_data\filter\FilterResults"
    root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\pt3-2"

    results_dir = os.path.join(root, "results_voxel")
    # results_dir = os.path.join(root, "results_skeletonize")

    save_swc_dir = os.path.join(root, "results_voxel_swc")
    # save_swc_dir = os.path.join(root, "results_skeletonize_swc")
    os.makedirs(save_swc_dir, exist_ok=True)
    names = [n for n in os.listdir(results_dir) if n.lower().endswith('.tif')]

    extractor = VascularSkeletonExtractor(
        vessel_scale=1.0,
        threshold=0.1,
        min_vessel_length=2
    )

    for i, name in enumerate(names):
        # if name != '0016-6_6_9-1_2_0.tif':
        # if name != '00_00_01.tif':
        #     continue
        print(i + 1, '/', len(names), name)
        results_path = os.path.join(results_dir, name)
        save_swc_path = os.path.join(save_swc_dir, Path(name).stem + ".swc")
        skeleton = tifffile.imread(results_path)
        G = extractor.build_skeleton_graph(skeleton)
        if G.number_of_nodes() == 0:
            # print("Error: Graph construction failed")
            break
        # 找到根节点
        # root_id = extractor.find_root_node(G)
        # print(f"根节点ID: {root_id}")
        # Convert ToSWC
        # swc_nodes = extractor.convert_to_swc(G, root_id)
        swc_nodes = extractor.connected_components_to_swc(G)
        # SaveSWC
        extractor.save_swc(swc_nodes, save_swc_path, extractor.spacing)
        print("=" * 60)
        print("Processing completed successfully!")
        print(f"SWCFile: {save_swc_path}")
        print("=" * 60)

    print(time.time() - start_time)
