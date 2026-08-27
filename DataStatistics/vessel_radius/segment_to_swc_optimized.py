import time
from pathlib import Path
import os
import shutil

import numpy as np
from collections import defaultdict, deque

# 尝试导入C扩展模块
try:
    # 先尝试直接导入
    try:
        import _segment_to_swc_c
        build_adjacency = _segment_to_swc_c.build_adjacency
        bfs_traversal = _segment_to_swc_c.bfs_traversal
    except ImportError:
        # 尝试相对导入
        from ._segment_to_swc_c import build_adjacency, bfs_traversal
    _has_c_extension = True
except ImportError:
    print("Warning: C extension module not found, will use pure Python implementation")
    _has_c_extension = False

def segments_to_swc(data):
    """
    将线段列表转换为SWC格式的树结构

    Parameter:
        data: 线段列表，每个元素为两个点的坐标 ([x1,y1,z1], [x2,y2,z2])

    返回:
        swc_array: numpyArray，每行对应一个节点 [id, type, x, y, z, radius, parent_id]
    """

    if not data:
        return np.array([])

    # 使用C扩展模块（如果可用）
    if _has_c_extension:
        return _segments_to_swc_c(data)
    else:
        return _segments_to_swc_pure_python(data)

def _segments_to_swc_c(data):
    """
    使用C扩展模块将线段列表转换为SWCFormat
    """
    # 构建邻接表
    adjacency = build_adjacency(data)
    
    # 选择起始节点（选择连接数最多的节点作为根节点）
    if not adjacency:
        return np.array([])
    
    start_node = max(adjacency.items(), key=lambda x: len(x[1]))[0]
    
    # BFS遍历构建树
    swc_nodes = bfs_traversal(adjacency, start_node)
    
    # Convert TonumpyArray
    swc_array = np.array(swc_nodes, dtype=np.float64)
    
    return swc_array

def _segments_to_swc_pure_python(data):
    """
    纯Python实现的线段转SWCFormat
    """
    # 1. 构建邻接表
    adjacency = defaultdict(list)
    node_degrees = {}
    
    # Preprocessing：计算节点连接度，避免重复计算
    for (p1, p2) in data:
        p1_tuple = tuple(p1)
        p2_tuple = tuple(p2)
        adjacency[p1_tuple].append(p2_tuple)
        adjacency[p2_tuple].append(p1_tuple)
        
        # 更新节点度数
        node_degrees[p1_tuple] = node_degrees.get(p1_tuple, 0) + 1
        node_degrees[p2_tuple] = node_degrees.get(p2_tuple, 0) + 1

    # 2. 选择起始节点（选择连接数最多的节点作为根节点）
    if not adjacency:
        return np.array([])

    # 使用预计算的度数来选择根节点，提高效率
    start_node = max(node_degrees.items(), key=lambda x: x[1])[0]

    # 3. BFS遍历构建树，避免环
    visited = set()
    node_mapping = {}  # Coordinate -> 节点ID
    swc_nodes = []  # 存储SWC节点
    node_id_counter = 1

    # 初始化队列: (当前节点, 父节点ID)
    queue = deque([(start_node, -1)])
    visited.add(start_node)

    while queue:
        current_node, parent_id = queue.popleft()

        # 分配节点ID
        node_id = node_id_counter
        node_id_counter += 1
        node_mapping[current_node] = node_id

        # 创建SWC节点: [id, type, x, y, z, radius, parent_id]
        swc_nodes.append([
            node_id,  # id
            0,  # type (0=未定义)
            current_node[0],  # x
            current_node[1],  # y
            current_node[2],  # z
            1.0,  # radius (默认为1.0)
            parent_id  # parent_id
        ])

        # 获取所有邻居节点并按连接度排序
        neighbors = []
        for neighbor in adjacency[current_node]:
            if neighbor not in visited:
                # 使用预计算的度数
                degree = node_degrees[neighbor]
                neighbors.append((neighbor, degree))

        # 按连接度降序排序（分支优先）
        neighbors.sort(key=lambda x: x[1], reverse=True)

        # 将邻居加入队列
        for neighbor, _ in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, node_id))

    # 4. 处理孤立的线段（如果存在）
    for node in adjacency:
        if node not in visited:
            # 从孤立节点开始新的子树
            subtree_start = node
            subtree_queue = deque([(subtree_start, -1)])
            visited.add(subtree_start)

            while subtree_queue:
                current_node, parent_id = subtree_queue.popleft()

                # 分配节点ID
                node_id = node_id_counter
                node_id_counter += 1
                node_mapping[current_node] = node_id

                # 创建SWC节点
                swc_nodes.append([
                    node_id,
                    0,
                    current_node[0],
                    current_node[1],
                    current_node[2],
                    1.0,
                    parent_id
                ])

                # 处理邻居
                for neighbor in adjacency[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        subtree_queue.append((neighbor, node_id))

    # 5. Convert TonumpyArray
    swc_array = np.array(swc_nodes, dtype=np.float64)

    return swc_array

# 示例使用
if __name__ == "__main__":
    test_dir = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\python\test"
    test_dir = Path(test_dir)

    swc_dir = test_dir.joinpath("swc")
    save_dir = test_dir.joinpath("res_optimized")
    os.makedirs(save_dir, exist_ok=True)

    swc_names = [n for n in os.listdir(swc_dir) if ".swc" in n][:1]

    for swc_name in swc_names:
        example_data = []

        swc_path = swc_dir.joinpath(swc_name)
        save_path = save_dir.joinpath(swc_name)

        swc_data = np.loadtxt(swc_path, ndmin=2)
        # Preprocessing：提取所有节点的位置，避免重复索引操作
        node_positions = swc_data[:, 2:5]
        parent_ids = swc_data[:, -1].astype(int)

        # 只处理有父节点的节点（parent_id != -1）
        valid_indices = np.where(parent_ids != -1)[0]

        for i in valid_indices:
            parent_id = parent_ids[i]
            parent_index = parent_id - 1

            # 跳过无效的父节点索引
            if parent_index < 0 or parent_index >= len(swc_data):
                continue

            # 获取当前节点和父节点的位置
            current_pos = node_positions[i]
            parent_pos = node_positions[parent_index]

            example_data.append((list(parent_pos), list(current_pos)))

        # Convert ToSWCFormat
        start_time = time.time()
        swc_result = segments_to_swc(example_data)
        print(f"处理时间: {time.time() - start_time:.6f}Second")
        print(f"使用C扩展: {_has_c_extension}")

        with open(save_path, "w") as swc_file:
            for d in swc_result:
                swc_file.write(" ".join([str(n) for n in d]) + "\n")
