import os
import shutil
import time
from collections import Counter
import numpy as np


def SplitSwcData(swcData):
    # 获取所有根节点id为-1的位置序号，并加上最后一个节点位置序号
    indLs = np.where(swcData[:, -1] == -1)[0].tolist() + [swcData.shape[0]]
    swcDataLs = []
    for i in range(len(indLs) - 1):
        data = swcData[indLs[i]: indLs[i + 1]]  # 提取分支树
        sp = data[0, 0]  # 原数据根节点序号
        data[:, 0] -= sp - 1  # 更新分支树所有节点序号
        data[1:, -1] -= sp - 1  # 除第一个节点（根节点），其余节点更新id
        swcDataLs.append(data)
    return swcDataLs


def SplitSwcToBranchData(swcData):
    """
    将SWC数据分割成多个独立的树

    Parameter:
        swcData: numpyArray，SWC格式数据，每行包含 [id, type, x, y, z, radius, parent_id]

    返回:
        swcDataLs: List，每个元素是一个独立的树（SWCArray）
    """
    if len(swcData) == 0:
        return []

    # Method1: 按根节点分割（父节点为-1）
    indLs = np.where(swcData[:, -1] == -1)[0].tolist() + [swcData.shape[0]]
    basic_trees = []
    for i in range(len(indLs) - 1):
        data = swcData[indLs[i]: indLs[i + 1]]
        if len(data) > 0:
            sp = data[0, 0]
            data[:, 0] -= sp - 1
            data[1:, -1] -= sp - 1
            basic_trees.append(data)

    # Method2: 进一步分割每个树的分支
    swcDataLs = []

    for tree in basic_trees:
        # 构建节点映射：id -> 索引
        id_to_idx = {}
        for idx, row in enumerate(tree):
            node_id = int(row[0])
            id_to_idx[node_id] = idx

        # 构建子节点列表
        children = {}
        for idx, row in enumerate(tree):
            node_id = int(row[0])
            parent_id = int(row[-1])

            if parent_id != -1:
                if parent_id not in children:
                    children[parent_id] = []
                children[parent_id].append(node_id)

        # 找到分支节点（有多个子节点的节点）
        branch_nodes = [node_id for node_id, child_list in children.items()
                        if len(child_list) > 1]

        # 如果没有分支节点，直接添加整个树
        if not branch_nodes:
            swcDataLs.append(tree)
            continue

        # 从根节点开始迭代遍历，收集路径
        root_id = int(tree[0, 0])  # 第一个节点是根节点

        # 使用栈进行迭代DFS，栈元素: (当前节点, 当前路径列表)
        stack = [(root_id, [root_id])]
        all_paths = []

        while stack:
            node, path = stack.pop()
            child_list = children.get(node, [])

            if len(child_list) == 0:
                # 叶子节点，保存路径
                all_paths.append(path)
            elif len(child_list) == 1:
                # 只有一个子节点，继续深入
                child = child_list[0]
                new_path = path + [child]
                stack.append((child, new_path))
            else:
                # 分支节点，保存当前路径（到分支节点）
                all_paths.append(path)
                # 对每个子节点，开始新路径
                for child in child_list:
                    # 新路径从分支节点开始，包含分支节点和子节点
                    new_path = [node, child]
                    stack.append((child, new_path))

        # 将路径转换为SWCArray
        for path in all_paths:
            if len(path) < 2:   # 跳过单节点路径（保持与原逻辑一致）
                continue

            # 提取路径对应的节点
            path_nodes = []
            for node_id in path:
                idx = id_to_idx[node_id]
                path_nodes.append(tree[idx])

            path_nodes = np.array(path_nodes)

            # 重新编号节点ID
            new_tree = path_nodes.copy()
            for i, row in enumerate(new_tree):
                new_tree[i, 0] = i + 1          # 新的节点ID
                if i == 0:
                    new_tree[i, -1] = -1        # 根节点
                else:
                    new_tree[i, -1] = i          # 父节点是前一个节点

            swcDataLs.append(new_tree)

    return swcDataLs


# 更简单的版本：只分割每个分支为独立线段
def SplitSwcToSegments(swcData):
    """
    将SWC数据分割成独立的线段（每个父子对为一个线段）

    Parameter:
        swcData: numpyArray，SWC格式数据

    返回:
        segments: List，每个元素是一个线段，包含两个节点的信息
    """
    segments = []

    if len(swcData) == 0:
        return segments

    # 构建ID到节点索引的映射
    id_to_idx = {}
    for idx, row in enumerate(swcData):
        node_id = int(row[0])
        id_to_idx[node_id] = idx

    # 遍历所有节点，找到父子对
    for idx, row in enumerate(swcData):
        node_id = int(row[0])
        parent_id = int(row[-1])

        if parent_id != -1 and parent_id in id_to_idx:
            parent_idx = id_to_idx[parent_id]

            # 创建一个包含父节点和当前节点的线段
            segment = np.array([
                [1, 0, swcData[parent_idx, 2], swcData[parent_idx, 3], swcData[parent_idx, 4], swcData[parent_idx, 5], -1],
                [2, 0, row[2], row[3], row[4], row[5], 1]
            ])
            segments.append(segment)
    return segments


def branch_test():
    # 测试函数
    # 创建示例SWCData（简单的树形结构）
    # example_swc = np.array([
    #     [1, 0, 0, 0, 0, 1.0, -1],  # 根节点
    #     [2, 0, 1, 1, 1, 1.0, 1],  # 子节点1
    #     [3, 0, 2, 2, 2, 1.0, 2],  # 子节点2
    #     [4, 0, 1, 0, 0, 1.0, 2],  # 分支节点（根节点的另一个子节点）
    #     [5, 0, 2, 0, 0, 1.0, 4],  # 分支节点的子节点
    #     [6, 0, 3, 0, 0, 1.0, 5],  # 继续延伸
    # ])

    example_swc = np.loadtxt(r"D:\SY\xueguan\Voxel_To_Model_Lunwen\python\branch_points\cut_0_5_6_0_0_6_0000_00_01.swc")
    # example_swc = np.loadtxt(r"D:\SY\xueguan\Voxel_To_Model_Lunwen\python\branch_points\0000_00_00.swc")

    save_dir = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\python\branch_points\save"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    SplitSwcData_path = os.path.join(save_dir, "SplitSwcData")
    SplitSwcToBranchData_path = os.path.join(save_dir, "SplitSwcToBranchData")
    os.makedirs(SplitSwcData_path, exist_ok=True)
    os.makedirs(SplitSwcToBranchData_path, exist_ok=True)

    # print("原始SWCData:")
    # print(example_swc)

    # 测试分割函数
    print("\n初始方法: SplitSwcData, 只分割主分支")
    trees = SplitSwcData(example_swc)
    print(f"分割成 {len(trees)} 棵树")
    for i, tree in enumerate(trees):
        # print(f"\nTree {i + 1}:")
        # print(tree)
        with open(os.path.join(SplitSwcData_path, f"{i}.swc"), "w") as f:
            for node in tree:
                f.write(" ".join([str(n) for n in node]) + "\n")

    start_time = time.time()
    print("\nMethod1: SplitSwcToBranchData, 分支路径包含分支点")
    trees1 = SplitSwcToBranchData(example_swc)
    print(f"分割成 {len(trees1)} 棵树")
    print("耗时：", time.time() - start_time)
    for i, tree in enumerate(trees1):
        # print(f"\nTree {i + 1}:")
        # print(tree)
        with open(os.path.join(SplitSwcToBranchData_path, f"{i}.swc"), "w") as f:
            for node in tree:
                f.write(" ".join([str(n) for n in node]) + "\n")

    start_time = time.time()
    print("\nMethod2: SplitSwcToSegments, 以线段形式分割")
    segments = SplitSwcToSegments(example_swc)
    print(f"分割成 {len(segments)} 个线段")
    print("耗时：", time.time() - start_time)
    # for i, seg in enumerate(segments):
    #     print(f"\n线段 {i + 1}:")
    #     print(seg)


def collect_branch_points(trees):
    # start_time = time.time()
    root_list = []
    for tree in trees:
        if len(tree) > 1:
            root_list.append(",".join([str(round(n, 2)) for n in tree[0][2: 5]]))
            root_list.append(",".join([str(round(n, 2)) for n in tree[-1][2: 5]]))

    # 1. 计数
    counts = Counter(root_list)
    # 2. 分支点
    branch_points = [[float(n) for n in item.split(",")] for item, cnt in counts.items() if cnt > 2]
    # 3. 末端点
    end_points = [[float(n) for n in item.split(",")] for item, cnt in counts.items() if cnt == 1]
    # print(f"The time taken is：{time.time() - start_time}s")
    return branch_points, end_points


if __name__ == '__main__':
    # Test
    # branch_test()
    # 获取分支点
    example_swc = np.loadtxt(r"D:\SY\xueguan\Voxel_To_Model_Lunwen\python\branch_points\cut_0_5_6_0_0_6_0000_00_01.swc")
    # example_swc = np.loadtxt(r"D:\SY\xueguan\Voxel_To_Model_Lunwen\python\branch_points\0000_00_00.swc")
    trees1 = SplitSwcToBranchData(example_swc)
    branch_points, end_points = collect_branch_points(trees1)
    # 获取分支角度方向
    # # swc_dir = Path(r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_108\results_voxel_swc_connect\add_res")
    # # swc_dir = Path(r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_108\results_voxel_swc")
    # swc_dir = Path(r"D:\SY\xueguan\Voxel_To_Model_Lunwen\python\branch_points")
    #
    # txt_list = []
    # for i, p in enumerate(branch_points):
    #     txt_list.append(f"{i + 1} {0} {p[0]} {p[1]} {p[2]} {0} {-1}\n")
    #
    # save_path = swc_dir.joinpath("branch_points1.swc")
    # with open(save_path, "w") as f:
    #     f.writelines(txt_list)
    #
    # # swc_path = swc_dir.joinpath("cut_0_5_6_0_0_6_0000_00_00.swc")
    # # # swc_path = swc_dir.joinpath("test.swc")
    # # # swc_path = swc_dir.joinpath("cut_0_5_6_0_0_6_0000_00_01.swc")
    # # swcData = np.loadtxt(swc_path, ndmin=2)  # 读取swcFile
    # swcDataLs = SplitSwcData(example_swc)  # swc多树拆分
    #
    # branch_points = []
    # start_time = time.time()
    # for swcData in swcDataLs:
    #     root_list = list((swcData[..., -1]).astype(np.int32))
    #
    #     # 1. 计数
    #     counts = Counter(root_list)
    #
    #     # 2. 提取重复元素及其全部索引
    #     dup_items = {item: [i for i, x in enumerate(root_list) if x == item]
    #                  for item, cnt in counts.items() if cnt > 1}
    #     # print(dup_items)
    #
    #     for key, val in dup_items.items():
    #         if key == 1 and len(val) < 3:
    #             continue
    #         if int(key) != -1:
    #             parent_item = swcData[key - 1]
    #             branch_points.append(parent_item[2: 5])
    #             main_item = None
    #             main_branch = None
    #             branch_v_list = []
    #             for i, v in enumerate(val):
    #                 if main_item is None and i == 0:  # 第一个分支作为主分支
    #                     main_item = swcData[v]
    #                     main_branch = main_item[2: 5] - parent_item[2: 5]  # 末端 - 开端
    #                 else:
    #                     child_item = swcData[v]
    #                     child_branch = child_item[2: 5] - parent_item[2: 5]
    #                     branch_v_list.append([main_branch, child_branch])
    # print(len(branch_points))
    # print("耗时：", time.time() - start_time)
    # txt_list = []
    # for i, p in enumerate(branch_points):
    #     txt_list.append(f"{i + 1} {0} {p[0]} {p[1]} {p[2]} {0} {-1}\n")
    #
    # save_path = swc_dir.joinpath("branch_points2.swc")
    # with open(save_path, "w") as f:
    #     f.writelines(txt_list)
