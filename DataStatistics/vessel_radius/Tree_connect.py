# -*- coding: utf-8 -*-
import os
import tifffile as tiff
import numpy as np
from DataStatistics.vessel_radius.segment_to_swc_optimized import segments_to_swc
import warnings
import sys
from pathlib import Path
from collections import Counter
import tempfile
import traceback
warnings.filterwarnings("ignore")


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


def box_filter_asym(pts,
                    center,
                    axis_u,
                    len_pos,  # +u 方向长度
                    len_neg,  # −u 方向长度（可小于 len_pos）
                    len_v=10.0,  # 垂直方向总宽
                    len_w=10.0):
    pts = np.asarray(pts, dtype=float)
    center = np.asarray(center, dtype=float)
    u = axis_u / np.linalg.norm(axis_u)
    ref = np.array([1, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1, 0])
    v = np.cross(u, ref)
    v /= np.linalg.norm(v)
    w = np.cross(u, v)

    vec = pts - center
    cu = vec @ u
    cv = vec @ v
    cw = vec @ w

    # 不对称区间：[-len_neg, +len_pos]
    mask_u = (cu >= -len_neg) & (cu <= len_pos)
    mask_v = np.abs(cv) <= len_v/2
    mask_w = np.abs(cw) <= len_w/2
    return mask_u & mask_v & mask_w


class SwcConnect:
    def __init__(self):
        self.img = None
        self.shapes = []
        pass

    """读图"""

    def read_img(self, img_path):
        self.img = tiff.imread(img_path)
        if self.img.dtype != np.uint8:
            self.img = ((1 * self.img - self.img.min()) /
                        (max(1e-5, self.img.max() - self.img.min())) * 255).astype(np.uint8)
        self.shapes = self.img.shape
        return self.img, self.shapes

    """读取swc"""

    def read_swc(self, swc_path):
        if os.path.exists(swc_path):
            if os.path.isfile(swc_path):
                if os.path.getsize(swc_path):
                    swcData = np.loadtxt(swc_path, ndmin=2)
                else:
                    swcData = np.array([])
            else:
                raise Exception(f'swc_path {swc_path} is not file')
        else:
            print(f'swc_path {swc_path} does not exist')
            with open(swc_path, 'w') as f:
                pass
            swcData = np.array([])
        return swcData

    """分组点集"""

    def swc_data_segment(self, swc_data):
        """
        :param swc_data:
        """
        tuple_points = []
        for ii, item in enumerate(swc_data):  # 分支树信息
            if int(item[-1]) == -1:
                continue
            else:
                op = item[2: 5]
                sp = swc_data[int(item[-1]) - 1][2: 5]
                tuple_points.append([tuple(sp), tuple(op)])

        return tuple_points

    """优化分组点集"""

    def swc_data_segment_optimized(self, swc_data):
        """
        :param swc_data:
        """
        # 优化：使用NumPy向量化操作替代循环
        # 找到所有非根节点（父节点ID不是-1的节点）
        non_root_indices = np.where(swc_data[:, -1] != -1)[0]
        if len(non_root_indices) == 0:
            return []

        # 获取非根节点的父节点ID
        parent_ids = swc_data[non_root_indices, -1].astype(int)

        # 获取所有点的坐标
        all_coords = swc_data[:, 2:5]

        # 使用向量化操作获取起始点和结束点
        start_points = all_coords[parent_ids - 1]  # 父节点坐标
        end_points = all_coords[non_root_indices]  # 当前节点坐标

        # 将结果转换为所需的列表格式，每个点使用元组
        segments = np.stack([start_points, end_points], axis=1)
        tuple_points = [[tuple(point1), tuple(point2)] for point1, point2 in segments.tolist()]
        return tuple_points

    """合并分组点集"""

    def swc_data_combine(self, tuple_points, save_path):
        """
        :param tuple_points:
        :param save_path:
        """
        swc_result = []
        if tuple_points:
            try:
                swc_result = segments_to_swc(tuple_points)
            except Exception as e:
                print("\n=== Error message ===")
                print(f"Exception type: {type(e).__name__}")
                print(f"Error message: {e}")
                print("=== Error location ===")
                tb = sys.exc_info()[2]
                for frame in traceback.extract_tb(tb):
                    print(f"  File: {frame.filename}")
                    print(f"  Line number: {frame.lineno}")
                    print(f"  Function: {frame.name}")
                    print(f"  Code: {frame.line}\n")

        lines = []
        if len(swc_result):
            for d in swc_result:
                lines.append(" ".join([str(n) for n in d]) + "\n")
        safe_write(save_path, ''.join(lines), sync=True)
        # points = np.array(list(all_points.keys()))
        # return points

    """获取swc所有树"""

    def collect_swc_tree(self, swc_data):
        """
        :param swc_data:
        """
        swcDataLs = SplitSwcData(swc_data)  # swc多树拆分，保留分支
        return swcDataLs

    def adp_cross(self, _v, adp, min_p, max_p):
        d = 1.0
        v = _v / (np.linalg.norm(_v) + 1e-12)
        ref = np.array([1, 0, 0]) if abs(v[0]) < 0.9 else np.array([0, 1, 0])
        e1 = np.cross(v, ref)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(v, e1)

        offs = np.array([d * e1,
                         -d * e1,
                         d * e2,
                         -d * e2,
                         d * (e1 - e2),
                         d * (e1 + e2),
                         d * (e2 - e1),
                         -d * (e2 + e1)])
        nb_pts = adp + offs  # (4,3)

        # 边界掩码
        inside = np.all((nb_pts >= min_p) & (nb_pts < max_p), axis=1)  # (4,)
        if not np.any(inside):  # 一个合格邻居都没有
            is_over = 1
            # break                   # 按需决定是否退出

        # 只保留未越界的
        valid_pts = nb_pts[inside]  # (M,3)  M<=4
        valid_gray = np.empty(len(valid_pts), dtype=self.img.dtype)
        for i, (z, y, x) in enumerate(valid_pts.astype(int)):
            valid_gray[i] = self.img[z, y, x]
        return valid_gray, valid_pts

    """末端增长"""

    def add_tree(self, data, endpoints):
        """
        :param data:
        :param endpoints:
        """
        min_p = np.zeros(3)
        max_p = np.array(self.shapes) - 1

        dataLen = len(data)
        tuple_point = []
        new_endpoints = []

        if dataLen > 1:
            for pp in endpoints:
                sp = np.array(pp[0])  # 向量起点
                ep = np.array(pp[1])  # 向量终点
                _v = ep - sp
                if np.sum(_v == 0) == 3:
                    continue
                g = self.img[int(ep[2]), int(ep[1]), int(ep[0])]
                tp0 = ep
                is_over = 0
                for i in range(1, 2):
                    adp = _v * i + ep
                    # 判断点是否在图像内
                    if np.sum(adp - min_p >= 0) != 3 or np.sum(max_p - adp > 0) != 3:
                        is_over = 1
                        break
                    # 获取增长点灰度
                    adp_g = self.img[int(adp[2]), int(adp[1]), int(adp[0])]

                    # 判断是否记录增长点
                    if adp_g >= g * 0.4:
                        tuple_point.append([tuple(tp0), tuple(adp)])
                        tp0 = np.uint(adp)
                    else:
                        break

                if not is_over:
                    if np.sum(tp0 - ep == 0) == 3:
                        new_endpoints.append([tuple(sp), tuple(ep)])
                    else:
                        new_endpoints.append(tuple_point[-1])

        return tuple_point, new_endpoints

    """获取swc所有末端向量和坐标"""

    def check_swc_endpoint(self, swc_data, min_s=(1, 1, 1)):
        min_s = np.array(min_s)
        max_s = np.array(self.shapes) - min_s

        # 找末端点
        all_points = []
        branch_set = set()
        points = swc_data[:, 2: 5]
        # 分组点集
        # tuple_points = self.swc_data_segment(swc_data)
        tuple_points = self.swc_data_segment_optimized(swc_data)
        for points in tuple_points:
            for point in points:
                p = f"{round(point[0], 2)},{round(point[1], 2)},{round(point[2], 2)}"
                all_points.append(p)
        # 1. 计数
        counts = Counter(all_points)
        # 2. 提取重复元素及其全部索引
        dup_items = {item: [i for i, x in enumerate(all_points) if x == item]
                     for item, cnt in counts.items() if cnt == 1}
        for key, val in dup_items.items():
            branch_set.add(key)

        # 获取分支索引
        res_index = swc_data[..., 0] != swc_data[..., -1] + 1
        # 获取分支点数据，并去除父节点
        true_coords = np.where(res_index)[0][1:]  # 或 np.nonzero(res_index)
        # 判断总数据长度
        endpoints = []
        if len(swc_data) > 1:
            # 判断总数据和父节点有关系点总数：如果只有一个则记录父节点为末端点，反之不记录
            if len(np.where(swc_data[:, -1] == 1)[0]) == 1:
                endpoints = [[tuple(swc_data[1][2: 5]),
                              tuple(swc_data[0][2: 5])]]
            # 找到所有末端点
            for n in true_coords:
                if n > 1:
                    p_1 = swc_data[n - 1][2: 5]
                    p = f"{round(p_1[0], 2)},{round(p_1[1], 2)},{round(p_1[2], 2)}"
                    if p in branch_set:
                        endpoints += [[tuple(swc_data[int(swc_data[n - 1][-1]) - 1][2: 5]),
                                       tuple(swc_data[n - 1][2: 5])]]
            # endpoints += [[tuple(swc_data[int(swc_data[n - 1][-1]) - 1][2: 5]),
            #                tuple(swc_data[n - 1][2: 5])] for n in true_coords if n > 1]
            # 总数据最后一个点默认是末端点
            endpoints += [[tuple(swc_data[int(swc_data[-1][-1]) - 1][2: 5]),
                           tuple(swc_data[-1][2: 5])]]

        new_endpoints = endpoints

        new_endpoints_2 = []
        for point_list in new_endpoints:
            point0 = np.array(point_list[0])
            point = np.array(point_list[-1])

            if np.any(point0 < min_s) or np.any(point0 > max_s):
                continue

            if np.any(point < min_s) or np.any(point > max_s):
                continue

            new_endpoints_2.append(point_list)

        new_endpoints = new_endpoints_2

        # tuple_point, new_endpoints = self.add_tree(swc_data, endpoints)

        for i, ep in enumerate(new_endpoints):
            points = np.concatenate([points, np.array([np.array(ep[1])])], axis=0)

        # tuple_points += tuple_point

        return points, new_endpoints, tuple_points


def safe_write(path: str | os.PathLike, data: bytes | str,
               *, encoding='utf-8', sync=True) -> None:
    """
    原子写文件：
      - 文件已存在则抛 FileExistsError
      - 写中途崩溃不会留下半写文件
      - 可选落盘
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)          # 确保目录存在

    # 1. 先写同目录临时文件（同设备保证 rename 原子）
    with tempfile.NamedTemporaryFile(mode='wb' if isinstance(data, bytes) else 'w',
                                     dir=p.parent, delete=False,
                                     encoding=encoding if isinstance(data, str) else None) as tmp:
        tmp.write(data)
        tmp.flush()                    # 刷到内核缓冲区
        if sync:                       # 真正落盘
            os.fsync(tmp.fileno())
        tmp_name = tmp.name            # 记住临时文件路径

    # 2. 原子改名：要么全新出现，要么抛 FileExistsError
    try:
        os.replace(tmp_name, p)        # 3.3+ 保证原子
    except BaseException:              # 任何失败都清理临时文件
        os.unlink(tmp_name)
        raise
