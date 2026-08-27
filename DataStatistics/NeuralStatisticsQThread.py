# -*- coding: utf-8 -*-
import os
from os.path import join
import time
import sys
import traceback
from pathlib import Path
import tempfile
from collections import namedtuple
import tifffile as tiff
import cv2
import csv
import shutil
import numpy as np
from math import sqrt
from numba import njit, prange
from decimal import Decimal, InvalidOperation
from pyexcelerate import Alignment, Font, Style, Workbook
from openpyxl import load_workbook
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.transform import resize
from multiprocessing import Process, Queue
from PyQt5.QtCore import QThread, pyqtSignal
from DataStatistics.CollectBranchPaths import SplitSwcToBranchData, collect_branch_points
from DataStatistics.vessel_radius.DataInterpolation import swc_data_interpolation
from DataStatistics.vessel_radius.Tree_connect import SwcConnect, box_filter_asym


# Get Program Running Path
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))


FeatureSet = namedtuple(
    "FeatureSet",
    """
    length, 
    tortuosity,
    coords_list
    """,
)


class NeuralStatisticsQThread(QThread):
    finish0 = pyqtSignal(str)
    progress_text = pyqtSignal(str, int)
    error0 = pyqtSignal(str)
    feature_files = pyqtSignal(str, str)
    chart_files = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super(NeuralStatisticsQThread, self).__init__()
        self.win = kwargs.get('win')
        self.logger = self.win.logger
        self.sct = SwcConnect()

    def run(self):
        try:
            self.new_text = ""
            analyze_dict = self.win.analyze_dict
            data_type = analyze_dict['data_type']
            save_path = analyze_dict['save_path']
            need_connect = analyze_dict['need_connect']
            if data_type == 'bv':
                print('bv')
                resolution_ratio = analyze_dict['resolution_ratio']
                CutWorkFilesDir = analyze_dict['CutWorkFilesDir']
                self.bv_data_analyze(CutWorkFilesDir,
                                     save_path,
                                     resolution_ratio,
                                     need_connect)
            elif data_type == 'small':
                print('small')
                img_path = analyze_dict['img_path']
                swc_path = analyze_dict['swc_path']
                resolution_ratio = analyze_dict['resolution_ratio']
                self.data_analyze(img_path,
                                  swc_path,
                                  save_path,
                                  resolution_ratio,
                                  need_connect)
            self.progress_text.emit("Statistics completed!", 0)
            self.finish0.emit(self.new_text)
        except Exception as e:
            self.error0.emit(str(e))
            self.error_logger(e)

    """Error message"""

    def error_logger(self, e):
        self.logger.error("\n=== Error message ===")
        self.logger.error(f"Exception type: {type(e).__name__}")
        self.logger.error(f"Error message: {e}")
        self.logger.error("=== Error location ===")
        tb = sys.exc_info()[2]
        for frame in traceback.extract_tb(tb):
            self.logger.error(f"  File: {frame.filename}")
            self.logger.error(f"  Line number: {frame.lineno}")
            self.logger.error(f"  Function: {frame.name}")
            self.logger.error(f"  Code: {frame.line}\n")

    def bv_data_analyze(self, root_dir, save_path, resolution, need_connect):
        root_dir = Path(root_dir)
        resolution = np.array(resolution)
        save_path = Path(save_path)
        # Get folder names
        names = [n for n in os.listdir(root_dir) if "." not in n]
        if not len(names):
            self.new_text = f"There is no block data in the {root_dir} folder"
            return

        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)
        results_folder = save_path.joinpath("AnalyzeResults")
        if results_folder.exists():
            shutil.rmtree(results_folder)
        results_folder.mkdir(exist_ok=True)

        start_time = time.time()
        results_file = ""
        for i, name in enumerate(names):
            root = root_dir.joinpath(name)
            analyze_dir = root.joinpath("neural_analysis")
            if analyze_dir.exists():
                shutil.rmtree(analyze_dir)
            analyze_dir.mkdir(exist_ok=True)
            img_dir = root.joinpath("images")
            swc_connect_dir = analyze_dir.joinpath("swc_connect")
            swc_inter_dir = analyze_dir.joinpath("swc_inter")

            img_names = [n for n in os.listdir(img_dir) if ".tif" in n]
            swc_dir = root.joinpath("swc")
            _swc_dir = swc_dir
            if len(img_names):
                if need_connect:
                    _swc_dir = self.swc_tree_connect_inter(img_dir,
                                                           swc_dir,
                                                           swc_connect_dir,
                                                           swc_inter_dir,
                                                           img_names=img_names,
                                                           resolution=resolution)

                for _dir in [swc_connect_dir]:
                    if _dir.exists():
                        shutil.rmtree(_dir)

            results_file = self.feature_analyze(img_names, _swc_dir, results_folder, resolution, name=name)

            userTime = time.time() - start_time
            surplusTime = userTime / (i + 1) * (len(names) - i - 1)
            if surplusTime >= 0:
                logInfo = '[Total progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                    (i + 1) / len(names) * 100, userTime, surplusTime)
                self.progress_text.emit(logInfo, 0)

        self.chart_files.emit(results_file)

    def data_analyze(self, img_dir, swc_dir, save_dir, resolution, need_connect):
        img_dir = Path(img_dir)
        swc_dir = Path(swc_dir)
        save_dir = Path(save_dir)
        resolution = np.array(resolution)
        img_names = [n for n in os.listdir(img_dir) if ".tif" in n]
        if not len(img_names):
            self.new_text = "The input image folder has no Tif format files"
            return

        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
        results_folder = save_dir.joinpath("AnalyzeResults")
        if results_folder.exists():
            shutil.rmtree(results_folder)
        results_folder.mkdir(exist_ok=True)

        root = img_dir.parent
        analyze_dir = root.joinpath("neural_analysis")
        if analyze_dir.exists():
            shutil.rmtree(analyze_dir)
        analyze_dir.mkdir(exist_ok=True)

        swc_connect_dir = analyze_dir.joinpath("swc_connect")
        swc_inter_dir = analyze_dir.joinpath("swc_inter")

        if need_connect:
            _swc_dir = self.swc_tree_connect_inter(img_dir,
                                                   swc_dir,
                                                   swc_connect_dir,
                                                   swc_inter_dir,
                                                   img_names=img_names,
                                                   resolution=resolution)
        else:
            _swc_dir = swc_dir

        for _dir in [swc_connect_dir]:
            if _dir.exists():
                shutil.rmtree(_dir)

        results_file = self.feature_analyze(img_names, _swc_dir, results_folder, resolution)

        self.chart_files.emit(results_file)

    def feature_analyze(self, img_names, swc_dir, results_folder, resolution, name=None):
        for img_name in img_names:
            swc_name = Path(img_name).stem + ".swc"
            swc_path = swc_dir.joinpath(swc_name)
            swc_data = np.loadtxt(swc_path, ndmin=2)
            trees = SplitSwcToBranchData(swc_data)

            features = []
            for tree in trees:
                features.append(feature_extraction(tree, resolution))

            (
                lengths,
                tortuosities,
                coords_lists,
            ) = record_results(features)

            network_length = np.sum(lengths)
            segment_count = lengths.shape[0]
            if network_length:
                segment_partitioning = segment_count / network_length
            else:
                segment_partitioning = 0

            # Branch points and end points
            branch_points, end_points = collect_branch_points(trees)
            branchpoints = len(branch_points)
            endpoints = len(end_points)

            filename = str(Path(img_name).stem)
            if name:
                filename = filename + f"({name})"

            network_features = [
                filename,
                network_length,
                branchpoints,
                endpoints,
                lengths.shape[0],
                segment_partitioning,
            ]

            avg_length = np.mean(lengths) if len(lengths) else 0
            avg_tortuosity = np.mean(tortuosities) if len(tortuosities) else 0

            segment_features = [
                avg_length,
                avg_tortuosity,
            ]

            results = network_features + segment_features

            ids = np.arange(0, lengths.shape[0])
            segment_results = np.array(
                [
                    ids,
                    lengths,
                    tortuosities,
                ]
            ).T.tolist()

            cache_result(results)  # Cache results

            file = write_seg_results(segment_results, results_folder, filename)

            # 统计图选项名称和路径
            self.feature_files.emit(filename, file)

        results_file = write_results(results_folder)
        return results_file

    def swc_tree_connect_inter(self,
                               img_dir,
                               swc_dir,
                               swc_connect_dir,
                               swc_inter_dir,
                               img_names=None,
                               resolution=(1, 1, 1)
                               ):
        """
        Connect and interpolate SWC tree structures
        Parameters:
        img_dir: str  Original image directory
        swc_dir: str  SWC label directory
        swc_connect_dir: str  Path to save connected SWC results
        swc_inter_dir: str  Path to save interpolated SWC results
        img_names: list  List of image names to process (optional)
        resolution: tuple/ndarray  Resolution ratio (x, y, z)
        """
        resolution = np.array(resolution)
        for _dir in [swc_connect_dir, swc_inter_dir]:
            if _dir.exists():
                if _dir.is_dir():
                    shutil.rmtree(_dir)
            _dir.mkdir(parents=True, exist_ok=True)

        if img_names is None:
            img_names = [n for n in os.listdir(img_dir) if ".tif" in n]

        MNumber = min(int(os.cpu_count() / 2), 6)

        pathLsQue = Queue()
        finishQue = Queue()
        errorQue = Queue()

        for img_name in img_names:
            stem = Path(img_name).stem
            swc_name = stem + ".swc"
            img_path = join(img_dir, img_name)  # Image path
            swc_path = join(swc_dir, swc_name)  # SWC label path
            swc_connect_path = join(swc_connect_dir, swc_name)  # Connected SWC path
            swc_inter_path = join(swc_inter_dir, swc_name)  # Interpolated SWC path
            if os.path.exists(swc_path):
                if os.path.getsize(swc_path) > 0:
                    pathLsQue.put((img_path, swc_path, swc_connect_path, swc_inter_path))
                else:
                    safe_write(swc_connect_path, "", sync=True)
                    safe_write(swc_inter_path, "", sync=True)

        original_work_len = pathLsQue.qsize()

        for i in range(MNumber):
            pathLsQue.put(())

        ps = []
        for i in range(MNumber):
            ps.append(Process(target=swc_tree_connect_inter_process, args=(pathLsQue,
                                                                           finishQue,
                                                                           errorQue,
                                                                           self.sct,
                                                                           resolution)))
            ps[-1].daemon = True
            ps[-1].start()

        start_time = time.time()
        curMakeSize = 0
        finished_processes = 0
        is_stop_count = 0

        while True:
            # Check if all processes have finished
            while finishQue.qsize() > 0:
                is_stop_count = 0
                result = finishQue.get()
                if result == -1:
                    finished_processes += 1
                elif result == 1:
                    curMakeSize += 1

            # All actual work tasks completed
            if curMakeSize >= original_work_len:
                break

            # Process hang detection
            if is_stop_count == 120:  # Approximately one minute without update
                print("Starting process hang detection")
                # Detect files not processed in the connection directory
                names = [Path(n).stem + ".tif" for n in os.listdir(swc_connect_dir) if ".swc" in n]
                if len(names) == len(img_names):
                    break
                else:
                    add_list = []
                    s_names = set(names)
                    for img_name in img_names:
                        if img_name in s_names:
                            continue
                        else:
                            add_list.append(img_name)
                    if len(add_list) > 0:
                        for img_name in add_list:
                            print(f"{curMakeSize + 1} / {original_work_len}")
                            stem = Path(img_name).stem
                            swc_name = stem + ".swc"
                            img_path = join(img_dir, img_name)  # Image path
                            swc_path = join(swc_dir, swc_name)  # SWC label path
                            swc_connect_path = join(swc_connect_dir, swc_name)  # Connected SWC path
                            swc_inter_path = join(swc_inter_dir, swc_name)  # Interpolated SWC path
                            _swc_tree_connect_inter(img_path,
                                                    swc_path,
                                                    swc_connect_path,
                                                    swc_inter_path,
                                                    self.sct,
                                                    resolution)
                            curMakeSize += 1
            # Process error
            if errorQue.qsize():
                e = errorQue.get()
                for p in ps:
                    p.terminate()
                self.error_logger(e)
                self.is_error = 1
                self.error0.emit(str(e))
                return

            # Update progress
            if curMakeSize > 0:
                userTime = time.time() - start_time
                surplusTime = userTime / curMakeSize * (original_work_len - curMakeSize)
                logInfo = f'[Connection progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                    curMakeSize / original_work_len * 100, userTime, surplusTime)
                self.progress_text.emit(logInfo, 1)

            is_stop_count += 1
            time.sleep(0.5)

        # Wait for all processes to finish
        for i, p in enumerate(ps):
            p.join(timeout=1)  # Wait at most 1 second
            if p.is_alive():
                p.terminate()
            if i == 0:
                userTime = time.time() - start_time
                logInfo = f'[Connection progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                    100, userTime, 0)
                self.progress_text.emit(logInfo, 1)
            logInfo = f'[End process progress %.2f%%]' % ((i + 1) / MNumber * 100)
            self.progress_text.emit(logInfo, i)
            time.sleep(0.2)

        return swc_inter_dir


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


def total_length_vox(segments, voxel_size):
    """
    segments: (N,2,3) 的整数体素坐标
    voxel_size: (3,) Array，Sequential [vx, vy, vz]
    """
    # 例：x,y,z Resolution 0.5×0.5×1.0 µm
    # vox = np.array([0.5, 0.5, 1.0])
    # lines = [[(x0,y0,z0), (x1,y1,z1)], [(x2,y2,z2), (x3,y3,z3)], ...]
    # print(total_length_vox(lines, vox))
    segs = np.asarray(segments, dtype=float)
    delta = (segs[:, 1] - segs[:, 0]) * np.array(voxel_size)  # 先减再乘，广播
    return np.sqrt(np.einsum('ij,ij->i', delta, delta)).sum()


# 长度计算
# @njit(fastmath=True)
def length_calc(coords, resolution):
    # Calculate square roots
    deltas = coords[0:-1] - coords[1:]
    squares = (deltas * resolution) ** 2
    results = np.sqrt(np.sum(squares, axis=1))
    return np.sum(results)


@njit(parallel=True, cache=True)
def table_generation(resolution=np.array([1, 1, 1]), size=150):
    # size = min(500, ceil(max_radius / np.min(resolution))) # Hard code size limit at 500 mb
    LUT = np.zeros((size, size, size))

    correction = resolution / 2

    for z in prange(size):
        for y in range(size):
            for x in range(size):
                coords = np.array([z, y, x])
                coords = coords * resolution
                non_zeros = np.count_nonzero(coords)

                # To correct for radii lines along 1D planes, remove half of resolution length.
                if non_zeros == 1:
                    # Two of the values will be 0 and therefore negative after correction.
                    corrected = coords - correction
                    # Remove to isolate true correction.
                    corrected = corrected[corrected > 0][0]
                    LUT[z, y, x] = corrected

                else:
                    a = np.sum(coords**2)
                    LUT[z, y, x] = sqrt(a)
    return LUT


@njit(parallel=True, cache=True)
def calculate_3Dradii(volume, points, LUT):
    volume = volume
    points = points
    LUT = LUT

    # Have to keep this as a list for igraph
    skeleton_radii = np.zeros(points.shape[0])
    empty = np.zeros(150)

    # Iterate through each skeleton point to find
    # local zeros for radius identifications
    for p in prange(points.shape[0]):
        for i in range(empty.shape[0]):
            point = points[p]
            mins = point - i
            # mins = point * -1 ### lots of weird Numba finagling here...
            mins[mins < 0] = 0  # Find the minimum onset of the search box
            zeros = np.vstack(
                np.where(
                    volume[
                        mins[0] : point[0] + i + 1,
                        mins[1] : point[1] + i + 1,
                        mins[2] : point[2] + i + 1,
                    ]
                    == 0
                )
            ).T

            # If there's more than 4 zeros, find radii for average.
            if zeros.shape[0] > 3:
                point_radii = np.zeros(zeros.shape[0])
                zeros = np.abs((zeros + mins) - point)
                for j in range(zeros.shape[0]):
                    point_radii[j] = LUT[zeros[j, 0], zeros[j, 1], zeros[j, 2]]

                point_radii = np.sort(point_radii)
                radius = np.sum(point_radii[:4]) / 4
                skeleton_radii[p] = radius
                break

            # Arbitrary number to cutoff and prevent deadlock.
            # Hard-coding this could be problematic.
            if i == 149:
                skeleton_radii[p] = LUT[-1, -1, -1]
                break

    return skeleton_radii


# 树连接、插值进程
def swc_tree_connect_inter_process(pathLsQue, finishQue, errorQue, sct, resolution):
    while True:
        try:
            add = pathLsQue.get(timeout=5)
            if len(add) == 0:
                finishQue.put(-1)
                return
            img_path, swc_path, swc_connect_path, swc_inter_path = add

            is_ok = False
            if os.path.exists(swc_path):
                if os.path.isfile(swc_path):
                    if os.path.getsize(swc_path):
                        is_ok = True
            if not is_ok:
                safe_write(swc_connect_path, "", sync=True)
                safe_write(swc_inter_path, "", sync=True)
                finishQue.put(1)
                continue

            # 存在标签
            img, shapes = sct.read_img(img_path)  # 读图
            if np.max(img) == 0:
                safe_write(swc_connect_path, "", sync=True)
                safe_write(swc_inter_path, "", sync=True)
                finishQue.put(1)
                continue

            swc_data = sct.read_swc(swc_path)  # 读swcTag
            swcDataLs = sct.collect_swc_tree(swc_data)  # Group

            new_points_dict = {}  # 新点集
            new_endpoints_dict = {}  # 新末端点
            tuple_points_all = []  # 线段集
            # 收集
            for ii, swcData in enumerate(swcDataLs):
                new_points, new_endpoints, tuple_points = sct.check_swc_endpoint(swcData, min_s=(2, 2, 2))
                # 去除长度较小的分支
                if len(tuple_points) == 0:
                    continue
                if len(swcData) <= 10:
                    v_len = total_length_vox(tuple_points, (1, 1, 1))
                    if v_len <= 3:
                        continue
                new_points_dict[str(ii)] = new_points
                new_endpoints_dict[str(ii)] = new_endpoints
                tuple_points_all += tuple_points
            # 分支树连接关系
            tree_connect_set = set()
            # Connect
            for name, new_endpoints in new_endpoints_dict.items():
                np_check_points = None
                for n, new_points in new_points_dict.items():
                    if not len(new_points):
                        continue
                    if name != n:
                        if np_check_points is None:
                            np_check_points = new_points
                        else:
                            if len(np_check_points):
                                np_check_points = np.concatenate([np_check_points, new_points], axis=0)

                if np_check_points is None or not len(np_check_points):
                    continue

                for point_list in new_endpoints:
                    point0 = np.array(point_list[0])
                    point = np.array(point_list[-1])

                    if tuple(point) in tree_connect_set:
                        continue

                    axis_u = point - point0

                    # mask = box_filter(np_check_points, point, axis_u, 40, 4, 4)
                    mask = box_filter_asym(np_check_points, point, axis_u, 6, 3, len_v=6, len_w=6)

                    # print('落入长方体的点数:', mask.sum())
                    selected = np_check_points[mask]  # 即为所求

                    # 3. 最近点
                    if selected.size:  # 框内非空
                        dist2 = np.sum((selected - point) ** 2, axis=1)  # 平方距离
                        nearest = selected[dist2.argmin()]  # 形状 (3,)
                        min_dist = np.sqrt(dist2.min())  # 实际距离
                    else:
                        nearest = None
                        min_dist = np.inf

                    if nearest is not None:
                        if min_dist > 0:
                            tuple_points_all.append([tuple(nearest), tuple(point)])
                            tree_connect_set.add(tuple(nearest))
                            # tuple_points_all.append([tuple(nearest), tuple(point + axis_u * 0.1)])
                            # tuple_points_all.append([tuple(nearest), tuple(point)])
                            # 更新树点集
                            new_points_dict[name] = np.concatenate([new_points_dict[name],
                                                                    np.array([nearest])], axis=0)
            # 重组
            sct.swc_data_combine(tuple_points_all, swc_connect_path)
            # 插值
            swc_data_interpolation(swc_connect_path, swc_inter_path)
            finishQue.put(1)
        except Exception as e:
            if errorQue.qsize() == 0:
                finishQue.put(1)
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
            errorQue.put(e)


# 树连接、插值
def _swc_tree_connect_inter(img_path, swc_path, swc_connect_path, swc_inter_path, sct, resolution):
    is_ok = False
    if os.path.exists(swc_path):
        if os.path.isfile(swc_path):
            if os.path.getsize(swc_path):
                is_ok = True

    if not is_ok:
        safe_write(swc_connect_path, "", sync=True)
        safe_write(swc_inter_path, "", sync=True)
        return

    # 存在标签
    img, shapes = sct.read_img(img_path)  # 读图
    if np.max(img) == 0:
        safe_write(swc_connect_path, "", sync=True)
        safe_write(swc_inter_path, "", sync=True)
        return

    swc_data = sct.read_swc(swc_path)  # 读swcTag
    swcDataLs = sct.collect_swc_tree(swc_data)  # Group

    new_points_dict = {}  # 新点集
    new_endpoints_dict = {}  # 新末端点
    tuple_points_all = []  # 线段集
    # 收集
    for ii, swcData in enumerate(swcDataLs):
        new_points, new_endpoints, tuple_points = sct.check_swc_endpoint(swcData, min_s=(2, 2, 2))
        # 去除长度较小的分支
        if len(tuple_points) == 0:
            continue
        if len(swcData) <= 10:
            v_len = total_length_vox(tuple_points, (1, 1, 1))
            if v_len <= 3:
                continue

        new_points_dict[str(ii)] = new_points
        new_endpoints_dict[str(ii)] = new_endpoints
        tuple_points_all += tuple_points
    # 分支树连接关系
    tree_connect_set = set()
    # Connect
    for name, new_endpoints in new_endpoints_dict.items():
        np_check_points = None
        for n, new_points in new_points_dict.items():
            if not len(new_points):
                continue
            if name != n:
                if np_check_points is None:
                    np_check_points = new_points
                else:
                    if len(np_check_points):
                        np_check_points = np.concatenate([np_check_points, new_points], axis=0)

        if np_check_points is None or not len(np_check_points):
            continue

        for point_list in new_endpoints:
            point0 = np.array(point_list[0])
            point = np.array(point_list[-1])

            if tuple(point) in tree_connect_set:
                continue

            axis_u = point - point0

            # mask = box_filter(np_check_points, point, axis_u, 40, 4, 4)
            mask = box_filter_asym(np_check_points, point, axis_u, 6, 3, len_v=6, len_w=6)

            # print('落入长方体的点数:', mask.sum())
            selected = np_check_points[mask]  # 即为所求

            # 3. 最近点
            if selected.size:  # 框内非空
                dist2 = np.sum((selected - point) ** 2, axis=1)  # 平方距离
                nearest = selected[dist2.argmin()]  # 形状 (3,)
                min_dist = np.sqrt(dist2.min())  # 实际距离
            else:
                nearest = None
                min_dist = np.inf

            if nearest is not None:
                if min_dist > 0:
                    tuple_points_all.append([tuple(nearest), tuple(point)])
                    tree_connect_set.add(tuple(nearest))
                    # tuple_points_all.append([tuple(nearest), tuple(point + axis_u * 0.1)])
                    # tuple_points_all.append([tuple(nearest), tuple(point)])
                    # 更新树点集
                    new_points_dict[name] = np.concatenate([new_points_dict[name],
                                                            np.array([nearest])], axis=0)
    # 重组
    sct.swc_data_combine(tuple_points_all, swc_connect_path)
    # 插值
    swc_data_interpolation(swc_connect_path, swc_inter_path)


def SplitSwcData(swcData):
    """优化版本：使用向量化操作分割SWCData"""
    indLs = np.where(swcData[:, -1] == -1)[0].tolist() + [swcData.shape[0]]
    swcDataLs = []
    for i in range(len(indLs) - 1):
        data = swcData[indLs[i]: indLs[i + 1]]
        sp = data[0, 0]
        data[:, 0] -= sp - 1
        data[1:, -1] -= sp - 1
        swcDataLs.append(data)
    return swcDataLs


def GetPcKernelPc(pc, kernelArr, imgShape):
    curPc = (pc[:, None] + kernelArr[None]).reshape([-1, 3])
    curPc = np.round(curPc).astype(np.int32)
    curPc[curPc < 0] = 0
    curPc[curPc[:, 2] > imgShape[0] - 1, 2] = imgShape[0] - 1
    curPc[curPc[:, 1] > imgShape[1] - 1, 1] = imgShape[1] - 1
    curPc[curPc[:, 0] > imgShape[2] - 1, 0] = imgShape[2] - 1
    curPc = np.unique(curPc, axis=0)
    return curPc


def swc_to_mask(shapes, swcPath):
    kernelLen = 1
    # shapes = mask.shape[::-1]  # xyz
    imgShape = np.array(shapes, dtype=np.int32)  # xyz
    maxD = ((kernelLen ** 2) * 3) ** 0.5
    minD = 0.0697
    maskMax = -np.log(minD)
    dfImg = np.zeros(imgShape[::-1], dtype=np.float32)
    kernelArr = []
    for z in range(-kernelLen, kernelLen + 1):
        for y in range(-kernelLen, kernelLen + 1):
            for x in range(-kernelLen, kernelLen + 1):
                kernelArr.append([x, y, z])
    kernelArr = np.array(kernelArr, dtype=np.int32)

    Data = np.loadtxt(swcPath, ndmin=2)  # 读取骨架文件
    swcDataLs = SplitSwcData(Data)
    dfImg[...] = maxD
    for swcData in swcDataLs:
        for ii, item in enumerate(swcData):
            if item[-1] == -1 or len(item) != 7:
                continue
            p0 = item[2: 5]
            p1 = swcData[int(item[-1]) - 1, 2: 5]
            v = (p1 - p0).reshape([1, 3])
            if np.linalg.norm(v) < 0.1:
                continue
            # 插值
            d = np.linalg.norm(p0 - p1)
            if d > kernelLen:
                d2 = int(d + 1)
                xLs = (np.arange(1, d2 + 1, 1) / d2).reshape([-1, 1])
                data = p0 * xLs + p1 * (1 - xLs)
            else:
                data = np.array([p0, p1])
            curPc = GetPcKernelPc(data, kernelArr, imgShape[::-1])

            proT = (curPc - p0).dot(v.T) / v.dot(v.T)
            proT = np.clip(proT, 0, 1)
            proP = p0 + proT * v
            d = np.linalg.norm(proP - curPc, axis=1)
            if np.isnan(d).any():
                print()

            dfImg[curPc[:, 2], curPc[:, 1], curPc[:, 0]] = np.min([dfImg[curPc[:, 2], curPc[:, 1],
            curPc[:, 0]], d], axis=0)

    dfImg = -np.log(minD + dfImg / maxD * (1 - minD))
    dfImg2 = (dfImg / maskMax * 255).astype(np.uint8)
    return dfImg2

def upsample_to_original(mask_cleaned, volume):
    """
    将处理后的掩膜上采样到原始尺寸
    """
    # print("=== 上采样到原始尺寸 ===")

    if mask_cleaned.shape == volume.shape:
        # print("尺寸已匹配，无需上采样")
        mask_final = mask_cleaned
    else:
        # print(f"上采样从 {mask_cleaned.shape} 到 {volume.shape}")

        # 使用最近邻插值保持二值特性
        mask_final = resize(
            mask_cleaned,
            volume.shape,
            order=0,  # 最近邻插值
            preserve_range=True,
            anti_aliasing=False
        ).astype(np.uint8)
    # print("")
    return mask_final

def feature_extraction(tree, resolution):
    coords = tree[..., 2: 5]  # xyz

    tree_len = length_calc(coords, resolution)

    delta = np.array([coords[0], coords[-1]])
    cord_length = length_calc(delta, resolution)
    if cord_length >= 1:
        tortuosity = tree_len / cord_length
    else:
        tortuosity = 0

    features = FeatureSet(
        tree_len,
        tortuosity,
        coords,
    )
    return features


def record_results(features):
    segment_count = len(features)
    lengths = np.zeros(segment_count)
    tortuosities = np.zeros(segment_count)
    coords_lists = []

    for i, feature in enumerate(features):
        lengths[i] = feature.length
        tortuosities[i] = feature.tortuosity
        coords_lists.append(feature.coords_list)

    return (
        lengths,
        tortuosities,
        coords_lists,
    )


def load_headers():
    results_topper = [""] * 8
    results_topper[2] = "Main Results"
    results_header = [
        "File Name",
        "Network Length",
        "Branchpoints",
        "Endpoints",
        "Number of Segments",
        "Segment Partitioning",
        "Mean Segment Length",
        "Mean Segment Tortuosity",
    ]
    results_header = [results_topper, results_header]

    segment_results_header = [
        "Segment ID",
        "Length",
        "Tortuosity",
    ]
    return results_header, segment_results_header


def write_seg_results(seg_results, results_folder, filename):
    _, segment_results_header = load_headers()

    # Make sure the folder exists
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    segments_folder = os.path.join(results_folder, "Segment_Results")
    if not os.path.exists(segments_folder):
        os.mkdir(segments_folder)

    file = os.path.join(segments_folder, filename + ".csv")

    # Save the info
    with open(file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename:", filename])
        writer.writerow(segment_results_header)
        writer.writerows(seg_results)
    return file


# Store the result in the cache csv file.
def cache_result(result):
    results_cache = get_cache_path()
    if not os.path.exists(results_cache):
        with open(results_cache, "w") as f:
            writer = csv.writer(f)
            writer.writerow(result)
    else:
        with open(results_cache, "a") as f:
            writer = csv.writer(f)
            writer.writerow(result)
    return


def read_ws(ws):
    results = []
    for row in ws.iter_rows():
        results.append([cell.value for cell in row])
    return results


def create_results_file(results_file, data):
    wb = Workbook()
    ws = wb.new_sheet("Main Results", data=data)

    row_style = Style(
        font=Font(bold=True),
        alignment=Alignment(wrap_text=True, horizontal="center", vertical="center"),
        size=60,
    )
    # col_style = Style(font=Font(bold=True), alignment=Alignment(wrap_text=True, horizontal='fill', vertical='bottom'))
    col_style = Style(font=Font(bold=True))
    ws.set_col_style((2, 3, 4, 5, 6, 7, 8), Style(size=12))
    ws.set_row_style((1, 2), row_style)
    ws.set_col_style((1, 2), col_style)
    wb.save(results_file)


# Read the results stored in the cache csv.
def read_cache_results():
    results_cache = get_cache_path()
    results = []
    if os.path.exists(results_cache):
        with open(results_cache, "r") as f:
            reader = csv.reader(f)
            for result in reader:
                # 第一列保持字符串，其他列转 Decimal
                for i in range(1, len(result)):  # 从第2列开始（索引1）
                    try:
                        result[i] = Decimal(result[i]).quantize(Decimal("1.000000"))
                    except InvalidOperation:
                        continue
                results.append(result)
    return results


def get_cache_path():
    results_cache = Path(base_path).joinpath("cache")
    if not results_cache.exists():
        os.makedirs(results_cache, exist_ok=True)
    results_cache = Path(base_path).joinpath("results_cache.csv")
    return results_cache


# Delete cache file after successfully exporting the results
def delete_results_cache():
    results_cache = get_cache_path()
    if os.path.exists(results_cache):
        os.remove(results_cache)
    return


def write_results(results_folder):
    results_header, segment_results_header = load_headers()

    results_file = os.path.join(
        results_folder, "3D Dataset Analysis Results.xlsx"
    )

    if not os.path.exists(results_file):
        results = results_header + read_cache_results()
        create_results_file(results_file, results)
    else:
        # Kind of silly to do it like this, but oh well.
        # This keeps the formatting nice
        # xlsx writing is already slow enough as it is
        wb = load_workbook(results_file, read_only=True)
        ws = wb["Main Results"]
        # 设置单元格格式为文本
        results = read_ws(ws) + read_cache_results()
        create_results_file(results_file, results)
        wb.close()

    delete_results_cache()

    return results_file


if __name__ == '__main__':
    print()
    # need_connect = True
    # vs = VesselStatisticsQThread()
    # root = Path(r"D:\SY\DEMO\save\PredictResults\CutWorkFiles\cut_0_5_4_0_0_6")
    # img_dir = root.joinpath("images")
    # swc_dir = root.joinpath("swc")
    # # swc_dir = root.joinpath("swc2")
    # analyze_dir = root.joinpath("analyze")
    # if analyze_dir.exists():
    #     shutil.rmtree(analyze_dir, ignore_errors=True)
    # os.makedirs(analyze_dir, exist_ok=True)
    # results_folder = analyze_dir.joinpath("results")
    # os.makedirs(results_folder, exist_ok=True)
    #
    # swc_connect_dir = analyze_dir.joinpath("swc_connect")
    # swc_inter_dir = analyze_dir.joinpath("swc_inter")
    # swc_radius_dir = analyze_dir.joinpath("swc_radius")
    # swc_smooth_dir = analyze_dir.joinpath("swc_smooth")
    #
    # resolution = np.array([1, 1, 2])  # xyz
    #
    # start = time.time()
    #
    # img_names = [n for n in os.listdir(img_dir) if ".tif" in n][:2]
    #
    # if need_connect:
    #     # 断点连接、插值
    #     _swc_dir = vs.swc_tree_connect_inter(img_dir,
    #                                          swc_dir,
    #                                          swc_connect_dir,
    #                                          swc_inter_dir,
    #                                          img_names=img_names,
    #                                          resolution=resolution)
    # else:
    #     _swc_dir = swc_dir
    #
    # # 调用优化版本的函数，计算半径，Smooth
    # vs.calculate_vessel_radius(img_dir, _swc_dir, swc_radius_dir, swc_smooth_dir,
    #                            img_names=img_names, resolution=resolution)
    #
    # for img_name in img_names:
    #     swc_name = Path(img_name).stem + ".swc"
    #     img_path = img_dir.joinpath(img_name)
    #     swc_path = swc_smooth_dir.joinpath(swc_name)
    #     img = tiff.imread(img_path)
    #     swc_data = np.loadtxt(swc_path, ndmin=2)
    #     trees = SplitSwcToBranchData(swc_data)
    #
    #     c = 0
    #     features = []
    #     for tree in trees:
    #         features.append(feature_extraction(tree))
    #
    #     (
    #         volumes_or_PAFs,
    #         surface_areas,
    #         lengths,
    #         tortuosities,
    #         radii_avgs,
    #         radii_maxes,
    #         radii_mins,
    #         radii_SD,
    #         coords_lists,
    #         radii_lists
    #     ) = record_results(features)
    #
    #     # Length
    #     network_length = np.sum(lengths)
    #     segment_count = lengths.shape[0]
    #     if network_length:
    #         segment_partitioning = segment_count / network_length
    #     else:
    #         segment_partitioning = 0
    #
    #     # Volume/Percent Area Fraction
    #     network_volume_or_PAF = np.sum(volumes_or_PAFs)
    #
    #     # Surface area
    #     network_SA = np.sum(surface_areas)
    #
    #     # Branch points and end points
    #     branch_points, end_points = collect_branch_points(trees)
    #     branchpoints = len(branch_points)
    #     endpoints = len(end_points)
    #
    #     filename = str(Path(img_name).stem)
    #
    #     network_features = [
    #         filename,
    #         network_volume_or_PAF,
    #         network_length,
    #         network_SA,
    #         branchpoints,
    #         endpoints,
    #         lengths.shape[0],
    #         segment_partitioning,
    #     ]
    #
    #     ## Segment characteristics
    #     avg_radius = np.mean(radii_avgs) if len(radii_avgs) else 0
    #     avg_length = np.mean(lengths) if len(lengths) else 0
    #     avg_tortuosity = np.mean(tortuosities) if len(tortuosities) else 0
    #
    #     # Average volume/PAF of each segment
    #     avg_volume_or_PAF = np.mean(volumes_or_PAFs) if len(volumes_or_PAFs) else 0
    #
    #     avg_SA = np.mean(surface_areas) if len(surface_areas) else 0
    #
    #     segment_features = [
    #         avg_radius,
    #         avg_length,
    #         avg_tortuosity,
    #         avg_volume_or_PAF,
    #         avg_SA,
    #     ]
    #
    #     # Add current results to our results list.
    #     results = network_features + segment_features
    #
    #     ## Distributions
    #     # Find histogram of radii distributions
    #     bins = np.arange(0, 22)
    #     bins[21] = 500
    #     radii_bins = np.histogram(radii_avgs, bins)[0]
    #     # Add these histogram values to our results.
    #     results += radii_bins.tolist()
    #
    #     # Create tortuosity and length bins
    #     tortuosity_bins = [""] * 21
    #     length_bins = [""] * 21
    #     # SA_bins = [''] * 21
    #
    #     # Find mean tortuosity & lengths of vessels in bins.  # 计算各分类区间内血管的平均弯曲度及长度。
    #     for i in range(21):
    #         locations = np.argwhere(
    #             (radii_avgs >= bins[i]) & (radii_avgs < bins[i + 1])
    #         ).transpose()[0]
    #         if len(locations) > 0:
    #             tortuosity_bins[i] = np.sum(tortuosities[locations]) / len(locations)
    #             length_bins[i] = np.sum(lengths[locations]) / len(locations)
    #             # SA_bins[i] = np.sum(surface_areas[locations]) / len(locations)
    #
    #     results += length_bins + tortuosity_bins  # + SA_bins
    #
    #     ids = np.arange(0, volumes_or_PAFs.shape[0])
    #     segment_results = np.array(
    #         [
    #             ids,
    #             volumes_or_PAFs,
    #             lengths,
    #             surface_areas,
    #             tortuosities,
    #             radii_avgs,
    #             radii_maxes,
    #             radii_mins,
    #             radii_SD,
    #         ]
    #     ).T.tolist()
    #
    #     cache_result(results)  # Cache results
    #
    #     write_seg_results(segment_results, results_folder, filename)
    #
    # write_results(results_folder)
    #
    # print("Times:", time.time() - start)

    # root = r"D:\SY\10GTestData\10GVesselTestData\Vessel_DataSet\TrainDataSet\swc"
    root = r"D:\SY\10GTestData\10GVesselTestData\Vessel_DataSet\KS_TrainSet\swc"
    col_dict = {
        'dense': 0.0,
        'medium': 0.0,
        'sparse': 0.0,
        'none': 0.0
    }
    names = os.listdir(root)
    for name in names:
        swc_path = os.path.join(root, name)
        swc_data = np.loadtxt(swc_path, ndmin=2)
        trees = SplitSwcToBranchData(swc_data)
        resolution = np.array([1, 1, 1])
        if len(trees) != 0:
            tree_len = 0
            for tree in trees:
                coords = tree[..., 2: 5]  # xyz
                tree_len += length_calc(coords, resolution)
            if tree_len < 1500:
                col_dict['sparse'] += 1
            elif 1500 <= tree_len < 8000:
                col_dict['medium'] += 1
                # print(name)
            else:
                col_dict['dense'] += 1
        else:
            col_dict['none'] += 1

    print(col_dict)
    col_dict['none'] /= float(len(names))
    col_dict['sparse'] /= float(len(names))
    col_dict['medium'] /= float(len(names))
    col_dict['dense'] /= float(len(names))
    print(col_dict)
