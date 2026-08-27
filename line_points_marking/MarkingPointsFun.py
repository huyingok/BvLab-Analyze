# -*- coding: utf-8 -*-
import json
import vtkmodules.all as vtk
import numpy as np
import pyqtgraph as pg
from typing import Tuple
from PyQt5.QtCore import Qt
from DataStatistics.vessel_radius.segment_to_swc_optimized import segments_to_swc
from line_points_marking.ThreeLineSwcCurve import ThreeCurveFitPc
from line_points_marking.ControlStyle import button_style, button_alpha_style


# '''Swc多树拆分'''
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


class MarkingPointsFun:
    @classmethod
    def change_mark_function(cls, self):  # 切换标记功能
        if self.dataImporter_index:  # 存在可标签数据,Update
            self.judge_actor()  # 删除多余的点线
            if self.function_index:
                self.function_index = 0
                self.fun_label = 'Peak mode'
            elif self.function_index == 0:
                self.function_index = 1
                self.fun_label = 'Max mode'
            if self.is_marking:
                self.fun_label = f"{self.fun_label}(Revision Mode)"
            self.function_change_label.setText(self.fun_label)

    @classmethod
    def get_picker_position(cls, self):
        clickpos = self.interactor.GetEventPosition()  # 获取坐标
        self.picker.Pick(clickpos[0], clickpos[1], 0, self.ren)  # 拾取点
        pick_pos = self.picker.GetPickPosition()  # 转换为世界坐标

        pick_pos = np.array(pick_pos)
        min_b = pick_pos == np.array([0, 0, 0])
        max_b = pick_pos == np.array(self.img3d.shape) - 1
        if np.sum(min_b) > 0:
            pick_pos += min_b * 0.001
        if np.sum(max_b) > 0:
            pick_pos -= max_b * 0.001
        return pick_pos

    '''交叉定点'''
    @classmethod
    def Cross_onLeftButtonPressEvent(cls, is_left, self):
        pick_pos = cls.get_picker_position(self)
        camera_pos = self.camera.GetPosition()  # 获取相机位置坐标
        # 交点计算
        if self.one_sphere_count:  # 是否触发生成球
            if self.is_sphere_point:
                # 获取第一条直线的中点
                i1 = self.forward_spinBox.value()
                i2 = self.backward_spinBox.value()
                self.sphere_point = (np.array(self.statistics_dict['point_list'][i1]) +
                                     np.array(self.statistics_dict['point_list'][i2])) / 2
                self.is_sphere_point = False

            # p1 = np.array(self.sphere_point)  # l1点击坐标
            # p2 = np.array(pick_pos)  # l2点击坐标
            # v1 = np.array(self.normal)  # l1方向向量
            # v2 = np.array(camera_pos) - np.array(pick_pos)  # l2方向向量
            # u = p2 - p1  # 两直线距离向量
            # d = np.dot(u, v1) / np.dot(v1, v1)  # 投影向量的长度
            # M = p1 + d * v1  # 求映射点坐标
            # # print("球体坐标：", M)
            # # 计算两条直线上的最近点
            # # 计算两点向量在(v1 × v2)上的投影：((P2 - P1) · (v1 × v2) / ||v1 × v2||^2) * (v1 × v2)
            # # Linel1的垂线向量：(P2 - P1) - ((P2 - P1) · (v1 × v2) / ||v1 × v2||^2) * (v1 × v2)
            # # [((P2 - P1) · (v1 × v2) / ||v1 × v2||^2) * (v1 × v2)] · v1 = 0
            # # 首先计算直线2上到直线1的垂直距离的参数s0： s0 = ((P1 - P2) × v1) · (v1 × v2) / ||v1 × v2||^2
            # # 然后计算直线1上最近点的参数t0： t0 = ((P2 - P1) × v2) · (v1 × v2) / ||v1 × v2||^2
            # # 最近点P1' = P1 + t0 * v1 最近点P2' = P2 + s0 * v2
            # N0 = np.cross(v1, v2)  # (v1 × v2)
            # NN = np.linalg.norm(N0)  # |v1 × v2|
            # t0 = np.dot((np.cross(u, v2)), N0) / NN ** 2  # Line1上到直线2的垂直距离的参数
            # lp1 = p1 + t0 * v1  # 标签位置坐标
            # print("lp1:", lp1)

            # cls._cross_line_points(self, camera_pos, lp1)
            cls._cross_line_points(self, camera_pos, self.sphere_point, is_left)

            # xmax, ymax, zmax = (self.img3d.shape[2], self.img3d.shape[1],
            #                     self.img3d.shape[0])
            # if self.sift_points([0, xmax, 0, ymax, 0, zmax], lp1, 0):
            #     coincidence, pos = self.point_coincidence_sure(lp1, pick_pos, camera_pos)
            #     if not coincidence:
            #         self.create_point(pos, is_left)
            #         # self.MarkProcess._create_point(pos, p_size, self, (0, 0, 1))
            #         # self.create_sphere([lp1[0], lp1[1], lp1[2]])

            # for actor in self.cross_actor_points:
            #     self.ren.RemoveActor(actor)  # 删除渲染的点
            # self.renWin.Render()  # 重新渲染
            # self.one_sphere_count = False  # 未触发
        # 直线计算
        if self.point_line_count:  # 是否触发生成点线
            # 找到直线与图像的边界交点
            intersection_points = cls.set_intersection_bound(pick_pos, camera_pos, self)
            # self.normal = [camera_pos[0] - pick_pos[0], camera_pos[1] - pick_pos[1],
            #                camera_pos[2] - pick_pos[2]]  # 直线的方向向量
            self.statistics_dict = {}
            if len(intersection_points):
                # 直线和图像数据的交点集
                points, count = self.collect_points(intersection_points[0], intersection_points[1])
                if count:
                    maxPoint, statistics_dict = self.points_pos_sure(points)
                    if len(maxPoint):
                        maxPoint = [maxPoint[0], maxPoint[1], maxPoint[2]]
                        self.sphere_point = maxPoint  # 获取点击位置坐标
                        self.focal_pos = maxPoint
                        statistics_dict['maxPoint'] = maxPoint
                        # 未完成调节，调节界面可显示或隐藏
                        self.is_cross_finish = False
                        # 更新调节器
                        cls._update_plotWidget(self, statistics_dict)
                        self.res_save_Button.setEnabled(False)
                        self.res_save_Button.setStyleSheet(button_alpha_style)
                        self.cross_actor_points_copy = [n for n in self.cross_actor_points]
                        # 完成第一条线
                        self.is_sphere_point = True

        if self.one_sphere_count:  # 如果可以触发生成球, 不触发生成点线
            self.point_line_count = False
        else:  # 如果不可以触发生成球, 可以触发生成点线
            self.point_line_count = True

    '''更新调节器'''

    @classmethod
    def _update_plotWidget(cls, self, statistics_dict):
        if statistics_dict:
            point_list = statistics_dict['point_list']
            if point_list:
                actor_list = []
                for point in point_list:
                    actor = self.CreateImageData.create_point_actor(point, self.p_size + 2)
                    actor.GetProperty().SetColor((0, 0, 1))  # 颜色
                    actor.VisibilityOn()
                    actor_list.append(actor)
                    self.ren.AddActor(actor)  # 加入渲染器
                self.cross_actor_points = actor_list
                maxIndex = statistics_dict['maxIndex']
                self.renWin.Render()  # 重新渲染
                self.one_sphere_count = True  # 可以触发生成球

                self.statistics_dict = statistics_dict
                # 禁止信号传递
                # self.forward_spinBox.blockSignals(True)
                # self.backward_spinBox.blockSignals(True)
                self.forward_spinBox.setMinimum(0)
                self.forward_spinBox.setMaximum(maxIndex - 1)
                self.forward_spinBox.setValue(0)
                self.backward_spinBox.setMinimum(maxIndex + 1)
                self.backward_spinBox.setMaximum(len(actor_list) - 1)
                self.backward_spinBox.setValue(len(actor_list) - 1)
                # 恢复信号传递
                # self.forward_spinBox.blockSignals(False)
                # self.backward_spinBox.blockSignals(False)
                self.radius_dialog.show()
                self.res_save_Button.setEnabled(True)
                self.res_save_Button.setStyleSheet(button_style)
                cls._create_plotWidget(self)

    '''交叉线提取'''

    @classmethod
    def _cross_line_points(cls, self, camera_pos, lp1, is_left):
        # 找到直线与图像的边界交点
        intersection_points = cls.set_intersection_bound(lp1, camera_pos, self)
        self.statistics_dict = {}
        if len(intersection_points):
            # 直线和图像数据的交点集
            points, count = self.collect_points(intersection_points[0], intersection_points[1])
            if count:
                # maxPoint, statistics_dict = self.points_pos_sure(points)
                maxPoint, statistics_dict = self.points_pos_nearest(points, lp1)
                if len(maxPoint):
                    maxPoint = [maxPoint[0], maxPoint[1], maxPoint[2]]
                    self.focal_pos = maxPoint
                    # 激活按钮
                    self.res_save_Button.setEnabled(True)
                    self.res_save_Button.setStyleSheet(button_style)
                    # 更新调节器
                    cls._update_plotWidget(self, statistics_dict)

                    self.is_left = is_left
                    # xmax, ymax, zmax = (self.img3d.shape[2], self.img3d.shape[1],
                    #                     self.img3d.shape[0])
                    # if self.sift_points([0, xmax, 0, ymax, 0, zmax], maxPoint, 0):
                    #     coincidence, pos = self.point_coincidence_sure(maxPoint, pick_pos, camera_pos)
                    #     if not coincidence:
                    #         self.create_point(pos, is_left)

    '''创建调节器'''

    @classmethod
    def _create_plotWidget(cls, self):
        max_index = self.statistics_dict['maxIndex']
        index_list = self.statistics_dict['index_list']
        gray_list = self.statistics_dict['gray_list']
        forward_value = index_list[0]
        backward_value = index_list[-1]
        center_value = int((forward_value + backward_value) / 2)

        self.pw.clear()
        self.pw.addLegend()
        # r, g, b, y, m, c
        self.pw.plot(index_list, gray_list, pen=pg.mkPen('b', width=2))
        # 2. 两条可拖动垂线
        cl = pg.InfiniteLine(pos=max_index, angle=90, movable=False,
                             pen=pg.mkPen('r', width=1, style=Qt.DashLine))  # 红色虚线
        self.forw_line = pg.InfiniteLine(pos=forward_value, angle=90, movable=True,
                                         pen=pg.mkPen('y', width=6))
        self.center_line = pg.InfiniteLine(pos=center_value, angle=90, movable=False,
                                           pen=pg.mkPen('c', width=3))
        self.back_line = pg.InfiniteLine(pos=backward_value, angle=90, movable=True,
                                         pen=pg.mkPen('g', width=6))
        self.pw.addItem(cl)
        self.pw.addItem(self.forw_line)
        self.pw.addItem(self.center_line)
        self.pw.addItem(self.back_line)

        # 垂线手动拖动 -> 反向更新控件
        self.forw_line.sigPositionChanged.connect(self.forw_line_moved)
        self.back_line.sigPositionChanged.connect(self.back_line_moved)

        self.forw_line.setPos(forward_value)
        self.back_line.setPos(backward_value)

        # 鼠标点击移动
        self.pw.scene().sigMouseClicked.connect(self.pw_mouse_clicked)
        self.active_line = self.forw_line

    '''Max值定点'''

    @classmethod
    def Max_onLeftButtonPressEvent(cls, is_left, self):
        pick_pos = cls.get_picker_position(self)
        camera_pos = self.camera.GetPosition()  # 获取相机位置坐标
        # camera_focal_pos = self.camera.GetFocalPoint()  # 获取相机焦点坐标
        # 找到直线与图像的边界交点
        intersection_points = cls.set_intersection_bound(pick_pos, camera_pos, self)

        if len(intersection_points):
            # 直线和图像数据的交点集
            points, count = self.collect_points(intersection_points[0], intersection_points[1])
            if count:
                maxPoint, statistics_dict = self.points_pos_sure(points)
                if len(maxPoint):
                    maxPoint = [maxPoint[0], maxPoint[1], maxPoint[2]]
                    coincidence, maxPoint = self.point_coincidence_sure(maxPoint, intersection_points[0], intersection_points[1])
                    # 是否重合
                    if not coincidence:  # 没有重合
                        # self.create_sphere([maxPoint[0], maxPoint[1], maxPoint[2]])
                        # self.create_point(maxPoint, is_left)
                        # if len(self.new_Position) and is_left:
                        #     maxPoint = self.MarkProcess.max_centroid(maxPoint, self)

                        # if len(self.new_Position) == 0:
                        #     v = np.array(maxPoint) - np.array(pick_pos)
                        #     if not (v[0] == 0 and v[1] == 0):
                        #         v_ = np.array([-v[1], v[0], 0])
                        #         A = np.array(maxPoint) + v_
                        #         maxPoint = self.MarkProcess.vector_space_centroid(A, maxPoint,
                        #                                                           int(self.centroid_size), self)

                        self.create_point(maxPoint, is_left)
                    else:  # 如果重合
                        if len(self.new_Position) == 0:  # 初始点
                            self.create_point(maxPoint, is_left)

                    self.focal_pos = maxPoint

    """设置边界"""

    @classmethod
    def set_intersection_bound(cls, pick_pos, camera_pos, self):
        img = self.img3d
        if self.volume.GetVisibility():
            min_x, min_y, min_z = 0, 0, 0  # 最小边界
            max_x, max_y, max_z = img.shape[2] - 1, img.shape[1] - 1, img.shape[0] - 1  # 最大边界
        else:
            if self.volume_mark is not None:
                min_x, max_x, min_y, max_y, min_z, max_z = [self.img3d_mark_info[0][0], self.img3d_mark_info[1][0] - 1,
                                                            self.img3d_mark_info[0][1], self.img3d_mark_info[1][1] - 1,
                                                            self.img3d_mark_info[0][2], self.img3d_mark_info[1][2] - 1
                                                            ]
            else:
                min_x, max_x, min_y, max_y, min_z, max_z = [0 for i in range(6)]

        # 计算方向向量
        direction = np.array(pick_pos) - np.array(camera_pos)
        bounds = [min_x, min_y, min_z, max_x, max_y, max_z]

        intersection_points = cls.collect_intersection_points(camera_pos, direction, bounds)

        return intersection_points

    """找到直线与图像的边界交点"""

    @classmethod
    def collect_intersection_points(cls, camera_pos, direction, bounds):
        min_x, min_y, min_z, max_x, max_y, max_z = bounds
        # 初始化交点
        intersection_points = []

        # 求解与每个边界的交点
        def find_intersection(min_val, max_val, camera_val, direction_val):
            if direction_val == 0:
                return None, None  # 平行于边界，无交点
            t_min = (min_val - camera_val) / direction_val
            t_max = (max_val - camera_val) / direction_val
            return min(t_min, t_max), max(t_min, t_max)

        # 求解每个维度的交点
        t_min, t_max = find_intersection(min_x, max_x, camera_pos[0], direction[0])
        if t_min is not None:
            t_min_x, t_max_x = t_min, t_max
        else:
            t_min_x, t_max_x = float('-inf'), float('inf')

        t_min, t_max = find_intersection(min_y, max_y, camera_pos[1], direction[1])
        if t_min is not None:
            t_min_y, t_max_y = t_min, t_max
        else:
            t_min_y, t_max_y = float('-inf'), float('inf')

        t_min, t_max = find_intersection(min_z, max_z, camera_pos[2], direction[2])
        if t_min is not None:
            t_min_z, t_max_z = t_min, t_max
        else:
            t_min_z, t_max_z = float('-inf'), float('inf')

        # 求解最终的交点范围
        t_min = max(t_min_x, t_min_y, t_min_z)
        t_max = min(t_max_x, t_max_y, t_max_z)

        if t_min != float('-inf') and t_max != float('inf'):
            intersection_points.append(list(camera_pos + t_min * direction))
            intersection_points.append(list(camera_pos + t_max * direction))

        return intersection_points

    '''直线和图像数据的交点集'''

    @classmethod
    def _collect_points(cls, intersection1, intersection2, self):
        d_list = np.array(intersection2) - np.array(intersection1)  # 方向向量
        dx, dy, dz = d_list
        ad_list = np.abs(d_list)
        xmin, xmax, ymin, ymax, zmin, zmax = self.sift_box()

        d = 0.2  # 设置最大间隔
        ad = np.argmax(ad_list)  # 判断最大方向
        m = ad_list[ad]
        if m == 0:
            m = 1e-5
        kx = d * dx / m  # 各方向坐标变化间距
        ky = d * dy / m
        kz = d * dz / m
        i = 0
        points = vtk.vtkPoints()  # 收集点集点按方向向量延申的点
        x, y, z = intersection1[0], intersection1[1], intersection1[2]
        while xmax >= x >= xmin and ymax >= y >= ymin and zmax >= z >= zmin:
            points.InsertPoint(i, (x - xmin, y - ymin, z - zmin))
            x += kx
            y += ky
            z += kz
            i += 1
        # x, y, z = pick_pos[0], pick_pos[1], pick_pos[2]
        # while xmax >= x >= xmin and ymax >= y >= ymin and zmax >= z >= zmin:
        #     points.InsertPoint(i, (x - xmin, y - ymin, z - zmin))
        #     x -= kx
        #     y -= ky
        #     z -= kz
        #     i += 1
        return points, i

    @staticmethod
    def find_nearest_index_numpy(points, target):
        points = np.array(points)  # shape: (N, D)
        target = np.array(target)  # shape: (D,)

        # 计算所有点到目标点的距离平方
        distances = np.sum((points - target) ** 2, axis=1)
        return np.argmin(distances)

    '''获取列表中最近的点'''

    @classmethod
    def _points_pos_nearest(cls, points, lp, self):
        line_source = vtk.vtkLineSource()  # 线绘制
        line_source.SetPoints(points)  # 充实线上的部分点
        # 创建一个投影滤波器
        projectFilter = vtk.vtkProbeFilter()  # 探针类，在指定点位置采样数据值
        projectFilter.SetInputConnection(line_source.GetOutputPort())  # 输入点集

        if self.volume.GetVisibility():
            projectFilter.SetSourceData(self.dataImporter.GetOutput())  # 对应数据
        else:
            projectFilter.SetSourceData(self.dataImporter_mark.GetOutput())  # 对应数据

        projectFilter.Update()
        filter_output = projectFilter.GetOutput()
        filter_output_nums = filter_output.GetNumberOfPoints()
        if filter_output_nums == 0:
            return [], {}

        xmin, xmax, ymin, ymax, zmin, zmax = self.sift_box()
        statistics_dict = {}
        point_list = []
        gray_list = []
        max_nums = self.adjust_nums

        for i in range(filter_output_nums):
            point = projectFilter.GetOutput().GetPoint(i)  # 获取直线上的点
            grayValue = filter_output.GetPointData().GetScalars().GetTuple1(i)  # 各点对应的灰度值
            point_list.append(tuple([point[0] + xmin, point[1] + ymin, point[2] + zmin]))
            gray_list.append(grayValue)

        pi = cls.find_nearest_index_numpy(point_list, lp)

        index_list = [i for i in range(filter_output_nums) if abs(pi - i) <= max_nums]
        statistics_dict['index_list'] = [i for i in range(len(index_list))]
        statistics_dict['point_list'] = [point_list[i] for i in range(filter_output_nums) if abs(pi - i) <= max_nums]
        statistics_dict['gray_list'] = [gray_list[i] for i in range(filter_output_nums) if abs(pi - i) <= max_nums]
        statistics_dict['maxIndex'] = pi - index_list[0]

        return point_list[pi], statistics_dict

    '''确认点位置'''

    @classmethod
    def _points_pos_sure(cls, points, self):
        # 获取直线上灰度值最大的整数点
        maxGrayValue = 0

        line_source = vtk.vtkLineSource()  # 线绘制
        line_source.SetPoints(points)  # 充实线上的部分点
        # 创建一个投影滤波器
        projectFilter = vtk.vtkProbeFilter()  # 探针类，在指定点位置采样数据值
        projectFilter.SetInputConnection(line_source.GetOutputPort())  # 输入点集

        if self.volume.GetVisibility():
            projectFilter.SetSourceData(self.dataImporter.GetOutput())  # 对应数据
        else:
            projectFilter.SetSourceData(self.dataImporter_mark.GetOutput())  # 对应数据

        projectFilter.Update()
        filter_output = projectFilter.GetOutput()
        filter_output_nums = filter_output.GetNumberOfPoints()
        if filter_output_nums == 0:
            return [], {}

        xmin, xmax, ymin, ymax, zmin, zmax = self.sift_box()
        statistics_dict = {}

        lvp = []
        mpi = 0
        for i in range(filter_output_nums):
            grayValue = filter_output.GetPointData().GetScalars().GetTuple1(i)  # 各点对应的灰度值
            lvp.append(grayValue)
            if grayValue > maxGrayValue:
                maxGrayValue = grayValue
                mpi = i

        if self.function_index == 0 and self.is_marking:
            # 极大值
            s = 0
            pi = mpi
            similar_r = 0.5
            for j in range(len(lvp) - 1):
                now_fit = float(lvp[j] / lvp[mpi])
                next_fit = float(lvp[j + 1] / lvp[mpi])
                if now_fit >= similar_r and next_fit >= similar_r:
                    pi = j + 1
                    s += 1
                if now_fit >= similar_r and next_fit < similar_r:
                    break
            pi -= round(s / 2)
        else:
            # Maximum
            max_gray_index_list = []  # 最大灰度值序号列表
            near_count = 1  # 最大灰度值临近个数
            fit = 0.98  # 最大灰度值相似度
            near_count_list = []  # 临近个数列表
            end_near_index_list = []  # 末尾临近值序号列表
            for j in range(len(lvp) - 1):  # 灰度值列表
                now_fit = float(lvp[j] / max(lvp[mpi], 1e-5))
                next_fit = float(lvp[j + 1] / max(lvp[mpi], 1e-5))
                if now_fit >= fit:
                    max_gray_index_list.append(j)
                if j == len(lvp) - 2:
                    if next_fit >= fit:
                        max_gray_index_list.append(j + 1)
                    break
            if len(max_gray_index_list) > 1:  # 至少两个临近团
                for k in range(1, len(max_gray_index_list)):
                    if max_gray_index_list[k] - max_gray_index_list[k - 1] == 1:
                        near_count += 1
                        if k == len(max_gray_index_list) - 1:
                            near_count_list.append(near_count)
                            end_near_index_list.append(max_gray_index_list[k])
                    else:
                        near_count_list.append(near_count)
                        end_near_index_list.append(max_gray_index_list[k - 1])
                        near_count = 1
                m_nci = np.argmax(near_count_list)  # 最大个数序号
                if near_count_list[m_nci] % 2 == 0 and int(near_count_list[m_nci] / 2) != 0:  # 个数为偶数，至少有2个
                    fin_id = int(end_near_index_list[m_nci] - near_count_list[m_nci] / 2)  # 中位数取前一个
                    if lvp[fin_id] < lvp[fin_id + 1]:  # 中位数灰度值对比
                        fin_id += 1
                else:  # 个数为奇数
                    fin_id = end_near_index_list[m_nci] - int(near_count_list[m_nci] / 2)
            else:
                try:
                    fin_id = max_gray_index_list[0]
                except IndexError as e:
                    self.error_print(e)
                    return [], {}
            pi = fin_id
        maxPoint = projectFilter.GetOutput().GetPoint(pi)  # 获取直线上的点
        maxPoint = [maxPoint[0] + xmin,
                    maxPoint[1] + ymin,
                    maxPoint[2] + zmin]

        return maxPoint, statistics_dict

    '''确认点是否重合'''

    @classmethod
    def _point_coincidence_sure(cls, mypoint, pick_pos, camera_pos, self):
        new_coincidence = False
        new_l = len(self.new_Position)

        if new_l:  # 已存在标记点
            # 计算方向向量v
            v_ = np.array(pick_pos) - np.array(camera_pos)
            pick_pos = np.array(pick_pos)
            all_points = np.array(self.new_Position)
            nearest_point, min_distance, min_idx = nearest_point_to_line_fast(all_points, pick_pos, v_)
            coin_r = 2  # 点重合
            if min_distance <= coin_r:
                new_coincidence = True
        if not new_coincidence and not new_l:  # 第一个标记点
            if not new_l:  # 重新标记，清空重合情况
                self.is_marking_coincidence = {}
            if self.vtk_Points:  # 存在已确认的标记树
                select_r = 2
                # 寻找最近树和点
                norms, norms_pos, norms_index = cls.check_nearest_tree_point(camera_pos, pick_pos, select_r, self)

                i = np.argmin(norms)  # 选中树序号
                min_norm = norms[i]  # 最近距离
                if min_norm <= select_r:
                    mypoint = norms_pos[i]  # 选中点
                    new_coincidence = True
                    # 保存选中树序号和重合情况
                    self.is_marking_coincidence['coincidence'] = True
                    self.is_marking_coincidence['tree_i'] = i
                    self.is_marking_coincidence['tree_pos'] = mypoint
                    self.is_marking_coincidence['tree_index'] = norms_index[i]
        return new_coincidence, mypoint

    """PloyDataConnect"""

    @classmethod
    def PloyData_connect(cls, self):
        details = []
        # 同一树禁止连接
        left_index1 = self.connect_list[1].get("left_index")
        if self.connect_list[0].get("tree_index") < self.connect_list[1].get("tree_index"):
            self.connect_list = self.connect_list[::-1]
        lines_list = []
        connect_pos = []

        points_sum = 0
        points_num_min = 1000000
        min_tree_index = None

        # 判断树点集数
        for ii, data in enumerate(self.connect_list):
            tree_index = data.get("tree_index")
            pos_index = data.get("pos_index")
            points = self.vtk_Points[tree_index]  # 点集对象
            # 获取所有点坐标
            num_points = points.GetNumberOfPoints()
            pos_list = []
            for i in range(num_points):
                coord = points.GetPoint(i)
                pos = [round(coord[0], 2), round(coord[1], 2), round(coord[2], 2)]
                pos_list.append(pos)
            connect_pos.append(tuple(pos_list[pos_index]))
            points_sum += num_points
            if points_num_min > num_points:
                min_tree_index = ii
                points_num_min = num_points

        if points_sum > 10000:  # 如果树点集总数大于2500，不进行连接，而是生成断点连接树
            if min_tree_index is not None:
                data = self.connect_list[min_tree_index]
                tree_index = data.get("tree_index")
                points = self.vtk_Points[tree_index]  # 点集对象
                # 获取所有点坐标
                num_points = points.GetNumberOfPoints()
                if num_points < 10000:  # 最小树点集数小于2500，生成断点连接树，否则生成连接点树
                    lines = self.vtk_Lines[tree_index]  # 线集对象
                    vertices = self.vtk_Vertices[tree_index]  # 顶点集对象
                    pos_list = []
                    for i in range(num_points):
                        coord = points.GetPoint(i)
                        pos = [round(coord[0], 2), round(coord[1], 2), round(coord[2], 2)]
                        pos_list.append(pos)
                    # pos_index = data.get("pos_index")
                    # connect_pos.append(tuple(pos_list[pos_index]))
                    # 遍历所有线段
                    lines.InitTraversal()
                    id_list = vtk.vtkIdList()

                    while lines.GetNextCell(id_list):
                        id_ = []
                        for j in range(id_list.GetNumberOfIds()):
                            id_.append(id_list.GetId(j))
                        lines_list.append([tuple(pos_list[id_[0]]), tuple(pos_list[id_[1]])])

                    actor = self.vtk_LP_Actor[tree_index]  # 树对象
                    details.append({
                        "actor": actor,
                        "points": points,
                        "lines": lines,
                        "vertices": vertices,
                        "del": 1
                    })
                    self.ren.RemoveActor(actor)
                    self.vtk_LP_Actor.pop(tree_index)
                    self.vtk_Points.pop(tree_index)
                    self.vtk_Lines.pop(tree_index)
                    self.vtk_Vertices.pop(tree_index)
            else:
                return
        else:  # 树结合
            for data in self.connect_list:
                tree_index = data.get("tree_index")
                points = self.vtk_Points[tree_index]  # 点集对象
                lines = self.vtk_Lines[tree_index]  # 线集对象
                vertices = self.vtk_Vertices[tree_index]  # 顶点集对象
                pos_list = []
                # 获取所有点坐标
                num_points = points.GetNumberOfPoints()
                for i in range(num_points):
                    coord = points.GetPoint(i)
                    pos = [round(coord[0], 2), round(coord[1], 2), round(coord[2], 2)]
                    pos_list.append(pos)
                # pos_index = data.get("pos_index")
                # connect_pos.append(tuple(pos_list[pos_index]))
                # 遍历所有线段
                lines.InitTraversal()
                id_list = vtk.vtkIdList()

                while lines.GetNextCell(id_list):
                    id_ = []
                    for j in range(id_list.GetNumberOfIds()):
                        id_.append(id_list.GetId(j))
                    lines_list.append([tuple(pos_list[id_[0]]), tuple(pos_list[id_[1]])])

                actor = self.vtk_LP_Actor[tree_index]  # 树对象

                details.append({
                    "actor": actor,
                    "points": points,
                    "lines": lines,
                    "vertices": vertices,
                    "del": 1
                })
                self.ren.RemoveActor(actor)
                self.vtk_LP_Actor.pop(tree_index)
                self.vtk_Points.pop(tree_index)
                self.vtk_Lines.pop(tree_index)
                self.vtk_Vertices.pop(tree_index)

        if left_index1:
            # 插值点
            sp = np.array(connect_pos[0], dtype=np.float32)  # 起点
            ep = np.array(connect_pos[1], dtype=np.float32)  # 终点

            img = self.img3d.copy()
            min_th = max(0, self.gray_min)
            max_th = min(255, self.gray_max)
            img[img < min_th] = 0
            img[img > max_th] = max_th

            inter_points = ThreeCurveFitPc(img, sp, ep)[1:-1, 2:5]
            inter_points = [connect_pos[0]] + [[round(p[0], 2), round(p[1], 2), round(p[2], 2)] for p in
                                               inter_points][1:-1] + [connect_pos[1]]

            inter_lines = [[tuple(inter_points[k]), tuple(rp)] for k, rp in enumerate(inter_points[1:])]
            lines_list += inter_lines
        else:
            lines_list.append(connect_pos)
        if lines_list:
            # 重新组合
            details = cls._lines_update_combination(lines_list, details, self)
        # 备份删除
        self.PolyData_copy.append(
            details
        )

    """PloyDataDisconnect"""

    @classmethod
    def PloyData_disconnect(cls, self):
        try:
            details = []
            data = self.connect_list[0]
            tree_index = data.get("tree_index")
            pos_index = data.get("pos_index")
            left_index = data.get("left_index")
            points = self.vtk_Points[tree_index]  # 点集对象
            lines = self.vtk_Lines[tree_index]  # 线集对象
            pos_list = []
            # 获取所有点坐标
            num_points = points.GetNumberOfPoints()
            for i in range(num_points):
                coord = points.GetPoint(i)
                pos = [round(coord[0], 2), round(coord[1], 2), round(coord[2], 2)]
                pos_list.append(pos)
            # 遍历所有线段
            lines.InitTraversal()
            id_list = vtk.vtkIdList()
            lines_list = []
            while lines.GetNextCell(id_list):
                id_ = []
                for i in range(id_list.GetNumberOfIds()):
                    id_.append(id_list.GetId(i))
                if left_index:  # 左键去点
                    if id_[0] == pos_index or id_[1] == pos_index:
                        continue
                    else:
                        lines_list.append([tuple(pos_list[id_[0]]), tuple(pos_list[id_[1]])])
                else:  # 右键去线
                    if id_[1] == pos_index:
                        continue
                    else:
                        lines_list.append([tuple(pos_list[id_[0]]), tuple(pos_list[id_[1]])])

            actor = self.vtk_LP_Actor[tree_index]  # 树对象
            # 备份删除的对象
            details.append({
                "actor": actor,
                "points": points,
                "lines": lines,
                "vertices": self.vtk_Vertices[tree_index],
                "del": 1
            })

            self.ren.RemoveActor(actor)
            self.vtk_LP_Actor.pop(tree_index)
            self.vtk_Points.pop(tree_index)
            self.vtk_Lines.pop(tree_index)
            self.vtk_Vertices.pop(tree_index)

            # 重新组合
            if lines_list:
                details = cls.lines_update_combination(lines_list, details, self)
            # 备份删除
            self.PolyData_copy.append(
                details
            )
        except IndexError as e:
            self.error_print(e)
            return

    """线段重新组合"""

    @classmethod
    def lines_update_combination(cls, lines_list, details, self):
        # 线段分组
        new_data = segments_to_swc(lines_list)

        if len(new_data):
            swcDataLs = SplitSwcData(np.array(new_data))  # swc多树拆分

            for swcData in swcDataLs:
                points = vtk.vtkPoints()  # 创建点集合
                lines = vtk.vtkCellArray()  # 创建单元数组
                # 创建顶点单元（每个点是一个顶点）
                vertices = vtk.vtkCellArray()
                for ii, item in enumerate(swcData):  # 分支树信息
                    p0 = item[2: 5]  # 当前位置坐标x，y，z
                    index0 = int(item[-1])
                    id0 = int(item[0])
                    points.InsertNextPoint(p0)

                    if index0 != -1:  # 父节点
                        index1 = index0 - 1  # 前一个点序号
                        # 创建线段
                        line = vtk.vtkLine()
                        if id0 != index0 + 1:  # 分支点
                            index0 = id0 - 1
                        line.GetPointIds().SetId(0, index1)  # 第一个点
                        line.GetPointIds().SetId(1, index0)  # 第二个点
                        # 单元数组添加线段
                        lines.InsertNextCell(line)
                    vertices.InsertNextCell(1)
                    vertices.InsertCellPoint(ii)

                # 创建 PolyData 并设置点和线
                actor = self.create_PolyData(points, lines, vertices)

                self.vtk_Points.append(points)
                self.vtk_Lines.append(lines)
                self.vtk_Vertices.append(vertices)
                self.vtk_LP_Actor.append(actor)

                details.append({
                    "points": points,
                    "lines": lines,
                    "vertices": vertices,
                    "actor": actor,
                    "del": 0
                })

        return details

    """连接合并"""

    @classmethod
    def _lines_update_combination(cls, group, details, self):
        if len(group) > 0:
            # 线段分组
            new_data = segments_to_swc(group)

            if len(new_data):
                swcDataLs = SplitSwcData(np.array(new_data))  # swc多树拆分

                for swcData in swcDataLs:
                    points = vtk.vtkPoints()  # 创建点集合
                    lines = vtk.vtkCellArray()  # 创建单元数组
                    # 创建顶点单元（每个点是一个顶点）
                    vertices = vtk.vtkCellArray()
                    for ii, item in enumerate(swcData):  # 分支树信息
                        p0 = item[2: 5]  # 当前位置坐标x，y，z
                        index0 = int(item[-1])
                        id0 = int(item[0])
                        points.InsertNextPoint(p0)

                        if index0 != -1:  # 父节点
                            index1 = index0 - 1  # 前一个点序号
                            # 创建线段
                            line = vtk.vtkLine()
                            if id0 != index0 + 1:  # 分支点
                                index0 = id0 - 1
                            line.GetPointIds().SetId(0, index1)  # 第一个点
                            line.GetPointIds().SetId(1, index0)  # 第二个点
                            # 单元数组添加线段
                            lines.InsertNextCell(line)
                        vertices.InsertNextCell(1)
                        vertices.InsertCellPoint(ii)

                    # 创建 PolyData 并设置点和线
                    actor = self.create_PolyData(points, lines, vertices)

                    self.vtk_Points.append(points)
                    self.vtk_Lines.append(lines)
                    self.vtk_Vertices.append(vertices)
                    self.vtk_LP_Actor.append(actor)

                    details.append({
                        "points": points,
                        "lines": lines,
                        "vertices": vertices,
                        "actor": actor,
                        "del": 0
                    })
        return details

    """新数据提取"""

    @classmethod
    def create_new_mark(cls, pos, size, self):
        shape = self.img3d.shape
        xmin = int(max(pos[0] - size, 0))
        xmax = int(min(pos[0] + size, shape[2]))
        ymin = int(max(pos[1] - size, 0))
        ymax = int(min(pos[1] + size, shape[1]))
        zmin = int(max(pos[2] - size, 0))
        zmax = int(min(pos[2] + size, shape[0]))
        # img = np.zeros(shape, dtype=str(self.img3d.dtype))
        # img[int(zmin):int(zmax), int(ymin):int(ymax), int(xmin):int(xmax)] = self.img3d[int(zmin):int(zmax), int(ymin):int(ymax), int(xmin):int(xmax)]
        img = self.img3d[zmin:zmax, ymin:ymax, xmin:xmax]
        self.img3d_mark_info = [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        return img

    """寻找最近树和点"""

    @classmethod
    def check_nearest_tree_point(cls, camera_pos, pick_pos, select_r, self):
        # 计算方向向量v
        pick_pos = np.array(pick_pos)
        v = np.array(pick_pos) - np.array(camera_pos)
        norms = []
        norms_pos = []
        norms_index = []
        for ii, points in enumerate(self.vtk_Points):
            num_points = points.GetNumberOfPoints()
            all_points = np.array([list(points.GetPoint(j)) for j in range(num_points)])

            pos, min_norms_, min_norms_index = nearest_point_to_line_fast(all_points, pick_pos, v)

            if self.volume_mark is not None:
                if self.volume_mark.GetVisibility():  # 如果切取图像显示
                    if not (self.img3d_mark_info[0][0] <= pos[0] <= self.img3d_mark_info[1][0] and
                            self.img3d_mark_info[0][1] <= pos[1] <= self.img3d_mark_info[1][1] and
                            self.img3d_mark_info[0][2] <= pos[2] <= self.img3d_mark_info[1][2]):
                        min_norms_index = 0
                        min_norms_ = select_r + 1
            norms.append(min_norms_)  # 当前树最近距离
            norms_pos.append(pos)  # 当前树最近距离点坐标
            norms_index.append(min_norms_index)  # 当前树最近距离点索引

        # 各树最小距离， 各树最小距离点坐标， 各树最小距离点索引
        return norms, norms_pos, norms_index


def nearest_point_to_line_fast(
        points: np.ndarray,
        line_point: np.ndarray,
        line_dir: np.ndarray
) -> Tuple[np.ndarray, float, int]:
    """
    内存优化的快速版本，返回最近点坐标和最小距离

    Parameter:
        points: (N, 3) 点集数组
        line_point: (3,) 直线上一点
        line_dir: (3,) 直线方向向量（不需要单位化）

    返回:
        nearest_point: (3,) 最近点的坐标
        min_distance: float 最小距离
    """
    # 归一化方向向量
    line_dir = line_dir / np.linalg.norm(line_dir)

    # 计算点到直线上一点的向量
    vec = points - line_point  # (N, 3)

    # 计算在直线方向上的投影长度
    t = np.einsum('ij,j->i', vec, line_dir)  # (N,)

    # 计算距离平方 = |vec|^2 - (vec·dir)^2
    # 这是点到直线的垂直距离（勾股定理）
    dist_sq = np.einsum('ij,ij->i', vec, vec) - t ** 2

    # 确保数值稳定（处理浮点误差导致的极小负数）
    dist_sq = np.maximum(dist_sq, 0)

    # 找到最小距离的索引
    min_idx = np.argmin(dist_sq)

    # 获取最近点坐标
    nearest_point = points[min_idx]

    # 计算实际距离（开方）
    min_distance = np.sqrt(dist_sq[min_idx])

    return nearest_point, min_distance, min_idx
