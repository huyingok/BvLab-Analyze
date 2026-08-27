# -*- coding: utf-8 -*-
import vtkmodules.all as vtk
import numpy as np
from typing import Tuple


class MarkingPointsFun:
    @classmethod
    def change_mark_function(cls, self):  # 切换标记功能
        if self.dataImporter_index:  # 存在可标签数据,Update
            self.judge_actor()  # 删除多余的点线
            self.renWin.Render()  # 重新渲染
            if self.function_index == 1:
                self.function_index = 0
                self.fun_label = 'Intersection mode'
            elif self.function_index == 0:
                self.function_index = 1
                self.fun_label = 'Max mode'
            if self.is_marking_button:
                self.fun_label = f"{self.fun_label}(Revision Mode)"
            self.function_change_label.setText(self.fun_label)

    @classmethod
    def get_picker_position(cls, self):
        clickpos = self.interactor.GetEventPosition()  # 获取坐标
        cellpicker = vtk.vtkCellPicker()  # 单位拾取
        cellpicker.SetTolerance(0.0005)  # 距离容差
        cellpicker.Pick(clickpos[0], clickpos[1], 0, self.ren)  # 拾取点
        pick_pos = cellpicker.GetPickPosition()  # 转换为世界坐标

        pick_pos = np.array(pick_pos)
        min_b = pick_pos == np.array([0, 0, 0])
        max_b = pick_pos == np.array(self.img3d.shape) - 1
        if np.sum(min_b) > 0:
            pick_pos += min_b * 0.001
        if np.sum(max_b) > 0:
            pick_pos -= max_b * 0.001
        return pick_pos

    '''Max值定点'''

    @classmethod
    def Max_onLeftButtonPressEvent(cls, self):
        pick_pos = cls.get_picker_position(self)
        camera_pos = self.camera.GetPosition()  # 获取相机位置坐标
        # camera_focal_pos = self.camera.GetFocalPoint()  # 获取相机焦点坐标
        # 找到直线与图像的边界交点
        intersection_points = cls.set_intersection_bound(pick_pos, camera_pos, self)
        if len(intersection_points) > 0:
            # 直线和图像数据的交点集
            points, count = self.collect_points(intersection_points[0], intersection_points[1])
            if count:
                maxPoint = self.points_pos_sure(points)
                if len(maxPoint):
                    maxPoint = [maxPoint[0], maxPoint[1], maxPoint[2]]
                    coincidence = self.point_coincidence_sure(maxPoint, pick_pos, camera_pos)
                    if not coincidence:
                        self.create_sphere([maxPoint[0], maxPoint[1], maxPoint[2]])

    """设置边界"""

    @classmethod
    def set_intersection_bound(cls, pick_pos, camera_pos, self):
        # 计算方向向量
        direction = np.array(pick_pos) - np.array(camera_pos)
        bounds = self.sift_box(self.img3d, 1)
        bounds = [bounds[0], bounds[2], bounds[4], bounds[1], bounds[3], bounds[5]]
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
        img = self.img3d  # 图数据
        dx, dy, dz = d_list
        ad_list = np.abs(d_list)
        xmin, xmax, ymin, ymax, zmin, zmax = self.sift_box(img, 1)

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
            points.InsertPoint(i, (x - xmin,
                                   y - ymin,
                                   z - zmin))
            x += kx
            y += ky
            z += kz
            i += 1
        # x, y, z = intersection1[0], intersection1[1], intersection1[2]
        # while xmax >= x >= xmin and ymax >= y >= ymin and zmax >= z >= zmin:
        #     points.InsertPoint(i, (x - xmin,
        #                            y - ymin,
        #                            z - zmin))
        #     x -= kx
        #     y -= ky
        #     z -= kz
        #     i += 1
        return points, i

    '''vtk鼠标左击事件'''
    @classmethod
    def onLeftButtonPressEvent(cls, self):
        pick_pos = cls.get_picker_position(self)
        camera_pos = self.camera.GetPosition()  # 获取相机位置坐标
        # 交点计算
        if self.one_sphere_count:  # 是否触发生成球
            p1 = np.array(self.sphere_point)  # l1点击坐标
            p2 = np.array(pick_pos)  # l2点击坐标
            v1 = np.array(self.normal)  # l1方向向量
            v2 = np.array(camera_pos) - np.array(pick_pos)  # l2方向向量
            u = p2 - p1  # 两直线距离向量
            d = np.dot(u, v1) / np.dot(v1, v1)  # 投影向量的长度
            M = p1 + d * v1  # 求映射点坐标
            # print("球体坐标：", M)
            # 计算两条直线上的最近点
            # 计算两点向量在(v1 × v2)上的投影：((P2 - P1) · (v1 × v2) / ||v1 × v2||^2) * (v1 × v2)
            # Linel1的垂线向量：(P2 - P1) - ((P2 - P1) · (v1 × v2) / ||v1 × v2||^2) * (v1 × v2)
            # [((P2 - P1) · (v1 × v2) / ||v1 × v2||^2) * (v1 × v2)] · v1 = 0
            # 首先计算直线2上到直线1的垂直距离的参数s0： s0 = ((P1 - P2) × v1) · (v1 × v2) / ||v1 × v2||^2
            # 然后计算直线1上最近点的参数t0： t0 = ((P2 - P1) × v2) · (v1 × v2) / ||v1 × v2||^2
            # 最近点P1' = P1 + t0 * v1 最近点P2' = P2 + s0 * v2
            N0 = np.cross(v1, v2)  # (v1 × v2)
            NN = np.linalg.norm(N0)  # |v1 × v2|
            t0 = np.dot((np.cross(u, v2)), N0) / NN ** 2  # Line1上到直线2的垂直距离的参数
            lp1 = p1 + t0 * v1  # 标签位置坐标
            # print("lp1:", lp1)
            xmax, ymax, zmax = (self.img3d.shape[2] * self.x_px, self.img3d.shape[1] * self.y_px,
                                self.img3d.shape[0] * self.z_px)
            if self.sift_points([0, xmax, 0, ymax, 0, zmax], lp1, 0):
                coincidence = self.point_coincidence_sure(lp1, pick_pos, camera_pos)
                if not coincidence:
                    self.create_sphere([lp1[0], lp1[1], lp1[2]])

            self.ren.RemoveActor(self.one_point_actor)  # 删除渲染的点
            self.ren.RemoveActor(self.one_line_actor)  # 删除渲染的线
            self.renWin.Render()  # 重新渲染
            self.one_sphere_count = False  # 未触发
        # 直线计算
        if self.point_line_count:  # 是否触发生成点线
            self.sphere_point = pick_pos  # 获取点击位置坐标
            sphere = vtk.vtkPointSource()  # 点绘制
            sphere.SetCenter(pick_pos)  # 设置中心位置
            sphere.SetNumberOfPoints(1)  # 设置要生成的点的数量
            sphere.SetRadius(0)  # 点云数据范围半径
            sphere.Update()  # 数据更新
            mapper = vtk.vtkPolyDataMapper()  # 数据映射器
            mapper.SetInputConnection(sphere.GetOutputPort())  # 连接数据
            self.one_point_actor = vtk.vtkActor()  # 创建点actor
            self.one_point_actor.GetProperty().SetColor(0, 0, 1)  # 颜色
            self.one_point_actor.GetProperty().SetPointSize(10)  # 点大小
            self.one_point_actor.SetMapper(mapper)  # 点连接映射器
            self.ren.AddActor(self.one_point_actor)  # Renderingactor
            line_glyph = vtk.vtkLineSource()  # 线绘制
            self.normal = [camera_pos[0] - pick_pos[0], camera_pos[1] - pick_pos[1],
                           camera_pos[2] - pick_pos[2]]  # 直线的方向向量
            x1, y1, z1 = (pick_pos[0] + self.normal[0] * 10, pick_pos[1] + self.normal[1] * 10,
                          pick_pos[2] + self.normal[2] * 10)
            x2, y2, z2 = (pick_pos[0] - self.normal[0] * 10, pick_pos[1] - self.normal[1] * 10,
                          pick_pos[2] - self.normal[2] * 10)
            line_glyph.SetPoint1(x1, y1, z1)  # 通过两点确定直线
            line_glyph.SetPoint2(x2, y2, z2)
            line_mapper1 = vtk.vtkPolyDataMapper()  # 数据映射器
            line_mapper1.SetInputConnection(line_glyph.GetOutputPort())  # 连接数据
            self.one_line_actor = vtk.vtkActor()  # 创建线actor
            self.one_line_actor.SetMapper(line_mapper1)  # 线连接映射器
            self.one_line_actor.GetProperty().SetColor(0, 0, 1)  # 颜色
            self.one_line_actor.GetProperty().SetLineWidth(2)  # 线宽
            self.ren.AddActor(self.one_line_actor)  # Renderingactor
            self.renWin.Render()  # 重新渲染
            self.one_sphere_count = True  # 可以触发生成球

        if self.one_sphere_count:  # 如果可以触发生成球, 不触发生成点线
            self.point_line_count = False
        else:  # 如果不可以触发生成球, 可以触发生成点线
            self.point_line_count = True

    '''确认点位置'''
    @classmethod
    def _points_pos_sure(cls, points, self):
        # def update_point(view, d, fit, max_gray, lxy, x, y, z):
        #     xy_count = 0
        #     while 1:
        #         y += d
        #         print(view.img3d[z, y, x])
        #         if (float(view.img3d[z, y, x]) / max_gray) >= fit:
        #             xy_count += 1
        #         else:
        #             lxy.append(xy_count)
        #             break
        #     return lxy

        # 正交模式
        # rectMin = [xmin, ymin, zmin]  # 框架尺寸
        # rectMax = [xmax, ymax, zmax]
        #
        # op = np.array(pick_pos)  # 点击位置坐标
        # # v_ = np.array(pick_pos) - np.array(camera_pos)  # 相机位置与点击点的方向向量
        # v_ = np.array(camera_focal_pos) - np.array(camera_pos)  # 相机位置与焦点的方向向量
        # dim = np.argmax(np.abs(v_))  # 获取绝对值最大值序号
        # if v_[dim] >= 0:  # Orientation
        #     tLs = np.arange(rectMin[dim], rectMax[dim], 1)
        # else:
        #     tLs = np.arange(rectMax[dim], rectMin[dim], -1)
        # vLen = ((tLs - op[dim]) / v_[dim]).reshape([-1, 1])  # Spacing
        # # print('vLen =', vLen)
        # newP = np.round((op + vLen * v_)).reshape([-1, 3]).astype((np.int32))
        # if len(newP) == 0:
        #     return False, []
        # newP2 = newP.reshape([-1, 3]) - rectMin
        # newP2[newP2 < 0] = 0
        # img = self.img3d[zmin:zmax, ymin:ymax, xmin:xmax]
        # shape = img.shape
        # newP2[newP2[:, 0] > shape[2] - 1, 0] = shape[2] - 1
        # newP2[newP2[:, 1] > shape[1] - 1, 1] = shape[1] - 1
        # newP2[newP2[:, 2] > shape[0] - 1, 2] = shape[0] - 1
        # grayLs = img[newP2[:, 2], newP2[:, 1], newP2[:, 0]]
        # idx = np.argmax(grayLs)
        # pos = list(newP2[idx] + rectMin)
        # print(newP2)
        # 透视模式
        # 获取直线上灰度值最大的整数点
        maxGrayValue = 0

        line_source = vtk.vtkLineSource()  # 线绘制
        line_source.SetPoints(points)  # 充实线上的部分点
        # 创建一个投影滤波器
        projectFilter = vtk.vtkProbeFilter()  # 探针类，在指定点位置采样数据值
        projectFilter.SetInputConnection(line_source.GetOutputPort())  # 输入点集
        # if self.volume.GetVisibility():
        #     projectFilter.SetSourceData(self.dataImporter.GetOutput())  # 对应数据
        # else:
        #     projectFilter.SetSourceData(self.box_dataImporter.GetOutput())  # 对应数据
        if self.box_volume.GetVisibility():
            projectFilter.SetSourceData(self.box_dataImporter.GetOutput())  # 对应数据
        else:
            return []
        projectFilter.Update()
        filter_output = projectFilter.GetOutput()
        filter_output_nums = filter_output.GetNumberOfPoints()
        if filter_output_nums == 0:
            return []

        lvp = []
        mpi = 0
        for i in range(filter_output_nums):
            # point = projectFilter.GetOutput().GetPoint(i)  # 获取直线上的点
            grayValue = filter_output.GetPointData().GetScalars().GetTuple1(i)  # 各点对应的灰度值
            lvp.append(grayValue)
            if grayValue > maxGrayValue:
                maxGrayValue = grayValue
                mpi = i
        # Maximum
        max_gray_index_list = []  # 最大灰度值序号列表
        near_count = 1  # 最大灰度值临近个数
        fit = 0.95  # 最大灰度值相似度
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
            except IndexError:
                return []
        pi = fin_id
        maxPoint = projectFilter.GetOutput().GetPoint(pi)  # 获取直线上的点
        maxPoint = [maxPoint[0] + self.origin[0] * self.x_px,
                    maxPoint[1] + self.origin[1] * self.y_px,
                    maxPoint[2] + self.origin[2] * self.z_px]
        # xy_sure = 1
        # lxy = []
        # d = 0.5
        # mx, my, mz = maxPoint
        # if xy_sure:
        #     for i in [-d, d, -d, d]:
        #         lxy = update_point(self, i, fit, maxGrayValue, lxy, mx, my, mz)
        #     mx = (lxy[1] - lxy[0]) * d / 2 + mx
        #     my = (lxy[3] - lxy[2]) * d / 2 + my
        # maxPoint = [mx, my, mz]
        # else:
        #     print("没有点数据")
        #     maxPoint = []
        # print("最大灰度值点坐标：", maxPoint)
        # print("最大灰度值：", maxGrayValue)
        return maxPoint

    '''确认点是否重合'''
    @classmethod
    def _point_coincidence_sure(cls, mypoint, pick_pos, camera_pos, self):
        pos = [mypoint[0], mypoint[1], mypoint[2]]  # 灰度值最大点坐标
        box_size = self.sift_box(self.img3d, 0)
        if len(self.points_list):
            i = self.index
            if self.sift_points(box_size, self.Position[i], self.sphere_size):
                if self.sift_points(box_size, self.Position[i], -self.sphere_size / 2):
                    if self.points_list[i].GetProperty().GetColor() == (1, 0, 0):
                        if self.mark_render_text == "Single-color":
                            cell_rgb = self.create_cell_random_color(3)
                        else:
                            cell_rgb = self.create_cell_random_color()
                        # self.points_list[i].GetProperty().SetColor(0, 0, 1)
                        self.points_list[i].GetProperty().SetColor(cell_rgb)

                    if len(self.points_mark_list):
                        for j in range(len(self.points_mark_list)):
                            if (self.points_mark_list[j][0] == self.Position[i][0]
                                    and self.points_mark_list[j][1] == self.Position[i][1]
                                    and self.points_mark_list[j][2] == self.Position[i][2]):
                                self.points_list[i].GetProperty().SetColor(self.mark_color)
                                break
                else:
                    if self.points_list[i].GetProperty().GetColor() == (1, 0, 0):
                        if self.mark_render_text == "Single-color":
                            cell_rgb = self.create_cell_random_color(2)
                        else:
                            cell_rgb = self.create_cell_random_color()
                        # self.points_list[i].GetProperty().SetColor(0, 1, 0)
                        self.points_list[i].GetProperty().SetColor(cell_rgb)

        coincidence = False
        # If no points have been marked yet, return False directly
        if len(self.Position) == 0:
            return coincidence
        # 计算方向向量v
        v_ = np.array(pick_pos) - np.array(camera_pos)
        pick_pos = np.array(pick_pos)
        all_points = np.array(self.Position) * np.array([self.x_px, self.y_px, self.z_px])
        nearest_point, min_distance, min_idx = nearest_point_to_line_fast(all_points, pick_pos, v_)
        # for i in range(len(self.Position)):  # 判断点集位置是否重合
        #     x = self.Position[i][0] * self.x_px
        #     y = self.Position[i][1] * self.y_px
        #     z = self.Position[i][2] * self.z_px
        x, y, z = nearest_point
        coin_r = self.sphere_size / 2
        if self.sift_points([x, x, y, y, z, z], pos, coin_r):
            coincidence = True
            # break
        return coincidence


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
