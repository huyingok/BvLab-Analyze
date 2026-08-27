# -*- coding: utf-8 -*-
import vtkmodules.all as vtk
import numpy as np
from line_points_marking.CreateImageData import CreateImageData
from line_points_marking.ThreeLineSwcCurve import ThreeCurveFitPc


class MarkProcess:

    """生成点"""

    @classmethod
    def _create_point(cls, pos, p_size, self, rgb):
        actor = CreateImageData.create_point_actor(pos, p_size)
        actor.GetProperty().SetColor(rgb)  # 颜色
        actor.VisibilityOn()

        self.new_points_list.append(actor)
        self.new_Position.append(pos)

        self.ren.AddActor(actor)  # 加入渲染器

    """生成线"""

    @classmethod
    def _create_line(cls, pos2, is_left, self):
        pos1 = self.new_Position[-1]

        if is_left:
            sp = np.array(pos1, dtype=np.float32)  # 起点
            ep = np.array(pos2, dtype=np.float32)  # 终点

            img = self.img3d.copy()
            # min_th = max(0, self.gray_min)
            # max_th = min(255, self.gray_max)
            # img[img < min_th] = 0
            # img[img > max_th] = max_th

            inter_points = ThreeCurveFitPc(img, sp, ep)[1:-1, 2:5]
            points = [sp] + [np.round(p, 2) for p in inter_points][:-1] + [ep]

            for i, point in enumerate(points[1:]):
                line_actor = CreateImageData.create_line_actor(points[i], point)
                line_actor.GetProperty().SetColor(0, 0, 1)  # 颜色
                self.new_mark_lines.append(line_actor)
                self.ren.AddActor(line_actor)  # 加入渲染器
                if i != len(points[1:]) - 1:
                    # 创建插值点对象
                    p_size = self.p_size + 2
                    cls._create_point(point, p_size, self, (0, 0, 1))
        else:
            line_actor = CreateImageData.create_line_actor(pos1, pos2)
            line_actor.GetProperty().SetColor(0, 0, 1)  # 颜色
            self.new_mark_lines.append(line_actor)
            self.ren.AddActor(line_actor)  # 加入渲染器

        if len(self.new_mark_lines) == 0 and len(self.new_Position) == 2:
            print("Marking error, skipping")

    """生成点线PolyData"""

    @classmethod
    def _create_PolyData(cls, points, lines, vertices, self):
        actor = CreateImageData.create_PolyData_actor(points, lines, vertices, self)
        color_ = self.create_random_color()  # 获取随机颜色
        actor.GetProperty().SetColor(color_)  # 颜色
        actor.GetProperty().SetLineWidth(2)  # 线宽，Default2
        actor.GetProperty().SetPointSize(self.p_size)  # 点大小
        self.ren.AddActor(actor)  # 加入渲染器
        actor.VisibilityOn()
        return actor

    """删除点线PolyData"""

    @classmethod
    def delete_PolyData(cls, self):
        details = []
        actor = self.vtk_LP_Actor[self.color_i]
        # 备份删除的对象
        details.append({
            "actor": actor,
            "points": self.vtk_Points[self.color_i],
            "lines": self.vtk_Lines[self.color_i],
            "vertices": self.vtk_Vertices[self.color_i],
            "del": 1
        })
        self.ren.RemoveActor(actor)
        self.vtk_LP_Actor.pop(self.color_i)
        self.vtk_Points.pop(self.color_i)
        self.vtk_Lines.pop(self.color_i)
        self.vtk_Vertices.pop(self.color_i)
        # Backup
        self.PolyData_copy.append(
            details
        )
        self.renWin.Render()
        self.color_i = None

        self.tree_count = len(self.vtk_LP_Actor)
        self.label_count.setText(f"{self.tree_count}")

    """删除所有点线PolyData"""

    @classmethod
    def delete_all_PolyData(cls, self):
        if self.vtk_LP_Actor:
            details = []
            for i in range(len(self.vtk_LP_Actor)):
                actor = self.vtk_LP_Actor[i]
                points = self.vtk_Points[i]
                lines = self.vtk_Lines[i]
                vertices = self.vtk_Vertices[i]
                # 备份删除的对象
                details.append({
                    "actor": actor,
                    "points": points,
                    "lines": lines,
                    "vertices": vertices,
                    "del": 1
                })
                self.ren.RemoveActor(actor)
            # Backup
            self.PolyData_copy.append(
                details
            )
            self.vtk_LP_Actor = []
            self.vtk_Points = []
            self.vtk_Lines = []
            self.vtk_Vertices = []
            self.renWin.Render()
            self.color_i = None

            self.tree_count = len(self.vtk_LP_Actor)
            self.label_count.setText(f"{self.tree_count}")

    """删除所有限制长度PolyData"""

    @classmethod
    def delete_all_PolyData_limit(cls, self):
        if self.vtk_LP_Actor:
            details = []
            del_i = []
            for i in range(len(self.vtk_LP_Actor)):
                points = self.vtk_Points[i]
                num_points = points.GetNumberOfPoints()  # 获取点数
                if num_points <= self.length_size:
                    del_i.append(i)
                    actor = self.vtk_LP_Actor[i]
                    points = self.vtk_Points[i]
                    lines = self.vtk_Lines[i]
                    vertices = self.vtk_Vertices[i]
                    # 备份删除的对象
                    details.append({
                        "actor": actor,
                        "points": points,
                        "lines": lines,
                        "vertices": vertices,
                        "del": 1
                    })
                    self.ren.RemoveActor(actor)
                    self.color_i = None
            # Backup
            self.PolyData_copy.append(
                details
            )
            del_i = del_i[::-1]
            for i in del_i:
                self.vtk_LP_Actor.pop(i)
                self.vtk_Points.pop(i)
                self.vtk_Lines.pop(i)
                self.vtk_Vertices.pop(i)
            self.renWin.Render()

            self.tree_count = len(self.vtk_LP_Actor)
            self.label_count.setText(f"{self.tree_count}")

    """恢复点线PolyData"""

    @classmethod
    def PolyData_withdraw(cls, self):
        if len(self.PolyData_copy):
            details = self.PolyData_copy[-1]
            for detail in details:
                del_index = detail.get("del", None)
                actor = detail.get("actor", None)
                points = detail.get("points", None)
                lines = detail.get("lines", None)
                vertices = detail.get("vertices", None)
                if del_index == 1:  # Restore
                    # 更新样式
                    color_ = self.create_random_color()  # 获取随机颜色
                    actor.GetProperty().SetColor(color_)  # 颜色
                    actor.GetProperty().SetLineWidth(2)  # 线宽
                    actor.GetProperty().SetPointSize(self.p_size)  # 点大小
                    self.ren.AddActor(actor)
                    self.vtk_LP_Actor.append(actor)
                    self.vtk_Points.append(points)
                    self.vtk_Lines.append(lines)
                    self.vtk_Vertices.append(vertices)
                else:  # Delete
                    indices = [i for i, act in enumerate(self.vtk_LP_Actor) if actor == act]
                    if len(indices) > 0:
                        self.ren.RemoveActor(actor)
                        self.vtk_LP_Actor.remove(actor)
                        self.vtk_Points.remove(points)
                        self.vtk_Lines.remove(lines)
                        self.vtk_Vertices.remove(vertices)

            self.renWin.Render()
            self.PolyData_copy.pop()

            self.tree_count = len(self.vtk_LP_Actor)
            self.label_count.setText(f"{self.tree_count}")

    '''标记显示和隐藏'''

    @classmethod
    def LP_show_hide(cls, self):
        if self.dataImporter_index:
            if self.LP_hide:  # 如果标记隐藏：True
                for act in self.vtk_LP_Actor:
                    act.VisibilityOn()
                self.LP_hide = False
                self.show_lines_button.setStyleSheet("QPushButton {background-color: red; width: 20px; height: 20px;"
                                                     "border-radius: 0px;}")
            else:
                for act in self.vtk_LP_Actor:
                    act.VisibilityOff()
                self.LP_hide = True
                self.show_lines_button.setStyleSheet("QPushButton {background-color: white; width: 20px; height: 20px;"
                                                     "border-radius: 0px;}")
            self.renWin.Render()

    """包围盒显示和隐藏"""

    @classmethod
    def outline_show_hide(cls, self):
        if self.dataImporter_index:
            if self.outline_hide:
                self.outlineActor.VisibilityOn()
                self.outline_hide = False
            else:
                self.outlineActor.VisibilityOff()
                self.outline_hide = True
            self.renWin.Render()

    """确认标记"""

    @classmethod
    def mark_sure(cls, self):
        for p in self.new_points_list:  # 删除点标记
            self.ren.RemoveActor(p)
        for l in self.new_mark_lines:  # 删除线标记
            self.ren.RemoveActor(l)

        details = []

        if len(self.new_Position) > 1:
            if not len(self.is_marking_coincidence):  # 没有重合，直接确认标记
                # color_ = self.create_random_color()
                points = vtk.vtkPoints()  # 创建点集合
                lines = vtk.vtkCellArray()  # 创建单元数组
                # 创建顶点单元（每个点是一个顶点）
                vertices = vtk.vtkCellArray()
                for i, point in enumerate(self.new_Position):
                    points.InsertNextPoint(point)
                    if i > 0:
                        # 创建线段
                        line = vtk.vtkLine()
                        line.GetPointIds().SetId(0, i - 1)  # 第一个点
                        line.GetPointIds().SetId(1, i)  # 第二个点
                        # 单元数组添加线段
                        lines.InsertNextCell(line)

                    vertices.InsertNextCell(1)
                    vertices.InsertCellPoint(i)
                actor = cls._create_PolyData(points, lines, vertices, self)
                self.vtk_Points.append(points)
                self.vtk_Lines.append(lines)
                self.vtk_Vertices.append(vertices)
                self.vtk_LP_Actor.append(actor)
                # 备份新增的对象
                details.append({
                    "points": points,
                    "lines": lines,
                    "vertices": vertices,
                    "actor": actor,
                    "del": 0
                })
                # Backup
                self.PolyData_copy.append(
                    details
                )
            else:
                is_coincidence = self.is_marking_coincidence.get('coincidence', None)
                if is_coincidence:
                    # 找到树的分支序号
                    branch_i = self.is_marking_coincidence.get('tree_i')
                    index0 = self.is_marking_coincidence.get('tree_index')

                    old_actor = self.vtk_LP_Actor[branch_i]
                    points = self.vtk_Points[branch_i]
                    num_points = points.GetNumberOfPoints()  # 点集数
                    vertices = self.vtk_Vertices[branch_i]
                    lines = self.vtk_Lines[branch_i]
                    # 创建新对象替换旧对象，方便备份
                    new_points = vtk.vtkPoints()  # 创建点集合
                    new_lines = vtk.vtkCellArray()  # 创建单元数组
                    new_vertices = vtk.vtkCellArray()  # 创建顶点单元（每个点是一个顶点）
                    # 备份删除的对象
                    details.append({
                        "points": points,
                        "lines": lines,
                        "vertices": vertices,
                        "actor": old_actor,
                        "del": 1
                    })

                    for j in range(num_points):
                        coord = points.GetPoint(j)
                        # 创建新对象
                        new_points.InsertNextPoint(coord)

                    lines.InitTraversal()
                    id_list = vtk.vtkIdList()

                    id_l = []
                    count = 0
                    while lines.GetNextCell(id_list):
                        id_ = []
                        for k in range(id_list.GetNumberOfIds()):
                            id_.append(id_list.GetId(k))
                        id_l.append(id_)

                        # 创建线段
                        line = vtk.vtkLine()
                        line.GetPointIds().SetId(0, id_[0])  # 第一个点
                        line.GetPointIds().SetId(1, id_[1])  # 第二个点
                        # 单元数组添加线段
                        new_lines.InsertNextCell(line)
                        if count == 0:
                            new_vertices.InsertNextCell(1)
                            new_vertices.InsertCellPoint(count)
                            count += 1
                        new_vertices.InsertNextCell(1)
                        new_vertices.InsertCellPoint(count)
                        count += 1

                    index1 = num_points
                    for m, point in enumerate(self.new_Position):  # 从第二个点开始
                        if m > 0:
                            new_points.InsertNextPoint(point)
                            # 创建线段
                            line = vtk.vtkLine()
                            line.GetPointIds().SetId(0, index0)  # 第一个点
                            line.GetPointIds().SetId(1, index1)  # 第二个点
                            # 单元数组添加线段
                            new_lines.InsertNextCell(line)
                            index0 = index1
                            index1 = index0 + 1
                            new_vertices.InsertNextCell(1)
                            new_vertices.InsertCellPoint(num_points + m - 1)

                    color_ = old_actor.GetProperty().GetColor()  # 颜色
                    self.ren.RemoveActor(old_actor)
                    # 创建 PolyData 并设置点和线
                    actor = cls._create_PolyData(new_points, new_lines, new_vertices, self)
                    actor.GetProperty().SetColor(color_)
                    # 替换对象
                    self.vtk_Points[branch_i] = new_points
                    self.vtk_Lines[branch_i] = new_lines
                    self.vtk_Vertices[branch_i] = new_vertices
                    self.vtk_LP_Actor[branch_i] = actor
                    # 备份新对象
                    details.append({
                        "points": new_points,
                        "lines": new_lines,
                        "vertices": new_vertices,
                        "actor": actor,
                        "del": 0
                    })
                    # Backup
                    self.PolyData_copy.append(
                        details
                    )

        self.new_points_list = []
        self.new_Position = []
        self.new_mark_lines = []
        self.is_marking_coincidence = {}

        if self.LP_hide:
            for act in self.vtk_LP_Actor:
                act.VisibilityOn()
            self.LP_hide = False
        # 重新渲染
        self.renWin.Render()

        self.tree_count = len(self.vtk_LP_Actor)
        self.label_count.setText(f"{self.tree_count}")

    """鼠标左键点击：添加识别点"""

    @classmethod
    def LeftButtonPressSet(cls, self):
        if self.vtk_LP_Actor:
            ctrlKey = self.interactor.GetControlKey()
            shiftKey = self.interactor.GetShiftKey()
            altKey = self.interactor.GetAltKey()
            if altKey and not ctrlKey and not shiftKey:
                if self.color_i is not None:
                    try:
                        color_ = self.create_random_color()
                        act = self.vtk_LP_Actor[self.color_i]
                        act.GetProperty().SetColor(color_)  # 更新颜色
                        act.GetProperty().SetPointSize(self.p_size)  # 点大小
                    except IndexError as e:
                        self.error_print(e)
                    self.color_i = None

                pick_pos = self.MarkingPointsFun.get_picker_position(self)
                camera_pos = self.camera.GetPosition()  # 获取相机位置坐标

                select_r = 2
                # 寻找最近树和点
                norms, norms_pos, norms_index = self.MarkingPointsFun.check_nearest_tree_point(camera_pos, pick_pos,
                                                                                               select_r, self)
                i = np.argmin(norms)  # 选中树序号
                min_norm = norms[i]  # 最近距离

                if min_norm <= select_r:
                    if len(self.connect_list):
                        if self.connect_list[0].get("tree_index") == i:  # 第二个点是否在同一树上
                            return

                    min_pos_index = norms_index[i]  # 选中点序号

                    if len(self.connect_list) > 1 and self.new_points_list and self.new_Position:
                        self.connect_list.pop()
                        self.ren.RemoveActor(self.new_points_list[-1])
                        self.new_points_list.pop()
                        self.new_Position.pop()

                    self.connect_list.append({
                        "tree_index": i,
                        "pos_index": min_pos_index,
                        "left_index": 1,
                    })
                    pos = norms_pos[i]
                    self.focal_pos = pos
                    cls._create_point(pos, self.p_size + 2, self, (0, 0, 1))  # 创建点
                    self.renWin.Render()  # 重新渲染

    """鼠标右键点击：无识别点"""

    @classmethod
    def RightButtonPressSet(cls, self):
        if self.vtk_LP_Actor:
            ctrlKey = self.interactor.GetControlKey()
            shiftKey = self.interactor.GetShiftKey()
            altKey = self.interactor.GetAltKey()
            if altKey and not ctrlKey and not shiftKey:
                if self.color_i is not None:
                    try:
                        color_ = self.create_random_color()
                        act = self.vtk_LP_Actor[self.color_i]
                        act.GetProperty().SetColor(color_)  # 更新颜色
                        act.GetProperty().SetPointSize(self.p_size)  # 点大小
                    except IndexError as e:
                        self.error_print(e)
                    self.color_i = None

                pick_pos = self.MarkingPointsFun.get_picker_position(self)
                camera_pos = self.camera.GetPosition()  # 获取相机位置坐标

                select_r = 2
                # 寻找最近树和点
                norms, norms_pos, norms_index = self.MarkingPointsFun.check_nearest_tree_point(camera_pos, pick_pos,
                                                                                               select_r, self)
                i = np.argmin(norms)  # 选中树序号
                min_norm = norms[i]  # 最近距离

                if min_norm <= select_r:
                    if len(self.connect_list):
                        if self.connect_list[0].get("tree_index") == i:  # 第二个点是否在同一树上
                            return

                    min_pos_index = norms_index[i]  # 选中点序号
                    if len(self.connect_list) > 1 and self.new_points_list and self.new_Position:
                        self.connect_list.pop()
                        self.ren.RemoveActor(self.new_points_list[-1])
                        self.new_points_list.pop()
                        self.new_Position.pop()

                    self.connect_list.append({
                        "tree_index": i,
                        "pos_index": min_pos_index,
                        "left_index": 0,
                    })
                    pos = norms_pos[i]
                    self.focal_pos = pos
                    cls._create_point(pos, self.p_size + 2, self, (0, 0, 1))  # 创建点
                    self.renWin.Render()  # 重新渲染

    """鼠标中键点击：取消连接或断开"""

    @classmethod
    def MiddleButtonPressSet(cls, self):
        if self.vtk_LP_Actor:
            ctrlKey = self.interactor.GetControlKey()
            shiftKey = self.interactor.GetShiftKey()
            altKey = self.interactor.GetAltKey()
            if altKey and not ctrlKey and not shiftKey:
                if self.connect_list:
                    for act in self.new_points_list:
                        self.ren.RemoveActor(act)
                    self.connect_list = []
                    self.new_points_list = []
                    self.new_Position = []
                    # 重新渲染
                    self.renWin.Render()

    '''撤回标记：点撤回'''

    @classmethod
    def points_withdraw(cls, self):
        if self.one_sphere_count:
            # self.ren.RemoveActor(self.one_point_actor)  # 删除点
            # self.ren.RemoveActor(self.one_line_actor)  # 删除直线
            for actor in self.cross_actor_points:
                self.ren.RemoveActor(actor)  # 删除渲染的点
            if not self.res_save_Button.isEnabled():
                self.point_line_count = True
                self.one_sphere_count = False
            self.radius_dialog.hide()
            self.res_save_Button.setEnabled(False)
            if len(self.cross_actor_points):
                self.cross_actor_points = []
            else:
                if len(self.cross_actor_points_copy):
                    for actor in self.cross_actor_points_copy:
                        self.ren.RemoveActor(actor)  # 删除渲染的点
                    self.cross_actor_points_copy = []
        else:
            if len(self.new_Position) > 1 and len(self.new_mark_lines) > 0:
                self.ren.RemoveActor(self.new_mark_lines[-1])  # 删除线
                self.new_mark_lines.pop()

                if len(self.new_radius_dict):
                    pos = self.new_Position[-1]
                    pos_key = self.pos_to_key(pos)
                    if self.new_radius_dict.get(pos_key):
                        self.new_radius_dict.pop(pos_key)  # 删除半径记录

                self.new_Position.pop()  # 删除点坐标
                self.ren.RemoveActor(self.new_points_list[-1])  # 删除点
                self.new_points_list.pop()
            else:
                if len(self.new_Position) > 0:
                    if len(self.new_radius_dict):
                        pos = self.new_Position[-1]
                        pos_key = self.pos_to_key(pos)
                        self.new_radius_dict.pop(pos_key)  # 删除半径记录

                    self.new_Position.pop()  # 删除点坐标
                    self.ren.RemoveActor(self.new_points_list[-1])  # 删除点
                    self.new_points_list.pop()
        self.renWin.Render()

    """撤回标记：全部撤回"""

    @classmethod
    def marks_withdraw(cls, self):
        if self.one_sphere_count:
            for actor in self.cross_actor_points:
                self.ren.RemoveActor(actor)  # 删除渲染的点
            for actor in self.cross_actor_points_copy:
                self.ren.RemoveActor(actor)  # 删除渲染的点
            # self.ren.RemoveActor(self.one_point_actor)  # 删除点
            # self.ren.RemoveActor(self.one_line_actor)  # 删除直线
            self.point_line_count = True
            self.one_sphere_count = False
            self.radius_dialog.hide()
            self.res_save_Button.setEnabled(False)
            self.cross_actor_points = []
            self.cross_actor_points_copy = []
        for i, line in enumerate(self.new_mark_lines):
            self.ren.RemoveActor(line)  # 删除线
        for i, point in enumerate(self.new_points_list):
            if i > 0:
                self.ren.RemoveActor(point)  # 删除点
        # 清空
        # self.new_Position = []
        self.new_mark_lines = []
        # self.new_points_list = []
        # self.new_radius_dict = {}
        # 保留第一个
        if self.new_Position:
            self.new_Position = self.new_Position[:1]
        if self.new_points_list:
            self.new_points_list = self.new_points_list[:1]
        if self.new_radius_dict:
            if self.new_Position:
                pos = self.new_Position[0]
                pos_key = self.pos_to_key(pos)
                r = self.new_radius_dict.get(pos_key)
                if r is not None:
                    self.new_radius_dict = {}
                    self.new_radius_dict[pos_key] = r
                else:
                    r = self.radius_dict.get(pos_key)
                    self.new_radius_dict = {}
                    self.new_radius_dict[pos_key] = r
        self.renWin.Render()

    '''更新点大小'''

    @classmethod
    def _update_point_size(cls, self):
        for act in self.vtk_LP_Actor:
            act.GetProperty().SetPointSize(self.p_size)  # 点大小
        self.renWin.Render()
