# -*- coding: utf-8 -*-
import vtkmodules.all as vtk
import numpy as np
from cell_points_marking.CreateImageData import CreateImageData
from cell_points_marking.MarkingPointsFun import nearest_point_to_line_fast
from cell_points_marking.ControlStyle import sphere_button_show_style, sphere_button_hide_style


class MarkSphereProcess:
    '''生成球体'''
    @classmethod
    def _create_sphere(cls, pos, self):
        actor = CreateImageData.create_sphere_actor(pos, self.sphere_size)
        cell_rgb = self.create_cell_random_color(1)
        actor.GetProperty().SetColor(cell_rgb)  # 颜色
        actor.VisibilityOn()
        pos = [pos[0] / self.x_px, pos[1] / self.y_px, pos[2] / self.z_px]
        self.Position.append(pos)
        self.points_list.append(actor)  # 加入球体列表
        self.ren.AddActor(actor)  # 加入渲染器
        self.points_count = len(self.points_list)
        # print("点数为：", self.points_count)
        self.nums_label.setText(str(self.points_count))
        self.renWin.Render()  # 重新渲染
        self.index = self.points_count - 1
        self.select_sphere_whether = True  # 生成球体可以执行删除

        details = []
        # 点信息
        details.append({
            "actor": actor,
            "pos": pos,
            "is_del": 0
        })
        # Backup
        self.points_withdraw_info.append(details)

    '''球体显示和隐藏'''
    @classmethod
    def sphere_show_hide(cls, self):
        if self.dataImporter_index:
            if self.sphere_hide:
                if self.data_volume_show_hide_count:
                    for i in range(len(self.Position)):
                        self.points_list[i].VisibilityOn()
                else:
                    for i in range(len(self.Position)):
                        if self.sift_points(self.box_size, self.Position[i], self.sphere_size):
                            self.points_list[i].VisibilityOn()
                        else:
                            self.points_list[i].VisibilityOff()
                self.sphere_hide = False
                self.sphere_button.setStyleSheet(sphere_button_show_style)
            else:
                for i in range(len(self.Position)):
                    self.points_list[i].VisibilityOff()
                self.sphere_hide = True
                self.sphere_button.setStyleSheet(sphere_button_hide_style)
            self.renWin.Render()

    '''选中球体'''
    @classmethod
    def sphere_select_and_mark(cls, self):
        """选择球体"""
        if len(self.points_list):
            pick_pos = self.MarkingPointsFun.get_picker_position(self)
            camera_pos = self.camera.GetPosition()  # 获取相机位置坐标
            # 计算方向向量v
            v_ = np.array(camera_pos) - np.array(pick_pos)
            # 计算向量diffSet
            box_size = self.sift_box(self.img3d, 0)

            all_points = np.array(self.Position) * np.array([self.x_px, self.y_px, self.z_px])

            for i in range(len(self.Position)):
                if self.sift_points(box_size, self.Position[i], self.sphere_size):
                    if self.sift_points(box_size, self.Position[i], -self.sphere_size / 2):
                        if self.mark_render_text == "Single-color":
                            cell_rgb = self.create_cell_random_color(3)
                        else:
                            cell_rgb = self.create_cell_random_color()
                        if self.points_list[i].GetProperty().GetColor() == tuple(self.mark_select_color):
                            # self.points_list[i].GetProperty().SetColor(0, 0, 1)
                            self.points_list[i].GetProperty().SetColor(cell_rgb)

                        if len(self.points_mark_list):
                            for j in range(len(self.points_mark_list)):
                                if (self.points_mark_list[j][0] == self.Position[i][0]
                                        and self.points_mark_list[j][1] == self.Position[i][1]
                                        and self.points_mark_list[j][2] == self.Position[i][2]):
                                    self.points_list[i].GetProperty().SetColor(self.mark_color)
                    else:
                        if self.mark_render_text == "Single-color":
                            cell_rgb = self.create_cell_random_color(2)
                        else:
                            cell_rgb = self.create_cell_random_color()
                        if self.points_list[i].GetProperty().GetColor() == tuple(self.mark_select_color):
                            # self.points_list[i].GetProperty().SetColor(0, 1, 0)
                            self.points_list[i].GetProperty().SetColor(cell_rgb)

            nearest_point, min_distance, min_idx = nearest_point_to_line_fast(all_points, np.array(pick_pos),
                                                                              v_)

            if min_distance <= self.sphere_size:
                if self.sift_points(box_size, self.Position[min_idx], self.sphere_size):
                    # print("Selected")
                    pickedActor = self.points_list[min_idx]
                    cell_rgb = self.create_cell_random_color(1)
                    pickedActor.GetProperty().SetColor(cell_rgb)
                    self.index = min_idx
                    self.select_sphere_whether = True  # 选中球体
            else:
                self.select_sphere_whether = False  # 未选中球体
            self.renWin.Render()
            return

    '''标记球体'''
    @classmethod
    def sphere_mark(cls, self):
        shiftKey = self.interactor.GetShiftKey()
        if shiftKey and len(self.points_list):
            i = self.index
            if len(self.points_mark_list):
                for j in range(len(self.points_mark_list)):
                    if (self.points_mark_list[j][0] == self.Position[i][0]
                            and self.points_mark_list[j][1] == self.Position[i][1]
                            and self.points_mark_list[j][2] == self.Position[i][2]):
                        return
            pickedActor = self.points_list[i]
            pickedActor.GetProperty().SetColor(self.mark_color)
            self.points_mark_list.append(self.Position[i])
            self.renWin.Render()

    '''删除框架球体'''
    @classmethod
    def delete_sphere(cls, self):
        details = []

        if len(self.points_list):
            '''选中删除'''
            if self.select_sphere_whether:  # 如果选中球体，可以删除
                i = self.index
                if len(self.points_mark_list):
                    for j in range(len(self.points_mark_list)):
                        # print("self.points_mark_list[j]:", self.points_mark_list[j])
                        if (self.points_mark_list[j][0] == self.Position[i][0]
                                and self.points_mark_list[j][1] == self.Position[i][1]
                                and self.points_mark_list[j][2] == self.Position[i][2]):
                            self.points_mark_list.remove(self.points_mark_list[j])
                            break
                # 点信息
                details.append({
                    "actor": self.points_list[i],
                    "pos": self.Position[i],
                    "is_del": 1
                })
                # Backup
                self.points_withdraw_info.append(details)

                self.ren.RemoveActor(self.points_list[i])  # 删除渲染的演员
                self.points_list.remove(self.points_list[i])  # 删除列表的演员
                self.Position.remove(self.Position[i])

                self.points_count = len(self.Position)
                self.nums_label.setText(str(self.points_count))

                if self.points_count:  # 调整球体序号
                    self.index = self.points_count - 1
                self.select_sphere_whether = False  # 成功删除后重新判定
                self.renWin.Render()

    '''清除球体'''

    @classmethod
    def delete_all_sphere(cls, self):
        details = []
        if len(self.points_list):
            box_size = self.sift_box(self.img3d, 0)
            index = 0
            for i in range(len(self.Position)):
                if self.sift_points(box_size, self.Position[index], 0):  # Delete，序号index不变
                    if len(self.points_mark_list):
                        for j in range(len(self.points_mark_list)):
                            if (self.points_mark_list[j][0] == self.Position[index][0]
                                    and self.points_mark_list[j][1] == self.Position[index][1]
                                    and self.points_mark_list[j][2] == self.Position[index][2]):
                                self.points_mark_list.remove(self.points_mark_list[j])
                                break
                    # 点信息
                    details.append({
                        "actor": self.points_list[index],
                        "pos": self.Position[index],
                        "is_del": 1
                    })
                    self.ren.RemoveActor(self.points_list[index])  # 删除渲染的演员
                    self.points_list.remove(self.points_list[index])  # 删除列表的演员
                    self.Position.remove(self.Position[index])
                else:  # 未删除，序号index增加
                    index += 1
            # Backup
            self.points_withdraw_info.append(details)
            self.points_count = len(self.Position)
            self.nums_label.setText(str(self.points_count))
            if self.points_count:
                self.index = self.points_count - 1
            self.renWin.Render()

    '''去除边缘标记'''
    @classmethod
    def remove_edge_sphere(cls, self, rx, ry, rz, shapes):
        def is_inside(nx, ny, nz):
            """判断坐标是否在内部有效范围内（0-based 索引）"""
            return (rx <= nx <= shapes[2] - rx and
                    ry <= ny <= shapes[1] - ry and
                    rz <= nz <= shapes[0] - rz)

        details = []
        if len(self.points_list):  # 对象列表
            index = 0
            for i in range(len(self.Position)):  # 坐标列表
                # if len(self.points_mark_list):
                #     for j in range(len(self.points_mark_list)):
                #         if (self.points_mark_list[j][0] == self.Position[index][0]
                #                 and self.points_mark_list[j][1] == self.Position[index][1]
                #                 and self.points_mark_list[j][2] == self.Position[index][2]):
                #             self.points_mark_list.remove(self.points_mark_list[j])
                #             break
                if is_inside(self.Position[index][0], self.Position[index][1], self.Position[index][2]):
                    index += 1
                    continue
                # 点信息
                details.append({
                    "actor": self.points_list[index],
                    "pos": self.Position[index],
                    "is_del": 1
                })
                self.ren.RemoveActor(self.points_list[index])  # 删除渲染的演员
                self.points_list.remove(self.points_list[index])  # 删除列表的演员
                self.Position.remove(self.Position[index])
            # Backup
            self.points_withdraw_info.append(details)
            self.points_count = len(self.Position)
            self.nums_label.setText(str(self.points_count))
            if self.points_count:
                self.index = self.points_count - 1
            self.renWin.Render()

    '''重新生成删除的球体'''

    @classmethod
    def shpere_withdraw(cls, self):
        if self.one_sphere_count:
            self.ren.RemoveActor(self.one_point_actor)  # 删除点
            self.ren.RemoveActor(self.one_line_actor)  # 删除直线
            self.point_line_count = True
            self.one_sphere_count = False
        else:
            if len(self.points_withdraw_info):  # 备份列表

                if self.select_sphere_whether:  # 如果选中球体，Update
                    if len(self.points_list) and self.index < len(self.points_list):
                        actor = self.points_list[self.index]
                        if self.mark_render_text == "Single-color":
                            cell_rgb = self.create_cell_random_color(3)
                        else:
                            cell_rgb = self.create_cell_random_color()
                        # actor.GetProperty().SetColor(0, 0, 1)
                        actor.GetProperty().SetColor(cell_rgb)
                        self.select_sphere_whether = False

                details = self.points_withdraw_info[-1]

                for detail in details:
                    actor = detail.get("actor", None)
                    pos = detail.get("pos", None)
                    is_del = detail.get("is_del")

                    if is_del:  # Restore
                        if self.mark_render_text == "Single-color":
                            cell_rgb = self.create_cell_random_color(3)
                        else:
                            cell_rgb = self.create_cell_random_color()
                        # actor.GetProperty().SetColor(0, 0, 1)
                        actor.GetProperty().SetColor(cell_rgb)
                        self.ren.AddActor(actor)
                        self.points_list.append(actor)
                        self.Position.append(pos)
                    else:  # Delete
                        indices = [i for i, act in enumerate(self.points_list) if actor == act]
                        if len(indices) > 0 and self.points_list and self.Position:
                            self.ren.RemoveActor(actor)
                            self.points_list.remove(actor)
                            self.Position.remove(pos)

                self.points_withdraw_info.pop()
                self.points_count = len(self.Position)
                self.nums_label.setText(str(self.points_count))
                if self.points_count:
                    self.index = self.points_count - 1
        self.renWin.Render()

    '''更新球体颜色'''
    @classmethod
    def update_sphere_color(cls, self, _mode):
        if len(self.points_list):
            # 计算向量diffSet
            box_size = self.sift_box(self.img3d, 0)
            for i in range(len(self.Position)):
                if self.sift_points(box_size, self.Position[i], self.sphere_size):
                    if self.sift_points(box_size, self.Position[i], -self.sphere_size / 2):
                        if _mode == "Single-color":
                            cell_rgb = self.create_cell_random_color(3)
                        else:
                            cell_rgb = self.create_cell_random_color()
                        self.points_list[i].GetProperty().SetColor(cell_rgb)

                        if len(self.points_mark_list):
                            for j in range(len(self.points_mark_list)):
                                if (self.points_mark_list[j][0] == self.Position[i][0]
                                        and self.points_mark_list[j][1] == self.Position[i][1]
                                        and self.points_mark_list[j][2] == self.Position[i][2]):
                                    self.points_list[i].GetProperty().SetColor(self.mark_color)
                    else:
                        if _mode == "Single-color":
                            cell_rgb = self.create_cell_random_color(2)
                        else:
                            cell_rgb = self.create_cell_random_color()
                        self.points_list[i].GetProperty().SetColor(cell_rgb)
            self.renWin.Render()

    '''更新球体位置'''
    @classmethod
    def update_sphere_position(cls, self):
        points_list_copy = []
        for i in range(len(self.Position)):
            pos = [self.Position[i][0] * self.x_px, self.Position[i][1] * self.y_px,
                   self.Position[i][2] * self.z_px]

            actor = CreateImageData.create_sphere_actor(pos, self.sphere_size)

            points_list_copy.append(actor)  # 加入球体列表
            self.ren.RemoveActor(self.points_list[i])
            self.ren.AddActor(actor)  # 加入渲染器
            if i == self.index:
                cell_rgb = self.create_cell_random_color(1)
                actor.GetProperty().SetColor(cell_rgb)  # 颜色
            else:
                if self.sift_points(self.box_size, self.Position[i], self.sphere_size):
                    if self.sift_points(self.box_size, self.Position[i], -self.sphere_size/2):
                        if self.mark_render_text == "Single-color":
                            cell_rgb = self.create_cell_random_color(3)
                            actor.GetProperty().SetColor(cell_rgb)  # 颜色
                        else:
                            cell_rgb = self.points_list[i].GetProperty().GetColor()
                            if cell_rgb == (1.0, 1.0, 1.0):
                                # if self.mark_render_text == "Single-color":
                                #     cell_rgb = self.create_cell_random_color(3)
                                # else:
                                cell_rgb = self.create_cell_random_color()
                                actor.GetProperty().SetColor(cell_rgb)  # 颜色
                            else:
                                # actor.GetProperty().SetColor(0, 0, 1)  # 颜色
                                actor.GetProperty().SetColor(cell_rgb)  # 颜色

                        if len(self.points_mark_list):
                            for j in range(len(self.points_mark_list)):
                                if (self.points_mark_list[j][0] == self.Position[i][0]
                                        and self.points_mark_list[j][1] == self.Position[i][1]
                                        and self.points_mark_list[j][2] == self.Position[i][2]):
                                    actor.GetProperty().SetColor(self.mark_color)
                                    break
                    else:
                        if self.mark_render_text == "Single-color":
                            cell_rgb = self.create_cell_random_color(2)
                            actor.GetProperty().SetColor(cell_rgb)  # 颜色
                        else:
                            cell_rgb = self.points_list[i].GetProperty().GetColor()
                            if cell_rgb == (1.0, 1.0, 1.0):
                                # if self.mark_render_text == "Single-color":
                                #     cell_rgb = self.create_cell_random_color(2)
                                # else:
                                cell_rgb = self.create_cell_random_color()
                                actor.GetProperty().SetColor(cell_rgb)  # 颜色
                            else:
                                # actor.GetProperty().SetColor(0, 1, 0)  # 颜色
                                actor.GetProperty().SetColor(cell_rgb)  # 颜色

            if self.sift_points(self.box_size, self.Position[i], self.sphere_size):
                actor.VisibilityOn()
            else:
                actor.VisibilityOff()

        self.renWin.Render()
        self.points_list = points_list_copy

