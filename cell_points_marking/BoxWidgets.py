# -*- coding: utf-8 -*-
import vtkmodules.all as vtk
import numpy as np


class BoxWidgets:
    @classmethod
    def create_box(cls, box_volume, self):  # 创建box框
        if self.boxWidget:
            self.boxWidget.Off()
            del self.boxWidget
        self.boxWidget = vtk.vtkBoxWidget()
        self.boxWidget.SetProp3D(box_volume)  # 选择对象
        self.boxWidget.SetInteractor(self.interactor)  # 交互器
        self.boxWidget.SetPlaceFactor(1.0)  # 设置盒子小部件相对于所选择的对象的大小或位置的因子
        self.boxWidget.PlaceWidget()  # 用于设置或更新立方体部件的位置

        self.boxWidget.AddObserver(vtk.vtkCommand.InteractionEvent, self.boxCallback)  # 交互时触发事件
        self.boxWidget.Off()  # Hide
        self.box_show_hide_count = 0

    '''获取boxWidgetParameter'''
    @classmethod
    def box_call_back(cls, obj, event, self):
        transform = vtk.vtkTransform()  # 空间变换
        obj.GetTransform(transform)
        obj.GetProp3D().SetUserTransform(transform)
        box_Widget_size = obj.GetProp3D().GetBounds()
        self.box_Widget_cut(box_Widget_size)

    '''box切取的信息处理'''
    @classmethod
    def box_cut_info(cls, bw_size, self):
        if self.dataImporter_index:
            self.cut_label_size = []
            bw_size_min = [bw_size[0], bw_size[2], bw_size[4]]  # xyz
            bw_size_max = [bw_size[1], bw_size[3], bw_size[5]]  # xyz
            for i in range(len(bw_size_min)):
                _min_i = bw_size_min[i]
                if 0 <= _min_i <= self.img3d.shape[2 - i]:
                    _min_i = int(_min_i)
                else:
                    _min_i = 0
                self.cut_label_size.append(_min_i)
                _max_i = bw_size_max[i]
                if 0 <= _max_i < self.img3d.shape[2 - i]:
                    _max_i = int(_max_i) + 1
                else:
                    _max_i = self.img3d.shape[2 - i]
                self.cut_label_size.append(_max_i)

    '''切取box'''
    @classmethod
    def box_cut_label_image(cls, self):
        if self.dataImporter_index:
            if self.box_show_hide_count:
                self.box_size = self.cut_label_size
                self.origin = [self.cut_label_size[0], self.cut_label_size[2], self.cut_label_size[4]]
                self.xyz_cut_index = True  # 步长冗余变化，避免图像重复生成
                self.x_step_num.setValue(self.cut_label_size[1] - self.cut_label_size[0])
                self.y_step_num.setValue(self.cut_label_size[3] - self.cut_label_size[2])
                self.z_step_num.setValue(self.cut_label_size[5] - self.cut_label_size[4])
                self.create_new_image()
                self.xyz_cut_index = False

    '''显示隐藏boxWidget'''
    @classmethod
    def box_show_hide_update(cls, self):
        if self.dataImporter_index:
            if self.box_show_hide_count:
                self.boxWidget.Off()
                self.box_show_hide_count = 0
            else:
                self.boxWidget.On()
                self.box_show_hide_count = 1
            self.renWin.Render()

    '''点击选取感兴趣区域'''
    @classmethod
    def point_to_box_image(cls, self):
        if self.dataImporter_index:
            shiftKey = self.interactor.GetShiftKey()
            if not shiftKey:
                return
            img = self.img3d
            pick_pos = self.MarkingPointsFun.get_picker_position(self)
            camera_pos = self.camera.GetPosition()  # 获取相机位置坐标
            # 找到直线与图像的边界交点
            intersection_points = self.MarkingPointsFun.set_intersection_bound(pick_pos, camera_pos, self)
            if len(intersection_points) > 0:
                # 直线和图像数据的交点集
                points, count = self.collect_points(intersection_points[0], intersection_points[1])
                if count:
                    maxPoint = self.points_pos_sure(points)
                    if len(maxPoint):
                        dx = img.shape[2] / 6
                        dy = img.shape[1] / 6
                        x1, x2 = int(maxPoint[0] - dx), int(maxPoint[0] + dx)
                        y1, y2 = int(maxPoint[1] - dy), int(maxPoint[1] + dy)
                        b = x1, x2, y1, y2, self.box_size[-2], self.box_size[-1]
                        self.box_Widget_cut(b)
                        self.box_size = self.cut_label_size
                        self.origin = [self.cut_label_size[0], self.cut_label_size[2], self.cut_label_size[4]]
                        self.xyz_cut_index = True  # 步长冗余变化，避免图像重复生成
                        self.x_step_num.setValue(self.cut_label_size[1] - self.cut_label_size[0])
                        self.y_step_num.setValue(self.cut_label_size[3] - self.cut_label_size[2])
                        self.create_new_image()
                        self.xyz_cut_index = False
