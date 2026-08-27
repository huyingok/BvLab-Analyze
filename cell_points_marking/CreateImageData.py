# -*- coding: utf-8 -*-
import math
import vtkmodules.all as vtk
import numpy as np


class CreateImageData:
    def __init__(self, image_info, render_mode_text):
        self.image_info = image_info
        self.render_mode_text = render_mode_text

    '''创建图形'''
    def create_img3d(self):  # 创建图
        data_matrix = self.image_info[0]  # 复制获取文件数据
        x_px, y_px, z_px = self.image_info[1:]
        import_data = vtk.vtkImageImport()  # 创建vtk.vtkImageImport()
        # 设置坐标轴
        import_data.SetWholeExtent(0, data_matrix.shape[2] - 1,
                                   0, data_matrix.shape[1] - 1,
                                   0, data_matrix.shape[0] - 1)
        import_data.SetDataExtentToWholeExtent()
        # 复制数据
        data_string = data_matrix.tobytes()
        import_data.CopyImportVoidPointer(data_string, len(data_string))
        # 设置间隔
        import_data.SetDataSpacing(x_px, y_px, z_px)
        # 设置数据类型
        if data_matrix.dtype == np.uint16:
            import_data.SetDataScalarTypeToUnsignedShort()
        else:
            import_data.SetDataScalarTypeToUnsignedChar()
        import_data.Update()

        # 将数据转换为图形数据
        volumeMapper = vtk.vtkSmartVolumeMapper()
        volumeMapper.SetBlendModeToMaximumIntensity()
        # volumeMapper.SetBlendModeToComposite()
        volumeMapper.SetInputConnection(import_data.GetOutputPort())
        # ⚡ MIP性能优化：计算量是Composite的5倍，适当放宽采样距离
        volumeMapper.SetInteractiveAdjustSampleDistances(False)
        volumeMapper.SetAutoAdjustSampleDistances(True)  # MIP开启自动采样加速（视觉影响小）
        volumeMapper.SetSampleDistance(1.5)  # MIP用1.5采样距离，Speed×2，画质损失可忽略
        # 直接修改mapper的混合模式
        if self.render_mode_text == "MIP":
            volumeMapper.SetBlendModeToMaximumIntensity()
        else:  # COMPOSITE
            volumeMapper.SetBlendModeToComposite()
        # ClassvtkVolume用于配对先前声明的卷以及属性，在渲染该体积时使用。
        volume = vtk.vtkVolume()
        volume.SetMapper(volumeMapper)
        # RGBSettings
        colorFunc = vtk.vtkColorTransferFunction()
        # 透明度设置
        alphaChannelFunc = vtk.vtkPiecewiseFunction()
        # 设置属性
        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(colorFunc)
        volumeProperty.SetScalarOpacity(alphaChannelFunc)
        volumeProperty.SetInterpolationType(3)
        volumeProperty.ShadeOn()
        volumeProperty.SetScalarOpacityUnitDistance(0.01)
        volume.SetProperty(volumeProperty)

        return import_data, volume, alphaChannelFunc, colorFunc

    '''创建包围盒'''
    @classmethod
    def create_outline(cls, data_importer):  # 创建包围盒
        # 添加包围盒
        outline = vtk.vtkOutlineFilter()
        outline.SetInputData(data_importer.GetOutput())
        outline.Update()
        outline_mapper = vtk.vtkPolyDataMapper()
        outline_mapper.SetInputData(outline.GetOutput())
        outline_actor = vtk.vtkActor()
        outline_actor.SetMapper(outline_mapper)
        outline_actor.GetProperty().SetColor(0, 1, 0)
        outline_actor.GetProperty().SetLineWidth(1)
        return data_importer, outline, outline_actor

    '''感兴趣区域更新'''
    @classmethod
    def create_box_image(cls, self):
        if self.box_volume:
            self.ren.RemoveActor(self.box_volume)  # 删除之前的区域
            self.ren.RemoveActor(self.box_outlineActor)
            self.ren.RemoveActor(self.box_outlineActor_copy)
        x1, x2, y1, y2, z1, z2 = np.array(self.box_size, dtype=np.int32)  # 获取尺寸

        image_info = [self.img3d[z1:z2, y1:y2, x1:x2], self.x_px, self.y_px, self.z_px]
        (self.box_dataImporter, self.box_volume,
         self.alphaChannelFunc, self.colorFunc) = CreateImageData(image_info, self.render_mode_text).create_img3d()

        CreateImageData.update_color_alpha(self.alphaChannelFunc, self.colorFunc,
                                           self.gray_min, self.gray_max, self.r, self.g, self.b)

        (dataImporter_box_outline, self.box_outline,
         self.box_outlineActor) = CreateImageData.create_outline(self.box_dataImporter)

        del dataImporter_box_outline

        (dataImporter_box_outline_copy, self.box_outline_copy,
         self.box_outlineActor_copy) = CreateImageData.create_outline(self.box_dataImporter)

        del dataImporter_box_outline_copy

        self.box_outlineActor.GetProperty().SetColor(1, 1, 0)
        self.box_outlineActor_copy.SetPosition([self.origin[0] * self.x_px, self.origin[1] * self.y_px,
                                                self.origin[2] * self.z_px])
        self.ren.AddActor(self.box_volume)
        self.ren.AddActor(self.box_outlineActor)
        self.box_outlineActor.SetPosition([self.origin[0] * self.x_px, self.origin[1] * self.y_px,
                                           self.origin[2] * self.z_px])
        self.box_volume.SetPosition([self.origin[0] * self.x_px, self.origin[1] * self.y_px,
                                     self.origin[2] * self.z_px])

        self.select_sphere_whether = False  # 更新图像后重新判定是否选中球体

        for i in range(len(self.Position)):  # 显示或隐藏区域内的标记
            if self.sift_points([x1, x2, y1, y2, z1, z2], self.Position[i], self.sphere_size):
                if self.sift_points([x1, x2, y1, y2, z1, z2], self.Position[i], -self.sphere_size / 2):
                    if self.mark_render_text == "Single-color":
                        cell_rgb = self.create_cell_random_color(3)
                    else:
                        cell_rgb = self.create_cell_random_color()
                    if self.points_list[i].GetProperty().GetColor() == tuple(self.mark_select_color):
                        # self.points_list[i].GetProperty().SetColor(0, 0, 1)
                        self.points_list[i].GetProperty().SetColor(cell_rgb)
                    elif self.points_list[i].GetProperty().GetColor() == (1.0, 1.0, 1.0):
                        self.points_list[i].GetProperty().SetColor(cell_rgb)  # 颜色

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
                    elif self.points_list[i].GetProperty().GetColor() == (1.0, 1.0, 1.0):
                        self.points_list[i].GetProperty().SetColor(cell_rgb)  # 颜色

                if self.sphere_hide:
                    self.points_list[i].VisibilityOff()
                else:
                    self.points_list[i].VisibilityOn()
            else:
                self.points_list[i].VisibilityOff()

        if self.dataImporter_index and self.box_volume:  # 选择区域
            # if self.data_volume_show_hide_count:  # 图文件显示
            #     self.volume.VisibilityOn()
            # else:
            #     self.volume.VisibilityOff()
            if self.outline_show_hide_count:
                self.box_volume.VisibilityOn()
                self.box_outlineActor.VisibilityOn()
            else:
                self.box_volume.VisibilityOff()
                self.box_outlineActor.VisibilityOff()

        self.box_Widget(self.box_outlineActor_copy)

        self.renWin.Render()

    '''更新图形颜色和灰度'''
    @classmethod
    def update_color_alpha(cls, alphaChannelFunc, colorFunc, gray_min, gray_max, r, g, b):
        alphaChannelFunc.RemoveAllPoints()
        alphaChannelFunc.AddPoint(gray_min, 0)
        alphaChannelFunc.AddPoint(gray_max, 1)
        colorFunc.RemoveAllPoints()
        colorFunc.AddRGBPoint(gray_min, 0, 0, 0)
        colorFunc.AddRGBPoint(gray_max, r, g, b)

    '''创建球体对象'''
    @classmethod
    def create_sphere_actor(cls, pos, sphere_size):
        sphere = vtk.vtkSphereSource()  # 球体
        sphere.SetCenter(pos)  # Coordinate
        sphere.SetRadius(sphere_size)  # Radius
        sphere.Update()  # Update
        mapper = vtk.vtkPolyDataMapper()  # 映射器
        mapper.SetInputConnection(sphere.GetOutputPort())  # 连接映射
        actor = vtk.vtkActor()  # actor
        actor.SetMapper(mapper)  # 映射球体
        return actor

    '''修改图形的分辨率'''
    @classmethod
    def change_image_for_px(cls, self):
        self.xyz_size_copy = [self.x_px, self.y_px, self.z_px]  # 获取修改前分辨率
        self.x_px = float(self.x_px_spinbox.value())
        self.y_px = float(self.y_px_spinbox.value())
        self.z_px = float(self.z_px_spinbox.value())

        # 更新标记位置
        self.update_sphere_size_position()
        # 修改图像尺寸
        # self.dataImporter.SetDataSpacing(self.x_px, self.y_px, self.z_px)  # Spacing
        # self.dataImporter.Update()
        self.dataImporter_outline.SetDataSpacing(self.x_px, self.y_px, self.z_px)  # Spacing
        self.dataImporter_outline.Update()
        self.outline.Update()
        self.box_dataImporter.SetDataSpacing(self.x_px, self.y_px, self.z_px)  # Spacing
        self.box_dataImporter.Update()
        self.box_outline.Update()
        self.box_outline_copy.Update()
        # 更新图像
        self.create_new_image()

    '''灰度变化'''
    @classmethod
    def gray_change(cls, index, self):
        if self.dataImporter_index:
            if index == 1:
                if self.gray_min == self.range_min:
                    return
                else:
                    # print("灰度减少")
                    gray_redun = int(self.gray_max - self.gray_min)
                    move_num = math.ceil(gray_redun / 4)
                    if self.gray_min < self.range_min + move_num:
                        self.gray_min = self.range_min
                        self.gray_max = self.gray_min + gray_redun
                    else:
                        self.gray_min -= move_num
                        self.gray_max -= move_num
            if index == 2:
                if self.gray_max == self.range_max:
                    return
                else:
                    # print("灰度增加")
                    gray_redun = int(self.gray_max - self.gray_min)
                    move_num = math.ceil(gray_redun / 4)
                    if self.gray_max > self.range_max - move_num:
                        self.gray_max = self.range_max
                        self.gray_min = self.gray_max - gray_redun
                    else:
                        self.gray_min += move_num
                        self.gray_max += move_num
            if index == 3:
                self.gray_min = self.min_auto_gray
                # if self.gray_min < self.range_min:
                #     self.gray_min = self.range_min
                self.gray_max = self.max_auto_gray
            # self.grayWidget.setLeftValue(self.gray_min)
            # self.grayWidget.setRightValue(self.gray_max)
            self.gray_min_num.setValue(self.gray_min)
            self.gray_max_num.setValue(self.gray_max)
            self.grayWidget.setLeftValue(self.gray_min)
            self.grayWidget.setRightValue(self.gray_max)
