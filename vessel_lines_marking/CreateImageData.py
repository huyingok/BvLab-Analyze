# -*- coding: utf-8 -*-
import math
import vtkmodules.all as vtk
import numpy as np


class CreateImageData:
    def __init__(self, image_info, render_mode_text):
        self.image_info = image_info
        self.render_mode_text = render_mode_text

    '''创建图形'''

    def create_img3d(self, x_px=1.0, y_px=1.0, z_px=1.0):  # 创建图
        data_matrix = self.image_info[0]  # 复制获取文件数据
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
        outline_actor.GetProperty().SetColor(1, 1, 1)
        outline_actor.GetProperty().SetLineWidth(2)

        return data_importer, outline, outline_actor

    '''更新图形颜色和灰度'''

    @classmethod
    def update_color_alpha(cls, alphaChannelFunc, colorFunc, gray_min, gray_max, r, g, b):
        alphaChannelFunc.RemoveAllPoints()
        alphaChannelFunc.AddPoint(gray_min, 0)
        alphaChannelFunc.AddPoint(gray_max, 1)
        colorFunc.RemoveAllPoints()
        colorFunc.AddRGBPoint(gray_min, 0, 0, 0)
        colorFunc.AddRGBPoint(gray_max, r, g, b)

    """创建点对象"""

    @classmethod
    def create_point_actor(cls, pos, size):
        sphere = vtk.vtkPointSource()  # 点绘制
        sphere.SetCenter(pos)  # 设置中心位置
        sphere.SetNumberOfPoints(1)  # 设置要生成的点的数量
        sphere.SetRadius(0)  # 点云数据范围半径
        sphere.Update()  # 数据更新
        mapper = vtk.vtkPolyDataMapper()  # 数据映射器
        mapper.SetInputConnection(sphere.GetOutputPort())  # 连接数据
        actor = vtk.vtkActor()  # 创建点actor
        actor.SetMapper(mapper)  # 点连接映射器
        # actor.GetProperty().SetColor(0, 0, 1)  # 颜色
        actor.GetProperty().SetPointSize(size)  # 点大小
        # actor.point = pos
        return actor

    """创建线对象"""

    @classmethod
    def create_line_actor(cls, pos1, pos2):
        line_glyph = vtk.vtkLineSource()  # 线绘制
        line_glyph.SetPoint1(pos1)  # 通过两点确定直线
        line_glyph.SetPoint2(pos2)
        line_mapper = vtk.vtkPolyDataMapper()  # 数据映射器
        line_mapper.SetInputConnection(line_glyph.GetOutputPort())  # 连接数据
        actor = vtk.vtkActor()  # 创建线actor
        actor.SetMapper(line_mapper)  # 线连接映射器
        # actor.GetProperty().SetColor(0, 0, 1)  # 颜色
        actor.GetProperty().SetLineWidth(2)  # 线宽
        # 获取端点
        # list(line_glyph.GetPoint1())
        # list(line_glyph.GetPoint2())
        # actor.point1 = pos1
        # actor.point2 = pos2
        return actor

    """创建点线对象"""

    @classmethod
    def create_PolyData_actor(cls, points, lines, vertices, self):
        # 创建 PolyData 并设置点和线
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        polydata.SetVerts(vertices)

        # # 创建 Mapper
        # mapper = vtk.vtkPolyDataMapper()  # 数据映射器
        # mapper.SetInputData(polydata)
        # # 创建 Actor
        # actor = vtk.vtkActor()  # 创建线actor
        # actor.SetMapper(mapper)  # 线连接映射器
        # return actor

        # 假设你已经有了 polydata、points、lines、vertices
        # 1. 定义裁剪框（世界坐标）
        box = vtk.vtkBox()
        xmin, ymin, zmin = 0, 0, 0
        xmax, ymax, zmax = self.img3d.shape[::-1]
        if self.volume_mark is not None:
            if self.volume_mark.GetVisibility():  # 如果切取图像显示
                xmin, ymin, zmin = self.img3d_mark_info[0]
                xmax, ymax, zmax = self.img3d_mark_info[1]
        box.SetBounds(xmin, xmax, ymin, ymax, zmin, zmax)  # 世界坐标
        # 2. Cropping polydata
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(polydata)
        clipper.SetClipFunction(box)
        clipper.InsideOutOn()  # 保留盒内；想保留盒外就 Off
        clipper.Update()

        # 3. 创建 mapper → actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(clipper.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        return actor

    '''灰度变化'''

    @classmethod
    def gray_change(cls, index, self):
        if self.dataImporter_index:
            if index == 1:
                if self.gray_min == self.range_min:
                    return
                else:
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
            self.gray_min_num.setValue(self.gray_min)
            self.gray_max_num.setValue(self.gray_max)
            self.grayWidget.setLeftValue(self.gray_min)
            self.grayWidget.setRightValue(self.gray_max)
