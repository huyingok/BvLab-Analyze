# -*- coding: utf-8 -*-
import traceback
import numpy as np
import sys
import random
from PyQt5.QtGui import QPainter, QColor, QFont
from PyQt5.QtCore import Qt, QSize
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox, QSplitter
from cell_points_marking.CreateImageData import CreateImageData
from cell_points_marking.CreateDialog import CreateDialog
from cell_points_marking.SetStepRedun import SetStepRedun
from cell_points_marking.ReadPath import ReadPath
from cell_points_marking.SetConnect import SetConnect
from cell_points_marking.MarkingPointsFun import MarkingPointsFun
from cell_points_marking.BoxWidgets import BoxWidgets
from cell_points_marking.MarkSphereProcess import MarkSphereProcess
from cell_points_marking.GrayAdjustWidget import GrayAdjustWidget
from cell_points_marking.CustomMainWindow import MyWidget
from cell_points_marking.InitializeInfo import InitializeInfo
from cell_points_marking.ControlStyle import (messagebox_style, image_button_show_style, image_button_hide_style, sphere_button_show_style,
                                              openColorDialog_button_style, MarkColorDialog_button_style)
from BVExample.BVMoudle.BVMoudle import BVReader


class CellViewWidget(MyWidget, CreateDialog):
    '''界面可视化'''
    def __init__(self, parent=None):
        super(CellViewWidget, self).__init__(parent)
        self.CreateImageData = CreateImageData  # 创建三维可视图
        self.SetStepRedun = SetStepRedun  # 三维可视图设置步长和冗余
        self.ReadPath = ReadPath  # 读取文件
        self.SetConnect = SetConnect  # 信号和快捷方式
        self.MarkingPointsFun = MarkingPointsFun  # 标记功能，Point/球的生成
        self.BoxWidgets = BoxWidgets  # box框
        self.MarkSphereProcess = MarkSphereProcess  # 对生成的球体操作

        # self.setStyleSheet(menu_style)
        # 取消右键菜单
        self.setContextMenuPolicy(Qt.NoContextMenu)
        # 自定义菜单栏
        # self.SetConnect(self.title_bar).set_title_menubar_connect(self)
        """初始界面设置"""
        self.action_widgets()
        # 图像列表信号
        self.SetConnect.img_list_connect(self)
        # 创建路径选择窗口
        self.create_path_select_dialog()
        self.SetConnect.select_path_connect(self)  # 路径选择信号
        # 筛选保存路径窗口
        self.create_filter_dialog()
        self.SetConnect.filter_connect(self)  # 筛选保存路径信号
        # 工具栏操作
        self.SetConnect.tool_button_clicked(self)  # 工具栏按钮信号
        # 快捷方式操作
        self.SetConnect.short_cut_connect(self)  # 快捷方式信号
        # 浮动窗口位置
        # self.addDockWidget(Qt.LeftDockWidgetArea, self.images_dockWidget)  # 浮动窗口靠左
        # self.addDockWidget(Qt.LeftDockWidgetArea, self.fun_dockWidget)  # 浮动窗口靠左
        self.Revising_verticalLayout.addWidget(self.images_dockWidget)
        self.Revising_verticalLayout.addWidget(self.fun_dockWidget)
        self.RevisingLayer_verticalLayout.addWidget(self.layer_dockWidget)
        # 设置拉伸
        """QSplitter 可以实现可拉伸的功能"""
        splitter_horizontal = QSplitter(Qt.Horizontal)
        splitter_horizontal.setHandleWidth(0)
        self.Revising_horizontalLayout.addWidget(splitter_horizontal)
        splitter_horizontal.addWidget(self.RevisingWidget)
        splitter_horizontal.addWidget(self.RevisingFrame)
        splitter_horizontal.addWidget(self.RevisingLayerFrame)
        # Prevent child windows from being completely hidden
        splitter_horizontal.setChildrenCollapsible(False)

        splitter_vertical = QSplitter(Qt.Vertical)
        splitter_vertical.setHandleWidth(0)
        self.Revising_verticalLayout.addWidget(splitter_vertical)
        splitter_vertical.addWidget(self.images_dockWidget)
        splitter_vertical.addWidget(self.fun_dockWidget)
        splitter_vertical.setChildrenCollapsible(False)
        # 设置不可关闭和不可折叠
        self.images_dockWidget.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.fun_dockWidget.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.layer_dockWidget.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        # 设置大小化
        # self.images_dockWidget.setFeatures(QtWidgets.QDockWidget.DockWidgetFloatable)
        # self.fun_dockWidget.setFeatures(QtWidgets.QDockWidget.DockWidgetFloatable)
        # self.layer_dockWidget.setFeatures(QtWidgets.QDockWidget.DockWidgetFloatable)
        # 初始化vtk视图区域
        self.create_vtk_view()
        # 设置坐标系函数
        self.AxesWidgt()
        self.images_dockWidget.installEventFilter(self)
        self.fun_dockWidget.installEventFilter(self)
        self.layer_dockWidget.installEventFilter(self)

        self.bvReader = BVReader()

    '''界面居中'''
    def center(self):
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    '''布局修饰'''
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QColor(70, 80, 100, 255))  # 设置矩形颜色和透明度
        painter.drawRect(-50, -50, self.width() + 50, self.height() + 50)  # 绘制矩形
        # painter.setOpacity(0.05)
        # pixmap = QPixmap(r".\icon_image\vstar.png")
        # painter.drawPixmap(self.rect(), pixmap)

    '''清除界面'''
    def clear_view(self):
        if self.dataImporter_index:  # 存在可标签数据,Update
            info = self.mess_set("Clear the interface?", 'Prompt', 2)
            if info == QtWidgets.QMessageBox.Yes:
                self.image_lineEdit.clear()
                self.swc_lineEdit.clear()

                self.fun_stacked_widget.removeWidget(self.widgetFun)  # 更新操作栏
                self.layer_stacked_widget.removeWidget(self.layer_cut_widget)

                self.widgetFun.deleteLater()
                self.layer_cut_widget.deleteLater()
                self.fun_stacked_widget.addWidget(self.fun_widget)
                self.layer_stacked_widget.addWidget(self.layer_widget)

                self.checkBox_zero.setChecked(False)

                if self.boxWidget:
                    self.boxWidget.Off()
                    del self.boxWidget

                del self.img3d  # Image
                del self.dataImporter  # 图像数据
                del self.box_dataImporter  # 图像块数据
                del self.dataImporter_outline  # 包围盒数据
                del self.outline  # 包围盒数据

                del self.volume  # Imageactor
                del self.box_volume  # 图像块actor
                del self.outlineActor

                InitializeInfo.__init__(self)
                self.img_list.clear()
                self.ren.RemoveAllViewProps()  # 去除所有渲染对象
                self.renWin.Render()
                self.dataImporter_index = 0
                self.set_button(self.start_mark_button, "Revision Mode", "marking.png")
                self.filter_button.setVisible(True)
                self.filter_finish_button.setVisible(False)

    '''设置相机朝上方向'''
    def set_camera_view_up_y(self):
        if self.dataImporter_index:
            dz = self.max_shape * 4
            d_near, d_far = self.camera.GetClippingRange()
            d = d_far - d_near
            d_min = self.max_shape
            if d_near >= dz - d_min or dz + d_min >= d_far:
                dz = d_near + d / 2
            else:
                self.camera.SetClippingRange(dz - d_min, dz + d_min)
                # print("Update")
            self.camera.SetViewUp(0, 1, 0)
            self.camera.SetPosition(self.img3d.shape[2] / 2, self.img3d.shape[1] / 2, dz)
            self.camera.SetFocalPoint(self.img3d.shape[2] / 2, self.img3d.shape[1] / 2, self.img3d.shape[0] / 2)
            # print(self.camera.GetFocalPoint())
            # print(self.camera.GetPosition())
            self.renWin.Render()

    def set_camera_view_up_x(self):
        if self.dataImporter_index:
            dy = self.max_shape * 4
            d_near, d_far = self.camera.GetClippingRange()
            d = d_far - d_near
            d_min = self.max_shape
            if d_near >= dy - d_min or dy + d_min >= d_far:
                dy = d_near + d / 2
            else:
                self.camera.SetClippingRange(dy - d_min, dy + d_min)
            self.camera.SetViewUp(1, 0, 0)
            self.camera.SetPosition(self.img3d.shape[2] / 2, dy, self.img3d.shape[0] / 2)
            self.camera.SetFocalPoint(self.img3d.shape[2] / 2, self.img3d.shape[1] / 2, self.img3d.shape[0] / 2)
            self.renWin.Render()

    def set_camera_view_up_z(self):
        if self.dataImporter_index:
            dx = self.max_shape * 4
            d_near, d_far = self.camera.GetClippingRange()
            d = d_far - d_near
            d_min = self.max_shape
            if d_near >= dx - d_min or dx + d_min >= d_far:
                dx = d_near + d / 2
            else:
                self.camera.SetClippingRange(dx - d_min, dx + d_min)
            self.camera.SetPosition(dx, self.img3d.shape[1] / 2, self.img3d.shape[0] / 2)
            self.camera.SetFocalPoint(self.img3d.shape[2] / 2, self.img3d.shape[1] / 2, self.img3d.shape[0] / 2)
            self.camera.SetViewUp(0, 0, 1)
            self.renWin.Render()

    '''删除多余的点线'''

    def judge_actor(self):
        try:
            self.ren.RemoveActor(self.one_point_actor)  # 删除点
            self.ren.RemoveActor(self.one_line_actor)  # 删除直线
            self.point_line_count = True
            self.one_sphere_count = False
            self.renWin.Render()
        except AttributeError:
            pass

    '''切换模式'''

    def change_function(self):
        self.MarkingPointsFun.change_mark_function(self)

    '''切换渲染模式'''

    def _on_render_mode_changed(self, render_mode):
        if self.dataImporter_index:
            print(render_mode)
            if self.volume:
                # 如果已有数据，直接更新mapper和property，不重建volume
                mapper = self.volume.GetMapper()
                # 直接修改mapper的混合模式
                if render_mode == "MIP":
                    mapper.SetBlendModeToMaximumIntensity()
                else:  # COMPOSITE
                    mapper.SetBlendModeToComposite()
            if self.box_volume:
                # 如果已有数据，直接更新mapper和property，不重建volume
                mapper = self.box_volume.GetMapper()
                if render_mode == "MIP":
                    mapper.SetBlendModeToMaximumIntensity()
                else:  # COMPOSITE
                    mapper.SetBlendModeToComposite()
            self.renWin.Render()
            self.render_mode_text = render_mode

    '''切换标记渲染'''
    def _on_mark_render_changed(self, _mode):
        if self.dataImporter_index:
            # print(_mode)
            self.MarkSphereProcess.update_sphere_color(self, _mode)
            self.mark_render_text = _mode

    '''创建球体'''

    def create_sphere(self, pos):
        self.MarkSphereProcess._create_sphere(pos, self)

    '''直线和图像数据的交点集'''

    def collect_points(self, intersection1, intersection2):
        points, count = self.MarkingPointsFun._collect_points(intersection1, intersection2, self)
        return points, count

    '''Max值定点'''

    # def Max_onRightButtonPressEvent(self, obj, event):
    #     self.MarkingPointsFun.Max_onRightButtonPressEvent(self)

    '''vtk鼠标右击事件'''

    # def onRightButtonPressEvent(self, obj, event):
    #     self.MarkingPointsFun.onRightButtonPressEvent(self)

    '''标点模式选择'''
    def onSelectMarkingFun(self):
        if self.is_marking_button:
            if self.function_index == 0:
                self.MarkingPointsFun.onLeftButtonPressEvent(self)
            else:
                self.MarkingPointsFun.Max_onLeftButtonPressEvent(self)
        else:
            self.select_actor()

    '''读取文件路径'''

    def read_file_path(self):
        self.ReadPath._read_file_path(self)

    '''上下切换图'''

    def on_item_selection(self, now_item, pre_item):
        # self.inherit_pos = self.camera.GetPosition()
        # self.inherit_focal = self.camera.GetFocalPoint()
        # self.inherit_range = self.camera.GetClippingRange()
        if self.selection:
            # print("选中列表单元")
            # print("当前路径为:", self.swc_file_list[self.img_list.row(now_item)])
            if pre_item:  # 获取之前选择项的序号
                self.image_file_list_index = self.img_list.row(pre_item)
                name = pre_item.text().split("   ")[-1]
                if self.filter_res["filenames"].get(name) is None:
                    pre_item.setForeground(QColor("green"))  # 字体颜色为绿色

            save_swc_path = self.swc_file_list[self.image_file_list_index]
            self.save_txt_swc(save_swc_path)

            self.image_file_list_index = self.img_list.currentRow()
            self.reOpen = False
            self.Visualization_tif_file()

    '''保存坐标为txt'''

    def save_position(self):
        self.ReadPath.save_sphere_pos(self)

    """设置用户功能"""

    def set_user_function(self):
        self.ShortCutDialog.stackedWidget.setCurrentIndex(0)
        self.ShortCutDialog.listWidget.setCurrentRow(0)
        self.ShortCutDialog.show()
        # print("设置用户功能")

    """显示筛选窗口"""

    def filter_dialog_show(self):
        if self.dataImporter_index:
            self.filter_dialog.show()

    """完成筛选"""

    def filter_finish(self):
        self.filter_button.setVisible(True)
        self.filter_finish_button.setVisible(False)
        self.ReadPath.treat_filter_finish(self)

    """开始筛选"""

    def filter_start(self):
        if len(self.save_lineEdit.text()):
            self.filter_button.setVisible(False)
            self.filter_finish_button.setVisible(True)
            self.filter_dialog.hide()
        else:
            self.mess_set("There is an empty path!", 'Prompt', 1)

    """取消筛选"""

    def filter_cancel(self):
        self.filter_dialog.hide()

    """选择筛选保存文件夹"""

    def filter_save_path_open(self):
        self.ReadPath.select_filter_path(self.save_lineEdit, self)

    '''Saveswc和txt'''

    def save_txt_swc(self, save_swc_path):
        self.ReadPath.save_pos_swc(save_swc_path, self)

    '''VisualizationtifFile'''

    def Visualization_tif_file(self):
        # start_time = time.time()
        self.tif_to_view(self)
        # print(time.time() - start_time)

    '''DataImporter_create,tifRendering'''

    def DataImporter_create(self):
        img = np.zeros(self.img3d.shape, dtype=np.uint8)
        image_info = [img, self.x_px, self.y_px, self.z_px]
        # 对类进行实例化，并调用方法（Function）
        (self.dataImporter, self.volume, self.opacityTransferFunction,
         self.colorTransferFunction) = self.CreateImageData(image_info, self.render_mode_text).create_img3d()

    '''Outline_create,包围盒'''

    def Outline_create(self):
        (self.dataImporter_outline, self.outline,
         self.outlineActor) = self.CreateImageData.create_outline(self.dataImporter)

    '''选择区域显示隐藏'''

    def outline_show_hide(self):
        if self.dataImporter_index and self.box_volume:
            if self.outline_show_hide_count:
                # Hide
                self.box_volume.VisibilityOff()
                self.box_outlineActor.VisibilityOff()
                self.outline_show_hide_count = 0
                self.image_button.setStyleSheet(image_button_hide_style)
            else:
                # Display
                self.box_volume.VisibilityOn()
                self.box_outlineActor.VisibilityOn()
                self.outline_show_hide_count = 1
                self.image_button.setStyleSheet(image_button_show_style)
            self.renWin.Render()

    '''图文件显示隐藏'''

    def data_volume_show_hide(self):
        if self.dataImporter_index:
            # if self.data_volume_show_hide_count:  # 图文件显示
            #     self.volume.VisibilityOff()
            #     self.data_volume_show_hide_count = 0
            # else:
            #     self.volume.VisibilityOn()
            #     self.data_volume_show_hide_count = 1
            self.renWin.Render()

    '''Modifystep'''

    def change_z_step(self):
        self.z_step = self.SetStepRedun.set_step(self.z_step, self.z_redun,
                                                 self.z_step_num, self.z_redun_num)
        self.z_redun_limit.setText(f"- {int(self.z_redun_num.maximum())}")
        self.change_z_step_redun()
        self.get_z_layers()

    '''Modifyredun'''

    def change_z_redun(self):
        self.z_redun = self.SetStepRedun.set_redun(self.z_step, self.z_redun,
                                                   self.z_step_num, self.z_redun_num, self.img3d.shape[0])
        self.z_step_limit.setText(f"{int(self.z_step_num.minimum())} -")
        self.change_z_step_redun()
        self.get_z_layers()

    '''获取z轴切块层数'''

    def get_z_layers(self):
        self.z_num = self.SetStepRedun.get_layers(self.z_step, self.z_redun, self.img3d.shape[0], 2)
        if self.z_list:
            self.update_z_list()

    '''修改步长和冗余'''

    def change_z_step_redun(self):
        (self.box_size, self.origin, self.z_up_over) = self.SetStepRedun.update_box_size(
            [self.x_step, self.y_step, self.z_step], self.img3d.shape[0], self.origin, 2)
        if not self.xyz_cut_index:
            self.create_new_image()

    '''包围盒上移'''

    def box_up(self):
        # print("上移")
        # print("self.z_down_over:", self.z_down_over)
        # print("self.z_up_over:", self.z_up_over)
        if self.dataImporter_index and self.box_volume:
            if not self.z_up_over:
                (self.box_size, self.origin, self.z_up_over, self.z_down_over) = self.SetStepRedun.box_to_up(
                    self.z_step, self.z_redun, self.box_size, self.origin, self.img3d.shape[0], self.z_up_over, 2)
                self.create_new_image()
                save_swc_path = self.swc_file_list[self.img_list.currentRow()]  # 保存标记
                self.save_txt_swc(save_swc_path)

    '''包围盒下移'''

    def box_down(self):
        # print("下移")
        # print("self.z_down_over:", self.z_down_over)
        # print("self.z_up_over:", self.z_up_over)
        if self.dataImporter_index and self.box_volume:
            if not self.z_down_over:
                (self.box_size, self.origin, self.z_up_over, self.z_down_over) = self.SetStepRedun.box_to_down(
                    self.z_step, self.z_redun, self.box_size, self.origin, self.z_down_over, 2)

                self.create_new_image()
                save_swc_path = self.swc_file_list[self.img_list.currentRow()]  # 保存标记
                self.save_txt_swc(save_swc_path)

    '''感兴趣区域更新'''

    def create_new_image(self):
        self.CreateImageData.create_box_image(self)

    '''所有自定义功能'''

    def object_message(self):
        # 更新图像灰度和颜色
        self.CreateImageData.update_color_alpha(self.opacityTransferFunction, self.colorTransferFunction,
                                                self.gray_min, self.gray_max, self.r, self.g, self.b)

        # 获取x轴层数
        x_num = 1
        s = self.x_step
        while s < self.img3d.shape[2]:
            x_num += 1
            s = s + self.x_step - self.x_redun
        self.x_num = x_num

        # 获取y轴层数
        y_num = 1
        s = self.y_step
        while s < self.img3d.shape[1]:
            y_num += 1
            s = s + self.y_step - self.y_redun
        self.y_num = y_num

        # 获取z轴层数
        z_num = 1
        s = self.z_step
        while s < self.img3d.shape[0]:
            z_num += 1
            s = s + self.z_step - self.z_redun
        self.z_num = z_num
        if self.reOpen:
            # 功能栏
            self.fun_widgets()
            # xyz切块界面
            self.layer_widgets()
            # 功能栏信号
            self.SetConnect.fun_widget_info_connect(self)
            # 切块列表信号
            self.SetConnect.layer_widget_info_connect(self)
            # 灰度调节
            self.grayWidget = GrayAdjustWidget(self.graphframe)
            self.grayWidget.leftValueChanged.connect(self.grayTransCallback)  # 设置灰度图槽函数
            self.grayWidget.rightValueChanged.connect(self.grayTransCallback)
            self.graphWidget_form.addRow(self.gray_label, self.grayWidget)
        else:
            # Image、标记显示和隐藏
            self.image_button.setStyleSheet(image_button_show_style)
            self.sphere_button.setStyleSheet(sphere_button_show_style)
            # 标记按钮颜色
            # self.MarkColorDialog_button.setStyleSheet(MarkColorDialog_button_style)
            # 颜色按钮
            # self.openColorDialog_button.setStyleSheet(openColorDialog_button_style)

        # 图像是否重新加载或切换
        self.isReload = True

        # 更新切块选择
        self.update_z_list()
        self.update_xy_table()
        # 图像灰度信息
        gray_message = self.gray_widget_message()
        # 灰度调节
        self.grayWidget.setLeftValue(self.gray_min)
        self.grayWidget.setRightValue(self.gray_max)
        self.grayWidget.setHist(gray_message)

        # 更新切块列表信息
        self.update_layer_info()
        # 切块列表窗口加入堆叠窗口
        self.layer_stacked_widget.addWidget(self.layer_cut_widget)

        # 更新功能框信息
        self.update_fun_box_info()
        # 功能栏窗口加入堆叠窗口
        self.fun_stacked_widget.addWidget(self.widgetFun)

    """更新切块列表信息"""

    def update_layer_info(self):
        # print(self.z_step)
        # 禁止信号传递
        self.z_step_num.blockSignals(True)
        self.z_redun_num.blockSignals(True)
        self.x_step_num.blockSignals(True)
        self.x_redun_num.blockSignals(True)
        self.y_step_num.blockSignals(True)
        self.y_redun_num.blockSignals(True)

        z_down = int(self.img3d.shape[0] / 10) + 1
        z_up = int(self.img3d.shape[0]) - 1
        if not self.isReload:
            z_down = max(int(self.z_redun_num.value()) + 1, z_down)
            z_up = min(int(self.z_step_num.value()) - 1, z_up)
        self.z_step_num.setRange(z_down, int(self.img3d.shape[0]))
        self.z_redun_num.setRange(0, z_up)

        # if not self.sr_inherit_index:  # 是否继承步长冗余
        #     self.z_step_num.setValue(int(self.img3d.shape[0]))  # 初始值
        #     self.z_redun_num.setValue(0)
        #     self.sr_inherit_index = True
        #     self.z_step = self.z_step_num.value()
        #     self.z_redun = self.z_redun_num.value()
        # else:
        #     self.z_step_num.setValue(int(self.z_sr_copy[0]))  # 初始值
        #     self.z_redun_num.setValue(int(self.z_sr_copy[1]))

        x_down = int(self.img3d.shape[2] / 10) + 1
        x_up = int(self.img3d.shape[2]) - 1
        y_down = int(self.img3d.shape[1] / 10) + 1
        y_up = int(self.img3d.shape[1]) - 1

        if not self.isReload:
            x_down = max(int(self.x_redun_num.value()) + 1, x_down)
            x_up = min(int(self.x_step_num.value()) - 1, x_up)
            y_down = max(int(self.y_redun_num.value()) + 1, y_down)
            y_up = min(int(self.y_step_num.value()) - 1, y_up)
        self.x_step_num.setRange(x_down, int(self.img3d.shape[2]))
        self.x_redun_num.setRange(0, x_up)
        self.y_step_num.setRange(y_down, int(self.img3d.shape[1]))
        self.y_redun_num.setRange(0, y_up)

        self.z_step_limit.setText(f"{z_down} -")
        self.x_step_limit.setText(f"{x_down} -")
        self.y_step_limit.setText(f"{y_down} -")

        self.z_redun_limit.setText(f"- {z_up}")
        self.x_redun_limit.setText(f"- {x_up}")
        self.y_redun_limit.setText(f"- {y_up}")

        # print(self.z_step)
        self.z_step_num.setValue(self.z_step)  # 初始值
        self.z_redun_num.setValue(self.z_redun)
        self.x_step_num.setValue(self.x_step)  # 初始值
        self.x_redun_num.setValue(self.x_redun)
        self.y_step_num.setValue(self.y_step)  # 初始值
        self.y_redun_num.setValue(self.y_redun)

        # 恢复信号传递
        self.z_step_num.blockSignals(False)
        self.z_redun_num.blockSignals(False)
        self.x_step_num.blockSignals(False)
        self.x_redun_num.blockSignals(False)
        self.y_step_num.blockSignals(False)
        self.y_redun_num.blockSignals(False)

    """更新功能框信息"""

    def update_fun_box_info(self):
        # 标记功能切换
        if self.is_marking_button:
            self.fun_label = self.function_change_label.text().split('(')[0]
            self.fun_label = f"{self.fun_label}(Revision Mode)"
        if self.reOpen:
            self.function_index = 1
        self.function_change_label.setText('%s' % self.fun_label)
        # Information
        self.info_num_label.setText('%d × %d × %d %s' %
                                    (self.img3d.shape[2], self.img3d.shape[1],
                                     self.img3d.shape[0], str(self.img3d.dtype)))
        # 渲染模式切换
        # 初始化渲染模式选项（与RenderMode枚举一致）
        self.render_mode_combo.blockSignals(True)
        self.render_mode_combo.clear()
        self.render_mode_combo.addItems(["MIP", "Composite"])
        self.render_mode_combo.setCurrentText(self.render_mode_text)
        self.render_mode_combo.blockSignals(False)
        # 标记渲染
        self.mark_render_combo.blockSignals(True)
        self.mark_render_combo.clear()
        self.mark_render_combo.addItems(["Single-color", "Multi-color"])
        self.mark_render_combo.setCurrentText(self.mark_render_text)
        self.mark_render_combo.blockSignals(False)
        # 标记颜色按钮
        rgb = (self.mark_color[0] * 255, self.mark_color[1] * 255, self.mark_color[2] * 255)
        self.MarkColorDialog_button.setStyleSheet("")  # 清除旧的样式表
        style_sheet = (
            f'QPushButton {{ background-color: rgb{rgb}; color: #19232d; '
            f'border-radius: 0px; width: 20px; height: 20px;}}'
            'QPushButton:hover { background-color: #379eff;}'
        )
        self.MarkColorDialog_button.setStyleSheet(style_sheet)
        # 筛选计数
        if not self.dataImporter_index:
            self.filter_nums_label.setText("0")
        # Resolution
        self.x_px_spinbox.setValue(self.x_px)  # 初始值
        self.y_px_spinbox.setValue(self.y_px)
        self.z_px_spinbox.setValue(self.z_px)
        # 自动调节灰度
        self.gray_auto_change_button.setText(f"{self.min_auto_gray} / {self.max_auto_gray}")
        # 颜色
        rgb = (self.r * 255, self.g * 255, self.b * 255)
        self.openColorDialog_button.setStyleSheet("")  # 清除旧的样式表
        style_sheet = (
            f'QPushButton {{ background-color: rgb{rgb}; color: #19232d; '
            f'border-radius: 0px; width: 20px; height: 20px;}}'
            'QPushButton:hover { background-color: #379eff;}'
        )
        self.openColorDialog_button.setStyleSheet(style_sheet)
        # 标记大小
        if not self.dataImporter_index:
            self.sphere_size_slider.setRange(1, 30)
            self.sphere_size_slider.setValue(max(int(self.max_shape / 30), 1))  # Default value
        self.sphere_size_label.setText(f'Sphere radius = {self.sphere_size}')
        self.sphere_size_slider.setValue(int(self.sphere_size))
        # 灰度调节
        self.gray_min_num.setValue(self.gray_min)
        self.gray_max_num.setValue(self.gray_max)

    '''读取txtFile,读取选择文件的txtFile,渲染标签球体'''

    def read_txt_file(self, index):
        self.ReadPath._read_txt_file(index, self)

    '''标记显示和隐藏'''

    def _Sphere_show_hide(self):
        self.MarkSphereProcess.sphere_show_hide(self)

    """路径选择"""

    def path_radioButton_clicked(self):
        self.path_widget.setVisible(True)
        self.config_widget.setVisible(False)  # 隐藏窗口
        self.name_widget.setVisible(False)

    """配置文件选择"""

    def config_radioButton_clicked(self):
        self.path_widget.setVisible(False)
        self.config_widget.setVisible(True)
        self.name_widget.setVisible(False)

    """预测配置文件"""

    def name_radioButton_clicked(self):
        self.path_widget.setVisible(False)
        self.config_widget.setVisible(False)
        self.name_widget.setVisible(True)

    '''open file,选择文件路径,路径选择框,是否保存当前文件的txt和swc'''

    def select_file_path_show(self):
        self.select_path_dialog.show()

    '''路径选择提示,路径选择框,取消按钮'''

    def cancel_path_select(self):
        self.select_path_dialog.hide()

    '''路径选择框确定按钮,image路径问题,获取文件名列表,路径列表'''

    def sure_path_select(self):
        if self.config_widget.isVisible():
            if not self.ReadPath.read_config_question(self):
                return
        elif self.name_widget.isVisible():
            if not self.ReadPath.read_name_question(self):
                return
        self.ReadPath.read_path_question(self)

    '''imagePath,SelecttifPath'''

    def image_path_open(self):
        # 获取文件路径
        self.ReadPath.select_image_path(self.image_lineEdit, self)

    '''swcPath,SelectswcPath'''

    def swc_path_open(self):
        # 获取文件夹路径
        self.swc_path = self.ReadPath.select_swc_path(self.swc_lineEdit, self)

    '''config，路径选择'''

    def config_path_open(self):
        # 选择文件路径
        self.ReadPath.select_config_path(self.config_lineEdit, self)

    '''预测配置文件，路径选择'''

    def name_path_open(self):
        self.ReadPath.select_name_path(self.name_lineEdit, self)

    '''修改图像分辨率'''

    def update_image_px(self):
        self.CreateImageData.change_image_for_px(self)

    '''图列表浮动窗口显示和隐藏'''

    def images_list_widget(self):
        if self.images_dockWidget.isHidden():
            self.images_dockWidget.show()
        else:
            self.images_dockWidget.hide()

    '''功能浮动窗口显示和隐藏'''

    def funs_widget_show_hide(self):
        if self.fun_dockWidget.isHidden():
            self.fun_dockWidget.show()
        else:
            self.fun_dockWidget.hide()

    '''最小灰度值'''

    def textinput_gray_min(self):
        # 控制最小灰度
        self.gray_min = self.gray_min_num.value()
        self.grayWidget.setLeftValue(self.gray_min)

    '''最大灰度值'''

    def textinput_gray_max(self):
        # 控制最大灰度
        self.gray_max = self.gray_max_num.value()
        self.grayWidget.setRightValue(self.gray_max)

    '''固定灰度'''

    def sure_gray_auto(self):
        if self.dataImporter_index == 1:
            self.min_auto_gray = self.gray_min
            self.max_auto_gray = self.gray_max
            self.gray_auto_change_button.setText(f'{self.min_auto_gray} / {self.max_auto_gray}')

    '''自动更新灰度'''

    def gray_auto_change(self):
        self.CreateImageData.gray_change(3, self)

    '''标记颜色'''

    def MarkColorDialog(self):
        self.create_mark_color_dialog(self)

    '''修改颜色'''

    def change_mark_color(self):
        if self.mark_render_text == 'Single-color':
            self.MarkSphereProcess.update_sphere_color(self, self.mark_render_text)

    '''Color dialog'''

    def openColorDialog(self):
        self.create_color_dialog(self)

    '''修改颜色和透明度'''

    def change_color_opacity(self):
        # if self.volume:
        #     self.CreateImageData.update_color_alpha(self.opacityTransferFunction, self.colorTransferFunction,
        #                                             self.gray_min, self.gray_max, self.r, self.g, self.b)
        # print("当前最小灰度:", self.gray_min)
        # print("当前最大灰度:", self.gray_max)
        if self.box_volume:
            self.CreateImageData.update_color_alpha(self.alphaChannelFunc, self.colorFunc,
                                                    self.gray_min, self.gray_max, self.r, self.g, self.b)
        self.renWin.Render()

    '''选择对象'''

    def select_actor(self):
        if self.dataImporter_index:
            # Selected
            self.MarkSphereProcess.sphere_select_and_mark(self)

    '''删除对象'''

    def del_actor(self, obj, event):
        if self.dataImporter_index:
            # Selected
            self.MarkSphereProcess.sphere_select_and_mark(self)
            # Delete
            self.Actor_Delete()

    '''切换投影方式'''

    def toggle_projection_mode(self):
        if self.dataImporter_index:
            if self.camera:
                is_parallel = self.camera.GetParallelProjection()  # 默认是False
                self.camera.SetParallelProjection(not is_parallel)
                self.renWin.Render()
                if is_parallel:
                    # 切换到透视投影模式
                    self.set_button(self.projection_button, "Perspective Projection", "perspective_projection.png")
                else:
                    # 切换到正交投影模式
                    self.set_button(self.projection_button, "Rectangular Projection", "rectangular_projection.png")

    """开启修订模式"""

    def start_mark_button_clicked(self):
        if self.dataImporter_index:
            if self.is_marking_button:
                self.is_marking_button = False
                text = self.function_change_label.text().split('(')[0]
                self.function_change_label.setText(text)
                self.set_button(self.start_mark_button, "Revision Mode", "marking.png")
            else:
                self.is_marking_button = True
                text = self.function_change_label.text()
                self.function_change_label.setText(f"{text}(Revision Mode)")
                self.judge_actor()  # 删除多余点线
                self.set_button(self.start_mark_button, "Revision Mode", "is_marking.png")

    '''标记对象'''

    def mark_actor(self):
        if self.dataImporter_index:
            self.MarkSphereProcess.sphere_mark(self)

    '''球体删除'''

    def Actor_Delete(self):
        if self.dataImporter_index:
            self.MarkSphereProcess.delete_sphere(self)

    """清空标记"""

    def all_actor_delete(self):
        if self.dataImporter_index:
            self.MarkSphereProcess.delete_all_sphere(self)

    '''点击撤回'''

    def Point_Withdraw(self):
        if self.dataImporter_index:
            self.MarkSphereProcess.shpere_withdraw(self)

    '''Focus'''

    def camera_focal(self):
        if self.dataImporter_index and len(self.points_list):
            pos = self.Position[self.index]
            self.camera.SetFocalPoint(pos[0] * self.x_px, pos[1] * self.y_px, pos[2] * self.z_px)
            self.renWin.Render()

    '''球体半径'''

    def change_sphere_size(self):
        self.sphere_size = self.sphere_size_slider.value()
        self.sphere_size_label.setText(f'Sphere radius = {self.sphere_size}')
        self.update_sphere_size_position()

    '''刷新球体'''

    def update_sphere_size_position(self):
        self.MarkSphereProcess.update_sphere_position(self)

    '''灰度减少'''

    def gray_left(self):
        self.CreateImageData.gray_change(1, self)

    '''灰度增加'''

    def gray_right(self):
        self.CreateImageData.gray_change(2, self)

    '''图像切换'''

    def item_up(self):
        if self.dataImporter_index:
            if self.img_list.currentRow() == 0:
                return
            else:
                self.image_file_list_index = self.img_list.currentRow() - 1
                self.img_list.setCurrentRow(self.image_file_list_index)

    def item_down(self):
        if self.dataImporter_index:
            if self.img_list.currentRow() == self.img_list.count() - 1 or self.img_list.currentRow() == -1:
                return
            else:
                self.image_file_list_index = self.img_list.currentRow() + 1
                self.img_list.setCurrentRow(self.image_file_list_index)

    """筛选图像"""

    def filter_images(self):
        if self.dataImporter_index:
            if self.filter_finish_button.isVisible():
                item = self.img_list.currentItem()
                text = item.text()
                name = text.split("   ")[-1]
                if self.filter_res["filenames"].get(name) is None:
                    # print(f"Middle button clicked on item: {item.text()}")
                    self.filter_res["filenames"][name] = item
                    item.setForeground(QColor("yellow"))  # 字体颜色为黄色
                    if "√ " not in item.text():
                        item.setText("√ " + text)
                    self.filter_nums_label.setText(f"{len(self.filter_res['filenames'])}")
                else:
                    # print(f"Right button clicked on item: {item.text()}")
                    if "√ " in item.text():
                        text = text.replace("√ ", "")
                        item.setText(text)
                    self.filter_res["filenames"].pop(name, None)
                    self.filter_nums_label.setText(f"{len(self.filter_res['filenames'])}")

    '''z轴选择'''

    def change_z(self, now_item, pre_item):
        if self.z_list.count() > 1:
            if pre_item:
                pre_item.setForeground(QColor("green"))  # 字体颜色为绿色
            self.z_index = self.z_list.currentRow()
            dz = self.z_num - self.z_index - 1
            # print("z轴切换:", dz)
            z2 = (dz + 1) * self.z_step - dz * self.z_redun
            # print(self.img3d.shape[0])
            if z2 > self.img3d.shape[0]:
                z2 = self.img3d.shape[0]
                # self.z_down_over = 0
            z1 = z2 - self.z_step
            self.box_size[-2:] = z1, z2
            # print(self.box_size)
            self.origin[2] = z1
            self.create_new_image()

    '''Updatez轴列表'''

    def update_z_list(self):
        self.z_list.blockSignals(True)
        self.z_list.clear()
        for i in range(self.z_num):
            # check_box = QtWidgets.QCheckBox()  # 修饰框
            # check_box.setStyleSheet("QCheckBox::indicator{width:0px;height:0px;}")
            item = QtWidgets.QListWidgetItem()
            # lg = QLinearGradient(0, 0, 300, 0)
            # lg.setColorAt(0, QColor(25, 35, 45))
            # lg.setColorAt(1, QColor(90, 100, 120))
            # item.setBackground(QBrush(lg))
            # item.setBackground(QColor(70, 80, 100))  # 设置背景颜色
            item.setText(f"{self.z_num - 1 - i}")  # 列表项名字
            item.setFont(QFont("微软雅黑", 9))  # 字体
            item.setSizeHint(QSize(200, 18))  # 设置列表项尺寸
            self.z_list.addItem(item)  # 添加列表项
            # self.z_list.setItemWidget(item, check_box)  # 设置组合
            self.z_list.update()  # 更新列表项
        self.z_list.blockSignals(False)
        if not self.isReload:
            self.z_list.setCurrentRow(self.z_num - 1)

    '''xySelect'''

    def change_xy(self):
        row = self.table_widget.currentRow()  # 行标
        column = self.table_widget.currentColumn()  # 列标
        # row = self.tableview.currentIndex().row()  # 行标
        # column = self.tableview.currentIndex().column()  # 列标
        # print("xyToggle[x, y]:", [column, self.y_num - 1 - row])
        dx = column
        dy = self.y_num - 1 - row
        x2 = (dx + 1) * self.x_step - dx * self.x_redun
        y2 = (dy + 1) * self.y_step - dy * self.y_redun
        if x2 > self.img3d.shape[2]:
            x2 = self.img3d.shape[2]
        if y2 > self.img3d.shape[1]:
            y2 = self.img3d.shape[1]
        x1 = x2 - self.x_step
        y1 = y2 - self.y_step
        self.box_size[:4] = x1, x2, y1, y2
        # print(self.box_size)
        self.origin[:2] = x1, y1
        self.create_new_image()

    '''Updatexy轴列表'''

    def update_xy_table(self):
        self.table_widget.blockSignals(True)
        self.table_widget.setColumnCount(self.x_num)
        self.table_widget.setRowCount(self.y_num)
        # 设置序号样式
        # header = QtWidgets.QHeaderView(Qt.Horizontal)
        # header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)  # 自适应内容大小
        # header.setStyleSheet("QHeaderView::section { background-color: lightblue; color: red; font-size: 16px; }")
        for row in range(self.y_num):
            for column in range(self.x_num):
                item = QtWidgets.QTableWidgetItem(f'{column}-{self.y_num - 1 - row}')
                item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)  # 上下左右居中
                self.table_widget.setItem(row, column, item)
        self.table_widget.blockSignals(False)

    '''复原xy'''

    def recovery_xy(self):
        if self.dataImporter_index:
            if (self.x_step_num.value() == int(self.img3d.shape[2]) and
                    self.y_step_num.value() == int(self.img3d.shape[1])):
                return
            self.x_step_num.setValue(int(self.img3d.shape[2]))
            self.y_step_num.setValue(int(self.img3d.shape[1]))

            # self.change_x_step()
            # self.change_x_redun()
            # self.change_y_step()
            # self.change_y_redun()

    '''复原z'''

    def recovery_z(self):
        if self.dataImporter_index:
            if self.z_step_num.value() == int(self.img3d.shape[0]):
                return
            self.z_step_num.setValue(int(self.img3d.shape[0]))

            # self.change_z_step()
            # self.change_z_redun()

    '''去除边缘标记点'''

    def remove_edge_annotations(self):
        if self.dataImporter_index:
            rx = self.x_redun_num.value()
            ry = self.y_redun_num.value()
            rz = self.z_redun_num.value()
            shapes = self.img3d.shape  # zyx
            self.MarkSphereProcess.remove_edge_sphere(self, rx, ry, rz, shapes)

    '''x轴切块'''

    def change_x_step(self):
        self.x_step = self.SetStepRedun.set_step(self.x_step, self.x_redun,
                                                 self.x_step_num, self.x_redun_num)
        self.x_redun_limit.setText(f"- {int(self.x_redun_num.maximum())}")
        self.change_x_step_redun()
        self.get_x_layers()

    def change_x_redun(self):
        self.x_redun = self.SetStepRedun.set_redun(self.x_step, self.x_redun,
                                                   self.x_step_num, self.x_redun_num, self.img3d.shape[2])
        self.x_step_limit.setText(f"{int(self.x_step_num.minimum())} -")
        self.change_x_step_redun()
        self.get_x_layers()

    def get_x_layers(self):
        self.x_num = self.SetStepRedun.get_layers(self.x_step, self.x_redun, self.img3d.shape[2], 0)
        if self.table_widget:
            self.update_xy_table()

    def change_x_step_redun(self):
        (self.box_size, self.origin, self.x_up_over) = self.SetStepRedun.update_box_size(
            [self.x_step, self.y_step, self.z_step], self.img3d.shape[2], self.origin, 0)
        if not self.xyz_cut_index:
            self.create_new_image()

    '''x轴移动'''

    def box_x_to_add(self):
        if self.dataImporter_index and self.box_volume:
            if not self.x_up_over:
                (self.box_size, self.origin, self.x_up_over, self.x_down_over) = self.SetStepRedun.box_to_up(
                    self.x_step, self.x_redun, self.box_size, self.origin, self.img3d.shape[2], self.x_up_over, 0)
                self.create_new_image()
                save_swc_path = self.swc_file_list[self.img_list.currentRow()]  # 保存标记
                self.save_txt_swc(save_swc_path)

    def box_x_to_down(self):
        if self.dataImporter_index and self.box_volume:
            if not self.x_down_over:
                (self.box_size, self.origin, self.x_up_over, self.x_down_over) = self.SetStepRedun.box_to_down(
                    self.x_step, self.x_redun, self.box_size, self.origin, self.x_down_over, 0)
                self.create_new_image()
                save_swc_path = self.swc_file_list[self.img_list.currentRow()]  # 保存标记
                self.save_txt_swc(save_swc_path)

    '''y轴切块'''

    def change_y_step(self):
        self.y_step = self.SetStepRedun.set_step(self.y_step, self.y_redun,
                                                 self.y_step_num, self.y_redun_num)
        self.y_redun_limit.setText(f"- {int(self.y_redun_num.maximum())}")
        self.change_y_step_redun()
        self.get_y_layers()

    def change_y_redun(self):
        self.y_redun = self.SetStepRedun.set_redun(self.y_step, self.y_redun,
                                                   self.y_step_num, self.y_redun_num, self.img3d.shape[1])
        self.y_step_limit.setText(f"{int(self.y_step_num.minimum())} -")
        self.change_y_step_redun()
        self.get_y_layers()

    def get_y_layers(self):
        self.y_num = self.SetStepRedun.get_layers(self.y_step, self.y_redun, self.img3d.shape[1], 1)
        if self.table_widget:
            self.update_xy_table()

    def change_y_step_redun(self):
        (self.box_size, self.origin, self.y_up_over) = self.SetStepRedun.update_box_size(
            [self.x_step, self.y_step, self.z_step], self.img3d.shape[1], self.origin, 1)
        if not self.xyz_cut_index:
            self.create_new_image()

    '''y轴移动'''

    def box_y_to_add(self):
        if self.dataImporter_index and self.box_volume:
            if not self.y_up_over:
                (self.box_size, self.origin, self.y_up_over, self.y_down_over) = self.SetStepRedun.box_to_up(
                    self.y_step, self.y_redun, self.box_size, self.origin, self.img3d.shape[1], self.y_up_over, 1)
                self.create_new_image()
                save_swc_path = self.swc_file_list[self.img_list.currentRow()]  # 保存标记
                self.save_txt_swc(save_swc_path)

    def box_y_to_down(self):
        if self.dataImporter_index and self.box_volume:
            if not self.y_down_over:
                (self.box_size, self.origin, self.y_up_over, self.y_down_over) = self.SetStepRedun.box_to_down(
                    self.y_step, self.y_redun, self.box_size, self.origin, self.y_down_over, 1)
                self.create_new_image()
                save_swc_path = self.swc_file_list[self.img_list.currentRow()]  # 保存标记
                self.save_txt_swc(save_swc_path)

    '''创建boxWidget'''

    def box_Widget(self, box_volume):
        self.BoxWidgets.create_box(box_volume, self)

    '''获取boxWidgetParameter'''

    def boxCallback(self, obj, event):
        self.BoxWidgets.box_call_back(obj, event, self)

    '''box切取的信息处理'''

    def box_Widget_cut(self, bw_size):
        self.BoxWidgets.box_cut_info(bw_size, self)

    '''切取box'''

    def box_cut_label(self):
        self.BoxWidgets.box_cut_label_image(self)

    '''显示隐藏boxWidget'''

    def box_show_hide(self):
        self.BoxWidgets.box_show_hide_update(self)

    '''点击选取感兴趣区域'''

    def point_to_box(self, obj, event):
        self.BoxWidgets.point_to_box_image(self)

    """生成随机颜色"""

    def create_cell_random_color(self, rgb=0):
        if self.mark_color == [1.0, 0.0, 0.0]:
            self.mark_select_color = [0, 1, 0]
        else:
            self.mark_select_color = [1, 0, 0]
        if rgb == 1:
            numbers = self.mark_select_color
        elif rgb == 2 or rgb == 3:
            numbers = self.mark_color
        else:
            # 创建一个列表
            # numbers = [random.randint(120, 255) / 255, 120 / 255, 255 / 255]
            numbers = [random.randrange(120, 255, 5) / 255, 120 / 255, 255 / 255]
            # # 使用 shuffle 函数随机打乱列表顺序
            random.shuffle(numbers)
            # print(color_)
            # numbers = [120 / 255, 120 / 255, 255 / 255]
        return numbers

    '''筛选点'''

    def sift_points(self, sift_box_size, pos, r):
        return (sift_box_size[0] - r <= pos[0] <= sift_box_size[1] + r and
                sift_box_size[2] - r <= pos[1] <= sift_box_size[3] + r and
                sift_box_size[4] - r <= pos[2] <= sift_box_size[5] + r)

    '''选择框架'''

    def sift_box(self, img, need_px):
        if self.data_volume_show_hide_count:  # 判断图文件是否显示
            if need_px:
                box_size = [0, img.shape[2] * self.x_px - 1,
                            0, img.shape[1] * self.y_px - 1,
                            0, img.shape[0] * self.z_px - 1]
            else:
                box_size = [0, img.shape[2] - 1,
                            0, img.shape[1] - 1,
                            0, img.shape[0] - 1]
        else:
            if need_px:
                box_size = [self.box_size[0] * self.x_px, self.box_size[1] * self.x_px,
                            self.box_size[2] * self.y_px, self.box_size[3] * self.y_px,
                            self.box_size[4] * self.z_px, self.box_size[5] * self.z_px]
            else:
                box_size = self.box_size
        return box_size

    '''确认点位置'''

    def points_pos_sure(self, points):
        pos = self.MarkingPointsFun._points_pos_sure(points, self)
        return pos

    '''确认点是否重合'''

    def point_coincidence_sure(self, mypoint, pick_pos, camera_pos):
        coincidence = self.MarkingPointsFun._point_coincidence_sure(mypoint, pick_pos, camera_pos, self)
        return coincidence

    '''关闭窗口事件'''

    def closeEvent(self, QCloseEvent):
        info = self.mess_set("Close the window?", 'Prompt', 2)
        if info == QtWidgets.QMessageBox.Yes:
            # 快捷弹窗关闭
            # self.ShortCutDialog.close()
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()

    '''滑轮前滑'''

    def mouseWheelForward(self, obj, event):
        # print("滚轮向前滚动")
        camera_focal_pos = self.camera.GetFocalPoint()
        camera_pos = self.camera.GetPosition()
        v_ = np.array(camera_focal_pos) - np.array(camera_pos)
        distance = np.linalg.norm(v_)
        # print("相机焦距为：", distance)

    '''滑轮后滑'''

    def mouseWheelBackward(self, obj, event):
        # print("滚轮向后滚动")
        camera_focal_pos = self.camera.GetFocalPoint()
        camera_pos = self.camera.GetPosition()
        v_ = np.array(camera_focal_pos) - np.array(camera_pos)
        distance = np.linalg.norm(v_)
        # print("相机焦距为：", distance)

    """Error message"""

    def error_print(self, e):
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

    """弹窗"""

    def mess_set(self, text, title, nums, icon=QMessageBox.Question):
        box = QMessageBox()
        if nums == 1:
            box.setStandardButtons(QMessageBox.Ok)
        if nums == 2:
            box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setStyleSheet(messagebox_style)
        box.setText(text)
        box.setWindowTitle(title)
        box.setIcon(icon)
        # 设置窗口标志，使其始终在最上面
        box.setWindowFlags(box.windowFlags() | Qt.WindowStaysOnTopHint)

        r = box.exec_()
        return r


if __name__ == '__main__':
    # cd ViewWidget
    # pyinstaller CellViewWidget.py
    # pyinstaller CellViewWidget.spec
    app = QtWidgets.QApplication(sys.argv)
    window = CellViewWidget()
    window.show()
    sys.exit(app.exec_())
