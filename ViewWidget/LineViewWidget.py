# -*- coding: UTF-8 -*-
import sys
import json
import time
import traceback
import numpy as np
import vtkmodules.all as vtk
import random
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPainter, QColor, QFont
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox, QSplitter
from line_points_marking.CreateImageData import CreateImageData
from line_points_marking.CreateDialog import CreateDialog
from line_points_marking.ReadPath import ReadPath
from line_points_marking.SetConnect import SetConnect
from line_points_marking.MarkingPointsFun import MarkingPointsFun
from line_points_marking.MarkProcess import MarkProcess
from line_points_marking.GrayAdjustWidget import GrayAdjustWidget
from line_points_marking.CustomMainWindow import MyWidget
from line_points_marking.InitializeInfo import InitializeInfo
from line_points_marking.ControlStyle import (messagebox_style, lines_button_show_style, image_button_show_style,
                                              image_button_hide_style,
                                              button_alpha_style, button_style)
from BVExample.BVMoudle.BVMoudle import BVReader
from line_points_marking.MarkingPointsFun import nearest_point_to_line_fast


class LineViewWidget(MyWidget, CreateDialog):
    def __init__(self, parent=None):
        super(LineViewWidget, self).__init__(parent)
        self.CreateImageData = CreateImageData  # 创建三维可视图
        self.ReadPath = ReadPath  # 读取文件
        self.SetConnect = SetConnect  # 信号和快捷方式
        self.MarkingPointsFun = MarkingPointsFun  # 标记功能，Point/球的生成
        self.MarkProcess = MarkProcess  # 对生成的球体操作

        # 取消右键菜单
        self.setContextMenuPolicy(Qt.NoContextMenu)
        """初始界面设置"""
        self.action_widgets()
        # 更新工具栏
        self.update_tool_button()
        # 图像列表信号
        self.SetConnect.img_list_connect(self)
        # 创建路径选择窗口
        self.create_path_select_dialog()
        self.SetConnect.select_path_connect(self)  # 路径选择信号
        # 筛选保存路径窗口
        self.create_filter_dialog()
        self.SetConnect.filter_connect(self)  # 筛选保存路径信号
        # 半径调节窗口
        self.create_radius_dialog()
        self.SetConnect.select_radius_connect(self)  # 半径调节信号
        # 工具栏操作
        self.SetConnect.tool_button_clicked(self)  # 工具栏按钮信号
        # 快捷方式操作
        self.SetConnect.short_cut_connect(self)  # 快捷方式信号
        # 浮动窗口位置
        # self.addDockWidget(Qt.LeftDockWidgetArea, self.images_dockWidget)  # 浮动窗口靠左
        # self.addDockWidget(Qt.LeftDockWidgetArea, self.fun_dockWidget)  # 浮动窗口靠左
        self.Revising_verticalLayout.addWidget(self.images_dockWidget)
        self.Revising_verticalLayout.addWidget(self.fun_dockWidget)
        self.FilterWidget_verticalLayout.addWidget(self.branch_dockWidget)
        # 设置拉伸
        """QSplitter 可以实现可拉伸的功能"""
        splitter_horizontal = QSplitter(Qt.Horizontal)
        splitter_horizontal.setHandleWidth(0)
        self.Revising_horizontalLayout.addWidget(splitter_horizontal)
        splitter_horizontal.addWidget(self.RevisingWidget)
        splitter_horizontal.addWidget(self.RevisingFrame)
        splitter_horizontal.addWidget(self.FilterFrame)
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
        self.branch_dockWidget.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        # 设置大小化
        # self.images_dockWidget.setFeatures(QtWidgets.QDockWidget.DockWidgetFloatable)
        # self.fun_dockWidget.setFeatures(QtWidgets.QDockWidget.DockWidgetFloatable)

        # 初始化vtk视图区域
        self.create_vtk_view()
        # 设置坐标系函数
        self.AxesWidgt()
        self.images_dockWidget.installEventFilter(self)
        self.fun_dockWidget.installEventFilter(self)
        self.branch_dockWidget.installEventFilter(self)

        self.bvReader = BVReader()

        # Calculate
        offset = 200
        x = self.x() + offset
        y = self.y() + offset
        self.radius_dialog.move(x, y)

    '''清除界面'''

    def clear_view(self):
        if self.dataImporter_index:  # 存在可标签数据,Update
            info = self.mess_set("Clear the interface?", 'Prompt', 2)
            if info == QtWidgets.QMessageBox.Yes:
                self.image_lineEdit.clear()
                self.swc_lineEdit.clear()

                self.fun_stacked_widget.removeWidget(self.widgetFun)  # 更新操作栏
                self.branch_stacked_widget.removeWidget(self.branch_select_widget)

                self.widgetFun.deleteLater()
                self.branch_select_widget.deleteLater()  # 正确删除并释放内存

                self.fun_stacked_widget.addWidget(self.fun_widget)
                self.branch_stacked_widget.addWidget(self.branch_widget)

                del self.img3d
                if self.img3d_mark is not None:
                    del self.img3d_mark
                del self.dataImporter
                if self.dataImporter_mark is not None:
                    del self.dataImporter_mark
                del self.dataImporter_outline
                del self.volume
                del self.outline
                del self.outlineActor
                if self.volume_mark is not None:
                    del self.volume_mark
                if self.outlineActor0 is not None:
                    del self.outlineActor0

                InitializeInfo.__init__(self)
                self.img_list.clear()
                self.branch_select_listwidget.clear()
                self.ren.RemoveAllViewProps()  # 去除所有渲染对象
                self.ren_v.RemoveAllViewProps()  # 去除所有渲染对象
                self.renWin.Render()
                self.dataImporter_index = 0

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
            for actor in self.cross_actor_points:
                self.ren.RemoveActor(actor)  # 删除渲染的点
            for actor in self.cross_actor_points_copy:
                self.ren.RemoveActor(actor)  # 删除渲染的点
            # self.ren.RemoveActor(self.one_point_actor)  # 删除点
            # self.ren.RemoveActor(self.one_line_actor)  # 删除直线
            if not self.res_save_Button.isEnabled():
                self.point_line_count = True  # 点击生成线，1为激活状态
                self.one_sphere_count = False  # 单击生成点，0为未触发
            self.radius_dialog.hide()
            self.res_save_Button.setEnabled(False)
            # 重新渲染
            self.renWin.Render()
            self.cross_actor_points = []
            self.cross_actor_points_copy = []
        except Exception as e:
            print("No object to delete!")
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
            if self.volume_mark:
                # 如果已有数据，直接更新mapper和property，不重建volume
                mapper = self.volume_mark.GetMapper()
                if render_mode == "MIP":
                    mapper.SetBlendModeToMaximumIntensity()
                else:  # COMPOSITE
                    mapper.SetBlendModeToComposite()
            self.renWin.Render()
            self.render_mode_text = render_mode

    """切换修订模式"""

    def change_marking_mode(self):
        if self.dataImporter_index:
            if self.is_marking:
                self.end_marking()
            else:
                self.start_marking()

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

    """开始标记"""

    def start_marking(self):
        if self.dataImporter_index:
            if not self.is_marking:
                text = self.function_change_label.text()
                if "(" in text:
                    text = text.split("(")[0]
                self.function_change_label.setText(f"{text}(Revision Mode)")
                self.is_marking = True
                self.new_points_list = []
                self.new_Position = []
                self.new_mark_lines = []
                self.set_button(self.marking_button, "Close Revision Mode", "is_marking.png")
                # 更新工具栏
                self.update_tool_button()

    """结束标记"""

    def end_marking(self):
        if self.dataImporter_index:
            if self.is_marking:
                # 记录半径
                self.remember_radius()
                # 确认标记
                self.MarkProcess.mark_sure(self)
                text = self.function_change_label.text().split('(')[0]
                self.function_change_label.setText(text)
                self.is_marking = False
                self.judge_actor()  # 删除多余点线
                self.set_button(self.marking_button, "Start Revision Mode", "marking.png")
                # 更新工具栏
                self.update_tool_button()

    """更新工具栏"""

    def update_tool_button(self):
        if self.is_marking:
            self.sure_marking_button.show()
            self.withdraw_all_button.show()

            self.delete_mark_button.hide()
            self.delete_limit_button.hide()
            self.clear_mark_button.hide()
        else:
            self.sure_marking_button.hide()
            self.withdraw_all_button.hide()

            self.delete_mark_button.show()
            self.delete_limit_button.show()
            self.clear_mark_button.show()

    """创建点"""

    def create_point(self, pos, is_left):
        pos = [round(pos[0], 2), round(pos[1], 2), round(pos[2], 2)]
        # 生成线
        if len(self.new_Position) > 0:
            self.MarkProcess._create_line(pos, is_left, self)
        # 生成点
        p_size = self.p_size + 2
        self.MarkProcess._create_point(pos, p_size, self, (0, 0, 1))
        self.renWin.Render()  # 重新渲染

    """创建点线"""

    def create_PolyData(self, points, lines, vertices):
        actor = self.MarkProcess._create_PolyData(points, lines, vertices, self)
        return actor

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

    def onSelectMarkingFun(self, is_left):
        # if self.function_index == 0:
        #     if not self.res_save_Button.isEnabled():
        #         self.MarkingPointsFun.Cross_onLeftButtonPressEvent(is_left, self)
        # else:
        self.MarkingPointsFun.Max_onLeftButtonPressEvent(is_left, self)

    """Releasedalt,连接或断开"""

    def PloyData_connect_disconnect(self):
        if self.dataImporter_index:
            if not self.is_marking:
                if self.vtk_LP_Actor:
                    l = len(self.connect_list)
                    if l:
                        if l == 2:
                            self.MarkingPointsFun.PloyData_connect(self)
                            # print("Connect")
                        elif l == 1:
                            self.MarkingPointsFun.PloyData_disconnect(self)
                            # print("Disconnect")
                        for act in self.new_points_list:
                            self.ren.RemoveActor(act)
                        self.connect_list = []
                        self.new_points_list = []
                        self.new_Position = []
                        self.color_i = None
                        # 重新渲染
                        self.renWin.Render()

                        self.tree_count = len(self.vtk_LP_Actor)
                        self.label_count.setText(f"{self.tree_count}")

    """左键选择点线"""

    def onSelectPolyData(self):
        if self.vtk_LP_Actor and not self.LP_hide:
            if self.color_i is not None:
                try:
                    color_ = self.create_random_color()
                    act = self.vtk_LP_Actor[self.color_i]
                    act.GetProperty().SetColor(color_)  # 更新颜色
                    act.GetProperty().SetPointSize(self.p_size)  # 点大小
                except IndexError as e:
                    self.error_print(e)

            pick_pos = self.MarkingPointsFun.get_picker_position(self)
            camera_pos = self.camera.GetPosition()  # 获取相机位置坐标

            # 计算向量diffSet
            # box_size = self.sift_box(self.img3d, 0)

            select_r = 2
            # 寻找最近树和点
            norms, norms_pos, norms_index = self.MarkingPointsFun.check_nearest_tree_point(camera_pos, pick_pos,
                                                                                           select_r, self)
            if not len(norms):
                return
            i = np.argmin(norms)  # 选中树序号
            min_norm = norms[i]
            # print('min_norm: ', min_norm)

            if min_norm <= select_r:
                print(min_norm, i)
                actor = self.vtk_LP_Actor[i]
                actor.GetProperty().SetColor(1, 0, 0)
                actor.GetProperty().SetPointSize(self.p_size + 2)  # 点大小
                self.color_i = i
                self.focal_pos = norms_pos[i]  # 选中点
                points = self.vtk_Points[i]
                num_points = points.GetNumberOfPoints()  # 获取点数
                print("Number of points:", num_points)
                # print(self.img3d_mark_info)
                # print(self.focal_pos)
                # print("选中树！")
                # break
            else:
                self.color_i = None
                self.focal_pos = None
                # print("未选中树！")

            self.renWin.Render()

    '''读取文件路径'''

    def read_file_path(self):
        self.ReadPath._read_file_path(self)

    """更新分支点选择列表项"""

    def update_branch_list(self):
        if self.branch_dict:
            if len(self.branch_dict) <= 1:
                self.branch_forward_Button.setStyleSheet(button_alpha_style)
                self.branch_backward_Button.setStyleSheet(button_alpha_style)
                self.branch_forward_Button.setEnabled(False)
                self.branch_backward_Button.setEnabled(False)
            else:
                self.branch_forward_Button.setStyleSheet(button_style)
                self.branch_backward_Button.setStyleSheet(button_style)
                self.branch_forward_Button.setEnabled(True)
                self.branch_backward_Button.setEnabled(True)
            self.branch_select_listwidget.blockSignals(True)
            self.branch_select_listwidget.clear()
            for i in self.branch_dict.keys():
                # check_box = QtWidgets.QCheckBox()  # 修饰框
                # check_box.setStyleSheet("QCheckBox::indicator{width:0px;height:0px;}")
                item = QtWidgets.QListWidgetItem()
                # lg = QLinearGradient(0, 0, 300, 0)
                # lg.setColorAt(0, QColor(25, 35, 45))
                # lg.setColorAt(1, QColor(90, 100, 120))
                # item.setBackground(QBrush(lg))
                # item.setBackground(QColor(70, 80, 100))  # 设置背景颜色
                item.setText(f"{i}")  # 列表项名字
                item.setFont(QFont("微软雅黑", 9))  # 字体
                item.setSizeHint(QSize(300, 18))  # 设置列表项尺寸
                self.branch_select_listwidget.addItem(item)  # 添加列表项
                # self.z_list.setItemWidget(item, check_box)  # 设置组合
                self.branch_select_listwidget.update()  # 更新列表项
            self.branch_select_listwidget.blockSignals(False)
            if not self.isReload:
                self.branch_select_listwidget.setCurrentRow(0)
        else:
            self.branch_forward_Button.setStyleSheet(button_alpha_style)
            self.branch_backward_Button.setStyleSheet(button_alpha_style)
            self.branch_forward_Button.setEnabled(False)
            self.branch_backward_Button.setEnabled(False)
            self.branch_select_listwidget.blockSignals(True)
            self.branch_select_listwidget.clear()
            self.branch_select_listwidget.blockSignals(False)

    """分支点区域显示"""

    def branch_region_show(self, now_item, pre_item):
        if pre_item:
            pre_item.setForeground(QColor("green"))  # 字体颜色为绿色
        now_index = self.branch_select_listwidget.currentRow()
        self.focal_pos = self.branch_dict.get(now_index)
        # print(self.focal_pos)
        if self.focal_pos is None:
            # print("没有分支点")
            return
        else:
            if now_index != 0:
                self.branch_forward_Button.setEnabled(True)
                self.branch_forward_Button.setStyleSheet(button_style)
            else:
                self.branch_forward_Button.setEnabled(False)
                self.branch_forward_Button.setStyleSheet(button_alpha_style)
            if now_index != self.branch_select_listwidget.count() - 1:
                self.branch_backward_Button.setEnabled(True)
                self.branch_backward_Button.setStyleSheet(button_style)
            else:
                self.branch_backward_Button.setEnabled(False)
                self.branch_backward_Button.setStyleSheet(button_alpha_style)
            self.new_mark_create()  # 存在分支点，创建裁剪标记图像
            self.camera_focal()  # 聚焦到分支点

    """上一个分支点"""

    def to_forward_branch(self):
        if self.branch_select_listwidget.count():
            now_index = self.branch_select_listwidget.currentRow()
            if now_index != 0:
                self.branch_select_listwidget.setCurrentRow(now_index - 1)
            if now_index - 1 <= 0:
                self.branch_forward_Button.setEnabled(False)
                self.branch_forward_Button.setStyleSheet(button_alpha_style)
            if now_index == self.branch_select_listwidget.count() - 1:
                self.branch_backward_Button.setEnabled(True)
                self.branch_backward_Button.setStyleSheet(button_style)

    """下一个分支点"""

    def to_backward_branch(self):
        if self.branch_select_listwidget.count():
            now_index = self.branch_select_listwidget.currentRow()
            if now_index != self.branch_select_listwidget.count() - 1:
                self.branch_select_listwidget.setCurrentRow(now_index + 1)
            if now_index + 1 >= self.branch_select_listwidget.count() - 1:
                self.branch_backward_Button.setEnabled(False)
                self.branch_backward_Button.setStyleSheet(button_alpha_style)
            if now_index == 0:
                self.branch_backward_Button.setEnabled(True)
                self.branch_backward_Button.setStyleSheet(button_style)

    """确认分支点"""

    def branch_finished(self):
        if self.branch_select_listwidget.count() > 0 and self.branch_dict:
            now_index = self.branch_select_listwidget.currentRow()
            now_item = self.branch_select_listwidget.item(now_index)
            if "√" not in now_item.text():
                name = now_item.text() + "  √"
                now_item.setText(name)
        # print("branch_finished")

    """取消分支点"""

    def branch_closed(self):
        if self.branch_select_listwidget.count() > 0 and self.branch_dict:
            now_index = self.branch_select_listwidget.currentRow()
            now_item = self.branch_select_listwidget.item(now_index)
            if "√" in now_item.text():
                name = now_item.text().split("  √")[0]
                now_item.setText(name)
        # print("branch_closed")

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

    """确定标记"""

    def sure_marking(self):
        # 存在图像
        if self.dataImporter_index:  # 存在可标签数据,Update
            if self.is_marking:
                # 记录半径
                self.remember_radius()
                # 确认标记
                self.MarkProcess.mark_sure(self)

    '''保存坐标为txt'''

    def save_position(self):
        # 存在图像
        if self.dataImporter_index:  # 存在可标签数据,Update
            # 保存坐标
            self.ReadPath.save_sphere_pos(self)

    """设置用户功能"""

    def set_user_function(self):
        self.ShortCutDialog.stackedWidget.setCurrentIndex(0)
        self.ShortCutDialog.listWidget.setCurrentRow(0)
        self.ShortCutDialog.show()

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
        image_info = [self.img3d]
        # 对类进行实例化，并调用方法（Function）
        (self.dataImporter, self.volume, self.opacityTransferFunction,
         self.colorTransferFunction) = self.CreateImageData(image_info, self.render_mode_text).create_img3d()

    '''DataImporter_mark_create,tifRendering'''

    def DataImporter_mark_create(self):
        image_info = [self.img3d_mark]
        # 对类进行实例化，并调用方法（Function）
        (self.dataImporter_mark, self.volume_mark, self.opacityTransferFunction_mark,
         self.colorTransferFunction_mark) = self.CreateImageData(image_info, self.render_mode_text).create_img3d()
        """裁剪轮廓"""
        outline = vtk.vtkOutlineSource()
        xmin = (self.img3d_mark_info[0][0])
        xmax = (self.img3d_mark_info[1][0] - 1)  # x
        ymin = (self.img3d_mark_info[0][1])
        ymax = (self.img3d_mark_info[1][1] - 1)  # y
        zmin = (self.img3d_mark_info[0][2])
        zmax = (self.img3d_mark_info[1][2] - 1)  # z
        outline.SetBounds(xmin, xmax, ymin, ymax, zmin, zmax)

        outlineMapper = vtk.vtkPolyDataMapper()
        outlineMapper.SetInputConnection(outline.GetOutputPort())

        if self.outlineActor0 is not None:
            self.ren.RemoveActor(self.outlineActor0)

        self.outlineActor0 = vtk.vtkActor()
        self.outlineActor0.SetMapper(outlineMapper)
        self.outlineActor0.GetProperty().SetColor(1, 1, 0)
        self.outlineActor0.GetProperty().SetLineWidth(1)

        self.ren.AddActor(self.outlineActor0)

        """裁剪分支"""
        if len(self.vtk_LP_Actor):
            new_list = []
            for i in range(len(self.vtk_LP_Actor)):
                actor = self.vtk_LP_Actor[i]
                self.ren.RemoveActor(actor)
                points = self.vtk_Points[i]
                lines = self.vtk_Lines[i]
                vertices = self.vtk_Vertices[i]
                # 创建 PolyData 并设置点和线
                actor = self.create_PolyData(points, lines, vertices)
                new_list.append(actor)
            self.vtk_LP_Actor = new_list
            self.PolyData_copy = []

    """新数据生成"""

    def new_mark_create(self):
        if self.dataImporter_index:
            if self.focal_pos is not None:
                if self.volume_mark is not None:
                    self.ren_v.RemoveActor(self.volume_mark)
                # 创建新数据块
                self.img3d_mark = self.MarkingPointsFun.create_new_mark(self.focal_pos, self.boxR_size, self)
                self.DataImporter_mark_create()
                self.volume_mark.SetPosition([self.img3d_mark_info[0][0],
                                              self.img3d_mark_info[0][1],
                                              self.img3d_mark_info[0][2]])
                self.ren_v.AddVolume(self.volume_mark)
                self.CreateImageData.update_color_alpha(self.opacityTransferFunction_mark,
                                                        self.colorTransferFunction_mark,
                                                        self.gray_min, self.gray_max, self.r, self.g, self.b)
                self.volume.VisibilityOff()
                self.volume_mark.VisibilityOn()
                self.renWin.Render()

    '''Outline_create,包围盒'''

    def Outline_create(self):
        (self.dataImporter_outline, self.outline,
         self.outlineActor) = self.CreateImageData.create_outline(self.dataImporter)

    '''自定义功能'''

    def object_message(self):
        # 更新图像灰度和颜色
        self.CreateImageData.update_color_alpha(self.opacityTransferFunction, self.colorTransferFunction,
                                                self.gray_min, self.gray_max, self.r, self.g, self.b)

        # self.CreateImageData.update_color_alpha(self.opacityTransferFunction_mark, self.colorTransferFunction_mark,
        #                                         self.gray_min, self.gray_max, self.r, self.g, self.b)

        if self.reOpen:
            # 功能栏
            self.fun_widgets()
            # 分支点选择界面
            self.branch_widgets()
            # 功能栏信号
            self.SetConnect.fun_widget_info_connect(self)
            # 分支点选择信号
            self.SetConnect.branch_widget_info_connect(self)
            # 灰度调节
            self.grayWidget = GrayAdjustWidget(self.graphframe)
            self.grayWidget.leftValueChanged.connect(self.grayTransCallback)  # 设置灰度图槽函数
            self.grayWidget.rightValueChanged.connect(self.grayTransCallback)
            self.graphWidget_form.addRow(self.gray_label, self.grayWidget)
        else:
            # Image、分支显示和隐藏
            self.show_image_button.setStyleSheet(image_button_show_style)
            self.show_lines_button.setStyleSheet(lines_button_show_style)

        # 图像是否重新加载或切换
        self.isReload = True
        hist = self.gray_widget_message()
        self.grayWidget.setLeftValue(self.gray_min)
        self.grayWidget.setRightValue(self.gray_max)
        self.grayWidget.setHist(hist)

        # 更新分支点选择列表信息
        self.update_branch_info()
        # 分支点选择列表窗口加入堆叠窗口
        self.branch_select_listwidget.show()
        self.branch_stacked_widget.addWidget(self.branch_select_widget)

        # 更新功能框信息
        self.update_fun_box_info()
        # 功能栏窗口加入堆叠窗口
        self.fun_stacked_widget.addWidget(self.widgetFun)

    """更新分支点选择列表信息"""

    def update_branch_info(self):
        pass

    """更新功能框信息"""

    def update_fun_box_info(self):
        # 标记功能切换
        if self.is_marking:
            self.fun_label = self.function_change_label.text().split('(')[0]
            self.fun_label = f"{self.fun_label}(Revision Mode)"
        else:
            self.set_button(self.marking_button, "Start Revision Mode", "marking.png")
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
        # 调节点数
        self.adjust_spinBox.setValue(self.adjust_nums)
        # 灰度
        self.gray_auto_change_button.setText(f"{self.min_auto_gray} / {self.max_auto_gray}")
        # 点大小
        self.size_spinBox.setValue(self.p_size)
        # 长度限制
        self.length_spinBox.setValue(self.length_size)
        # 框半径
        self.boxR_spinBox.setValue(self.boxR_size)
        # 筛选数
        if not self.dataImporter_index:
            self.filter_nums_label.setText("0")
        # 灰度调节
        self.gray_min_num.setValue(self.gray_min)
        self.gray_max_num.setValue(self.gray_max)

    '''读取txtFile,读取选择文件的txtFile,渲染标签球体'''

    def read_txt_file(self, index):
        self.ReadPath._read_txt_file(index, self)

    """Togglevolume"""

    def volume_change(self):
        if self.dataImporter_index:
            if self.volume.GetVisibility():
                self.volume.VisibilityOff()
                if self.volume_mark is not None:
                    self.volume_mark.VisibilityOn()
                if self.outlineActor0 is not None:
                    self.outlineActor0.VisibilityOn()
                self.show_image_button.setStyleSheet(image_button_hide_style)
            else:
                self.volume.VisibilityOn()
                if self.volume_mark is not None:
                    self.volume_mark.VisibilityOff()
                if self.outlineActor0 is not None:
                    self.outlineActor0.VisibilityOff()
                self.show_image_button.setStyleSheet(image_button_show_style)
            """裁剪分支"""
            if len(self.vtk_LP_Actor):
                new_list = []
                for i in range(len(self.vtk_LP_Actor)):
                    actor = self.vtk_LP_Actor[i]
                    self.ren.RemoveActor(actor)
                    points = self.vtk_Points[i]
                    lines = self.vtk_Lines[i]
                    vertices = self.vtk_Vertices[i]
                    # 创建 PolyData 并设置点和线
                    actor = self.create_PolyData(points, lines, vertices)
                    new_list.append(actor)
                self.vtk_LP_Actor = new_list
                self.PolyData_copy = []
            self.renWin.Render()

    '''球体快捷显示'''

    def _LP_show_hide(self):
        self.MarkProcess.LP_show_hide(self)

    """包围盒快捷显示"""

    def Outline_quick_key(self):
        self.MarkProcess.outline_show_hide(self)

    """更新调节点数"""

    def update_adjust_size(self):
        self.adjust_nums = self.adjust_spinBox.value()

    """显示和隐藏调节界面"""

    def show_dialog_hide(self):
        if self.function_index == 0 and self.is_marking:
            if self.is_cross_finish == False:  # 调节界面是否冻结
                if self.radius_dialog.isHidden():
                    self.radius_dialog.show()
                else:
                    self.radius_dialog.hide()

    """更新点大小"""

    def update_point_size(self):
        self.p_size = self.size_spinBox.value()
        if self.vtk_LP_Actor:
            self.MarkProcess._update_point_size(self)

    """更新长度限制"""

    def update_length_size(self):
        self.length_size = self.length_spinBox.value()

    """更新裁剪框半径"""

    def update_boxR_size(self):
        self.boxR_size = self.boxR_spinBox.value()

    '''open file,选择文件路径,路径选择框,是否保存当前文件的txt和swc'''

    def select_file_path_show(self):
        self.select_path_dialog.show()

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

    '''最小灰度值'''

    def textinput_gray_min(self):
        # 控制最小灰度
        self.gray_min = self.gray_min_num.value()
        self.grayWidget.setLeftValue(self.gray_min)

        # self.gray_limit = int(np.mean([self.gray_min, self.gray_max]))
        # print("gray_limit: ", self.gray_limit)

    '''最大灰度值'''

    def textinput_gray_max(self):
        # 控制最大灰度
        self.gray_max = self.gray_max_num.value()
        self.grayWidget.setRightValue(self.gray_max)
        # self.gray_limit = int(np.mean([self.gray_min, self.gray_max]))
        # print("gray_limit: ", self.gray_limit)

    '''固定灰度'''

    def sure_gray_auto(self):
        if self.dataImporter_index == 1:
            self.min_auto_gray = self.gray_min
            self.max_auto_gray = self.gray_max
            self.gray_auto_change_button.setText(f'{self.min_auto_gray} / {self.max_auto_gray}')

    '''自动更新灰度'''

    def gray_auto_change(self):
        self.CreateImageData.gray_change(3, self)

    '''修改颜色和透明度'''

    def change_color_opacity(self):
        self.CreateImageData.update_color_alpha(self.opacityTransferFunction, self.colorTransferFunction,
                                                self.gray_min, self.gray_max, self.r, self.g, self.b)
        if self.volume_mark is not None:
            self.CreateImageData.update_color_alpha(self.opacityTransferFunction_mark, self.colorTransferFunction_mark,
                                                    self.gray_min, self.gray_max, self.r, self.g, self.b)
        # print("当前最小灰度:", self.gray_min)
        # print("当前最大灰度:", self.gray_max)
        self.renWin.Render()

    """生成随机颜色"""

    def create_random_color(self):
        # 创建一个列表
        # numbers = [random.randint(120, 255) / 255, 120 / 255, 255 / 255]
        numbers = [random.randrange(120, 255, 5) / 255, 120 / 255, 255 / 255]
        # # 使用 shuffle 函数随机打乱列表顺序
        random.shuffle(numbers)
        # print(color_)
        # numbers = [120 / 255, 120 / 255, 255 / 255]
        return numbers

    '''鼠标左键点击'''

    def LeftButtonPress(self, obj, event):
        try:
            if self.dataImporter_index:
                if not self.is_marking:
                    self.MarkProcess.LeftButtonPressSet(self)
                    # print("鼠标左键点击！")
        except Exception as e:
            self.error_print(e)

    '''鼠标右键点击'''

    def RightButtonPress(self, obj, event):
        try:
            if self.dataImporter_index:
                if not self.is_marking:
                    self.MarkProcess.RightButtonPressSet(self)
                    # print("鼠标右键点击！")
        except Exception as e:
            self.error_print(e)

    '''鼠标中键点击'''

    def MiddleButtonPress(self, obj, event):
        try:
            if self.dataImporter_index:
                if not self.is_marking:
                    self.MarkProcess.MiddleButtonPressSet(self)
                    # print("鼠标中键点击！")
        except Exception as e:
            self.error_print(e)

    '''actorDelete'''

    def Actor_Delete(self):
        if self.dataImporter_index and not self.LP_hide:
            if not self.is_marking:
                if self.color_i is not None:  # 选中树序号
                    # self.MarkProcess.delete_shpere(self)
                    self.MarkProcess.delete_PolyData(self)

    """删除所有actor"""

    def Actor_All_Delete(self):
        if self.dataImporter_index and not self.LP_hide:
            if not self.is_marking:
                self.MarkProcess.delete_all_PolyData(self)

    """删除所有限制长度actor"""

    def Actor_All_Delete_limit(self):
        if self.dataImporter_index and not self.LP_hide:
            if not self.is_marking:
                self.MarkProcess.delete_all_PolyData_limit(self)

    '''撤回点：一个点
    恢复点线：恢复删除的点线'''

    def Point_Withdraw(self):
        if self.dataImporter_index:
            if self.is_marking:  # 修订模式
                if self.new_Position or self.one_sphere_count:
                    self.MarkProcess.points_withdraw(self)
            else:
                self.MarkProcess.PolyData_withdraw(self)

    """撤回标记：所有点"""

    def Mark_Withdraw(self):
        if self.dataImporter_index:
            if self.is_marking:  # 修订模式
                if self.new_Position or self.one_sphere_count:
                    self.MarkProcess.marks_withdraw(self)

    '''Focus'''

    def camera_focal(self):
        if self.dataImporter_index:
            if self.focal_pos is not None:
                pos = self.focal_pos
                print(pos)
                self.camera.SetFocalPoint(pos)
                self.renWin.Render()

    '''灰度减少'''

    def gray_left(self):
        self.CreateImageData.gray_change(1, self)

    '''灰度增加'''

    def gray_right(self):
        self.CreateImageData.gray_change(2, self)

    '''图像切换'''

    def item_up(self):
        if self.dataImporter_index:
            if not self.is_marking:
                if self.img_list.currentRow() == 0:
                    return
                else:
                    self.image_file_list_index = self.img_list.currentRow() - 1
                    self.img_list.setCurrentRow(self.image_file_list_index)

    def item_down(self):
        if self.dataImporter_index:
            if not self.is_marking:
                if self.img_list.currentRow() == self.img_list.count() - 1 or self.img_list.currentRow() == -1:
                    return
                else:
                    self.image_file_list_index = self.img_list.currentRow() + 1
                    self.img_list.setCurrentRow(self.image_file_list_index)

    """半径调节"""

    def update_forw_line(self, v):
        """SpinBox 改变 -> forw_line Move"""
        self.forw_line.setPos(v)
        v1, v2 = int(v), int(self.backward_spinBox.value())
        self.update_render(v1, v2)

    def update_back_line(self, v):
        """SpinBox 改变 -> back_line Move"""
        self.back_line.setPos(v)
        v1, v2 = int(self.forward_spinBox.value()), int(v)
        self.update_render(v1, v2)

    def update_render(self, v1, v2):
        self.center_line.setPos(int((v1 + v2) / 2))
        if not self.statistics_dict:
            return
        point_list = self.statistics_dict['point_list']
        if v2 > len(point_list) - 1:
            v2 = len(point_list) - 1
        if v1 < 0:
            v1 = 0
        points = [[tuple(point_list[v1]), tuple(point_list[v2])]]
        voxel_size = (self.x_resolution_doubleSpinBox.value(),
                      self.y_resolution_doubleSpinBox.value(),
                      self.z_resolution_doubleSpinBox.value())
        length = self.total_length_vox(points, voxel_size)
        if self.res_save_Button.isEnabled():
            self.d_h_nums_label.setText(f"{round(length, 3)}")
        else:
            self.d_v_nums_label.setText(f"{round(length, 3)}")
        actor_list = self.cross_actor_points
        for i, actor in enumerate(actor_list):
            if v1 <= i <= v2:
                actor.VisibilityOn()
            else:
                actor.VisibilityOff()
        self.renWin.Render()  # 重新渲染

    def total_length_vox(self, segments, voxel_size=(1, 1, 1)):
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

    def _clamp_line(self, line, spinbox):
        """把 line 的 x 坐标钳位到 [X_MIN, X_MAX]"""
        self.X_MIN = spinbox.minimum()
        self.X_MAX = spinbox.maximum()
        x = line.pos().x()
        if x < self.X_MIN:
            line.setPos(self.X_MIN)
            x = self.X_MIN
        elif x > self.X_MAX:
            line.setPos(self.X_MAX)
            x = self.X_MAX
        return x

    def forw_line_moved(self, line):
        """forw_line 被鼠标拖动 -> 更新控件"""
        x = self._clamp_line(line, self.forward_spinBox)
        self.forward_spinBox.setValue(round(x))

    def back_line_moved(self, line):
        """back_line 被鼠标拖动 -> 更新控件"""
        x = self._clamp_line(line, self.backward_spinBox)
        self.backward_spinBox.setValue(round(x))

    def pw_mouse_clicked(self, ev):
        if ev.button() != Qt.LeftButton:
            return
        vb = self.pw.plotItem.vb
        scene_pos = ev.scenePos()
        if not self.pw.sceneBoundingRect().contains(scene_pos):
            return
        mouse_point = vb.mapSceneToView(scene_pos)
        if self.statistics_dict:
            X_MIN = 0
            X_MAX = len(self.statistics_dict['index_list']) - 1
            x = max(X_MIN, min(X_MAX, mouse_point.x()))

            # Divider
            maxIndex = self.statistics_dict['maxIndex']

            self.active_line = self.forw_line if x < maxIndex else self.back_line
            self.active_line.setPos(x)
            if self.active_line is self.forw_line:
                self.forward_spinBox.setValue(round(x))
            else:
                self.backward_spinBox.setValue(round(x))

    def save_radius_res(self):
        for actor in self.cross_actor_points:
            self.ren.RemoveActor(actor)  # 删除渲染的点

        for actor in self.cross_actor_points_copy:
            self.ren.RemoveActor(actor)  # 删除渲染的点

        # 获取第二条直线的中点
        i1 = self.forward_spinBox.value()
        i2 = self.backward_spinBox.value()
        maxPoint = (np.array(self.statistics_dict['point_list'][i1]) +
                    np.array(self.statistics_dict['point_list'][i2])) / 2
        self.focal_pos = maxPoint
        self.statistics_dict['maxPoint'] = maxPoint

        self.renWin.Render()  # 重新渲染
        self.res_save_Button.setEnabled(False)
        self.radius_dialog.hide()
        self.one_sphere_count = False  # 未触发
        self.is_cross_finish = True  # 冻结调节界面
        self.create_point(self.statistics_dict['maxPoint'], self.is_left)

        add_len = len(self.new_Position)
        if add_len == 1:
            pos = self.new_Position[-1]
            pos_key = self.pos_to_key(pos)
            self.new_radius_dict[pos_key] = (float(self.d_v_nums_label.text()) +
                                             float(self.d_h_nums_label.text())) / 4.
        elif add_len > 1:
            back_pos = self.new_Position[-1]
            back_pos_key = self.pos_to_key(back_pos)
            self.new_radius_dict[back_pos_key] = (float(self.d_v_nums_label.text()) +
                                                  float(self.d_h_nums_label.text())) / 4.

            for i, pos in enumerate(self.new_Position[:-1]):
                pos_key = self.pos_to_key(pos)
                if not self.new_radius_dict.get(pos_key):
                    rB = self.new_radius_dict.get(back_pos_key)
                    if i == 0:  # 模式切换后继续标记，第一个点非交叉点
                        pos0 = self.new_Position[0]
                        pos0_key = self.pos_to_key(pos0)
                        r = self.radius_dict.get(pos0_key)
                        if not r:
                            rA = rB
                            self.new_radius_dict[pos0_key] = rA
                        else:
                            rA = self.radius_dict.get(pos0_key)
                        points_list = self.new_Position[1:-1]
                    else:
                        pos = self.new_Position[i - 1]
                        pos_key = self.pos_to_key(pos)
                        rA = self.new_radius_dict.get(pos_key)
                        points_list = self.new_Position[i:-1]

                    if points_list:
                        # 梯度插值计算半径
                        add_r = self.interp_segment(pos, back_pos, rA, rB, points_list)
                        # print(add_r)
                        # print(add_h)
                        for ii, p in enumerate(points_list):
                            p_key = self.pos_to_key(p)
                            self.new_radius_dict[p_key] = np.round(add_r[ii], 4)
                        break

    def remember_radius(self):
        # 半径集合合并
        self.radius_dict.update(self.new_radius_dict)
        self.new_radius_dict = {}

    def interp_segment(self, pA, pB, vA, vB, pts):
        """
        pA, pB : 起点、终点坐标，任意维度 ndarray/list
        vA, vB : 起点、终点对应的标量值
        pts    : 待插值点坐标数组，形状 (N, dim)
        return : 插值结果数组，Length N
        """
        pA, pB = np.asarray(pA), np.asarray(pB)
        pts = np.asarray(pts)
        # 向量 AB 与 AP
        AB = pB - pA
        L2 = np.dot(AB, AB)  # 长度平方
        AP = pts - pA
        # 投影比例 t
        t = np.dot(AP, AB) / L2
        t = np.clip(t, 0, 1)  # 若 pts 不在 AB 段上，可截断
        return vA + t * (vB - vA)

    def pos_to_key(self, pos):
        """
        将三维坐标转换为字符串键

        Args:
            pos: 三维坐标数组或列表

        Returns:
            str: 坐标的字符串表示，格式为"x,y,z"
        """
        return f"{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}"

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

    '''筛选点'''

    def sift_points(self, sift_box_size, pos, r):
        return (sift_box_size[0] - r <= pos[0] < sift_box_size[1] + r and
                sift_box_size[2] - r <= pos[1] < sift_box_size[3] + r and
                sift_box_size[4] - r <= pos[2] < sift_box_size[5] + r)

    '''选择框架'''

    def sift_box(self):
        # 原图显示
        if self.volume.GetVisibility():
            box_size = [0, self.img3d.shape[2] - 1,
                        0, self.img3d.shape[1] - 1,
                        0, self.img3d.shape[0] - 1]
        else:
            if self.volume_mark is not None:
                box_size = [self.img3d_mark_info[0][0], self.img3d_mark_info[1][0] - 1,
                            self.img3d_mark_info[0][1], self.img3d_mark_info[1][1] - 1,
                            self.img3d_mark_info[0][2], self.img3d_mark_info[1][2] - 1
                            ]
            else:
                box_size = [0 for i in range(6)]

        return box_size

    '''确认最近点位置'''

    def points_pos_nearest(self, points, lp):
        pos, statistics_dict = self.MarkingPointsFun._points_pos_nearest(points, lp, self)
        return pos, statistics_dict

    '''确认点位置'''

    def points_pos_sure(self, points):
        pos, statistics_dict = self.MarkingPointsFun._points_pos_sure(points, self)
        return pos, statistics_dict

    '''确认点是否重合'''

    def point_coincidence_sure(self, mypoint, pick_pos, camera_pos):
        if self.LP_hide:
            coincidence = False
            new_l = len(self.new_Position)
            if new_l:
                # 计算方向向量v
                v_ = np.array(pick_pos) - np.array(camera_pos)
                pick_pos = np.array(pick_pos)
                all_points = np.array(self.new_Position)
                nearest_point, min_distance, min_idx = nearest_point_to_line_fast(all_points, pick_pos, v_)
                coin_r = 2  # 点重合
                if min_distance <= coin_r:
                    print("New coincidence detected")
                    coincidence = True
        else:
            coincidence, mypoint = self.MarkingPointsFun._point_coincidence_sure(mypoint, pick_pos, camera_pos, self)
        return coincidence, mypoint

    '''关闭窗口事件'''

    def closeEvent(self, QCloseEvent):
        info = self.mess_set("Close the window?", 'Prompt', 2)
        if info == QtWidgets.QMessageBox.Yes:
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
        print("Camera focal distance:", distance)

    '''滑轮后滑'''

    def mouseWheelBackward(self, obj, event):
        # print("滚轮向后滚动")
        camera_focal_pos = self.camera.GetFocalPoint()
        camera_pos = self.camera.GetPosition()
        v_ = np.array(camera_focal_pos) - np.array(camera_pos)
        distance = np.linalg.norm(v_)
        print("Camera focal distance:", distance)

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
    app = QtWidgets.QApplication(sys.argv)
    window = LineViewWidget()
    window.show()
    sys.exit(app.exec_())
