# -*- coding: utf-8 -*-
import vtkmodules.all as vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QIcon, QKeySequence, QMouseEvent, QColor
from PyQt5.QtWidgets import QWidget, QShortcut, QApplication, QAbstractItemView
import qdarkstyle
import cv2
import os
import sys
from pathlib import Path
from cell_points_marking.ControlStyle import (button_style, slider_style, tool_button_style, tool_bar_style, tab_widget_style,
                                              layer_list_style, fun_list_style, layer_widget_style, fun_widget_style,
                                              images_list_qdarkstyle, double_spin_qdarkstyle, dock_widget_qdarkstyle,
                                              table_widget_qdarkstyle, z_list_qdarkstyle, comboBox_style,
                                              spin_qdarkstyle)


# 枚举左上右下以及四个定点
Left, Top, Right, Bottom, LeftTop, RightTop, LeftBottom, RightBottom = range(8)

import tifffile
from cell_points_marking.InitializeInfo import InitializeInfo
from cell_points_marking.Ui_fun_Form import Ui_fun_Form
from bv.BVUtil import getHist


class MyListWidget(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

    def mousePressEvent(self, event: QMouseEvent):
        super().mousePressEvent(event)  # 调用基类的事件处理
        item = self.itemAt(event.pos())  # 获取被点击的项
        if item:
            self.parent.handle_mouse_event(item, event)  # 将事件传递给父窗口


class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    '''交互器类型事件重写'''
    def __init__(self, interactor, view):
        super().__init__()
        self.inter_actor = interactor
        self.view = view
        self.left_pressed = False
        self.right_pressed = False
        self.middle_pressed = False
        self.key_pressed = False
        self.shift_pressed = False
        self.ctrl_pressed = False
        self.AddObserver("CharEvent", self.on_char_event)
        self.AddObserver("MouseMoveEvent", self.on_mouse_move)
        self.AddObserver("KeyReleaseEvent", self.on_key_release)
        self.AddObserver("LeftButtonPressEvent", self.on_left_button_press)
        self.AddObserver("LeftButtonReleaseEvent", self.on_left_button_release)
        # self.AddObserver("RightButtonPressEvent", self.on_right_button_press)
        # self.AddObserver("RightButtonReleaseEvent", self.on_right_button_release)
        self.AddObserver("MiddleButtonPressEvent", self.on_middle_button_press)
        self.AddObserver("MiddleButtonReleaseEvent", self.on_middle_button_release)

    '''键盘事件'''
    def OnChar(self):  # 所有键盘按键功能失效
        # print("OnChar")
        super(CustomInteractorStyle, self).OnChar()
        pass

    '''鼠标右键按下事件'''
    def OnRightButtonDown(self):
        super(CustomInteractorStyle, self).OnRightButtonDown()
        pass

    '''重写鼠标右键'''
    def on_right_button_press(self, obj, event):  # 右键按下重写
        # print("右键按下")
        self.right_pressed = True

    def on_right_button_release(self, obj, event):
        self.right_pressed = False

    '''重写鼠标移动'''
    def on_mouse_move(self, obj, event):
        if self.right_pressed:
            pass
        else:
            super().OnMouseMove()
            self.left_pressed = False
            # if self.middle_pressed and self.view.dataImporter_index == 1:
            #     camera_focal_pos = self.view.camera.GetFocalPoint()
            #     print("焦点：", camera_focal_pos)

    '''重写键盘事件'''
    def on_char_event(self, obj, event):
        self.key_pressed = True
        key = self.GetInteractor().GetKeySym()
        # if key == "Alt_L" or key == "Alt_R":
        #     print("Alt")
        #     self.inter_actor.SetAltKey(1)
        if key == "Shift_L" or key == "Shift_R":
            self.shift_pressed = True
            # print("Shift")
        if key == "Control_L" or key == "Control_R":
            self.ctrl_pressed = True
            # print("Ctrl")

    def on_key_release(self, obj, event):
        # print("Released")
        # self.inter_actor.SetAltKey(0)
        self.key_pressed = False
        self.shift_pressed = False
        self.ctrl_pressed = False

    '''重写鼠标左键'''
    def on_left_button_press(self, obj, event):
        if self.shift_pressed or self.ctrl_pressed:  # 只保留左键按下后的移动事件，不响应键盘事件
            self.FindPokedRenderer(self.GetInteractor().GetEventPosition()[0],
                                   self.GetInteractor().GetEventPosition()[1])
            self.StartRotate()
            self.inter_actor.GetRenderWindow().Render()
        else:
            super().OnLeftButtonDown()
            if not self.key_pressed:
                self.left_pressed = True

    def on_left_button_release(self, obj, event):
        super().OnLeftButtonUp()
        # print("左键释放")
        if self.view.dataImporter_index and self.left_pressed:  # 是否生成球
            # 没有按下shift，ctrl，并且是修订模式
            if not self.shift_pressed and not self.ctrl_pressed:
                self.view.onSelectMarkingFun()

    '''重写鼠标中键'''
    def on_middle_button_press(self, obj, event):
        super().OnMiddleButtonDown()
        self.middle_pressed = True

    def on_middle_button_release(self, obj, event):
        super().OnMiddleButtonUp()
        self.middle_pressed = False


# from CustomTitleBar import CustomTitleBar
# from Ui_NoMenubarWidget import Ui_NoMenubarWidget
# from PyQt5.QtGui import QEnterEvent
# from PyQt5.QtWidgets import QSizePolicy


# class CustomMainWindow(QtWidgets.QMainWindow, Ui_NoMenubarWidget, Ui_fun_Form, InitializeInfo):
#     '''界面初始化'''
#     # 四周边距
#     Margins = 5
#     # 窗口内边距
#     Widget_Margins = 7
#
#     def __init__(self, parent=None):
#         super(CustomMainWindow, self).__init__(parent)
#         # Get Program Running Path
#         if getattr(sys, 'frozen', False):
#             self.base_path = sys._MEIPASS
#         else:
#             self.base_path = os.path.dirname(os.path.abspath(__file__))
#
#         # 样式
#         self.qdarkstyle_sheet = qdarkstyle.load_stylesheet(qt_api='pyqt5')
#         self._pressed = False
#         self.Direction = None
#         # self.setAttribute(Qt.WA_PaintOnScreen)
#         # self.setAttribute(Qt.WA_TranslucentBackground, True)
#         self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
#         self.setMouseTracking(True)
#
#         self.title_bar = CustomTitleBar(self)
#         self.setMenuWidget(self.title_bar)
#         # 布局
#         layout = QtWidgets.QVBoxLayout(self, spacing=0)
#         layout.setContentsMargins(
#             self.Margins, self.Margins, self.Margins, self.Margins)
#         # 信号槽
#         self.title_bar.windowMinimumed.connect(self.showMinimized)
#         self.title_bar.windowMaximumed.connect(self.showMaximized)
#         self.title_bar.windowNormaled.connect(self.showNormal)
#         self.title_bar.windowClosed.connect(self.close)
#         self.title_bar.windowMoved.connect(self.move)
#         self.windowTitleChanged.connect(self.title_bar.setTitle)
#         self.windowIconChanged.connect(self.title_bar.setIcon)
#
#         self.custom_tool_bar()
#         self.setupUi_win(self)
#         self.myStatus.setSizeGripEnabled(False)  # 禁用状态栏的大小调整功能
#         InitializeInfo.__init__(self)
#
#     # def setIconSize(self, size):
#     #     self.title_bar.setIconSize(size)
#         """定时检测键盘按键是否按下"""
#         self.timer = QTimer(self)
#         self.timer.timeout.connect(self.check_key_press)
#         self.timer.start(1000)  # 每1000毫秒触发一次
#
#     """定时检测键盘按键是否按下"""
#     def check_key_press(self):
#         if QApplication.queryKeyboardModifiers() == Qt.NoModifier:
#             # self.interactor_style.alt_pressed = False
#             self.interactor_style.key_pressed = False
#             # self.interactor.SetAltKey(0)
#
#     '''鼠标移动-窗口或控件移动'''
#     def move(self, pos):
#         if self.windowState() == Qt.WindowMaximized or self.windowState() == Qt.WindowFullScreen:
#             return
#         super(CustomMainWindow, self).move(pos)
#
#     '''窗口尺寸变化'''
#     def showMaximized(self):
#         """最大化,要去除上下左右边界,如果不去除则边框地方会有空隙"""
#         super(CustomMainWindow, self).showMaximized()
#         # Activate Window
#         self.raise_()
#         self.activateWindow()
#         self.layout().setContentsMargins(0, 0, 0, 0)
#
#     def showNormal(self):
#         """Restore,要保留上下左右边界,否则没有边框无法调整"""
#         super(CustomMainWindow, self).showNormal()
#         # Activate Window
#         self.raise_()
#         self.activateWindow()
#         self.layout().setContentsMargins(
#             self.Margins, self.Margins, self.Margins, self.Margins)
#
#     '''鼠标变化'''
#     def eventFilter(self, obj, event):
#         if isinstance(event, QEnterEvent):
#             self.setCursor(Qt.ArrowCursor)
#         return super(CustomMainWindow, self).eventFilter(obj, event)
#
#     '''鼠标按下'''
#     def mousePressEvent(self, event):
#         super(CustomMainWindow, self).mousePressEvent(event)
#         if event.button() == Qt.LeftButton:
#             self._mpos = event.pos()
#             self._pressed = True
#
#     '''鼠标释放'''
#     def mouseReleaseEvent(self, event):
#         super(CustomMainWindow, self).mouseReleaseEvent(event)
#         if self.Direction:
#             self.frame.setVisible(True)  # Displayvtk所在窗口
#         self._pressed = False
#         self.Direction = None
#
#     '''鼠标移动'''
#     def mouseMoveEvent(self, event):
#         """鼠标移动事件"""
#         super(CustomMainWindow, self).mouseMoveEvent(event)
#         pos = event.pos()
#         xPos, yPos = pos.x(), pos.y()
#         wm, hm = self.width() - self.Margins, self.height() - self.Margins
#         if self.isMaximized() or self.isFullScreen():
#             self.Direction = None
#             self.setCursor(Qt.ArrowCursor)
#             return
#         if event.buttons() == Qt.LeftButton and self._pressed:
#             self._resizeWidget(pos)
#             if self.Direction:
#                 self.frame.setVisible(False)  # Hidevtk所在窗口
#             return
#         if xPos <= self.Margins and yPos <= self.Margins:
#             # 左上角
#             self.Direction = LeftTop
#             self.setCursor(Qt.SizeFDiagCursor)
#         elif wm <= xPos <= self.width() and hm <= yPos <= self.height():
#             # 右下角
#             self.Direction = RightBottom
#             self.setCursor(Qt.SizeFDiagCursor)
#         elif wm <= xPos and yPos <= self.Margins:
#             # 右上角
#             self.Direction = RightTop
#             self.setCursor(Qt.SizeBDiagCursor)
#         elif xPos <= self.Margins and hm <= yPos:
#             # 左下角
#             self.Direction = LeftBottom
#             self.setCursor(Qt.SizeBDiagCursor)
#         elif 0 <= xPos <= self.Margins and self.Margins <= yPos <= hm:
#             # 左边
#             self.Direction = Left
#             self.setCursor(Qt.SizeHorCursor)
#         elif wm <= xPos <= self.width() and self.Margins <= yPos <= hm:
#             # 右边
#             self.Direction = Right
#             self.setCursor(Qt.SizeHorCursor)
#         elif self.Margins <= xPos <= wm and 0 <= yPos <= self.Margins:
#             # 上面
#             self.Direction = Top
#             self.setCursor(Qt.SizeVerCursor)
#         elif self.Margins <= xPos <= wm and hm <= yPos <= self.height():
#             # 下面
#             self.Direction = Bottom
#             self.setCursor(Qt.SizeVerCursor)
#         else:
#             self.Direction = None
#             self.setCursor(Qt.ArrowCursor)
#
#     '''窗口拉伸'''
#     def _resizeWidget(self, pos):
#         if self.Direction == None:
#             return
#         mpos = pos - self._mpos
#         xPos, yPos = mpos.x(), mpos.y()
#         geometry = self.geometry()
#         x, y, w, h = geometry.x(), geometry.y(), geometry.width(), geometry.height()
#         if self.Direction == LeftTop:  # 左上角
#             if w - xPos > self.minimumWidth():
#                 x += xPos
#                 w -= xPos
#             if h - yPos > self.minimumHeight():
#                 y += yPos
#                 h -= yPos
#         elif self.Direction == RightBottom:  # 右下角
#             if w + xPos > self.minimumWidth():
#                 w += xPos
#                 self._mpos = pos
#             if h + yPos > self.minimumHeight():
#                 h += yPos
#                 self._mpos = pos
#         elif self.Direction == RightTop:  # 右上角
#             if h - yPos > self.minimumHeight():
#                 y += yPos
#                 h -= yPos
#             if w + xPos > self.minimumWidth():
#                 w += xPos
#                 self._mpos.setX(pos.x())
#         elif self.Direction == LeftBottom:  # 左下角
#             if w - xPos > self.minimumWidth():
#                 x += xPos
#                 w -= xPos
#             if h + yPos > self.minimumHeight():
#                 h += yPos
#                 self._mpos.setY(pos.y())
#         elif self.Direction == Left:  # 左边
#             if w - xPos > self.minimumWidth():
#                 x += xPos
#                 w -= xPos
#             else:
#                 return
#         elif self.Direction == Right:  # 右边
#             if w + xPos > self.minimumWidth():
#                 w += xPos
#                 self._mpos = pos
#             else:
#                 return
#         elif self.Direction == Top:  # 上面
#             if h - yPos > self.minimumHeight():
#                 y += yPos
#                 h -= yPos
#             else:
#                 return
#         elif self.Direction == Bottom:  # 下面
#             if h + yPos > self.minimumHeight():
#                 h += yPos
#                 self._mpos = pos
#             else:
#                 return
#         self.setGeometry(x, y, w, h)
#
#     '''Toolbar'''
#     def custom_tool_bar(self):
#         self.tool_bar = QtWidgets.QToolBar("tool_bar", self)
#         self.clear_button = QtWidgets.QAction("&Clear", self)
#         self.clear_button.setIcon(QIcon(os.path.join(self.base_path, "icon_image", "clear.png")))
#         self.tool_xy_button = QtWidgets.QAction("&XY", self)
#         self.tool_xy_button.setIcon(QIcon(os.path.join(self.base_path, "icon_image", "XY.png")))
#         self.tool_xz_button = QtWidgets.QAction("&XZ", self)
#         self.tool_xz_button.setIcon(QIcon(os.path.join(self.base_path, "icon_image", "XZ.png")))
#         self.tool_yz_button = QtWidgets.QAction("&YZ", self)
#         self.tool_yz_button.setIcon(QIcon(os.path.join(self.base_path, "icon_image", "YZ.png")))
#
#         # spacer = QtWidgets.QWidget()
#         # spacer.setMaximumWidth(int(self.width() / 2))
#         # spacer.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
#         # self.tool_bar.addWidget(spacer)
#         # self.tool_bar.addSeparator()
#         self.tool_bar.addAction(self.tool_xy_button)
#         self.tool_bar.addAction(self.tool_xz_button)
#         self.tool_bar.addAction(self.tool_yz_button)
#         spacer_1 = QtWidgets.QWidget()
#         spacer_1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
#         self.tool_bar.addWidget(spacer_1)
#         self.tool_bar.addAction(self.clear_button)
#
#         self.tool_bar.setStyleSheet(tool_bar_style)
#         self.tool_bar.setMovable(False)
#         self.tool_bar.setFixedHeight(30)
#         # self.tool_bar.setMinimumHeight(30)
#         # self.tool_bar.setMaximumHeight(30)
#
#         self.addToolBar(self.tool_bar)
#
#     '''创建vtk界面'''
#     def create_vtk_view(self):
#         # QWidgetYesQFrame的父类，也就是说QFrameYesQWidget的一个子类。
#         # 这意味着QFrame继承了QWidget的所有属性和方法，并且可以使用QWidget中定义的所有功能。
#         # 同时，QFrame也有自己独特的功能和效果，例如边框、背景、阴影等
#         # self.frame = QtWidgets.QFrame()  # 创建子类窗口
#         # vl = QtWidgets.QVBoxLayout(self.frame)  # 垂直布局
#         # 窗口交互器
#         # 能捕捉渲染窗口中的鼠标和键盘事件，并将这些事件转变为对相机、演员和属性对象的相应操作，具体的转变由交互方式确定
#         self.vtkWidget = QVTKRenderWindowInteractor(self.frame)  # 在子类窗口下创建窗口交互器
#         self.frame_vl.addWidget(self.vtkWidget)  # 将交互器进行布局
#         self.frame_vl.setContentsMargins(0, 0, 0, 0)
#         # vl.addWidget(self.vtkWidget)  # 将交互器进行布局
#         # self.setCentralWidget(self.frame)  # 设置子类窗口为中心窗口
#         # vl.setContentsMargins(0, 0, 0, 0)
#         # self.menu_bar()  # 菜单栏函数
#         # self.setup_ui()  # 框架函数
#         # 绘制器，负责管理场景的渲染过程。
#         # 组成场景的所有对象包括Prop，照相机(Camera)和光照(Light)都被集中在一个vtkRenderer对象中。
#         # 一个vtkRenderWindow中可以有多个vtkRendererObject，而这些vtkRenderer可以渲染在窗口中不同的矩形区域中(即视口)，
#         # 或者覆盖整个窗口区域。
#         # 创建一个坐标轴对象
#         self.ren = vtk.vtkRenderer()  # 绘制器 renderer
#         # renWin 背景颜色
#         self.ren.SetBackground(0, 0, 0)  # 设置背景颜色为黑色
#         # self.ren.SetBackground(255, 255, 255)  # 设置背景颜色为白色
#         self.renWin = self.vtkWidget.GetRenderWindow()  # 场景
#         self.renWin.AddRenderer(self.ren)  # 在场景添加绘制器
#         self.renWin.SetDoubleBuffer(True)
#         self.interactor = self.renWin.GetInteractor()  # 交互器
#         # 创建一个 vtkRenderWindowInteractor
#         self.interactor.SetRenderWindow(self.renWin)  # 场景交互
#         # 当我们的mapper采样距离设置较低或者硬件性能不太好时，体渲染交互会有卡顿现象。
#         # 为了提高交互时的流畅性，可以设置交互器的SetDesiredUpdateRate来降低采样率进而避免卡顿现象。
#         # 当鼠标处于活动状态时，期望的渲染帧率会提高。当鼠标松开时，期望的渲染帧率会降回原来的值。
#         # 也就是图像进行旋转和缩放时会执行此操作。
#         '''当vtk鼠标处于活动状态时，期望的渲染帧率会提高是因为vtk会实时响应用户的交互操作，
#         并采用一些优化技术来提高渲染效率，从而使用户能够更流畅地观察到场景的变化'''
#         self.interactor.SetDesiredUpdateRate(0.2)  # vtk体渲染设置帧率
#         # 交互器样式的一种，该样式下，用户是通过控制相机对物体作旋转、Zoom in、缩小等操作
#         self.interactor_style = CustomInteractorStyle(self.interactor, self)
#         self.interactor_style.SetDefaultRenderer(self.ren)
#         self.interactor.SetInteractorStyle(self.interactor_style)
#         # 创建一个vtkCamera
#         self.camera = self.ren.GetActiveCamera()  # 获取渲染器的相机
#         self.cc_pos = self.camera.GetPosition()
#         self.cc_focal = self.camera.GetFocalPoint()
#
#         # 相机投影方式
#         # 透视投影：
#         # 透视投影是一种模拟人眼观察世界的投影方式，它会根据物体与相机的距离远近而产生近大远小的效果。
#         # 在VTK中，可以通过设置相机的视角、近裁剪面和远裁剪面等参数来实现透视投影。
#         #
#         # 正交投影： 正交投影是一种将物体投影到相机平面上时保持物体大小不变的投影方式，不受物体与相机距离的影响。
#         # 在VTK中，可以通过设置相机的正交投影模式和缩放因子等参数来实现正交投影。
#
#         # self.camera.SetParallelProjection(True)  # 启用正交投影，禁用透视投影
#
#         self.show()  # 窗口展示
#         # 启动事件循环--Start()
#         # 该方法表示开始进入事件响应循环，交互器处于等待状态，等待用户交互事件的发生。
#         # 一般在Start()前，先调用Initialize()Method
#         # 初始化interactior--Initialize()
#         # 如果没有初始化interactior，Start()将自动调用它，
#         # 但是如果需要在初始化和事件循环开始之间执行任何操作，则可以手动调用它开启交互模式
#         self.interactor.Initialize()
#
#     '''设置坐标系'''
#     def AxesWidgt(self):
#         # 创建一个vtkAxesActor
#         self.axes = vtk.vtkAxesActor()  # 空间坐标系对象
#         # 创建一个vtkOrientationMarkerWidget
#         self.orientation_marker_widget = vtk.vtkOrientationMarkerWidget()  # 方向标记组件
#         self.orientation_marker_widget.SetOrientationMarker(self.axes)  # 设置方向标记，需要传入vtkProp类型标记
#         self.orientation_marker_widget.SetInteractor(self.interactor)  # 设置交互器
#         self.orientation_marker_widget.SetViewport(-0.05, -0.05, 0.15, 0.15)
#         self.orientation_marker_widget.SetEnabled(True)  # 设置方向标记组件可用
#         self.orientation_marker_widget.InteractiveOff()  # 坐标系是否可移动
#         self.ren.AddViewProp(self.orientation_marker_widget.GetOrientationMarker())  # 将小部件添加到渲染器视图属性
#         self.ren.RemoveActor(self.axes)  # 从渲染器中去除对象坐标系
#
#     '''创建功能窗口及控件'''
#     def fun_widgets(self):
#         self.widgetFun = QtWidgets.QWidget()  # 设置一个子窗口，添加更多控件
#         self.setupUi_fun(self.widgetFun)
#         self.fun_from_hbox.setContentsMargins(self.Widget_Margins, self.Widget_Margins,
#                                               self.Widget_Margins, self.Widget_Margins)
#         self.z_step_num.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
#         self.z_redun_num.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
#         self.x_step_num.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
#         self.x_redun_num.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
#         self.y_step_num.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
#         self.y_redun_num.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
#         self.gray_min_num.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
#         self.gray_max_num.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
#         self.gray_auto_change_button.setText(f"{self.min_auto_gray} / {self.max_auto_gray}")
#         self.openColorDialog_button.setStyleSheet(
#             'QPushButton{ background-color: rgb(255, 255, 255); color: rgb(25, 35, 45); '
#             'border: 1px solid rgb(255, 255, 255); padding: 2px 2px; border-radius: 0px; min-width: 40px;}'
#             'QPushButton:hover { background-color: #379eff; }')
#         self.dialog_xyz_size_show_button.setStyleSheet(button_style)
#         self.gray_auto_change_button.setStyleSheet(button_style)
#         self.sphere_size_slider.setStyleSheet(slider_style)
#         self.gray_min_num.valueChanged.connect(self.change_color_opacity)
#         self.gray_max_num.valueChanged.connect(self.change_color_opacity)
#         self.gray_min_num.setValue(self.gray_min)
#         self.gray_max_num.setValue(self.gray_max)
#         # self.graphWidget_form.addRow(self.gray_label, graph_widget)  # 添加表格
#         self.function_change_button.setStyleSheet(button_style)
#         self.sphere_button.setStyleSheet("QPushButton {background-color: red; width: 20px; height: 20px; "
#                                          "border-radius: 0px;}")
#
#     '''创建初始活动窗口'''
#     def action_widgets(self):
#         self.images_dockWidget = QtWidgets.QDockWidget('images_widget')  # 浮动窗口，只能添加一个控件
#         self.images_dockWidget.setStyleSheet(dock_widget_qdarkstyle + self.qdarkstyle_sheet)
#
#         self.image_widget = QtWidgets.QWidget()  # 设置一个子窗口，添加更多控件
#         self.image_widget.setStyleSheet(
#             'QWidget {background-color: rgb(25, 35, 45);}'
#                                         )
#         self.tab_widget = QtWidgets.QTabWidget()  # 设置选项卡窗口
#         self.tab_widget.setStyleSheet(tab_widget_style)
#         self.tab_widget.setTabPosition(QtWidgets.QTabWidget.South)
#         self.tab_widget.setMovable(True)
#         self.tab_widget.currentChanged.connect(self.on_tab_changed)
#         # 设置列表项
#         self.img_list = QtWidgets.QListWidget()  # 设置列表项，展示文件夹文件
#         self.img_list.setStyleSheet(self.qdarkstyle_sheet)  # 筛选
#         self.img_list.setStyleSheet(None)
#         self.img_list.setStyleSheet(images_list_qdarkstyle)
#
#         # 禁用垂直和水平滚动条
#         # self.img_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 垂直
#         self.img_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
#
#         self.fun_dockWidget = QtWidgets.QDockWidget('fun_widget')  # 浮动窗口，只能添加一个控件
#         self.fun_dockWidget.setStyleSheet(dock_widget_qdarkstyle + self.qdarkstyle_sheet)
#         self.fun_widget = QtWidgets.QWidget()  # 设置一个子窗口，添加更多控件
#         self.fun_widget.setStyleSheet(fun_widget_style)
#         self.fun_stacked_widget = QtWidgets.QStackedWidget()  # 堆叠窗口
#         # self.fun_stacked_widget.setStyleSheet(
#         #     'QStackedWidget {font: 9pt "微软雅黑"; background-color: rgb(25, 35, 45); border-width: 1px; '
#         #     'border-style: solid; border-color: rgb(70, 80, 100); color: rgb(255, 255, 255);}'
#         # )
#         self.fun_list = QtWidgets.QListWidget()  # 设置列表项，展示标注
#         self.fun_list.setStyleSheet(fun_list_style)
#
#         # self.layer_dockWidget = QtWidgets.QDockWidget('cut_layer')
#         # self.layer_dockWidget.setStyleSheet(
#         #     'QDockWidget {font: 9pt "微软雅黑"; background-color: rgb(25, 35, 45); color: rgb(255, 255, 255);}'
#         #                                     )
#         self.layer_stacked_widget = QtWidgets.QStackedWidget()  # 堆叠窗口
#         self.layer_stacked_widget.setStyleSheet(
#             'QStackedWidget {font: 9pt "微软雅黑"; background-color: rgb(25, 35, 45); color: rgb(255, 255, 255);}'
#                                             )
#         self.layer_stacked_widget.setContentsMargins(2, 2, 2, 2)
#
#         self.tab_widget.addTab(self.image_widget, 'images_list')
#         self.tab_widget.addTab(self.layer_stacked_widget, 'cut_layer')
#
#         self.images_dockWidget.setWidget(self.image_widget)  # 在Dock窗口区域设置QWidget
#         # self.images_dockWidget.setWidget(self.tab_widget)  # 在Dock窗口区域设置QWidget
#         self.fun_dockWidget.setWidget(self.fun_stacked_widget)  # 浮动窗口添加堆叠窗口
#
#         image_vbox = QtWidgets.QVBoxLayout(self.image_widget)  # 垂直布局
#         image_vbox.addWidget(self.img_list)  # 添加控件
#         fun_vbox = QtWidgets.QVBoxLayout(self.fun_widget)  # 垂直布局
#         fun_vbox.addWidget(self.fun_list)  # 添加控件
#         image_vbox.setContentsMargins(self.Widget_Margins, self.Widget_Margins,
#                                       self.Widget_Margins, self.Widget_Margins)
#         fun_vbox.setContentsMargins(self.Widget_Margins, self.Widget_Margins,
#                                     self.Widget_Margins, self.Widget_Margins)
#
#         self.fun_stacked_widget.addWidget(self.fun_widget)  # 堆叠窗口添加窗口
#
#         # self.layer_dockWidget.setWidget(self.layer_stacked_widget)
#
#     def on_tab_changed(self):
#         text = self.tab_widget.tabText(self.tab_widget.currentIndex())
#         self.images_dockWidget.setWindowTitle(text)
#
#     '''直方图信息收集'''
#     def gray_widget_message(self):
#         self.range_min = self.img3d.min()
#         self.range_max = self.img3d.max()
#         hist = getHist(self.img3d)
#         # 创建两个滑动块的初始值
#         min_val = self.range_min
#         max_val = int(self.range_max / 2)
#         if max_val - min_val <= 5:
#             max_val = self.range_max
#             self.min_auto_gray = self.range_min
#             self.max_auto_gray = int(self.range_max / 4)
#         else:
#             max_index = np.argmax(hist[int(min_val)+5:max_val])
#             self.min_auto_gray = round((max_index + self.range_min) / 2)
#             self.max_auto_gray = round((max_index + self.range_min) * 13 / 2)
#             if self.max_auto_gray == self.min_auto_gray:
#                 self.min_auto_gray = self.range_min
#                 self.max_auto_gray = int(self.range_max / 4)
#         # if self.gray_inherit_index:  # 设置初始化时灰度值
#         self.gray_min = min_val
#         self.gray_max = max_val
#             # self.gray_inherit_index = False
#         return hist
#
#     def grayTransCallback(self):
#         minGray = self.grayWidget.getLeftValue()
#         maxGray = self.grayWidget.getRightValue()
#         self.gray_min_num.setValue(minGray)
#         self.gray_max_num.setValue(maxGray)
#         self.gray_min = minGray
#         self.gray_max = maxGray
#         # self.txtMinGray.setValue(minGray)
#         # self.txtMaxGray.setValue(maxGray)
#
#     '''VisualizationtifFile'''
#     @classmethod
#     def tif_to_view(cls, self):
#         # new_index = 0
#         if self.dataImporter_index:  # 存在可标签数据,Update
#             self.fun_stacked_widget.removeWidget(self.widgetFun)  # 更新操作栏
#             self.layer_stacked_widget.removeWidget(self.layer_cut_widget)
#             self.widgetFun.deleteLater()
#             self.layer_cut_widget.deleteLater()
#             if self.boxWidget:
#                 self.boxWidget.Off()
#                 del self.boxWidget
#
#             cls.update_initialize_info(self)
#
#             self.ren.RemoveAllViewProps()  # 去除所有渲染对象
#             # self.renWin.Render()  # 重新渲染
#             # new_index = 1
#         else:
#             self.camera.SetPosition(self.cc_pos)
#             self.camera.SetFocalPoint(self.cc_focal)
#             self.camera.SetViewUp(0, 1, 0)
#             # self.camera.SetClippingRange(-999999, 999999)
#             self.fun_stacked_widget.removeWidget(self.fun_widget)  # 更新操作栏
#
#         self.index = -1  # 更新选中的球体
#
#         index = self.img_list.currentRow()
#         self.img3d = tifffile.imread(self.image_file_list[index])  # 读取tifFile
#         self.max_shape = max(self.img3d.shape)
#         if len(self.img3d.shape) == 2:
#             # 在z方向上加一层，创建三维数组
#             # self.img3d = np.expand_dims(self.img3d, axis=0)
#             # 复制一份原始二维图像
#             image_copy = np.copy(self.img3d)
#             # 将复制的图像叠加在原始图像上，创建一个带有两层的三维数组
#             self.img3d = np.stack([self.img3d, image_copy], axis=0)
#
#         self.x_step, self.y_step, self.z_step = self.img3d.shape[2], self.img3d.shape[1], self.img3d.shape[0]
#         self.x_redun, self.y_redun, self.z_redun = 0, 0, 0
#
#         # 获取x轴层数
#         x_num = 1
#         s = self.x_step
#         while s < self.img3d.shape[2]:
#             x_num += 1
#             s = s + self.x_step - self.x_redun
#         self.x_num = x_num
#
#         # 获取y轴层数
#         y_num = 1
#         s = self.y_step
#         while s < self.img3d.shape[1]:
#             y_num += 1
#             s = s + self.y_step - self.y_redun
#         self.y_num = y_num
#
#         # 获取z轴层数
#         z_num = 1
#         s = self.z_step
#         while s < self.img3d.shape[0]:
#             z_num += 1
#             s = s + self.z_step - self.z_redun
#         self.z_num = z_num
#
#         self.r = self.g = self.b = 1  # Settingsrgb
#
#         self.DataImporter_create()
#
#         self.Outline_create()  # 包围盒创建函数
#
#         # 创建一个 vtkRenderer 和 vtkRenderWindow
#         # vtkLight vtkCamera
#         self.ren.AddVolume(self.volume)  # 添加演员
#         self.ren.AddActor(self.outlineActor)  # 添加包围盒
#         self.ren.ResetCameraClippingRange()  # 自动重设渲染范围
#         self.ren.ResetCamera()  # 自动设置相机
#         self.data_volume_show_hide_count = 0  # 图文件隐藏
#         self.outline_show_hide_count = 1  # 选择区域显示
#         self.volume.VisibilityOff()  # Hidevolume
#
#         self.object_message()  # 操作栏函数
#
#         if not self.eye_left:
#             # 添加左键点击观察者
#             # self.interactor.RemoveObserver(self.eye_left)
#             self.eye_left = self.interactor.AddObserver("LeftButtonPressEvent", self.select_actor)
#
#         if not self.eye_right:
#             # self.interactor.RemoveObserver(self.eye_right)  # 删除右键点击观察者
#             self.eye_right = self.interactor.AddObserver("RightButtonPressEvent", self.point_to_box)
#
#         # if not self.eye_mid:
#             # 添加左键点击观察者
#             # self.interactor.RemoveObserver(self.eye_left1)
#             # self.eye_mid = self.interactor.AddObserver("MiddleButtonPressEvent", self.point_to_box)
#
#         # 设置滑轮事件回调函数
#         # self.interactor.AddObserver("MouseWheelForwardEvent", self.mouseWheelForward)
#         # self.interactor.AddObserver("MouseWheelBackwardEvent", self.mouseWheelBackward)
#
#         self.change_z_step_redun()  # 渲染感兴趣区域
#
#         self.judge_actor()
#
#         self.dataImporter_index = True  # 创建成功
#
#         self.read_txt_file(index)
#
#         # if new_index:
#         #     print("修改焦距")
#         #     self.camera.SetPosition(self.inherit_pos)
#         #     self.camera.SetFocalPoint(self.inherit_focal)
#         #     self.camera.SetClippingRange(self.inherit_range)
#
#         self.renWin.Render()  # 重新渲染
#
#         self.selection = True


from cell_points_marking.Ui_CellRevisingWidget import Ui_CellRevisingWidget
from cell_points_marking.ShortCutDialog import ShortCutDialog
from cell_points_marking.Ui_layer_cut_Form import Ui_layer_cut_Form


class MyWidget(QWidget, Ui_CellRevisingWidget, Ui_fun_Form, Ui_layer_cut_Form, InitializeInfo):
    def __init__(self, parent=None):
        super(MyWidget, self).__init__(parent)
        # Get Program Running Path
        if getattr(sys, 'frozen', False):
            self.base_path = sys._MEIPASS
        else:
            self.base_path = os.path.dirname(os.path.abspath(__file__))

        # 样式
        self.qdarkstyle_sheet = qdarkstyle.load_stylesheet(qt_api='pyqt5')

        InitializeInfo.__init__(self)

        # 窗口内边距
        self.Widget_Margins = 0

        self.setupUi_win(self)
        self.resize(1600, 900)

        # 鼠标快捷方式
        self.mouse_short_cuts_dict = {
            "Mark": "Left click in revision mode (visualization area)",
            "Select label": "Left click in non-revision mode (visualization area)",
            "Delete label": "Right click (visualization area)",
            "Rotate image": "Left click and drag (visualization area)",
            "Pan image": "Middle click and drag (visualization area)",
            "Select filter": "Middle click on option in filter mode (image list)",
            "Cancel filter": "Right click on option in filter mode (image list)"
        }

        # 更新快捷方式
        self.short_cuts_dict = {
            "Open file": "Ctrl+Shift+Q",
            "Save file": "Ctrl+S",
            "Revision mode": "V",
            "Undo": "Ctrl+Z",
            # "Delete selected label": "Ctrl+D",
            "Clear labels": "C",
            "Toggle marking function": "M",
            "Show/hide labels": "Q",
            "Show/hide image block": "Ctrl+Q",
            # "Show/hide full image": "Shift+Q",
            "Auto adjust grayscale": "G",
            "Focus": "F",
            "Previous image": "A",
            "Next image": "D",
            "Previous block": "Shift+A",
            "Next block": "Shift+D",
            "Restore XY layer": "R",
            "Restore Z layer": "Ctrl+R",
            "Filter images": "Space"
        }

        self.shortcut_button()
        self.ShortCutDialog = ShortCutDialog(self)

        """定时检测键盘按键是否按下"""
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_key_press)
        self.timer.start(5000)  # 每1000毫秒触发一次

    def show(self):
        # super().show()
        # Activate Window
        self.raise_()
        self.activateWindow()
        # Bring Window to Front
        self.setWindowState(self.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)
        self.showNormal()

    """自定义QListWidget鼠标检测"""
    def handle_mouse_event(self, item, event: QMouseEvent):
        if self.filter_finish_button.isVisible():
            text = item.text()
            if event.button() == Qt.LeftButton:
                return
                # print(f"Left button clicked on item: {item.text()}")
            elif event.button() == Qt.MiddleButton:
                name = text.split("   ")[-1]
                if self.filter_res["filenames"].get(name) is None:
                    # print(f"Middle button clicked on item: {item.text()}")
                    self.filter_res["filenames"][name] = item
                    item.setForeground(QColor("yellow"))  # 字体颜色为黄色
                    if "√ " not in item.text():
                        item.setText("√ " + text)
                    self.filter_nums_label.setText(f"{len(self.filter_res['filenames'])}")
            elif event.button() == Qt.RightButton:
                name = text.split("   ")[-1]
                if self.filter_res["filenames"].get(name) is not None:
                    if "√ " in item.text():
                        text = text.replace("√ ", "")
                        item.setText(text)
                    # print(f"Right button clicked on item: {item.text()}")
                    self.filter_res["filenames"].pop(name, None)
                    self.filter_nums_label.setText(f"{len(self.filter_res['filenames'])}")

    """定时检测键盘按键是否按下"""

    def check_key_press(self):
        try:
            if QApplication.queryKeyboardModifiers() == Qt.NoModifier:
                # self.interactor_style.alt_pressed = False
                self.interactor_style.key_pressed = False
                # self.interactor.SetAltKey(0)
        except Exception as e:
            print(f"Periodic keyboard key press detection: {e}")

    def update_user_short_cut(self, short_cuts_dict):
        self.open_file_button.setShortcut(QKeySequence(short_cuts_dict['Open file']))
        self.save_pos_button.setShortcut(QKeySequence(short_cuts_dict['Save file']))
        self.start_mark_shortcut.setKey(QKeySequence(short_cuts_dict['Revision mode']))
        self.point_withdraw_shortcut.setKey(QKeySequence(short_cuts_dict['Undo']))
        # self.actor_delete_shortcut.setKey(QKeySequence(short_cuts_dict['Delete selected label']))
        self.all_actor_delete_shortcut.setKey(QKeySequence(short_cuts_dict['Clear labels']))
        self.change_function_shortcut.setKey(QKeySequence(short_cuts_dict['Toggle marking function']))
        self.sphere_quick_key_shortcut.setKey(QKeySequence(short_cuts_dict['Show/hide labels']))
        self.outline_show_hide_shortcut.setKey(QKeySequence(short_cuts_dict['Show/hide image block']))
        # self.data_volume_show_hide_shortcut.setKey(QKeySequence(short_cuts_dict['Show/hide full image']))
        self.gray_auto_change_shortcut.setKey(QKeySequence(short_cuts_dict['Auto adjust grayscale']))
        self.camera_focal_shortcut.setKey(QKeySequence(short_cuts_dict['Focus']))
        self.item_up_shortcut.setKey(QKeySequence(short_cuts_dict['Previous image']))
        self.item_down_shortcut.setKey(QKeySequence(short_cuts_dict['Next image']))
        self.box_to_up_shortcut.setKey(QKeySequence(short_cuts_dict['Previous block']))
        self.box_to_down_shortcut.setKey(QKeySequence(short_cuts_dict['Next block']))
        self.recovery_xy_shortcut.setKey(QKeySequence(short_cuts_dict['Restore XY layer']))
        self.recovery_z_shortcut.setKey(QKeySequence(short_cuts_dict['Restore Z layer']))
        self.filter_images_shortcut.setKey(QKeySequence(short_cuts_dict['Filter images']))

    def shortcut_button(self):
        self.set_button(self.clear_button, "Clear interface", "clear.png")
        self.set_button(self.projection_button, "perspective projection",
                        "perspective_projection.png")  # perspective projection / rectangular projection
        self.set_button(self.tool_xy_button, "XY plane projection", "XY.png")
        self.set_button(self.tool_xz_button, "XZ plane projection", "XZ.png")
        self.set_button(self.tool_yz_button, "YZ plane projection", "YZ.png")
        self.set_button(self.focal_button, "Mark focus", "focal.png")

        self.set_button(self.open_file_button, "Open file", "open_file.png")
        self.set_button(self.save_pos_button, "Save file", "save.png")
        self.set_button(self.set_function_button, "Function settings", "set.png")
        self.set_button(self.filter_button, "Start filtering", "filtering.png")
        self.set_button(self.filter_finish_button, "Finish filtering", "is_filtering.png")
        self.filter_finish_button.setVisible(False)

        self.set_button(self.start_mark_button, "Revision mode", "marking.png")
        self.set_button(self.clear_mark_button, "Clear marks", "clear_mark.png")
        # self.set_button(self.delete_mark_button, "Delete mark", "delete_mark.png")
        self.set_button(self.withdraw_button, "Undo", "withdraw.png")

        self.clear_button.setStyleSheet(tool_button_style)
        self.projection_button.setStyleSheet(tool_button_style)
        self.tool_xy_button.setStyleSheet(tool_button_style)
        self.tool_xz_button.setStyleSheet(tool_button_style)
        self.tool_yz_button.setStyleSheet(tool_button_style)
        self.focal_button.setStyleSheet(tool_button_style)
        self.open_file_button.setStyleSheet(tool_button_style)
        self.save_pos_button.setStyleSheet(tool_button_style)
        self.set_function_button.setStyleSheet(tool_button_style)
        self.filter_button.setStyleSheet(tool_button_style)
        self.filter_finish_button.setStyleSheet(tool_button_style)

        self.start_mark_button.setStyleSheet(tool_button_style)
        self.clear_mark_button.setStyleSheet(tool_button_style)
        # self.delete_mark_button.setStyleSheet(tool_button_style)
        self.withdraw_button.setStyleSheet(tool_button_style)

        self.open_file_button.setShortcut(QKeySequence(self.short_cuts_dict['Open file']))
        self.save_pos_button.setShortcut(QKeySequence(self.short_cuts_dict['Save file']))

        # 创建一个独立的快捷键
        self.start_mark_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Revision mode']), self)
        self.point_withdraw_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Undo']), self)
        # 修改快捷方式
        # self.point_withdraw_shortcut.setKey(QKeySequence('Ctrl+Z'))

        # self.actor_delete_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Delete selected label']), self)
        self.all_actor_delete_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Clear labels']), self)
        self.change_function_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Toggle marking function']), self)

        self.sphere_quick_key_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Show/hide labels']), self)
        self.outline_show_hide_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Show/hide image block']), self)
        # self.data_volume_show_hide_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Show/hide full image']), self)
        self.box_to_up_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Previous block']), self)
        self.box_to_down_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Next block']), self)
        self.gray_auto_change_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Auto adjust grayscale']), self)
        self.camera_focal_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Focus']), self)
        # self.sure_gray_auto_shortcut = QShortcut(QKeySequence('Ctrl+G'), self)
        self.item_up_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Previous image']), self)
        self.item_down_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Next image']), self)
        # self.box_show_hide_shortcut = QShortcut(QKeySequence('B'), self)
        # self.box_cut_label_shortcut = QShortcut(QKeySequence('Ctrl+B'), self)
        self.recovery_xy_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Restore XY layer']), self)
        self.recovery_z_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Restore Z layer']), self)
        self.filter_images_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Filter images']), self)

    def set_button(self, button, tool_tip, icon):
        button.setToolTip(str(tool_tip))
        button.setIcon(QIcon(os.path.join(self.base_path, "icon_image", str(icon))))
        button.setIconSize(QSize(26, 26))
        return button

    '''创建vtk界面'''

    def create_vtk_view(self):
        # QWidgetYesQFrame的父类，也就是说QFrameYesQWidget的一个子类。
        # 这意味着QFrame继承了QWidget的所有属性和方法，并且可以使用QWidget中定义的所有功能。
        # 同时，QFrame也有自己独特的功能和效果，例如边框、背景、阴影等
        # self.frame = QtWidgets.QFrame()  # 创建子类窗口
        # vl = QtWidgets.QVBoxLayout(self.frame)  # 垂直布局
        # 窗口交互器
        # 能捕捉渲染窗口中的鼠标和键盘事件，并将这些事件转变为对相机、演员和属性对象的相应操作，具体的转变由交互方式确定
        self.vtkWidget = QVTKRenderWindowInteractor(self.RevisingFrame)  # 在子类窗口下创建窗口交互器
        self.RevisingFrame_verticalLayout.addWidget(self.vtkWidget)  # 将交互器进行布局
        self.RevisingFrame_verticalLayout.setContentsMargins(0, 0, 0, 0)
        # vl.addWidget(self.vtkWidget)  # 将交互器进行布局
        # self.setCentralWidget(self.frame)  # 设置子类窗口为中心窗口
        # vl.setContentsMargins(0, 0, 0, 0)
        # self.menu_bar()  # 菜单栏函数
        # self.setup_ui()  # 框架函数
        # 绘制器，负责管理场景的渲染过程。
        # 组成场景的所有对象包括Prop，照相机(Camera)和光照(Light)都被集中在一个vtkRenderer对象中。
        # 一个vtkRenderWindow中可以有多个vtkRendererObject，而这些vtkRenderer可以渲染在窗口中不同的矩形区域中(即视口)，
        # 或者覆盖整个窗口区域。
        # 创建一个坐标轴对象
        self.ren = vtk.vtkRenderer()  # 绘制器 renderer
        # renWin 背景颜色
        self.ren.SetBackground(0, 0, 0)  # 设置背景颜色为黑色
        # self.ren.SetBackground(255, 255, 255)  # 设置背景颜色为白色
        self.renWin = self.vtkWidget.GetRenderWindow()  # 场景
        self.renWin.AddRenderer(self.ren)  # 在场景添加绘制器
        self.renWin.SetDoubleBuffer(True)
        self.interactor = self.renWin.GetInteractor()  # 交互器
        # 创建一个 vtkRenderWindowInteractor
        self.interactor.SetRenderWindow(self.renWin)  # 场景交互
        # 当我们的mapper采样距离设置较低或者硬件性能不太好时，体渲染交互会有卡顿现象。
        # 为了提高交互时的流畅性，可以设置交互器的SetDesiredUpdateRate来降低采样率进而避免卡顿现象。
        # 当鼠标处于活动状态时，期望的渲染帧率会提高。当鼠标松开时，期望的渲染帧率会降回原来的值。
        # 也就是图像进行旋转和缩放时会执行此操作。
        '''当vtk鼠标处于活动状态时，期望的渲染帧率会提高是因为vtk会实时响应用户的交互操作，
        并采用一些优化技术来提高渲染效率，从而使用户能够更流畅地观察到场景的变化'''
        # self.interactor.SetDesiredUpdateRate(0.2)  # vtk体渲染设置帧率
        self.interactor.SetDesiredUpdateRate(5)  # vtk体渲染设置帧率
        # 交互器样式的一种，该样式下，用户是通过控制相机对物体作旋转、Zoom in、缩小等操作
        self.interactor_style = CustomInteractorStyle(self.interactor, self)
        self.interactor_style.SetDefaultRenderer(self.ren)
        self.interactor.SetInteractorStyle(self.interactor_style)
        # 创建一个vtkCamera
        self.camera = self.ren.GetActiveCamera()  # 获取渲染器的相机
        self.cc_pos = self.camera.GetPosition()
        self.cc_focal = self.camera.GetFocalPoint()

        # 相机投影方式
        # 透视投影：
        # 透视投影是一种模拟人眼观察世界的投影方式，它会根据物体与相机的距离远近而产生近大远小的效果。
        # 在VTK中，可以通过设置相机的视角、近裁剪面和远裁剪面等参数来实现透视投影。
        #
        # 正交投影： 正交投影是一种将物体投影到相机平面上时保持物体大小不变的投影方式，不受物体与相机距离的影响。
        # 在VTK中，可以通过设置相机的正交投影模式和缩放因子等参数来实现正交投影。

        # self.camera.SetParallelProjection(True)  # 启用正交投影，禁用透视投影

        # self.show()  # 窗口展示
        # 启动事件循环--Start()
        # 该方法表示开始进入事件响应循环，交互器处于等待状态，等待用户交互事件的发生。
        # 一般在Start()前，先调用Initialize()Method
        # 初始化interactior--Initialize()
        # 如果没有初始化interactior，Start()将自动调用它，
        # 但是如果需要在初始化和事件循环开始之间执行任何操作，则可以手动调用它开启交互模式
        self.interactor.Initialize()
        self.interactor.Start()

    '''设置坐标系'''

    def AxesWidgt(self):
        # 创建一个vtkAxesActor
        self.axes = vtk.vtkAxesActor()  # 空间坐标系对象
        # 创建一个vtkOrientationMarkerWidget
        self.orientation_marker_widget = vtk.vtkOrientationMarkerWidget()  # 方向标记组件
        self.orientation_marker_widget.SetOrientationMarker(self.axes)  # 设置方向标记，需要传入vtkProp类型标记
        self.orientation_marker_widget.SetInteractor(self.interactor)  # 设置交互器
        self.orientation_marker_widget.SetViewport(-0.05, -0.05, 0.15, 0.15)
        self.orientation_marker_widget.SetEnabled(True)  # 设置方向标记组件可用
        self.orientation_marker_widget.InteractiveOff()  # 坐标系是否可移动
        self.ren.AddViewProp(self.orientation_marker_widget.GetOrientationMarker())  # 将小部件添加到渲染器视图属性
        self.ren.RemoveActor(self.axes)  # 从渲染器中去除对象坐标系

    '''创建功能窗口及控件'''

    def fun_widgets(self):
        self.widgetFun = QtWidgets.QWidget()  # 设置一个子窗口，添加更多控件
        self.setupUi_fun(self.widgetFun)
        self.fun_from_hbox.setContentsMargins(self.Widget_Margins, self.Widget_Margins,
                                              self.Widget_Margins, self.Widget_Margins)
        # 标记功能
        self.function_change_button.setStyleSheet(button_style)
        # Information
        # self.info_num_label
        # 渲染模式切换
        self.render_mode_combo.setStyleSheet(self.qdarkstyle_sheet + comboBox_style)
        # 标记渲染
        self.mark_render_combo.setStyleSheet(self.qdarkstyle_sheet + comboBox_style)
        # 标记颜色按钮
        # self.MarkColorDialog_button.setStyleSheet(
        #     'QPushButton{ background-color: rgb(0, 255, 0); color: rgb(25, 35, 45); '
        #     'border-radius: 0px; width: 20px; height: 20px;}'
        #     'QPushButton:hover { background-color: #379eff; }')
        # Resolution
        self.x_px_spinbox.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
        self.y_px_spinbox.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
        self.z_px_spinbox.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
        # Image、标记显示和隐藏
        self.image_button.setStyleSheet("QPushButton {background-color: red; width: 20px; height: 20px;"
                                         "border-radius: 0px;}")
        self.sphere_button.setStyleSheet("QPushButton {background-color: red; width: 20px; height: 20px;"
                                         "border-radius: 0px;}")
        # 筛选计数
        self.filter_nums_label.setStyleSheet("color: rgb(255, 255, 0); font-size: 10pt;")
        # 颜色按钮
        # self.openColorDialog_button.setStyleSheet(
        #     'QPushButton{ background-color: rgb(255, 255, 255); color: rgb(25, 35, 45); '
        #     'border-radius: 0px; width: 20px; height: 20px;}'
        #     'QPushButton:hover { background-color: #379eff; }')
        # 自动调节灰度
        self.gray_auto_change_button.setStyleSheet(button_style)
        # 计数
        self.nums_label.setStyleSheet("color: #4C9C4C; font-size: 10pt;")
        # 标记大小
        self.sphere_size_slider.setStyleSheet(slider_style)
        # 灰度调节
        # self.graphWidget_form
        self.gray_min_num.setStyleSheet(self.qdarkstyle_sheet + spin_qdarkstyle)
        self.gray_max_num.setStyleSheet(self.qdarkstyle_sheet + spin_qdarkstyle)
        # self.gray_min_num.valueChanged.connect(self.change_color_opacity)
        # self.gray_max_num.valueChanged.connect(self.change_color_opacity)

    """切块列表窗口"""

    def layer_widgets(self):
        self.layer_cut_widget = QtWidgets.QWidget()  # 设置一个子窗口，添加更多控件

        self.setupUi_layer(self.layer_cut_widget)
        self.layer_cut_Form_hbox.setContentsMargins(2, 2, 2, 2)

        self.z_list.setStyleSheet(self.qdarkstyle_sheet)  # 筛选样式
        self.z_list.setStyleSheet(None)
        self.z_list.setStyleSheet(z_list_qdarkstyle)
        self.z_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 禁用水平滑块

        self.table_widget.setStyleSheet(self.qdarkstyle_sheet)
        self.table_widget.setStyleSheet(None)
        self.table_widget.setStyleSheet(table_widget_qdarkstyle)
        self.table_widget.horizontalHeader().setVisible(False)
        self.table_widget.verticalHeader().setVisible(False)

        self.table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # Adaptive
        self.table_widget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents |
                                                                  QtWidgets.QHeaderView.Stretch)
        self.table_widget.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents |
                                                                QtWidgets.QHeaderView.Stretch)
        self.recovery_xy_button.setStyleSheet(button_style)
        self.recovery_z_button.setStyleSheet(button_style)
        self.remove_edge_button.setStyleSheet(button_style)
        # 步长和冗余
        self.z_step_num.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
        self.z_redun_num.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
        self.x_step_num.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
        self.x_redun_num.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
        self.y_step_num.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
        self.y_redun_num.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)

    '''创建初始活动窗口'''

    def action_widgets(self):
        """Image list"""
        self.images_dockWidget = QtWidgets.QDockWidget('Image List')  # 浮动窗口，只能添加一个控件
        self.images_dockWidget.setStyleSheet(dock_widget_qdarkstyle + self.qdarkstyle_sheet)

        self.image_widget = QtWidgets.QWidget()  # 设置一个子窗口，添加更多控件
        self.image_widget.setStyleSheet(
            'QWidget {background-color: rgb(25, 35, 45);}'
        )
        # self.tab_widget = QtWidgets.QTabWidget()  # 设置选项卡窗口
        # self.tab_widget.setStyleSheet(tab_widget_style)
        # self.tab_widget.setTabPosition(QtWidgets.QTabWidget.South)
        # self.tab_widget.setMovable(True)
        # self.tab_widget.currentChanged.connect(self.on_tab_changed)
        # 设置列表项
        # self.img_list = QtWidgets.QListWidget()  # 设置列表项，展示文件夹文件
        self.img_list = MyListWidget(self)  # 设置列表项，展示文件夹文件
        self.img_list.setStyleSheet(self.qdarkstyle_sheet)  # 筛选
        self.img_list.setStyleSheet(None)
        self.img_list.setStyleSheet(images_list_qdarkstyle)

        # 禁用垂直和水平滚动条
        # self.img_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 垂直
        self.img_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        """Function bar"""
        self.fun_dockWidget = QtWidgets.QDockWidget('Function Bar')  # 浮动窗口，只能添加一个控件
        self.fun_dockWidget.setStyleSheet(dock_widget_qdarkstyle + self.qdarkstyle_sheet)
        self.fun_widget = QtWidgets.QWidget()  # 设置一个子窗口，添加更多控件
        self.fun_widget.setStyleSheet(fun_widget_style)
        self.fun_stacked_widget = QtWidgets.QStackedWidget()  # 堆叠窗口
        self.fun_stacked_widget.setStyleSheet(
            "QStackedWidget {border: 1px solid rgb(70, 80, 100);}")
        self.fun_list = QtWidgets.QListWidget()  # 设置列表项，展示标注
        self.fun_list.setStyleSheet(fun_list_style)
        """Block list"""
        self.layer_dockWidget = QtWidgets.QDockWidget('Block List')
        self.layer_dockWidget.setStyleSheet(dock_widget_qdarkstyle + self.qdarkstyle_sheet)
        self.layer_widget = QtWidgets.QWidget()  # 设置一个子窗口，添加更多控件
        self.layer_widget.setStyleSheet(layer_widget_style)
        self.layer_stacked_widget = QtWidgets.QStackedWidget()  # 堆叠窗口
        self.layer_stacked_widget.setStyleSheet(
            "QStackedWidget {border: 1px solid rgb(70, 80, 100);}")
        self.layer_list = QtWidgets.QListWidget()  # 设置列表项，展示标注
        self.layer_list.setStyleSheet(layer_list_style)

        # self.tab_widget.addTab(self.image_widget, 'images_list')
        # self.tab_widget.addTab(self.layer_stacked_widget, 'cut_layer')

        image_vbox = QtWidgets.QVBoxLayout(self.image_widget)  # 垂直布局
        image_vbox.addWidget(self.img_list)  # 添加控件
        fun_vbox = QtWidgets.QVBoxLayout(self.fun_widget)  # 垂直布局
        fun_vbox.addWidget(self.fun_list)  # 添加控件
        layer_vbox = QtWidgets.QVBoxLayout(self.layer_widget)  # 垂直布局
        layer_vbox.addWidget(self.layer_list)

        image_vbox.setContentsMargins(self.Widget_Margins, self.Widget_Margins,
                                      self.Widget_Margins, self.Widget_Margins)
        fun_vbox.setContentsMargins(self.Widget_Margins, self.Widget_Margins,
                                    self.Widget_Margins, self.Widget_Margins)
        layer_vbox.setContentsMargins(self.Widget_Margins, self.Widget_Margins,
                                      self.Widget_Margins, self.Widget_Margins)
        self.fun_stacked_widget.addWidget(self.fun_widget)  # 堆叠窗口添加窗口
        self.layer_stacked_widget.addWidget(self.layer_widget)  # 堆叠窗口添加窗口

        self.images_dockWidget.setWidget(self.image_widget)  # 在Dock窗口区域设置QWidget
        # self.images_dockWidget.setWidget(self.tab_widget)  # 在Dock窗口区域设置QWidget
        self.fun_dockWidget.setWidget(self.fun_stacked_widget)  # 浮动窗口添加堆叠窗口
        self.layer_dockWidget.setWidget(self.layer_stacked_widget)

    def on_tab_changed(self):
        text = self.tab_widget.tabText(self.tab_widget.currentIndex())
        self.images_dockWidget.setWindowTitle(text)

    '''直方图信息收集'''

    def gray_widget_message(self):
        # startTime = time.time()
        self.range_min = self.img3d.min()
        self.range_max = self.img3d.max()
        hist = getHist(self.img3d)
        self.autoAdjustGray()
        # self.autoAdjustGrayPlus()
        # print(time.time() - startTime)
        return hist

    """简单下采样"""

    def simple_downsample(self, volume, factor):
        """
        简单下采样
        :param volume: 三维图像 (numpyArray)
        :param factor: 下采样因子 (整数)
        :return: 下采样后的三维图像
        """
        return volume[::factor, ::factor, ::factor]

    def _otsu_threshold(self, img_data: np.ndarray, min_val: int, max_val: int) -> int:
        """
        OTSU阈值算法

        Args:
            img_data: 图像数据
            min_val: 最小值
            max_val: 最大值

        Returns:
            阈值
        """
        import cv2

        if max_val > 255:
            # 归一化到0-255范围
            img_normalized = ((img_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            img_normalized = img_data.astype(np.uint8)

        # 使用OpenCV的OTSU阈值
        threshold, _ = cv2.threshold(img_normalized, min_val, 255, cv2.THRESH_OTSU)

        # 映射回原始范围
        if max_val > 255:
            threshold = threshold * (max_val - min_val) / 255 + min_val

        return int(threshold)

    """获取最优灰度"""

    def autoAdjustGray(self):
        limit_nums = 300 ** 3
        img = self.img3d
        self.min_auto_gray = 0
        self.max_auto_gray = 1
        if self.isReload:
            tImg = img[(img > 0)]
            if tImg.size < 2:
                return
            while len(tImg) > limit_nums:
                img = self.simple_downsample(img, 4)
                tImg = img[(img > 0)]
                if tImg.size < 2:
                    return
            grayMax = int(tImg.max())
            tMin = int(tImg.min())
            grayMin = self._otsu_threshold(tImg, tMin, grayMax)
            minInd = tImg < grayMin
            if minInd.sum() / tImg.size > 0.9:
                tImg = tImg[minInd]
                grayMin = self._otsu_threshold(tImg, tMin, grayMax)
            grayMax = min((grayMin + grayMax) / 2, grayMin * 2)
            if grayMax == 0:
                grayMax = 1
        else:
            grayMax = (self.MIPParam['minGray2'] + self.MIPParam['maxGray2']) / 2
            tImg = img[(img > 0) & (img < grayMax)]
            if tImg.size < 2:
                return
            while len(tImg) > limit_nums:
                img = self.simple_downsample(img, 4)
                tImg = img[(img > 0) & (img < grayMax)]
                if tImg.size < 2:
                    return
            tMin = int(tImg.min())
            grayMin = self._otsu_threshold(tImg, tMin, grayMax)
            grayMax = (grayMin + grayMax) / 2
            if grayMax == 0:
                grayMax = 1
        grayMax = int(grayMax)
        self.gray_min = self.range_min
        self.gray_max = grayMax
        # 新灰度调节参数
        self.MIPParam['minGray'] = self.range_min
        self.MIPParam['maxGray'] = grayMax
        self.MIPParam['minGray2'] = grayMin
        self.MIPParam['maxGray2'] = grayMax
        self.min_auto_gray = max(int(self.MIPParam['minGray']), self.range_min)
        self.max_auto_gray = min(int(self.MIPParam['maxGray']), self.range_max)

    def autoAdjustGrayPlus(self):
        img = self.img3d
        if self.isReload:
            tImg = img[(img > 0)]
            if tImg.size < 2:
                return

            grayMax = tImg.max()
            tMin = tImg.min()
            grayMin = cv2.threshold(tImg, tMin, grayMax, cv2.THRESH_OTSU)[0]
            minInd = tImg < grayMin
            if minInd.sum() / tImg.size > 0.9:
                tImg = tImg[minInd]
                grayMin = cv2.threshold(tImg, tMin, grayMax, cv2.THRESH_OTSU)[0]
            grayMax = min((grayMin + grayMax) / 2, grayMin * 2)
            # 同步界面，兼容之前版本
            self.gray_min = 0
            self.gray_max = grayMax
            # 新灰度调节参数
            self.MIPParam['minGray'] = 0
            self.MIPParam['maxGray'] = grayMax
            self.MIPParam['minGray2'] = grayMin
            self.MIPParam['maxGray2'] = grayMax
        else:
            grayMin = (self.MIPParam['minGray2'] + self.MIPParam['maxGray2']) / 2
            # grayMin = self.MIPParam['minGray2']
            tImg = img[(img > grayMin)]
            if tImg.size < 2: return
            grayMax = cv2.threshold(tImg, grayMin, tImg.max(), cv2.THRESH_OTSU)[0]
            grayMax = max((grayMin + grayMax) / 2,
                          self.MIPParam['maxGray2'] + (
                                  self.MIPParam['maxGray2'] - self.MIPParam['minGray2']) * 0.1)
            self.gray_min = 0
            self.gray_max = grayMax
            self.MIPParam['minGray'] = 0
            self.MIPParam['maxGray'] = grayMax
            self.MIPParam['minGray2'] = grayMin
            self.MIPParam['maxGray2'] = grayMax

        self.min_auto_gray = self.MIPParam['minGray2']
        self.max_auto_gray = self.MIPParam['maxGray2']

        if self.min_auto_gray == self.max_auto_gray and self.max_auto_gray == 0:
            self.MIPParam['maxGray'] = 1
            self.MIPParam['maxGray2'] = 1
            self.max_auto_gray = 1

        self.gray_min = self.min_auto_gray
        self.gray_max = self.max_auto_gray

    def grayTransCallback(self):
        minGray = self.grayWidget.getLeftValue()
        maxGray = self.grayWidget.getRightValue()
        self.gray_min_num.setValue(minGray)
        self.gray_max_num.setValue(maxGray)
        self.gray_min = minGray
        self.gray_max = maxGray
        # self.txtMinGray.setValue(minGray)
        # self.txtMaxGray.setValue(maxGray)

    '''VisualizationtifFile'''

    @classmethod
    def tif_to_view(cls, self):
        # 图像列表当前索引
        index = self.img_list.currentRow()
        img_path = self.image_file_list[index]
        if not os.path.exists(img_path):
            self.mess_set(f"{img_path} path does not exist, cannot read!", 'Prompt', 1)
            return

        if self.dataImporter_index:  # 存在可标签数据,Update
            if self.boxWidget:
                self.boxWidget.Off()
                del self.boxWidget
            # 更新参数
            cls.update_initialize_info(self)
            if self.reOpen:
                # 删除功能框
                self.fun_stacked_widget.removeWidget(self.widgetFun)
                # 删除切块列表
                self.layer_stacked_widget.removeWidget(self.layer_cut_widget)
                self.widgetFun.deleteLater()
                self.layer_cut_widget.deleteLater()
            # 去除所有渲染对象
            self.ren.RemoveAllViewProps()
        else:
            # 相机重新定位
            self.camera.SetPosition(self.cc_pos)
            self.camera.SetFocalPoint(self.cc_focal)
            self.camera.SetViewUp(0, 1, 0)
            # self.camera.SetClippingRange(-999999, 999999)
            # 删除功能框
            self.fun_stacked_widget.removeWidget(self.fun_widget)
            # 删除切块列表
            self.layer_stacked_widget.removeWidget(self.layer_widget)

        self.index = -1  # 更新选中的球体

        name_suffix = Path(img_path).suffix
        if name_suffix == '.bv':
            self.img3d = np.array(self.bvReader.readBV(img_path))  # 读取bvFile
        else:
            self.img3d = tifffile.imread(img_path)  # 读取tifFile

        # 仅在非 uint8 且非 uint16 时进行归一化
        if self.img3d.dtype not in (np.uint8, np.uint16):
            min_val = self.img3d.min()
            max_val = self.img3d.max()
            if min_val == max_val:
                # 防止除零：全图同值，直接置为0（或根据需要置为65535）
                self.img3d = np.zeros_like(self.img3d, dtype=np.uint16)
            else:
                # 转换为浮点数进行归一化，再缩放至[0, 65535]并转为uint16
                self.img3d = ((self.img3d.astype(np.float32) - min_val) /
                              (max_val - min_val) * 65535).astype(np.uint16)

        # 获取最大尺寸，用于点定位的间隔
        self.max_shape = max(self.img3d.shape)
        # 如果输入图像是二维
        if len(self.img3d.shape) == 2:
            # 在z方向上加一层，创建三维数组
            # self.img3d = np.expand_dims(self.img3d, axis=0)
            # 复制一份原始二维图像
            image_copy = np.copy(self.img3d)
            # 将复制的图像叠加在原始图像上，创建一个带有两层的三维数组
            self.img3d = np.stack([self.img3d, image_copy], axis=0)
        # 初始化步长和冗余
        if self.reOpen:  # 首次打开图像
            self.x_step, self.y_step, self.z_step = self.img3d.shape[2], self.img3d.shape[1], self.img3d.shape[0]
            self.x_redun, self.y_redun, self.z_redun = 0, 0, 0
        else:
            self.x_step = min(self.img3d.shape[2], int(self.x_step_num.value()))
            self.y_step = min(self.img3d.shape[1], int(self.y_step_num.value()))
            self.z_step = min(self.img3d.shape[0], int(self.z_step_num.value()))
            self.x_redun = max(0, self.x_redun_num.value())
            self.y_redun = max(0, self.y_redun_num.value())
            self.z_redun = max(0, self.z_redun_num.value())
        # 设置颜色
        if not self.dataImporter_index:
            self.r = self.g = self.b = 1  # Settingsrgb
        # 创建图像对象
        self.DataImporter_create()
        # 包围盒创建函数
        self.Outline_create()

        # vtkLight vtkCamera
        self.ren.AddVolume(self.volume)  # 添加演员
        self.ren.AddActor(self.outlineActor)  # 添加包围盒
        self.ren.ResetCameraClippingRange()  # 自动重设渲染范围
        self.ren.ResetCamera()  # 自动设置相机
        self.data_volume_show_hide_count = 0  # 图文件隐藏
        self.outline_show_hide_count = 1  # 选择区域显示
        # self.volume.VisibilityOff()  # Hidevolume
        self.outlineActor.VisibilityOff()  # HideoutlineActor
        self.ren.RemoveActor(self.volume)

        # 设置自定义功能函数
        self.object_message()
        # if not self.eye_left:
        #     # 添加左键点击观察者
        #     # self.interactor.RemoveObserver(self.eye_left)
        #     self.eye_left = self.interactor.AddObserver("LeftButtonPressEvent", self.select_actor)

        if not self.eye_right:
            # self.interactor.RemoveObserver(self.eye_right)  # 删除右键点击观察者
            self.eye_right = self.interactor.AddObserver("RightButtonPressEvent", self.del_actor)  # 选取标记

        # if not self.eye_right:
        #     # self.interactor.RemoveObserver(self.eye_right)  # 删除右键点击观察者
        #     self.eye_right = self.interactor.AddObserver("RightButtonPressEvent", self.point_to_box)  # 选取感兴趣区域

        if not self.eye_mid:
            # 添加左键点击观察者
            # self.interactor.RemoveObserver(self.eye_left1)
            self.eye_mid = self.interactor.AddObserver("MiddleButtonPressEvent", self.point_to_box)

        # 设置滑轮事件回调函数
        # self.interactor.AddObserver("MouseWheelForwardEvent", self.mouseWheelForward)
        # self.interactor.AddObserver("MouseWheelBackwardEvent", self.mouseWheelBackward)

        # 渲染感兴趣区域
        self.change_z_step_redun()
        self.outlineActor.VisibilityOn()
        # 删除多余的点线
        self.judge_actor()
        # 图像对象创建成功
        self.dataImporter_index = True
        # 读取选择文件的txtFile,渲染标签球体
        self.read_txt_file(index)
        # 重新渲染
        self.renWin.Render()
        # 图像选择框是否执行切换
        self.selection = True
        # 图像是否重新加载或切换
        self.isReload = False
