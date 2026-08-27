# -*- coding: utf-8 -*-
import time

import vtkmodules.all as vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QIcon, QKeySequence, QMouseEvent, QColor
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout,
                             QDockWidget, QListWidget, QStackedWidget, QShortcut)
import qdarkstyle
import os
import sys
from pathlib import Path
import cv2
from vessel_lines_marking.ControlStyle import (button_style, tool_bar_style,
                                               fun_list_style, fun_widget_style, branch_widget_style,
                                               branch_list_style, tool_button_style, branch_select_listwidget_qdarkstyle,
                                               images_list_qdarkstyle, double_spin_qdarkstyle, dock_widget_qdarkstyle,
                                               image_button_show_style, lines_button_show_style, label_count_style,
                                               dialog_button_style, filter_nums_label_style, spin_qdarkstyle,
                                               comboBox_style)


# 枚举左上右下以及四个定点
Left, Top, Right, Bottom, LeftTop, RightTop, LeftBottom, RightBottom = range(8)

import tifffile
from vessel_lines_marking.InitializeInfo import InitializeInfo
from vessel_lines_marking.Ui_fun_Form2 import Ui_fun_Form
from bv.BVUtil import getHist


class MyListWidget(QListWidget):
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
        self.interactor = interactor
        self.view = view
        self.left_pressed = False
        self.right_pressed = False
        self.middle_pressed = False
        self.key_pressed = False
        self.shift_pressed = False
        self.ctrl_pressed = False
        self.alt_pressed = False
        self.AddObserver("CharEvent", self.on_char_event)
        self.AddObserver("MouseMoveEvent", self.on_mouse_move)
        self.AddObserver("KeyReleaseEvent", self.on_key_release)
        self.AddObserver("LeftButtonPressEvent", self.on_left_button_press)
        self.AddObserver("LeftButtonReleaseEvent", self.on_left_button_release)
        self.AddObserver("RightButtonPressEvent", self.on_right_button_press)
        self.AddObserver("RightButtonReleaseEvent", self.on_right_button_release)
        self.AddObserver("MiddleButtonPressEvent", self.on_middle_button_press)
        self.AddObserver("MiddleButtonReleaseEvent", self.on_middle_button_release)

    '''键盘事件'''
    def OnChar(self):  # 所有键盘按键功能失效
        print("OnChar")
        super(CustomInteractorStyle, self).OnChar()
        pass

    '''鼠标右键按下事件'''
    def OnRightButtonDown(self):
        super(CustomInteractorStyle, self).OnRightButtonDown()
        pass

    '''重写鼠标右键'''
    def on_right_button_press(self, obj, event):  # 右键按下重写
        # print("右键按下")
        # if self.view.dataImporter_index:  # 是否生成
        #     if self.view.is_marking:
        #         self.view.onSelectMarkingFun(0)
        # self.right_pressed = True
        if self.shift_pressed or self.ctrl_pressed:  # 只保留左键按下后的移动事件，不响应键盘事件
            self.FindPokedRenderer(self.GetInteractor().GetEventPosition()[0],
                                   self.GetInteractor().GetEventPosition()[1])
            self.StartRotate()
            self.interactor.GetRenderWindow().Render()
        else:
            super().OnRightButtonDown()
            # if not self.key_pressed and not self.shift_pressed and not self.ctrl_pressed and not self.alt_pressed:
            self.right_pressed = True

    def on_right_button_release(self, obj, event):
        # self.right_pressed = False
        super().OnRightButtonUp()
        if self.view.dataImporter_index:  # 是否生成
            # print(f"左键释放 {self.left_pressed}\n")
            if self.right_pressed:
                if self.view.is_marking:
                    self.view.onSelectMarkingFun(0)

    '''重写鼠标移动'''
    def on_mouse_move(self, obj, event):
        if self.right_pressed:
            # pass
            super().OnMouseMove()
            self.right_pressed = False
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
        if key == "Alt_L" or key == "Alt_R":
            # print("Alt")
            self.alt_pressed = True
            self.interactor.SetAltKey(1)
        if key == "Shift_L" or key == "Shift_R":
            self.shift_pressed = True
            # print("Shift")
        elif key == "Control_L" or key == "Control_R":
            self.ctrl_pressed = True
            # print("Ctrl")

    """键盘按键释放"""

    def on_key_release(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        if key == "Control_L" or key == "Control_R":
            # print("CtrlReleased")
            self.ctrl_pressed = False
        elif key == "Shift_L" or key == "Shift_R":
            # print("ShiftReleased")
            self.shift_pressed = False
        elif key == "Alt_L" or key == "Alt_R":
            # print("AltReleased")
            self.alt_pressed = False
            self.interactor.SetAltKey(0)
            self.view.PloyData_connect_disconnect()
        else:
            self.key_pressed = False

    '''重写鼠标左键按下'''
    def on_left_button_press(self, obj, event):
        if self.shift_pressed or self.ctrl_pressed:  # 只保留左键按下后的移动事件，不响应键盘事件
            self.FindPokedRenderer(self.GetInteractor().GetEventPosition()[0],
                                   self.GetInteractor().GetEventPosition()[1])
            self.StartRotate()
            self.interactor.GetRenderWindow().Render()
        else:
            super().OnLeftButtonDown()
            # if not self.key_pressed and not self.shift_pressed and not self.ctrl_pressed and not self.alt_pressed:
            self.left_pressed = True

    """重写鼠标左键释放"""

    def on_left_button_release(self, obj, event):
        super().OnLeftButtonUp()
        if self.view.dataImporter_index:  # 是否生成
            # print(f"左键释放 {self.left_pressed}\n")
            if self.left_pressed:
                if self.view.is_marking:
                    self.view.onSelectMarkingFun(1)
                else:
                    if not self.interactor.GetAltKey():  # 如果alt没有按下
                        self.view.onSelectPolyData()

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
# from PyQt5.QtWidgets import QSizePolicy, QMainWindow, QToolBar, QAction


# class CustomMainWindow(QMainWindow, Ui_NoMenubarWidget, Ui_fun_Form, InitializeInfo):
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
#         layout = QVBoxLayout(self, spacing=0)
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
#     def show(self):
#         # super().show()
#         # Activate Window
#         self.raise_()
#         self.activateWindow()
#         # Bring Window to Front
#         self.setWindowState(self.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)
#         self.showNormal()
#
#     """定时检测键盘按键是否按下"""
#     def check_key_press(self):
#         if QApplication.queryKeyboardModifiers() == Qt.NoModifier:
#             self.interactor_style.alt_pressed = False
#             self.interactor.SetAltKey(0)
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
#         self.tool_bar = QToolBar("tool_bar", self)
#         self.clear_button = QAction("&Clear", self)
#         self.clear_button.setIcon(QIcon(os.path.join(self.base_path, "icon_image", "clear.png")))
#         self.tool_xy_button = QAction("&XY", self)
#         self.tool_xy_button.setIcon(QIcon(os.path.join(self.base_path, "icon_image", "XY.png")))
#         self.tool_xz_button = QAction("&XZ", self)
#         self.tool_xz_button.setIcon(QIcon(os.path.join(self.base_path, "icon_image", "XZ.png")))
#         self.tool_yz_button = QAction("&YZ", self)
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
#         spacer_1 = QWidget()
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
#         self.ren_v = vtk.vtkRenderer()  # 绘制器 renderer
#         self.ren = vtk.vtkRenderer()  # 绘制器 renderer
#         # renWin 背景颜色
#         # self.ren.SetBackground(0, 0, 0)  # 设置背景颜色为黑色
#         self.ren_v.SetBackground(0.05, 0.08, 0.1)  # 设置背景颜色为黑色
#         # self.ren_line.SetInteractive(0)  # 禁用交互
#         # self.ren_line.SetBackground(255, 255, 255)  # 设置背景颜色为白色
#         self.renWin = self.vtkWidget.GetRenderWindow()  # 场景
#         # # 设置渲染器的层次
#         self.ren_v.SetLayer(0)
#         self.ren.SetLayer(1)
#         # # 启用多层渲染
#         self.renWin.SetNumberOfLayers(2)
#         self.renWin.AddRenderer(self.ren_v)  # 在场景添加绘制器
#         self.renWin.AddRenderer(self.ren)  # 在场景添加绘制器
#
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
#         # 两个图层交互同步
#         self.ren_v.SetActiveCamera(self.camera)
#         self.ren.ResetCamera()
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
#         self.widgetFun = QWidget()  # 设置一个子窗口，添加更多控件
#         self.setupUi_fun(self.widgetFun)
#         self.fun_from_hbox.setContentsMargins(self.Widget_Margins, self.Widget_Margins,
#                                               self.Widget_Margins, self.Widget_Margins)
#         self.gray_min_num.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
#         self.gray_max_num.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
#         self.gray_auto_change_button.setText(f"{self.min_auto_gray} / {self.max_auto_gray}")
#         # self.openColorDialog_button.setStyleSheet(
#         #     'QPushButton{ background-color: rgb(255, 255, 255); color: rgb(25, 35, 45); '
#         #     'border: 1px solid rgb(255, 255, 255); padding: 2px 2px; border-radius: 0px;}'
#         #     'QPushButton:hover { background-color: #379eff; }')
#         # self.dialog_xyz_size_show_button.setStyleSheet(button_style)
#         self.gray_auto_change_button.setStyleSheet(button_style)
#         self.gray_min_num.valueChanged.connect(self.change_color_opacity)
#         self.gray_max_num.valueChanged.connect(self.change_color_opacity)
#         self.gray_min_num.setValue(self.gray_min)
#         self.gray_max_num.setValue(self.gray_max)
#         # self.graphWidget_form.addRow(self.gray_label, graph_widget)  # 添加表格
#         self.size_spinBox.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
#         self.length_spinBox.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
#         self.gray_limit_spinBox.setMaximum(self.img3d.max())
#         self.gray_limit_spinBox.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
#         self.centroid_spinBox.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
#         self.boxR_spinBox.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
#         self.step_spinBox.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
#
#     '''创建初始活动窗口'''
#     def action_widgets(self):
#         self.images_dockWidget = QDockWidget('images_widget')  # 浮动窗口，只能添加一个控件
#         self.images_dockWidget.setStyleSheet(dock_widget_qdarkstyle + self.qdarkstyle_sheet)
#
#         self.image_widget = QWidget()  # 设置一个子窗口，添加更多控件
#         self.image_widget.setStyleSheet(
#             'QWidget {background-color: rgb(25, 35, 45);}'
#                                         )
#         # 设置列表项
#         self.img_list = QListWidget()  # 设置列表项，展示文件夹文件
#         self.img_list.setStyleSheet(self.qdarkstyle_sheet)  # 筛选
#         self.img_list.setStyleSheet(None)
#         self.img_list.setStyleSheet(images_list_qdarkstyle)
#
#         # 禁用垂直和水平滚动条
#         # self.img_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 垂直
#         self.img_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
#
#         self.fun_dockWidget = QDockWidget('fun_widget')  # 浮动窗口，只能添加一个控件
#         self.fun_dockWidget.setStyleSheet(dock_widget_qdarkstyle + self.qdarkstyle_sheet)
#         self.fun_widget = QWidget()  # 设置一个子窗口，添加更多控件
#         self.fun_widget.setStyleSheet(fun_widget_style)
#         self.fun_stacked_widget = QStackedWidget()  # 堆叠窗口
#         self.fun_list = QListWidget()  # 设置列表项，展示标注
#         self.fun_list.setStyleSheet(fun_list_style)
#
#         self.images_dockWidget.setWidget(self.image_widget)  # 在Dock窗口区域设置QWidget
#         self.fun_dockWidget.setWidget(self.fun_stacked_widget)  # 浮动窗口添加堆叠窗口
#
#         image_vbox = QVBoxLayout(self.image_widget)  # 垂直布局
#         image_vbox.addWidget(self.img_list)  # 添加控件
#         fun_vbox = QVBoxLayout(self.fun_widget)  # 垂直布局
#         fun_vbox.addWidget(self.fun_list)  # 添加控件
#         image_vbox.setContentsMargins(self.Widget_Margins, self.Widget_Margins,
#                                       self.Widget_Margins, self.Widget_Margins)
#         fun_vbox.setContentsMargins(self.Widget_Margins, self.Widget_Margins,
#                                     self.Widget_Margins, self.Widget_Margins)
#
#         self.fun_stacked_widget.addWidget(self.fun_widget)  # 堆叠窗口添加窗口
#
#     def on_tab_changed(self):
#         text = self.tab_widget.tabText(self.tab_widget.currentIndex())
#         self.images_dockWidget.setWindowTitle(text)
#
#     '''直方图信息收集'''
#     def graywidget_message(self):
#         # 检查图像数据是否存在
#         if not hasattr(self, 'img3d') or self.img3d is None:
#             return None
#
#         # 获取图像的灰度范围
#         self.range_min = self.img3d.min()
#         self.range_max = self.img3d.max()
#
#         # 计算直方图
#         hist = getHist(self.img3d)
#
#         # 根据数据类型设置参数
#         rc, nc, kc = 100, 100, 0  # 默认参数
#         if self.img3d.dtype == np.uint8:
#             rc, nc, kc = 1, 10, 10
#         elif self.img3d.dtype == np.uint16:
#             if np.sum(hist[256:]) == 0:
#                 rc, nc, kc = 1, 10, 10
#                 hist = hist[:256]  # 只使用前256个值
#             else:
#                 rc, nc, kc = 200, 1000, 0
#
#         # 计算直方图最大值位置，增加安全检查避免索引越界
#         start_idx = max(self.range_min + rc, 0)
#         end_idx = min(rc * 200, len(hist))
#
#         if start_idx < end_idx:
#             max_index = np.argmax(hist[start_idx:end_idx]) + start_idx
#         else:
#             max_index = self.range_min
#
#         # 计算灰度滑动块的初始值
#         if self.range_min == self.range_max and self.range_min == 0:
#             min_val, max_val = 0, 1
#         else:
#             # 计算最小和最大值，确保在有效范围内
#             min_val = max(round((max_index + self.range_min) / 16), 0)
#             max_val_candidate = round((max_index + self.range_min) * 25 / 16) + kc
#             max_val = min(max_val_candidate, len(hist) - 1)
#
#         # 根据直方图峰值调整最大值
#         if len(hist) > max_val and hist[max_val] > 100 and self.range_max > nc:
#             max_val = min(max_val + rc * 5, len(hist) - 1)
#
#         # 设置自动灰度范围和当前灰度范围
#         self.min_auto_gray = min_val
#         self.max_auto_gray = max_val
#         self.gray_min = min_val
#         self.gray_max = max_val
#
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
#         # self.gray_limit = int(np.mean([self.gray_min, self.gray_max]))
#         # print("gray_limit: ", self.gray_limit)
#
#     '''VisualizationtifFile'''
#     @classmethod
#     def tif_to_view(cls, self):
#         # new_index = 0
#         if self.dataImporter_index:  # 存在可标签数据,Update
#             self.fun_stacked_widget.removeWidget(self.widgetFun)  # 更新操作栏
#             self.widgetFun.deleteLater()  # 正确删除self.widget并释放内存
#             cls.update_initialize_info(self)
#
#             self.ren_v.RemoveAllViewProps()  # 去除所有渲染对象
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
#         index = self.img_list.currentRow()
#         img3d_path = self.image_file_list[index]
#         name_suffix = Path(img3d_path).suffix
#         if name_suffix == '.bv':
#             self.img3d = np.array(self.bvReader.readBV(img3d_path))  # 读取bvFile
#         else:
#             self.img3d = tifffile.imread(img3d_path)  # 读取tifFile
#
#         # 按信号值从大到小排序
#         # signals = self.img3d.ravel()
#         # sorted_signals = np.sort(signals)[::-1]  # 降序排列
#         # # sorted_signals = np.sort(signals)  # 升序排列
#         # # 计算阈值（取信号值前10%的最小值）
#         # threshold_index = int(len(sorted_signals) * 0.044) - 1
#         # if threshold_index < 0:
#         #     threshold_index = 0
#         # self.gray_limit = sorted_signals[threshold_index]
#         # print("灰度：", self.gray_limit)
#
#         shape = self.img3d.shape
#         # self.focal_pos = [shape[2] / 2, shape[1] / 2, shape[0] / 2]
#
#         img0 = np.zeros(shape, dtype=str(self.img3d.dtype))
#         self.img3d_mark = img0
#
#         self.max_shape = max(shape)
#         if len(shape) == 2:
#             # 在z方向上加一层，创建三维数组
#             # self.img3d = np.expand_dims(self.img3d, axis=0)
#             # 复制一份原始二维图像
#             image_copy = np.copy(self.img3d)
#             # 将复制的图像叠加在原始图像上，创建一个带有两层的三维数组
#             self.img3d = np.stack([self.img3d, image_copy], axis=0)
#
#             img3d_mark_copy = np.copy(self.img3d_mark)
#             self.img3d_mark = np.stack([self.img3d_mark, img3d_mark_copy], axis=0)
#
#         self.r = self.g = self.b = 1  # Settingsrgb
#
#         self.DataImporter_create()
#
#         self.DataImporter_mark_create()
#
#         self.Outline_create()  # 包围盒创建函数
#
#         # self.sphere_create()  # 创建球体
#         # self.ren.AddActor(self.sphere)
#
#         # 创建一个 vtkRenderer 和 vtkRenderWindow
#         # vtkLight vtkCamera
#         self.ren_v.AddVolume(self.volume)  # 添加演员
#         self.ren_v.AddVolume(self.volume_mark)
#         self.ren.AddActor(self.outlineActor)  # 添加包围盒
#
#         self.volume_mark.VisibilityOff()
#         print(self.volume.GetVisibility())
#
#         self.ren.ResetCameraClippingRange()  # 自动重设渲染范围
#         # self.ren_v.SetActiveCamera(self.ren.GetActiveCamera())
#         self.ren.ResetCamera()  # 自动设置相机
#
#         self.object_message()  # 操作栏函数
#
#         if not self.eye_left:
#             # 添加左键点击观察者
#             # self.interactor.RemoveObserver(self.eye_left)
#             self.eye_left = self.interactor.AddObserver("LeftButtonPressEvent", self.LeftButtonPress)
#
#         if not self.eye_right:
#             # self.interactor.RemoveObserver(self.eye_right)  # 删除右键点击观察者
#             self.eye_right = self.interactor.AddObserver("RightButtonPressEvent", self.RightButtonPress)
#
#         if not self.eye_mid:
#             # 添加中键点击观察者
#             # self.interactor.RemoveObserver(self.eye_left1)
#             self.eye_mid = self.interactor.AddObserver("MiddleButtonPressEvent", self.MiddleButtonPress)
#
#         # 设置滑轮事件回调函数
#         # self.interactor.AddObserver("MouseWheelForwardEvent", self.mouseWheelForward)
#         # self.interactor.AddObserver("MouseWheelBackwardEvent", self.mouseWheelBackward)
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


from vessel_lines_marking.Ui_VesselRevisingWidget import Ui_VesselRevisingWidget
from vessel_lines_marking.ShortCutDialog import ShortCutDialog
from vessel_lines_marking.Ui_branch_select_Form import Ui_branch_select_Form


class MyWidget(QWidget, Ui_VesselRevisingWidget, Ui_fun_Form, Ui_branch_select_Form,
               InitializeInfo):
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
        self.resize(1500, 900)

        # 鼠标快捷方式
        self.mouse_short_cuts_dict = {
            "Mark": "Left click in revision mode (visualization area)",
            "Select branch": "Left click in non-revision mode (visualization area)",
            "Rotate image": "Left click and drag (visualization area)",
            "Pan image": "Middle click and drag (visualization area)",
            "Select filter": "Middle click on option in filter mode (image list)",
            "Cancel filter": "Right click on option in filter mode (image list)"
        }

        # 更新快捷方式
        self.short_cuts_dict = {
            "Open file": "Ctrl+Shift+Q",
            "Save file": "Shift+S",
            "Undo": "Ctrl+Z",
            "Undo pending branch": "Z",
            "Delete selected branch": "D",
            "Clear branches": "C",
            "Delete branches shorter than length limit": "Alt+D",
            "Confirm annotation": "Ctrl+S",
            "Show/hide image": "Q",
            "Show/hide branches": "A",
            "Toggle marking mode": "M",
            "Adjust optimal grayscale": "G",
            "Focus on branch point": "F",
            "Start revision mode": "V",
            "Close revision mode": "S",
            "Previous image": "Ctrl+W",
            "Next image": "Ctrl+X",
            "Show positioning box": "R",
            "Show/hide frame": "B",
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
                self.interactor_style.alt_pressed = False
                self.interactor_style.ctrl_pressed = False
                self.interactor_style.shift_pressed = False
                self.interactor_style.key_pressed = False
                self.interactor.SetAltKey(0)
                self.interactor.SetShiftKey(0)
                self.interactor.SetControlKey(0)
        except Exception as e:
            print(f"Periodic keyboard key press detection: {e}")

    def update_user_short_cut(self, short_cuts_dict):
        self.open_file_button.setShortcut(QKeySequence(short_cuts_dict['Open file']))
        self.save_pos_button.setShortcut(QKeySequence(short_cuts_dict['Save file']))

        self.point_withdraw_shortcut.setKey(QKeySequence(short_cuts_dict['Undo']))
        self.mark_withdraw_shortcut.setKey(QKeySequence(short_cuts_dict['Undo pending branch']))
        self.actor_delete_shortcut.setKey(QKeySequence(short_cuts_dict['Delete selected branch']))
        self.all_actor_delete_shortcut.setKey(QKeySequence(short_cuts_dict['Clear branches']))
        self.all_actor_limit_delete_shortcut.setKey(QKeySequence(short_cuts_dict['Delete branches shorter than length limit']))
        self.sure_marking_shortcut.setKey(QKeySequence(short_cuts_dict['Confirm annotation']))
        self.volume_change_shortcut.setKey(QKeySequence(short_cuts_dict['Show/hide image']))
        self.LP_quick_key_shortcut.setKey(QKeySequence(short_cuts_dict['Show/hide branches']))
        self.change_function_shortcut.setKey(QKeySequence(short_cuts_dict['Toggle marking mode']))
        self.gray_auto_change_shortcut.setKey(QKeySequence(short_cuts_dict['Adjust optimal grayscale']))
        self.camera_focal_shortcut.setKey(QKeySequence(short_cuts_dict['Focus on branch point']))
        self.start_marking_shortcut.setKey(QKeySequence(short_cuts_dict['Start revision mode']))
        self.end_marking_shortcut.setKey(QKeySequence(short_cuts_dict['Close revision mode']))
        self.item_up_shortcut.setKey(QKeySequence(short_cuts_dict['Previous image']))
        self.item_down_shortcut.setKey(QKeySequence(short_cuts_dict['Next image']))
        self.new_mark_create_shortcut.setKey(QKeySequence(short_cuts_dict['Show positioning box']))
        self.outline_quick_key_shortcut.setKey(QKeySequence(short_cuts_dict['Show/hide frame']))
        self.filter_images_shortcut.setKey(QKeySequence(short_cuts_dict['Filter images']))

    def shortcut_button(self):
        self.set_button(self.open_file_button, "Open file", "open_file.png")
        self.set_button(self.save_pos_button, "Save file", "save.png")
        self.set_button(self.set_function_button, "Function settings", "set.png")
        self.set_button(self.filter_button, "Start filtering", "filtering.png")
        self.set_button(self.filter_finish_button, "Finish filtering", "is_filtering.png")
        self.filter_finish_button.setVisible(False)
        self.set_button(self.tool_xy_button, "XY plane projection", "XY.png")
        self.set_button(self.tool_xz_button, "XZ plane projection", "XZ.png")
        self.set_button(self.tool_yz_button, "YZ plane projection", "YZ.png")
        self.set_button(self.marking_button, "Start revision mode", "marking.png")
        self.set_button(self.sure_marking_button, "Confirm annotation", "sure.png")
        self.set_button(self.delete_mark_button, "Delete selected branch", "delete_mark.png")
        self.set_button(self.delete_limit_button, "Delete branches shorter than length limit", "limit.png")
        self.set_button(self.clear_mark_button, "Clear branches", "clear_mark.png")
        self.set_button(self.withdraw_all_button, "Undo pending branch", "withdraw_all.png")
        self.set_button(self.withdraw_button, "Undo", "withdraw.png")
        self.set_button(self.focal_button, "Focus on branch point", "focal.png")
        self.set_button(self.clear_button, "Clear interface", "clear.png")
        self.set_button(self.projection_button, "perspective projection",
                        "perspective_projection.png")  # perspective projection / rectangular projection

        self.open_file_button.setStyleSheet(tool_button_style)
        self.save_pos_button.setStyleSheet(tool_button_style)
        self.set_function_button.setStyleSheet(tool_button_style)
        self.filter_button.setStyleSheet(tool_button_style)
        self.filter_finish_button.setStyleSheet(tool_button_style)
        self.tool_xy_button.setStyleSheet(tool_button_style)
        self.tool_xz_button.setStyleSheet(tool_button_style)
        self.tool_yz_button.setStyleSheet(tool_button_style)
        self.marking_button.setStyleSheet(tool_button_style)
        self.sure_marking_button.setStyleSheet(tool_button_style)
        self.delete_mark_button.setStyleSheet(tool_button_style)
        self.delete_limit_button.setStyleSheet(tool_button_style)
        self.clear_mark_button.setStyleSheet(tool_button_style)
        self.withdraw_button.setStyleSheet(tool_button_style)
        self.withdraw_all_button.setStyleSheet(tool_button_style)
        self.focal_button.setStyleSheet(tool_button_style)
        self.clear_button.setStyleSheet(tool_button_style)
        self.projection_button.setStyleSheet(tool_button_style)

        self.open_file_button.setShortcut(QKeySequence(self.short_cuts_dict['Open file']))
        self.save_pos_button.setShortcut(QKeySequence(self.short_cuts_dict['Save file']))

        # 创建一个独立的快捷键
        self.point_withdraw_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Undo']), self)
        # 修改快捷方式
        # self.point_withdraw_shortcut.setKey(QKeySequence('Ctrl+Z'))

        self.mark_withdraw_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Undo pending branch']), self)
        self.actor_delete_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Delete selected branch']), self)
        self.all_actor_delete_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Clear branches']), self)
        self.all_actor_limit_delete_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Delete branches shorter than length limit']), self)
        self.sure_marking_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Confirm annotation']), self)
        self.volume_change_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Show/hide image']), self)
        self.LP_quick_key_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Show/hide branches']), self)
        self.change_function_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Toggle marking mode']), self)
        self.gray_auto_change_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Adjust optimal grayscale']), self)
        self.camera_focal_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Focus on branch point']), self)
        self.start_marking_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Start revision mode']), self)
        self.end_marking_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Close revision mode']), self)
        self.item_up_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Previous image']), self)
        self.item_down_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Next image']), self)
        self.new_mark_create_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Show positioning box']), self)
        self.outline_quick_key_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Show/hide frame']), self)
        self.filter_images_shortcut = QShortcut(QKeySequence(self.short_cuts_dict['Filter images']), self)

        self.branch_forward_shortcut = QShortcut(QKeySequence("1"), self)
        self.branch_backward_shortcut = QShortcut(QKeySequence("2"), self)

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
        self.ren_v = vtk.vtkRenderer()  # 绘制器 renderer
        self.ren = vtk.vtkRenderer()  # 绘制器 renderer
        # renWin 背景颜色
        self.ren.SetBackground(0, 0, 0)  # 设置背景颜色为黑色
        # self.ren_v.SetBackground(0.05, 0.08, 0.1)  # 设置背景颜色为黑色
        # self.ren_line.SetInteractive(0)  # 禁用交互
        # self.ren_line.SetBackground(255, 255, 255)  # 设置背景颜色为白色
        self.renWin = self.vtkWidget.GetRenderWindow()  # 场景
        # # 设置渲染器的层次
        self.ren_v.SetLayer(0)
        self.ren.SetLayer(1)
        # # 启用多层渲染
        self.renWin.SetNumberOfLayers(2)
        self.renWin.AddRenderer(self.ren_v)  # 在场景添加绘制器
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
        # 交互器样式的一种，该样式下，用户是通过控制相机对物体作旋转、Zoom in、缩小等操作
        self.interactor_style = CustomInteractorStyle(self.interactor, self)
        self.interactor_style.SetDefaultRenderer(self.ren)
        self.interactor.SetInteractorStyle(self.interactor_style)
        # 设置拾取器
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.0005)  # 距离容差
        self.interactor.SetPicker(self.picker)
        # 创建一个vtkCamera
        self.camera = self.ren.GetActiveCamera()  # 获取渲染器的相机
        self.cc_pos = self.camera.GetPosition()
        self.cc_focal = self.camera.GetFocalPoint()
        # 两个图层交互同步
        self.ren_v.SetActiveCamera(self.camera)
        self.ren.ResetCamera()

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
        self.widgetFun = QWidget()  # 设置一个子窗口，添加更多控件
        self.setupUi_fun(self.widgetFun)
        self.fun_from_hbox.setContentsMargins(self.Widget_Margins, self.Widget_Margins,
                                              self.Widget_Margins, self.Widget_Margins)
        # 标记功能
        self.function_change_button.setStyleSheet(button_style)
        # Information
        # self.info_num_label
        # 渲染模式切换
        self.render_mode_combo.setStyleSheet(self.qdarkstyle_sheet + comboBox_style)
        # Image、分支显示和隐藏
        self.show_image_button.setStyleSheet(image_button_show_style)
        self.show_lines_button.setStyleSheet(lines_button_show_style)
        # 调节点数
        self.adjust_spinBox.setStyleSheet(self.qdarkstyle_sheet + spin_qdarkstyle)
        self.dialog_show_Button.setStyleSheet(dialog_button_style)
        # 灰度
        self.gray_auto_change_button.setStyleSheet(button_style)
        # 点大小
        self.size_spinBox.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
        # 长度限制
        self.length_spinBox.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
        # # 识别间距
        # self.step_spinBox.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
        # # 识别半径
        # self.centroid_spinBox.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
        # 框半径
        self.boxR_spinBox.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
        # # 灰度限制
        # self.gray_limit_spinBox.setMaximum(self.img3d.max())
        # self.gray_limit_spinBox.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
        # 分支数
        self.label_count.setStyleSheet(label_count_style)
        # 筛选数
        self.filter_nums_label.setStyleSheet(filter_nums_label_style)
        # 灰度调节
        # self.graphWidget_form
        self.gray_min_num.setStyleSheet(self.qdarkstyle_sheet + spin_qdarkstyle)
        self.gray_max_num.setStyleSheet(self.qdarkstyle_sheet + spin_qdarkstyle)
        # self.gray_min_num.valueChanged.connect(self.change_color_opacity)
        # self.gray_max_num.valueChanged.connect(self.change_color_opacity)
        # self.gray_min_num.setValue(self.gray_min)
        # self.gray_max_num.setValue(self.gray_max)

    """分支点选择列表窗口"""

    def branch_widgets(self):
        self.branch_select_widget = QWidget()  # 设置一个子窗口，添加更多控件

        self.setupUi_branch(self.branch_select_widget)
        self.branch_select_Form_hbox.setContentsMargins(2, 2, 2, 2)

        self.branch_select_listwidget.setStyleSheet(self.qdarkstyle_sheet)  # 筛选样式
        self.branch_select_listwidget.setStyleSheet(None)
        self.branch_select_listwidget.setStyleSheet(branch_select_listwidget_qdarkstyle)
        self.branch_select_listwidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 禁用水平滑块

        self.branch_sure_button.setStyleSheet(button_style)
        self.branch_close_button.setStyleSheet(button_style)
        self.branch_forward_Button.setStyleSheet(button_style)
        self.branch_backward_Button.setStyleSheet(button_style)

    '''创建初始活动窗口'''

    def action_widgets(self):
        self.images_dockWidget = QDockWidget('Image List')  # 浮动窗口，只能添加一个控件
        self.images_dockWidget.setStyleSheet(dock_widget_qdarkstyle + self.qdarkstyle_sheet)

        self.image_widget = QWidget()  # 设置一个子窗口，添加更多控件
        self.image_widget.setStyleSheet(
            'QWidget {background-color: rgb(25, 35, 45);}'
        )
        # 设置列表项
        # self.img_list = QListWidget()  # 设置列表项，展示文件夹文件
        self.img_list = MyListWidget(self)  # 设置列表项，展示文件夹文件
        self.img_list.setStyleSheet(self.qdarkstyle_sheet)  # 筛选
        self.img_list.setStyleSheet(None)
        self.img_list.setStyleSheet(images_list_qdarkstyle)

        # 禁用垂直和水平滚动条
        # self.img_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 垂直
        self.img_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        """功能栏"""
        self.fun_dockWidget = QDockWidget('Function Bar')  # 浮动窗口，只能添加一个控件
        self.fun_dockWidget.setStyleSheet(dock_widget_qdarkstyle + self.qdarkstyle_sheet)
        self.fun_widget = QWidget()  # 设置一个子窗口，添加更多控件
        self.fun_widget.setStyleSheet(fun_widget_style)
        self.fun_stacked_widget = QStackedWidget()  # 堆叠窗口
        self.fun_list = QListWidget()  # 设置列表项，展示标注
        self.fun_list.setStyleSheet(fun_list_style)
        """分支点列表"""
        self.branch_dockWidget = QDockWidget('Node List')
        self.branch_dockWidget.setStyleSheet(dock_widget_qdarkstyle + self.qdarkstyle_sheet)
        self.branch_widget = QWidget()  # 设置一个子窗口，添加更多控件
        self.branch_widget.setStyleSheet(branch_widget_style)
        self.branch_stacked_widget = QStackedWidget()  # 堆叠窗口
        self.branch_list = QListWidget()  # 设置列表项，展示标注
        self.branch_list.setStyleSheet(branch_list_style)
        """交叉调节栏"""

        image_vbox = QVBoxLayout(self.image_widget)  # 垂直布局
        image_vbox.addWidget(self.img_list)  # 添加控件
        fun_vbox = QVBoxLayout(self.fun_widget)  # 垂直布局
        fun_vbox.addWidget(self.fun_list)  # 添加控件
        branch_vbox = QVBoxLayout(self.branch_widget)  # 垂直布局
        branch_vbox.addWidget(self.branch_list)

        image_vbox.setContentsMargins(self.Widget_Margins, self.Widget_Margins,
                                      self.Widget_Margins, self.Widget_Margins)
        fun_vbox.setContentsMargins(self.Widget_Margins, self.Widget_Margins,
                                    self.Widget_Margins, self.Widget_Margins)
        branch_vbox.setContentsMargins(self.Widget_Margins, self.Widget_Margins,
                                       self.Widget_Margins, self.Widget_Margins)

        self.fun_stacked_widget.addWidget(self.fun_widget)  # 堆叠窗口添加窗口
        self.branch_stacked_widget.addWidget(self.branch_widget)  # 堆叠窗口添加窗口

        self.images_dockWidget.setWidget(self.image_widget)  # 在Dock窗口区域设置QWidget
        self.fun_dockWidget.setWidget(self.fun_stacked_widget)  # 浮动窗口添加堆叠窗口
        self.branch_dockWidget.setWidget(self.branch_stacked_widget)  # 浮动窗口添加堆叠窗口

    '''直方图信息收集'''

    def gray_widget_message(self):
        # startTime = time.time()
        self.range_min = int(self.img3d.min())
        self.range_max = int(self.img3d.max())
        hist = getHist(self.img3d)
        self.autoAdjustGray()
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

    def grayTransCallback(self):
        minGray = self.grayWidget.getLeftValue()
        maxGray = self.grayWidget.getRightValue()
        self.gray_min_num.setValue(minGray)
        self.gray_max_num.setValue(maxGray)
        self.gray_min = minGray
        self.gray_max = maxGray
        # self.txtMinGray.setValue(minGray)
        # self.txtMaxGray.setValue(maxGray)
        # self.gray_limit = int(np.mean([self.gray_min, self.gray_max]))
        # print("gray_limit: ", self.gray_limit)

    '''VisualizationtifFile'''

    @classmethod
    def tif_to_view(cls, self):
        index = self.img_list.currentRow()
        img3d_path = self.image_file_list[index]

        if not os.path.exists(img3d_path):
            self.mess_set(f"{img3d_path} path does not exist, cannot read!", "Prompt", 1)
            return

        if self.dataImporter_mark is not None:
            del self.dataImporter_mark
        if self.volume_mark is not None:
            del self.volume_mark

        # new_index = 0
        if self.dataImporter_index:  # 存在可标签数据,Update
            if self.reOpen:
                # 删除功能栏
                self.fun_stacked_widget.removeWidget(self.widgetFun)
                self.widgetFun.deleteLater()  # 正确删除并释放内存
                # 删除分支点列表
                self.branch_stacked_widget.removeWidget(self.branch_select_widget)
                self.branch_select_widget.deleteLater()  # 正确删除并释放内存
            # 更新参数
            cls.update_initialize_info(self)
            self.ren_v.RemoveAllViewProps()  # 去除所有渲染对象
            self.ren.RemoveAllViewProps()  # 去除所有渲染对象
        else:
            self.camera.SetPosition(self.cc_pos)
            self.camera.SetFocalPoint(self.cc_focal)
            self.camera.SetViewUp(0, 1, 0)
            # self.camera.SetClippingRange(-999999, 999999)
            # 删除功能栏
            self.fun_stacked_widget.removeWidget(self.fun_widget)  # 更新操作栏
            # 删除分支点列表
            self.branch_stacked_widget.removeWidget(self.branch_widget)

        self.update_tool_button()

        # index = self.img_list.currentRow()
        # img3d_path = self.image_file_list[index]
        name_suffix = Path(img3d_path).suffix
        if name_suffix == '.bv':
            self.img3d = np.array(self.bvReader.readBV(img3d_path))  # 读取bvFile
        else:
            self.img3d = tifffile.imread(img3d_path)  # 读取tifFile

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

        shape = self.img3d.shape
        # self.focal_pos = [shape[2] / 2, shape[1] / 2, shape[0] / 2]
        # img0 = np.zeros(shape, dtype=str(self.img3d.dtype))
        # self.img3d_mark = img0

        self.max_shape = max(shape)
        if len(shape) == 2:
            # 在z方向上加一层，创建三维数组
            # self.img3d = np.expand_dims(self.img3d, axis=0)
            # 复制一份原始二维图像
            image_copy = np.copy(self.img3d)
            # 将复制的图像叠加在原始图像上，创建一个带有两层的三维数组
            self.img3d = np.stack([self.img3d, image_copy], axis=0)
            # img3d_mark_copy = np.copy(self.img3d_mark)
            # self.img3d_mark = np.stack([self.img3d_mark, img3d_mark_copy], axis=0)

        self.r = self.g = self.b = 1  # Settingsrgb

        self.DataImporter_create()
        # self.DataImporter_mark_create()
        self.Outline_create()  # 包围盒创建函数

        self.ren_v.AddVolume(self.volume)  # 添加演员
        # self.ren_v.AddVolume(self.volume_mark)
        self.ren.AddActor(self.outlineActor)  # 添加包围盒

        # self.volume_mark.VisibilityOff()
        # print(self.volume.GetVisibility())

        self.ren.ResetCameraClippingRange()  # 自动重设渲染范围
        # self.ren_v.SetActiveCamera(self.ren.GetActiveCamera())
        self.ren.ResetCamera()  # 自动设置相机
        # 自定义功能函数
        self.object_message()

        if not self.eye_left:
            # 添加左键点击观察者
            # self.interactor.RemoveObserver(self.eye_left)
            self.eye_left = self.interactor.AddObserver("LeftButtonPressEvent", self.LeftButtonPress)

        if not self.eye_right:
            # self.interactor.RemoveObserver(self.eye_right)  # 删除右键点击观察者
            self.eye_right = self.interactor.AddObserver("RightButtonPressEvent", self.RightButtonPress)

        if not self.eye_mid:
            # 添加中键点击观察者
            # self.interactor.RemoveObserver(self.eye_left1)
            self.eye_mid = self.interactor.AddObserver("MiddleButtonPressEvent", self.MiddleButtonPress)

        # 设置滑轮事件回调函数
        # self.interactor.AddObserver("MouseWheelForwardEvent", self.mouseWheelForward)
        # self.interactor.AddObserver("MouseWheelBackwardEvent", self.mouseWheelBackward)
        # 删除多余的点线
        self.judge_actor()
        # 图像对象创建成功
        self.dataImporter_index = True
        # 读取选择文件的txtFile,渲染标签
        # start_time = time.time()
        self.read_txt_file(index)
        # print("Read time: ", time.time() - start_time)

        # if new_index:
        #     print("修改焦距")
        #     self.camera.SetPosition(self.inherit_pos)
        #     self.camera.SetFocalPoint(self.inherit_focal)
        #     self.camera.SetClippingRange(self.inherit_range)
        # 重新渲染
        self.renWin.Render()
        # 图像选择框是否执行切换
        self.selection = True
        # 图像是否重新加载或切换
        self.isReload = False
