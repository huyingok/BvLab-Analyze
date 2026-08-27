# -*- coding: utf-8 -*-
from PyQt5.QtGui import QColor, QIcon, QFont
from PyQt5.QtCore import pyqtSignal
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt, QPoint
from control_style.ControlStyle import menubar_style, menu_style, titleLabel_style
import os
import sys


class CustomTitleBar(QWidget):
    '''自定义菜单栏'''
    windowMinimumed = pyqtSignal()
    windowMaximumed = pyqtSignal()
    windowNormaled = pyqtSignal()
    windowClosed = pyqtSignal()
    windowMoved = pyqtSignal(QPoint)

    def __init__(self, parent=None):
        super(CustomTitleBar, self).__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        # Get Program Running Path
        if getattr(sys, 'frozen', False):
            self.base_path = sys._MEIPASS
        else:
            self.base_path = os.path.dirname(os.path.abspath(__file__))

        self.mPos = None
        self.maxP = None
        # self.max_view = True
        self.max_view = False
        self.iconSize = 28
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(palette.Window, QColor(240, 240, 240))
        self.setPalette(palette)
        layout = QHBoxLayout(self, spacing=0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setObjectName("menubar")
        self.menubar.setStyleSheet(menubar_style)
        # self.menubar.setMinimumWidth(35)
        # self.menubar.setMinimumHeight(30)
        # self.menubar.setMaximumHeight(38)

        self.menubar_message()

        self.iconLabel = QLabel(self)
        layout.addWidget(self.iconLabel)
        layout.addWidget(self.menubar)
        layout.addSpacerItem(QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum))
        self.titleLabel = QLabel(self)
        self.titleLabel.setStyleSheet(titleLabel_style)
        self.titleLabel.setMargin(2)
        layout.addWidget(self.titleLabel)
        layout.addSpacerItem(QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum))
        font = self.font() or QFont()
        font.setFamily('Webdings')
        # 最小化按钮
        # self.buttonMinimum = QPushButton(
        #     '0', self, clicked=self.windowMinimumed.emit, font=font, objectName='buttonMinimum')
        self.buttonMinimum = QPushButton(self, clicked=self.windowMinimumed.emit,
                                         font=font, objectName='buttonMinimum')
        self.buttonMinimum.setIcon(QIcon(os.path.join(self.base_path, "icon_image", "minimize.png")))
        layout.addWidget(self.buttonMinimum)
        # 最大化/还原按钮
        # self.buttonMaximum = QPushButton(
        #     '1', self, clicked=self.showMaximized, font=font, objectName='buttonMaximum')
        self.buttonMaximum = QPushButton(self, clicked=self.showMaximized,
                                         font=font, objectName='buttonMaximum')
        self.buttonMaximum.setIcon(QIcon(os.path.join(self.base_path, "icon_image", "maximize.png")))
        # self.buttonMaximum.setIcon(QIcon(os.path.join(self.base_path, "icon_image", "restore.png")))
        layout.addWidget(self.buttonMaximum)
        # 关闭按钮
        # self.buttonClose = QPushButton(
        #     'r', self, clicked=self.windowClosed.emit, font=font, objectName='buttonClose')
        self.buttonClose = QPushButton(self, clicked=self.windowClosed.emit,
                                       font=font, objectName='buttonClose')
        self.buttonClose.setIcon(QIcon(os.path.join(self.base_path, "icon_image", "close.png")))
        layout.addWidget(self.buttonClose)
        layout.setStretch(1, 1)
        layout.setStretch(2, 5)
        layout.setStretch(3, 6)
        # 初始高度
        self.setHeight()

    '''Menu bar'''
    def menubar_message(self):
        self.file = QtWidgets.QMenu(self.menubar)
        self.file.setStyleSheet(menu_style)
        self.file.setObjectName("menu")
        self.file.setTitle('File')
        self.menubar.addMenu(self.file)
        self.open_file_act = QtWidgets.QAction('Open File', self.file)
        self.open_file_act.setShortcut("Ctrl+Shift+Q")
        self.open_file_act.setStatusTip('>>>Open file')
        self.save_pos_act = QtWidgets.QAction('Save File', self.file)
        self.save_pos_act.setShortcut("Ctrl+S")
        self.save_pos_act.setStatusTip('>>>Save file')
        self.file.addAction(self.open_file_act)
        # self.file.addSeparator()
        self.file.addAction(self.save_pos_act)

        self.edit = self.menubar.addMenu('')
        self.images_list_act = QtWidgets.QAction(self.edit)  # 浮动窗口1
        self.images_list_act.setShortcut("Ctrl+1")
        self.funs_widget_act = QtWidgets.QAction(self.edit)  # 浮动窗口2
        self.funs_widget_act.setShortcut("Ctrl+2")
        self.edit.addAction(self.images_list_act)
        self.edit.addAction(self.funs_widget_act)

        self.hide_bar = self.menubar.addMenu("")
        self.point_withdraw_act = QtWidgets.QAction(self.hide_bar)
        self.point_withdraw_act.setShortcut('Ctrl+Z')
        self.actor_delete_act = QtWidgets.QAction(self.hide_bar)
        self.actor_delete_act.setShortcut('Ctrl+X')
        self.all_actor_delete_act = QtWidgets.QAction(self.hide_bar)
        self.all_actor_delete_act.setShortcut('Alt+X')
        self.change_function_act = QtWidgets.QAction(self.hide_bar)
        self.change_function_act.setShortcut('M')
        self.sphere_quick_key_act = QtWidgets.QAction(self.hide_bar)
        self.sphere_quick_key_act.setShortcut('Q')
        self.outline_show_hide_act = QtWidgets.QAction(self.hide_bar)
        self.outline_show_hide_act.setShortcut('Ctrl+Q')
        self.data_volume_show_hide_act = QtWidgets.QAction(self.hide_bar)
        self.data_volume_show_hide_act.setShortcut('Shift+Q')
        self.box_to_up_act = QtWidgets.QAction(self.hide_bar)
        self.box_to_up_act.setShortcut('D')
        self.box_to_down_act = QtWidgets.QAction(self.hide_bar)
        self.box_to_down_act.setShortcut('A')
        self.gray_auto_change_act = QtWidgets.QAction(self.hide_bar)
        self.gray_auto_change_act.setShortcut('G')
        self.camera_focal_act = QtWidgets.QAction(self.hide_bar)
        self.camera_focal_act.setShortcut('F')
        self.sure_gray_auto_act = QtWidgets.QAction(self.hide_bar)
        self.sure_gray_auto_act.setShortcut('Ctrl+G')

        self.gray_left_act = QtWidgets.QAction(self.hide_bar)
        self.gray_left_act.setShortcut('1')
        self.gray_right_act = QtWidgets.QAction(self.hide_bar)
        self.gray_right_act.setShortcut('2')

        self.item_up_act = QtWidgets.QAction(self.hide_bar)
        # self.item_up_act.setShortcut(Qt.Key_Up)
        self.item_up_act.setShortcut('Alt+W')
        self.item_down_act = QtWidgets.QAction(self.hide_bar)
        # self.item_down_act.setShortcut(Qt.Key_Down)
        self.item_down_act.setShortcut('Alt+S')

        self.box_x_to_down_act = QtWidgets.QAction(self.hide_bar)
        self.box_x_to_down_act.setShortcut('J')
        self.box_x_to_add_act = QtWidgets.QAction(self.hide_bar)
        self.box_x_to_add_act.setShortcut('L')
        self.box_y_to_add_act = QtWidgets.QAction(self.hide_bar)
        self.box_y_to_add_act.setShortcut('I')
        self.box_y_to_down_act = QtWidgets.QAction(self.hide_bar)
        self.box_y_to_down_act.setShortcut('K')
        self.box_show_hide_act = QtWidgets.QAction(self.hide_bar)
        self.box_show_hide_act.setShortcut('B')
        self.box_cut_label_act = QtWidgets.QAction(self.hide_bar)
        self.box_cut_label_act.setShortcut('Ctrl+B')
        self.recovery_xy_act = QtWidgets.QAction(self.hide_bar)
        self.recovery_xy_act.setShortcut('R')
        self.recovery_z_act = QtWidgets.QAction(self.hide_bar)
        self.recovery_z_act.setShortcut('Ctrl+R')

        self.hide_bar.addAction(self.point_withdraw_act)
        self.hide_bar.addAction(self.actor_delete_act)
        self.hide_bar.addAction(self.all_actor_delete_act)
        self.hide_bar.addAction(self.change_function_act)
        self.hide_bar.addAction(self.sphere_quick_key_act)
        self.hide_bar.addAction(self.outline_show_hide_act)
        self.hide_bar.addAction(self.data_volume_show_hide_act)
        self.hide_bar.addAction(self.box_to_up_act)
        self.hide_bar.addAction(self.box_to_down_act)
        self.hide_bar.addAction(self.gray_auto_change_act)
        self.hide_bar.addAction(self.camera_focal_act)
        self.hide_bar.addAction(self.sure_gray_auto_act)
        self.hide_bar.addAction(self.gray_left_act)
        self.hide_bar.addAction(self.gray_right_act)
        self.hide_bar.addAction(self.item_up_act)
        self.hide_bar.addAction(self.item_down_act)
        self.hide_bar.addAction(self.box_x_to_down_act)
        self.hide_bar.addAction(self.box_x_to_add_act)
        self.hide_bar.addAction(self.box_y_to_add_act)
        self.hide_bar.addAction(self.box_y_to_down_act)
        self.hide_bar.addAction(self.box_show_hide_act)
        self.hide_bar.addAction(self.box_cut_label_act)
        self.hide_bar.addAction(self.recovery_xy_act)
        self.hide_bar.addAction(self.recovery_z_act)

    '''最大化'''
    def showMaximized(self):
        # if self.buttonMaximum.text() == '1':
        #     self.buttonMaximum.setText('2')
        #     self.windowMaximumed.emit()
        # else:
        #     self.buttonMaximum.setText('1')
        #     self.windowNormaled.emit()
        if not self.max_view:
            self.windowMaximumed.emit()
            self.buttonMaximum.setIcon(QIcon(os.path.join(self.base_path, "icon_image", "restore.png")))
            self.max_view = True
        else:
            self.windowNormaled.emit()
            self.buttonMaximum.setIcon(QIcon(os.path.join(self.base_path, "icon_image", "maximize.png")))
            self.max_view = False

    '''菜单栏高度'''
    def setHeight(self, height=38):
        self.setMinimumHeight(height)
        self.setMaximumHeight(height)
        self.buttonMinimum.setMinimumSize(height, height)
        self.buttonMinimum.setMaximumSize(height, height)
        self.buttonMaximum.setMinimumSize(height, height)
        self.buttonMaximum.setMaximumSize(height, height)
        self.buttonClose.setMinimumSize(height, height)
        self.buttonClose.setMaximumSize(height, height)

    '''菜单栏标题'''
    def setTitle(self, title):
        self.titleLabel.setText(title)

    '''菜单栏图标'''
    def setIcon(self, icon):
        self.iconLabel.setPixmap(icon.pixmap(self.iconSize, self.iconSize))

    '''菜单栏标题大小'''
    def setIconSize(self, size):
        self.iconSize = size

    '''鼠标样式'''
    def enterEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        super(CustomTitleBar, self).enterEvent(event)

    '''鼠标双击'''
    def mouseDoubleClickEvent(self, event):
        super(CustomTitleBar, self).mouseDoubleClickEvent(event)
        self.showMaximized()

    '''鼠标按下'''
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mPos = event.pos()
        event.accept()

    '''鼠标释放'''
    def mouseReleaseEvent(self, event):
        self.mPos = None
        if self.maxP:
            self.showMaximized()
            self.maxP = None
        event.accept()

    '''鼠标移动'''
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.mPos:
            self.windowMoved.emit(self.mapToGlobal(event.pos() - self.mPos))

            screen_geometry = QApplication.desktop().availableGeometry()
            if event.globalY() <= screen_geometry.top():
                self.maxP = True
            # if self.buttonMaximum.text() == '2':
            #     self.showMaximized()
            if self.max_view:
                self.showMaximized()
        event.accept()
