# -*- coding: utf-8 -*-
import qdarkstyle
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
from cell_points_marking.Ui_select_path_dialog import Ui_select_path_dialog
from cell_points_marking.Ui_filter_dialog import Ui_filter_dialog
from cell_points_marking.ControlStyle import button_style, lineedit_style, color_dialog_style


class CreateDialog(Ui_select_path_dialog, Ui_filter_dialog):
    '''创建窗口'''
    def __init__(self):
        super().__init__()
        # 样式
        self.qdarkstyle_sheet = qdarkstyle.load_stylesheet(qt_api='pyqt5')

    '''选择路径'''
    def create_path_select_dialog(self):
        self.select_path_dialog = QtWidgets.QDialog()
        self.setupUi(self.select_path_dialog)
        self.path_radioButton.setChecked(True)  # 默认激活
        self.path_widget.setVisible(True)
        self.config_widget.setVisible(False)  # 隐藏窗口
        self.name_widget.setVisible(False)  # 隐藏窗口
        self.path_radioButton.setStyleSheet(self.qdarkstyle_sheet)
        self.config_radioButton.setStyleSheet(self.qdarkstyle_sheet)
        self.name_radioButton.setStyleSheet(self.qdarkstyle_sheet)
        self.checkBox_zero.setStyleSheet(self.qdarkstyle_sheet)
        self.image_lineEdit.setStyleSheet(lineedit_style)
        self.swc_lineEdit.setStyleSheet(lineedit_style)
        self.config_lineEdit.setStyleSheet(lineedit_style)
        self.name_lineEdit.setStyleSheet(lineedit_style)
        # self.image_lineEdit.setReadOnly(True)  # 设置为只读模式
        # self.swc_lineEdit.setReadOnly(True)  # 设置为只读模式
        self.image_path_open_button.setStyleSheet(button_style)
        self.swc_path_open_button.setStyleSheet(button_style)
        self.config_path_open_Button.setStyleSheet(button_style)
        self.name_path_open_Button.setStyleSheet(button_style)
        # self.image_path_open_button.setMinimumWidth(80)
        # self.swc_path_open_button.setMinimumWidth(80)
        self.select_sure_button.setStyleSheet(button_style)
        self.select_cancel_button.setStyleSheet(button_style)
        # 使用setWindowModality方法设置对话框的模态性，可以防止对话框被其他窗口遮挡
        self.select_path_dialog.setWindowModality(Qt.ApplicationModal)

    '''筛选保存路径'''
    def create_filter_dialog(self):
        self.filter_dialog = QtWidgets.QDialog()
        self.setupUi_filter(self.filter_dialog)
        self.save_lineEdit.setStyleSheet(lineedit_style)
        self.save_lineEdit.setReadOnly(True)
        self.save_path_open_button.setStyleSheet(button_style)
        self.save_path_open_button.setMinimumWidth(80)
        self.filter_ok_button.setStyleSheet(button_style)
        self.filter_cancel_button.setStyleSheet(button_style)
        self.filter_dialog.setWindowModality(Qt.ApplicationModal)

    '''颜色'''
    @classmethod
    def create_color_dialog(cls, self):
        def update_color(color):
            self.openColorDialog_button.setStyleSheet("")  # 清除旧的样式表
            style_sheet = (
                f'QPushButton {{ background-color: {color.name()}; color: #19232d; '
                f'border-radius: 0px; width: 20px; height: 20px;}}'
                'QPushButton:hover { background-color: #379eff;}'
            )
            self.openColorDialog_button.setStyleSheet(style_sheet)

            self.r = color.red() / 255
            self.g = color.green() / 255
            self.b = color.blue() / 255

        if not self.color_dialog_index:  # 只生成一次颜色对话框
            self.color_dialog = QtWidgets.QColorDialog(self)  # Color dialog
            self.color_dialog.setStyleSheet(color_dialog_style)
            self.color_dialog.setOption(QtWidgets.QColorDialog.NoButtons)  # 设置关闭按钮为隐藏
            self.color_dialog_index = True
            self.color_dialog.currentColorChanged.connect(update_color)
            self.color_dialog.currentColorChanged.connect(self.change_color_opacity)

        if self.color_dialog.isHidden():
            self.color_dialog.show()

    '''标记颜色'''
    @classmethod
    def create_mark_color_dialog(cls, self):
        def update_color(color):
            self.MarkColorDialog_button.setStyleSheet("")  # 清除旧的样式表
            style_sheet = (
                f'QPushButton {{ background-color: {color.name()}; color: #19232d; '
                f'border-radius: 0px; width: 20px; height: 20px;}}'
                'QPushButton:hover { background-color: #379eff;}'
            )
            self.MarkColorDialog_button.setStyleSheet(style_sheet)

            self.mark_color = [color.red() / 255, color.green() / 255, color.blue() / 255]

        if not self.mark_color_dialog_index:  # 只生成一次颜色对话框
            self.mark_color_dialog = QtWidgets.QColorDialog(self)  # Color dialog
            self.mark_color_dialog.setStyleSheet(color_dialog_style)
            self.mark_color_dialog.setOption(QtWidgets.QColorDialog.NoButtons)  # 设置关闭按钮为隐藏
            self.mark_color_dialog_index = True
            self.mark_color_dialog.currentColorChanged.connect(update_color)
            self.mark_color_dialog.currentColorChanged.connect(self.change_mark_color)

        if self.mark_color_dialog.isHidden():
            self.mark_color_dialog.show()
