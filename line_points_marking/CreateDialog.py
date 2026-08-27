# -*- coding: utf-8 -*-
import qdarkstyle
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
from line_points_marking.Ui_select_path_dialog import Ui_select_path_dialog
from line_points_marking.Ui_filter_dialog import Ui_filter_dialog
from line_points_marking.Ui_select_radius_dialog import Ui_select_radius_dialog
from line_points_marking.ControlStyle import button_style, lineedit_style, spin_qdarkstyle, double_spin_qdarkstyle
import pyqtgraph as pg


class CreateDialog(Ui_select_path_dialog, Ui_filter_dialog, Ui_select_radius_dialog):

    """创建窗口"""

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

    '''半径调节器'''

    def create_radius_dialog(self):
        self.radius_dialog = QtWidgets.QDialog()
        self.setupUi_r(self.radius_dialog)

        self.x_resolution_doubleSpinBox.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
        self.y_resolution_doubleSpinBox.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
        self.z_resolution_doubleSpinBox.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)

        self.res_save_Button.setStyleSheet(button_style)
        self.res_save_Button.setEnabled(False)

        self.forward_spinBox.setStyleSheet(self.qdarkstyle_sheet + spin_qdarkstyle)
        self.forward_spinBox.setValue(0)
        self.forward_spinBox.setMinimum(0)
        self.forward_spinBox.setMaximum(49)

        self.backward_spinBox.setStyleSheet(self.qdarkstyle_sheet + spin_qdarkstyle)
        self.backward_spinBox.setValue(101)
        self.backward_spinBox.setMinimum(51)
        self.backward_spinBox.setMaximum(101)

        # 1. 画布
        self.pw = pg.PlotWidget()
        self.pw.setLabel('left', 'Gray')
        self.pw.setLabel('bottom', 'Index')
        self.verticalLayout_direction.addWidget(self.pw)
        # 2. 两条可拖动垂线
        pg.InfiniteLine(pos=50, angle=90, movable=False,
                        pen=pg.mkPen('r', width=3))
        self.forw_line = pg.InfiniteLine(pos=0, angle=90, movable=True,
                                         pen=pg.mkPen('y', width=6))
        self.center_line = pg.InfiniteLine(pos=50, angle=90, movable=False,
                                         pen=pg.mkPen('c', width=3))
        self.back_line = pg.InfiniteLine(pos=101, angle=90, movable=True,
                                         pen=pg.mkPen('g', width=6))
        self.pw.addItem(self.forw_line)
        self.pw.addItem(self.center_line)
        self.pw.addItem(self.back_line)
        # 禁止拖放、左下角按钮、右键菜单
        self.pw.setMouseEnabled(x=False, y=False)
        self.pw.setAntialiasing(True)
        self.pw.setMenuEnabled(False)
        self.pw.hideButtons()
        # self.pw.hide()

        self.radius_dialog.setWindowFlags(
            Qt.Tool | Qt.WindowStaysOnTopHint
            # Qt.Tool | Qt.WindowStaysOnTopHint | Qt.CustomizeWindowHint
        )

        self.active_line = self.forw_line
