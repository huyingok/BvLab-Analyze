import sys

from PyQt5.uic.properties import QtWidgets

from DataStatistics.ui_create_chart import Ui_MainWindow
from UI.Ui_Train_Param_Form import Ui_Train_Param_Form
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QSplitter, QMessageBox, QWidget, QApplication, QListWidget
from PyQt5.QtGui import QIcon
import qdarkstyle
from control_style.ControlStyle import (button_alpha_style, button_style, lineedit_style, spin_style,
                                        double_spin_qdarkstyle, messagebox_style, comboBox_style, tool_button_style)


class TrainParamWidget(QWidget, Ui_Train_Param_Form):
    def __init__(self, parent=None):
        super(TrainParamWidget, self).__init__(parent)

        # 训练参数界面
        self.setupUi_tpf(self)
        self.qdarkstyle_sheet = qdarkstyle.load_stylesheet(qt_api='pyqt5')

        self.default_train_param = {
            "learning_rate": 0.0001,
            "weight_decay": 0.00002,
            "batch_size": 1,
            "val_batch_size": 2,
            "val_counts": 150,
            "early_stop_patience": 100
        }

        self.train_param = {
            "learning_rate": 0.0001,
            "weight_decay": 0.00002,
            "batch_size": 1,
            "val_batch_size": 2,
            "val_counts": 150,
            "early_stop_patience": 100
        }

        self.set_connect()
        self.set_style()
        # 使用setWindowModality方法设置对话框的模态性，可以防止对话框被其他窗口遮挡
        self.setWindowModality(Qt.ApplicationModal)

    def set_style(self):
        self.tp_lr_dsBox.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
        self.tp_wd_dsBox.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
        self.tp_bs_sBox.setStyleSheet(self.qdarkstyle_sheet + spin_style)
        self.tp_vbs_sBox.setStyleSheet(self.qdarkstyle_sheet + spin_style)
        self.tp_vs_sBox.setStyleSheet(self.qdarkstyle_sheet + spin_style)
        self.tp_esp_sBox.setStyleSheet(self.qdarkstyle_sheet + spin_style)
        self.tp_default_btn.setStyleSheet(button_style)

    def set_connect(self):
        self.tp_default_btn.clicked.connect(self.set_default)

    def set_default(self):
        self.tp_lr_dsBox.setValue(self.default_train_param.get("learning_rate"))
        self.tp_wd_dsBox.setValue(self.default_train_param.get("weight_decay"))
        self.tp_bs_sBox.setValue(self.default_train_param.get("batch_size"))
        self.tp_vbs_sBox.setValue(self.default_train_param.get("val_batch_size"))
        self.tp_vs_sBox.setValue(self.default_train_param.get("val_counts"))
        self.tp_esp_sBox.setValue(self.default_train_param.get("early_stop_patience"))

    def update_train_param(self):
        self.tp_lr_dsBox.setValue(self.train_param.get("learning_rate"))
        self.tp_wd_dsBox.setValue(self.train_param.get("weight_decay"))
        self.tp_bs_sBox.setValue(self.train_param.get("batch_size"))
        self.tp_vbs_sBox.setValue(self.train_param.get("val_batch_size"))
        self.tp_vs_sBox.setValue(self.train_param.get("val_counts"))
        self.tp_esp_sBox.setValue(self.train_param.get("early_stop_patience"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = TrainParamWidget()
    win.show()
    sys.exit(app.exec_())
