# -*- coding: utf-8 -*-
import imagecodecs._imcd
import imagecodecs._shared
import sys
import os
from os.path import join
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QSplitter, QMessageBox, QWidget, QApplication, QListWidget
from PyQt5.QtGui import QIcon
import datetime
import psutil
from multiprocessing import freeze_support
import qdarkstyle
from UI.Ui_NeuralMultifunctionWidget import Ui_NeuralMultifunctionWidget
from DataSetMake.NeuralDataSetMake import NeuralDataSetMake
from DataPredict.NeuralDataPredict import NeuralDataPredict
from DataTrain.NeuralDataTrain import NeuralDataTrain
from DataStatistics.NeuralStatistics import NeuralDataStatistics
from ViewWidget.LineViewWidget import LineViewWidget
from control_style.ControlStyle import (button_alpha_style, button_style, lineedit_style, spin_style,
                                        double_spin_qdarkstyle, messagebox_style, comboBox_style, tool_button_style)
from config import exe_cfg, cfgPath
from Logging import setLoggerConfig
from DataStatistics.create_chart_neural import MainWindow


class NeuralMultifunctionWidget(QWidget, Ui_NeuralMultifunctionWidget):
    def __init__(self, IntegrateWindow):
        super().__init__()
        self.win = IntegrateWindow

        # Get Program Running Path
        if getattr(sys, 'frozen', False):
            self.base_path = sys._MEIPASS
        else:
            self.base_path = os.path.dirname(os.path.abspath(__file__))

        if self.win:
            self.logger = self.win.logger
        else:
            # Get Log Path
            self.logging_path = self.get_logger_path()
            self.setLoggerConfig = setLoggerConfig()
            self.logger = self.setLoggerConfig.get_logger_object(self.logging_path)
            # 获取进程pid
            self.now_program_pid = self.get_current_pid()

        # 模型日志路径
        logs_path = join("./logs", "neural_logs")
        os.makedirs(logs_path, exist_ok=True)
        exe_cfg.cfg['ExePath']['neural_logs_path'] = str(os.path.abspath(logs_path))

        # 模型保存路径
        models_path = join("./ModelSave", "NeuralTrainModel")
        os.makedirs(models_path, exist_ok=True)
        exe_cfg.cfg['ExePath']['neural_models_path'] = str(os.path.abspath(models_path))

        # Program Path
        exe_cfg.cfg['ExePath']['base_path'] = self.base_path

        # 端口，地址
        port = exe_cfg.cfg['Port']['neural_port']
        exe_cfg.cfg['Port']['neural_port'] = port
        host = exe_cfg.cfg['Host']['neural_host']
        exe_cfg.cfg['Host']['neural_host'] = host

        with open(cfgPath, 'w') as configfile:
            exe_cfg.cfg.write(configfile)

        # with open("gpu.txt", "w", encoding="utf-8") as f:
        #     f.write(f"{8}")

        self.Icon_size = 30

        # 默认模型路径
        self.default_models_Dir = join(self.base_path, 'DefaultModels', 'NeuralModels')
        os.makedirs(self.default_models_Dir, exist_ok=True)

        # 样式
        self.qdarkstyle_sheet = qdarkstyle.load_stylesheet(qt_api='pyqt5')
        self.setupUi(self)
        self.resize(1500, 920)

        # 设置初始值
        self.set_init()
        # Signal
        # self.set_connect()
        # 样式
        self.set_style()

        # self.resize(1650, 922)

        """设置拉伸"""
        # 数据集制作
        self.set_splitter(
            left_widget=self.MakingWidget,
            right_widget=self.MakingFrame,
            horizontal_layout=self.Making_horizontalLayout,
            top_widget=self.making_func_widget,
            bottom_widget=self.making_info_widget,
            vertical_layout=self.Making_verticalLayout)
        # 模型预测
        self.set_splitter(
            left_widget=self.PredictWidget,
            right_widget=self.PredictFrame,
            horizontal_layout=self.Predict_horizontalLayout,
            top_widget=self.predict_func_widget,
            bottom_widget=self.predict_info_widget,
            vertical_layout=self.Predict_verticalLayout)
        # Train
        self.set_splitter(
            left_widget=self.TrainWidget,
            right_widget=self.TrainFrame,
            horizontal_layout=self.Train_horizontalLayout,
            top_widget=self.train_func_widget,
            bottom_widget=self.train_info_widget,
            vertical_layout=self.Train_verticalLayout)
        # Statistics
        self.set_splitter(
            left_widget=self.StatisticsWidget,
            right_widget=self.StatisticsFrame,
            horizontal_layout=self.Statistics_horizontalLayout,
            top_widget=self.statistics_func_widget,
            bottom_widget=self.statistics_info_widget,
            vertical_layout=self.Statistics_verticalLayout)
        self.set_splitter(left_widget=self.statistics_files_progress_widget,
                          right_widget=self.statistics_analyze_widget,
                          horizontal_layout=self.StatisticsFrame_horizontalLayout,
                          )
        self.set_splitter(top_widget=self.statistics_files_frame,
                          bottom_widget=self.statistics_progress_frame,
                          vertical_layout=self.progress_verticalLayout,
                          )
        self.set_splitter(top_widget=self.analyze_chart_frame,
                          bottom_widget=self.analyze_total_chart_frame,
                          vertical_layout=self.analyze_verticalLayout,
                          )
        # Toolbar
        self.DataMakeButton.setText("Creation")  # 设置按钮文本
        self.ModelPredictButton.setText("Prediction")  # 设置按钮文本
        self.DataReviseButton.setText("Annotation")  # 设置按钮文本
        self.ModelTrainButton.setText("Training")  # 设置按钮文本
        self.DataStatisticsButton.setText("Statistics")  # 设置按钮文本
        self.change_icon(revising_icon="revising2.png")
        self.DataMakeButton.clicked.connect(self.DataMakingButton_clicked)
        self.ModelPredictButton.clicked.connect(self.ModelPredictButton_clicked)
        self.DataReviseButton.clicked.connect(self.DataReviseButton_clicked)
        self.ModelTrainButton.clicked.connect(self.ModelTrainButton_clicked)
        self.DataStatisticsButton.clicked.connect(self.DataStatisticsButton_clicked)
        # 修订
        self.LineViewWidget = LineViewWidget()
        self.revising_verticalLayout.addWidget(self.LineViewWidget)

        self.NeuralDataSetMake = NeuralDataSetMake(self)  # 制作
        self.NeuralDataTrain = NeuralDataTrain(self)  # Train
        self.NeuralDataPredict = NeuralDataPredict(self)  # Predict
        self.NeuralDataStatistics = NeuralDataStatistics(self)  # Statistics

    def show(self):
        if not self.win:
            self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
            # Activate Window
            self.raise_()
            self.activateWindow()
            # Bring Window to Front
            self.setWindowState(self.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)
            self.showNormal()
            self.setWindowFlag(Qt.WindowStaysOnTopHint, False)
            # Activate Window
            # self.raise_()
            # self.activateWindow()
            # # Bring Window to Front
            # self.setWindowState(self.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)
            # self.showMaximized()
            self.showNormal()

    def get_logger_path(self):
        # 获取 .exe 文件所在的目录
        projectDir = os.path.dirname(os.path.abspath(sys.argv[0]))
        os.chdir(projectDir)  # 切换工作目录
        # print("Program Path：", projectDir)

        # 日志路径
        log_dir = os.path.join(projectDir, 'logs')
        if not os.path.exists(log_dir):  # 日志文件夹
            os.makedirs(log_dir, exist_ok=True)
        # 获取当前日期的年、Month、日
        now = datetime.datetime.now()
        year = now.year
        month = now.month
        day = now.day
        logging_path = os.path.join(log_dir, f'{year}-{month}-{day}-info.log')
        # print("日志路径：", self.logging_path)

        return logging_path

    def get_current_pid(self):
        pid = os.getpid()
        self.logger.info(f"Current process PID: {pid}")
        return pid

    """设置初始值"""

    def set_init(self):
        # 训练参数设置
        self.train_arguments = {
            "epochs": 500,
        }
        self.set_train_arguments()
        # 训练集制作参数
        self.making_arguments = {
            "dataNums": 100,
            "MNumber": 5,
            "smallSize": [192, 192, 192],
            "divisionRatio": [7, 2, 1]
        }
        self.set_making_arguments()

        # 控件初始化
        self.stackedWidget.setCurrentIndex(0)  # 默认第一个界面
        # 数据制作
        self.path_widget.hide()  # Hide
        self.data_type_comboBox.setCurrentIndex(0)
        self.select_model_comboBox.setCurrentIndex(0)  # 默认第一个模型
        self.mask_widget.hide()
        self.bv_widget.hide()
        self.making_level_spinBox.setValue(0)  # 设置默认等级
        # self.bv_radioButton.setVisible(False)
        # Train
        self.train_end_Button.setVisible(False)
        # Predict
        self.predict_stackedWidget.setCurrentIndex(0)  # 默认预测界面
        self.predict_radioButton.setChecked(True)  # 默认激活
        self.splice_start_Button.setVisible(False)
        self.predict_level_spinBox.setValue(0)
        self.bv_ROI_widget.setVisible(False)
        self.bv_ROI_checkBox.setChecked(False)
        # Statistics
        self.import_cfg_radioButton.setChecked(True)  # 默认激活
        self.input_img_widget.hide()  # 隐藏输入图像窗口
        self.unit_comboBox.setCurrentIndex(0)  # 默认第一个单位
        self.connect_checkBox.setChecked(False)

        # 文本
        self.making_info_textBrowser.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 取消垂直滑块
        # self.making_result_listWidget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.making_result_listWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 取消水平滑块
        self.making_result_listWidget.setFocusPolicy(Qt.NoFocus)
        self.making_result_listWidget.setStyleSheet(self.qdarkstyle_sheet)  # 筛选

        self.predict_info_textBrowser.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # self.predict_result_listWidget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.predict_result_listWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 取消水平滑块
        self.predict_result_listWidget.setFocusPolicy(Qt.NoFocus)
        self.predict_result_listWidget.setStyleSheet(self.qdarkstyle_sheet)  # 筛选

        self.train_info_textBrowser.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # self.train_result_listWidget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.train_result_listWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 取消水平滑块
        self.train_result_listWidget.setFocusPolicy(Qt.NoFocus)
        self.train_result_listWidget.setStyleSheet(self.qdarkstyle_sheet)  # 筛选
        # Settings QListWidget 不可选择
        self.train_result_listWidget.setSelectionMode(QListWidget.NoSelection)

        # Statistics
        self.statistics_info_textBrowser.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.statistics_result_listWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 取消水平滑块
        self.statistics_result_listWidget.setFocusPolicy(Qt.NoFocus)
        self.statistics_result_listWidget.setStyleSheet(self.qdarkstyle_sheet)  # 筛选
        self.statistics_files_listWidget.setFocusPolicy(Qt.NoFocus)
        self.statistics_files_listWidget.setStyleSheet(self.qdarkstyle_sheet)  # 筛选
        self.statistics_files_listWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 取消水平滑块

        self.chart_main = MainWindow()
        self.analyze_total_chart_verticalLayout.addWidget(self.chart_main.centralwidget)

    """训练集制作参数"""

    def set_making_arguments(self):
        self.dataNums_spinBox.setValue(self.making_arguments.get("dataNums"))
        self.MNumber_spinBox.setValue(self.making_arguments.get("MNumber"))
        self.data_x_spinBox.setValue(self.making_arguments.get("smallSize")[0])
        self.data_y_spinBox.setValue(self.making_arguments.get("smallSize")[1])
        self.data_z_spinBox.setValue(self.making_arguments.get("smallSize")[2])
        self.train_ratio_doubleSpinBox.setValue(self.making_arguments.get("divisionRatio")[0])
        self.val_ratio_doubleSpinBox.setValue(self.making_arguments.get("divisionRatio")[1])
        self.test_ratio_doubleSpinBox.setValue(self.making_arguments.get("divisionRatio")[2])

    """训练参数"""

    def set_train_arguments(self):
        self.epoch_spinBox.setValue(self.train_arguments.get("epochs"))

    """设置样式"""

    def set_style(self):
        # Toolbar
        self.DataMakeButton.setStyleSheet(tool_button_style)
        self.ModelPredictButton.setStyleSheet(tool_button_style)
        self.DataReviseButton.setStyleSheet(tool_button_style)
        self.ModelTrainButton.setStyleSheet(tool_button_style)
        self.DataStatisticsButton.setStyleSheet(tool_button_style)
        self.DataStatisticsButton.hide()
        """数据制作"""
        self.making_start_Button.setStyleSheet(button_style)
        self.path_lineEdit.setStyleSheet(lineedit_style)
        self.select_model_comboBox.setStyleSheet(self.qdarkstyle_sheet + comboBox_style)
        self.data_type_comboBox.setStyleSheet(self.qdarkstyle_sheet + comboBox_style)
        self.dataNums_spinBox.setStyleSheet(self.qdarkstyle_sheet + spin_style)
        self.MNumber_spinBox.setStyleSheet(self.qdarkstyle_sheet + spin_style)
        self.data_x_spinBox.setStyleSheet(self.qdarkstyle_sheet + spin_style)
        self.data_y_spinBox.setStyleSheet(self.qdarkstyle_sheet + spin_style)
        self.data_z_spinBox.setStyleSheet(self.qdarkstyle_sheet + spin_style)
        self.train_ratio_doubleSpinBox.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
        self.val_ratio_doubleSpinBox.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
        self.test_ratio_doubleSpinBox.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
        self.Min_x_spinBox.setStyleSheet(self.qdarkstyle_sheet + spin_style)
        self.Min_y_spinBox.setStyleSheet(self.qdarkstyle_sheet + spin_style)
        self.Min_z_spinBox.setStyleSheet(self.qdarkstyle_sheet + spin_style)
        self.Max_x_spinBox.setStyleSheet(self.qdarkstyle_sheet + spin_style)
        self.Max_y_spinBox.setStyleSheet(self.qdarkstyle_sheet + spin_style)
        self.Max_z_spinBox.setStyleSheet(self.qdarkstyle_sheet + spin_style)
        # 无标签数据集
        self.no_mask_images_lineEdit.setStyleSheet(lineedit_style)  # 输入图像路径
        self.no_mask_images_Button.setStyleSheet(button_style)
        self.no_mask_cfg_lineEdit.setStyleSheet(lineedit_style)  # 输出配置文件
        self.no_mask_cfg_Button.setStyleSheet(button_style)
        # 有标签数据集
        self.making_image_lineEdit.setStyleSheet(lineedit_style)  # 输入图像路径
        self.making_image_Button.setStyleSheet(button_style)
        self.making_swc_lineEdit.setStyleSheet(lineedit_style)  # 输入标签路径
        self.making_swc_Button.setStyleSheet(button_style)
        self.making_cfg_lineEdit.setStyleSheet(lineedit_style)  # 输出配置文件
        self.making_cfg_Button.setStyleSheet(button_style)
        # bv格式数据
        self.bv_images_lineEdit.setStyleSheet(lineedit_style)  # 输入图像路径
        self.bv_images_Button.setStyleSheet(button_style)
        self.bv_cfg_lineEdit.setStyleSheet(lineedit_style)  # 输出配置文件
        self.bv_cfg_Button.setStyleSheet(button_style)
        self.making_level_spinBox.setStyleSheet(self.qdarkstyle_sheet + spin_style)  # 降采样等级

        """Train"""
        self.train_cfg_lineEdit.setStyleSheet(lineedit_style)
        self.train_cfg_Button.setStyleSheet(button_style)
        self.train_save_lineEdit.setStyleSheet(lineedit_style)
        self.train_save_Button.setStyleSheet(button_style)
        self.epoch_spinBox.setStyleSheet(self.qdarkstyle_sheet + spin_style)

        self.train_start_Button.setStyleSheet(button_style)
        self.train_end_Button.setStyleSheet(button_style)
        self.train_preview_Button.setEnabled(False)
        self.train_preview_Button.setStyleSheet(button_alpha_style)

        """Predict"""
        self.predict_cfg_images_lineEdit.setStyleSheet(lineedit_style)
        self.predict_cfg_images_Button.setStyleSheet(button_style)
        self.predict_cfg_lineEdit.setStyleSheet(lineedit_style)
        self.predict_cfg_comboBox.setStyleSheet(self.qdarkstyle_sheet + comboBox_style)
        self.predict_cfg_save_lineEdit.setStyleSheet(lineedit_style)
        self.predict_cfg_save_Button.setStyleSheet(button_style)
        self.predict_start_Button.setStyleSheet(button_style)
        self.predict_radioButton.setStyleSheet(self.qdarkstyle_sheet)
        self.splice_cfg_lineEdit.setStyleSheet(lineedit_style)
        self.splice_save_lineEdit.setStyleSheet(lineedit_style)
        self.splice_start_Button.setStyleSheet(button_style)
        self.splice_cfg_Button.setStyleSheet(button_style)
        self.splice_save_Button.setStyleSheet(button_style)
        self.splice_radioButton.setStyleSheet(self.qdarkstyle_sheet)
        self.predict_level_spinBox.setStyleSheet(self.qdarkstyle_sheet + spin_style)  # 降采样等级
        self.bv_ROI_checkBox.setStyleSheet(self.qdarkstyle_sheet)
        self.MinX_spinBox.setStyleSheet(self.qdarkstyle_sheet + spin_style)
        self.MinY_spinBox.setStyleSheet(self.qdarkstyle_sheet + spin_style)
        self.MinZ_spinBox.setStyleSheet(self.qdarkstyle_sheet + spin_style)
        self.MaxX_spinBox.setStyleSheet(self.qdarkstyle_sheet + spin_style)
        self.MaxY_spinBox.setStyleSheet(self.qdarkstyle_sheet + spin_style)
        self.MaxZ_spinBox.setStyleSheet(self.qdarkstyle_sheet + spin_style)

        """Statistics"""
        self.import_img_radioButton.setStyleSheet(self.qdarkstyle_sheet)
        self.import_cfg_radioButton.setStyleSheet(self.qdarkstyle_sheet)
        self.input_img_lineEdit.setStyleSheet(lineedit_style)
        self.input_swc_lineEdit.setStyleSheet(lineedit_style)
        self.input_cfg_lineEdit.setStyleSheet(lineedit_style)
        self.input_save_lineEdit.setStyleSheet(lineedit_style)
        self.input_img_Button.setStyleSheet(button_style)
        self.input_swc_Button.setStyleSheet(button_style)
        self.input_cfg_Button.setStyleSheet(button_style)
        self.input_save_Button.setStyleSheet(button_style)
        self.statistics_start_Button.setStyleSheet(button_style)
        self.x_resolution_doubleSpinBox.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
        self.y_resolution_doubleSpinBox.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
        self.z_resolution_doubleSpinBox.setStyleSheet(self.qdarkstyle_sheet + double_spin_qdarkstyle)
        self.unit_comboBox.setStyleSheet(self.qdarkstyle_sheet + comboBox_style)
        self.connect_checkBox.setStyleSheet(self.qdarkstyle_sheet)

    def set_connect(self):
        pass

    def show_training_arguments_dialog(self):
        self.TrainingArgumentsDialog.show()

    """设置拉伸"""

    def set_splitter(self,
                     left_widget=None,
                     right_widget=None,
                     horizontal_layout=None,
                     top_widget=None,
                     bottom_widget=None,
                     vertical_layout=None):
        """QSplitter 可以实现可拉伸的功能"""
        if left_widget is not None:
            splitter_horizontal = QSplitter(Qt.Horizontal)
            splitter_horizontal.setHandleWidth(0)
            horizontal_layout.addWidget(splitter_horizontal)
            splitter_horizontal.addWidget(left_widget)
            splitter_horizontal.addWidget(right_widget)
            # Prevent child windows from being completely hidden
            splitter_horizontal.setChildrenCollapsible(False)

        if top_widget is not None:
            splitter_vertical = QSplitter(Qt.Vertical)
            splitter_vertical.setHandleWidth(0)
            vertical_layout.addWidget(splitter_vertical)
            splitter_vertical.addWidget(top_widget)
            splitter_vertical.addWidget(bottom_widget)
            splitter_vertical.setChildrenCollapsible(False)

    """分页跳转"""

    def DataReviseButton_clicked(self):
        # print("DataRevising")
        if self.stackedWidget.currentIndex() != 0:
            self.change_icon(revising_icon="revising2.png")
            self.stackedWidget.setCurrentIndex(0)

    def DataMakingButton_clicked(self):
        # print("DataMaking")
        if self.stackedWidget.currentIndex() != 1:
            self.change_icon(making_icon="making2.png")
            self.stackedWidget.setCurrentIndex(1)

    def ModelTrainButton_clicked(self):
        # print("ModelTraining")
        if self.stackedWidget.currentIndex() != 2:
            self.change_icon(training_icon="training2.png")
            self.stackedWidget.setCurrentIndex(2)

    def ModelPredictButton_clicked(self):
        # print("ModelPredicting")
        if self.stackedWidget.currentIndex() != 3:
            self.change_icon(predicting_icon="predicting2.png")
            self.stackedWidget.setCurrentIndex(3)

    def DataStatisticsButton_clicked(self):
        # print("Statistics")
        if self.stackedWidget.currentIndex() != 4:
            self.change_icon(statistics_icon="statistics2.png")
            self.stackedWidget.setCurrentIndex(4)

    """设置图标"""

    def change_icon(self,
                    making_icon="making.png",
                    predicting_icon="predicting.png",
                    revising_icon="revising.png",
                    training_icon="training.png",
                    statistics_icon="statistics.png"):
        self.DataMakeButton.setIcon(QIcon(join(self.base_path, "icon_images", making_icon)))
        self.ModelPredictButton.setIcon(QIcon(join(self.base_path, "icon_images", predicting_icon)))
        self.DataReviseButton.setIcon(QIcon(join(self.base_path, "icon_images", revising_icon)))
        self.ModelTrainButton.setIcon(QIcon(join(self.base_path, "icon_images", training_icon)))
        self.DataStatisticsButton.setIcon(QIcon(join(join(self.base_path, "icon_images", statistics_icon))))
        self.DataMakeButton.setIconSize(QSize(self.Icon_size, self.Icon_size))
        self.ModelPredictButton.setIconSize(QSize(self.Icon_size, self.Icon_size))
        self.DataReviseButton.setIconSize(QSize(self.Icon_size, self.Icon_size))
        self.ModelTrainButton.setIconSize(QSize(self.Icon_size, self.Icon_size))
        self.DataStatisticsButton.setIconSize(QSize(self.Icon_size, self.Icon_size))

    def kill_program_pid(self, pid):
        try:
            self.logger.info(f"Closing window, killing process {pid} and its children")
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        except Exception as e:
            self.logger.error(f"Error killing process: {e}")

    def closeEvent(self, event):
        if self.win:
            event.accept()
        else:
            box = QMessageBox()
            box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            box.setStyleSheet(messagebox_style)
            box.setText('Close the window or not？')
            box.setWindowTitle('Prompt')
            box.setIcon(QMessageBox.Question)
            box.setWindowModality(Qt.ApplicationModal)
            if box.exec_() == QMessageBox.Yes:
                # self.CellDataTrain.TrainResultPreview.close()
                self.kill_program_pid(self.now_program_pid)
                event.accept()
            else:
                event.ignore()


if __name__ == "__main__":
    # python -m nuitka --version
    # pyinstaller NeuralMultifunctionWidget.py
    # pyinstaller --onefile NeuralMultifunctionWidget.py
    # pyinstaller NeuralMultifunctionWidget.spec
    # pyinstaller NeuralMultifunctionWidget_files.spec
    # pyinstaller NeuralMultifunctionWidget_one.spec
    # pyinstaller --hidden-import tensorboard --add-data "D:\anaconda3\envs\IntegratePose_py39\Lib\site-packages\tensorboard;./tensorboard" CellMultifunctionWidget.py
    app = QApplication(sys.argv)
    freeze_support()
    win = NeuralMultifunctionWidget(None)
    icon = QIcon(os.path.join(win.base_path, 'icon_image', 'icon_cell2.png'))
    win.setWindowIcon(icon)
    # win.showMaximized()
    win.show()
    sys.exit(app.exec_())
