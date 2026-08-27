# -*- coding: utf-8 -*-
import imagecodecs._imcd
import imagecodecs._shared
import sys
import os
from os.path import join
import datetime
import psutil
import traceback
from multiprocessing import freeze_support
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QMessageBox
from PyQt5.QtGui import QIcon
from UI.Ui_IntegrateWindow import Ui_IntegrateWindow
from CellMultifunctionWidget import CellMultifunctionWidget
from NeuralMultifunctionWidget import NeuralMultifunctionWidget
from VesselMultifunctionWidget import VesselMultifunctionWidget
# from nnUNetWidget import nnUNetWidget
from control_style.ControlStyle import tool_bar_style, messagebox_style
from config import exe_cfg, cfgPath
from Logging import setLoggerConfig


class IntegrateWindow(QMainWindow, Ui_IntegrateWindow):
    def __init__(self, parent=None):
        super(IntegrateWindow, self).__init__(parent)
        # Get Program Running Path
        if getattr(sys, 'frozen', False):
            self.base_path = sys._MEIPASS
        else:
            self.base_path = os.path.dirname(os.path.abspath(__file__))

        # Get Log Path
        self.logging_path = self.get_logger_path()
        self.setLoggerConfig = setLoggerConfig()
        self.logger = self.setLoggerConfig.get_logger_object(self.logging_path)
        # 获取进程pid
        self.now_program_pid = self.get_current_pid()

        # Program Path
        exe_cfg.cfg['ExePath']['base_path'] = self.base_path

        with open(cfgPath, 'w') as configfile:
            exe_cfg.cfg.write(configfile)

        self.IconSize = 40
        self.setupUi(self)

        self.resize(1550, 920)

        self.setContextMenuPolicy(Qt.NoContextMenu)  # Disable Right-click Menu

        self.cell_view = QAction(QIcon(join(self.base_path, "icon_images", "cell2.png")), "Cell", self)
        self.neural_view = QAction(QIcon(join(self.base_path, "icon_images", "neural.png")), "Neural", self)
        self.vessel_view = QAction(QIcon(join(self.base_path, "icon_images", "xue_guan.png")), "Vessel", self)
        self.nnUNet_view = QAction(QIcon(join(self.base_path, "icon_images", "nnUNet.png")), "nnUNet", self)
        self.nnUNet_view.setVisible(False)
        self.change_icon(cell="cell2.png")
        self.toolBar.addAction(self.cell_view)
        self.toolBar.addAction(self.neural_view)
        self.toolBar.addAction(self.vessel_view)
        self.toolBar.addAction(self.nnUNet_view)
        # self.toolBar.setMovable(False)
        self.toolBar.setIconSize(QSize(self.IconSize + 15, self.IconSize))
        self.toolBar.setMinimumHeight(36)
        self.toolBar.setMinimumWidth(36)
        self.toolBar.layout().setSpacing(10)
        self.toolBar.setStyleSheet(tool_bar_style)
        # 设置工具栏按钮样式
        self.toolBar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)  # Text Under Icon
        # self.toolBar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)  # Text Beside Icon

        self.cell_view.triggered.connect(self.cell_view_click)
        self.neural_view.triggered.connect(self.neural_view_click)
        self.vessel_view.triggered.connect(self.vessel_view_click)
        self.nnUNet_view.triggered.connect(self.nnUNet_view_click)

        self.stackedWidget.setCurrentIndex(0)

        # # Cell
        self.CellMultifunctionWidget = CellMultifunctionWidget(self)
        self.cell_verticalLayout.addWidget(self.CellMultifunctionWidget)
        # # Neural
        self.NeuralMultifunctionWidget = NeuralMultifunctionWidget(self)
        # self.NeuralMultifunctionWidget.setVisible(True)
        self.neural_verticalLayout.addWidget(self.NeuralMultifunctionWidget)
        # # Vessel
        self.VesselMultifunctionWidget = VesselMultifunctionWidget(self)
        # self.VesselMultifunctionWidget.setVisible(True)
        self.vessel_verticalLayout.addWidget(self.VesselMultifunctionWidget)
        # nnUNet
        # self.nnUNetWidget = nnUNetWidget(self)
        # self.nnUNet_verticalLayout.addWidget(self.nnUNetWidget)

    def show(self):
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        # Activate Window
        self.raise_()
        self.activateWindow()
        # Bring Window to Front
        self.setWindowState(self.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)
        # self.showNormal()
        self.setWindowFlag(Qt.WindowStaysOnTopHint, False)
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

        # 如果保存的日志文件数大于30，去掉之前的日志
        if len([l for l in os.listdir(log_dir) if ".log" in l]) > 30:
            self.delete_oldest_file(log_dir)

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

    def cell_view_click(self):
        # 展示模块
        # print("cell module")
        self.stackedWidget.setCurrentIndex(0)
        self.change_icon(cell="cell2.png")

    def neural_view_click(self):
        # print("neural module")
        self.stackedWidget.setCurrentIndex(1)
        self.change_icon(neuron="neural2.png")

    def vessel_view_click(self):
        # print("vessel module")
        self.stackedWidget.setCurrentIndex(2)
        self.change_icon(vessel="xue_guan2.png")

    def nnUNet_view_click(self):
        # print("vessel module")
        self.stackedWidget.setCurrentIndex(3)
        self.change_icon(nnUNet="nnUNet2.png")

    def change_icon(self,
                    cell="cell.png",
                    neuron="neural.png",
                    vessel="xue_guan.png",
                    nnUNet="nnUNet.png"):
        self.cell_view.setIcon(QIcon(join(self.base_path, "icon_images", cell)))
        self.neural_view.setIcon(QIcon(join(self.base_path, "icon_images", neuron)))
        self.vessel_view.setIcon(QIcon(join(self.base_path, "icon_images", vessel)))
        self.nnUNet_view.setIcon(QIcon(join(self.base_path, "icon_images", nnUNet)))

    def delete_oldest_file(self, directory):
        # 获取目录中的所有文件
        files = [l for l in os.listdir(directory) if ".log" in l]
        if not files:
            print("The directory is empty, no files to delete.")
            return

        # 创建一个列表，存储文件的路径和创建时间
        files_with_ctime = []
        for file in files:
            file_path = join(directory, file)
            if os.path.isfile(file_path):  # 确保是文件而不是目录
                creation_time = os.path.getctime(file_path)
                files_with_ctime.append((file_path, creation_time))

        # 按创建时间排序，最早的文件在前面
        files_with_ctime.sort(key=lambda x: x[1])

        # 删除最早的文件
        oldest_file_path = files_with_ctime[0][0]
        os.remove(oldest_file_path)
        # print(f"已删除最早的文件：{oldest_file_path}")

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
        box = QMessageBox()
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setStyleSheet(messagebox_style)
        if self.CellMultifunctionWidget.CellDataSetMake.cm:
            box.setText('Cell data is being generated, close the window?')
        elif self.CellMultifunctionWidget.CellDataTrain.training:
            box.setText('Cell model is being trained, close the window?')
        elif self.CellMultifunctionWidget.CellDataPredict.predicting:
            box.setText('Cell data is being predicted, close the window?')
        elif self.NeuralMultifunctionWidget.NeuralDataSetMake.nm:
            box.setText('Neural data is being generated, close the window?')
        elif self.NeuralMultifunctionWidget.NeuralDataTrain.training:
            box.setText('Neural model is being trained, close the window?')
        elif self.NeuralMultifunctionWidget.NeuralDataPredict.predicting:
            box.setText('Neural data is being predicted, close the window?')
        # elif self.NeuralMultifunctionWidget.NeuralDataStatistics.analyzing:
        #     box.setText('Neural data is being analyzed, close the window?')
        elif self.VesselMultifunctionWidget.VesselDataSetMake.vm:
            box.setText('Vessel data is being generated, close the window?')
        elif self.VesselMultifunctionWidget.VesselDataTrain.training:
            box.setText('Vessel model is being trained, close the window?')
        elif self.VesselMultifunctionWidget.VesselDataPredict.predicting:
            box.setText('Vessel data is being predicted, close the window?')
        elif self.VesselMultifunctionWidget.VesselDataStatistics.analyzing:
            box.setText('Vessel data is being analyzed, close the window?')
        # elif self.nnUNetWidget.nnUNetDataTrain.training:
        #     box.setText('nnUNet model is being trained, close the window?')
        # elif self.nnUNetWidget.nnUNetDataPredict.predicting:
        #     box.setText('nnUNet data is being predicted, close the window?')
        else:
            box.setText('Close the window?')
        box.setWindowTitle('Prompt')
        box.setIcon(QMessageBox.Question)
        box.setWindowModality(Qt.ApplicationModal)
        if box.exec_() == QMessageBox.Yes:
            # self.CellDataTrain.TrainResultPreview.close()
            self.kill_program_pid(self.now_program_pid)
            event.accept()
        else:
            event.ignore()


def show_crash_box(exc_type, exc_value, exc_tb):
    """统一的未处理异常弹框"""
    msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
    box = QMessageBox()
    box.setIcon(QMessageBox.Critical)
    box.setWindowTitle('Program Crash')
    box.setText('Sorry, the program encountered an unhandled exception:')
    box.setDetailedText(msg)
    box.exec_()


class _SafeStdout:
    """当终端被关闭导致 stdout 管道断开时 (WinError 233)，自动静默重定向到 os.devnull，
    避免 GUI 因 print() 之类的无害代码直接崩溃。"""
    def __init__(self, original):
        self._original = original
        self._devnull = None

    def write(self, s):
        try:
            self._original.write(s)
        except OSError:
            if self._devnull is None:
                self._devnull = open(os.devnull, 'w')
            sys.stdout = self._devnull
            self._devnull.write(s)

    def flush(self):
        try:
            self._original.flush()
        except OSError:
            pass

    def __getattr__(self, name):
        return getattr(self._original, name)


if __name__ == "__main__":
    # 1. 追踪结果分支处断开的情况比较普遍，添加断点连接处理，在统计血管分支之前处理
    # 2. Connect、插值、计算半径、平滑处理
    # 3. 拼接优化
    # 4. 异常退出，File name
    # 5. 全脑新数据处理
    # pyinstaller
    # from PIL import Image
    # Image.open(r"icon_cell2.png").save("icon.ico")

    # --add-data=D:\python+vtk+Qt\cell_points_marking\icon_image\;icon_image\
    # --add-data=D:\python+vtk+Qt\cell_points_marking\bv\;bv\
    # --add-data=D:\python+vtk+Qt\cell_points_marking\imagecodecs\;imagecodecs\
    # python BVLabAnalyzer.py

    # pyinstaller BVLabAnalyzer.spec
    # pyinstaller BVLabAnalyzer_files.spec -y
    # pyinstaller BVLabAnalyzer_one.spec
    # pyinstaller --onefile BVLabAnalyzer.py

    # app = QApplication(sys.argv)
    # freeze_support()
    # main_window = IntegrateWindow()
    # # main_window.showMaximized()
    # icon = QIcon(os.path.join(main_window.base_path, 'icon_image', 'icon_cell2.png'))
    # main_window.setWindowIcon(icon)
    # main_window.show()
    # sys.exit(app.exec_())

    # icon='D:\CellNeuralBloodVessel\IntegratePoseOptimization\icon_image\icon.ico',

    # 将代码中的中文文本提示和输出转为英文

    freeze_support()
    # 保护 stdout/stderr：终端关闭后 print()/日志输出不再导致 OSError 233，静默写入 devnull
    sys.stdout = _SafeStdout(sys.stdout)
    sys.stderr = _SafeStdout(sys.stderr)
    app = QApplication(sys.argv)

    # 全局异常钩子，任何线程未捕获的异常都会走到这里
    sys.excepthook = show_crash_box

    try:
        main_window = IntegrateWindow()
        icon = QIcon(os.path.join(main_window.base_path,
                                  'icon_image', 'icon_cell2.png'))
        main_window.setWindowIcon(icon)
        main_window.show()
        sys.exit(app.exec_())
    except Exception as e:
        # 万一初始化前就炸了，也抓一下
        show_crash_box(*sys.exc_info())
