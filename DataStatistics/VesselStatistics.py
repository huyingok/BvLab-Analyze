# -*- coding: utf-8 -*-
import shutil
from os.path import join
import matplotlib.pyplot as plt
import os
import sys
import traceback
from pathlib import Path
from PyQt5.QtWidgets import (QFileDialog, QMessageBox, QListWidgetItem, QComboBox, QPushButton, QVBoxLayout, QWidget,
                             QSizePolicy)
from PyQt5.QtCore import Qt, QSettings, QUrl, QSize
from PyQt5.QtGui import QDesktopServices, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import time
import json
import pandas as pd
import numpy as np
import subprocess
from CustomListItem import CustomListItem
# from DataStatistics.vessel_radius.RadiusCalculateQThread import RadiusCalculateQThread
from DataStatistics.VesselStatisticsQThread import VesselStatisticsQThread
from control_style.ControlStyle import (button_alpha_style, button_style, lineedit_alpha_style, lineedit_style,
                                        messagebox_style, checkBox_alpha_style, checkBox_style, comboBox_style)


# Set Global Font to Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.family'] = 'Microsoft YaHei'  # '微软雅黑'


"""Vessel"""


class VesselDataStatistics(object):
    def __init__(self, win):
        self.win = win
        self.logger = self.win.logger
        # Initialization
        self.analyze_dict = {}
        self.analyzing = False
        self.count = 0
        self.unit_dict = {
            "µm": "µm³",
            "mm": "mm³"
        }
        self.unit_text = "µm"
        self.statistics_files_dict = {}
        self.analyze_chart_widget = None
        self.idx = 0

        # Parent class variables
        self.qdarkstyle_sheet = self.win.qdarkstyle_sheet
        self.input_img_widget = self.win.input_img_widget
        self.input_cfg_widget = self.win.input_cfg_widget
        self.import_img_radioButton = self.win.import_img_radioButton
        self.import_cfg_radioButton = self.win.import_cfg_radioButton
        self.input_img_lineEdit = self.win.input_img_lineEdit
        self.input_cfg_lineEdit = self.win.input_cfg_lineEdit
        self.input_swc_lineEdit = self.win.input_swc_lineEdit
        self.input_save_lineEdit = self.win.input_save_lineEdit
        self.input_img_Button = self.win.input_img_Button
        self.input_cfg_Button = self.win.input_cfg_Button
        self.input_swc_Button = self.win.input_swc_Button
        self.input_save_Button = self.win.input_save_Button
        self.x_resolution_doubleSpinBox = self.win.x_resolution_doubleSpinBox
        self.y_resolution_doubleSpinBox = self.win.y_resolution_doubleSpinBox
        self.z_resolution_doubleSpinBox = self.win.z_resolution_doubleSpinBox
        self.unit_comboBox = self.win.unit_comboBox
        self.unit_label = self.win.unit_label
        self.connect_checkBox = self.win.connect_checkBox
        self.statistics_start_Button = self.win.statistics_start_Button
        self.analyze_chart_verticalLayout = self.win.analyze_chart_verticalLayout
        self.chart_main = self.win.chart_main
        # Set unit
        self.unit_names = list(self.unit_dict.keys())
        self.unit_comboBox.clear()
        self.unit_comboBox.addItems(self.unit_names)
        # File
        self.statistics_files_listWidget = self.win.statistics_files_listWidget
        # Progress
        self.statistics_result_listWidget = self.win.statistics_result_listWidget
        # Signal
        self.import_img_radioButton.toggled.connect(lambda: self.radioButton_toggled(0))
        self.import_cfg_radioButton.toggled.connect(lambda: self.radioButton_toggled(1))
        self.statistics_start_Button.clicked.connect(self.statistics_start)
        self.input_img_Button.clicked.connect(lambda: self.select_dir(self.input_img_lineEdit))
        self.input_swc_Button.clicked.connect(lambda: self.select_dir(self.input_swc_lineEdit))
        self.input_cfg_Button.clicked.connect(lambda: self.select_path(1))
        self.input_save_Button.clicked.connect(lambda: self.select_dir(self.input_save_lineEdit, save=1))
        self.unit_comboBox.activated.connect(self.select_unit)
        self.statistics_files_listWidget.currentItemChanged.connect(self.show_chart)
        # Thread
        # self.RadiusCalculateQThread = RadiusCalculateQThread(win=self)
        # self.RadiusCalculateQThread.finish0.connect(self.radius_calculate_finish)
        # self.RadiusCalculateQThread.progress_text.connect(self.radius_calculate_progress)
        # self.RadiusCalculateQThread.error0.connect(self.radius_calculate_error)

        self.VesselStatisticsQThread = VesselStatisticsQThread(win=self)
        self.VesselStatisticsQThread.finish0.connect(self.statistics_finish)
        self.VesselStatisticsQThread.progress_text.connect(self.statistics_progress)
        self.VesselStatisticsQThread.error0.connect(self.statistics_error)
        self.VesselStatisticsQThread.feature_files.connect(self.statistics_files)
        self.VesselStatisticsQThread.chart_files.connect(self.statistics_chart)

    def radioButton_toggled(self, index):
        if index == 0:  # Input image
            self.input_img_widget.show()
            self.input_cfg_widget.hide()
        if index == 1:  # Config file
            self.input_img_widget.hide()
            self.input_cfg_widget.show()

    def select_unit(self, index):
        text = self.unit_comboBox.itemText(index)
        self.unit_text = self.unit_dict[text]
        self.unit_label.setText(self.unit_text)

    def statistics_start(self):
        try:
            if not self.analyzing:
                data_type = "small"
                img_path = swc_path = CutWorkFilesDir = ""
                # Resolution ratio
                resolution_ratio = [self.x_resolution_doubleSpinBox.value(),
                                    self.y_resolution_doubleSpinBox.value(),
                                    self.z_resolution_doubleSpinBox.value()]

                save_path = self.input_save_lineEdit.text()
                need_connect = self.connect_checkBox.isChecked()

                if not save_path:
                    self.mess("Save path is empty, cannot start!", "Prompt", 1)
                    return

                if min(resolution_ratio) == 0:
                    self.mess("Resolution ratio must be non-zero!", "Prompt", 1)
                    return

                if self.import_img_radioButton.isChecked():  # Select input image
                    img_path = self.input_img_lineEdit.text()
                    swc_path = self.input_swc_lineEdit.text()
                elif self.import_cfg_radioButton.isChecked():  # Select config file
                    cfg_path = self.input_cfg_lineEdit.text()
                    if cfg_path:
                        if os.path.exists(cfg_path) and ".json" in cfg_path:
                            with open(cfg_path, 'r') as f:
                                cfgInfo = json.loads(f.read())
                            dataType = cfgInfo.get('dataType', None)
                            if dataType is None:
                                self.mess("The input configuration file is incorrect, cannot start!", "Prompt", 1)
                                return
                            else:
                                if dataType == "BV":  # BVFormat
                                    CutWorkFilesDir = cfgInfo.get('CutWorkFilesDir', None)
                                    if CutWorkFilesDir is None or len(CutWorkFilesDir) == 0:
                                        self.mess("The input configuration file has no CutWorkFilesDir parameter, cannot start!", "Prompt", 1)
                                        return
                                    else:
                                        if not os.path.isdir(CutWorkFilesDir):
                                            self.mess("The CutWorkFilesDir path in the configuration file does not exist, cannot start!", "Prompt", 1)
                                            return
                                    data_type = "bv"
                                elif dataType == "TIF":
                                    img_path = cfgInfo.get('cut_img_dir', None)
                                    swc_path = cfgInfo.get('cut_swc_dir', None)
                        else:
                            self.mess("The input configuration file does not exist, cannot start!", "Prompt", 1)
                            return
                    else:
                        self.mess("Input configuration file is empty, cannot start!", "Prompt", 1)
                        return
                if data_type == "bv":
                    res = self.mess("Start statistics?", "Prompt", 2)
                    if res == QMessageBox.Yes:
                        self.analyze_dict = {
                            "img_path": img_path,
                            "swc_path": swc_path,
                            "save_path": save_path,
                            "resolution_ratio": resolution_ratio,
                            "CutWorkFilesDir": CutWorkFilesDir,
                            "data_type": data_type,
                            "need_connect": need_connect
                        }
                        # Disable
                        self.forbid_control()
                        self.analyzing = True
                        self.statistics_files_listWidget.blockSignals(True)
                        self.statistics_files_listWidget.clear()
                        self.statistics_files_listWidget.blockSignals(False)
                        if self.analyze_chart_widget is not None:
                            self.analyze_chart_widget.deleteLater()  # Properly delete and free memory
                            self.analyze_chart_widget = None
                        self.idx = 0
                        self.statistics_files_dict = {}
                        self.statistics_result_listWidget.clear()
                        self.count = 0
                        self.start_time = time.time()
                        self.add_items("Statistics started!", None, None, )
                        # self.RadiusCalculateQThread.start()
                        self.VesselStatisticsQThread.start()
                else:
                    if img_path:
                        if swc_path:
                            if os.path.isdir(img_path):
                                if os.path.isdir(swc_path):
                                    res = self.mess("Start statistics?", "Prompt", 2)
                                    if res == QMessageBox.Yes:
                                        self.analyze_dict = {
                                            "img_path": img_path,
                                            "swc_path": swc_path,
                                            "save_path": save_path,
                                            "resolution_ratio": resolution_ratio,
                                            "CutWorkFilesDir": CutWorkFilesDir,
                                            "data_type": data_type,
                                            "need_connect": need_connect
                                        }
                                        # Disable
                                        self.forbid_control()
                                        self.analyzing = True
                                        self.statistics_files_listWidget.blockSignals(True)
                                        self.statistics_files_listWidget.clear()
                                        self.statistics_files_listWidget.blockSignals(False)
                                        if self.analyze_chart_widget is not None:
                                            self.analyze_chart_widget.deleteLater()  # Properly delete and free memory
                                            self.analyze_chart_widget = None
                                        self.idx = 0
                                        self.statistics_files_dict = {}
                                        self.statistics_result_listWidget.clear()
                                        self.count = 0
                                        self.start_time = time.time()
                                        self.add_items("Statistics started!", None, None, )
                                        # self.RadiusCalculateQThread.start()
                                        self.VesselStatisticsQThread.start()
                                else:
                                    self.mess("Input label path does not exist, cannot start!", "Prompt", 1)
                                    return
                            else:
                                self.mess("Input image path does not exist, cannot start!", "Prompt", 1)
                                return
                        else:
                            self.mess("Input label path is empty, cannot start!", "Prompt", 1)
                            return
                    else:
                        self.mess("Input image path is empty, cannot start!", "Prompt", 1)
                        return
        except Exception as e:
            self.error_logger(e)

    """Disable controls"""

    def forbid_control(self):
        # Disable
        self.import_img_radioButton.setEnabled(False)
        self.import_cfg_radioButton.setEnabled(False)
        self.input_img_lineEdit.setEnabled(False)
        self.input_img_lineEdit.setStyleSheet(lineedit_alpha_style)
        self.input_swc_lineEdit.setEnabled(False)
        self.input_swc_lineEdit.setStyleSheet(lineedit_alpha_style)
        self.input_cfg_lineEdit.setEnabled(False)
        self.input_cfg_lineEdit.setStyleSheet(lineedit_alpha_style)
        self.input_save_lineEdit.setEnabled(False)
        self.input_save_lineEdit.setStyleSheet(lineedit_alpha_style)
        self.input_img_Button.setEnabled(False)
        self.input_img_Button.setStyleSheet(button_alpha_style)
        self.input_swc_Button.setEnabled(False)
        self.input_swc_Button.setStyleSheet(button_alpha_style)
        self.input_cfg_Button.setEnabled(False)
        self.input_cfg_Button.setStyleSheet(button_alpha_style)
        self.input_save_Button.setEnabled(False)
        self.input_save_Button.setStyleSheet(button_alpha_style)
        self.x_resolution_doubleSpinBox.setEnabled(False)
        self.y_resolution_doubleSpinBox.setEnabled(False)
        self.z_resolution_doubleSpinBox.setEnabled(False)
        self.unit_comboBox.setEnabled(False)
        self.connect_checkBox.setEnabled(False)
        self.connect_checkBox.setStyleSheet(checkBox_alpha_style)
        # Hide
        self.statistics_start_Button.setEnabled(False)
        self.statistics_start_Button.setStyleSheet(button_alpha_style)

    """Restore controls"""

    def recover_control(self):
        self.analyzing = False
        self.import_img_radioButton.setEnabled(True)
        self.import_cfg_radioButton.setEnabled(True)
        self.input_img_lineEdit.setEnabled(True)
        self.input_img_lineEdit.setStyleSheet(lineedit_style)
        self.input_swc_lineEdit.setEnabled(True)
        self.input_swc_lineEdit.setStyleSheet(lineedit_style)
        self.input_cfg_lineEdit.setEnabled(True)
        self.input_cfg_lineEdit.setStyleSheet(lineedit_style)
        self.input_save_lineEdit.setEnabled(True)
        self.input_save_lineEdit.setStyleSheet(lineedit_style)
        self.input_img_Button.setEnabled(True)
        self.input_img_Button.setStyleSheet(button_style)
        self.input_swc_Button.setEnabled(True)
        self.input_swc_Button.setStyleSheet(button_style)
        self.input_cfg_Button.setEnabled(True)
        self.input_cfg_Button.setStyleSheet(button_style)
        self.input_save_Button.setEnabled(True)
        self.input_save_Button.setStyleSheet(button_style)
        self.x_resolution_doubleSpinBox.setEnabled(True)
        self.y_resolution_doubleSpinBox.setEnabled(True)
        self.z_resolution_doubleSpinBox.setEnabled(True)
        self.unit_comboBox.setEnabled(True)
        self.connect_checkBox.setEnabled(True)
        self.connect_checkBox.setStyleSheet(checkBox_style)

        self.statistics_start_Button.setEnabled(True)
        self.statistics_start_Button.setStyleSheet(button_style)

    def show_chart(self, now_item, pre_item):
        if self.analyze_chart_widget is not None:
            self.analyze_chart_widget.deleteLater()  # Properly delete and free memory
        self.analyze_chart_widget = QWidget()

        self.analyze_chart_verticalLayout.addWidget(self.analyze_chart_widget)
        self.analyze_chart_widget.setStyleSheet("border-radius: 0px; "
                                                "border: 1px solid rgb(70, 80, 100);")

        main_layout = QVBoxLayout(self.analyze_chart_widget)
        name = now_item.text()
        path = self.statistics_files_dict[name]
        print(path)
        self.create_chart(path, main_layout)

    def load_data(self, csv_path):
        # df = pd.read_csv(io.StringIO(DATA_STR), skiprows=1, index_col='Segment ID')
        df = pd.read_csv(csv_path, skiprows=1, index_col='Segment ID')
        return df

    def create_chart(self, path, main_layout):
        # Load data
        self.data = self.load_data(path)
        # Define plot options (index 0 is comprehensive, 1~6 are subplots)
        self.plot_options = [
            # "[Comprehensive] Six-in-one Overview",
            "Volume Distribution Histogram",
            "Length vs Volume Scatter Plot (colored by Tortuosity)",
            "Tortuosity Distribution Histogram",
            "Radius Parameters Box Plot (Min/Mean/Max)",
            "Surface Area vs Volume Scatter Plot + Trend Line",
            "Radius Std. Dev. Distribution Histogram"
        ]
        self.setup_ui(main_layout)
        # Use QTimer to call update_display with a delay to ensure the layout is fully stable
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(10, self.update_display)  # 10ms Delay
        # self.update_display()  # Default display comprehensive chart

    def setup_ui(self, main_layout):
        self.combo = QComboBox()
        self.combo.addItems(self.plot_options)
        self.combo.setCurrentIndex(self.idx)
        self.combo.currentIndexChanged.connect(lambda: self.update_display(0))
        self.combo.setStyleSheet(self.qdarkstyle_sheet + comboBox_style)

        # Create canvas: do not specify figsize, let canvas decide automatically
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        # Set stretch policy
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        self.export_btn = QPushButton("Export Current Chart")
        self.export_btn.clicked.connect(self.export_current)
        self.export_btn.setStyleSheet(button_style)

        # Add canvas to layout with stretch factor (e.g., 1)
        main_layout.addWidget(self.combo)
        main_layout.addWidget(self.canvas, 1)  # Stretch factor
        main_layout.addWidget(self.export_btn)

    def update_display(self, n=1):
        if n:
            idx = self.idx
        else:
            idx = self.combo.currentIndex()
            self.idx = idx
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._draw_single_plot(idx, ax)
        # Force canvas to recalculate layout
        self.figure.tight_layout()
        self.canvas.draw()

    def _draw_single_plot(self, idx, ax):
        """Draw a single subplot based on index"""
        if idx == 0:
            self._draw_volume_hist(ax)
        elif idx == 1:
            self._draw_length_volume_scatter(ax)
        elif idx == 2:
            self._draw_tortuosity_hist(ax)
        elif idx == 3:
            self._draw_radius_boxplot(ax)
        elif idx == 4:
            self._draw_surface_volume_scatter(ax)
        elif idx == 5:
            self._draw_radius_std_hist(ax)

    # ---------- Drawing functions for each subplot (data comes from self.data) ----------
    def _draw_volume_hist(self, ax):
        ax.hist(self.data['Volume'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_title('Volume Distribution')
        ax.set_xlabel('Volume')
        ax.set_ylabel('Frequency')
        ax.axvline(self.data['Volume'].mean(), color='red', linestyle='--',
                   label=f'Mean: {self.data["Volume"].mean():.2f}')
        ax.legend()

    def _draw_length_volume_scatter(self, ax):
        scatter = ax.scatter(self.data['Length'], self.data['Volume'],
                             c=self.data['Tortuosity'], cmap='viridis', alpha=0.6)
        ax.set_title('Length vs Volume (colored by Tortuosity)')
        ax.set_xlabel('Length')
        ax.set_ylabel('Volume')
        cbar = self.figure.colorbar(scatter, ax=ax)
        cbar.set_label('Tortuosity')

    def _draw_tortuosity_hist(self, ax):
        ax.hist(self.data['Tortuosity'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
        ax.set_title('Tortuosity Distribution')
        ax.set_xlabel('Tortuosity')
        ax.set_ylabel('Frequency')
        ax.axvline(self.data['Tortuosity'].mean(), color='blue', linestyle='--',
                   label=f'Mean: {self.data["Tortuosity"].mean():.2f}')
        ax.legend()

    def _draw_radius_boxplot(self, ax):
        radius_data = [self.data['Min Radius'], self.data['Mean Radius'], self.data['Max Radius']]
        # ax.boxplot(radius_data, labels=['Min', 'Mean', 'Max'])
        ax.boxplot(radius_data, tick_labels=['Min', 'Mean', 'Max'])
        ax.set_title('Radius Parameters Comparison')
        ax.set_ylabel('Radius')

    def _draw_surface_volume_scatter(self, ax):
        ax.scatter(self.data['Volume'], self.data['Surface Area'], c='green', alpha=0.5)
        ax.set_title('Surface Area vs Volume')
        ax.set_xlabel('Volume')
        ax.set_ylabel('Surface Area')
        # Add linear trend line
        z = np.polyfit(self.data['Volume'], self.data['Surface Area'], 1)
        p = np.poly1d(z)
        ax.plot(self.data['Volume'], p(self.data['Volume']), "r--", alpha=0.8,
                label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        ax.legend()

    def _draw_radius_std_hist(self, ax):
        ax.hist(self.data['Radius Std. Dev.'], bins=30, color='gold', edgecolor='black', alpha=0.7)
        ax.set_title('Radius Std. Dev. Distribution')
        ax.set_xlabel('Radius Std. Dev.')
        ax.set_ylabel('Frequency')

    # ---------- Export function ----------
    def export_current(self):
        idx = self.combo.currentIndex()
        # Chart: save current figure regardless of comprehensive or single plot
        file_path, _ = QFileDialog.getSaveFileName(None,
                                                   "Save Image",
                                                   "",
                                                   "PNG Image (*.png);;PDF (*.pdf)")
        if file_path:
            self.figure.savefig(file_path, dpi=150, bbox_inches='tight')
        # If exporting statistical table is needed, additional button can be added; here only chart export is kept

    def radius_calculate_progress(self, text, count):
        self.add_items(text, None, None, is_remove=True, count=count)

    def radius_calculate_finish(self, new_text):
        self.recover_control()
        text = "Statistics finished!"
        self.add_items(text, None, None, )
        t = (time.time() - self.start_time) / 60
        if new_text:
            self.mess(f"{new_text}, cannot perform vessel data statistics!", "Prompt", 1)
        else:
            self.mess(f"Vessel data statistics finished! Time elapsed: {t:.4f} Minute", "Prompt", 1)

    def radius_calculate_error(self, e):
        self.recover_control()
        self.mess(f"Vessel data statistics error: {e}", "Prompt", 1)

    def statistics_progress(self, text, count):
        self.add_items(text, None, None, is_remove=True, count=count)

    def statistics_finish(self, new_text):
        self.recover_control()
        text = "Statistics finished!"
        self.add_items(text, None, None, )
        t = (time.time() - self.start_time) / 60
        if new_text:
            self.mess(f"{new_text}, cannot perform vessel data statistics!", "Prompt", 1)
        else:
            self.mess(f"Vessel data statistics finished! Time elapsed: {t:.4f} Minute", "Prompt", 1)

    def statistics_error(self, e):
        self.recover_control()
        self.mess(f"Vessel data statistics error: {e}", "Prompt", 1)

    def statistics_files(self, filename, filepath):
        if not self.statistics_files_dict.get(filename, None):
            self.statistics_files_dict[filename] = filepath
            item = QListWidgetItem()
            item.setText(filename)  # List item name
            item.setFont(QFont("Microsoft YaHei", 9))  # Font
            item.setSizeHint(QSize(1600, 18))  # Set item size
            self.statistics_files_listWidget.addItem(item)

    def statistics_chart(self, filepath):
        self.chart_main.read_analyze_file(filepath)

    def add_items(self, text, img=None, seg=None, is_remove=False, count=0):
        if is_remove:
            self.count = self.statistics_result_listWidget.count()
            if count > 0:
                self.statistics_result_listWidget.takeItem(self.count - 1)
        aItem = QListWidgetItem()
        custom_widget = CustomListItem(text, self.statistics_result_listWidget, aItem, img=img, seg=seg)
        aItem.setSizeHint(custom_widget.sizeHint())
        self.statistics_result_listWidget.addItem(aItem)
        self.statistics_result_listWidget.setItemWidget(aItem, custom_widget)
        if is_remove:
            if count == 0:
                self.statistics_result_listWidget.scrollToBottom()
            return
        self.statistics_result_listWidget.scrollToBottom()

    def mess(self, text, title, nums, icon=QMessageBox.Question):
        box = QMessageBox()
        if nums == 1:
            box.setStandardButtons(QMessageBox.Ok)
        if nums == 2:
            box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setStyleSheet(messagebox_style)
        box.setText(text)
        box.setWindowTitle(title)
        box.setIcon(icon)
        # Set window flags to keep on top
        box.setWindowFlags(box.windowFlags() | Qt.WindowStaysOnTopHint)
        r = box.exec_()
        return r

    """Path navigation"""

    def toSelectPath(self, line_edit):
        path = line_edit.text()
        if path:
            try:
                if os.path.exists(path):
                    # Open folder and locate the file using explorer
                    subprocess.run(['explorer', '/select,', os.path.abspath(path)])
                else:
                    # Open folder location using QUrl
                    # Get the directory path of the file
                    folderPath = os.path.dirname(path)
                    # Try to open folder location using QUrl
                    result = QDesktopServices.openUrl(QUrl.fromLocalFile(folderPath))
                    # Check if open succeeded based on return value
                    if not result:
                        self.mess("Path does not exist!", "Prompt", 1)

            except Exception as e:
                self.mess(f"Open failed! {e}", "Prompt", 1)
                self.error_logger(e)

    """Error message"""

    def error_logger(self, e):
        self.logger.error("\n=== Error message ===")
        self.logger.error(f"Exception type: {type(e).__name__}")
        self.logger.error(f"Error message: {e}")
        self.logger.error("=== Error location ===")
        tb = sys.exc_info()[2]
        for frame in traceback.extract_tb(tb):
            self.logger.error(f"  File: {frame.filename}")
            self.logger.error(f"  Line number: {frame.lineno}")
            self.logger.error(f"  Function: {frame.name}")
            self.logger.error(f"  Code: {frame.line}\n")

    '''Select folder'''

    def select_dir(self, line_edit, look=0, save=0):
        try:
            if look and self.analyzing:
                self.toSelectPath(line_edit)
            else:
                # Create QSettings object
                settings = QSettings("MyCompany", "MyApp")
                # Read previously saved path
                if save:
                    initial_path = settings.value("VesselResultsSavePath", "")
                else:
                    initial_path = settings.value("VesselAnalysisPath", "")

                path = QFileDialog.getExistingDirectory(None, "Select folder", initial_path)

                # If a file was selected, save the selected file path
                if path:
                    if not self.contains_chinese(path):
                        if save:
                            settings.setValue("VesselResultsSavePath", path)
                        else:
                            settings.setValue("VesselAnalysisPath", path)
                        line_edit.setText(path)
                    else:
                        self.mess(f"Path contains Chinese characters or spaces", "Prompt", 1)
        except Exception as e:
            self.mess(f"Open failed! {e}", "Prompt", 1)
            self.error_logger(e)

    def select_path(self, se):
        try:
            # Create QSettings object
            settings = QSettings("MyCompany", "MyApp")
            filename = ""
            # Get file path
            if se == 1:
                initial_path = settings.value("VesselAnalysisJsonPath", "")
                path = os.path.join(initial_path, "cut_infos.json")
                filename, filetype = QFileDialog.getOpenFileName(None, "Select file", path, "Json File (*.json)")

            # If a file was selected, save the selected file path
            if filename:
                if not self.contains_chinese(filename):
                    if se == 1:
                        settings.setValue("VesselAnalysisJsonPath", str(Path(filename).parent))
                        self.input_cfg_lineEdit.setText(filename)
                else:
                    self.mess(f"Path contains Chinese characters or spaces", "Prompt", 1)
        except Exception as e:
            self.mess(f"Open failed! {e}", "Prompt", 1)
            self.error_logger(e)

    def contains_chinese(self, path):
        try:
            str(path).encode('ascii')
        except UnicodeEncodeError:
            return True
        if " " in str(path):
            return True
        return False