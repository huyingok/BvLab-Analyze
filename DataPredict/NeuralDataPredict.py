# -*- coding: utf-8 -*-
from os.path import join
import os
import sys
import traceback
from pathlib import Path
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QListWidgetItem
from PyQt5.QtCore import Qt, QSettings, QUrl
from PyQt5.QtGui import QDesktopServices
import time
import json
import subprocess
from CustomListItem import CustomListItem
from Unet3D.PredictQThread import NeuralDataPredictQThread
from ImageSplice.LineImgSpliceQThread import ImgSpliceQThread
from control_style.ControlStyle import (button_alpha_style, button_style, lineedit_alpha_style, lineedit_style,
                                        messagebox_style)


"""Neural"""


class NeuralDataPredict(object):
    def __init__(self, win):
        self.win = win
        self.logger = self.win.logger
        # Initialization
        self.data_predict_dict = {}
        self.img_splice_dict = {}
        self.predicting = False
        self.splicing = False
        self.count = 0
        self.cut_infos_path = ""

        # Parent class variables
        self.splice_widget = self.win.splice_widget
        self.pred_widget = self.win.pred_widget
        self.predict_cfg_images_lineEdit = self.win.predict_cfg_images_lineEdit
        self.predict_cfg_lineEdit = self.win.predict_cfg_lineEdit
        self.predict_cfg_save_lineEdit = self.win.predict_cfg_save_lineEdit
        self.predict_cfg_images_Button = self.win.predict_cfg_images_Button
        self.predict_cfg_comboBox = self.win.predict_cfg_comboBox
        self.predict_cfg_save_Button = self.win.predict_cfg_save_Button
        self.predict_start_Button = self.win.predict_start_Button
        self.predict_radioButton = self.win.predict_radioButton
        self.splice_cfg_lineEdit = self.win.splice_cfg_lineEdit
        self.splice_save_lineEdit = self.win.splice_save_lineEdit
        self.splice_start_Button = self.win.splice_start_Button
        self.splice_cfg_Button = self.win.splice_cfg_Button
        self.splice_save_Button = self.win.splice_save_Button
        self.splice_radioButton = self.win.splice_radioButton
        self.predict_level_spinBox = self.win.predict_level_spinBox
        self.predict_stackedWidget = self.win.predict_stackedWidget
        self.bv_ROI_checkBox = self.win.bv_ROI_checkBox
        self.bv_ROI_widget = self.win.bv_ROI_widget
        self.MinX_spinBox = self.win.MinX_spinBox
        self.MinY_spinBox = self.win.MinY_spinBox
        self.MinZ_spinBox = self.win.MinZ_spinBox
        self.MaxX_spinBox = self.win.MaxX_spinBox
        self.MaxY_spinBox = self.win.MaxY_spinBox
        self.MaxZ_spinBox = self.win.MaxZ_spinBox
        # Default model names
        self.default_models_Dir = self.win.default_models_Dir
        self.model_names = [Path(l).stem for l in os.listdir(self.default_models_Dir) if ".pth" in l]
        self.model_names.append("Select model")
        self.model_names.append("Select file")
        self.predict_cfg_comboBox.addItems(self.model_names)
        # Last selection
        self.last_index = self.predict_cfg_comboBox.count() - 1
        self.predict_cfg_comboBox.setCurrentIndex(self.last_index)
        # Progress
        self.predict_result_listWidget = self.win.predict_result_listWidget
        # Signal
        self.predict_radioButton.toggled.connect(lambda: self.radioButton_toggled(0))
        self.splice_radioButton.toggled.connect(lambda: self.radioButton_toggled(1))
        self.bv_ROI_checkBox.clicked.connect(self.on_set_bv_ROI)
        self.predict_start_Button.clicked.connect(self.data_predict_start)
        self.predict_cfg_comboBox.activated.connect(self.comboBox_select)
        self.predict_cfg_images_Button.clicked.connect(lambda: self.select_dir(self.predict_cfg_images_lineEdit))
        self.predict_cfg_save_Button.clicked.connect(lambda: self.select_dir(self.predict_cfg_save_lineEdit, look=1, save=1))
        self.splice_start_Button.clicked.connect(self.img_splice_start)
        self.splice_cfg_Button.clicked.connect(lambda: self.select_path(1))
        self.splice_save_Button.clicked.connect(lambda: self.select_dir(self.splice_save_lineEdit, save=1))
        # Thread
        self.NeuralDataPredictQThread = NeuralDataPredictQThread(win=self, source="pred")
        self.NeuralDataPredictQThread.finish0.connect(self.data_predict_finish)
        self.NeuralDataPredictQThread.progress0.connect(self.data_predict_progress)
        self.NeuralDataPredictQThread.progress_text.connect(self.data_predict_progress_text)
        self.NeuralDataPredictQThread.error0.connect(self.data_predict_error)

        self.ImgSpliceQThread = ImgSpliceQThread(win=self)
        self.ImgSpliceQThread.finish0.connect(self.img_splice_finish)
        self.ImgSpliceQThread.progress0.connect(self.img_splice_progress)
        self.ImgSpliceQThread.error0.connect(self.img_splice_error)

    def radioButton_toggled(self, index):
        self.predict_stackedWidget.setCurrentIndex(index)
        if index == 0:  # Predict
            self.predict_start_Button.setVisible(True)
            self.splice_start_Button.setVisible(False)
        if index == 1:
            self.predict_start_Button.setVisible(False)
            self.splice_start_Button.setVisible(True)

    def on_set_bv_ROI(self):
        if self.bv_ROI_checkBox.isChecked():
            self.bv_ROI_widget.setVisible(True)
        else:
            self.bv_ROI_widget.setVisible(False)

    def comboBox_select(self, index):
        if index == self.last_index or index == -1:  # Custom
            # Select path
            self.select_path(0)
        else:
            text = self.predict_cfg_comboBox.currentText()
            if text == "Select model":
                # Select path
                self.select_path(2)
            else:
                self.predict_cfg_lineEdit.setText(text + ".json")

    def img_splice_start(self):
        try:
            if not self.splicing:
                cfg_path = self.splice_cfg_lineEdit.text()
                save_dir = self.splice_save_lineEdit.text()
                if cfg_path:
                    if os.path.exists(cfg_path):
                        with open(cfg_path, "r") as f:
                            cut_infos = json.loads(f.read())
                        dataType = cut_infos.get("dataType", None)
                        if dataType is None:
                            self.mess("Configuration file has no block information, cannot splice!", "Prompt", 1)
                            return
                    else:
                        self.mess("Input configuration file does not exist, cannot splice!", "Prompt", 1)
                        return
                else:
                    self.mess("Input configuration file is empty, cannot splice!", "Prompt", 1)
                    return
                if save_dir:
                    if os.path.exists(save_dir):
                        if not os.path.isdir(save_dir) or "." in save_dir:
                            self.mess("Result save path is not a folder path, cannot start!", "Prompt", 1)
                            return
                    else:
                        os.makedirs(save_dir, exist_ok=True)
                    res = self.mess("Start prediction?", "Prompt", 2)
                    if res == QMessageBox.Yes:
                        self.img_splice_dict = {
                            "cut_infos": cut_infos,
                            "cfg_path": cfg_path,
                            "save_dir": save_dir
                        }
                        # Disable
                        self.splice_forbid_control()
                        self.splicing = True
                        self.predict_result_listWidget.clear()
                        self.count = 0
                        self.start_time = time.time()
                        self.add_items("Splicing started!", None, None, )
                        self.ImgSpliceQThread.start()
                else:
                    self.mess("Save result path is empty, cannot splice!", "Prompt", 1)
                    return
        except Exception as e:
            self.error_logger(e)

    def splice_forbid_control(self):
        # Disable
        self.predict_radioButton.setEnabled(False)
        self.splice_radioButton.setEnabled(False)
        self.predict_level_spinBox.setEnabled(False)
        self.splice_cfg_lineEdit.setEnabled(False)
        self.splice_cfg_lineEdit.setStyleSheet(lineedit_alpha_style)
        self.splice_save_lineEdit.setEnabled(False)
        self.splice_save_lineEdit.setStyleSheet(lineedit_alpha_style)
        self.splice_cfg_Button.setEnabled(False)
        self.splice_cfg_Button.setStyleSheet(button_alpha_style)
        self.splice_save_Button.setEnabled(False)
        self.splice_save_Button.setStyleSheet(button_alpha_style)
        # Hide
        self.splice_start_Button.setEnabled(False)
        self.splice_start_Button.setStyleSheet(button_alpha_style)

    def splice_recover_control(self):
        self.splicing = False
        self.predict_radioButton.setEnabled(True)
        self.splice_radioButton.setEnabled(True)
        self.predict_level_spinBox.setEnabled(True)
        self.splice_cfg_lineEdit.setEnabled(True)
        self.splice_cfg_lineEdit.setStyleSheet(lineedit_style)
        self.splice_save_lineEdit.setEnabled(True)
        self.splice_save_lineEdit.setStyleSheet(lineedit_style)
        self.splice_cfg_Button.setEnabled(True)
        self.splice_cfg_Button.setStyleSheet(button_style)
        self.splice_save_Button.setEnabled(True)
        self.splice_save_Button.setStyleSheet(button_style)
        # Hide
        self.splice_start_Button.setEnabled(True)
        self.splice_start_Button.setStyleSheet(button_style)

    def img_splice_progress(self, text, count):
        self.add_items(text, None, None, is_remove=True, count=count)

    def img_splice_finish(self, text):
        self.splice_recover_control()
        self.add_items(text, None, None, )
        t = (time.time() - self.start_time) / 60
        self.mess(f"Neural image splicing finished! Time elapsed: {t:.4f} Minute", "Prompt", 1)

    def img_splice_error(self, e):
        self.splice_recover_control()
        self.mess(f"Neural image splicing error: {e}", "Prompt", 1)

    def data_predict_start(self):
        # Check whether to continue prediction
        is_keep_on = 0  # Restart prediction
        try:
            if not self.predicting:
                small_size = [192, 192, 192]
                image_dir = self.predict_cfg_images_lineEdit.text()
                save_dir = self.predict_cfg_save_lineEdit.text()
                makerInfo_path = self.predict_cfg_lineEdit.text()
                cfg_level = self.predict_level_spinBox.value()
                bv_ROI_check = self.bv_ROI_checkBox.isChecked()
                bv_ROI = []
                if bv_ROI_check:
                    MinX = self.MinX_spinBox.value()
                    MaxX = self.MaxX_spinBox.value()
                    MinY = self.MinY_spinBox.value()
                    MaxY = self.MaxY_spinBox.value()
                    MinZ = self.MinZ_spinBox.value()
                    MaxZ = self.MaxZ_spinBox.value()
                    if (MaxX - MinX < small_size[0]
                            or MaxY - MinY < small_size[1]
                            or MaxZ - MinZ < small_size[2]):
                        self.mess(
                            f"The minimum size (Max-Min) of the ROI region in BV format cannot be less than [x, y, z]={small_size}. Please reset the ROI range!",
                            "Prompt", 1)
                        return
                    else:
                        bv_ROI = [MinX, MaxX, MinY, MaxY, MinZ, MaxZ]
                if self.predict_cfg_comboBox.currentText() == "Select file" and ".json" in makerInfo_path:
                    if makerInfo_path:
                        if os.path.exists(makerInfo_path):
                            with open(makerInfo_path, 'r') as f:
                                makerInfo = json.loads(f.read())
                            model_path = makerInfo.get('train_model', None)
                            make_arguments = makerInfo.get('make_arguments', None)
                            if model_path is None:
                                self.mess("Configuration file has no training model path, cannot predict!", "Prompt", 1)
                                return
                            if make_arguments is None:
                                self.mess("Configuration file has no make_arguments parameter, cannot predict!", "Prompt", 1)
                                return
                            else:
                                small_size = make_arguments.get('small_size', None)
                                if small_size is None:
                                    self.mess("Configuration file has no small_size parameter, cannot predict!", "Prompt", 1)
                                    return
                        else:
                            self.mess("Input configuration file does not exist, cannot predict!", "Prompt", 1)
                            return
                    else:
                        self.mess("Input configuration file is empty, cannot predict!", "Prompt", 1)
                        return
                else:
                    if self.predict_cfg_comboBox.currentText() == "Select model" and ".pth" in makerInfo_path:
                        model_path = makerInfo_path
                    else:
                        name = self.predict_cfg_comboBox.currentText()
                        model_path = join(self.default_models_Dir, name + ".pth")
                # print(model_path)
                if image_dir:
                    if save_dir:
                        if not os.path.isdir(save_dir):
                            self.mess("Result save path is not a folder path, cannot start!", "Prompt", 1)
                            return
                        if model_path:
                            print(model_path)
                            if os.path.isdir(image_dir):
                                if os.path.exists(model_path):
                                    res = self.mess("Continue prediction?\n\nYes: Continue prediction  No: Restart prediction", "Question", 2)
                                    if res == QMessageBox.Yes:
                                        is_keep_on = 1
                                    res = self.mess("Parameter update completed, start prediction?", "Prompt", 2)
                                    if res == QMessageBox.Yes:
                                        self.data_predict_dict = {
                                            "imageDir": image_dir,
                                            "saveDir": save_dir,
                                            "modelPath": model_path,
                                            "small_size": small_size,
                                            "cfg_level": int(cfg_level),
                                            "is_keep_on": is_keep_on,
                                            "bv_ROI": bv_ROI
                                        }
                                        # Disable
                                        self.forbid_control()
                                        self.cut_infos_path = ""
                                        self.predicting = True
                                        self.predict_result_listWidget.clear()
                                        self.count = 0
                                        self.start_time = time.time()
                                        self.add_items("Prediction started!", None, None)
                                        self.NeuralDataPredictQThread.start()
                                else:
                                    self.mess("Model path does not exist, cannot predict!", "Prompt", 1)
                                    return
                            else:
                                self.mess("Input image path does not exist, cannot predict!", "Prompt", 1)
                                return
                        else:
                            self.mess("Model path is empty, cannot predict!", "Prompt", 1)
                            return
                    else:
                        self.mess("Save result path is empty, cannot predict!", "Prompt", 1)
                        return
                else:
                    self.mess("Input image path is empty, cannot predict!", "Prompt", 1)
                    return
        except Exception as e:
            self.error_logger(e)

    """Disable controls"""

    def forbid_control(self):
        # Disable
        self.predict_radioButton.setEnabled(False)
        self.splice_radioButton.setEnabled(False)
        self.predict_level_spinBox.setEnabled(False)
        self.predict_cfg_lineEdit.setEnabled(False)
        self.predict_cfg_lineEdit.setStyleSheet(lineedit_alpha_style)
        self.predict_cfg_images_lineEdit.setEnabled(False)
        self.predict_cfg_images_lineEdit.setStyleSheet(lineedit_alpha_style)
        self.predict_cfg_images_Button.setEnabled(False)
        self.predict_cfg_images_Button.setStyleSheet(button_alpha_style)
        self.predict_cfg_save_lineEdit.setEnabled(False)
        self.predict_cfg_save_lineEdit.setStyleSheet(lineedit_alpha_style)
        self.predict_cfg_comboBox.setEnabled(False)
        self.predict_cfg_save_Button.setText("Look")
        self.bv_ROI_checkBox.setEnabled(False)
        self.MinX_spinBox.setEnabled(False)
        self.MaxX_spinBox.setEnabled(False)
        self.MinY_spinBox.setEnabled(False)
        self.MaxY_spinBox.setEnabled(False)
        self.MinZ_spinBox.setEnabled(False)
        self.MaxZ_spinBox.setEnabled(False)
        self.predict_start_Button.setEnabled(False)
        self.predict_start_Button.setStyleSheet(button_alpha_style)

    """Restore controls"""

    def recover_control(self):
        self.predicting = False
        self.predict_radioButton.setEnabled(True)
        self.splice_radioButton.setEnabled(True)
        self.predict_level_spinBox.setEnabled(True)
        self.predict_cfg_lineEdit.setEnabled(True)
        self.predict_cfg_lineEdit.setStyleSheet(lineedit_style)
        self.predict_cfg_images_lineEdit.setEnabled(True)
        self.predict_cfg_images_lineEdit.setStyleSheet(lineedit_style)
        self.predict_cfg_images_Button.setEnabled(True)
        self.predict_cfg_images_Button.setStyleSheet(button_style)
        self.predict_cfg_save_lineEdit.setEnabled(True)
        self.predict_cfg_save_lineEdit.setStyleSheet(lineedit_style)
        self.predict_cfg_comboBox.setEnabled(True)
        self.predict_cfg_save_Button.setText("Open")
        self.bv_ROI_checkBox.setEnabled(True)
        self.MinX_spinBox.setEnabled(True)
        self.MaxX_spinBox.setEnabled(True)
        self.MinY_spinBox.setEnabled(True)
        self.MaxY_spinBox.setEnabled(True)
        self.MinZ_spinBox.setEnabled(True)
        self.MaxZ_spinBox.setEnabled(True)
        self.predict_start_Button.setEnabled(True)
        self.predict_start_Button.setStyleSheet(button_style)

    def data_predict_progress(self, text, img, seg):
        self.add_items(text, img=img, seg=seg)

    def data_predict_progress_text(self, text, count):
        self.add_items(text, None, None, is_remove=True, count=count)

    def data_predict_finish(self, new_text):
        self.recover_control()
        text = "Prediction finished!"
        self.add_items(text, None, None)
        t = (time.time() - self.start_time) / 60
        if new_text:
            self.mess(f"{new_text}, cannot predict neural data!", "Prompt", 1)
        else:
            self.mess(f"Neural data prediction finished! Time elapsed: {t:.4f} Minute", "Prompt", 1)
        self.splice_cfg_lineEdit.setText(self.cut_infos_path)

    def data_predict_error(self, e):
        self.recover_control()
        self.mess(f"Neural data prediction error: {e}", "Prompt", 1)

    def add_items(self, text, img=None, seg=None, is_remove=False, count=0):
        if is_remove:
            self.count = self.predict_result_listWidget.count()
            if count > 0:
                self.predict_result_listWidget.takeItem(self.count - 1)
        aItem = QListWidgetItem()
        custom_widget = CustomListItem(text, self.predict_result_listWidget, aItem, img=img, seg=seg)
        aItem.setSizeHint(custom_widget.sizeHint())
        self.predict_result_listWidget.addItem(aItem)
        self.predict_result_listWidget.setItemWidget(aItem, custom_widget)
        if is_remove:
            if count == 0:
                self.predict_result_listWidget.scrollToBottom()
            return
        self.predict_result_listWidget.scrollToBottom()

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
                    folderPath = os.path.dirname(path)
                    result = QDesktopServices.openUrl(QUrl.fromLocalFile(folderPath))
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
            if look and self.predicting:
                self.toSelectPath(line_edit)
            else:
                # Create QSettings object
                settings = QSettings("MyCompany", "MyApp")
                # Read previously saved path
                if save:
                    initial_path = settings.value("NeuralResultsSavePath", "")
                else:
                    initial_path = settings.value("NeuralPredictPath", "")

                path = QFileDialog.getExistingDirectory(None, "Select folder", initial_path)

                if path:
                    if not self.contains_chinese(path):
                        if save:
                            settings.setValue("NeuralResultsSavePath", path)
                        else:
                            settings.setValue("NeuralPredictPath", path)
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
            if se == 0:
                initial_path = settings.value("NeuralPredictConfigPath", "")
                path = os.path.join(initial_path, "config.json")
                filename, filetype = QFileDialog.getOpenFileName(None, "Select file", path, "Json File (*.json)")
            if se == 1:
                initial_path = settings.value("NeuralPredictJsonPath", "")
                path = os.path.join(initial_path, "cut_infos.json")
                filename, filetype = QFileDialog.getOpenFileName(None, "Select file", path, "Json File (*.json)")
            if se == 2:
                initial_path = settings.value("NeuralPredictModelPath", "")
                filename, filetype = QFileDialog.getOpenFileName(None, "Select model file", initial_path,
                                                                 "Model File (*.pth)")

            if filename:
                if not self.contains_chinese(filename):
                    if se == 0:
                        settings.setValue("NeuralPredictConfigPath", str(Path(filename).parent))
                        self.predict_cfg_lineEdit.setText(filename)
                    if se == 1:
                        settings.setValue("NeuralPredictJsonPath", str(Path(filename).parent))
                        self.splice_cfg_lineEdit.setText(filename)
                    if se == 2:
                        settings.setValue("NeuralPredictModelPath", str(Path(filename).parent))
                        self.predict_cfg_lineEdit.setText(filename)
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
