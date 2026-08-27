# -*- coding: utf-8 -*-
import os
import shutil
import sys
import traceback
from os.path import join
from pathlib import Path
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QListWidgetItem
from PyQt5.QtCore import Qt, QSettings, QUrl
from PyQt5.QtGui import QDesktopServices
import json
import subprocess
from DataMake.DataFilterQThread import DataFilterQThread
from Unet3D.PredictQThread import NeuralDataPredictQThread
from ImageCut.LineImageCutThread import LineImageCutThread
from CustomListItem import CustomListItem
from control_style.ControlStyle import (button_alpha_style, button_style, lineedit_alpha_style, lineedit_style,
                                        messagebox_style)


"""Neural"""


class NeuralDataSetMake(object):
    def __init__(self, win):
        """
        Unlabeled dataset: cut, filter, make, predict
        Labeled dataset: cut, filter, make
        Large data:
        :param win:
        """
        self.win = win
        self.logger = self.win.logger
        # Initialization
        self.making_arguments = {}  # Backup parameters
        # Thread status
        self.cutting = False
        self.filtering = False
        self.making = False
        self.predicting = False
        self.nm = False
        self.select_input_index = 0  # Select input image
        self.count = 0  # Progress count for same type
        # Parent class variables
        """Data making"""
        self.making_start_Button = self.win.making_start_Button
        self.path_lineEdit = self.win.path_lineEdit
        self.path_widget = self.win.path_widget
        self.select_model_comboBox = self.win.select_model_comboBox
        self.data_type_comboBox = self.win.data_type_comboBox
        self.data_counts_label = self.win.data_counts_label
        self.mask_widget = self.win.mask_widget
        self.no_mask_widget = self.win.no_mask_widget
        self.tif_widget = self.win.tif_widget
        self.bv_widget = self.win.bv_widget
        self.dataNums_spinBox = self.win.dataNums_spinBox
        self.MNumber_spinBox = self.win.MNumber_spinBox
        self.data_x_spinBox = self.win.data_x_spinBox
        self.data_y_spinBox = self.win.data_y_spinBox
        self.data_z_spinBox = self.win.data_z_spinBox
        self.train_ratio_doubleSpinBox = self.win.train_ratio_doubleSpinBox
        self.val_ratio_doubleSpinBox = self.win.val_ratio_doubleSpinBox
        self.test_ratio_doubleSpinBox = self.win.test_ratio_doubleSpinBox
        self.Min_x_spinBox = self.win.Min_x_spinBox
        self.Min_y_spinBox = self.win.Min_y_spinBox
        self.Min_z_spinBox = self.win.Min_z_spinBox
        self.Max_x_spinBox = self.win.Max_x_spinBox
        self.Max_y_spinBox = self.win.Max_y_spinBox
        self.Max_z_spinBox = self.win.Max_z_spinBox
        # Default model names
        self.default_models_Dir = self.win.default_models_Dir
        self.model_names = [Path(l).stem for l in os.listdir(self.default_models_Dir) if ".pth" in l]
        self.model_names.append("Self-trained model")
        self.select_model_comboBox.clear()
        self.select_model_comboBox.addItems(self.model_names)
        # Unlabeled dataset
        self.no_mask_images_lineEdit = self.win.no_mask_images_lineEdit
        self.no_mask_images_Button = self.win.no_mask_images_Button
        self.no_mask_cfg_lineEdit = self.win.no_mask_cfg_lineEdit
        self.no_mask_cfg_Button = self.win.no_mask_cfg_Button
        # Labeled dataset
        self.making_image_lineEdit = self.win.making_image_lineEdit
        self.making_image_Button = self.win.making_image_Button
        self.making_swc_lineEdit = self.win.making_swc_lineEdit
        self.making_swc_Button = self.win.making_swc_Button
        self.making_cfg_lineEdit = self.win.making_cfg_lineEdit
        self.making_cfg_Button = self.win.making_cfg_Button
        # BV format data
        self.bv_images_lineEdit = self.win.bv_images_lineEdit
        self.bv_images_Button = self.win.bv_images_Button
        self.bv_cfg_lineEdit = self.win.bv_cfg_lineEdit
        self.bv_cfg_Button = self.win.bv_cfg_Button
        self.making_level_spinBox = self.win.making_level_spinBox

        # Progress bar
        self.making_result_listWidget = self.win.making_result_listWidget
        # Get the last index of QComboBox
        self.last_index = self.select_model_comboBox.count() - 1

        # Signal
        self.making_start_Button.clicked.connect(self.data_filter_start)
        self.select_model_comboBox.activated.connect(self.select_predict_model)
        self.data_type_comboBox.activated.connect(self.select_input_data)
        # Unlabeled dataset
        self.no_mask_images_Button.clicked.connect(lambda: self.select_dir(self.no_mask_images_lineEdit))
        self.no_mask_cfg_Button.clicked.connect(lambda: self.select_dir(self.no_mask_cfg_lineEdit, look=1, save=1))
        # Labeled dataset
        self.making_image_Button.clicked.connect(lambda: self.select_dir(self.making_image_lineEdit))
        self.making_swc_Button.clicked.connect(lambda: self.select_dir(self.making_swc_lineEdit))
        self.making_cfg_Button.clicked.connect(lambda: self.select_dir(self.making_cfg_lineEdit, look=1, save=1))
        # BV format data
        self.bv_images_Button.clicked.connect(lambda: self.select_dir(self.bv_images_lineEdit))
        self.bv_cfg_Button.clicked.connect(lambda: self.select_dir(self.bv_cfg_lineEdit, look=1, save=1))

        # Thread
        # Filter
        self.DataFilterQThread = DataFilterQThread(win=self)
        self.DataFilterQThread.finish0.connect(self.data_filter_finish)
        self.DataFilterQThread.progress0.connect(self.data_filter_progress)
        self.DataFilterQThread.error0.connect(self.data_filter_error)
        self.DataFilterQThread.make_finish.connect(self.data_set_finish)
        # Predict
        self.NeuralDataPredictQThread = NeuralDataPredictQThread(win=self, source="make")
        self.NeuralDataPredictQThread.progress0.connect(self.data_predict_progress)
        self.NeuralDataPredictQThread.progress_text.connect(self.data_predict_progress_text)
        self.NeuralDataPredictQThread.finish0.connect(self.data_predict_finish)
        self.NeuralDataPredictQThread.error0.connect(self.data_predict_error)
        # Cut
        self.LineImageCutThread = LineImageCutThread(win=self)
        self.LineImageCutThread.progress0.connect(self.data_cut_progress)
        self.LineImageCutThread.finish0.connect(self.data_cut_finish)
        self.LineImageCutThread.error0.connect(self.data_cut_error)

    """Start filtering"""

    def data_filter_start(self):
        try:
            if not self.cutting and not self.filtering and not self.making and not self.predicting:
                # Reference parameters
                dataNums, m_number, small_size, division_ratio = self.update_make_arguments()
                cut_redun_size = [4, 4, 2]
                cfg_level = 0
                img_dir = ""
                swc_dir = ""
                save_dir = ""
                model_path = ""
                MinX = self.Min_x_spinBox.value()
                MaxX = self.Max_x_spinBox.value()
                MinY = self.Min_y_spinBox.value()
                MaxY = self.Max_y_spinBox.value()
                MinZ = self.Min_z_spinBox.value()
                MaxZ = self.Max_z_spinBox.value()
                bv_ROI = [MinX, MaxX, MinY, MaxY, MinZ, MaxZ]
                # Determine input image selection
                if (self.select_input_index == 0 or
                        self.select_input_index == 2 or
                        self.select_input_index == 3):  # Unlabeled dataset / BVFormat
                    model_name = self.select_model_comboBox.currentText()
                    if model_name == "Self-trained model":
                        model_path = self.path_lineEdit.text()  # Model path
                    else:
                        # Built-in model path
                        model_path = join(self.default_models_Dir, model_name + ".pth")
                    if model_path:
                        if not os.path.exists(model_path):
                            self.mess(f"Model {model_path} does not exist, cannot start!", "Prompt", 1)
                            return
                    else:
                        self.mess(f"Model path parameter is empty, cannot start!", "Prompt", 1)
                        return

                    if self.select_input_index == 0:
                        img_dir = self.no_mask_images_lineEdit.text()  # Image folder
                        save_dir = self.no_mask_cfg_lineEdit.text()  # Save folder
                    else:
                        img_dir = self.bv_images_lineEdit.text()  # Image folder
                        save_dir = self.bv_cfg_lineEdit.text()  # Save folder
                        cfg_level = self.making_level_spinBox.value()  # Downsampling level
                    if img_dir:
                        if save_dir:
                            if not os.path.exists(img_dir):
                                self.mess(f"{img_dir} does not exist, cannot start!", "Prompt", 1)
                                return
                        else:
                            self.mess(f"Output configuration file is empty, cannot start!", "Prompt", 1)
                            return
                    else:
                        self.mess(f"Input image path is empty, cannot start!", "Prompt", 1)
                        return

                elif self.select_input_index == 1:  # Labeled dataset
                    img_dir = self.making_image_lineEdit.text()  # Image folder
                    swc_dir = self.making_swc_lineEdit.text()  # Label folder
                    save_dir = self.making_cfg_lineEdit.text()  # Save folder
                    if img_dir:
                        if swc_dir:
                            if save_dir:
                                if os.path.exists(img_dir):
                                    if not os.path.exists(swc_dir):
                                        self.mess(f"Input label path does not exist, cannot start!", "Prompt", 1)
                                        return
                                else:
                                    self.mess(f"Input image path does not exist, cannot start!", "Prompt", 1)
                                    return
                            else:
                                self.mess(f"Output configuration file is empty, cannot start!", "Prompt", 1)
                                return
                        else:
                            self.mess(f"Input label path is empty, cannot start!", "Prompt", 1)
                            return
                    else:
                        self.mess(f"Input image path is empty, cannot start!", "Prompt", 1)
                        return

                res = self.mess("Start dataset making?", "Prompt", 2)

                if res == QMessageBox.Yes:
                    if os.path.exists(save_dir):
                        if not os.path.isdir(save_dir) or "." in save_dir:
                            self.mess("Result save path is not a folder path, cannot start!", "Prompt", 1)
                            return

                    os.makedirs(save_dir, exist_ok=True)

                    results_dir = join(save_dir, f"MakeResults")
                    if os.path.exists(results_dir):
                        shutil.rmtree(results_dir, ignore_errors=True)
                    os.makedirs(results_dir, exist_ok=True)
                    info_json_path = join(save_dir, f"config.json")

                    cut_results_dir = join(results_dir, f"CutResults")
                    cut_img_dir = join(cut_results_dir, f"images")
                    cut_swc_dir = join(cut_results_dir, f"swc")

                    divide_results_dir = join(results_dir, f"DivideResults")
                    divide_img_dir = join(divide_results_dir, f"images")
                    divide_swc_dir = join(divide_results_dir, f"swc")

                    self.making_result_listWidget.clear()
                    self.count = 0
                    self.cutting = True
                    self.making = False
                    self.filtering = False
                    self.predicting = False
                    self.nm = True

                    self.making_arguments = {
                        "info_json_path": info_json_path,
                        "img_dir": img_dir,
                        "swc_dir": swc_dir,
                        "results_dir": results_dir,
                        "cut_results_dir": cut_results_dir,
                        "cut_img_dir": cut_img_dir,
                        "cut_swc_dir": cut_swc_dir,
                        "divide_results_dir": divide_results_dir,
                        "divide_img_dir": divide_img_dir,
                        "divide_swc_dir": divide_swc_dir,
                        "make_arguments": {
                            "dataNums": dataNums,
                            "m_number": m_number,
                            "small_size": small_size,
                            "division_ratio": division_ratio,
                            "cut_redun_size": cut_redun_size,
                            "cfg_level": int(cfg_level)
                        },
                        "model_path": model_path,
                        "select_input_index": self.select_input_index,
                        "bv_ROI": bv_ROI
                    }
                    # Disable
                    self.forbid_control()

                    if self.select_input_index == 2 or self.select_input_index == 3:
                        self.add_items("Filtering started!", None, None)
                        self.data_counts_label.setText("0 / 0")
                        self.DataFilterQThread.start()
                    else:
                        self.add_items("Cutting started!", None, None)
                        self.data_counts_label.setText("0 / 0")
                        self.LineImageCutThread.start()
            else:
                if self.filtering:
                    self.mess("Dataset is being filtered!", "Prompt", 1)
                elif self.cutting:
                    self.mess("Dataset is being cut!", "Prompt", 1)
                elif self.making:
                    self.mess("Dataset is being made!", "Prompt", 1)
                elif self.predicting:
                    self.mess("Dataset is being predicted!", "Prompt", 1)
        except Exception as e:
            self.error_logger(e)

    """Disable controls"""

    def forbid_control(self):
        # Disable
        self.data_type_comboBox.setEnabled(False)
        self.select_model_comboBox.setEnabled(False)
        self.path_lineEdit.setEnabled(False)
        self.path_lineEdit.setStyleSheet(lineedit_alpha_style)
        self.dataNums_spinBox.setEnabled(False)
        self.MNumber_spinBox.setEnabled(False)
        self.data_x_spinBox.setEnabled(False)
        self.data_y_spinBox.setEnabled(False)
        self.data_z_spinBox.setEnabled(False)
        self.train_ratio_doubleSpinBox.setEnabled(False)
        self.test_ratio_doubleSpinBox.setEnabled(False)
        self.val_ratio_doubleSpinBox.setEnabled(False)
        self.Min_x_spinBox.setEnabled(False)
        self.Min_y_spinBox.setEnabled(False)
        self.Min_z_spinBox.setEnabled(False)
        self.Max_x_spinBox.setEnabled(False)
        self.Max_y_spinBox.setEnabled(False)
        self.Max_z_spinBox.setEnabled(False)
        self.making_start_Button.setEnabled(False)
        self.making_start_Button.setStyleSheet(button_alpha_style)
        if self.select_input_index == 0:
            self.no_mask_images_Button.setEnabled(False)
            self.no_mask_images_Button.setStyleSheet(button_alpha_style)
            self.no_mask_cfg_Button.setText("Look")
            self.no_mask_images_lineEdit.setEnabled(False)
            self.no_mask_cfg_lineEdit.setEnabled(False)
            self.no_mask_images_lineEdit.setStyleSheet(lineedit_alpha_style)
            self.no_mask_cfg_lineEdit.setStyleSheet(lineedit_alpha_style)
        if self.select_input_index == 1:
            self.making_image_Button.setEnabled(False)
            self.making_swc_Button.setEnabled(False)
            self.making_image_Button.setStyleSheet(button_alpha_style)
            self.making_swc_Button.setStyleSheet(button_alpha_style)
            self.making_cfg_Button.setText("Look")
            self.making_image_lineEdit.setEnabled(False)
            self.making_swc_lineEdit.setEnabled(False)
            self.making_cfg_lineEdit.setEnabled(False)
            self.making_image_lineEdit.setStyleSheet(lineedit_alpha_style)
            self.making_swc_lineEdit.setStyleSheet(lineedit_alpha_style)
            self.making_cfg_lineEdit.setStyleSheet(lineedit_alpha_style)
        if self.select_input_index == 2 or self.select_input_index == 3:
            self.bv_images_Button.setEnabled(False)
            self.bv_images_Button.setStyleSheet(button_alpha_style)
            self.bv_cfg_Button.setText("Look")
            self.bv_images_lineEdit.setEnabled(False)
            self.bv_cfg_lineEdit.setEnabled(False)
            self.bv_images_lineEdit.setStyleSheet(lineedit_alpha_style)
            self.bv_cfg_lineEdit.setStyleSheet(lineedit_alpha_style)
            self.making_level_spinBox.setEnabled(False)

    """Restore controls"""

    def recover_control(self):
        # Whether threads are running
        self.cutting = False
        self.making = False
        self.filtering = False
        self.predicting = False
        self.nm = False
        # Re-enable
        self.data_type_comboBox.setEnabled(True)
        self.select_model_comboBox.setEnabled(True)
        self.path_lineEdit.setEnabled(True)
        self.path_lineEdit.setStyleSheet(lineedit_style)
        self.dataNums_spinBox.setEnabled(True)
        self.MNumber_spinBox.setEnabled(True)
        self.data_x_spinBox.setEnabled(True)
        self.data_y_spinBox.setEnabled(True)
        self.data_z_spinBox.setEnabled(True)
        self.train_ratio_doubleSpinBox.setEnabled(True)
        self.test_ratio_doubleSpinBox.setEnabled(True)
        self.val_ratio_doubleSpinBox.setEnabled(True)
        self.Min_x_spinBox.setEnabled(True)
        self.Min_y_spinBox.setEnabled(True)
        self.Min_z_spinBox.setEnabled(True)
        self.Max_x_spinBox.setEnabled(True)
        self.Max_y_spinBox.setEnabled(True)
        self.Max_z_spinBox.setEnabled(True)
        self.making_start_Button.setEnabled(True)
        self.making_start_Button.setStyleSheet(button_style)
        if self.select_input_index == 0:
            self.no_mask_images_Button.setEnabled(True)
            self.no_mask_images_Button.setStyleSheet(button_style)
            self.no_mask_cfg_Button.setText("Open")
            self.no_mask_images_lineEdit.setEnabled(True)
            self.no_mask_cfg_lineEdit.setEnabled(True)
            self.no_mask_images_lineEdit.setStyleSheet(lineedit_style)
            self.no_mask_cfg_lineEdit.setStyleSheet(lineedit_style)
        if self.select_input_index == 1:
            self.making_image_Button.setEnabled(True)
            self.making_swc_Button.setEnabled(True)
            self.making_image_Button.setStyleSheet(button_style)
            self.making_swc_Button.setStyleSheet(button_style)
            self.making_cfg_Button.setText("Open")
            self.making_image_lineEdit.setEnabled(True)
            self.making_swc_lineEdit.setEnabled(True)
            self.making_cfg_lineEdit.setEnabled(True)
            self.making_image_lineEdit.setStyleSheet(lineedit_style)
            self.making_swc_lineEdit.setStyleSheet(lineedit_style)
            self.making_cfg_lineEdit.setStyleSheet(lineedit_style)
        if self.select_input_index == 2 or self.select_input_index == 3:
            self.bv_images_Button.setEnabled(True)
            self.bv_images_Button.setStyleSheet(button_style)
            self.bv_cfg_Button.setText("Open")
            self.bv_images_lineEdit.setEnabled(True)
            self.bv_cfg_lineEdit.setEnabled(True)
            self.bv_images_lineEdit.setStyleSheet(lineedit_style)
            self.bv_cfg_lineEdit.setStyleSheet(lineedit_style)
            self.making_level_spinBox.setEnabled(True)

    """Select prediction model"""

    def select_predict_model(self, index):
        if index == self.last_index or index == -1:  # Custom
            self.path_widget.show()
            # Select path
            self.select_path()
        else:
            self.path_widget.hide()

    """Select input image"""

    def select_input_data(self):
        self.select_input_index = self.data_type_comboBox.currentIndex()
        if self.select_input_index == 0:  # Unlabeled data
            self.mask_widget.hide()
            self.no_mask_widget.show()
            self.tif_widget.show()
            self.bv_widget.hide()
        if self.select_input_index == 1:  # Labeled data
            self.no_mask_widget.hide()
            self.mask_widget.show()
        if self.select_input_index == 2:  # BV format data
            self.mask_widget.hide()
            self.no_mask_widget.show()
            self.tif_widget.hide()
            self.bv_widget.show()
        if self.select_input_index == 3:  # OME-Zarr format data
            self.mask_widget.hide()
            self.no_mask_widget.show()
            self.tif_widget.hide()
            self.bv_widget.show()

    """Dataset filtering"""

    def data_filter_progress(self, text, count):
        self.add_items(text, None, None, is_remove=True, count=count)

    def data_filter_finish(self, res):
        self.filtering = False
        self.data_counts_label.setText(res)
        self.add_items("Filtering finished!", None, None)

    def update_make_arguments(self):
        # Update parameters
        dataNums = self.dataNums_spinBox.value()
        m_number = self.MNumber_spinBox.value()
        small_size = [self.data_x_spinBox.value(),
                      self.data_y_spinBox.value(),
                      self.data_z_spinBox.value()]
        division_ratio = [self.train_ratio_doubleSpinBox.value(),
                          self.val_ratio_doubleSpinBox.value(),
                          self.test_ratio_doubleSpinBox.value()]
        return dataNums, m_number, small_size, division_ratio

    def data_filter_error(self, e):
        self.mess(f"Neural dataset filtering error: {e}", "Prompt", 1)
        self.recover_control()

    def data_set_finish(self):
        # Start prediction
        self.making = False
        if (self.select_input_index == 0 or
                self.select_input_index == 2 or
                self.select_input_index == 3):
            self.predicting = True
            self.add_items("Prediction started!", None, None)
            self.NeuralDataPredictQThread.start()
        if self.select_input_index == 1:
            self.recover_control()
            text = "Dataset making finished!"
            self.add_items(text, None, None)
            info_json_path = self.making_arguments['info_json_path']

            try:
                with open(info_json_path, 'w') as f:
                    f.write(json.dumps(self.making_arguments, indent=4))
            except PermissionError:
                if os.path.exists(info_json_path):
                    shutil.rmtree(info_json_path)
                with open(info_json_path, 'w') as f:
                    f.write(json.dumps(self.making_arguments, indent=4))

            self.win.train_cfg_lineEdit.setText(info_json_path)
            # self.win.predict_cfg_lineEdit.setText(info_json_path)
            self.making_cfg_lineEdit.setText(info_json_path)

            self.mess(f"Neural dataset making finished!", "Prompt", 1)

    """Predict"""

    def data_predict_progress(self, text, img, seg):
        self.add_items(text, img, seg)

    def data_predict_progress_text(self, text, count):
        self.add_items(text, None, None, is_remove=True, count=count)

    def data_predict_finish(self, new_text):
        self.recover_control()
        text = "Dataset making finished!"
        self.add_items(text, None, None)

        info_json_path = self.making_arguments['info_json_path']

        try:
            with open(info_json_path, 'w') as f:
                f.write(json.dumps(self.making_arguments, indent=4))
        except PermissionError:
            if os.path.exists(info_json_path):
                shutil.rmtree(info_json_path)
            with open(info_json_path, 'w') as f:
                f.write(json.dumps(self.making_arguments, indent=4))

        self.win.train_cfg_lineEdit.setText(info_json_path)
        # self.win.predict_cfg_lineEdit.setText(info_json_path)
        if self.select_input_index == 0:
            self.no_mask_cfg_lineEdit.setText(info_json_path)
        if self.select_input_index == 1:
            self.making_cfg_lineEdit.setText(info_json_path)
        if self.select_input_index == 2 or self.select_input_index == 3:
            self.bv_cfg_lineEdit.setText(info_json_path)
        if new_text:
            self.mess(f"{new_text}, cannot make neural dataset (predict)!", "Prompt", 1)
        else:
            self.mess(f"Neural dataset making finished!", "Prompt", 1)

    def data_predict_error(self, e):
        self.mess(f"Neural dataset making error: {e}", "Prompt", 1)
        self.recover_control()

    """Image cutting"""

    def data_cut_progress(self, text, count):
        self.add_items(text, None, None, is_remove=True, count=count)

    def data_cut_finish(self, text, cut_image_count):
        self.cutting = False
        self.add_items(text, None, None)
        if cut_image_count == 0:
            self.mess("Removing blank images, got 0 blocks after cutting, please select input image again!", "Prompt", 1)
            self.recover_control()
        else:
            self.filtering = True
            self.add_items("Filtering started!", None, None)
            self.DataFilterQThread.start()

    def data_cut_error(self, e):
        self.mess(f"Neural image cutting error: {e}", "Prompt", 1)
        self.recover_control()

    def add_items(self, text, img=None, seg=None, is_remove=False, count=0):
        if is_remove:
            self.count = self.making_result_listWidget.count()
            if count > 0:
                self.making_result_listWidget.takeItem(self.count - 1)
        aItem = QListWidgetItem()
        custom_widget = CustomListItem(text, self.making_result_listWidget, aItem, img=img, seg=seg)
        aItem.setSizeHint(custom_widget.sizeHint())
        self.making_result_listWidget.addItem(aItem)
        self.making_result_listWidget.setItemWidget(aItem, custom_widget)
        if is_remove:
            if count == 0:
                self.making_result_listWidget.scrollToBottom()
            return
        self.making_result_listWidget.scrollToBottom()

    """Navigate to configuration file path"""

    def toCfgPath(self, line_edit):
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
            if look and (self.cutting or self.filtering or self.making or self.predicting):
                self.toCfgPath(line_edit)
            else:
                # Create QSettings object
                settings = QSettings("MyCompany", "MyApp")
                # Read previously saved path
                if save:
                    initial_path = settings.value("NeuralResultsSavePath", "")
                else:
                    initial_path = settings.value("NeuralMakePath", "")
                # Get folder path
                path = QFileDialog.getExistingDirectory(None, "Select folder", initial_path)
                if path:
                    if not self.contains_chinese(path):
                        if save:
                            settings.setValue("NeuralResultsSavePath", path)
                        else:
                            settings.setValue("NeuralMakePath", path)
                        line_edit.setText(path)
                    else:
                        self.mess(f"Path contains Chinese characters or spaces, please select again!", "Prompt", 1)
        except Exception as e:
            self.mess(f"Open failed! {e}", "Prompt", 1)
            self.error_logger(e)

    def select_path(self):
        try:
            # Create QSettings object
            settings = QSettings("MyCompany", "MyApp")
            # Read previously saved path
            initial_path = settings.value("NeuralMakeModelPath", "")
            # Get file path
            filename, filetype = QFileDialog.getOpenFileName(None, "Select model file", initial_path, "Model File (*.pth)")

            if filename:
                if not self.contains_chinese(filename):
                    settings.setValue("NeuralMakeModelPath", str(Path(filename).parent))
                    self.path_lineEdit.setText(filename)
                else:
                    self.mess(f"Path contains Chinese characters or spaces", "Prompt", 1)
        except Exception as e:
            self.mess(f"Open failed! {e}", "Prompt", 1)
            self.error_logger(e)

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

    def contains_chinese(self, path):
        try:
            str(path).encode('ascii')
        except UnicodeEncodeError:
            return True
        if " " in str(path):
            return True
        return False
