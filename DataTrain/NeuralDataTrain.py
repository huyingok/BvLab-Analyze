# -*- coding: utf-8 -*-
import os
import shutil
import sys
import traceback
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QListWidgetItem
from PyQt5.QtCore import Qt, QSettings, QUrl
from PyQt5.QtGui import QDesktopServices
import time
import json
import subprocess
from pathlib import Path
import webbrowser
from CustomListItem import CustomListItem2
from Unet3D.TrainQThread import NeuralDataTrainQThread
from SwcToMask.NeuralSwcToMask import NeuralSwcToMaskQThread
from config import exe_cfg
from control_style.ControlStyle import (button_alpha_style, button_style, lineedit_alpha_style, lineedit_style,
                                        messagebox_style)


"""Neural"""


class NeuralDataTrain(object):
    def __init__(self, win):
        """
        With config file: read config file
        Without config file: read images and SWC files
        :param win:
        """
        self.win = win
        self.logger = self.win.logger
        # self.TrainResultPreview = TrainResultPreview()

        self.data_train_dict = {}  # Backup parameters
        # Thread status
        self.training = False
        self.train_stop = True
        self.count = 0

        # Parent class variables
        self.train_cfg_lineEdit = self.win.train_cfg_lineEdit
        self.train_cfg_Button = self.win.train_cfg_Button
        self.train_save_lineEdit = self.win.train_save_lineEdit
        self.train_save_Button = self.win.train_save_Button
        self.epoch_spinBox = self.win.epoch_spinBox

        self.train_start_Button = self.win.train_start_Button
        self.train_end_Button = self.win.train_end_Button
        self.train_preview_Button = self.win.train_preview_Button
        # Progress bar
        self.train_result_listWidget = self.win.train_result_listWidget

        # Signal
        self.train_start_Button.clicked.connect(self.data_train_start)
        self.train_end_Button.clicked.connect(self.data_train_stop)
        self.train_preview_Button.clicked.connect(self.show_TrainResultPreview)
        self.train_cfg_Button.clicked.connect(self.select_path)
        self.train_save_Button.clicked.connect(lambda: self.select_dir(self.train_save_lineEdit, look=1, save=1))

        # Thread
        # Train
        # self.NeuralDataTrainQThread = NeuralTrainQThread(win=self)
        self.NeuralDataTrainQThread = NeuralDataTrainQThread(win=self)
        self.NeuralDataTrainQThread.finish0.connect(self.data_train_finish)
        self.NeuralDataTrainQThread.progress0.connect(self.data_train_progress)
        self.NeuralDataTrainQThread.progress1.connect(self.data_train_progress1)
        self.NeuralDataTrainQThread.error0.connect(self.data_train_error)
        self.NeuralDataTrainQThread.update_modSavePath.connect(self.train_model_update)  # Update latest model path
        # self.NeuralDataTrainQThread.update_plot.connect(self.train_plot)  # Training loss curve
        self.NeuralDataTrainQThread.preview0.connect(self.train_preview)  # Training result preview
        self.NeuralDataTrainQThread.show_preview.connect(self.show_preview_button)
        # Conversion
        self.NeuralSwcToMaskQThread = NeuralSwcToMaskQThread(win=self)
        self.NeuralSwcToMaskQThread.finish0.connect(self.swc_to_mask_finish)
        self.NeuralSwcToMaskQThread.warning0.connect(self.swc_to_mask_warning)
        self.NeuralSwcToMaskQThread.progress0.connect(self.swc_to_mask_progress)
        self.NeuralSwcToMaskQThread.progress1.connect(self.swc_to_mask_progress1)
        self.NeuralSwcToMaskQThread.error0.connect(self.swc_to_mask_error)

    def train_preview(self, img_i, mask_i, net_seg_i):
        # self.TrainResultPreview.start_preview(img_i, mask_i, net_seg_i)
        self.add_items("", img=img_i, mask=mask_i, pre=net_seg_i)

    def show_preview_button(self, is_show):
        if is_show:
            time.sleep(3)
            self.train_preview_Button.setEnabled(True)
            self.train_preview_Button.setStyleSheet(button_style)

    def train_model_update(self, path):
        # self.train_model_lineEdit.setText(path)
        try:
            if os.path.exists(path):
                makerInfo_path = self.data_train_dict.get('makerInfo_path', None)
                save_dir = self.data_train_dict['results_dir']
                if makerInfo_path is not None:
                    makerInfo_path = str(makerInfo_path)
                    if os.path.exists(makerInfo_path):
                        self.train_save_lineEdit.setText(makerInfo_path)
                        self.win.predict_cfg_lineEdit.setText(makerInfo_path)
                        name = "neural_train.pth"
                        new_path = os.path.join(save_dir, name)
                        shutil.copy(path, new_path)

                        # If more than one saved model, remove the previous one
                        model_save_root = Path(path).parent
                        if len([l for l in os.listdir(model_save_root) if ".pth" in l]) > 1:
                            self.delete_oldest_file(model_save_root)

                        with open(makerInfo_path, 'r') as f:
                            makerInfo = json.loads(f.read())
                        makerInfo['train_model'] = new_path
                        with open(makerInfo_path, 'w') as f:
                            f.write(json.dumps(makerInfo, indent=4))
        except Exception as e:
            self.mess(f"Neural model update error: {e}", "Prompt", 1)
            self.error_logger(e)

    def delete_oldest_file(self, directory):
        # Get all files in the directory
        files = [l for l in os.listdir(directory) if ".pth" in l]
        if not files:
            print("Directory is empty, no files to delete.")
            return

        # Create a list of file paths and creation times
        files_with_ctime = []
        for file in files:
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):  # Ensure it is a file, not a directory
                creation_time = os.path.getctime(file_path)
                files_with_ctime.append((file_path, creation_time))

        # Sort by creation time, oldest first
        files_with_ctime.sort(key=lambda x: x[1])

        # Delete the oldest file
        oldest_file_path = files_with_ctime[0][0]
        os.remove(oldest_file_path)
        # print(f"Deleted oldest file: {oldest_file_path}")

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

    # Define a function to open a web page
    def show_TrainResultPreview(self):
        # Call the function to open the specified web page
        port = exe_cfg.cfg['Port']['neural_port']  # Default port is 6008
        host = exe_cfg.cfg['Host']['neural_host']  # Default host is 127.0.0.1
        url = f"http://{host}:{port}/"
        # print(url)
        webbrowser.open(url)

    def train_plot(self, text_eval_path, text_loss_path, text_save_model_path):
        if self.training and not self.train_stop:
            if text_eval_path and text_loss_path and text_save_model_path:
                l0 = 10
                m = 0
                loss_list = []
                loss_indices = []
                eval_list = []
                eval_indices = []
                # prec_list = []
                # prec_indices = []
                # rec_list = []
                # rec_indices = []
                if os.path.exists(text_loss_path):
                    if os.path.getsize(text_loss_path) > 0:
                        with open(text_loss_path, "r", encoding="utf-8") as f1:
                            for i1, line in enumerate(f1.readlines()):
                                # Remove specified characters (default spaces or newlines) from both ends
                                # Split
                                line = line.strip().split(" ")
                                if i1 == 0:
                                    l0 = float(line[0])
                                loss_list.append(float(line[1]))
                                loss_indices.append(int(float(line[0]) / l0))
                        f1.close()

                if os.path.exists(text_save_model_path):
                    if os.path.getsize(text_save_model_path) > 0:
                        with open(text_save_model_path, "r", encoding="utf-8") as f2:
                            m = f2.readlines()[-1]
                            m = float(m.strip().split(" ")[0])
                        f2.close()
                if os.path.exists(text_eval_path):
                    if os.path.getsize(text_eval_path) > 0:
                        with open(text_eval_path, "r", encoding="utf-8") as f3:
                            for i3, line in enumerate(f3.readlines()):
                                line = line.strip().split(" ")
                                eval_list.append(float(line[1]))
                                eval_indices.append(int(float(line[0]) / l0))
                        f3.close()

                # with open(path4, "r", encoding="utf-8") as f4:
                #     for i4, line in enumerate(f4.readlines()):
                #         line = line.strip().split(" ")
                #         prec_list.append(float(line[1]))
                #         prec_indices.append(int(float(line[0]) / l0))
                # f4.close()
                #
                # with open(path5, "r", encoding="utf-8") as f5:
                #     for i5, line in enumerate(f5.readlines()):
                #         line = line.strip().split(" ")
                #         rec_list.append(float(line[1]))
                #         rec_indices.append(int(float(line[0]) / l0))
                # f5.close()
                #
                # f1_list = calculate_f1_list(prec_list, rec_list)
                # f1_list = [0] + f1_list

                cut_line = int(m / l0)
                loss_list = np.array([1] + loss_list)
                eval_list = np.array([0] + eval_list)
                # prec_list = [0] + prec_list
                # rec_list = [0] + rec_list
                loss_indices = np.array([0] + loss_indices)
                eval_indices = np.array([0] + eval_indices)
                # prec_indices = [0] + prec_indices
                # rec_indices = [0] + rec_indices

                # self.TrainResultPreview.update_line(loss_indices, loss_list, eval_indices, eval_list, cut_line)

                # self.TrainResultPreview.update_pg(loss_indices, loss_list, eval_indices, eval_list, cut_line)

    def data_train_start(self):
        is_keep_on = False
        try:
            if not self.training:
                epochs = self.epoch_spinBox.value()
                makerInfo_path = self.train_cfg_lineEdit.text()
                save_dir = self.train_save_lineEdit.text()

                if save_dir:
                    if not os.path.isdir(save_dir):
                        self.mess("Result save path is not a folder path, cannot train!", "Prompt", 1)
                        return
                else:
                    self.mess("Result save path is empty, cannot train!", "Prompt", 1)
                    return

                if makerInfo_path:
                    if os.path.exists(makerInfo_path):
                        with open(makerInfo_path, 'r') as f:
                            makerInfo = json.loads(f.read())

                        os.makedirs(save_dir, exist_ok=True)

                        results_dir = os.path.join(save_dir, f"TrainResults")
                        os.makedirs(results_dir, exist_ok=True)

                        checkpoint_path = makerInfo.get('checkpoint_path', None)

                        img_dir = makerInfo.get('divide_img_dir', None)
                        swc_dir = makerInfo.get('divide_swc_dir', None)
                        if img_dir is None:
                            self.mess("Configuration file has no divide_img_dir training image parameter, cannot train!", "Prompt", 1)
                            return
                        if swc_dir is None:
                            self.mess("Configuration file has no divide_swc_dir training label parameter, cannot train!", "Prompt", 1)
                            return
                        if checkpoint_path is not None:
                            if os.path.exists(checkpoint_path):
                                is_keep_on = True  # Checkpoint file exists, determine whether to continue previous training
                    else:
                        self.mess("Input configuration file does not exist, cannot train!", "Prompt", 1)
                        return
                else:
                    self.mess("Input configuration file is empty, cannot train!", "Prompt", 1)
                    return

                if img_dir:
                    if swc_dir:
                        if os.path.exists(img_dir):
                            if os.path.exists(swc_dir):
                                if is_keep_on:
                                    res = self.mess("Checkpoint file exists, continue previous training?", "Prompt", 2)
                                    if res == QMessageBox.No:
                                        is_keep_on = False
                                res = self.mess("Start training?", "Prompt", 2)
                                if res == QMessageBox.Yes:
                                    self.data_train_dict = {
                                        'makerInfo_path': makerInfo_path,
                                        'results_dir': results_dir,
                                        'save_dir': save_dir,
                                        'imgDir': img_dir,
                                        'swcDir': swc_dir,
                                        'epochs': epochs,
                                        'shapes': None,
                                        'is_keep_on': is_keep_on
                                    }
                                    self.forbid_control()
                                    self.train_stop = False
                                    self.training = True
                                    self.count = 0
                                    self.train_result_listWidget.clear()
                                    self.start_time = time.time()
                                    # Label conversion
                                    self.add_items("Starting label conversion!")
                                    # self.NeuralSwcToMaskQThread.run()
                                    self.NeuralSwcToMaskQThread.start()
                            else:
                                self.mess("Input label path does not exist, cannot train!", "Prompt", 1)
                                return
                        else:
                            self.mess("Input image path does not exist, cannot train!", "Prompt", 1)
                            return
                    else:
                        self.mess(f"Input label path is empty, cannot train!", "Prompt", 1)
                        return
                else:
                    self.mess("Input image path is empty, cannot train!", "Prompt", 1)
                    return
        except Exception as e:
            self.error_logger(e)

    """Disable controls"""

    def forbid_control(self):
        # Disable
        self.train_cfg_lineEdit.setEnabled(False)
        self.train_cfg_Button.setEnabled(False)
        self.train_save_lineEdit.setEnabled(False)
        # self.train_save_Button.setEnabled(False)
        self.train_save_Button.setText("Look")
        self.train_cfg_lineEdit.setStyleSheet(lineedit_alpha_style)
        self.train_cfg_Button.setStyleSheet(button_alpha_style)
        self.train_save_lineEdit.setStyleSheet(lineedit_alpha_style)
        # self.train_save_Button.setStyleSheet(button_alpha_style)
        self.epoch_spinBox.setEnabled(False)
        # Hide
        self.train_preview_Button.setEnabled(False)
        self.train_preview_Button.setStyleSheet(button_alpha_style)
        self.train_start_Button.setVisible(False)
        self.train_end_Button.setVisible(True)

    """Restore controls"""

    def recover_control(self):
        self.training = False
        # Restore
        self.train_cfg_lineEdit.setEnabled(True)
        self.train_cfg_Button.setEnabled(True)
        self.train_save_lineEdit.setEnabled(True)
        # self.train_save_Button.setEnabled(True)
        self.train_save_Button.setText("Open")
        self.train_cfg_lineEdit.setStyleSheet(lineedit_style)
        self.train_cfg_Button.setStyleSheet(button_style)
        self.train_save_lineEdit.setStyleSheet(lineedit_style)
        # self.train_save_Button.setStyleSheet(button_style)
        self.epoch_spinBox.setEnabled(True)
        # Hide
        # self.train_preview_Button.setEnabled(True)
        # self.train_preview_Button.setStyleSheet(button_style)
        self.train_start_Button.setVisible(True)
        self.train_end_Button.setVisible(False)

    def data_train_finish(self, new_text):
        self.recover_control()
        t = (time.time() - self.start_time) / 60
        text = f"Training finished! Time elapsed: {t:.4f} Minute"
        self.add_items(text)
        if new_text:
            self.mess(f"{new_text}, cannot train neural model!", "Prompt", 1)
        else:
            self.mess(f"Neural model training finished! Time elapsed: {t:.4f} Minute", "Prompt", 1)

    def data_train_progress(self, text):
        self.add_items(text)

    def data_train_progress1(self, text, count):
        self.add_items(text, is_remove=True, count=count)

    def data_train_error(self, e):
        self.train_stop = True
        text = f"Training error: {e}"
        self.add_items(text)
        self.mess(f"Neural model training error: {e}", "Prompt", 1)
        self.NeuralDataTrainQThread.to_stop()
        if self.NeuralDataTrainQThread.isRunning():
            self.NeuralDataTrainQThread.quit()
            if not self.NeuralDataTrainQThread.wait(5000):
                self.NeuralDataTrainQThread.terminate()
                self.NeuralDataTrainQThread.wait(3000)
        self.recover_control()

    def data_train_stop(self):
        r = self.mess("Training in progress, stop training?", "Prompt", 2)
        if r == QMessageBox.Yes:
            self.train_stop = True
            if self.NeuralDataTrainQThread.start_train:
                self.NeuralDataTrainQThread.to_stop()
                if self.NeuralDataTrainQThread.isRunning():
                    self.NeuralDataTrainQThread.quit()
                    if not self.NeuralDataTrainQThread.wait(5000):
                        self.NeuralDataTrainQThread.terminate()
                        self.NeuralDataTrainQThread.wait(3000)

    def swc_to_mask_finish(self):
        t = (time.time() - self.start_time) / 60
        text = f"Label conversion finished! Time elapsed: {t:.4f} Minute"
        self.add_items(text)
        if self.train_stop:
            self.recover_control()
            text = f"Training finished!"
            self.add_items(text)
        else:
            # Start training
            text = f"Training started!"
            self.add_items(text)
            self.start_time = time.time()
            self.NeuralDataTrainQThread.start()
            # self.TrainResultPreview.start_preview_init()

    def swc_to_mask_warning(self, text):
        self.train_stop = True
        self.mess(f"{text}", "Prompt", 1)
        self.recover_control()

    def swc_to_mask_progress(self, text, img, mask):
        self.add_items(text, img=img, mask=mask)

    def swc_to_mask_progress1(self, text, count):
        self.add_items(text, is_remove=True, count=count)

    def add_items(self, text, img=None, mask=None, pre=None, is_remove=False, count=0):
        if is_remove:
            self.count = self.train_result_listWidget.count()
            if count > 0:
                self.train_result_listWidget.takeItem(self.count - 1)
        aItem = QListWidgetItem()
        custom_widget = CustomListItem2(text, self.train_result_listWidget, aItem, img=img, mask=mask, pre=pre)
        aItem.setSizeHint(custom_widget.sizeHint())
        self.train_result_listWidget.addItem(aItem)
        self.train_result_listWidget.setItemWidget(aItem, custom_widget)
        if is_remove:
            if count == 0:
                self.train_result_listWidget.scrollToBottom()
            return
        self.train_result_listWidget.scrollToBottom()

    def swc_to_mask_error(self, e):
        self.train_stop = True
        self.mess(f"Neural label conversion error: {e}", "Prompt", 1)
        self.recover_control()

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

    '''Select folder'''

    def select_dir(self, line_edit, line_edit_2=None, look=0, save=0):
        try:
            if look and self.training:
                self.toSelectPath(line_edit)
            else:
                # Create QSettings object
                settings = QSettings("MyCompany", "MyApp")
                # Read previously saved path
                if save:
                    initial_path = settings.value("NeuralResultsSavePath", "")
                else:
                    initial_path = settings.value("NeuralTrainPath", "")
                # Get folder path
                path = QFileDialog.getExistingDirectory(None, "Select folder", initial_path)
                if path:
                    if not self.contains_chinese(path):
                        if save:
                            settings.setValue("NeuralResultsSavePath", path)
                        else:
                            settings.setValue("NeuralTrainPath", path)
                        line_edit.setText(path)
                        if line_edit_2:
                            pass
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
            initial_path = settings.value("NeuralTrainConfigPath", "")
            path = os.path.join(initial_path, "config.json")
            # Get file path
            filename, filetype = QFileDialog.getOpenFileName(None, "Select file", path, "Json File (*.json)")

            # If a file was selected, save the selected file path
            if filename:
                if not self.contains_chinese(filename):
                    settings.setValue("NeuralTrainConfigPath", str(Path(filename).parent))
                    self.train_cfg_lineEdit.setText(filename)
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
