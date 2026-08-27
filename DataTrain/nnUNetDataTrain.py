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
from PIL import Image
import webbrowser
from CustomListItem import CustomListItem2
from nnUNet.TrainQThread import nnUNetDataTrainQThread
from SwcToMask.nnUNetSwcToMask import nnUNetSwcToMaskQThread
from config import exe_cfg
from control_style.ControlStyle import (button_alpha_style, button_style, lineedit_alpha_style, lineedit_style,
                                        messagebox_style)


"""nnUNet"""


class nnUNetDataTrain(object):
    def __init__(self, win):
        """
        With config file: read config file
        Without config file: read images and SWC files
        :param win:
        """
        self.win = win
        self.logger = self.win.logger

        self.data_train_dict = {}  # Backup parameters
        # Thread status
        self.training = False
        self.train_stop = True
        self.count = 0
        self.progress_png_path = None

        # Parent class variables
        self.train_images_lineEdit = self.win.train_images_lineEdit
        self.train_images_Button = self.win.train_images_Button
        self.train_SWC_lineEdit = self.win.train_SWC_lineEdit
        self.train_SWC_Button = self.win.train_SWC_Button
        self.train_save_lineEdit = self.win.train_save_lineEdit
        self.train_save_Button = self.win.train_save_Button
        self.images_type_combo = self.win.images_type_combo
        self.train_epochs_combo = self.win.train_epochs_combo
        self.continue_training_checkBox = self.win.continue_training_checkBox
        self.continue_training_widget = self.win.continue_training_widget
        self.continue_training_lineEdit = self.win.continue_training_lineEdit
        self.continue_training_Button = self.win.continue_training_Button
        self.no_continue_training_widget = self.win.no_continue_training_widget

        self.train_start_Button = self.win.train_start_Button
        self.train_end_Button = self.win.train_end_Button
        self.train_preview_Button = self.win.train_preview_Button
        # Progress bar
        self.train_result_listWidget = self.win.train_result_listWidget

        # Signal
        self.train_images_Button.clicked.connect(lambda: self.select_dir(self.train_images_lineEdit))
        self.train_SWC_Button.clicked.connect(lambda: self.select_dir(self.train_SWC_lineEdit))
        self.train_save_Button.clicked.connect(lambda: self.select_dir(self.train_save_lineEdit, look=1, save=1))
        self.continue_training_checkBox.stateChanged.connect(self.on_continue_training)
        self.continue_training_Button.clicked.connect(lambda:
                                                      self.on_select_checkpoints_path(self.continue_training_lineEdit))
        self.train_start_Button.clicked.connect(self.data_train_start)
        self.train_end_Button.clicked.connect(self.data_train_stop)
        self.train_preview_Button.clicked.connect(self.show_TrainResultPreview)

        # Thread
        # Train
        self.nnUNetDataTrainQThread = nnUNetDataTrainQThread(win=self)
        self.nnUNetDataTrainQThread.finish0.connect(self.data_train_finish)
        self.nnUNetDataTrainQThread.progress0.connect(self.data_train_progress)
        self.nnUNetDataTrainQThread.progress1.connect(self.data_train_progress1)
        self.nnUNetDataTrainQThread.error0.connect(self.data_train_error)
        self.nnUNetDataTrainQThread.preview0.connect(self.show_preview_button)
        # Conversion
        self.nnUNetSwcToMaskQThread = nnUNetSwcToMaskQThread(win=self)
        self.nnUNetSwcToMaskQThread.finish0.connect(self.swc_to_mask_finish)
        self.nnUNetSwcToMaskQThread.warning0.connect(self.swc_to_mask_warning)
        self.nnUNetSwcToMaskQThread.progress0.connect(self.swc_to_mask_progress)
        self.nnUNetSwcToMaskQThread.progress1.connect(self.swc_to_mask_progress1)
        self.nnUNetSwcToMaskQThread.error0.connect(self.swc_to_mask_error)

    def on_continue_training(self, state):
        if state:
            self.continue_training_widget.setVisible(True)
            self.no_continue_training_widget.setVisible(False)
        else:
            self.continue_training_widget.setVisible(False)
            self.no_continue_training_widget.setVisible(True)

    def train_preview(self, img_i, mask_i, net_seg_i):
        # self.TrainResultPreview.start_preview(img_i, mask_i, net_seg_i)
        self.add_items("", img=img_i, mask=mask_i, pre=net_seg_i)

    def show_preview_button(self, progress_png_path):
        if progress_png_path:
            time.sleep(3)
            self.train_preview_Button.setEnabled(True)
            self.train_preview_Button.setStyleSheet(button_style)
            self.progress_png_path = progress_png_path

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

    def show_TrainResultPreview(self):
        # 打开训练进度图片
        if self.progress_png_path is not None:
            if os.path.exists(self.progress_png_path):
                image = Image.open(self.progress_png_path)
                image.show()

    def data_train_start(self):
        try:
            if not self.training:
                images_dir = self.train_images_lineEdit.text()
                SWC_dir = self.train_SWC_lineEdit.text()
                save_dir = self.train_save_lineEdit.text()
                epochs = self.train_epochs_combo.currentText()
                images_type = self.images_type_combo.currentText()
                radio_checked = self.continue_training_checkBox.isChecked()
                continue_training = str(radio_checked).lower() == "true"
                print(continue_training)
                checkpoints_path = "checkpoints_path"
                if radio_checked:
                    checkpoints_path = self.continue_training_lineEdit.text()
                    if checkpoints_path:
                        if not os.path.exists(checkpoints_path):
                            self.mess("The checkpoint path 'checkpoint_final.pth' does not exist, and training cannot continue!", "Prompt", 1)
                            return
                        condition1 = "fold" != str(Path(checkpoints_path).parent.stem).split("_")[0]
                        condition2 = ("nnUNetTrainer" in checkpoints_path and "nnUNet_results" in checkpoints_path
                                      and "nnUNetFrame" in checkpoints_path)
                        if condition1 or not condition2:
                            self.mess('The path of "checkpoints_path" does not correspond to the path of the saved folder. Please refer to the "Hint Text"!', "Prompt", 1)
                            return
                    else:
                        self.mess("The checkpoint path 'checkpoint_final.pth' is empty, and training cannot continue!",
                                  "Prompt", 1)
                        return

                if save_dir:
                    if not os.path.isdir(save_dir):
                        self.mess("Result save path is not a folder path, cannot train!", "Prompt", 1)
                        return
                else:
                    self.mess("Result save path is empty, cannot train!", "Prompt", 1)
                    return

                if images_dir:
                    if SWC_dir:
                        if os.path.exists(images_dir):
                            if os.path.exists(SWC_dir):
                                res = self.mess("Start training?", "Prompt", 2)
                                if res == QMessageBox.Yes:
                                    self.data_train_dict = {
                                        'images_dir': images_dir,
                                        'SWC_dir': SWC_dir,
                                        'mask_dir': None,
                                        'save_dir': save_dir,
                                        'epochs': epochs,
                                        'images_type': images_type,
                                        'continue_training': continue_training,
                                        'checkpoints_path': checkpoints_path
                                    }
                                    self.forbid_control()
                                    self.train_stop = False
                                    self.training = True
                                    self.count = 0
                                    self.train_result_listWidget.clear()
                                    self.start_time = time.time()
                                    # Label conversion
                                    self.add_items("Starting label conversion!")
                                    self.nnUNetSwcToMaskQThread.start()
                            else:
                                self.mess("Input SWC path does not exist, cannot train!", "Prompt", 1)
                                return
                        else:
                            self.mess("Input images path does not exist, cannot train!", "Prompt", 1)
                            return
                    else:
                        self.mess(f"Input SWC path is empty, cannot train!", "Prompt", 1)
                        return
                else:
                    self.mess("Input images path is empty, cannot train!", "Prompt", 1)
                    return
        except Exception as e:
            self.error_logger(e)

    """Disable controls"""

    def forbid_control(self):
        # Disable
        self.train_images_lineEdit.setEnabled(False)
        self.train_images_Button.setEnabled(False)
        self.train_SWC_lineEdit.setEnabled(False)
        self.train_SWC_Button.setEnabled(False)
        self.train_save_lineEdit.setEnabled(False)
        self.train_save_Button.setText("Look")
        self.train_images_lineEdit.setStyleSheet(lineedit_alpha_style)
        self.train_images_Button.setStyleSheet(button_alpha_style)
        self.train_SWC_lineEdit.setStyleSheet(lineedit_alpha_style)
        self.train_SWC_Button.setStyleSheet(button_alpha_style)
        self.train_save_lineEdit.setStyleSheet(lineedit_alpha_style)
        self.images_type_combo.setEnabled(False)
        self.train_epochs_combo.setEnabled(False)
        self.continue_training_checkBox.setEnabled(False)
        self.continue_training_lineEdit.setEnabled(False)
        self.continue_training_Button.setEnabled(False)
        self.continue_training_lineEdit.setStyleSheet(lineedit_alpha_style)
        self.continue_training_Button.setStyleSheet(button_alpha_style)
        # Hide
        self.train_preview_Button.setEnabled(False)
        self.train_preview_Button.setStyleSheet(button_alpha_style)
        self.train_start_Button.setVisible(False)
        self.train_end_Button.setVisible(True)

    """Restore controls"""

    def recover_control(self):
        self.training = False
        # Restore
        self.train_images_lineEdit.setEnabled(True)
        self.train_images_Button.setEnabled(True)
        self.train_SWC_lineEdit.setEnabled(True)
        self.train_SWC_Button.setEnabled(True)
        self.train_save_lineEdit.setEnabled(True)
        self.train_save_Button.setText("Open")
        self.train_images_lineEdit.setStyleSheet(lineedit_style)
        self.train_images_Button.setStyleSheet(button_style)
        self.train_SWC_lineEdit.setStyleSheet(lineedit_style)
        self.train_SWC_Button.setStyleSheet(button_style)
        self.train_save_lineEdit.setStyleSheet(lineedit_style)
        self.images_type_combo.setEnabled(True)
        self.train_epochs_combo.setEnabled(True)
        self.continue_training_checkBox.setEnabled(True)
        self.continue_training_lineEdit.setEnabled(True)
        self.continue_training_Button.setEnabled(True)
        self.continue_training_lineEdit.setStyleSheet(lineedit_style)
        self.continue_training_Button.setStyleSheet(button_style)
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
            self.mess(f"{new_text}, cannot train model!", "Prompt", 1)
        else:
            self.mess(f"Model training finished! Time elapsed: {t:.4f} Minute", "Prompt", 1)

    def data_train_progress(self, text):
        self.add_items(text)

    def data_train_progress1(self, text, count):
        self.add_items(text, is_remove=True, count=count)

    def data_train_error(self, e):
        self.train_stop = True
        text = f"Training error: {e}"
        self.add_items(text)
        self.mess(f"Model training error: {e}", "Prompt", 1)
        self.nnUNetDataTrainQThread.to_stop()
        if self.nnUNetDataTrainQThread.isRunning():
            self.nnUNetDataTrainQThread.quit()
            if not self.nnUNetDataTrainQThread.wait(5000):
                self.nnUNetDataTrainQThread.terminate()
                self.nnUNetDataTrainQThread.wait(3000)
        self.recover_control()

    def data_train_stop(self):
        r = self.mess("Training in progress, stop training?", "Prompt", 2)
        if r == QMessageBox.Yes:
            self.train_stop = True
            if self.nnUNetDataTrainQThread.is_training:
                # Step 1: 通知线程做清理（清 GPU cache 等）
                self.nnUNetDataTrainQThread.to_stop()
                # Step 2: 尝试优雅退出
                self.nnUNetDataTrainQThread.quit()
                self.add_items("Stopping training, please wait...")
                # Step 3: 5 秒后如果仍未退出，强制终止（不阻塞 GUI）
                from PyQt5.QtCore import QTimer
                self._stop_timeout_timer = QTimer(self.win)
                self._stop_timeout_timer.setSingleShot(True)
                self._stop_timeout_timer.timeout.connect(self._force_stop_training_thread)
                self._stop_timeout_timer.start(5000)

    def _force_stop_training_thread(self):
        """超时后强制终止训练线程"""
        if self.nnUNetDataTrainQThread.isRunning():
            self.nnUNetDataTrainQThread.terminate()
            self.add_items("Training forcefully stopped")
        # terminate() 不会执行 finally 块，手动清理资源
        # 但清理可能因进程管道断开而失败，必须用 try/except 保护，
        # 否则 recover_control() 不被调用，GUI 永久禁用
        try:
            self.nnUNetDataTrainQThread._cleanup_resources()
        except Exception:
            pass
        self.recover_control()
        self.mess("Training forcefully stopped!", "Prompt", 1)

    def swc_to_mask_finish(self):
        t = (time.time() - self.start_time) / 60
        text = f"Label conversion finished! Time elapsed: {t:.4f} Minute"
        self.add_items(text)

        self.add_items(text)
        if self.train_stop:
            self.recover_control()
            text = f"Training finished!"
            self.add_items(text)
        else:
            self.start_time = time.time()
            self.nnUNetDataTrainQThread.start()

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
        self.mess(f"Label conversion error: {e}", "Prompt", 1)
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

    def select_dir(self, line_edit, look=0, save=0):
        try:
            if look and self.training:
                self.toSelectPath(line_edit)
            else:
                # Create QSettings object
                settings = QSettings("MyCompany", "MyApp")
                # Read previously saved path
                if save:
                    initial_path = settings.value("nnUNetResultsSavePath", "")
                else:
                    initial_path = settings.value("nnUNetTrainPath", "")
                # Get folder path
                path = QFileDialog.getExistingDirectory(None, "Select folder", initial_path)
                if path:
                    if not self.contains_chinese(path):
                        if save:
                            settings.setValue("nnUNetResultsSavePath", path)
                        else:
                            settings.setValue("nnUNetTrainPath", path)
                        line_edit.setText(path)
                    else:
                        self.mess(f"Path contains Chinese characters or spaces, please select again!", "Prompt", 1)
        except Exception as e:
            self.mess(f"Open failed! {e}", "Prompt", 1)
            self.error_logger(e)

    def on_select_checkpoints_path(self, line_edit):
        try:
            # Create QSettings object
            settings = QSettings("MyCompany", "MyApp")
            # Read previously saved path
            initial_path = settings.value("nnUNetTrainCheckPointsPath", "")
            path = os.path.join(initial_path, "checkpoint_final.pth")
            # Get file path
            filename, filetype = QFileDialog.getOpenFileName(None, "Select File", path, "Model File (*.pth)")

            # If a file was selected, save the selected file path
            if filename:
                if not self.contains_chinese(filename):
                    settings.setValue("nnUNetTrainCheckPointsPath", str(Path(filename).parent))
                    line_edit.setText(filename)
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
