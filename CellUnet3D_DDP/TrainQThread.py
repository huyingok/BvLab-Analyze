# -*- coding: utf-8 -*-
import random
import time
import sys
import tifffile
import traceback
import os
import cv2
from pathlib import Path
import json
# Must be executed before any import torch
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from .models.model import LoadModel
from torch.utils.data import DataLoader
from .DataLoader import GetMultiTypeMemoryDataSetAndCropQxz
import numpy as np
import torch
from tensorboardX import SummaryWriter
from .MyUtil import GetLossOptimiLr, TakeNotesLoss
from os.path import join
import cc3d
from speed_cc3d import speed_cc3d
from scipy.spatial import distance
import warnings

warnings.filterwarnings("ignore")
import socket
import subprocess

from config import exe_cfg, cfgPath
import gpu_device_use

from PyQt5.QtCore import QThread, pyqtSignal


class CellDataTrainQThread(QThread):
    finish0 = pyqtSignal(str)
    progress0 = pyqtSignal(str)
    progress1 = pyqtSignal(str, int)
    error0 = pyqtSignal(str)
    update_modSavePath = pyqtSignal(str)
    # update_plot = pyqtSignal(str, str, str)
    preview0 = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    show_preview = pyqtSignal(bool)

    def __init__(self, *args, **kwargs):
        super(CellDataTrainQThread, self).__init__()
        self.win = kwargs.get('win')
        self.logger = self.win.logger
        self.init_value()

    def init_value(self):
        self.lr = 0.0002
        self.weight_decay = 0.00001
        self.train_stop = False
        self.start_train = False
        self.s_gpu_id = 0
        self.wordSize = 4
        self.curRankId = 0
        self.device = None
        self.writer = None
        self.savePath = ""
        self.trainLoader = None
        self.valLoader = None
        self.rootPath = ""
        self.imgSize = None
        self.batchSize = 1
        self.valBatchSize = 2
        self.epochs = 500
        self.valCount = 150
        self.new_text = ""

        self.text_eval_path = ""
        self.text_save_model_path = ""
        self.text_loss_path = ""

        self.modSavePath = ""

        self.make_config_path = ""  # config file path
        self.logAdd = ""  # original log path
        self.checkpoint_path = ""  # checkpoint save path
        self.make_config = {}  # config file content
        self.is_keep_on = False  # whether to continue training
        self.checkpoints = None  # checkpoint

    def run(self):
        """Run the training thread, handling the complete model training process"""
        print("Cell training!")
        self.init_value()
        try:
            gpu_id = gpu_device_use.get_gpu_utilization()
            if gpu_id is not None:
                torch.cuda.set_device(gpu_id)
                self.device = torch.device("cuda", gpu_id)
            else:
                self.new_text = "No available GPU"
                return
            # Initialize training state
            self._initialize_training_state()

            # Execute training
            if not self.train_stop:
                self._execute_training_pipeline()

        except Exception as e:
            self._handle_training_exception(e)
        finally:
            self._cleanup_resources()
            self.start_train = False
            time.sleep(2)
            self.finish0.emit(self.new_text)

    def _initialize_training_state(self):
        """Initialize training state variables"""
        self.train_stop = False
        self.start_train = True

    def _execute_training_pipeline(self):
        """Execute the complete training pipeline: data loading, model configuration, training execution"""
        # Get training parameters
        imgPath = self.win.data_train_dict['imgDir']
        shapes = self.win.data_train_dict['shapes']  # xyz
        self.epochs = self.win.data_train_dict['epochs']
        self.make_config_path = self.win.data_train_dict['makerInfo_path']  # config file
        self.is_keep_on = self.win.data_train_dict.get('is_keep_on', self.is_keep_on)

        self.rootPath = str(Path(imgPath).parent.absolute())
        self.imgSize = np.array(shapes, dtype=np.int32)  # image size xyz

        # Initialize training environment
        self.train_init()

        # Load data
        train_loader, val_loader = self._load_training_data(imgPath)
        self.trainLoader = train_loader
        self.valLoader = val_loader

        # Configure model
        model_config = self._get_model_config()

        # Execute training
        self.start_trainer(model_config)

        # Close writer after training
        self.writer.close()

    def _load_training_data(self, imgPath):
        """Load training and validation datasets"""
        trainTxt = r"train.txt"  # training txt name
        valTxt = r"val.txt"  # validation txt name
        maskPath = join(self.rootPath, "mask")

        # Load training dataset
        train_dataset = GetMultiTypeMemoryDataSetAndCropQxz(
            self.rootPath, trainTxt, self.imgSize, imgPath, maskPath
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batchSize, sampler=None, pin_memory=True
        )

        # Load validation dataset
        val_dataset = GetMultiTypeMemoryDataSetAndCropQxz(
            self.rootPath, valTxt, self.imgSize, imgPath, maskPath
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.valBatchSize, sampler=None, pin_memory=True
        )

        return train_loader, val_loader

    def _get_model_config(self):
        """Get model configuration parameters"""
        return {
            'name': 'UNet3D',
            'in_channels': 4,
            'out_channels': 1,
            'fieldSpace': 1,
            'layer_order': 'gcr',
            'f_maps': [16, 32, 64, 128, 256],
            'num_groups': 8,
            'final_sigmoid': True,
            'is_segmentation': True
        }

    def _handle_training_exception(self, exception):
        """Handle exceptions during training"""
        self.error0.emit(str(exception))
        print("Training ended abnormally")

        # Log detailed error information
        self.logger.error("\n=== Error message ===")
        self.logger.error(f"Exception type: {type(exception).__name__}")
        self.logger.error(f"Error message: {exception}")
        self.logger.error("=== Error location ===")

        # Get and log stack trace
        tb = sys.exc_info()[2]
        for frame in traceback.extract_tb(tb):
            self.logger.error(f"  File: {frame.filename}")
            self.logger.error(f"  Line: {frame.lineno}")
            self.logger.error(f"  Function: {frame.name}")
            self.logger.error(f"  Code: {frame.line}\n")

    def _cleanup_resources(self):
        """Clean up resources used during training"""
        # Clean up distributed training environment
        if hasattr(self, 'model'):
            # Clean up model related resources
            attributes_to_clean = ['model', 'trainLoader', 'valLoader', 'device',
                                   'writer', 'loss_criterion', 'optimizer',
                                   'lr_scheduler', 'eval_metric']

            for attr in attributes_to_clean:
                if hasattr(self, attr):
                    try:
                        delattr(self, attr)
                    except Exception:
                        pass

        # Clean GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cache cleared")

    def load_train_checkpoint(self):
        # Try to load checkpoint
        self.model.load_state_dict(self.checkpoints['model_state'])
        self.optimizer.load_state_dict(self.checkpoints['optimizer_state'])
        self.lr_scheduler.load_state_dict(self.checkpoints['scheduler_state'])  # restore learning rate scheduler state
        self.start_epoch = self.checkpoints['epoch'] + 1
        self.small_epoch = self.checkpoints['small_epoch'] + 1
        self.big_epoch = self.checkpoints['big_epoch'] + 1
        self.eval_epoch = self.checkpoints['eval_epoch'] + 1
        iter_count = self.checkpoints['iter_count']
        lastEvalVal = self.checkpoints.get('lastEvalVal', 0)
        patience_counter = self.checkpoints.get('patience_counter', 0)
        print(f"Resuming training from checkpoint, starting epoch: {self.start_epoch}")
        return iter_count, lastEvalVal, patience_counter

    def start_trainer(self, modelCfg):
        try:
            # Initialize training tracking variables
            train_small_losses = TakeNotesLoss()
            train_big_losses = TakeNotesLoss()
            evalVal = TakeNotesLoss()
            lastEvalVal = 0
            view_count = 0
            iter_count = 0  # global iteration counter, used to track current iteration number
            progress1_count = 0
            patience_counter = 0  # early stopping counter for validation accuracy stagnation

            # Initialize model
            self.model = LoadModel(modelCfg)
            device = self.device
            self.model.to(device)

            # Get loss function, optimizer, learning rate scheduler and evaluation metric
            self.loss_criterion, self.optimizer, self.lr_scheduler, self.eval_metric = GetLossOptimiLr(self.model,
                                                                                                       learning_rate=self.lr,
                                                                                                       weight_decay=self.weight_decay)
            self.eval_metric.to(device)

            self.start_epoch = 0
            self.small_epoch = 0
            self.big_epoch = 0
            self.eval_epoch = 0
            if self.is_keep_on and self.checkpoints is not None:  # continue training
                # Load checkpoint
                iter_count, lastEvalVal, patience_counter = self.load_train_checkpoint()

            self.text_eval_path = join(self.savePath, 'text_eval.txt')
            self.text_save_model_path = join(self.savePath, 'text_save_model.txt')
            self.text_loss_path = join(self.savePath, 'text_loss.txt')
            if not self.eval_epoch:
                with open(self.text_eval_path, 'w', encoding='utf-8') as f:
                    f.write("")
                with open(self.text_save_model_path, 'w', encoding='utf-8') as f:
                    f.write("")
                with open(self.text_loss_path, 'w', encoding='utf-8') as f:
                    f.write("")

            self.model.train(True)

            start_time = time.time()
            self.valCount = min(self.valCount, len(self.trainLoader))
            train_loader_length = len(self.trainLoader)

            self.train_round = self.eval_epoch

            # Main training loop
            for epoch in range(self.epochs):
                epoch += self.start_epoch
                if self.train_stop:
                    return

                # Initialize progress info for the first epoch
                if epoch == 0:
                    text = f"TRAIN [Epoch {epoch} | {self.epochs}] [Proce {0} | {train_loader_length}] [Loss {1:.4f}]"
                    self.progress0.emit(text)

                # Clear cache and set training mode
                torch.cuda.empty_cache()
                torch.set_grad_enabled(True)
                self.model.train()

                # Batch training loop
                for kk, (img, mask, name) in enumerate(self.trainLoader):
                    if self.train_stop:
                        return
                    if img.shape[0] != self.batchSize:
                        continue

                    # Train one batch
                    reduced_loss = self._train_one_batch(img, mask, device)
                    if self.curRankId == 0:
                        train_small_losses.update(reduced_loss.item())
                        train_big_losses.update(reduced_loss.item())

                    # Handling at every 10th iteration and validation points
                    is_10th_iter = (iter_count + 1) % 10 == 0
                    is_val_point = (iter_count + 1) % self.valCount == 0
                    is_val10th_iter = self.valCount % 10 == 0

                    # Record training progress and loss
                    if is_10th_iter and self.curRankId == 0:
                        self._record_training_loss(epoch, kk, train_small_losses, iter_count)

                        # Record detailed progress
                        if is_val_point:
                            if is_val10th_iter:
                                userTime = time.time() - start_time
                                surplusTime = userTime / (progress1_count + 1) * (self.valCount - progress1_count - 1)
                                logInfo2 = '[Round %d progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                                    (self.train_round + 1), (progress1_count + 1) / self.valCount * 100, userTime, surplusTime)
                                self.progress1.emit(logInfo2, 0)
                            else:
                                self._update_progress_info(progress1_count, start_time)
                            start_time = time.time()
                            progress1_count = 0
                        else:
                            if kk != 0:
                                userTime = time.time() - start_time
                                surplusTime = userTime / (progress1_count + 1) * (self.valCount - progress1_count - 1)
                                logInfo2 = '[Round %d progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                                    (self.train_round + 1), (progress1_count + 1) / self.valCount * 100, userTime, surplusTime)
                                self.progress1.emit(logInfo2, 0)
                            progress1_count += 1
                    else:
                        # Update progress info
                        if is_val_point and self.curRankId == 0:
                            self._update_progress_info(progress1_count, start_time)
                            start_time = time.time()
                            progress1_count = 0
                        else:
                            userTime = time.time() - start_time
                            surplusTime = userTime / (progress1_count + 1) * (self.valCount - progress1_count - 1)
                            logInfo2 = '[Round %d progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                                (self.train_round + 1), (progress1_count + 1) / self.valCount * 100, userTime, surplusTime)
                            self.progress1.emit(logInfo2, progress1_count)
                            progress1_count += 1

                    # Perform validation every valCount iterations
                    if is_val_point:
                        # _perform_validation returns the new evaluation value and patience counter
                        new_eval_val, view_count, patience_counter = self._perform_validation(epoch, train_big_losses, evalVal,
                                                                                              lastEvalVal,
                                                                                              iter_count, view_count, patience_counter)
                        if self.train_stop:
                            return
                        if new_eval_val is not None:
                            lastEvalVal = new_eval_val

                        self.train_round = evalVal.id + self.eval_epoch + 1

                    iter_count += 1
                    torch.cuda.empty_cache()

                self.model.eval()
                with torch.no_grad():
                    # Save current state
                    torch.save({
                        # 检查点参数
                        'epoch': epoch,
                        'small_epoch': train_small_losses.id + self.small_epoch,
                        'big_epoch': train_big_losses.id + self.big_epoch,
                        'eval_epoch': evalVal.id + self.eval_epoch,
                        'model_state': self.model.state_dict(),
                        'iter_count': iter_count,
                        'optimizer_state': self.optimizer.state_dict(),
                        'scheduler_state': self.lr_scheduler.state_dict(),  # save learning rate scheduler state
                        # 训练模型参数
                        'state_dict': self.model.state_dict(),
                        'lastEvalVal': lastEvalVal,
                        'patience_counter': patience_counter,
                        'param': self.optimizer,
                    }, self.checkpoint_path)

                    if (self.checkpoint_path != self.make_config.get('checkpoint_path', None) or
                            self.logAdd != self.make_config.get('log_path', None)):
                        print("Updating configuration file!")
                        self.make_config['checkpoint_path'] = self.checkpoint_path
                        self.logAdd = self.make_config['log_path']
                        with open(self.make_config_path, 'w') as f:  # update config file
                            f.write(json.dumps(self.make_config, indent=4))
                torch.cuda.empty_cache()
                self.model.train(True)

        except Exception as e:
            torch.cuda.empty_cache()
            print("End")
            print("\n=== Error message ===")
            print(f"Exception type: {type(e).__name__}")
            print(f"Error message: {e}")
            print("=== Error location ===")
            tb = sys.exc_info()[2]
            for frame in traceback.extract_tb(tb):
                print(f"  File: {frame.filename}")
                print(f"  Line: {frame.lineno}")
                print(f"  Function: {frame.name}")
                print(f"  Code: {frame.line}\n")
        finally:
            torch.cuda.empty_cache()
            return

    def _train_one_batch(self, img, mask, device):
        """Train one batch of data"""
        img = img.to(device)
        mask = mask.to(device)
        seg = self.model(img)
        loss = self.loss_criterion(seg, mask)[0]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def _record_training_loss(self, epoch, kk, train_small_losses, iter_count):
        """Record training loss information"""
        tmpLoss = train_small_losses.update2()
        self.writer.add_scalar('Loss/TrainSmallLoss', tmpLoss, train_small_losses.id + self.small_epoch)

        text = f"TRAIN [Epoch {epoch} | {self.epochs}] [Proce {kk} | {len(self.trainLoader)}] [Loss {tmpLoss:.4f}]"
        self.progress1.emit(text, 1)

        # Write to loss log file
        with open(self.text_loss_path, "a", encoding="utf-8") as f:
            f.write(f"{iter_count + 1} {tmpLoss:.4f}\n")
        f.close()

    def _update_progress_info(self, progress1_count, start_time):
        """Update and display training progress information"""
        userTime = time.time() - start_time
        surplusTime = userTime / (progress1_count + 1) * (self.valCount - progress1_count - 1)
        logInfo2 = '[Round %d progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
            (self.train_round + 1), (progress1_count + 1) / self.valCount * 100, userTime, surplusTime)
        self.progress1.emit(logInfo2, progress1_count)

    def _perform_validation(self, epoch, train_big_losses, evalVal, lastEvalVal, iter_count, view_count, patience_counter):
        """Execute validation process and return the updated evaluation value and patience counter"""
        torch.cuda.empty_cache()
        if self.curRankId == 0:
            self.writer.add_scalar('Loss/TrainBigLoss', train_big_losses.update2(), train_big_losses.id + self.big_epoch)
        self.model.eval()

        try:
            with torch.no_grad():
                random_v = random.randint(0, len(self.valLoader) - 1)
                eval_start_time = time.time()
                for kk, (img, mask, name) in enumerate(self.valLoader):
                    if self.train_stop:
                        return None, view_count, patience_counter
                    img = img.to(self.device)
                    mask = mask.to(self.device)
                    seg = self.model(img)
                    eval = self.eval_metric(seg, mask)
                    evalVal.update(eval)

                    eval_userTime = time.time() - eval_start_time
                    eval_surplusTime = eval_userTime / (kk + 1) * (len(self.valLoader) - kk - 1)
                    eval_logInfo = '[Round %d progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                        (self.train_round + 1), (kk + 1) / len(self.valLoader) * 100, eval_userTime, eval_surplusTime)

                    if kk == random_v and (epoch + 1) % 10 == 0:
                        ns = seg.cpu().detach().numpy()  # training result -- Tag
                        ma = mask.cpu().detach().numpy()  # target label
                        imgs = img.cpu().detach().numpy()  # original image
                        ns_i = ns[:, 0]
                        ma_i = ma[:, 0]
                        img_i = imgs[:, 0]
                        self.save_result(ns_i, ma_i, img_i)
                        self.progress1.emit(eval_logInfo, 0)
                    else:
                        self.progress1.emit(eval_logInfo, kk)

                    torch.cuda.empty_cache()

                curEvalVal = evalVal.update2()
                curEvalVal = curEvalVal
                if self.curRankId == 0:
                    self.writer.add_scalar('Eval/EvalVal', curEvalVal, evalVal.id + self.eval_epoch)

                if view_count < 2 and curEvalVal > 0:
                    view_count += 1
                if view_count == 2 or epoch >= 5:
                    self.show_preview.emit(True)

                # Early stopping: check validation improvement
                early_stop_patience = 100
                if curEvalVal > lastEvalVal:
                    # Improved: reset patience counter
                    patience_counter = 0
                else:
                    # Not improved: increment patience counter
                    patience_counter += 1
                    if self.curRankId == 0:
                        self.progress0.emit(f'Validation accuracy not improved for {patience_counter} consecutive rounds')
                    
                    # Early stopping trigger
                    if patience_counter >= early_stop_patience:
                        if self.curRankId == 0:
                            self.progress0.emit(f'Early stopping! Validation accuracy not improved for {early_stop_patience} consecutive rounds')
                        self.train_stop = True
                        return None, view_count, patience_counter

                # Save best model
                if curEvalVal > lastEvalVal:
                    if self.curRankId == 0:
                        self.progress0.emit('Validation loss decreased, saving model')
                        self.modSavePath = join(self.savePath, "supernet_%s_best_%.4f.pth" % (
                            str(epoch).zfill(5), curEvalVal))
                        # Update
                        torch.save({'state_dict': self.model.state_dict(), 'param': self.optimizer},
                                   self.modSavePath)
                        # Copy
                        self.update_modSavePath.emit(self.modSavePath)
                        # If more than one model saved, delete the oldest one
                        # if len([l for l in os.listdir(self.savePath) if ".pth" in l]) > 1:
                        #     self.delete_oldest_file(self.savePath)
                    lastEvalVal = curEvalVal

                    with open(self.text_save_model_path, "a", encoding="utf-8") as f:
                        f.write(f"{iter_count + 1} Validation loss decreased, saving model {curEvalVal:.4f}\n")
                    f.close()

                # Update learning rate
                self.lr_scheduler.step(curEvalVal)
                lr = self.optimizer.param_groups[0]['lr']
                if self.curRankId == 0:
                    self.writer.add_scalar('TrainParam/Lr', lr, evalVal.id + self.eval_epoch)

                    text = f"VAL [Epoch {epoch} | {self.epochs}] [EvalVal; {curEvalVal:.4f}] [Lr: {lr:.8f}] [ValLen {len(self.valLoader)}]"
                    self.progress0.emit(text)

                    with open(self.text_eval_path, "a", encoding="utf-8") as f:
                        f.write(f"{iter_count + 1} {curEvalVal:.4f}\n")
                    f.close()

            return lastEvalVal, view_count, patience_counter

        finally:
            torch.cuda.empty_cache()
            if self.train_stop:
                return None, view_count, patience_counter
            self.model.train()

    def save_result(self, net_seg, mask, img):
        root = self.rootPath
        i = random.randint(0, len(mask) - 1)

        path = f"{root}/TrainResultsPreview/{i + 1}/"
        os.makedirs(path, exist_ok=True)
        net_seg_i = ((net_seg[i] - net_seg[i].min()) / (net_seg[i].max() - net_seg[i].min()) * 255).astype(np.uint8)
        mask_i = ((mask[i] - mask[i].min()) / (mask[i].max() - mask[i].min()) * 255).astype(np.uint8)
        img_i = ((img[i] - img[i].min()) / (img[i].max() - img[i].min()) * 255).astype(np.uint8)

        net_seg_i[net_seg_i < 103] = 0

        seg_name = f"net_seg{i + 1}.tif"
        tifffile.imwrite(join(path, seg_name), net_seg_i, compression="lzw")

        tifffile.imwrite(join(path, f"mask{i + 1}.tif"), mask_i, compression="lzw")
        tifffile.imwrite(join(path, f"img{i + 1}.tif"), img_i, compression="lzw")

        img_i_2d = self.MaxProject(img_i, 1)
        mask_i_2d = self.MaxProject(mask_i, 0)
        pre_i_2d = self.MaxProject(net_seg_i, 0)

        self.preview0.emit(img_i_2d, mask_i_2d, pre_i_2d)

    def MaxProject(self, img, is_enhance):
        img = np.max(img, axis=0)
        img = self.normalize_and_scale_to_uint8(img)
        if is_enhance:
            self.enhance_contrast_histogram_equalization(img)
        return img

    def enhance_contrast_histogram_equalization(self, img):
        """
        Enhance image contrast using histogram equalization.

        Parameter:
            img (numpy.ndarray): Input image.
        Returns:
            numpy.ndarray: Contrast enhanced image.
        """
        # Apply histogram equalization
        img_eq = cv2.equalizeHist(img)
        return img_eq

    def normalize_and_scale_to_uint8(self, data):
        """
        Normalize gray values to 0-1, then scale to 0-255 and convert to uint8.

        Parameter:
            data (numpy.ndarray): Input data.
        Returns:
            numpy.ndarray: Converted uint8 data.
        """
        data_min = np.min(data)
        data_max = np.max(data)
        normalized_data = (data - data_min) / (data_max - data_min)
        scaled_data = (normalized_data * 255).astype(np.uint8)
        return scaled_data

    def delete_oldest_file(self, directory):
        # Get all files in the directory
        files = [l for l in os.listdir(directory) if ".pth" in l]
        if not files:
            print("Directory is empty, no files to delete.")
            return

        # Create a list storing file paths and creation times
        files_with_ctime = []
        for file in files:
            file_path = join(directory, file)
            if os.path.isfile(file_path):  # ensure it's a file not a directory
                creation_time = os.path.getctime(file_path)
                files_with_ctime.append((file_path, creation_time))

        # Sort by creation time, oldest first
        files_with_ctime.sort(key=lambda x: x[1])

        # Delete the oldest file
        oldest_file_path = files_with_ctime[0][0]
        os.remove(oldest_file_path)
        # print(f"Deleted the oldest file: {oldest_file_path}")

    def train_init(self):
        with open(self.make_config_path, 'r') as f:  # read config file
            self.make_config = json.loads(f.read())

        self.logAdd = self.make_config.get("log_path", "")  # get original log file path

        # if self.is_keep_on:  # self.checkpoint_path exists
        #     self.checkpoint_path = self.make_config.get("checkpoint_path", "")  # get original checkpoint file path
        # if not self.checkpoint_path:

        results_dir = self.win.data_train_dict['results_dir']  # save path
        checkpoint_dir = join(results_dir, "CheckPoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_path = join(checkpoint_dir, "checkpoint.pth")  # checkpoint file path

        if self.is_keep_on and self.checkpoint_path:
            try:
                # Load checkpoint
                self.checkpoints = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            except Exception as e:
                # Load failed
                print("Checkpoints load failed!")
                self.checkpoints = None
        is_keep_on = False
        if self.is_keep_on and self.checkpoints is not None:
            logAdd = self.make_config.get("log_path", None)
            if logAdd is not None:
                if os.path.isdir(logAdd):  # log path exists
                    small_epoch = self.checkpoints['small_epoch']
                    # Use tensorboardX SummaryWriter to write logs
                    # self.writer = SummaryWriter(logAdd)
                    # purge_step parameter ensures continuation from specified step
                    # Avoid TensorBoard displaying duplicate step ranges
                    # Keep log directory unchanged to ensure data continuity
                    self.writer = SummaryWriter(log_dir=logAdd, purge_step=small_epoch)
                    is_keep_on = True
                    expName = Path(logAdd).stem

        logPath = exe_cfg.cfg['ExePath']['cell_logs_path']  # log path
        self.curRankId = 0
        # In single-GPU environment, directly use single-GPU training logic
        if self.curRankId == 0:
            if not is_keep_on:
                if not os.path.isdir(logPath):
                    os.makedirs(logPath)
                logName = len(os.listdir(logPath))
                expName = 'exp%s' % str(logName).zfill(3)
                logAdd = join(logPath, expName)
                while True:
                    if os.path.isdir(logAdd):
                        logName += 1
                        logAdd = join(logPath, f"exp{str(logName).zfill(3)}")
                    else:
                        break
                logAdd = os.path.abspath(logAdd)
                # Ensure log directory exists
                os.makedirs(logAdd, exist_ok=True)
                # Use tensorboardX SummaryWriter to write logs
                self.writer = SummaryWriter(logAdd)
                # Update log file
                self.make_config['log_path'] = str(logAdd)
            # Start TensorBoard
            print(logAdd)

            # if not os.path.exists(logAdd):
            #     os.makedirs(logAdd)
            #     # Set port and host
            # port = exe_cfg.cfg['Port']['cell_port']  # default port is 6008, can be changed
            # host = exe_cfg.cfg['Host']['cell_host']  # default host is 127.0.0.1, can be changed
            #
            # subprocess.Popen([
            #     "tensorboard", f"--logdir={logAdd}", f"--host={host}", f"--port={port}"], shell=True)

            if not os.path.exists(logAdd):
                os.makedirs(logAdd)
                # Set port and host
            port = exe_cfg.cfg['Port']['cell_port']  # default port is 6008, can be changed
            host = exe_cfg.cfg['Host']['cell_host']  # default host is 127.0.0.1, can be changed

            """Start TensorBoard"""
            while self.is_port_in_use(host, port):
                print(f"TensorBoard is already running on port {port}.")
                port = random.randint(6006, 6100)
                exe_cfg.cfg['Port']['cell_port'] = str(port)

            with open(cfgPath, 'w') as configfile:
                exe_cfg.cfg.write(configfile)

            # Function to start TensorBoard
            def start_tensorboard():
                # Method1: Try using TensorFlow's TensorBoardModule (should be packaged with the program)
                try:
                    import tensorboard.program
                    import mimetypes
                    # Ensure MIME types for JavaScript files are set correctly
                    mimetypes.add_type('application/javascript', '.js')
                    mimetypes.add_type('text/css', '.css')
                    mimetypes.add_type('application/json', '.json')

                    tb = tensorboard.program.TensorBoard()
                    tb.configure(
                        argv=[
                            None,
                            f"--logdir", f"{logAdd}",
                            f"--host", f"{host}",
                            f"--port", f"{port}",
                            # '--bind_all'  # allow access from any IP
                        ]
                    )
                    url = tb.launch()
                    print(f"TensorBoard service started (using TensorFlowModule): {url}")
                    return True
                except Exception as e:
                    print(f"Failed to start TensorBoard using TensorFlow module: {e}")

                # Method2: Try using the system TensorBoard command
                try:
                    # Start TensorBoard using command line arguments
                    command = [
                        "tensorboard",
                        f"--logdir={logAdd}",
                        f"--host={host}",
                        f"--port={port}"
                    ]

                    # Start TensorBoard process
                    subprocess.Popen(command)
                    print(f"TensorBoard service started (using system command), access at: http://{host}:{port}")
                    return True
                except Exception as e:
                    print(f"Failed to start TensorBoard using system command: {e}")

                # Method3: If both methods fail, provide manual start instructions
                print("TensorBoard startup failed, it is recommended to start manually:")
                print("1. Open command prompt")
                print(f"2. Run command: tensorboard --logdir={logAdd} --host={host} --port={port}")
                print(f"3. Access in browser: http://{host}:{port}")
                return False

            # Start TensorBoard
            start_tensorboard()

            # Even if TensorBoard fails to start, training can continue

            # subprocess.Popen([
            #     "tensorboard", f"--logdir {logAdd}", f"--host {host}", f"--port {port}"])

            # # Get current program environment variables
            # current_env = os.environ.copy()
            #
            # # Define command
            # command = [
            #     "tensorboard",
            #     f"--logdir={logAdd}",
            #     f"--host={host}",
            #     f"--port={port}"
            # ]
            #
            # # Start subprocess
            # process = subprocess.Popen(command, env=current_env)

            # Use TensorFlow's TensorBoard module to start TensorBoard
            # tb = tensorboard.program.TensorBoard()
            # tb.configure(
            #     argv=[
            #         None,
            #         f"--logdir", f"{logAdd}",
            #         f"--host", f"{host}",
            #         f"--port", f"{port}",
            #         # '--bind_all'
            #     ]
            # )
            # try:
            #     url = tb.launch()
            #     print(f"TensorBoard is running at {url}")
            #
            #     # Wait a few seconds to ensure TensorBoard service has started
            #     # time.sleep(2)
            #
            #     # # Open browser
            #     # webbrowser.open(url)
            # except Exception as e:
            #     print(f"Failed to start TensorBoard: {e}")

            self.savePath = join(exe_cfg.cfg['ExePath']['cell_models_path'], f"{expName}")  # model save path
            if not os.path.isdir(self.savePath):
                os.makedirs(self.savePath)

    def to_stop(self):
        self.train_stop = True
        time.sleep(3)
        try:
            # Delete model reference
            # if hasattr(self, 'model'):
            #     del self.model
            #     print("Model deleted")
            # del self.trainLoader
            # del self.valLoader
            # del self.device
            # del self.writer
            # del self.loss_criterion
            # del self.optimizer
            # del self.lr_scheduler
            # del self.eval_metric
            # Clean GPU cache
            torch.cuda.empty_cache()
            print("GPU cache cleared")
        except Exception as e:
            print(e)
            print("Stop")
        finally:
            time.sleep(3)
            self.progress1.emit("Training stopped!", 0)

    def is_port_in_use(self, host, port):
        """Check if the specified port is in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((str(host), int(float(port)))) == 0


if __name__ == '__main__':
    # import tensorboard.program
    # import subprocess
    # import time
    logAdd = r"D:\CellNeuralBloodVessel\IntegratePose\logs\cell_logs\exp001"
    host = r"localhost"
    port = 6008

    # tb = tensorboard.program.TensorBoard()
    # tb.configure(
    #     argv=[
    #         None,
    #         fr"--logdir={logAdd}",
    #         f"--host={host}",
    #         f"--port={port}",
    #     ]
    # )
    # url = tb.launch()
    # print(f"TensorBoard is running at {url}")

    # subprocess.Popen([
    #     "tensorboard", f"--logdir={logAdd}", f"--host={host}", f"--port={port}"], shell=True)
    #
    # time.sleep(100)