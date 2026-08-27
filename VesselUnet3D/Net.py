import os
from os.path import join
import torch
import shutil
import time
import sys
import traceback
import numpy as np
import random
import cv2
import subprocess
from multiprocessing import Queue, Process
from PyQt5.QtCore import QThread
import tifffile as tiff
from config import exe_cfg


class TakeNotesLoss:
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.id = -1

    def update(self, value):
        self.sum += value
        self.count += 1

    def update2(self):
        # print(self.sum, self.count)
        if self.count != 0:
            tmp = self.sum / self.count
        else:
            tmp = 0
        self.sum = 0
        self.count = 0
        self.id += 1
        return tmp


class Trainer:
    # Loaded training set, loaded validation set, loaded model, loss function, optimizer, learning rate scheduler,
    # evaluation metric, number of backpropagations per update, model save path, device, number of samples per training batch,
    def __init__(self, data_loader, test_loader, model, loss_criterion, optimizer, lr_scheduler, eval_metric,
                 backwardNumber=1, modelPath=None, rootPath=None, device=torch.device('cpu'), batchSize=1,
                 valBatchSize=None, progress0=None, progress1=None, update_modSavePath=None,
                 preview0=None, show_preview=None):
        self.backwardNumber = backwardNumber  # Number of backpropagations per update
        self.batchSize = batchSize  # Number of samples processed per training batch
        if valBatchSize is None:
            self.valBatchSize = batchSize
        else:
            self.valBatchSize = valBatchSize
        self.dataLoader = data_loader  # Loaded training set
        self.testLoader = test_loader  # Loaded validation set
        self.device = device  # Device to use
        # self.valCount = int(30 / self.batchSize)
        # self.valCount = 8
        self.valCount = 150  # Validate every N iterations
        # Network
        self.super_net = model  # Loaded model
        self.super_net.to(self.device)  # Move model to GPU or CPU
        '''Load loss function, optimizer, learning rate, etc.'''
        self.loss_criterion = loss_criterion  # Loss function
        self.optimizer = optimizer  # Optimizer
        self.lr_scheduler = lr_scheduler  # Learning rate scheduler
        self.eval_metric = eval_metric  # Evaluation metric
        self.modelPath = modelPath  # Model save path
        self.rootPath = rootPath
        if self.modelPath is None:
            self.modelPath = './saved_models/'
        if not os.path.isdir(self.modelPath):
            os.makedirs(self.modelPath)

        # self.text_eval_path = os.path.join(self.modelPath, 'text_eval.txt')
        # self.text_save_model_path = os.path.join(self.modelPath, 'text_save_model.txt')
        # self.text_loss_path = os.path.join(self.modelPath, 'text_loss.txt')
        # with open(self.text_eval_path, 'w', encoding='utf-8') as f:
        #     pass
        # with open(self.text_save_model_path, 'w', encoding='utf-8') as f:
        #     pass
        # with open(self.text_loss_path, 'w', encoding='utf-8') as f:
        #     pass

        self.stop_training = False  # Flag variable to control whether to stop training
        self.modSavePath = ""

        self.progress0 = progress0
        self.progress1 = progress1
        self.update_modSavePath = update_modSavePath
        self.preview0 = preview0
        self.show_preview = show_preview
        self.base_path = exe_cfg.cfg['ExePath']['base_path']

    def Train(self, turns=2, writer=None):
        """Main function for training the model

        Args:
            turns (int): Number of training epochs
            writer: Log writer for TensorBoard
        """
        try:
            self.view_count = 0
            train_small_losses = TakeNotesLoss()  # Record average loss over short term (every 10 iterations)
            train_big_losses = TakeNotesLoss()  # Record average loss over long term (every valCount iterations)
            evalVal = TakeNotesLoss()  # Record evaluation metric on validation set
            lastEvalVal = 0  # Record the best evaluation metric value from previous validation to decide whether to save model
            self.valCount = min(len(self.dataLoader), self.valCount)  # Validate every N iterations
            iter_count = 0  # Global iteration counter
            progress1_count = 0
            start_time = time.time()

            # Training epoch loop
            for turn in range(turns):
                if self.stop_training:  # Check if training should stop
                    break

                # Send initial progress info for the first epoch
                if turn == 0:
                    text = f"TRAIN [Epoch {turn} | {turns}] [Proce {0} | {len(self.dataLoader)}] [Loss {1:.4f}]"
                    self.progress0.emit(text)

                # Clear GPU cache and set to training mode
                torch.cuda.empty_cache()
                torch.set_grad_enabled(True)
                self.super_net.train()  # Set network to training mode (enable dropout and batch norm)

                # Batch loop
                for batch_idx, (img, mask, name) in enumerate(self.dataLoader):  # Load data
                    if self.stop_training:  # Check if training should stop
                        break

                    # Skip data that does not match batch size
                    if img.shape[0] != self.batchSize:
                        continue

                    # Forward and backward propagation
                    loss = self._train_one_batch(img, mask)
                    train_small_losses.update(loss.item())
                    train_big_losses.update(loss.item())

                    # Record training progress and loss value
                    if (iter_count + 1) % 10 == 0:
                        if (iter_count + 1) % self.valCount == 0:
                            self._record_training_progress(turn, turns, batch_idx, len(self.dataLoader),
                                                           train_small_losses, writer)
                        else:
                            self._record_training_progress(turn, turns, batch_idx, len(self.dataLoader),
                                                           train_small_losses, writer)
                            if batch_idx != 0:
                                # Record detailed progress
                                user_time = time.time() - start_time
                                surplus_time = user_time / (progress1_count + 1) * (self.valCount - progress1_count - 1)
                                log_info = f'[Epoch {turn + 1} training progress {progress1_count + 1 / self.valCount * 100:.2f}%] [Time elapsed {user_time:.0f}s] [Estimated remaining time {surplus_time:.0f}s]'
                                self.progress1.emit(log_info, 0)

                    # Update training progress info
                    if (iter_count + 1) % self.valCount == 0:
                        if self.valCount % 10 == 0:
                            user_time = time.time() - start_time
                            surplus_time = user_time / (progress1_count + 1) * (
                                        self.valCount - progress1_count - 1) if progress1_count + 1 > 0 else 0

                            progress_percent = (progress1_count + 1) / self.valCount * 100 if self.valCount > 0 else 0
                            log_info = f'[Epoch {turn + 1} training progress {progress_percent:.2f}%] [Time elapsed {user_time:.0f}s] [Estimated remaining time {surplus_time:.0f}s]'

                            self.progress1.emit(log_info, 0)
                        else:
                            self._update_progress_info(turn, progress1_count, start_time)
                        start_time = time.time()
                        progress1_count = 0
                    else:
                        self._update_progress_info(turn, progress1_count, start_time)
                        progress1_count += 1

                    # Perform validation
                    if (iter_count + 1) % self.valCount == 0:
                        writer.add_scalar('Loss/TrainBigLoss', train_big_losses.update2(), train_big_losses.id)
                        lastEvalVal = self.validate(writer, evalVal, lastEvalVal, turn, turns)

                    iter_count += 1

                # If the last batch didn't reach backwardNumber times, update parameters as well
                if iter_count % self.backwardNumber != 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Clean up resources
            self._cleanup_resources()
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
                print(f"  Line number: {frame.lineno}")
                print(f"  Function: {frame.name}")
                print(f"  Code: {frame.line}\n")
        finally:
            return

    def _train_one_batch(self, img, mask):
        """Train a single batch of data

        Args:
            img: Input image
            mask: Target mask

        Returns:
            loss: Loss value
        """
        img = img.to(self.device)
        mask = mask.to(self.device)
        seg = self.super_net(img)  # Use network self.super_net to segment the image, obtaining segmentation result seg
        loss = self.loss_criterion(img, seg, mask)[0]  # Returned loss value is a tensor
        self.optimizer.zero_grad()
        loss.backward()  # Backpropagation
        self.optimizer.step()  # Update parameters
        return loss

    def _record_training_progress(self, turn, total_turns, batch_idx, total_batches,
                                  train_small_losses, writer):
        """Record training progress and loss value

        Args:
            turn: Current epoch
            total_turns: Total epochs
            batch_idx: Current batch index
            total_batches: Total number of batches
            train_small_losses: Short-term loss recorder
            writer: Log writer
            iter_count: Global iteration count
        """
        tmp_loss = train_small_losses.update2()
        writer.add_scalar('Loss/TrainSmallLoss', tmp_loss, train_small_losses.id)

        # Send progress info
        text = f"TRAIN [Epoch {turn} | {total_turns}] [Proce {batch_idx} | {total_batches}] [Loss {tmp_loss:.4f}]"
        self.progress1.emit(text, 1)

        # Save loss value to text file
        # with open(self.text_loss_path, "a", encoding="utf-8") as f:
        #     f.write(f"{iter_count + 1} {tmp_loss:.4f}\n")

    def _update_progress_info(self, turn, progress_count, start_time):
        """Update training progress info

        Args:
            turn: Current epoch
            progress_count: Progress count
            start_time: Start time
        """
        user_time = time.time() - start_time
        surplus_time = user_time / (progress_count + 1) * (self.valCount - progress_count - 1) if progress_count + 1 > 0 else 0

        progress_percent = (progress_count + 1) / self.valCount * 100 if self.valCount > 0 else 0
        log_info = f'[Epoch {turn + 1} training progress {progress_percent:.2f}%] [Time elapsed {user_time:.0f}s] [Estimated remaining time {surplus_time:.0f}s]'

        self.progress1.emit(log_info, progress_count)

    def _cleanup_resources(self):
        """Clean up resources used during training"""
        # Delete model reference
        if hasattr(self, 'super_net'):
            del self.super_net
            print("Model deleted")

        # Clear GPU cache
        torch.cuda.empty_cache()
        print("Cleaned up resources used during training")

        # Delete other resource references
        for attr in ['dataLoader', 'testLoader', 'loss_criterion', 'optimizer', 'lr_scheduler', 'eval_metric']:
            if hasattr(self, attr):
                delattr(self, attr)

        if not self.stop_training:
            self.Train_Stop()

    def validate(self, writer, evalVal, lastEvalVal, epoch, total_epochs):
        torch.cuda.empty_cache()
        self.super_net.eval()  # Set network to evaluation mode
        with torch.no_grad():
            random_v = random.randint(0, len(self.testLoader) - 1)
            eval_start_time = time.time()
            for batch_idx, (img, mask, name) in enumerate(self.testLoader):
                if self.stop_training:  # Check if training should stop
                    # print("Training stopped")
                    break
                if img.shape[0] != self.valBatchSize:
                    continue
                img = img.to(self.device)
                mask = mask.to(self.device)
                # Use network to segment the validation set
                seg = self.super_net(img)
                # Calculate evaluation metric
                eval = self.eval_metric(img, seg, mask)
                evalVal.update(eval)

                eval_userTime = time.time() - eval_start_time
                eval_surplusTime = eval_userTime / (batch_idx + 1) * (len(self.testLoader) - batch_idx - 1)
                eval_logInfo = '[Epoch %d validation progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                    (epoch + 1), (batch_idx + 1) / len(self.testLoader) * 100, eval_userTime, eval_surplusTime)

                if batch_idx == random_v and (epoch + 1) % 10 == 0:
                    ns = seg.cpu().detach().numpy()  # Training result -- Tag
                    ma = mask.cpu().detach().numpy()  # Target label
                    imgs = img.cpu().detach().numpy()  # Original image
                    ns_i = ns[:, 0]
                    ma_i = ma[:, 0]
                    img_i = imgs[:, 0]
                    self.save_result(ns_i, ma_i, img_i)
                    self.progress1.emit(eval_logInfo, 0)
                else:
                    self.progress1.emit(eval_logInfo, batch_idx)

            if self.stop_training:
                return

            curEvalVal = evalVal.update2()
            writer.add_scalar('Eval/EvalVal', curEvalVal, evalVal.id)
            curEvalVal = curEvalVal
            # print(f'curEvalVal: {curEvalVal}')
            # print(f"lastEvalVal: {lastEvalVal}")

            if self.view_count < 2 and curEvalVal > 0:
                self.view_count += 1
            if self.view_count == 2:
                self.show_preview.emit(True)

            if curEvalVal > lastEvalVal:
                # print('Validation metric improved, saving model')
                self.progress0.emit('Validation loss decreased, saving model')
                self.modSavePath = os.path.abspath(join(self.modelPath, "supernet_%s_best_%.4f.pth" % (
                                    str(epoch).zfill(5), curEvalVal)))
                torch.save({'state_dict': self.super_net.state_dict(), 'param': self.optimizer},
                           self.modSavePath)
                self.update_modSavePath.emit(self.modSavePath)
                # If more than one saved model, remove the previous one
                if len([l for l in os.listdir(self.modelPath) if ".pth" in l]) > 1:
                    self.delete_oldest_file(self.modelPath)
                lastEvalVal = curEvalVal

                # with open(self.text_save_model_path, "a", encoding="utf-8") as f:
                #     f.write(f"{iter_count + 1} Validation loss decreased, saving model {curEvalVal:.4f}\n")
                # f.close()

            # Adjust learning rate
            self.lr_scheduler.step(curEvalVal)
            lr = self.optimizer.param_groups[0]['lr']
            writer.add_scalar('TrainParam/Lr', lr, evalVal.id)
            text = f'VAL [Epoch {epoch} | {total_epochs}] [EvalVal: {curEvalVal:.4f}] [Lr: {lr:.8f}]'
            self.progress0.emit(text)
            # print(f'VAL [Epoch {epoch} | {total_epochs}] [EvalVal: {curEvalVal:.4f}] [Lr: {lr:.8f}]')

            # with open(self.text_eval_path, "a", encoding="utf-8") as f:
            #     f.write(f"{iter_count + 1} {curEvalVal:.4f}\n")
            # f.close()
        torch.cuda.empty_cache()
        time.sleep(1)
        # After validation, set network back to training mode
        self.super_net.train()
        return lastEvalVal

    def save_result(self, net_seg, mask, img):
        if self.stop_training:
            return

        root = self.rootPath
        i = random.randint(0, len(mask) - 1)

        path = f"{root}/TrainResultsPreview/{i + 1}/"
        os.makedirs(path, exist_ok=True)
        net_seg_i = ((net_seg[i] - net_seg[i].min()) / (net_seg[i].max() - net_seg[i].min()) * 255).astype(np.uint8)
        mask_i = ((mask[i] - mask[i].min()) / (mask[i].max() - mask[i].min()) * 255).astype(np.uint8)
        img_i = ((img[i] - img[i].min()) / (img[i].max() - img[i].min()) * 255).astype(np.uint8)

        net_seg_i[net_seg_i < 103] = 0

        seg_name = f"net_seg{i + 1}.tif"
        tiff.imwrite(join(path, seg_name), net_seg_i, compression="lzw")
        tiff.imwrite(join(path, f"mask{i + 1}.tif"), mask_i, compression="lzw")
        tiff.imwrite(join(path, f"img{i + 1}.tif"), img_i, compression="lzw")

        img_i_2d = self.MaxProject(img_i, 1)
        mask_i_2d = self.MaxProject(mask_i, 0)
        pre_i_2d = self.MaxProject(net_seg_i, 0)

        self.preview0.emit(img_i_2d, mask_i_2d, pre_i_2d)

    def Train_Stop(self):
        self.stop_training = True
        time.sleep(3)
        try:
            # Delete model reference
            if hasattr(self, 'super_net'):
                del self.super_net
                # print("Model deleted")
            # Clear GPU cache
            torch.cuda.empty_cache()
            print("GPU cache cleared")
            del self.dataLoader
            del self.testLoader
            del self.loss_criterion
            del self.optimizer
            del self.lr_scheduler
            del self.eval_metric
        except Exception as e:
            print("Stop")

    def MaxProject(self, img, is_enhance):
        img = np.max(img, axis=0)
        img = self.normalize_and_scale_to_uint8(img)
        if is_enhance:
            self.enhance_contrast_histogram_equalization(img)
        return img

    def enhance_contrast_histogram_equalization(self, img):
        """
        Enhance image contrast using histogram equalization.

        Parameters:
            img (numpy.ndarray): Input image.
        Returns:
            numpy.ndarray: Contrast-enhanced image.
        """
        # Apply histogram equalization
        img_eq = cv2.equalizeHist(img)
        return img_eq

    def normalize_and_scale_to_uint8(self, data):
        """
        Normalize grayscale values to 0-1, then scale to 0-255 and convert to uint8.

        Parameters:
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

        # Create a list of file paths and creation times
        files_with_ctime = []
        for file in files:
            file_path = join(directory, file)
            if os.path.isfile(file_path):  # Ensure it is a file, not a directory
                creation_time = os.path.getctime(file_path)
                files_with_ctime.append((file_path, creation_time))

        # Sort by creation time, oldest first
        files_with_ctime.sort(key=lambda x: x[1])

        # Delete the oldest file
        oldest_file_path = files_with_ctime[0][0]
        os.remove(oldest_file_path)
        # print(f"Deleted oldest file: {oldest_file_path}")