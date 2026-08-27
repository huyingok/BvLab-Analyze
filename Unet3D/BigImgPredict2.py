'''大图预测'''
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time, torch
import numpy as np
import tifffile
import shutil
from os.path import join
from Unet3D.ModelPredictPy import ModelPredictClass
# from PyQt5.QtCore import QThread, pyqtSignal
#
#
# class BigImgPredict1(QThread):
#     finish1 = pyqtSignal()
#     seg_progress1 = pyqtSignal(str, str, float)
#     error1 = pyqtSignal(str)
#
#     def __init__(self, *args, **kwargs):
#         super(BigImgPredict1, self).__init__()
#         self.win = kwargs.get('win')
#
#     def run(self):
#         try:
#             imgPath = self.win.p_dict['imgDir']
#             segSaveDir = self.win.p_dict['segSaveDir']
#             modelPath = self.win.p_dict['modPath']
#             shapes = self.win.p_dict['shapes']
#             is_keep = self.win.p_dict["is_keep"]
#             is_xueguan = self.win.p_dict.get("is_xueguan")
#
#             modSegDir = join(segSaveDir, "mod_seg")  # 模型分割结果
#             os.makedirs(segSaveDir, exist_ok=True)
#
#             if is_xueguan:
#                 distSegDir = join(segSaveDir, "dist_seg")  # 距离场分割结果
#                 if is_keep:
#                     os.makedirs(distSegDir, exist_ok=True)
#                 else:
#                     shutil.rmtree(distSegDir, ignore_errors=True)
#                     os.makedirs(distSegDir, exist_ok=True)
#
#             ls = os.listdir(imgPath)
#             if is_keep:
#                 seg_names = []
#                 if os.path.exists(modSegDir):
#                     # 存在的追踪结果不为空
#                     seg_names = [l for l in os.listdir(modSegDir) if '.tif' in l]
#                 ls = [l for l in ls if '.tif' in l and l not in seg_names]
#             else:
#                 ls = [l for l in ls if '.tif' in l]
#                 shutil.rmtree(modSegDir, ignore_errors=True)
#
#             os.makedirs(modSegDir, exist_ok=True)
#
#             if len(ls) > 0:
#                 imgSize = np.array(shapes, dtype=np.int32)
#                 r = np.array([32, 32, 32], dtype=np.int32)  # 冗余
#                 rSp = np.array([16, 16, 16], dtype=np.int32)  # 边缘不要部分
#                 # batchSize = 1
#                 fieldLen = 16
#                 device = torch.device('cuda:0')
#                 model = ModelPredictClass(modelPath, fieldLen=fieldLen, device=device)  # 预测类
#
#                 if is_xueguan:
#                     total = len(ls) * 2
#                 else:
#                     total = len(ls)
#
#                 # saveInfo = []
#                 self.seg_progress1.emit(f"{0} / {total}", "", 0.0)
#                 for ii, name in enumerate(ls):
#                     if self.win.pred_is_stop:
#                         break
#                     # print(name)
#                     newName, fileType = os.path.splitext(name)
#                     if not fileType in ['.tif']:
#                         continue
#                     s1 = time.time()
#                     oriImg = tifffile.imread(join(imgPath, name))
#
#                     if oriImg.ndim != 3: continue
#
#                     bigImgSize = np.array(list(oriImg.shape)[::-1], dtype=np.int32)
#
#                     # print(list(oriImg.shape)[::-1])
#
#                     maskBigImg = np.zeros(bigImgSize[::-1], dtype=np.uint8)
#                     sliceNumber = np.ceil((bigImgSize - imgSize) / (imgSize - r)).astype(np.int32) + 1
#                     for nz in range(sliceNumber[2]):
#                         for ny in range(sliceNumber[1]):
#                             for nx in range(sliceNumber[0]):
#                                 sp = (imgSize - r) * [nx, ny, nz]
#                                 ep = np.min([sp + imgSize, bigImgSize], axis=0)
#                                 sp = np.min([sp, ep - imgSize], axis=0)
#                                 img = oriImg[sp[2]: ep[2], sp[1]: ep[1], sp[0]: ep[0]]
#                                 mask = model(img)
#                                 rsp2 = rSp * np.sign([nx, ny, nz])
#                                 sp += rsp2
#                                 maskBigImg[sp[2]: ep[2], sp[1]: ep[1], sp[0]: ep[0]] = mask[rsp2[2]:, rsp2[1]:, rsp2[0]:]
#                     maskBigImg[maskBigImg < 103] = 0
#                     tifffile.imwrite(join(modSegDir, '%s.tif' % newName), maskBigImg, compression='lzw')
#
#                     self.seg_progress1.emit(f"{ii + 1} / {total}", str(name), float(time.time() - s1))
#                     if self.win.pred_is_stop:
#                         break
#                 del model
#                 torch.cuda.empty_cache()
#             self.finish1.emit()
#         except Exception as e:
#             self.error1.emit(str(e))
#             import sys
#             import traceback
#             print("\n=== Error message ===")
#             print(f"Exception type: {type(e).__name__}")
#             print(f"Error message: {e}")
#             print("=== Error location ===")
#             tb = sys.exc_info()[2]
#             for frame in traceback.extract_tb(tb):
#                 print(f"  File: {frame.filename}")
#                 print(f"  Line number: {frame.lineno}")
#                 print(f"  Function: {frame.name}")
#                 print(f"  Code: {frame.line}\n")


def BigImgPredict2(imgPath, savePath, modelPath, shapes):
    # root = r"D:\xueguan\TrainDateSet2"
    # imgPath = os.path.join(root, "images")
    # savePath = os.path.join(root, "Predict2")
    # modelPath = r'D:\UNet3D\Unet3D\ModelSave\exp019\supernet_000.pth'
    # shapes = [192, 192, 192]

    imgSize = np.array(shapes, dtype=np.int32)
    r = np.array([32, 32, 32], dtype=np.int32)  # 冗余
    rSp = np.array([16, 16, 16], dtype=np.int32)  # 边缘不要部分
    # batchSize = 1
    fieldLen = 16
    device = torch.device('cuda:0')
    os.makedirs(savePath, exist_ok=True)
    model = ModelPredictClass(modelPath, fieldLen=fieldLen, device=device)  # 预测类
    ls = os.listdir(imgPath)
    # ls = [l for l in ls if '.tif' in l]
    ls = [l for l in ls if '.tif' in l][:3]
    # saveInfo = []
    for ii, name in enumerate(ls):
        print(name)
        newName, fileType = os.path.splitext(name)
        if not fileType in ['.tif']:
            continue
        s1 = time.time()
        oriImg = tifffile.imread(join(imgPath, name))

        if oriImg.ndim != 3: continue

        bigImgSize = np.array(list(oriImg.shape)[::-1], dtype=np.int32)

        print(list(oriImg.shape)[::-1])

        maskBigImg = np.zeros(bigImgSize[::-1], dtype=np.uint8)
        sliceNumber = np.ceil((bigImgSize - imgSize) / (imgSize - r)).astype(np.int32) + 1
        for nz in range(sliceNumber[2]):
            for ny in range(sliceNumber[1]):
                for nx in range(sliceNumber[0]):
                    sp = (imgSize - r) * [nx, ny, nz]
                    ep = np.min([sp + imgSize, bigImgSize], axis=0)
                    sp = np.min([sp, ep - imgSize], axis=0)
                    img = oriImg[sp[2]: ep[2], sp[1]: ep[1], sp[0]: ep[0]]
                    mask = model(img)
                    rsp2 = rSp * np.sign([nx, ny, nz])
                    sp += rsp2
                    maskBigImg[sp[2]: ep[2], sp[1]: ep[1], sp[0]: ep[0]] = mask[rsp2[2]:, rsp2[1]:, rsp2[0]:]
        maskBigImg[maskBigImg < 103] = 0
        tifffile.imwrite(join(savePath, '%s.tif' % newName), maskBigImg, compression='lzw')
        # tifffile.imwrite(join(savePath, '%s.tif' % newName), maskBigImg, compression='lzw')
        print(ii + 1, len(ls), time.time() - s1)
        print()


if __name__ == '__main__':
    # root = r"D:\xueguan\TrainDateSet2"
    root = r"D:\SY\VesselData\tif_data\tif_exist_swc\data_14"
    imgPath = os.path.join(root, "images")
    savePath = os.path.join(root, "Predict2")
    modelPath = r'D:\SY\CellNeuralBloodVessel\IntegratePoseOptimizationEnglish\DefaultModels\VesselModels\Vessel_M.pth'
    shapes = [192, 192, 192]
    BigImgPredict2(imgPath, savePath, modelPath, shapes)
