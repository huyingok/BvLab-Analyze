'''大图预测'''
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time, torch
import numpy as np
import tifffile
from os.path import join
from Seg_Net_main_PyQt.ModelPredictPy import ModelPredictClass
from PyQt5.QtCore import QThread, pyqtSignal


class BigImgPredict2(QThread):
    finish = pyqtSignal(int)
    seg_progress = pyqtSignal(str)
    error2 = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super(BigImgPredict2, self).__init__()
        self.win = kwargs.get('win')

    def run(self):
        try:
            imgPath = self.win.s2_dict['imgPath']
            savePath = self.win.s2_dict['savePath']
            modelPath = self.win.s2_dict['modelPath']
            shapes = self.win.s2_dict['shapes']

            # imgPath = r"D:\xueguan\TrainDateSet3\train_data\big_test_predict\dist"
            # savePath = r"D:\xueguan\TrainDateSet3\train_data\big_test_predict\save_192"
            # modelPath = r'D:\Seg_Net-main\ModelSave\exp031\supernet_000.pth'
            # shapes = [192, 192, 192]

            imgSize = np.array(shapes, dtype=np.int32)
            r = 32  # 冗余
            rSp = 16  # 边缘不要部分
            batchSize = 1
            fieldLen = 16
            # device = torch.device('cuda:1')
            device = torch.device('cuda:0')

            os.makedirs(savePath, exist_ok=True)
            model = ModelPredictClass(modelPath, fieldLen=fieldLen, device=device)  # 预测类
            # # ls = [os.listdir(imgPath)[220]]
            # ls = ['10623_8104_831.tif']
            ls = os.listdir(imgPath)
            saveInfo = []
            self.seg_progress.emit(f"{0} / {len(ls)}")
            for ii, name in enumerate(ls):
                s1 = time.time()
                oriImg = tifffile.imread(join(imgPath, name))
                bigImgSize = np.array(oriImg.shape, dtype=np.int32)
                # oriImg = (oriImg - oriImg.min()) / (oriImg.max() - oriImg.min())
                # oriImg = (exposure.equalize_hist(oriImg) * 2700).astype(np.uint16)
                # tifffile.imwrite(r'D:\qxz\MyProject\KKMarkCellBody\Code\NeuronTrack\DataSet\ProblemDataSet\SegProblemTest\SmallImages\ttt.tif', oriImg)
                # oriImg[oriImg > 125] = 125
                # oriImg
                if oriImg.ndim != 3:
                    continue
                maskBigImg = np.zeros(bigImgSize[::-1], dtype=np.uint8)
                sliceNumber = np.ceil((bigImgSize - imgSize) / (imgSize - r)).astype(np.int32) + 1
                newName = os.path.splitext(name)[0]
                for nz in range(sliceNumber[2]):
                    for ny in range(sliceNumber[1]):
                        for nx in range(sliceNumber[0]):
                            sp = (imgSize - r) * [nx, ny, nz]
                            ep = np.min([sp + imgSize, bigImgSize], axis=0)
                            sp = np.min([sp, ep - imgSize], axis=0)
                            img = oriImg[sp[2]: ep[2], sp[1]: ep[1], sp[0]: ep[0]]
                            # stdVal = img.std()
                            # if stdVal > 500:
                            #     print('**********************')
                            # print(name, nx, ny, nz, img.std())
                            mask = model(img)
                            rsp2 = rSp * np.sign([nx, ny, nz])
                            sp += rsp2
                            maskBigImg[sp[2]: ep[2], sp[1]: ep[1], sp[0]: ep[0]] = mask[rsp2[2]:, rsp2[1]:, rsp2[0]:]
                maskBigImg[maskBigImg < 103] = 0
                tifffile.imwrite(join(savePath, '%s.tif' % newName), maskBigImg)
                # saveInfo.append([join(saveRes, '%s.tif' % newName), join(imgPath, name), ''])
                # with open(cfgPath, 'w') as f:
                #     f.write(json.dumps(saveInfo))
                print(ii + 1, len(ls), time.time() - s1)
                self.seg_progress.emit(f"{ii + 1} / {len(ls)}")
            del model
            torch.cuda.empty_cache()
            self.finish.emit(2)
        except Exception as e:
            self.error2.emit(str(e))
            import sys
            import traceback
            print("=== Error message ===")
            print(f"异常类型: {type(e).__name__}")
            print(f"Error message: {e}")
            print("\n=== 错误位置 ===")
            tb = sys.exc_info()[2]
            frame = traceback.extract_tb(tb)[0]
            print(f"  File: {frame.filename}")
            print(f"  行号: {frame.lineno}")
            print(f"  Function: {frame.name}")
            print(f"  代码: {frame.line}\n")


def BigImgPredict(imgPath, savePath, modelPath, shapes):
    # imgPath = r"D:\xueguan\TrainDateSet3\train_data\big_test_predict\dist"
    # savePath = r"D:\xueguan\TrainDateSet3\train_data\big_test_predict\save_192"
    # modelPath = r'D:\Seg_Net-main\ModelSave\exp031\supernet_000.pth'
    # shapes = [192, 192, 192]

    imgSize = np.array(shapes, dtype=np.int32)
    r = 32          # 冗余
    rSp = 16        # 边缘不要部分
    batchSize = 1
    fieldLen = 16
    # device = torch.device('cuda:1')
    device = torch.device('cuda:0')

    os.makedirs(savePath, exist_ok=True)
    model = ModelPredictClass(modelPath, fieldLen=fieldLen, device=device)  # 预测类
    # # ls = [os.listdir(imgPath)[220]]
    # ls = ['10623_8104_831.tif']
    ls = os.listdir(imgPath)
    saveInfo = []
    for ii, name in enumerate(ls):
        s1 = time.time()
        oriImg = tifffile.imread(join(imgPath, name))
        bigImgSize = np.array(oriImg.shape, dtype=np.int32)
        # oriImg = (oriImg - oriImg.min()) / (oriImg.max() - oriImg.min())
        # oriImg = (exposure.equalize_hist(oriImg) * 2700).astype(np.uint16)
        # tifffile.imwrite(r'D:\qxz\MyProject\KKMarkCellBody\Code\NeuronTrack\DataSet\ProblemDataSet\SegProblemTest\SmallImages\ttt.tif', oriImg)
        # oriImg[oriImg > 125] = 125
        # oriImg
        if oriImg.ndim != 3:
            continue
        maskBigImg = np.zeros(bigImgSize[::-1], dtype=np.uint8)
        sliceNumber = np.ceil((bigImgSize - imgSize) / (imgSize - r)).astype(np.int32) + 1
        newName = os.path.splitext(name)[0]
        for nz in range(sliceNumber[2]):
            for ny in range(sliceNumber[1]):
                for nx in range(sliceNumber[0]):
                    sp = (imgSize - r) * [nx, ny, nz]
                    ep = np.min([sp + imgSize, bigImgSize], axis=0)
                    sp = np.min([sp, ep - imgSize], axis=0)
                    img = oriImg[sp[2]: ep[2], sp[1]: ep[1], sp[0]: ep[0]]
                    # stdVal = img.std()
                    # if stdVal > 500:
                    #     print('**********************')
                    # print(name, nx, ny, nz, img.std())
                    mask = model(img)
                    rsp2 = rSp * np.sign([nx, ny, nz])
                    sp += rsp2
                    maskBigImg[sp[2]: ep[2], sp[1]: ep[1], sp[0]: ep[0]] = mask[rsp2[2]:, rsp2[1]:, rsp2[0]:]
        maskBigImg[maskBigImg < 103] = 0
        tifffile.imwrite(join(savePath, '%s.tif' % newName), maskBigImg, compression="lzw")
        # saveInfo.append([join(saveRes, '%s.tif' % newName), join(imgPath, name), ''])
        # with open(cfgPath, 'w') as f:
        #     f.write(json.dumps(saveInfo))
        print(ii, len(ls), time.time() - s1)


if __name__ == '__main__':
    imgPath = r"D:\xueguan\TrainDateSet3\train_data\big_test_predict\dist"
    savePath = r"D:\xueguan\TrainDateSet3\train_data\big_test_predict\save_192"
    modelPath = r'D:\Seg_Net-main\ModelSave\exp031\supernet_000.pth'
    shapes = [192, 192, 192]
    BigImgPredict(imgPath, savePath, modelPath, shapes)
