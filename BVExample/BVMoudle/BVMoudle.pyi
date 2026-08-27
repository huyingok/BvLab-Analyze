from typing import Tuple, Dict, List

import numpy as np


def getHist(data: np.ndarray) -> np.ndarray:
    """
    计算数组直方图
    :param data:
    :return:
    """
    pass


class BVReader:
    def readBV(self, filePath, startIdx=0, slice=0) -> np.ndarray:
        """
        读取BV文件
        :param filePath:文件路径
        :param startIdx:开始帧
        :param slice:读取帧数
        :return:
        """
        pass

    def getShape(self, filePath) -> Tuple:
        """
        获取BV文件尺寸信息
        :param filePath:bv文件路径
        :return:(width,height,depth,pixSpace)
        """
        pass

    def readConfig(self, filePath) -> Dict:
        """
        读取BV大数据格式配置文件
        :param filePath:配置文件路径
        :return:
        """
        pass

    def readFrame(self, filePath, zIdx=0) -> np.ndarray:
        """
        读取BV文件指定帧
        :param filePath:
        :param zIdx:
        :return:
        """
        pass

    def loadROI(self, configPath, level, x1, x2, y1, y2, z1, z2) -> np.ndarray:
        """
        读取大数据格式ROI区域
        :param configPath:
        :param level:
        :param x1:
        :param x2:
        :param y1:
        :param y2:
        :param z1:
        :param z2:
        :return:
        """
        pass

    def addROIWork(self, configPath, level, x1, x2, y1, y2, z1, z2) -> np.ndarray:
        """
        添加读取大数据格式ROI区域任务
        :param configPath:
        :param level:
        :param x1:
        :param x2:
        :param y1:
        :param y2:
        :param z1:
        :param z2:
        :return:
        """
        pass

    def getROIResult(self) -> np.ndarray:
        """
        获取ROI区域读取结果
        :return:
        """
        pass

    def numpyToBV(self, savePath, img, encodeType='bv1') -> np.ndarray:
        """
        numpy数组转bv
        :param savePath:
        :param img:
        :param encodeType:
        :return:
        """
        pass

    def initMultiReader(self, threadNum=16, maxMakeLen=30):
        """
        初始化批量BV解码器
        :param threadNum:
        :param maxMakeLen:
        :return:
        """
        pass

    def getDecBV(self) -> List[str, np.ndarray]:
        """
        获取批量解码结果
        :return:
        """
        pass

    def addDecInfo(self, filePaths):
        """
        添加解码任务
        :param filePaths:
        :return:
        """
        pass

    def initMultiMeanReader(self, threadNum=16, maxMakeLen=30):
        """
        初始化批量BV解码器
        :param threadNum:
        :param maxMakeLen:
        :return:
        """
        pass

    def getDecBVMean(self) -> List[str, np.ndarray]:
        """
        获取批量解码结果
        :return:
        """
        pass

    def addDecInfoMean(self, filePaths):
        """
        添加解码任务
        :param filePaths:
        :return:
        """
        pass

    def setSpace(self, x, y, z):
        pass
