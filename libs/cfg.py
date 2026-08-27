import json
import os
from os.path import join

import numpy as np


def getCfgInstance(refresh=False, only_default=False):
    cfg = CfgClass()
    if refresh:
        cfg.importConfig(os.path.join(os.getcwd(), 'lockCfg.json'), only_default)
    return cfg


class CfgClass:

    def __init__(self):
        self.root = None
        self.saveRoot = None
        self.rectBoxLs = []
        self.n_clusters = 50
        self.getClsNum = 10
        self.MNumber = 4
        self.level = 0
        self.sampleXYZ = np.array([4, 4, 2], dtype=np.int32)
        self.smallBatchSize = np.array([192, 192, 192], dtype=np.int32)
        self.batchSize = np.array([512, 512, 192], dtype=np.int32)
        self.redunSize = np.array([32, 32, 32], dtype=np.int32)

    def exportConfig(self):
        """
        导出配置
        :return:
        """
        config = {
            'root': self.root,
            'saveRoot': self.saveRoot,
            'rectBoxLs': self.rectBoxLs,
            'n_clusters': self.n_clusters,
            'getClsNum': self.getClsNum,
            'MNumber': self.MNumber,
            'level': self.level,
            'sampleXYZ': self.sampleXYZ.tolist(),
            'smallBatchSize': self.smallBatchSize.tolist(),
            'batchSize': self.batchSize.tolist(),
            'redunSize': self.redunSize.tolist()
        }
        return config

    def importConfig(self, filePath, only_default=False):
        """
        导入配置
        :param filePath:
        :param only_default:
        :return:
        """
        with open(filePath, 'r') as f:
            config = json.loads(f.read())
        if only_default:
            self.root = ''
            self.saveRoot = ''
            self.rectBoxLs = []
        else:
            self.root = config.get('root')
            self.saveRoot = config.get('saveRoot')
            if self.saveRoot is None:
                self.saveRoot = join(self.root, 'HistAnaly')
            if not os.path.exists(self.saveRoot):
                os.makedirs(self.saveRoot, exist_ok=True)
            self.rectBoxLs = config.get('rectBoxLs')
            if self.rectBoxLs is None:
                self.rectBoxLs = []
        self.n_clusters = config.get('n_clusters')
        self.getClsNum = config.get('getClsNum')
        self.MNumber = config.get('MNumber')
        self.level = config.get('level')
        self.sampleXYZ = np.array(config.get('sampleXYZ'), dtype=np.int32)
        self.smallBatchSize = np.array(config.get('smallBatchSize'), dtype=np.int32)
        self.batchSize = np.array(config.get('batchSize'), dtype=np.int32)
        self.redunSize = np.array(config.get('redunSize'), dtype=np.int32)

    def lockCfg(self):
        """
        锁定配置
        :return:
        """
        lockPath = os.path.join(os.getcwd(), 'lockCfg.json')
        config = self.exportConfig()
        with open(lockPath, 'w') as f:
            f.write(json.dumps(config))

    def checkParam(self):
        """
        检查参数
        :return:
        """
        errInfo = ''
        ret = True
        if self.root is None:
            errInfo += 'DataDir is None\n'
        if self.saveRoot is None:
            errInfo += 'WorkDir is None\n'
        elif not os.path.exists(self.saveRoot):
            errInfo += self.saveRoot + 'Not Exists\n'
        if errInfo != '':
            ret = False
        return ret, errInfo
