import json
import os


config_path = os.path.join(os.getcwd(), 'lockCfg.json')
# [x,y,z]
config = {
    # 'root': "F:\\WM-Soma-8399dst-BV",
    'root': "F:\LiAnAn-233100-CH1-BV",
    # 'saveRoot': "D:\\SY\\10GTestData\\10GNeuralTestData\\block_2000\\Cell_Cluster_Results",
    'saveRoot': "D:\\SY\\10GTestData\\10GNeuralTestData\\block_2000\\Images2",
    # 'rectBoxLs': [[[0, 0, 0], [26600, 16949, 4400]]],  # 细胞
    # 'rectBoxLs': [[[0, 0, 0], [27868, 21000, 11611]]],  # 神经
    'rectBoxLs': [[[2139, 4047, 1794], [25657, 18002, 11344]]],  # 神经截取
    'n_clusters': 1000,
    'getClsNum': 1,
    'MNumber': 5,
    'level': 0,
    # 'sampleXYZ': [4, 4, 2],
    'sampleXYZ': [6, 6, 2],
    # 'smallBatchSize': [272, 272, 144],
    'smallBatchSize': [192, 192, 192],
    # 'batchSize': [512, 512, 144],
    'batchSize': [512, 512, 192],
    'redunSize': [32, 32, 32]
}


with open(config_path, "w", encoding="utf-8") as f:
    f.write(json.dumps(config, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    from ViewInfo import GetBigBVLevelInfo
    from MulGetHist import BvBigDataToHist
    from HistAnaly import HistFiterBv
    from BvBigDataAndBatchIdLsToTifLs import MulBatchLsToTifLs

    with open(config_path, 'r') as f:
        config = json.loads(f.read())
    print(config)

    # 查看块信息
    GetBigBVLevelInfo()
    # 提取每一块特征
    BvBigDataToHist()
    # Clustering
    HistFiterBv()
    # 聚类结果提取图像
    MulBatchLsToTifLs()
