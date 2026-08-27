'''数据集划分转Txt'''
import os, random
from os.path import join



def DataSetTxtMake(path, makePath, dataSetRadio=[0.9, 0.10, 0.00], setNameLs=['train', 'val', 'test'], logSignal=None):
    # 训练集,验证集,测试集比例
    # dataSetRadio = [0.9, 0.10, 0.00]
    setNameLs = ['train', 'val', 'test']
    # makePath = join(path, 'mask')
    ls = os.listdir(makePath)
    lsLen = len(ls)
    nameLs = []
    for ii, name in enumerate(ls):
        nameLs.append(name)
        if ii % 100 == 0:
            print('%d | %d' % (ii, lsLen), len(nameLs))
            if logSignal is not None:
                logSignal.emit('%d | %d %d\n' % (ii, lsLen, len(nameLs)))

    print(len(nameLs), lsLen)
    if logSignal is not None:
        logSignal.emit('%d %d\n' % (len(nameLs), lsLen))
    # 写入总数
    lsLen = len(nameLs)
    with open(join(path, 'totalName.txt'), 'w') as f:
        [f.write('%s\n' % name) for name in nameLs]
    spaceLs = [0, int(lsLen * dataSetRadio[0]), int(lsLen * (dataSetRadio[0] + dataSetRadio[1])),
               int(lsLen * (dataSetRadio[0] + dataSetRadio[1] + dataSetRadio[2]))]
    random.shuffle(nameLs)
    for ii in range(len(setNameLs)):
        curNameLS = nameLs[spaceLs[ii]: spaceLs[ii + 1]]
        with open(join(path, setNameLs[ii] + '.txt'), 'w') as f:
            [f.write('%s\n' % name) for name in curNameLS]


if __name__ == '__main__':
    path = r'D:\NeronDataSet\ZjHospital-C2804-DataSet\SegMentDataSet\TrainDataSet'
    maskPath = join(path, 'mask')
    dataSetRadio = [0.8, 0.1, 0.1]
    DataSetTxtMake(path, maskPath, dataSetRadio)
