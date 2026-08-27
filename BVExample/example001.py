import os
from tqdm import tqdm
import tifffile

from BVMoudle.BVMoudle import BVReader

if __name__ == '__main__':
    reader = BVReader()
    # img1 = reader.readBV(r'D:\CH2\CH2-bv\0\3_3.bv')  # 读单块
    # img2 = reader.readBV(r'E:\slice3D\1\0_0.bv', 0, 20)  # 读单块,带起始帧和帧数
    # roiImg = reader.loadROI(r'D:\CH2\CH2-bv\config.cfg', 0, 512, 1024, 512, 1024, 512, 1024)  # 读roi
    # cfg = reader.readConfig(r'D:\CH2\CH2-bv\config.cfg')  # 读cfg
    # shape = cfg['bigImgSize']
    # roiImg = reader.loadROI(r'D:\CH2\CH2-bv\config.cfg', 0, 0, shape[0], 0, shape[1], 0, shape[2])  # 读roi
    # roiImg = reader.loadROI(r'D:\SY\CellData\bv_data\2048_2560_512_data\images\config.cfg', 2, 0, 272, 0, 272, 0, 144)  # 读roi
    # tifffile.imwrite("res.tif", roiImg, compression="lzw")
    print()
    # tifffile.imwrite(r"D:\CH2\bv-test\bv-test\3_3.tif", img1, compression="lzw")

    # cfg = reader.readConfig(r"D:\SY\cell_data\data_bv\config.cfg")
    # shape = cfg['bigImgSize']
    # roiImg = reader.loadROI(r"D:\SY\cell_data\data_bv\config.cfg",
    #                         0, 0, shape[0], 0, shape[1], 0, shape[2])
    # tifffile.imwrite(r"D:\SY\cell_data\data_bv\config.tif", roiImg, compression="lzw")

    # 26600
    # 16949
    # 4400
    save_dir = r"D:\SY\CellData\bv_data\WM-Soma-8399dst-BV"
    x1 = 0
    x2 = 26600
    y1 = 0
    y2 = 16949
    lv = 2
    for i in tqdm(range(int(4400 / 4) - 1)):
        save_path = os.path.join(save_dir, f"slice_{str(i).zfill(4)}.tif")
        z1 = i * 2 ** lv
        z2 = (i + 1) * 2 ** lv
        roiImg = reader.loadROI(r'F:\WM-Soma-8399dst-BV\config.cfg', lv, x1, x2, y1, y2, z1, z2)
        image = roiImg[0]
        tifffile.imwrite(save_path, image, compression="lzw")
        # break
