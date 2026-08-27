import numpy as np

def SliceBatch(bigImgSize, imgSize, redunSize):
    sliceNumber = np.ceil((1. * bigImgSize - imgSize) / (imgSize - redunSize)).astype(np.int32) + 1
    # sliceNumber = np.round((1. * bigImgSize - imgSize) / (imgSize - redunSize)).astype(np.int32) + 1
    # sliceNumber = np.array((1. * bigImgSize - imgSize) / (imgSize - redunSize)).astype(np.int32) + 1
    info = []
    for nz in range(sliceNumber[2]):
        for ny in range(sliceNumber[1]):
            for nx in range(sliceNumber[0]):
                sp = (imgSize - redunSize) * [nx, ny, nz]
                ep = np.min([sp + imgSize, bigImgSize], axis=0)
                sp = np.max([np.min([sp, ep - imgSize], axis=0), [0, 0, 0]], axis=0)
                info.append([sp, ep, nx, ny, nz])
    return info


if __name__ == '__main__':
    # bigImgSize = np.array([512, 512, 512], dtype=np.int32)
    bigImgSize = np.array([300, 300, 512], dtype=np.int32)
    # imgSize = np.array([192, 192, 192], dtype=np.int32)
    imgSize = np.array([272, 272, 144], dtype=np.int32)
    redunSize = np.array([32, 32, 32], dtype=np.int32)
    info = SliceBatch(bigImgSize, imgSize, redunSize)
    print(info)
    print(len(info))
