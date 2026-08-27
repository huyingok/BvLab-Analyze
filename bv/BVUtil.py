import ctypes
import os
import sys
import time

import numpy as np

from bv.BaseType import BVImg, ImageROI, DataConfig, BVImg1, BVImgTwo, SwcData

if sys.version_info[:2] >= (3, 7):
    os.add_dll_directory(os.path.abspath(os.path.dirname(__file__)) + '/dll')
    # os.add_dll_directory(os.path.abspath(os.path.dirname(__file__)) + '/test')
    # ps = os.environ['path'].split(';')
    # for p in ps:
    #     try:
    #         os.add_dll_directory(p)
    #         projectionDll = ctypes.cdll.LoadLibrary('MultifunctHighDecoder.dll')
    #         print(p)
    #     except Exception as ex:
    #         print('ex======', ex)

else:
    os.environ['path'] = os.path.abspath(os.path.dirname(__file__)) + '/dll;' + os.environ['path']

# path = r'F:\dev\Gtree\bv\test'
# 寻找没用到的dll思路
# 程序成功运行时，已经加载的dll是无法删除的，所以遍历dllFolder，尝试删除，能删掉的就是程序没用到的
# ls = os.listdir(r'F:\dev\Gtree\bv\test')
# projectionDll = ctypes.cdll.LoadLibrary('MultifunctHighDecoder.dll')
# for name in ls:
#     try:
#         os.remove(os.path.join(path, name))
#     except Exception as e:
#         pass
#         # shutil.copy(os.path.join(r'F:\dev\Gtree\bv\test2', name), os.path.join(r'F:\dev\Gtree\bv\test', name))

dll = ctypes.cdll.LoadLibrary('BVUtil.dll')
dll.getHist.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_int]
dll.transGray.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int]
dll.getHist.restype = ctypes.POINTER(ctypes.c_double)
dll.getShape.restype = ctypes.POINTER(ctypes.c_int)


def getHist(img):
    img = img.copy()
    pixSpace = 1 if img.dtype == np.uint8 else 2
    histLen = 2 ** (8 * pixSpace)
    n = img.size
    ptr = img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)).contents
    histPtr = dll.getHist(ctypes.byref(ptr), n, pixSpace)
    p_arr_type = ctypes.POINTER(ctypes.c_double * 1 * histLen)
    obj = ctypes.cast(histPtr, p_arr_type).contents
    hist = np.asarray(obj)
    del img
    return hist


def freeHist(hist):
    ptr = hist.ctypes.data_as(ctypes.POINTER(ctypes.c_double)).contents
    dll.freeHist(ctypes.byref(ptr))
