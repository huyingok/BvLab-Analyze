import ctypes


class ImageROI(ctypes.Structure):
    _fields_ = [
        ("level", ctypes.c_int),
        ("pixSpace", ctypes.c_int),
        ("xStart", ctypes.c_int),
        ("yStart", ctypes.c_int),
        ("zStart", ctypes.c_int),
        ("xEnd", ctypes.c_int),
        ("yEnd", ctypes.c_int),
        ("zEnd", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("depth", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("configPath", ctypes.c_char_p)
    ]


class DataConfig(ctypes.Structure):
    _fields_ = [
        ("levelNum", ctypes.c_int),
        ("dataType", ctypes.c_int),
        ("scale", ctypes.POINTER(ctypes.c_int)),
        ("batchSize", ctypes.POINTER(ctypes.c_int)),
        ("zLen", ctypes.POINTER(ctypes.c_int)),
        ("bigImgSize", ctypes.POINTER(ctypes.c_int)),
        ("blockNum", ctypes.POINTER(ctypes.POINTER(ctypes.c_int))),
        ("resolution", ctypes.POINTER(ctypes.c_double)),
    ]


class BVImg(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("depth", ctypes.c_int),
        ("pixSpace", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
    ]


class BVImg1(ctypes.Structure):
    _fields_ = [
        ("dNum", ctypes.c_int),
        ("shape", ctypes.POINTER(ctypes.c_int)),
        ("pixSpace", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
    ]


class BVFiberImg(ctypes.Structure):
    _fields_ = [
        ("fileName", ctypes.c_char_p),
        ("zStart", ctypes.c_int),
        ("zEnd", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("depth", ctypes.c_int),
        ("pixSpace", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
    ]


class BVImgTwo(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("pixSpace", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
    ]


class Projection(ctypes.Structure):
    _fields_ = [
        ("dim", ctypes.c_int),
        ("xBatch", ctypes.c_int),
        ("yBatch", ctypes.c_int),
        ("xSize", ctypes.c_int),
        ("ySize", ctypes.c_int),
        ("zSize", ctypes.c_int),
        ("space", ctypes.c_int),
        ("bvPath", ctypes.c_char_p),
        ("savePath", ctypes.c_char_p),
        ("progress", ctypes.c_int),
        ("total", ctypes.c_int),
        ("finished", ctypes.c_bool)
    ]


class SwcData(ctypes.Structure):
    _fields_ = [
        ("treeNum", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("pointsNum", ctypes.POINTER(ctypes.c_int)),
        ("totalNum", ctypes.c_longlong)
    ]
