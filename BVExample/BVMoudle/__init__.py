"""
此模块封装了BV格式的常用工具，以及一些常用算法
"""
import os
import sys

if sys.version_info[:2] >= (3, 7):
    os.add_dll_directory(os.path.abspath(os.path.dirname(__file__)) + '/libs')
else:
    os.environ['path'] = os.path.abspath(os.path.dirname(__file__)) + '/libs;' + os.environ['path']

from BVExample.BVMoudle.BVMoudle import BVReader, getHist