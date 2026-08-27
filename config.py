# -*- coding: utf-8 -*-
import os
import sys
from configparser import ConfigParser


# Get Program Running Path
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))


configDir = os.path.join(base_path, 'config')
cfgPath = os.path.join(configDir, 'config.ini')


class MyConfig(object):
    def __init__(self):
        self.cfg = ConfigParser()
        self.cfg.read(cfgPath, encoding='utf-8')


exe_cfg = MyConfig()


if __name__ == '__main__':
    print(exe_cfg.__dir__())

"""
exe_cfg.cfg[''][''] = ""
exe_cfg.cfg[''][''] = ""
exe_cfg.cfg[''][''] = ""
exe_cfg.cfg[''][''] = ""
with open(cfgPath, 'w') as configfile:
    exe_cfg.cfg.write(configfile)
"""
