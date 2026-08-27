# -*- coding: utf-8 -*-
#  Pyinstaller -D --icon=./icon_image/icon_cell2.png StartProgramOfMark.py
import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from ControlStyle import StyleSheet
# from ViewMainWindow import MainWindow
from CellViewWidget import CellViewWidget


if __name__ == '__main__':
    # Get Program Running Path
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    app = QApplication(sys.argv)
    app.setStyleSheet(StyleSheet)
    # window = MainWindow()
    window = CellViewWidget()
    window.setWindowTitle('CellPoints[ Marking ]')
    icon = QIcon(os.path.join(base_path, 'icon_image', 'icon_cell2.png'))
    window.setWindowIcon(icon)
    # window.showMaximized()
    window.showNormal()
    sys.exit(app.exec_())


# from VTKViewer import VTKViewer
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     app.setStyleSheet(StyleSheet)
#     window = VTKViewer()
#     window.setWindowTitle("CellPointsMarkingWidgets")  # 设置标题
#     # window.showMaximized()
#     # window.setWindowIcon(QIcon(r'D:\python+vtk+Qt\cell_points_marking\pic.png'))  # 设置标题图标
#     # window.setWindowIcon(QIcon(r'D:\python3\icon\pic.png'))  # 设置标题图标
#     sys.exit(app.exec_())
