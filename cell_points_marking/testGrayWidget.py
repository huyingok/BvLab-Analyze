import sys

import tifffile
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QApplication

from bv.BVUtil import getHist


from PyQt5 import QtCore, QtWidgets

from GrayAdjustWidget import GrayAdjustWidget


class Ui_GrayWindow(object):
    def setupUi(self, GrayWindow):
        GrayWindow.setObjectName("GrayWindow")
        GrayWindow.resize(599, 227)
        self.centralwidget = QtWidgets.QWidget(GrayWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.grayWidget = GrayAdjustWidget()
        self.grayWidget.setObjectName("grayWidget")
        self.verticalLayout.addWidget(self.grayWidget)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.txtMinGray = QtWidgets.QSpinBox(self.centralwidget)
        self.txtMinGray.setMinimumSize(QtCore.QSize(65, 0))
        self.txtMinGray.setObjectName("txtMinGray")
        self.horizontalLayout.addWidget(self.txtMinGray)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.txtMaxGray = QtWidgets.QSpinBox(self.centralwidget)
        self.txtMaxGray.setMinimumSize(QtCore.QSize(65, 0))
        self.txtMaxGray.setObjectName("txtMaxGray")
        self.horizontalLayout.addWidget(self.txtMaxGray)
        self.verticalLayout.addLayout(self.horizontalLayout)
        GrayWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(GrayWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 599, 23))
        self.menubar.setObjectName("menubar")
        GrayWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(GrayWindow)
        self.statusbar.setObjectName("statusbar")
        GrayWindow.setStatusBar(self.statusbar)

        self.retranslateUi(GrayWindow)
        QtCore.QMetaObject.connectSlotsByName(GrayWindow)

    def retranslateUi(self, GrayWindow):
        _translate = QtCore.QCoreApplication.translate
        GrayWindow.setWindowTitle(_translate("GrayWindow", "MainWindow"))



class GrayWindow(QMainWindow, Ui_GrayWindow):

    def __init__(self, parent=None):
        super(GrayWindow, self).__init__(parent)
        self.setupUi(self)
        self.grayWidget.leftValueChanged.connect(self.grayTransCallback)
        self.grayWidget.rightValueChanged.connect(self.grayTransCallback)
        self.txtMinGray.setMinimum(-9999999)
        self.txtMaxGray.setMinimum(-9999999)
        self.txtMinGray.setMaximum(9999999)
        self.txtMaxGray.setMaximum(9999999)
        self.init_data()

    def init_data(self):
        bigImg = tifffile.imread(r'D:\SY\Global_Registration\New_Conding\TrainRes\RegistratDataSet001\FFT_3d\FFT_DLMab#180725.tif')
        minGray = bigImg.min()
        maxGray = bigImg.max()
        self.grayWidget.setLeftValue(minGray)
        self.grayWidget.setRightValue(maxGray)
        self.txtMinGray.setValue(minGray)
        self.txtMaxGray.setValue(maxGray)
        hist = getHist(bigImg)
        self.grayWidget.setHist(hist)

    def grayTransCallback(self):
        minGray = self.grayWidget.getLeftValue()
        maxGray = self.grayWidget.getRightValue()
        self.txtMinGray.setValue(minGray)
        self.txtMaxGray.setValue(maxGray)


if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    window = GrayWindow()
    # app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    window.show()
    sys.exit(app.exec_())
