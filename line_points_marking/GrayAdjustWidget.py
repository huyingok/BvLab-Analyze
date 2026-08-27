#!/usr/bin/python3
# -*- coding: utf-8 -*-
import threading
import time
from queue import Queue

import numpy as np
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QPointF
from PyQt5.QtGui import QPainter, QColor, QPen, QPolygon, QPainterPath
from pynput.mouse import Controller, Button


class GrayAdjustWidget(QWidget):
    """
    灰度直方图调节器
    """
    leftValueChanged = pyqtSignal(int)  # 左滑块值变动信号
    rightValueChanged = pyqtSignal(int)  # 右滑块值变动信号

    renderEvent = pyqtSignal()

    class SliderColor:
        border = QColor(0, 0, 0)  # 边框颜色
        background = QColor(0, 0, 0)  # 背景颜色
        hist = QColor(89, 229, 89, 100)  # 直方图颜色
        leftSliderBorder = QColor(132, 203, 191)  # 左滑块边框
        leftBrush = QColor('#e1e1e1')  # 左滑块填充
        leftLine = QColor(132, 203, 191)  # 左滑块竖线
        rightSliderBorder = QColor(132, 203, 191)  # 右滑块边框
        rightBrush = QColor('#e1e1e1')  # 右滑块填充
        rightLine = QColor(132, 203, 191)  # 右滑块竖线
        maskBorder = QColor(255, 255, 255, 20)  # 遮罩边框
        maskBrush = QColor(255, 255, 255, 25)  # 遮罩填充

    def __init__(self, parent=None):
        super().__init__()
        # 私有变量
        self._hist = None  # 直方图数据
        self.setMinimumSize(30, 45)  # 设置最小尺寸
        self._sliderWidth = 10  # 滑块宽度值
        self._left_slider_border = 0  # 左滑块边框宽度
        self._right_slider_border = 0  # 右滑块边框宽度
        self._left_value = 0  # 左滑块值
        self._right_value = 255  # 右滑块值
        self._leftMove = False  # 左滑块是否可移动
        self._rightMove = False  # 右滑块是否可移动
        self._rangeMove = False  # 两滑块区间是否可移动
        self._pressPos = None  # 鼠标左键按下时的坐标
        self._curPos = None  # 鼠标左键按下时的坐标
        self._size = self.size()
        # 颜色表
        self._style = self.getStyle()
        # 公共变量
        self.maxValue = 255  # 最大灰度值
        self.minValue = 0  # 最小灰度值
        self.lv = 0
        self.rv = 255
        self.first = True
        self.msgQueue = Queue()  # Message queue
        self.mouse = Controller()

        # self.bind_events()
        self.repaint()

    # =================================================外部接口开始============================================================
    def setHist(self, hist):
        """
        设置直方图数据
        :param hist:直方图数据
        :return:
        """
        self._hist = hist
        self.repaint()

    def setLeftValue(self, value):
        """
        设置左滑块的值
        :return:
        """
        self.lv = int(value)
        ps, pe, lp, rp, pLen, Length, minV, maxV = self._getWidgetInfo()
        self._left_value = lp
        self.repaint()

    def setRightValue(self, value):
        """
        设置右滑块的值,外部接口
        :return:
        """
        self.rv = int(value)
        ps, pe, lp, rp, pLen, Length, minV, maxV = self._getWidgetInfo()
        self._right_value = rp
        self.repaint()

    def getRightValue(self):
        """
        获取右滑块的值,外部接口
        :return:
        """
        ps, pe, lp, rp, pLen, Length, minV, maxV = self._getWidgetInfo()
        # return maxV
        v = ((self._right_value - ps) / Length) * (maxV - minV) + minV
        gray = int(v + 0.5)
        # gray = math.ceil(v)
        return gray

    def getLeftValue(self):
        """
        获取左滑块的值
        :return:
        """
        ps, pe, lp, rp, pLen, Length, minV, maxV = self._getWidgetInfo()
        v = ((self._left_value - ps) / Length) * (maxV - minV) + minV
        gray = int(v + 0.5)
        return gray

    def getStyle(self):
        return self.SliderColor()

    def setStyle(self, style):
        self._style = style

    # ==================================================外部接口结束========================================================

    def bind_events(self):
        self.renderEvent.connect(self.repaint)
        self.t = threading.Thread(target=self._autoAdjust)
        self.t.setDaemon(True)
        self.t.start()

    def _autoAdjust(self):
        while True:
            # 读空消息队列
            self.msgQueue.get()
            qsize = self.msgQueue.qsize()
            [self.msgQueue.get() for i in range(qsize)]
            while True:
                if self._pressPos is None:
                    break
                sleepTime = self.updateEdge()
                self.renderEvent.emit()
                time.sleep(sleepTime)

    def _getWidgetInfo(self):
        width = self._size.width()
        # Len = width - 2 * self._sliderWidth - self._left_slider_border - self._right_slider_border
        ps = self._sliderWidth + self._left_slider_border  # 控件指针左边起点坐标
        pe = width - self._sliderWidth - self._right_slider_border  # 控件指针右边起点坐标
        Length = pe - ps  # 控件区间长度，Pixel
        pLen = 0.6 * Length  # 中央区间长度
        lp = 0.2 * Length + ps  # 映射后左指针位置
        rp = pe - 0.2 * Length  # 映射后右指针位置
        space = int((self.rv - self.lv) / 0.6 * 0.2)  # 每像素代表刻度
        if space == 0:
            space = 5
        minV = self.lv - space
        maxV = self.rv + space
        # minV = 1.2 * self.lv - 0.2 * self.rv
        # maxV = 1.2 * self.rv + 0.2 * self.lv
        return ps, pe, lp, rp, pLen, Length, minV, maxV

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        """
        重写鼠标按下事件
        :param e:
        :return:
        """
        x = e.pos().x()
        if x <= self._left_value + self._sliderWidth and self._left_value - x < self._sliderWidth * 2:
            # 如果鼠标当前位置小于左值且处于滑块内
            self._left_slider_border = 0
            self._pressPos = e.pos()
            self._leftMove = True
            self._curPos = self.mapToGlobal(e.pos())
            # print('left Move')
        elif x >= self._right_value - self._sliderWidth and x - self._right_value < self._sliderWidth * 2:
            self._right_slider_border = 0
            self._pressPos = e.pos()
            self._rightMove = True
            # print('Right move Move')
        elif self._right_value >= x >= self._left_value:
            self._left_slider_border = 0
            self._right_slider_border = 0
            self._rangeMove = True
            self._pressPos = e.pos()
            # print('rangeMove')
        # print(x, self._leftMove, self._rightMove, self._rangeMove)
        self.repaint()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent) -> None:
        x = e.pos().x()
        # print(x)
        if self._rangeMove:
            self._setLeftValue(self._left_value + x - self._pressPos.x())
            self._setRightValue(self._right_value + x - self._pressPos.x())
            self._pressPos = e.pos()
        elif self._leftMove:
            if x < 0:
                pos = self._curPos
                self.mouse.release(Button.left)
                self.mouse.position = (pos.x(), pos.y())
                self.mouse.press(Button.left)
                # self._leftMove = True
            else:
                self._setLeftValue(x)
            # self._curPos = e.pos()
        elif self._rightMove:
            self._setRightValue(x)
        self.repaint()

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent) -> None:
        self._left_slider_border = 0
        self._right_slider_border = 0
        self._pressPos = None
        # self._curPos = None
        lv = self.getLeftValue()
        rv = self.getRightValue()
        ps, pe, lp, rp, pLen, Length, minV, maxV = self._getWidgetInfo()
        if abs(rv - lv) >= 10:
            if self._leftMove:
                self.lv = lv
                self._left_value = lp
            elif self._rightMove:
                self.rv = rv
                self._right_value = rp
            elif self._rangeMove:
                self.lv = lv
                self.rv = rv
                self._right_value = rp
                self._left_value = lp
        self._leftMove = False
        self._rightMove = False
        self._rangeMove = False
        self.repaint()

    def _setLeftValue(self, value, emit=True):
        """
        设置左滑块坐标值
        :param value:
        :return:
        """
        self._left_value = value
        if emit:
            self.leftValueChanged.emit(self.getLeftValue())

    def _setRightValue(self, value, emit=True):
        """
        设置右滑块坐标值
        :param value:
        :return:
        """
        self._right_value = value
        if emit:
            self.rightValueChanged.emit(self.getRightValue())

    def updateEdge(self):
        """
        处理鼠标边缘
        :return:
        """
        pos = self._curPos
        if pos is None:
            return 0.2
        x = pos.x()
        ps, pe, lp, rp, pLen, Length, minV, maxV = self._getWidgetInfo()
        if ps < x < pe:
            return 0.2
        # print('ps', ps)
        if x < ps and self._leftMove:
            # 左移
            space = int((self.rv - self.lv) / 0.6 * 0.2)  # 每像素代表刻度
            if space == 0:
                space = 5
            distance = space / Length
            deltaA = distance * (ps - x) / (ps + 10)
            deltaT = (ps - x) / (ps + 10)
            print(deltaA)
            self._setLeftValue(self._left_value - deltaA)
            return max(0, 1 - deltaT)
        return 0.2
        # print(space / Length)

    def paintEvent(self, e):
        """
        绘制事件
        :param e:
        :return:
        """
        # self.updateEdge()
        qp = QPainter()
        qp.begin(self)
        self.drawWidget(qp)
        qp.end()

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        self._size = self.size()
        ps, pe, lp, rp, pLen, Length, minV, maxV = self._getWidgetInfo()
        self._right_value = rp
        self._left_value = lp
        super(GrayAdjustWidget, self).resizeEvent(a0)

    def drawWidget(self, qp):
        """
        绘图逻辑
        :param qp:
        :return:
        """
        self._size = self.size()
        delta = self._sliderWidth
        size = self._size
        w = size.width()
        h = size.height()
        if self._left_value is None:
            # 首次渲染进行初始化滑块位置
            self._left_value = self._sliderWidth
        if self._right_value is None:
            # 首次渲染进行初始化滑块位置
            self._right_value = w - self._sliderWidth - 2 * self._left_slider_border

        # 画组件主区域
        pen = QPen(self._style.border, 1,
                   Qt.SolidLine)
        qp.setPen(pen)
        qp.setBrush(self._style.background)
        qp.drawRect(delta, 0, w - 2 * delta - 2, h - 1)

        # 画直方图
        if self._hist is not None:
            histPath = None
            qp.setPen(QPen(self._style.hist, 1, Qt.SolidLine))
            qp.setBrush(self._style.hist)

            hLen = len(self._hist)
            ps, pe, lp, rp, pLen, Length, minV, maxV = self._getWidgetInfo()
            p1 = max(0, minV)
            pointNum = Length  # 绘制500个点
            # print(pointNum)
            p2 = min(hLen - 1, maxV)
            p1p = ((p1 - minV) / (maxV - minV)) * Length + ps
            p2p = ((p2 - minV) / (maxV - minV)) * Length + ps
            histLen = p2p - p1p
            step = (p2 - p1) / pointNum
            n = int((p2 - p1) / step)
            histX = np.arange(0, hLen)
            histY = self._hist[:, 0].copy()
            x1 = np.arange(p1, p2, step)
            # 线性插值
            y1 = np.interp(x1, histX, histY)
            thre = np.sort(y1)[-30:].mean()
            # print(thre, y1.max())
            y1[y1 > thre] = thre

            for i in range(n):
                x = (i / n) * histLen + p1p
                p = y1[i]
                y = float(((y1.max() - p) / (y1.max() - y1.min())) * (size.height() - 2))
                if i == 0:
                    histPath = QPainterPath(QPointF(x, (size.height() - 2)))
                    histPath.lineTo(QPointF(x, y))
                else:
                    histPath.lineTo(QPointF(x, y))
            histPath.lineTo(QPointF(x, (size.height() - 2)))
            qp.drawPath(histPath)
            qp.fillPath(histPath, qp.brush())

        # 画左滑块
        left_pos = int(max(self._left_value, self._sliderWidth))
        left_pos = int(min(left_pos, self._size.width() - self._sliderWidth - self._right_slider_border))
        qp.setPen(QPen(self._style.leftSliderBorder, self._left_slider_border, Qt.SolidLine))
        qp.setBrush(self._style.leftBrush)
        points = QPolygon([
            QPoint(left_pos, h),
            QPoint(left_pos, h - delta * 2),
            QPoint(left_pos - delta, h - delta),
            QPoint(left_pos - delta, h)
        ])
        qp.drawPolygon(points)
        # 画右滑块
        right_pos = int(min(self._right_value, self._size.width() - self._sliderWidth - self._right_slider_border))
        right_pos = int(max(right_pos, self._sliderWidth))
        qp.setPen(QPen(self._style.rightSliderBorder, self._right_slider_border, Qt.SolidLine))
        qp.setBrush(self._style.rightBrush)
        points = QPolygon([
            QPoint(right_pos, h),
            QPoint(right_pos, h - delta * 2),
            QPoint(right_pos + delta, h - delta),
            QPoint(right_pos + delta, h)
        ])
        qp.drawPolygon(points)
        # 画滑块间的遮罩
        qp.setPen(QPen(self._style.maskBorder, self._left_slider_border, Qt.SolidLine))
        qp.setBrush(self._style.maskBrush)
        points = QPolygon([
            # QPoint(self._left_value, h - delta * 2),
            QPoint(left_pos, 0),
            QPoint(left_pos, h),
            QPoint(right_pos, h),
            # QPoint(self._right_value, h - delta * 2),
            QPoint(right_pos, 0)
        ])
        qp.drawPolygon(points)
        qp.setPen(QPen(self._style.leftLine, self._left_slider_border, Qt.SolidLine))
        qp.drawLine(left_pos, 0, left_pos, h - 2 * delta)
        qp.setPen(QPen(self._style.rightLine, self._right_slider_border, Qt.SolidLine))
        qp.drawLine(right_pos, 0, right_pos, h - 2 * delta)
