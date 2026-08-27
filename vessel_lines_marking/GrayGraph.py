# -*- coding: utf-8 -*-
import pyqtgraph as pg


class GrayHistogram:
    def __init__(self, dmix, gray_min, gray_max, graphWidget, bins, hist):
        self.dmix = dmix
        self.gray_max = gray_max
        self.gray_min = gray_min
        self.graphWidget = graphWidget
        self.bins = bins
        self.hist = hist

    '''创建直方图'''
    def create_gray_histogram(self):
        bins_x = self.bins[:-1][self.dmix[0]:self.dmix[1]]  # 灰度直方图x
        hist_y = self.hist[self.dmix[0]:self.dmix[1]]  # 灰度直方图y
        plotitem = self.graphWidget.addPlot()  # 添加直方图
        # 隐藏坐标系
        # plotitem.hideAxis('bottom')  # 隐藏底部坐标轴
        plotitem.hideAxis('left')  # 隐藏左侧坐标轴
        plotitem.plot(bins_x, hist_y,
                      pen=pg.mkPen(color=(0, 255, 0), width=1),
                      fillLevel=0, brush=(0, 255, 0))
        # 添加边界线
        lineItem = pg.LinearRegionItem([self.gray_min, self.gray_max],
                                       pen=pg.mkPen(color=(50, 255, 255), width=5))
        lineItem.setBrush(136, 136, 136, 80)
        lineItem.setZValue(10)
        plotitem.setMouseEnabled(x=False, y=False)
        plotitem.addItem(lineItem)
        return plotitem, lineItem

    '''Histogramx轴活动槽函数'''
    @classmethod
    def plotitem_x_change(cls, dmix, bins, hist, plotitem, lineItem):
        plotitem.clear()
        bins_x = bins[:-1][dmix[0]:dmix[1]]
        hist_y = hist[dmix[0]:dmix[1]]
        plotitem.plot(bins_x, hist_y,
                      pen=pg.mkPen(color=(0, 255, 0), width=1),
                      fillLevel=0, brush=(0, 255, 0))
        plotitem.addItem(lineItem)

    '''边界线值域'''
    @classmethod
    def bounds_of_lineitem(cls, self):
        bounds = self.lineItem.getRegion()
        print('Boundary line:', bounds[0], bounds[1])
        D = int((self.range_max - self.range_min) / 2)
        if not (bounds[0] < self.range_min or bounds[1] > self.range_max + D):
            Min = round(bounds[0])
            Max = round(bounds[1])
            self.gray_min_slider.setValue(Min)  # 修改滑块
            self.gray_max_slider.setValue(Max)  # 修改滑块

    '''设置直方图边界'''
    @classmethod
    def set_histogram_bounds(cls, self):
        int_gray_min = int(self.gray_min)
        int_gray_max = int(self.gray_max)
        gray_redun = int((self.gray_max - self.gray_min) / 2)
        if int_gray_min < self.range_min + gray_redun:
            dminn, dmaxn = self.range_min, int_gray_max + gray_redun
        else:
            dminn, dmaxn = int_gray_min - gray_redun, int_gray_max + gray_redun
        if int_gray_max > self.range_max - gray_redun:
            dmin, dmax = dminn, self.range_max
        else:
            dmin, dmax = dminn, dmaxn
        self.dmix = [dmin - self.range_min, dmax - self.range_min]

