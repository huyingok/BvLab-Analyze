# -*- coding: utf-8 -*-

class InitializeInfo:
    def __init__(self):
        self.eye_right = None
        self.eye_left = None
        self.eye_mid = None
        self.img3d = None
        self.dataImporter_index = False  # 定义一个vtkImageImportObject
        self.image_file_list = []  # 图文件列表
        self.image_filename_list = []  # 图名称列表
        self.image_file_list_index = 0  # 对应序号
        self.swc_file_list = []  # swc文件列表
        self.points_withdraw_info = []  # 点撤回信息
        self.points_list = []  # Point/球列表
        self.points_mark_list = []  # 标记列表
        self.Position = []  # 坐标列表
        self.box_size = []  # 区域尺寸
        self.box_volume = None  # 感兴趣区域
        self.boxWidget = None
        self.z_step = 0  # 间隔
        self.z_redun = 0  # 冗余
        self.z_up_over = True  # 1：上切结束，不能继续上切
        self.z_down_over = True  # 1：下切结束，不能继续下切
        self.x_up_over = True
        self.x_down_over = True
        self.y_up_over = True
        self.y_down_over = True
        self.points_count = 0  # 球体个数
        self.sphere_hide = False  # 1：球体隐藏
        self.origin = [0, 0, 0]  # 初始位置
        self.point_line_count = True  # 点击生成线，1为激活状态
        self.one_sphere_count = False  # 单击生成点，0为未触发
        self.function_index = 1  # 功能切换
        self.fun_label = 'Max mode'
        self.render_mode_text = 'MIP'
        self.mark_render_text = 'Single-color'
        self.mark_color = [0, 1, 0]
        self.mark_select_color = [1, 0, 0]
        self.index = -1  # 选择的actorTag
        self.color_dialog_index = False  # 是否生成颜色对话框
        self.mark_color_dialog_index = False
        self.range_min = 0  # 区间最小值
        self.range_max = 0  # 区间最大值
        self.gray_min = 0  # 初始灰度值
        self.gray_max = 1

        self.isReload = True  # 图像是否重新加载或切换
        # MIP模式灰度调节参数
        self.MIPParam = {
            'minGray': self.gray_min,  # 界面上的灰度直方图最小值
            'maxGray': self.gray_max,  # 界面上的灰度直方图最大值
            # 用于多次增强或减弱
            'minGray2': self.gray_min,
            'maxGray2': self.gray_max
        }
        self.min_auto_gray = 0  # 初始灰度值
        self.max_auto_gray = 1

        self.xyz_cut_index = False  # 步长冗余变化，避免图像重复生成
        self.gray_inherit_index = True
        self.sr_inherit_index = False
        self.x_px = 1
        self.y_px = 1
        self.z_px = 1
        self.xyz_size_copy = [1, 1, 1]
        self.selection = False  # 图像选择框是否执行切换
        self.sphere_size = 7  # 球体半径尺寸
        self.select_sphere_whether = False  # 是否选择球体
        self.is_marking_button = False  # 是否是修订模式
        self.reOpen = True  # 重新打开文件

        self.image_path_list = []  # 图像路径列表

        self.filter_res = {}  # 筛选结果列表

    '''更新初始化信息'''
    def update_initialize_info(self):
        # self.is_marking_button = False
        self.points_withdraw_info = []  # 点撤回信息
        self.points_list = []  # Point/球列表
        self.points_mark_list = []  # 标记列表
        self.Position = []  # 坐标列表
        self.points_count = 0  # 球体数
        self.sphere_hide = False  # 球体显示和隐藏
        self.z_up_over = True  # 上切
        self.z_down_over = True  # 下切
        self.x_up_over = True
        self.x_down_over = True
        self.y_up_over = True
        self.y_down_over = True
        self.selection = False  # 图像选择框是否执行切换
        self.xyz_cut_index = False
        self.point_line_count = True  # 点击生成线，1为激活状态
        self.one_sphere_count = False  # 单击生成点，0为未触发
        self.select_sphere_whether = False  # 是否选择球体
        self.origin = [0, 0, 0]  # 区域初始坐标
        self.box_size = []  # 区域尺寸
        self.box_volume = None  # 感兴趣区域
        self.boxWidget = None


# from PyQt5.QtWidgets import QApplication, QListWidget, QListWidgetItem, QVBoxLayout, QWidget
# from PyQt5.QtCore import Qt
# from PyQt5.QtGui import QMouseEvent
#
#
# class MyListWidget(QListWidget):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.parent = parent
#
#     def mousePressEvent(self, event: QMouseEvent):
#         super().mousePressEvent(event)  # 调用基类的事件处理
#         item = self.itemAt(event.pos())  # 获取被点击的项
#         if item:
#             self.parent.handle_mouse_event(item, event)  # 将事件传递给父窗口
#
#
# class MyWindow(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.list_widget = MyListWidget(self)  # 将父窗口传递给MyListWidget
#         self.layout = QVBoxLayout()
#         self.layout.addWidget(self.list_widget)
#         self.setLayout(self.layout)
#
#         # 添加一些项
#         for i in range(5):
#             item = QListWidgetItem(f"Item {i}")
#             self.list_widget.addItem(item)
#
#         self.list_widget.currentItemChanged.connect(self.on_item_changed)
#
#     def on_item_changed(self):
#         print("on_item_changed")
#
#     def handle_mouse_event(self, item, event: QMouseEvent):
#         if event.button() == Qt.LeftButton:
#             print(f"Left button clicked on item: {item.text()}")
#         elif event.button() == Qt.MiddleButton:
#             print(f"Middle button clicked on item: {item.text()}")
#         elif event.button() == Qt.RightButton:
#             print(f"Right button clicked on item: {item.text()}")
#
#
# if __name__ == "__main__":
#     app = QApplication([])
#     window = MyWindow()
#     window.show()
#     app.exec_()
