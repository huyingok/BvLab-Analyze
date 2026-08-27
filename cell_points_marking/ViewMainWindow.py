# import numpy as np
# from PyQt5.QtGui import QPainter, QColor
# from PyQt5.QtCore import Qt
# from PyQt5 import QtWidgets
# from PyQt5.QtWidgets import QMessageBox
# from CreateImageData import CreateImageData
# from CreateDialog import CreateDialog
# from SetStepRedun import SetStepRedun
# from ReadPath import ReadPath
# from SetConnect import SetConnect
# from MarkingPointsFun import MarkingPointsFun
# from BoxWidgets import BoxWidgets
# from MarkSphereProcess import MarkSphereProcess
# from GrayAdjustWidget import GrayAdjustWidget
# from CustomMainWindow import CustomMainWindow
# from InitializeInfo import InitializeInfo
# # 样式
# messagebox_style = """
#     QMessageBox { background-color: rgb(25, 35, 45); font: 9pt \"微软雅黑\"; color: rgb(255, 255, 255);}
#     QLabel {color: rgb(255, 255, 255); font: 9pt \"微软雅黑\";}
#     QPushButton { background-color: rgb(70, 80, 100); font: 9pt \"微软雅黑\"; color: rgb(255, 255, 255);
#     border: 1px solid rgb(70, 80, 100); padding: 3px 3px; min-width: 80px; height: 15px;}
#     QPushButton:hover { background-color: #379eff; border: 1px solid #379eff;}
# """
# menu_style = """
#     QMenu { color: rgb(255, 255, 255); background-color: rgb(70,80,100);}
#     QMenu::item:selected { background-color: #379eff;}
#     QMenu::Separator { background-color: rgb(90, 100, 120); height: 1px; }
# """
# fun_list_style = """
# QListWidget {font: 9pt "微软雅黑"; background-color: rgb(25, 35, 45); border-width: 1px;
# border-style: solid; border-color: rgb(70, 80, 100); border-radius: 0px}
# """
# fun_widget_style = """
# QWidget {background-color: rgb(25, 35, 45)}
# """
#
#
# class MainWindow(CustomMainWindow, CreateDialog):
#     '''界面可视化'''
#     def __init__(self, parent=None):
#         super(MainWindow, self).__init__(parent)
#         # self.setupUi_win(self)
#         self.CreateImageData = CreateImageData  # 创建三维可视图
#         self.SetStepRedun = SetStepRedun  # 三维可视图设置步长和冗余
#         self.ReadPath = ReadPath  # 读取文件
#         self.SetConnect = SetConnect  # 信号和快捷方式
#         self.MarkingPointsFun = MarkingPointsFun  # 标记功能，Point/球的生成
#         self.BoxWidgets = BoxWidgets  # box框
#         self.MarkSphereProcess = MarkSphereProcess  # 对生成的球体操作
#
#         self.center()
#         self.setContentsMargins(2, 1, 2, 2)
#         self.setStyleSheet(menu_style)
#         # 取消右键菜单
#         self.setContextMenuPolicy(Qt.NoContextMenu)
#         # 自定义菜单栏
#         self.SetConnect(self.title_bar).set_title_menubar_connect(self)
#         # 框架函数
#         self.action_widgets()  # 创建浮动窗口
#         self.SetConnect.img_list_connect(self)
#         # 路径选择
#         self.create_path_select_dialog()
#         # SetConnect_copy(self).select_path_connect()
#         self.SetConnect.select_path_connect(self)
#         # 工具栏操作
#         self.SetConnect.tool_bar_action(self)
#         # 浮动窗口位置
#         self.addDockWidget(Qt.LeftDockWidgetArea, self.images_dockWidget)  # 浮动窗口靠左
#         self.addDockWidget(Qt.LeftDockWidgetArea, self.fun_dockWidget)  # 浮动窗口靠左
#         # 设置不可关闭和不可折叠
#         self.images_dockWidget.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
#         self.fun_dockWidget.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
#         # 设置大小化
#         self.images_dockWidget.setFeatures(QtWidgets.QDockWidget.DockWidgetFloatable)
#         self.fun_dockWidget.setFeatures(QtWidgets.QDockWidget.DockWidgetFloatable)
#
#         # 初始化vtk视图区域
#         self.create_vtk_view()
#         # 设置坐标系函数
#         self.AxesWidgt()
#         self.images_dockWidget.installEventFilter(self)
#         self.fun_dockWidget.installEventFilter(self)
#         self.centralwidget.installEventFilter(self)
#         self.tool_bar.installEventFilter(self)
#         self.myStatus.installEventFilter(self)
#         self.fun_dockWidget.setMinimumWidth(435)
#         self.fun_dockWidget.setMaximumWidth(440)
#
#         # self.frame.setVisible(False)
#
#         # layout = QtWidgets.QVBoxLayout(self, spacing=0)
#         # layout.setContentsMargins(0, 0, 0, 0)
#
#     '''界面居中'''
#     def center(self):
#         qr = self.frameGeometry()
#         cp = QtWidgets.QDesktopWidget().availableGeometry().center()
#         qr.moveCenter(cp)
#         self.move(qr.topLeft())
#
#     '''布局修饰'''
#     def paintEvent(self, event):
#         painter = QPainter(self)
#         painter.setBrush(QColor(70, 80, 100, 255))  # 设置矩形颜色和透明度
#         painter.drawRect(-50, -50, self.width() + 50, self.height() + 50)  # 绘制矩形
#         # painter.setOpacity(0.05)
#         # pixmap = QPixmap(r".\icon_image\vstar.png")
#         # painter.drawPixmap(self.rect(), pixmap)
#
#     '''清除界面'''
#     def clear_view(self):
#         if self.dataImporter_index:  # 存在可标签数据,Update
#             box = QtWidgets.QMessageBox()
#             box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
#             box.setStyleSheet(messagebox_style)
#             box.setText('是否清空界面？')
#             box.setWindowTitle('Prompt')
#             box.setIcon(QtWidgets.QMessageBox.Question)
#             box.setWindowModality(Qt.ApplicationModal)
#             if box.exec_() == QtWidgets.QMessageBox.Yes:
#                 self.image_lineEdit.clear()
#                 self.swc_lineEdit.clear()
#                 self.fun_list = QtWidgets.QListWidget()  # 设置列表项，展示标注
#                 self.fun_list.setStyleSheet(self.qdarkstyle_sheet + fun_list_style)
#                 self.fun_widget = QtWidgets.QWidget()  # 设置一个子窗口，添加更多控件
#                 self.fun_widget.setStyleSheet(fun_widget_style)
#                 fun_vbox = QtWidgets.QVBoxLayout(self.fun_widget)  # 垂直布局
#                 fun_vbox.addWidget(self.fun_list)  # 添加控件
#
#                 self.fun_stacked_widget.removeWidget(self.widgetFun)  # 更新操作栏
#                 self.layer_stacked_widget.removeWidget(self.layer_cut_widget)
#
#                 self.widgetFun.deleteLater()
#                 self.layer_cut_widget.deleteLater()
#                 self.fun_stacked_widget.addWidget(self.fun_widget)
#
#                 if self.boxWidget:
#                     self.boxWidget.Off()
#                     del self.boxWidget
#
#                 InitializeInfo.__init__(self)
#                 self.img_list.clear()
#                 self.ren.RemoveAllViewProps()  # 去除所有渲染对象
#                 self.renWin.Render()
#                 self.dataImporter_index = 0
#
#                 self.fun_dockWidget.setMinimumWidth(435)
#
#     '''设置相机朝上方向'''
#     def set_camera_view_up_y(self):
#         if self.dataImporter_index:
#             dz = self.max_shape * 4
#             d_near, d_far = self.camera.GetClippingRange()
#             d = d_far - d_near
#             d_min = self.max_shape
#             if d_near >= dz - d_min or dz + d_min >= d_far:
#                 dz = d_near + d / 2
#             else:
#                 self.camera.SetClippingRange(dz - d_min, dz + d_min)
#                 # print("Update")
#             self.camera.SetViewUp(0, 1, 0)
#             self.camera.SetPosition(self.img3d.shape[2] / 2, self.img3d.shape[1] / 2, dz)
#             self.camera.SetFocalPoint(self.img3d.shape[2] / 2, self.img3d.shape[1] / 2, self.img3d.shape[0] / 2)
#             # print(self.camera.GetFocalPoint())
#             # print(self.camera.GetPosition())
#             self.renWin.Render()
#
#     def set_camera_view_up_x(self):
#         if self.dataImporter_index:
#             dy = self.max_shape * 4
#             d_near, d_far = self.camera.GetClippingRange()
#             d = d_far - d_near
#             d_min = self.max_shape
#             if d_near >= dy - d_min or dy + d_min >= d_far:
#                 dy = d_near + d / 2
#             else:
#                 self.camera.SetClippingRange(dy - d_min, dy + d_min)
#             self.camera.SetViewUp(1, 0, 0)
#             self.camera.SetPosition(self.img3d.shape[2] / 2, dy, self.img3d.shape[0] / 2)
#             self.camera.SetFocalPoint(self.img3d.shape[2] / 2, self.img3d.shape[1] / 2, self.img3d.shape[0] / 2)
#             self.renWin.Render()
#
#     def set_camera_view_up_z(self):
#         if self.dataImporter_index:
#             dx = self.max_shape * 4
#             d_near, d_far = self.camera.GetClippingRange()
#             d = d_far - d_near
#             d_min = self.max_shape
#             if d_near >= dx - d_min or dx + d_min >= d_far:
#                 dx = d_near + d / 2
#             else:
#                 self.camera.SetClippingRange(dx - d_min, dx + d_min)
#             self.camera.SetPosition(dx, self.img3d.shape[1] / 2, self.img3d.shape[0] / 2)
#             self.camera.SetFocalPoint(self.img3d.shape[2] / 2, self.img3d.shape[1] / 2, self.img3d.shape[0] / 2)
#             self.camera.SetViewUp(0, 0, 1)
#             self.renWin.Render()
#
#     '''删除多余的点线'''
#
#     def judge_actor(self):
#         try:
#             self.ren.RemoveActor(self.one_point_actor)  # 删除点
#             self.ren.RemoveActor(self.one_line_actor)  # 删除直线
#         except AttributeError:
#             print("没有删除的对象")
#
#     '''切换模式'''
#
#     def change_function(self):
#         self.MarkingPointsFun.change_mark_function(self)
#
#     '''创建球体'''
#
#     def create_sphere(self, pos):
#         self.MarkSphereProcess.create_sphere2(pos, self)
#
#     '''直线和图像数据的交点集'''
#
#     def collect_points(self, pick_pos, camera_pos):
#         points, count = self.MarkingPointsFun.collect_points2(pick_pos, camera_pos, self)
#         return points, count
#
#     '''Max值定点'''
#
#     # def Max_onRightButtonPressEvent(self, obj, event):
#     #     self.MarkingPointsFun.Max_onRightButtonPressEvent(self)
#
#     '''vtk鼠标右击事件'''
#
#     # def onRightButtonPressEvent(self, obj, event):
#     #     self.MarkingPointsFun.onRightButtonPressEvent(self)
#
#     '''标点模式选择'''
#     def onSelectMarkingFun(self):
#         if self.function_index == 0:
#             self.MarkingPointsFun.onLeftButtonPressEvent(self)
#         else:
#             self.MarkingPointsFun.Max_onLeftButtonPressEvent(self)
#
#     '''读取文件路径'''
#
#     def read_file_path(self):
#         self.ReadPath.read_file_path2(self)
#
#     '''上下切换图'''
#
#     def on_item_selection(self, now_item, pre_item):
#         # self.inherit_pos = self.camera.GetPosition()
#         # self.inherit_focal = self.camera.GetFocalPoint()
#         # self.inherit_range = self.camera.GetClippingRange()
#         if self.selection:
#             # print("选中列表单元")
#             print("当前路径为:", self.swc_file_list[self.img_list.row(now_item)])
#             if pre_item:  # 获取之前选择项的序号
#                 self.image_file_list_index = self.img_list.row(pre_item)
#
#             save_swc_path = self.swc_file_list[self.image_file_list_index]
#             self.save_txt_swc(save_swc_path)
#
#             self.image_file_list_index = self.img_list.currentRow()
#             self.Visualization_tif_file()
#
#     '''保存坐标为txt'''
#
#     def save_position(self):
#         self.ReadPath.save_sphere_pos(self)
#
#     '''Saveswc和txt'''
#
#     def save_txt_swc(self, save_swc_path):
#         self.ReadPath.save_pos_swc(save_swc_path, self)
#
#     '''VisualizationtifFile'''
#
#     def Visualization_tif_file(self):
#         self.tif_to_view(self)
#
#     '''DataImporter_create,tifRendering'''
#
#     def DataImporter_create(self):
#         image_info = [self.img3d, self.x_px, self.y_px, self.z_px]
#         # 对类进行实例化，并调用方法（Function）
#         (self.dataImporter, self.volume, self.opacityTransferFunction,
#          self.colorTransferFunction) = self.CreateImageData(image_info).create_img3d()
#
#     '''Outline_create,包围盒'''
#
#     def Outline_create(self):
#         (self.dataImporter_outline, self.outline,
#          self.outlineActor) = self.CreateImageData.create_outline(self.dataImporter)
#
#     '''选择区域显示隐藏'''
#
#     def outline_show_hide(self):
#         if self.dataImporter_index and self.box_volume:
#             if self.outline_show_hide_count:
#                 self.box_volume.VisibilityOff()
#                 self.box_outlineActor.VisibilityOff()
#                 self.outline_show_hide_count = 0
#             else:
#                 self.box_volume.VisibilityOn()
#                 self.box_outlineActor.VisibilityOn()
#                 self.outline_show_hide_count = 1
#             self.renWin.Render()
#
#     '''图文件显示隐藏'''
#
#     def data_volume_show_hide(self):
#         if self.dataImporter_index:
#             if self.data_volume_show_hide_count:  # 图文件显示
#                 self.volume.VisibilityOff()
#                 self.data_volume_show_hide_count = 0
#             else:
#                 self.volume.VisibilityOn()
#                 self.data_volume_show_hide_count = 1
#             self.renWin.Render()
#
#     '''Modifystep'''
#
#     def change_z_step(self):
#         self.z_step = self.SetStepRedun.set_step(self.z_step, self.z_redun,
#                                                  self.z_step_num, self.z_redun_num)
#         self.change_z_step_redun()
#         self.get_z_layers()
#
#     '''Modifyredun'''
#
#     def change_z_redun(self):
#         self.z_redun = self.SetStepRedun.set_redun(self.z_step, self.z_redun,
#                                                    self.z_step_num, self.z_redun_num, self.img3d.shape[0])
#         self.change_z_step_redun()
#         self.get_z_layers()
#
#     '''获取z轴切块层数'''
#
#     def get_z_layers(self):
#         self.z_num = self.SetStepRedun.get_layers(self.z_step, self.z_redun, self.img3d.shape[0], 2)
#         if self.z_list:
#             self.update_z_list()
#
#     '''修改步长和冗余'''
#
#     def change_z_step_redun(self):
#         (self.box_size, self.origin, self.z_up_over) = self.SetStepRedun.update_box_size(
#             [self.x_step, self.y_step, self.z_step], self.img3d.shape[0], self.origin, 2)
#         if not self.xyz_cut_index:
#             self.create_new_image()
#
#     '''包围盒上移'''
#
#     def box_up(self):
#         # print("上移")
#         # print("self.z_down_over:", self.z_down_over)
#         # print("self.z_up_over:", self.z_up_over)
#         if self.dataImporter_index and self.box_volume:
#             if not self.z_up_over:
#                 (self.box_size, self.origin, self.z_up_over, self.z_down_over) = self.SetStepRedun.box_to_up(
#                     self.z_step, self.z_redun, self.box_size, self.origin, self.img3d.shape[0], self.z_up_over, 2)
#                 self.create_new_image()
#                 save_swc_path = self.swc_file_list[self.img_list.currentRow()]  # 保存标记
#                 self.save_txt_swc(save_swc_path)
#
#     '''包围盒下移'''
#
#     def box_down(self):
#         # print("下移")
#         # print("self.z_down_over:", self.z_down_over)
#         # print("self.z_up_over:", self.z_up_over)
#         if self.dataImporter_index and self.box_volume:
#             if not self.z_down_over:
#                 (self.box_size, self.origin, self.z_up_over, self.z_down_over) = self.SetStepRedun.box_to_down(
#                     self.z_step, self.z_redun, self.box_size, self.origin, self.z_down_over, 2)
#                 self.create_new_image()
#                 save_swc_path = self.swc_file_list[self.img_list.currentRow()]  # 保存标记
#                 self.save_txt_swc(save_swc_path)
#
#     '''感兴趣区域更新'''
#
#     def create_new_image(self):
#         self.CreateImageData.create_box_image(self)
#
#     '''功能框'''
#
#     def object_message(self):
#         self.CreateImageData.update_color_alpha(self.opacityTransferFunction, self.colorTransferFunction,
#                                                 self.gray_min, self.gray_max, self.r, self.g, self.b)
#
#         hist = self.graywidget_message()
#
#         self.fun_widgets()
#
#         self.grayWidget = GrayAdjustWidget(self.graphframe)
#         self.grayWidget.leftValueChanged.connect(self.grayTransCallback)  # 设置灰度图槽函数
#         # self.grayWidget.leftValueChanged.connect(self.change_color_opacity)
#         self.grayWidget.rightValueChanged.connect(self.grayTransCallback)
#         # self.grayWidget.rightValueChanged.connect(self.change_color_opacity)
#         self.grayWidget.setLeftValue(self.gray_min)
#         self.grayWidget.setRightValue(self.gray_max)
#         # self.txtMinGray.setValue(self.gray_max)
#         # self.txtMaxGray.setValue(self.gray_max)
#         self.grayWidget.setHist(hist)
#         self.graphWidget_form.addRow(self.gray_label, self.grayWidget)
#
#         self.xyz_dialog()  # 修改间隔浮框函数
#
#         self.xyz_cut_dialog()
#
#         self.layer_stacked_widget.addWidget(self.layer_cut_widget)
#
#         self.SetConnect.fun_widget_info_connect(self)
#
#         self.fun_stacked_widget.addWidget(self.widgetFun)
#
#     '''修改间隔浮框'''
#
#     def xyz_dialog(self):
#         sure_button = self.create_xyz_px_dialog()
#         self.x_px_spinbox.setValue(self.x_px)  # 初始值
#         self.y_px_spinbox.setValue(self.y_px)
#         self.z_px_spinbox.setValue(self.z_px)
#         self.SetConnect.xyz_px_dialog_connect(sure_button, self)
#         self.xyz_px_dialog.hide()  # 隐藏窗口
#
#     '''读取txtFile,读取选择文件的txtFile,渲染标签球体'''
#
#     def read_txt_file(self, index):
#         self.ReadPath.read_txt_file2(index, self)
#
#     '''球体快捷显示'''
#
#     def Sphere_quick_key(self):
#         self.MarkSphereProcess.sphere_show_hide(self)
#
#     '''open file,选择文件路径,路径选择框,是否保存当前文件的txt和swc'''
#
#     def select_file_path_show(self):
#         self.select_path_dialog.show()
#
#     '''浮点数调节框修改间距'''
#
#     def dialog_xyz_size_show(self):
#         self.xyz_px_dialog.show()  # 显示浮动框
#
#     '''路径选择提示,路径选择框,取消按钮'''
#
#     def Warning_Select(self):
#         self.select_path_dialog.hide()
#
#     '''路径选择框确定按钮,image路径问题,获取文件名列表,路径列表'''
#
#     def read_path_question(self):
#         self.ReadPath.read_path_question2(self)
#
#     '''imagePath,SelecttifPath'''
#
#     def image_path_open(self):
#         # 获取文件路径
#         self.image_path, self.swc_path = self.ReadPath.image_path_open2(self.image_lineEdit, self.swc_lineEdit)
#
#     '''swcPath,SelectswcPath'''
#
#     def swc_path_open(self):
#         # 获取文件夹路径
#         self.swc_path = self.ReadPath.dir_path_open(self.swc_lineEdit)
#
#     '''点击确定修改间距'''
#
#     def xyz_sure_size(self):
#         self.CreateImageData.change_image_for_px(self)
#
#     '''图列表浮动窗口显示和隐藏'''
#
#     def images_list_widget(self):
#         if self.images_dockWidget.isHidden():
#             self.images_dockWidget.show()
#         else:
#             self.images_dockWidget.hide()
#
#     '''功能浮动窗口显示和隐藏'''
#
#     def funs_widget_show_hide(self):
#         if self.fun_dockWidget.isHidden():
#             self.fun_dockWidget.show()
#         else:
#             self.fun_dockWidget.hide()
#
#     '''最小灰度值'''
#
#     def textprint_gmin(self):
#         # 控制最小灰度
#         self.gray_min = self.gray_min_num.value()
#         self.grayWidget.setLeftValue(self.gray_min)
#
#     '''最大灰度值'''
#
#     def textprint_gmax(self):
#         # 控制最大灰度
#         self.gray_max = self.gray_max_num.value()
#         self.grayWidget.setRightValue(self.gray_max)
#
#     '''固定灰度'''
#
#     def sure_gray_auto(self):
#         if self.dataImporter_index == 1:
#             self.min_auto_gray = self.gray_min
#             self.max_auto_gray = self.gray_max
#             self.gray_auto_change_button.setText(f'{self.min_auto_gray} / {self.max_auto_gray}')
#
#     '''自动更新灰度'''
#
#     def gray_auto_change(self):
#         self.CreateImageData.gray_change(3, self)
#
#     '''Color dialog'''
#
#     def openColorDialog(self):
#         self.create_color_dialog(self)
#         # CreateDialog.create_color_dialog(self)
#
#     '''修改颜色和透明度'''
#
#     def change_color_opacity(self):
#         self.CreateImageData.update_color_alpha(self.opacityTransferFunction, self.colorTransferFunction,
#                                            self.gray_min, self.gray_max, self.r, self.g, self.b)
#         # print("当前最小灰度:", self.gray_min)
#         # print("当前最大灰度:", self.gray_max)
#         if self.box_volume:
#             self.CreateImageData.update_color_alpha(self.alphaChannelFunc, self.colorFunc,
#                                                self.gray_min, self.gray_max, self.r, self.g, self.b)
#         self.renWin.Render()
#
#     '''选择对象'''
#
#     def select_actor(self, obj, event):
#         if self.dataImporter_index:
#             self.MarkSphereProcess.sphere_select_and_mark(self, 0)
#
#     '''标记对象'''
#
#     def mark_actor(self):
#         if self.dataImporter_index:
#             self.MarkSphereProcess.sphere_mark(self)
#
#     '''球体删除'''
#
#     def Actor_Delete(self):
#         if self.dataImporter_index:
#             self.MarkSphereProcess.delete_sphere(self)
#
#     """清空标记"""
#
#     def all_actor_delete(self):
#         if self.dataImporter_index:
#             self.MarkSphereProcess.delete_all_sphere(self)
#
#     '''点击撤回'''
#
#     def Point_Withdraw(self):
#         if self.dataImporter_index:
#             self.MarkSphereProcess.shpere_withdraw(self)
#
#     '''Focus'''
#
#     def camera_focal(self):
#         if self.dataImporter_index and len(self.points_list):
#             pos = self.Position[self.index]
#             self.camera.SetFocalPoint(pos[0] * self.x_px, pos[1] * self.y_px, pos[2] * self.z_px)
#             self.renWin.Render()
#
#     '''球体半径'''
#
#     def change_sphere_size(self):
#         self.sphere_size = self.sphere_size_slider.value()
#         self.sphere_size_label.setText(f'球半径 = {self.sphere_size}')
#         self.update_sphere_size_position()
#
#     '''刷新球体'''
#
#     def update_sphere_size_position(self):
#         self.MarkSphereProcess.update_sphere_position(self)
#
#     '''灰度减少'''
#
#     def gray_left(self):
#         self.CreateImageData.gray_change(1, self)
#
#     '''灰度增加'''
#
#     def gray_right(self):
#         self.CreateImageData.gray_change(2, self)
#
#     '''图像切换'''
#
#     def item_up(self):
#         if self.dataImporter_index:
#             if self.img_list.currentRow() == 0:
#                 return
#             else:
#                 self.image_file_list_index = self.img_list.currentRow() - 1
#                 self.img_list.setCurrentRow(self.image_file_list_index)
#
#     def item_down(self):
#         if self.dataImporter_index:
#             if self.img_list.currentRow() == self.img_list.count() - 1 or self.img_list.currentRow() == -1:
#                 return
#             else:
#                 self.image_file_list_index = self.img_list.currentRow() + 1
#                 self.img_list.setCurrentRow(self.image_file_list_index)
#
#     '''xyz浮框'''
#
#     def xyz_cut_dialog(self):
#         (recovery_xy_button, recovery_z_button) = (
#             self.create_xyz_cut_dialog(self.x_num, self.y_num, self.z_num))
#
#         self.SetConnect.xyz_widget_connect(recovery_xy_button, recovery_z_button, self)
#
#     '''z轴选择'''
#
#     def change_z(self):
#         if self.z_list.count() > 1:
#             self.z_index = self.z_list.currentRow()
#             dz = self.z_num - self.z_index - 1
#             # print("z轴切换:", dz)
#             z2 = (dz + 1) * self.z_step - dz * self.z_redun
#             # print(self.img3d.shape[0])
#             if z2 > self.img3d.shape[0]:
#                 z2 = self.img3d.shape[0]
#                 # self.z_down_over = 0
#             z1 = z2 - self.z_step
#             self.box_size[-2:] = z1, z2
#             print(self.box_size)
#             self.origin[2] = z1
#             self.create_new_image()
#
#     '''Updatez轴列表'''
#
#     def update_z_list(self):
#         self.update_z_list_widget(self.z_num)
#
#     '''xySelect'''
#
#     def change_xy(self):
#         row = self.table_widget.currentRow()  # 行标
#         column = self.table_widget.currentColumn()  # 列标
#         # row = self.tableview.currentIndex().row()  # 行标
#         # column = self.tableview.currentIndex().column()  # 列标
#         # print("xyToggle[x, y]:", [column, self.y_num - 1 - row])
#         dx = column
#         dy = self.y_num - 1 - row
#         x2 = (dx + 1) * self.x_step - dx * self.x_redun
#         y2 = (dy + 1) * self.y_step - dy * self.y_redun
#         if x2 > self.img3d.shape[2]:
#             x2 = self.img3d.shape[2]
#         if y2 > self.img3d.shape[1]:
#             y2 = self.img3d.shape[1]
#         x1 = x2 - self.x_step
#         y1 = y2 - self.y_step
#         self.box_size[:4] = x1, x2, y1, y2
#         print(self.box_size)
#         self.origin[:2] = x1, y1
#         self.create_new_image()
#
#     '''Updatexy轴列表'''
#
#     def update_xy_table(self):
#         self.update_xy_tableview(self.x_num, self.y_num)
#
#     '''复原xy'''
#
#     def recovery_xy(self):
#         if self.dataImporter_index:
#             self.x_step_num.setValue(int(self.img3d.shape[2]))
#             self.y_step_num.setValue(int(self.img3d.shape[1]))
#
#     '''复原z'''
#
#     def recovery_z(self):
#         if self.dataImporter_index:
#             self.z_step_num.setValue(int(self.img3d.shape[0]))
#
#     '''x轴切块'''
#
#     def change_x_step(self):
#         self.x_step = self.SetStepRedun.set_step(self.x_step, self.x_redun,
#                                                  self.x_step_num, self.x_redun_num)
#         self.change_x_step_redun()
#         self.get_x_layers()
#
#     def change_x_redun(self):
#         self.x_redun = self.SetStepRedun.set_redun(self.x_step, self.x_redun,
#                                                    self.x_step_num, self.x_redun_num, self.img3d.shape[2])
#         self.change_x_step_redun()
#         self.get_x_layers()
#
#     def get_x_layers(self):
#         self.x_num = self.SetStepRedun.get_layers(self.x_step, self.x_redun, self.img3d.shape[2], 0)
#         if self.table_widget:
#             self.update_xy_table()
#
#     def change_x_step_redun(self):
#         (self.box_size, self.origin, self.x_up_over) = self.SetStepRedun.update_box_size(
#             [self.x_step, self.y_step, self.z_step], self.img3d.shape[2], self.origin, 0)
#         if not self.xyz_cut_index:
#             self.create_new_image()
#
#     '''x轴移动'''
#
#     def box_x_to_add(self):
#         if self.dataImporter_index and self.box_volume:
#             if not self.x_up_over:
#                 (self.box_size, self.origin, self.x_up_over, self.x_down_over) = self.SetStepRedun.box_to_up(
#                     self.x_step, self.x_redun, self.box_size, self.origin, self.img3d.shape[2], self.x_up_over, 0)
#                 self.create_new_image()
#                 save_swc_path = self.swc_file_list[self.img_list.currentRow()]  # 保存标记
#                 self.save_txt_swc(save_swc_path)
#
#     def box_x_to_down(self):
#         if self.dataImporter_index and self.box_volume:
#             if not self.x_down_over:
#                 (self.box_size, self.origin, self.x_up_over, self.x_down_over) = self.SetStepRedun.box_to_down(
#                     self.x_step, self.x_redun, self.box_size, self.origin, self.x_down_over, 0)
#                 self.create_new_image()
#                 save_swc_path = self.swc_file_list[self.img_list.currentRow()]  # 保存标记
#                 self.save_txt_swc(save_swc_path)
#
#     '''y轴切块'''
#
#     def change_y_step(self):
#         self.y_step = self.SetStepRedun.set_step(self.y_step, self.y_redun,
#                                                  self.y_step_num, self.y_redun_num)
#         self.change_y_step_redun()
#         self.get_y_layers()
#
#     def change_y_redun(self):
#         self.y_redun = self.SetStepRedun.set_redun(self.y_step, self.y_redun,
#                                                    self.y_step_num, self.y_redun_num, self.img3d.shape[1])
#         self.change_y_step_redun()
#         self.get_y_layers()
#
#     def get_y_layers(self):
#         self.y_num = self.SetStepRedun.get_layers(self.y_step, self.y_redun, self.img3d.shape[1], 1)
#         if self.table_widget:
#             self.update_xy_table()
#
#     def change_y_step_redun(self):
#         (self.box_size, self.origin, self.y_up_over) = self.SetStepRedun.update_box_size(
#             [self.x_step, self.y_step, self.z_step], self.img3d.shape[1], self.origin, 1)
#         if not self.xyz_cut_index:
#             self.create_new_image()
#
#     '''y轴移动'''
#
#     def box_y_to_add(self):
#         if self.dataImporter_index and self.box_volume:
#             if not self.y_up_over:
#                 (self.box_size, self.origin, self.y_up_over, self.y_down_over) = self.SetStepRedun.box_to_up(
#                     self.y_step, self.y_redun, self.box_size, self.origin, self.img3d.shape[1], self.y_up_over, 1)
#                 self.create_new_image()
#                 save_swc_path = self.swc_file_list[self.img_list.currentRow()]  # 保存标记
#                 self.save_txt_swc(save_swc_path)
#
#     def box_y_to_down(self):
#         if self.dataImporter_index and self.box_volume:
#             if not self.y_down_over:
#                 (self.box_size, self.origin, self.y_up_over, self.y_down_over) = self.SetStepRedun.box_to_down(
#                     self.y_step, self.y_redun, self.box_size, self.origin, self.y_down_over, 1)
#                 self.create_new_image()
#                 save_swc_path = self.swc_file_list[self.img_list.currentRow()]  # 保存标记
#                 self.save_txt_swc(save_swc_path)
#
#     '''创建boxWidget'''
#
#     def box_Widget(self, box_volume):
#         self.BoxWidgets.create_box(box_volume, self)
#
#     '''获取boxWidgetParameter'''
#
#     def boxCallback(self, obj, event):
#         self.BoxWidgets.box_call_back(obj, event, self)
#
#     '''box切取的信息处理'''
#
#     def box_Widget_cut(self, bw_size):
#         self.BoxWidgets.box_cut_info(bw_size, self)
#
#     '''切取box'''
#
#     def box_cut_label(self):
#         self.BoxWidgets.box_cut_label2(self)
#
#     '''显示隐藏boxWidget'''
#
#     def box_show_hide(self):
#         self.BoxWidgets.box_show_hide2(self)
#
#     '''点击选取感兴趣区域'''
#
#     def point_to_box(self, obj, event):
#         self.BoxWidgets.point_to_box2(self)
#
#     '''筛选点'''
#
#     def sift_points(self, sift_box_size, pos, r):
#         return (sift_box_size[0] - r <= pos[0] <= sift_box_size[1] + r and
#                 sift_box_size[2] - r <= pos[1] <= sift_box_size[3] + r and
#                 sift_box_size[4] - r <= pos[2] <= sift_box_size[5] + r)
#
#     '''选择框架'''
#
#     def sift_box(self, img, need_px):
#         if self.data_volume_show_hide_count:  # 判断图文件是否显示
#             if need_px:
#                 box_size = [0, img.shape[2] * self.x_px, 0, img.shape[1] * self.y_px,
#                             0, img.shape[0] * self.z_px]
#             else:
#                 box_size = [0, img.shape[2], 0, img.shape[1], 0, img.shape[0]]
#         else:
#             if need_px:
#                 box_size = [self.box_size[0], self.box_size[1] * self.x_px,
#                             self.box_size[2], self.box_size[3] * self.y_px,
#                             self.box_size[4], self.box_size[5] * self.z_px]
#             else:
#                 box_size = self.box_size
#         return box_size
#
#     '''确认点位置'''
#
#     def points_pos_sure(self, points):
#         pos = self.MarkingPointsFun.points_pos_sure2(points, self)
#         return pos
#
#     '''确认点是否重合'''
#
#     def point_coincidence_sure(self, mypoint):
#         coincidence = self.MarkingPointsFun.point_coincidence_sure2(mypoint, self)
#         return coincidence
#
#     '''关闭窗口事件'''
#
#     def closeEvent(self, QCloseEvent):
#         box = QtWidgets.QMessageBox()
#         box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
#         box.setStyleSheet(messagebox_style)
#         box.setText('是否关闭窗口？')
#         box.setWindowTitle('Prompt')
#         box.setIcon(QtWidgets.QMessageBox.Question)
#         box.setWindowModality(Qt.ApplicationModal)
#         if box.exec_() == QtWidgets.QMessageBox.Yes:
#             QCloseEvent.accept()
#         else:
#             QCloseEvent.ignore()
#
#     '''滑轮前滑'''
#
#     def mouseWheelForward(self, obj, event):
#         # print("滚轮向前滚动")
#         camera_focal_pos = self.camera.GetFocalPoint()
#         camera_pos = self.camera.GetPosition()
#         v_ = np.array(camera_focal_pos) - np.array(camera_pos)
#         distance = np.linalg.norm(v_)
#         # print("相机焦距为：", distance)
#
#     '''滑轮后滑'''
#
#     def mouseWheelBackward(self, obj, event):
#         # print("滚轮向后滚动")
#         camera_focal_pos = self.camera.GetFocalPoint()
#         camera_pos = self.camera.GetPosition()
#         v_ = np.array(camera_focal_pos) - np.array(camera_pos)
#         distance = np.linalg.norm(v_)
#         # print("相机焦距为：", distance)
#
