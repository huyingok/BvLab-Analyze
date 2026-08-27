# -*- coding: utf-8 -*-

class SetConnect:
    def __init__(self, title_bar):
        self.open_file_act = title_bar.open_file_act
        self.save_pos_act = title_bar.save_pos_act
        self.point_withdraw_act = title_bar.point_withdraw_act
        self.mark_withdraw_act = title_bar.mark_withdraw_act
        self.actor_delete_act = title_bar.actor_delete_act
        self.actor_all_delete_act = title_bar.actor_all_delete_act
        self.actor_all_delete_limit_act = title_bar.actor_all_delete_limit_act
        self.volume_change_act = title_bar.volume_change_act
        self.new_mark_create_act = title_bar.new_mark_create_act
        self.change_function_act = title_bar.change_function_act
        self.sphere_quick_key_act = title_bar.sphere_quick_key_act
        self.outline_quick_key_act = title_bar.outline_quick_key_act
        self.gray_auto_change_act = title_bar.gray_auto_change_act
        self.gray_limit_change_act = title_bar.gray_limit_change_act
        self.camera_focal_act = title_bar.camera_focal_act
        self.sure_gray_auto_act = title_bar.sure_gray_auto_act
        self.start_marking_act = title_bar.start_marking_act
        self.end_marking_act = title_bar.end_marking_act
        self.gray_left_act = title_bar.gray_left_act
        self.gray_right_act = title_bar.gray_right_act
        self.item_up_act = title_bar.item_up_act
        self.item_down_act = title_bar.item_down_act

    '''自定义菜单栏信号'''

    # def set_title_menubar_connect(self, clss):
    #     print("ok")
    #     self.open_file_act.triggered.connect(clss.select_file_path_show)
    #     self.save_pos_act.triggered.connect(clss.save_position)
    #     self.point_withdraw_act.triggered.connect(clss.Point_Withdraw)  # 撤回  Ctrl+Z
    #     self.mark_withdraw_act.triggered.connect(clss.Mark_Withdraw)  # 修订模式撤回所有点  Z
    #     self.actor_delete_act.triggered.connect(clss.Actor_Delete)  # Delete
    #     self.actor_all_delete_act.triggered.connect(clss.Actor_All_Delete)  # 删除所有actor
    #     self.actor_all_delete_limit_act.triggered.connect(clss.Actor_All_Delete_limit)  # 删除所有限制长度actor
    #     self.volume_change_act.triggered.connect(clss.volume_change)  # Togglevolume
    #     self.new_mark_create_act.triggered.connect(clss.new_mark_create)  # 创建裁剪标记图像
    #     self.change_function_act.triggered.connect(clss.change_function)  # 切换功能
    #     self.sphere_quick_key_act.triggered.connect(clss._LP_show_hide)  # 显示和隐藏标记
    #     self.outline_quick_key_act.triggered.connect(clss.Outline_quick_key)  # 显示和隐藏包围盒
    #     self.gray_auto_change_act.triggered.connect(clss.gray_auto_change)  # 自动更新灰度
    #     self.camera_focal_act.triggered.connect(clss.camera_focal)  # Focus
    #     self.sure_gray_auto_act.triggered.connect(clss.sure_gray_auto)  # 固定灰度
    #     self.start_marking_act.triggered.connect(clss.start_marking)  # 开始标记
    #     self.end_marking_act.triggered.connect(clss.end_marking)  # 结束标记
    #     self.gray_left_act.triggered.connect(clss.gray_left)  # 左移灰度
    #     self.gray_right_act.triggered.connect(clss.gray_right)  # 右移灰度
    #     self.item_up_act.triggered.connect(clss.item_up)  # 切换图像
    #     self.item_down_act.triggered.connect(clss.item_down)  # 切换图像

    '''路径选择信号'''

    @classmethod
    def select_path_connect(cls, self):
        self.path_radioButton.toggled.connect(self.path_radioButton_clicked)  # 路径选择
        self.config_radioButton.toggled.connect(self.config_radioButton_clicked)  # 配置文件选择
        self.name_radioButton.toggled.connect(self.name_radioButton_clicked)
        self.select_sure_button.clicked.connect(self.sure_path_select)  # Signal
        self.select_cancel_button.clicked.connect(self.cancel_path_select)  # Signal
        self.image_path_open_button.clicked.connect(self.image_path_open)  # Signal
        self.swc_path_open_button.clicked.connect(self.swc_path_open)  # Signal
        self.config_path_open_Button.clicked.connect(self.config_path_open)  # Signal
        self.name_path_open_Button.clicked.connect(self.name_path_open)

    '''筛选保存信号'''

    @classmethod
    def filter_connect(cls, self):
        self.filter_ok_button.clicked.connect(self.filter_start)  # 启动筛选模式
        self.filter_cancel_button.clicked.connect(self.filter_cancel)  # Cancel
        self.save_path_open_button.clicked.connect(self.filter_save_path_open)  # 选择筛选保存路径

    '''半径调节信号'''

    @classmethod
    def select_radius_connect(cls, self):
        # 数值变化 -> 移动垂线
        self.forward_spinBox.valueChanged.connect(self.update_forw_line)
        self.backward_spinBox.valueChanged.connect(self.update_back_line)
        # 垂线手动拖动 -> 反向更新控件
        self.forw_line.sigPositionChanged.connect(self.forw_line_moved)
        self.back_line.sigPositionChanged.connect(self.back_line_moved)
        # 调节结果保存
        self.res_save_Button.clicked.connect(self.save_radius_res)

    '''功能框信息及信号'''

    @classmethod
    def fun_widget_info_connect(cls, self):
        # 标记功能
        self.function_change_button.clicked.connect(self.change_function)
        # Information
        # self.info_num_label
        # 切换渲染模式
        self.render_mode_combo.currentTextChanged.connect(self._on_render_mode_changed)
        # Image、分支显示和预测
        self.show_image_button.clicked.connect(self.volume_change)
        self.show_lines_button.clicked.connect(self._LP_show_hide)
        # 调节点数
        self.adjust_spinBox.editingFinished.connect(self.update_adjust_size)
        self.dialog_show_Button.clicked.connect(self.show_dialog_hide)
        # 灰度
        self.gray_auto_change_button.clicked.connect(self.gray_auto_change)  # 设置快捷灰度
        # 点大小
        self.size_spinBox.editingFinished.connect(self.update_point_size)
        # 长度限制
        self.length_spinBox.editingFinished.connect(self.update_length_size)
        # 框半径
        self.boxR_spinBox.editingFinished.connect(self.update_boxR_size)
        # 分支数
        # self.label_count
        # 筛选数
        # self.filter_nums_label
        # 灰度调节
        self.gray_min_num.editingFinished.connect(self.textinput_gray_min)
        self.gray_max_num.editingFinished.connect(self.textinput_gray_max)
        self.gray_min_num.editingFinished.connect(self.change_color_opacity)
        self.gray_max_num.editingFinished.connect(self.change_color_opacity)
        self.gray_min_num.valueChanged.connect(self.change_color_opacity)
        self.gray_max_num.valueChanged.connect(self.change_color_opacity)

    '''分支点选择信号'''

    @classmethod
    def branch_widget_info_connect(cls, self):
        self.branch_select_listwidget.currentItemChanged.connect(self.branch_region_show)
        self.branch_forward_Button.clicked.connect(self.to_forward_branch)
        self.branch_backward_Button.clicked.connect(self.to_backward_branch)
        self.branch_sure_button.clicked.connect(self.branch_finished)
        self.branch_close_button.clicked.connect(self.branch_closed)

    '''图形列表项信号'''

    @classmethod
    def img_list_connect(cls, self):
        self.img_list.currentItemChanged.connect(self.on_item_selection)

    '''相机朝上方向'''

    @classmethod
    def tool_bar_action(cls, self):
        self.clear_button.triggered.connect(self.clear_view)
        self.tool_xy_button.triggered.connect(self.set_camera_view_up_y)
        self.tool_xz_button.triggered.connect(self.set_camera_view_up_x)
        self.tool_yz_button.triggered.connect(self.set_camera_view_up_z)

    '''自定义工具栏'''
    @classmethod
    def tool_button_clicked(cls, self):
        # 打开文件
        self.open_file_button.clicked.connect(self.select_file_path_show)
        # 保存文件
        self.save_pos_button.clicked.connect(self.save_position)
        # 功能设置
        self.set_function_button.clicked.connect(self.set_user_function)
        # 打开筛选窗口
        self.filter_button.clicked.connect(self.filter_dialog_show)
        # 筛选完成
        self.filter_finish_button.clicked.connect(self.filter_finish)
        # 平面投影
        self.tool_xy_button.clicked.connect(self.set_camera_view_up_y)
        self.tool_xz_button.clicked.connect(self.set_camera_view_up_x)
        self.tool_yz_button.clicked.connect(self.set_camera_view_up_z)
        # 修订模式
        self.marking_button.clicked.connect(self.change_marking_mode)
        # 确定标注
        self.sure_marking_button.clicked.connect(self.sure_marking)
        # 删除分支
        self.delete_mark_button.clicked.connect(self.Actor_Delete)
        # 删除小于长度限制的分支
        self.delete_limit_button.clicked.connect(self.Actor_All_Delete_limit)
        # 清空分支
        self.clear_mark_button.clicked.connect(self.Actor_All_Delete)
        # 撤回待定分支
        self.withdraw_all_button.clicked.connect(self.Mark_Withdraw)
        # 撤回
        self.withdraw_button.pressed.connect(self.Point_Withdraw)
        # 分支点聚焦
        self.focal_button.clicked.connect(self.camera_focal)
        # 清空界面
        self.clear_button.clicked.connect(self.clear_view)
        # 切换投影方式
        self.projection_button.clicked.connect(self.toggle_projection_mode)

    """快捷方式信号"""

    @classmethod
    def short_cut_connect(cls, self):
        self.point_withdraw_shortcut.activated.connect(self.Point_Withdraw)  # 撤回  Ctrl+Z
        self.mark_withdraw_shortcut.activated.connect(self.Mark_Withdraw)  # 修订模式撤回所有点  Z
        self.actor_delete_shortcut.activated.connect(self.Actor_Delete)  # Delete
        self.all_actor_delete_shortcut.activated.connect(self.Actor_All_Delete)  # 删除所有actor
        self.all_actor_limit_delete_shortcut.activated.connect(self.Actor_All_Delete_limit)  # 删除所有限制长度actor
        self.volume_change_shortcut.activated.connect(self.volume_change)  # Togglevolume
        self.new_mark_create_shortcut.activated.connect(self.new_mark_create)  # 创建裁剪标记图像
        self.change_function_shortcut.activated.connect(self.change_function)  # 切换功能
        self.LP_quick_key_shortcut.activated.connect(self._LP_show_hide)  # 显示和隐藏标记
        self.outline_quick_key_shortcut.activated.connect(self.Outline_quick_key)  # 显示和隐藏包围盒
        self.gray_auto_change_shortcut.activated.connect(self.gray_auto_change)  # 自动更新灰度
        self.camera_focal_shortcut.activated.connect(self.camera_focal)  # Focus
        self.start_marking_shortcut.activated.connect(self.start_marking)  # 修订模式
        self.sure_marking_shortcut.activated.connect(self.sure_marking)  # 确定标记
        self.end_marking_shortcut.activated.connect(self.end_marking)  # 关闭修订模式
        self.item_up_shortcut.activated.connect(self.item_up)  # 切换图像
        self.item_down_shortcut.activated.connect(self.item_down)  # 切换图像
        self.filter_images_shortcut.activated.connect(self.filter_images)  # 筛选图像

        self.branch_forward_shortcut.activated.connect(self.to_forward_branch)
        self.branch_backward_shortcut.activated.connect(self.to_backward_branch)
