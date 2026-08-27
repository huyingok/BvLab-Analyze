# -*- coding: utf-8 -*-

class SetConnect:
    def __init__(self, title_bar):
        self.open_file_act = title_bar.open_file_act
        self.save_pos_act = title_bar.save_pos_act
        self.images_list_act = title_bar.images_list_act
        self.funs_widget_act = title_bar.funs_widget_act
        self.point_withdraw_act = title_bar.point_withdraw_act
        self.actor_delete_act = title_bar.actor_delete_act
        self.all_actor_delete_act = title_bar.all_actor_delete_act
        self.change_function_act = title_bar.change_function_act
        self.sphere_quick_key_act = title_bar.sphere_quick_key_act
        self.outline_show_hide_act = title_bar.outline_show_hide_act
        self.data_volume_show_hide_act = title_bar.data_volume_show_hide_act
        self.box_to_up_act = title_bar.box_to_up_act
        self.box_to_down_act = title_bar.box_to_down_act
        self.gray_auto_change_act = title_bar.gray_auto_change_act
        self.camera_focal_act = title_bar.camera_focal_act
        self.sure_gray_auto_act = title_bar.sure_gray_auto_act
        self.gray_left_act = title_bar.gray_left_act
        self.gray_right_act = title_bar.gray_right_act
        self.item_up_act = title_bar.item_up_act
        self.item_down_act = title_bar.item_down_act
        self.box_x_to_down_act = title_bar.box_x_to_down_act
        self.box_x_to_add_act = title_bar.box_x_to_add_act
        self.box_y_to_add_act = title_bar.box_y_to_add_act
        self.box_y_to_down_act = title_bar.box_y_to_down_act
        self.box_show_hide_act = title_bar.box_show_hide_act
        self.box_cut_label_act = title_bar.box_cut_label_act
        self.recovery_xy_act = title_bar.recovery_xy_act
        self.recovery_z_act = title_bar.recovery_z_act

    '''自定义菜单栏信号'''
    # def set_title_menubar_connect(self, clss):
    #     print("ok")
    #     self.open_file_act.triggered.connect(clss.select_file_path_show)
    #     self.save_pos_act.triggered.connect(clss.save_position)
    #     self.images_list_act.triggered.connect(clss.images_list_widget)
    #     self.funs_widget_act.triggered.connect(clss.funs_widget_show_hide)
    #     self.point_withdraw_act.triggered.connect(clss.Point_Withdraw)  # 撤回
    #     # self.actor_delete_act.triggered.connect(clss.Actor_Delete)  # Delete
    #     self.all_actor_delete_act.triggered.connect(clss.all_actor_delete)  # 清空标记
    #     self.change_function_act.triggered.connect(clss.change_function)  # 切换功能
    #     self.sphere_quick_key_act.triggered.connect(clss._Sphere_show_hide)  # 显示和隐藏标记
    #     self.outline_show_hide_act.triggered.connect(clss.outline_show_hide)  # 显示和隐藏感兴趣区域
    #     self.data_volume_show_hide_act.triggered.connect(clss.data_volume_show_hide)  # 图像显示和隐藏
    #     self.box_to_up_act.triggered.connect(clss.box_up)  # 上切
    #     self.box_to_down_act.triggered.connect(clss.box_down)  # 下切
    #     self.gray_auto_change_act.triggered.connect(clss.gray_auto_change)  # 自动更新灰度
    #     self.camera_focal_act.triggered.connect(clss.camera_focal)  # Focus
    #     self.sure_gray_auto_act.triggered.connect(clss.sure_gray_auto)  # 固定灰度
    #     self.gray_left_act.triggered.connect(clss.gray_left)  # 左移灰度
    #     self.gray_right_act.triggered.connect(clss.gray_right)  # 右移灰度
    #     self.item_up_act.triggered.connect(clss.item_up)  # Toggle
    #     self.item_down_act.triggered.connect(clss.item_down)  # Toggle
    #     self.box_x_to_down_act.triggered.connect(clss.box_x_to_down)  # x轴上移
    #     self.box_x_to_add_act.triggered.connect(clss.box_x_to_add)  # x轴下移
    #     self.box_y_to_add_act.triggered.connect(clss.box_y_to_add)  # y轴上移
    #     self.box_y_to_down_act.triggered.connect(clss.box_y_to_down)  # y轴下移
    #     self.box_show_hide_act.triggered.connect(clss.box_show_hide)  # OK
    #     self.box_cut_label_act.triggered.connect(clss.box_cut_label)  # 切块
    #     self.recovery_xy_act.triggered.connect(clss.recovery_xy)  # Restore
    #     self.recovery_z_act.triggered.connect(clss.recovery_z)  # Restore

    # '''菜单栏信号'''
    # @classmethod
    # def set_menubar_connect(cls, self):  # 菜单栏信号
    #     self.open_file_act.triggered.connect(self.select_file_path_show)
    #     self.save_pos_act.triggered.connect(self.save_position)
    #     self.images_list_act.triggered.connect(self.images_list_widget)
    #     self.funs_widget_act.triggered.connect(self.funs_widget_show_hide)
    #     self.point_withdraw_act.triggered.connect(self.Point_Withdraw)  # 撤回
    #     self.actor_delete_act.triggered.connect(self.Actor_Delete)  # Delete
    #     self.change_function_act.triggered.connect(self.change_function)  # 切换功能
    #     self.sphere_quick_key_act.triggered.connect(self._Sphere_show_hide)  # 显示和隐藏标记
    #     self.outline_show_hide_act.triggered.connect(self.outline_show_hide)  # 显示和隐藏感兴趣区域
    #     self.data_volume_show_hide_act.triggered.connect(self.data_volume_show_hide)  # 图像显示和隐藏
    #     self.box_to_up_act.triggered.connect(self.box_to_up)  # 上切
    #     self.box_to_down_act.triggered.connect(self.box_to_down)  # 下切
    #     self.gray_auto_change_act.triggered.connect(self.gray_auto_change)  # 自动更新灰度
    #     self.camera_focal_act.triggered.connect(self.camera_focal)  # Focus
    #     self.sure_gray_auto_act.triggered.connect(self.sure_gray_auto)  # 固定灰度
    #     self.gray_left_act.triggered.connect(self.gray_left)  # 左移灰度
    #     self.gray_right_act.triggered.connect(self.gray_right)  # 右移灰度
    #     self.item_up_act.triggered.connect(self.item_up)  # Toggle
    #     self.item_down_act.triggered.connect(self.item_down)  # Toggle
    #     self.box_x_to_down_act.triggered.connect(self.box_x_to_down)  # x轴上移
    #     self.box_x_to_add_act.triggered.connect(self.box_x_to_add)  # x轴下移
    #     self.box_y_to_add_act.triggered.connect(self.box_y_to_add)  # y轴上移
    #     self.box_y_to_down_act.triggered.connect(self.box_y_to_down)  # y轴下移
    #     self.box_show_hide_act.triggered.connect(self.box_show_hide)  # OK)
    #     self.box_cut_label_act.triggered.connect(self.box_cut_label)  # 切块
    #     self.recovery_xy_act.triggered.connect(self.recovery_xy)  # Restore
    #     self.recovery_z_act.triggered.connect(self.recovery_z)  # Restore

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

    '''功能框信息及信号'''
    @classmethod
    def fun_widget_info_connect(cls, self):
        # 标记功能切换
        self.function_change_button.clicked.connect(self.change_function)  # 切换功能
        # 切换渲染模式
        self.render_mode_combo.currentTextChanged.connect(self._on_render_mode_changed)
        # 切换标记渲染
        self.mark_render_combo.currentTextChanged.connect(self._on_mark_render_changed)
        # 标记颜色
        self.MarkColorDialog_button.clicked.connect(self.MarkColorDialog)  # 打开颜色选择框
        # Resolution
        self.x_px_spinbox.editingFinished.connect(self.update_image_px)
        self.y_px_spinbox.editingFinished.connect(self.update_image_px)
        self.z_px_spinbox.editingFinished.connect(self.update_image_px)
        # Image
        self.image_button.clicked.connect(self.outline_show_hide)  # 显示和隐藏感兴趣区域
        # Mark
        self.sphere_button.clicked.connect(self._Sphere_show_hide)  # 显示和隐藏标记
        # 颜色
        self.openColorDialog_button.clicked.connect(self.openColorDialog)  # 打开颜色对话框
        # 自动调节灰度
        self.gray_auto_change_button.clicked.connect(self.gray_auto_change)  # 设置快捷灰度
        # 标记大小
        self.sphere_size_slider.valueChanged.connect(self.change_sphere_size)  # 修改大小
        # 灰度调节
        self.gray_min_num.editingFinished.connect(self.textinput_gray_min)
        self.gray_max_num.editingFinished.connect(self.textinput_gray_max)
        self.gray_min_num.editingFinished.connect(self.change_color_opacity)
        self.gray_max_num.editingFinished.connect(self.change_color_opacity)
        self.gray_min_num.valueChanged.connect(self.change_color_opacity)
        self.gray_max_num.valueChanged.connect(self.change_color_opacity)

    '''切块列表信号'''
    @classmethod
    def layer_widget_info_connect(cls, self):
        self.z_list.currentItemChanged.connect(self.change_z)
        self.recovery_xy_button.clicked.connect(self.recovery_xy)  # Signal
        self.recovery_z_button.clicked.connect(self.recovery_z)  # Signal
        self.table_widget.itemClicked.connect(self.change_xy)
        self.remove_edge_button.clicked.connect(self.remove_edge_annotations)

        # self.z_step_num.valueChanged.connect(self.change_z_step)
        self.z_step_num.valueChanged.connect(self.change_z_step)
        self.z_redun_num.valueChanged.connect(self.change_z_redun)

        self.x_step_num.valueChanged.connect(self.change_x_step)
        self.x_redun_num.valueChanged.connect(self.change_x_redun)
        self.y_step_num.valueChanged.connect(self.change_y_step)
        self.y_redun_num.valueChanged.connect(self.change_y_redun)

    '''xyz悬浮窗口信号'''
    @classmethod
    def xyz_widget_connect(cls, recovery_xy_button, recovery_z_button, self):
        self.z_list.currentItemChanged.connect(self.change_z)
        recovery_xy_button.clicked.connect(self.recovery_xy)  # Signal
        recovery_z_button.clicked.connect(self.recovery_z)  # Signal
        self.table_widget.itemClicked.connect(self.change_xy)

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

    """工具按钮点击"""

    @classmethod
    def tool_button_clicked(cls, self):
        self.clear_button.clicked.connect(self.clear_view)
        self.projection_button.clicked.connect(self.toggle_projection_mode)
        self.tool_xy_button.clicked.connect(self.set_camera_view_up_y)
        self.tool_xz_button.clicked.connect(self.set_camera_view_up_x)
        self.tool_yz_button.clicked.connect(self.set_camera_view_up_z)
        self.focal_button.clicked.connect(self.camera_focal)

        self.open_file_button.clicked.connect(self.select_file_path_show)  # Open
        self.save_pos_button.clicked.connect(self.save_position)  # Save
        self.set_function_button.clicked.connect(self.set_user_function)  # Settings
        self.filter_button.clicked.connect(self.filter_dialog_show)  # 打开筛选窗口
        self.filter_finish_button.clicked.connect(self.filter_finish)  # 筛选完成

        self.start_mark_button.clicked.connect(self.start_mark_button_clicked)  # 选择标记
        self.clear_mark_button.clicked.connect(self.all_actor_delete)  # 清空标记
        # self.delete_mark_button.clicked.connect(self.Actor_Delete)  # 删除标记
        self.withdraw_button.clicked.connect(self.Point_Withdraw)  # 撤回

    """快捷方式信号"""

    @classmethod
    def short_cut_connect(cls, self):
        self.start_mark_shortcut.activated.connect(self.start_mark_button_clicked)  # 修订模式
        self.point_withdraw_shortcut.activated.connect(self.Point_Withdraw)  # 撤回
        # self.actor_delete_shortcut.activated.connect(self.Actor_Delete)  # Delete
        self.all_actor_delete_shortcut.activated.connect(self.all_actor_delete)  # 清空标记
        self.change_function_shortcut.activated.connect(self.change_function)  # 切换功能
        self.sphere_quick_key_shortcut.activated.connect(self._Sphere_show_hide)  # 显示和隐藏标记
        self.outline_show_hide_shortcut.activated.connect(self.outline_show_hide)  # 显示和隐藏感兴趣区域
        # self.data_volume_show_hide_shortcut.activated.connect(self.data_volume_show_hide)  # 图像显示和隐藏
        self.box_to_up_shortcut.activated.connect(self.box_up)  # 上切
        self.box_to_down_shortcut.activated.connect(self.box_down)  # 下切
        self.gray_auto_change_shortcut.activated.connect(self.gray_auto_change)  # 自动更新灰度
        self.camera_focal_shortcut.activated.connect(self.camera_focal)  # Focus
        # self.sure_gray_auto_shortcut.activated.connect(self.sure_gray_auto)  # 固定灰度
        self.item_up_shortcut.activated.connect(self.item_up)  # Toggle
        self.item_down_shortcut.activated.connect(self.item_down)  # Toggle
        # self.box_show_hide_shortcut.activated.connect(self.box_show_hide)  # OK
        # self.box_cut_label_shortcut.activated.connect(self.box_cut_label)  # 切块
        self.recovery_xy_shortcut.activated.connect(self.recovery_xy)  # Restore
        self.recovery_z_shortcut.activated.connect(self.recovery_z)  # Restore
        self.filter_images_shortcut.activated.connect(self.filter_images)  # 筛选图像
