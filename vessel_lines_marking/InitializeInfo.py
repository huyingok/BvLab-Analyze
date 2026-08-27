# -*- coding: utf-8 -*-
class InitializeInfo:
    def __init__(self):
        self.img3d = None
        self.img3d_mark = None
        self.dataImporter = None
        self.dataImporter_mark = None
        self.dataImporter_outline = None
        self.volume = None
        self.outline = None
        self.outlineActor = None
        self.volume_mark = None
        self.outlineActor0 = None

        self.image_path_list = []  # 图像路径列表

        self.eye_right = None
        self.eye_left = None
        self.eye_mid = None
        self.dataImporter_index = False  # 定义一个vtkImageImportObject
        self.image_file_list = []  # 图文件列表
        self.image_filename_list = []  # 图名称列表
        self.image_file_list_index = 0  # 对应序号
        self.swc_file_list = []  # swc文件列表

        self.new_points_list = []  # 新增点
        self.new_Position = []  # 新增点坐标
        self.new_mark_lines = []  # 新增线

        self.vtk_Points = []  # 点集列表
        self.vtk_Lines = []  # 线集列表
        self.vtk_LP_Actor = []  # 点线actorList
        self.vtk_Vertices = []  # 顶点集
        self.color_i = None  # 选择点线序号
        self.PolyData_copy = []  # 备份点线
        self.connect_list = []  # 连接列表
        self.focal_pos = None  # 聚焦点

        self.LP_hide = False  # 1：球体隐藏
        self.outline_hide = False  # 包围盒显示和隐藏
        self.point_line_count = True  # 点击生成线，1为激活状态
        self.one_sphere_count = False  # 单击生成点，0为未触发
        self.function_index = 1  # 功能切换
        self.fun_label = 'Max mode'
        self.render_mode_text = 'MIP'
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

        self.selection = False  # 图像选择框是否执行切换
        self.p_size = 6  # 点大小
        self.length_size = 10  # 长度限制
        self.is_marking = False  # 是否在标记
        self.is_marking_coincidence = {}  # 第一个标记是否重合
        self.boxR_size = 50  # 裁剪框半径
        self.tree_count = 0  # 分支数
        self.img3d_mark_info = [[], []]  # 定位图像信息

        self.reOpen = True  # 重新打开文件

        self.filter_res = {}  # 筛选结果列表
        self.statistics_dict = {}
        self.cross_actor_points = []
        self.cross_actor_points_copy = []

        # 创建PyQtGraph绘图区域
        self.adjust_nums = 50
        self.radius_dict = {}
        self.new_radius_dict = {}
        self.is_sphere_point = False  # 第一条直线是否完成，如果完成后续变为False，防止重复
        self.is_cross_finish = True  # 冻结调节界面
        self.is_left = 1

        # 分支点字典
        self.branch_dict = {}

    '''更新初始化信息'''
    def update_initialize_info(self):
        self.img3d = None
        self.img3d_mark = None
        self.dataImporter = None
        self.dataImporter_mark = None
        self.dataImporter_outline = None
        self.volume = None
        self.outline = None
        self.outlineActor = None
        self.volume_mark = None
        self.outlineActor0 = None

        self.new_points_list = []  # 新增点
        self.new_Position = []  # 新增点坐标
        self.new_mark_lines = []  # 新增线
        self.vtk_Points = []  # 点集列表
        self.vtk_Lines = []  # 线集列表
        self.vtk_LP_Actor = []  # 点线actorList
        self.vtk_Vertices = []  # 顶点集
        self.color_i = None  # 选择点线序号
        self.PolyData_copy = []  # 备份点线
        self.connect_list = []  # 连接列表

        self.tree_count = 0  # 分支数
        self.function_index = 1  # 功能切换
        self.fun_label = 'Max mode'
        self.LP_hide = False  # 球体显示和隐藏
        self.outline_hide = False  # 包围盒显示和隐藏
        self.selection = False  #
        self.is_marking = False  # 是否在标记
        self.is_marking_coincidence = {}  # 第一个标记是否重合
        self.point_line_count = True  # 点击生成线，1为激活状态
        self.one_sphere_count = False  # 单击生成点，0为未触发
        self.img3d_mark_info = [[], []]  # 定位图像信息

        self.statistics_dict = {}
        self.cross_actor_points = []
        self.cross_actor_points_copy = []
        self.radius_dict = {}
        self.new_radius_dict = {}
        self.is_sphere_point = False  # 第一条直线是否完成，如果完成后续变为False，防止重复
        self.is_cross_finish = True  # 冻结调节界面
        self.is_left = 1

        # 分支点字典
        self.branch_dict = {}
