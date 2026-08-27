# -*- coding: utf-8 -*-
import os.path
import shutil
import json
import time
import vtkmodules.all as vtk
import numpy as np
from tqdm import tqdm
from pathlib import Path as pp
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import QSettings, QSize
from PyQt5 import QtWidgets
from vessel_lines_marking.InitializeInfo import InitializeInfo
from DataStatistics.CollectBranchPaths import SplitSwcData, SplitSwcToBranchData, collect_branch_points


class ReadPath(InitializeInfo):
    def __init__(self):
        InitializeInfo.__init__(self)
        pass

    """选择最优swcFolder"""

    @classmethod
    def collect_best_swc_name(cls, root, root_name):
        def lcs_length(s1, s2):
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            max_length = 0
            end_index = 0

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i - 1] == s2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                        if dp[i][j] > max_length:
                            max_length = dp[i][j]
                            end_index = i
                    else:
                        dp[i][j] = 0

            return max_length, end_index - max_length, end_index

        def find_best_match(target, string_list):
            best_match = None
            max_lcs_length = 0

            for string in string_list:
                lcs_len, _, _ = lcs_length(target, string)
                if lcs_len > max_lcs_length:
                    max_lcs_length = lcs_len
                    best_match = string

            return best_match

        dir_names = []
        names = os.listdir(root)
        for name in names:
            path = os.path.join(root, name)
            if os.path.isdir(path):
                try:
                    str(name).encode('ascii')
                except UnicodeEncodeError:
                    continue
                name_l = name.lower()
                if "swc" in name_l:
                    dir_names.append(name)
        try:
            dir_names.remove(root_name)
        except ValueError:
            pass

        if len(dir_names) != 0:
            best_name = find_best_match(root_name, dir_names)
            if best_name is not None:
                return best_name
        best_name = root_name + "_swc"
        return best_name

    '''获取tif文件路径'''
    @classmethod
    def select_image_path(cls, image_line_edit, self):
        try:
            # 创建QSettingsObject
            settings = QSettings("MyCompany", "MyApp")
            # 读取之前保存的路径信息
            initial_path = settings.value("ImportVesselImagePath", "")
            # 获取多个文件路径
            filenames, filetype = QtWidgets.QFileDialog.getOpenFileNames(None, "Select Files", str(initial_path),
                                                                         "Tif Files (*.tif; *.tiff;);;Bv Files (*.bv;)")

            # 如果选择了文件，保存选择的文件路径
            if filenames:
                if not cls.contains_chinese(pp(filenames[0]).parent):
                    last_path = str(pp(filenames[0]).parent)  # 获取第一个文件的文件夹路径
                    settings.setValue("ImportVesselImagePath", last_path)
                    image_line_edit.setText(last_path)  # 设置初始文本 -- 修改可视化数值
                    self.image_path_list = filenames
                else:
                    self.mess_set("Path contains Chinese characters or spaces", 'Prompt', 1)
        except Exception as e:
            print(f"Open failed! {e}", "Prompt", 1)
            self.error_print(e)

    '''获取文件夹路径'''
    @classmethod
    def select_swc_path(cls, dir_line_edit, self):
        try:
            # 创建QSettingsObject
            settings = QSettings("MyCompany", "MyApp")
            # 读取之前保存的路径信息
            initial_path = settings.value("ImportVesselAnnotationPath", "")

            # 获取文件夹路径
            path = QtWidgets.QFileDialog.getExistingDirectory(None, "Select annotation folder", str(initial_path))

            # 如果选择了文件，保存选择的文件路径
            if path:
                if not cls.contains_chinese(path):
                    settings.setValue("ImportVesselAnnotationPath", path)
                    # 打印选择的文件路径
                    # print(f"Selected dir: {path}")
                    dir_line_edit.setText(path)  # 设置初始文本 -- 修改可视化数值
                else:
                    self.mess_set("Path contains Chinese characters or spaces", 'Prompt', 1)
        except Exception as e:
            print(f"Open failed! {e}", "Prompt", 1)

        path = dir_line_edit.text()
        return path

    '''获取文件夹路径'''

    @classmethod
    def select_config_path(cls, config_lineEdit, self):
        try:
            # 创建QSettingsObject
            settings = QSettings("MyCompany", "MyApp")
            # 读取之前保存的路径信息
            initial_path = settings.value("VesselConfigPath", "")
            initial_path = os.path.join(initial_path, "config.json")
            # 获取文件路径
            filename, filetype = QtWidgets.QFileDialog.getOpenFileName(None, "Select File", initial_path,
                                                                       "Json File (*.json)")

            # 如果选择了文件，保存选择的文件路径
            if filename:
                if not cls.contains_chinese(filename):
                    settings.setValue("VesselConfigPath", str(pp(filename).parent))
                    # 打印选择的文件路径
                    # print(f"Selected dir: {path}")
                    config_lineEdit.setText(filename)  # 设置初始文本 -- 修改可视化数值
                else:
                    self.mess_set("Path contains Chinese characters or spaces", 'Prompt', 1)
        except Exception as e:
            print(f"Open failed! {e}", "Prompt", 1)

    @classmethod
    def select_name_path(cls, name_lineEdit, self):
        try:
            # 创建QSettingsObject
            settings = QSettings("MyCompany", "MyApp")
            # 读取之前保存的路径信息
            initial_path = settings.value("VesselJsonPath", "")
            initial_path = os.path.join(initial_path, "config.json")
            # 获取文件路径
            filename, filetype = QtWidgets.QFileDialog.getOpenFileName(None, "Select File", initial_path,
                                                                       "Json File (*.json)")

            # 如果选择了文件，保存选择的文件路径
            if filename:
                if not cls.contains_chinese(filename):
                    settings.setValue("VesselJsonPath", str(pp(filename).parent))
                    # 打印选择的文件路径
                    # print(f"Selected dir: {path}")
                    name_lineEdit.setText(filename)  # 设置初始文本 -- 修改可视化数值
                else:
                    self.mess_set("Path contains Chinese characters or spaces", 'Prompt', 1)
        except Exception as e:
            print(f"Open failed! {e}", "Prompt", 1)

    '''判断路径是否包含中文'''
    @classmethod
    def contains_chinese(cls, path):
        try:
            str(path).encode('ascii')
        except UnicodeEncodeError:
            return True
        if " " in str(path):
            return True
        return False

    '''读路径'''
    @classmethod
    def _read_file_path(cls, self):
        print("Current directory folder path:", pp().cwd())
        # print("选择文件序号:", self.image_file_list_index)
        # print("选择文件路径:", self.image_path)
        # print("swc文件路径列表:", self.swc_file_list)
        # print("图文件名列表:", self.image_filename_list)
        # print("图文件路径列表:", self.image_file_list)
        # 文件列表项
        self.img_list.clear()  # 清空列表项
        for i in range(len(self.image_filename_list)):
            item = QtWidgets.QListWidgetItem()
            item.setText(f"%s   %s" % (str(i + 1), self.image_filename_list[i]))  # 列表项名字
            item.setFont(QFont("微软雅黑", 10))  # 字体
            item.setSizeHint(QSize(self.img_list.width() - 8, 20))  # 设置列表项尺寸
            self.img_list.addItem(item)  # 添加列表项
            self.img_list.update()  # 更新列表项
        # 获取要设置为当前选择的项的索引
        index = self.img_list.model().index(self.image_file_list_index, 0)  # 选取第3个项（索引从0Start）
        self.img_list.setCurrentIndex(index)
        self.reOpen = True
        self.Visualization_tif_file()

    '''读取txtFile,读取选择文件的txtFile'''
    @classmethod
    def _read_txt_file(cls, index, self):
        # 读取swc，获取已有点
        save_swc_path = self.swc_file_list[index]
        if not pp(save_swc_path).exists():  # 是否存在文件
            with open(save_swc_path, 'w') as f:
                f.write("")
        pp(save_swc_path).chmod(0o666)  # 取消文件只读
        swcData_ = np.loadtxt(save_swc_path, ndmin=2)  # 读取swcFile
        swcDataLs = SplitSwcData(swcData_)  # swc多树拆分

        for i, swcData in enumerate(swcDataLs):  # 遍历每个分支树
            points = vtk.vtkPoints()  # 创建点集合
            lines = vtk.vtkCellArray()  # 创建单元数组
            # 创建顶点单元（每个点是一个顶点）
            vertices = vtk.vtkCellArray()
            for ii, item in enumerate(swcData):  # 分支树信息
                p0 = item[2: 5]  # 当前位置坐标x，y，z
                index0 = int(item[-1])
                id0 = int(item[0])
                points.InsertNextPoint(p0)

                # 备份直径
                if item[5]:
                    key = self.pos_to_key(p0)
                    if self.radius_dict.get(key, None) is None:
                        self.radius_dict[key] = item[5]

                if index0 != -1:  # 父节点
                    index1 = index0 - 1  # 前一个点序号
                    # 创建线段
                    line = vtk.vtkLine()
                    if id0 != index0 + 1:  # 分支点
                        index0 = id0 - 1
                    line.GetPointIds().SetId(0, index1)  # 第一个点
                    line.GetPointIds().SetId(1, index0)  # 第二个点
                    # 单元数组添加线段
                    lines.InsertNextCell(line)
                # vertices.InsertNextCell(1)
                # vertices.InsertCellPoint(ii)
                vert = vtk.vtkVertex()
                vert.GetPointIds().SetId(0, ii)
                vertices.InsertNextCell(vert)

            # 创建 PolyData 并设置点和线
            actor = self.create_PolyData(points, lines, vertices)
            # actor.VisibilityOff()
            self.vtk_Points.append(points)
            self.vtk_Vertices.append(vertices)
            self.vtk_Lines.append(lines)
            self.vtk_LP_Actor.append(actor)

            # # 获取所有点坐标
            # print("所有点坐标：")
            # num_points = points.GetNumberOfPoints()
            # for i in range(num_points):
            #     coord = points.GetPoint(i)
            #     print(f"Point {i}: {coord}")
            #
            # # 获取每条线段的点索引
            # print("\n每条线段的点索引：")
            # num_lines = polydata.GetNumberOfLines()
            # print(f"线段数量: {num_lines}")
            #
            # 遍历所有线段
            # lines.InitTraversal()
            # id_list = vtk.vtkIdList()
            # while lines.GetNextCell(id_list):
            #     print("线段包含的点索引:", end=" ")
            #     for i in range(id_list.GetNumberOfIds()):
            #         print(id_list.GetId(i), end=" ")
            #     print()

        self.setEnabled(False)  # 整棵控件树全灰
        for a in self.findChildren(QtWidgets.QAction):
            a.setEnabled(False)
        for s in self.findChildren(QtWidgets.QShortcut):
            s.setEnabled(False)

        # 记录分支点
        st_time = time.time()
        self.branch_dict = {}
        trees = SplitSwcToBranchData(swcData_)
        branch_points, end_points = collect_branch_points(trees)
        # 分支选择列表, 添加末端点
        for i, n in enumerate(branch_points + end_points):
            self.branch_dict[i] = n
        print("Branching took {} seconds".format(time.time() - st_time))

        # 更新分支点选择列表项
        self.update_branch_list()

        self.setEnabled(True)  # 恢复整棵控件树
        for a in self.findChildren(QtWidgets.QAction):
            a.setEnabled(True)
        for s in self.findChildren(QtWidgets.QShortcut):
            s.setEnabled(True)

        self.tree_count = len(self.vtk_LP_Actor)
        self.label_count.setText(f"{self.tree_count}")

    """配置文件问题"""

    @classmethod
    def read_config_question(cls, self):
        path = self.config_lineEdit.text()
        if os.path.exists(path):
            with open(path, 'r') as f:
                info = json.loads(f.read())
                img_dir = info.get('divide_img_dir', None)
                swc_dir = info.get('divide_swc_dir', None)
                if img_dir is None:
                    self.mess_set("Configuration file has no divide_img_dir parameter, cannot read!", "Prompt", 1)
                    return False
                else:
                    if os.path.exists(img_dir):
                        self.image_lineEdit.setText(img_dir)
                        self.image_path_list = [str(os.path.join(img_dir, n)) for n in os.listdir(img_dir)
                                                if ".tif" in n]
                    else:
                        self.mess_set("The divide_img_dir path in the configuration file does not exist, cannot read!", "Prompt", 1)
                        return False
                if swc_dir is None:
                    self.mess_set("Configuration file has no divide_swc_dir parameter, cannot read!", "Prompt", 1)
                    return False
                else:
                    if os.path.exists(swc_dir):
                        self.swc_lineEdit.setText(swc_dir)
                        self.swc_path = swc_dir
                    else:
                        self.mess_set("The divide_swc_dir path in the configuration file does not exist, cannot read!", "Prompt", 1)
                        return False
        else:
            self.mess_set("Configuration file does not exist!", "Prompt", 1)
            return False
        return True

    @classmethod
    def read_name_question(cls, self):
        path = self.name_lineEdit.text()
        if os.path.exists(path):
            with open(path, 'r') as f:
                info = json.loads(f.read())
                img_dir = info.get('cut_img_dir', None)
                swc_dir = info.get('cut_swc_dir', None)
                if img_dir is None:
                    self.mess_set("Configuration file has no cut_img_dir parameter, cannot read!", "Prompt", 1)
                    return False
                else:
                    if os.path.exists(img_dir):
                        self.image_lineEdit.setText(img_dir)
                        self.image_path_list = [str(os.path.join(img_dir, n)) for n in os.listdir(img_dir)
                                                if ".tif" in n]
                    else:
                        self.mess_set("The cut_img_dir path in the configuration file does not exist, cannot read!", "Prompt", 1)
                        return False
                if swc_dir is None:
                    self.mess_set("Configuration file has no cut_swc_dir parameter, cannot read!", "Prompt", 1)
                    return False
                else:
                    if os.path.exists(swc_dir):
                        self.swc_lineEdit.setText(swc_dir)
                        self.swc_path = swc_dir
                    else:
                        self.mess_set("The cut_swc_dir path in the configuration file does not exist, cannot read!", "Prompt", 1)
                        return False
        else:
            self.mess_set("Configuration file does not exist!", "Prompt", 1)
            return False
        return True

    '''路径选择框确定按钮,image路径问题,获取文件名列表,路径列表'''
    @classmethod
    def read_path_question(cls, self):
        if self.image_lineEdit.text() == "" or self.swc_lineEdit.text() == "":
            self.mess_set("Path selection incomplete", 'Prompt', 1)
        else:
            self.filter_res["img_dir"] = self.image_lineEdit.text()
            self.filter_res["swc_dir"] = self.swc_lineEdit.text()
            self.filter_res["filenames"] = {}
            self.image_lineEdit.setText("")
            self.swc_lineEdit.setText("")
            # 初始化
            image_file_list_all = []
            image_filename_list = []
            swc_file_list = []
            image_file_list = []

            os.makedirs(self.swc_path, exist_ok=True)

            image_file_list_all = self.image_path_list

            if self.checkBox_zero.isChecked():
                for file in image_file_list_all:
                    swc_path = os.path.join(self.swc_path, f"{pp(file).stem}.swc")  # 对应的swcFile
                    if not pp(swc_path).exists():  # swcFile does not exist,创建swcFile
                        with open(swc_path, 'w') as f:
                            f.write("")
                    if os.path.getsize(swc_path):  # swc文件不为空
                        image_filename_list.append(pp(file).name)  # tifFile name
                        swc_file_list.append(swc_path)
                        image_file_list.append(file)
                        if not pp(swc_path).exists():  # swcFile does not exist,创建swcFile
                            with open(swc_path, 'w') as f:
                                f.write("")
            else:
                for file in image_file_list_all:
                    swc_path = os.path.join(self.swc_path, f"{pp(file).stem}.swc")  # 对应的swcFile
                    if not pp(swc_path).exists():  # swcFile does not exist,创建swcFile
                        with open(swc_path, 'w') as f:
                            f.write("")

                    image_filename_list.append(pp(file).name)  # tifFile name
                    swc_file_list.append(swc_path)
                    image_file_list.append(file)
                    if not pp(swc_path).exists():  # swcFile does not exist,创建swcFile
                        with open(swc_path, 'w') as f:
                            f.write("")

            if len(image_file_list):
                self.image_file_list_index = 0
                if self.dataImporter_index:  # 存在可标签数据,Update
                    self.selection = False
                self.image_file_list = image_file_list
                self.image_filename_list = image_filename_list
                self.swc_file_list = swc_file_list
                self.select_path_dialog.hide()
                self.read_file_path()
            else:
                self.mess_set("No images meet the criteria, please select again!", 'Prompt', 1)

    '''保存坐标'''
    @classmethod
    def save_sphere_pos(cls, self):
        info = self.mess_set("Save all labels?", 'Prompt', 2)
        if info == QtWidgets.QMessageBox.Yes:
            save_swc_path = self.swc_file_list[self.img_list.currentRow()]
            self.save_txt_swc(save_swc_path)
            self.mess_set(f"Saved successfully\n{save_swc_path}", 'Prompt', 1)

    '''Saveswc'''

    @classmethod
    def save_pos_swc(cls, save_swc_path, self):
        # save_swc_path = r"D:\CH2\CH2_test2\que_test\test.swc"
        # Saveswc
        # print("保存路径为:", save_swc_path)

        swc_lines = []
        count = 1
        # 已存所有点长度
        all_len = 0
        for points, lines in zip(self.vtk_Points, self.vtk_Lines):
            num_points = points.GetNumberOfPoints()
            pos_list = []
            for ii in range(num_points):
                coord = points.GetPoint(ii)
                pos_list.append([round(p, 2) for p in coord])
            # 遍历所有线段
            lines.InitTraversal()
            id_list = vtk.vtkIdList()
            pi = 0
            while lines.GetNextCell(id_list):
                # print("线段包含的点索引:", end=" ")
                pos = pos_list[pi]
                id_l = []
                for i in range(id_list.GetNumberOfIds()):
                    id_ = id_list.GetId(i)
                    id_l.append(id_)

                r = 0
                pos_key = self.pos_to_key(pos)
                r_ = self.radius_dict.get(pos_key)
                if r_:
                    r = r_

                if id_l[0] == 0 and id_l[1] == 1:
                    swc_lines.append(f"{count} {0} "
                                     f"{pos[0]:.2f} {pos[1]:.2f} {pos[2]:.2f} "
                                     f"{r:.4f} {-1}\n")
                    count += 1
                    pi += 1
                    pos = pos_list[pi]

                    r = 0
                    pos_key = self.pos_to_key(pos)
                    r_ = self.radius_dict.get(pos_key)
                    if r_:
                        r = r_

                    swc_lines.append(f"{count} {0} "
                                     f"{pos[0]:.2f} {pos[1]:.2f} {pos[2]:.2f} "
                                     f"{r:.4f} {id_l[1] + all_len}\n")
                else:
                    if id_l[0] + 1 != id_l[1]:
                        swc_lines.append(f"{count} {0} "
                                         f"{pos[0]:.2f} {pos[1]:.2f} {pos[2]:.2f} "
                                         f"{r:.4f} {id_l[0] + 1 + all_len}\n")
                    else:
                        swc_lines.append(f"{count} {0} "
                                         f"{pos[0]:.2f} {pos[1]:.2f} {pos[2]:.2f} "
                                         f"{r:.4f} {id_l[1] + all_len}\n")
                count += 1
                pi += 1
            all_len += len(pos_list)

        with open(save_swc_path, 'w') as f:
            f.writelines(swc_lines)

    '''筛选路径选择'''

    @classmethod
    def select_filter_path(cls, lineedit, self):
        try:
            # 创建QSettingsObject
            settings = QSettings("MyCompany", "MyApp")
            # 读取之前保存的路径信息
            initial_path = settings.value("VesselFilterPath", "")

            # 获取文件夹路径
            path = QtWidgets.QFileDialog.getExistingDirectory(None, "Select Folder", initial_path)

            # 如果选择了文件，保存选择的文件路径
            if path:
                if not cls.contains_chinese(path):
                    settings.setValue("VesselFilterPath", path)
                    lineedit.setText(path)  # 设置初始文本 -- 修改可视化数值
                else:
                    self.mess_set("Path contains Chinese characters or spaces", 'Prompt', 1)
            # path = lineedit.text()
            # return path
        except Exception as e:
            print(f"Open failed! {e}", "Prompt", 1)

    """完成筛选"""

    @classmethod
    def treat_filter_finish(cls, self):
        info = self.mess_set("Extract filtered files?", 'Prompt', 2)
        if info == QtWidgets.QMessageBox.Yes:
            if self.filter_res:
                filenames = self.filter_res["filenames"]
                if filenames:
                    old_img_dir = self.filter_res["img_dir"]
                    old_swc_dir = self.filter_res["swc_dir"]

                    new_res_dir = self.save_lineEdit.text()
                    if not os.path.exists(new_res_dir):
                        os.makedirs(new_res_dir, exist_ok=True)
                    new_res_dir = os.path.join(new_res_dir, "FilterResults")
                    if os.path.exists(new_res_dir):  # 如果存在筛选保存文件夹
                        info = self.mess_set(
                            "The filter folder already exists. Do you want to remove the existing files? (No: Keep the existing files and replace them)",
                            'Prompt', 2)
                        if info == QtWidgets.QMessageBox.Yes:
                            shutil.rmtree(new_res_dir)
                        os.makedirs(new_res_dir, exist_ok=True)

                    new_img_dir = os.path.join(new_res_dir, "images")
                    new_swc_dir = os.path.join(new_res_dir, "swc")
                    os.makedirs(new_img_dir, exist_ok=True)
                    os.makedirs(new_swc_dir, exist_ok=True)

                    for res, item in tqdm(filenames.items()):
                        res_pp = pp(res)
                        stem = str(res_pp.stem)
                        name = str(res_pp.name)
                        old_img_path = os.path.join(old_img_dir, name)
                        old_swc_path = os.path.join(old_swc_dir, stem + ".swc")
                        new_img_path = os.path.join(new_img_dir, name)
                        new_swc_path = os.path.join(new_swc_dir, stem + ".swc")
                        shutil.copy(old_img_path, new_img_path)
                        shutil.copy(old_swc_path, new_swc_path)

                        item.setForeground(QColor("green"))  # 字体颜色为绿色
                        text = item.text()
                        if "√ " in text:
                            text = text.replace("√ ", "")
                            item.setText(text)
                    print(f"Filtering completed! {len(filenames)}")
                else:
                    print(f"No filtered images to extract!")
        else:
            if self.filter_res:
                filenames = self.filter_res["filenames"]
                if filenames:
                    for res, item in filenames.items():
                        item.setForeground(QColor("green"))  # 字体颜色为绿色
                        text = item.text()
                        if "√ " in text:
                            text = text.replace("√ ", "")
                            item.setText(text)
        self.filter_res["filenames"] = {}
        self.save_lineEdit.setText("")
        self.filter_nums_label.setText("0")
