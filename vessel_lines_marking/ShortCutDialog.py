# -*- coding: utf-8 -*-
import sys
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QDialog, QLineEdit, QLabel, QMessageBox, QPushButton,
                             QSpacerItem, QSizePolicy, QFormLayout)
from PyQt5.QtGui import QKeySequence, QMouseEvent
from vessel_lines_marking.Ui_ShortCutDialog import Ui_ShortCutDialog
from vessel_lines_marking.ControlStyle import messagebox_style, button_style, lineedit_style, images_list_qdarkstyle


class CustomLineEdit(QLineEdit):
    keyPressed = pyqtSignal(QKeySequence)
    updateText = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)  # 设置为只读，防止用户直接编辑

    # def mouseDoubleClickEvent(self, event: QMouseEvent):
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.setReadOnly(False)  # 双击时取消只读状态
            self.selectAll()  # 选中所有文本，方便用户编辑
            self.setFocus()  # 焦点聚焦到输入框
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if not self.isReadOnly():
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self.setReadOnly(True)  # 按下回车键时恢复只读状态
                self.clearFocus()  # 取消焦点
                self.updateText.emit()
            else:
                if event.key() in (Qt.Key_Backspace, Qt.Key_Delete):
                    super().keyPressEvent(event)  # 调用父类的 keyPressEvent 以保持正常行为
                    return
                # 将 event.modifiers() 转换为整数
                modifiers = int(event.modifiers())
                key_sequence = QKeySequence(event.key() | modifiers)
                # 忽略默认的快捷键处理
                if key_sequence.matches(QKeySequence.Paste):
                    event.ignore()
                    return
                self.keyPressed.emit(key_sequence)
            super().keyPressEvent(event)  # 调用父类的 keyPressEvent 以保持正常行为

    def focusOutEvent(self, event):
        self.setReadOnly(True)  # 失去焦点时恢复只读状态
        super().focusOutEvent(event)


class ShortCutDialog(QDialog, Ui_ShortCutDialog):
    def __init__(self, win):
        super().__init__()

        # 使用setWindowModality方法设置对话框的模态性，可以防止对话框被其他窗口遮挡
        self.setWindowModality(Qt.ApplicationModal)

        self.hide_button = QPushButton()
        self.hide_button.setText("")
        self.hide_button.hide()

        self.win = win

        self.short_cuts_dict = self.win.short_cuts_dict
        self.mouse_short_cuts_dict = self.win.mouse_short_cuts_dict

        self.short_cuts_dict_copy = self.short_cuts_dict.copy()

        self.setupUi(self)

        self.textBrowser_2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 取消垂直滑块

        self.listWidget.setStyleSheet(images_list_qdarkstyle)
        self.listWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 取消水平滑块

        self.stackedWidget.setCurrentIndex(0)
        self.listWidget.setCurrentRow(0)

        self.listWidget.currentRowChanged.connect(self.OnItemChanged)

        self.RestoreKeybindingsButton.clicked.connect(self.restore_keybindings_clicked)
        self.CancelButton.clicked.connect(self.cancel_clicked)
        self.OkButton.clicked.connect(self.ok_clicked)

        self.RestoreKeybindingsButton.setStyleSheet(button_style)
        self.CancelButton.setStyleSheet(button_style)
        self.OkButton.setStyleSheet(button_style)

        self.create_grid_layout(self.short_cuts_dict)
        self.create_formLayout(self.mouse_short_cuts_dict)

    """恢复默认快捷方式"""

    def restore_keybindings_clicked(self):
        box = QMessageBox()
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setStyleSheet(messagebox_style)
        box.setText('Restore default shortcuts?')
        box.setWindowTitle('Question')
        box.setIcon(QMessageBox.Question)
        if box.exec_() == QMessageBox.Yes:
            self.update_button_line_text(self.short_cuts_dict)  # 更新按钮和文本框的文本
            # print("Restore")

    def cancel_clicked(self):
        self.update_button_line_text(self.short_cuts_dict_copy)  # 更新按钮和文本框的文本
        self.hide()

    def ok_clicked(self):
        self.print_grid_layout_rows()

    """获取当前快捷方式"""

    def print_grid_layout_rows(self):
        short_cuts_dict = {}
        field_text = []

        total = len(self.short_cuts_dict.keys())
        for i in range(1, total + 1):
            # 获取特定位置的控件
            label = self.shortcut_gridLayout.itemAtPosition(i, 0).widget()
            label_text = label.text()
            button = self.shortcut_gridLayout.itemAtPosition(i, 1).widget()
            button_text = button.text()

            field_text.append(button_text)
            short_cuts_dict[label_text] = button_text

        if len(field_text) == len(list(set(field_text))):
            # 无重复快捷方式, 更新快捷方式
            self.short_cuts_dict_copy = short_cuts_dict
            self.update_short_cut(self.short_cuts_dict_copy)  # 更新快捷方式
            self.update_button_line_text(self.short_cuts_dict_copy)  # 更新按钮和文本框的文本
            self.hide()
        else:
            box = QMessageBox()
            box.setStandardButtons(QMessageBox.Ok)
            box.setStyleSheet(messagebox_style)
            box.setText('Multiple actions have the same shortcut, please modify again!')
            box.setWindowTitle('Prompt')
            box.setIcon(QMessageBox.Question)
            box.exec_()

    # 初始化 shortcut_gridLayout
    def create_grid_layout(self, short_cuts_dict):
        title_action_label = QLabel(self)
        title_action_label.setText("Action")
        title_action_label.setStyleSheet("border-bottom: 2px solid rgb(70, 80, 100);"
                                         "min-width: 120px; min-height: 30px;")
        self.shortcut_gridLayout.addWidget(title_action_label, 0, 0)
        title_keybinding_label = QLabel(self)
        title_keybinding_label.setText("Keybinding")
        title_keybinding_label.setStyleSheet("border-bottom: 2px solid rgb(70, 80, 100);"
                                             "min-width: 120px; min-height: 30px;")
        self.shortcut_gridLayout.addWidget(title_keybinding_label, 0, 1)
        title_edit_label = QLabel(self)
        title_edit_label.setText("Edit")
        title_edit_label.setStyleSheet("border-bottom: 2px solid rgb(70, 80, 100);"
                                       "min-width: 150px; min-height: 30px;")
        self.shortcut_gridLayout.addWidget(title_edit_label, 0, 2)
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.shortcut_gridLayout.addItem(spacerItem, 0, 3)

        # 添加一些按钮
        i = 1
        for key, value in short_cuts_dict.items():
            label = QLabel(self)
            label.setText(key)
            button = QPushButton(self)
            button.setText(value)
            button.setStyleSheet(button_style)
            button.clicked.connect(self.clicked_button_show_line)
            button.setCheckable(True)  # 设置按钮为可选中状态
            button.setChecked(False)

            line = CustomLineEdit(self)
            line.setText(value)
            line.setStyleSheet(lineedit_style)
            line.keyPressed.connect(self.change_short_cut)
            line.updateText.connect(self.update_button_text)
            line.hide()  # 初始时隐藏文本框

            spacerItem = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Expanding)

            self.shortcut_gridLayout.addWidget(label, i, 0)
            self.shortcut_gridLayout.addWidget(button, i, 1)
            self.shortcut_gridLayout.addWidget(line, i, 2)
            self.shortcut_gridLayout.addItem(spacerItem, i, 3)
            i += 1

    def create_formLayout(self, mouse_short_cuts_dict):
        title_action_label = QLabel(self)
        title_action_label.setText("Action")
        title_action_label.setStyleSheet("border-bottom: 2px solid rgb(70, 80, 100);"
                                         "min-width: 160px; min-height: 30px;")
        title_keybinding_label = QLabel(self)
        title_keybinding_label.setText("Keybinding")
        title_keybinding_label.setStyleSheet("border-bottom: 2px solid rgb(70, 80, 100);"
                                             "min-width: 0px; min-height: 30px;")
        self.formLayout.setWidget(0, QFormLayout.LabelRole, title_action_label)
        self.formLayout.setWidget(0, QFormLayout.FieldRole, title_keybinding_label)

        i = 1
        for key, value in mouse_short_cuts_dict.items():
            label = QLabel(self)
            label.setText(key)
            filed = QLabel(self)
            filed.setText(value)
            self.formLayout.setWidget(i, QFormLayout.LabelRole, label)
            self.formLayout.setWidget(i, QFormLayout.FieldRole, filed)
            i += 1

    # 更新按钮和文本框的文本
    def update_button_line_text(self, short_cuts_dict):
        i = 1
        for key, value in short_cuts_dict.items():
            # 获取特定位置的控件
            button = self.shortcut_gridLayout.itemAtPosition(i, 1).widget()
            line = self.shortcut_gridLayout.itemAtPosition(i, 2).widget()
            button.setText(value)
            line.setText(value)
            i += 1

    # 按钮点击
    def clicked_button_show_line(self):
        # 点击按钮更新隐藏按钮文本
        self.hide_button.setText("")
        # 按钮触发
        sender = self.sender()
        if isinstance(sender, QPushButton):
            # 获取特定控件的位置
            index = self.shortcut_gridLayout.indexOf(sender)
            if index != -1:
                row, column, rowspan, colspan = self.shortcut_gridLayout.getItemPosition(index)
                # print(f"Control '{sender.text()}' 位于第 {row} 行，第 {column} Columns。")

                # 获取特定位置的控件
                line = self.shortcut_gridLayout.itemAtPosition(row, column + 1).widget()
                if line:
                    if sender.isChecked():  # 判断按钮是否被选中
                        line.setText(sender.text())
                        # print(f"获取到的控件: {line.text()}")  # PyQt5 中使用 text() 获取单行编辑栏文本
                        line.show()
                        line.setFocus()  # 设置文本框获取焦点
                    else:
                        line.hide()
            else:
                print("Control not found.")

    # 回车更新按钮文本
    def update_button_text(self):
        # 文本框触发
        sender = self.sender()
        if isinstance(sender, CustomLineEdit):
            # 获取特定控件的位置
            index = self.shortcut_gridLayout.indexOf(sender)
            if index != -1:
                row, column, rowspan, colspan = self.shortcut_gridLayout.getItemPosition(index)
                # print(f"Control '{sender.text()}' 位于第 {row} 行，第 {column} Columns。")

                # 获取特定位置的控件
                button = self.shortcut_gridLayout.itemAtPosition(row, column - 1).widget()
                if button:
                    if sender.isVisible():
                        text = self.hide_button.text()
                        if len(text):
                            button.setText(self.hide_button.text())  # 设置按钮文本为文本框内容
                        sender.hide()  # 隐藏文本框
                        button.setChecked(False)  # 取消按钮选中状态
            else:
                print("Control not found.")

    def update_short_cut(self, short_cuts_dict):
        self.win.update_user_short_cut(short_cuts_dict)
        # print(short_cuts_dict)

    """Option"""

    def OnItemChanged(self, index):
        self.stackedWidget.setCurrentIndex(index)

    """修改快捷方式"""

    def change_short_cut(self, key_sequence):
        # 替换无效的 Unicode 字符：使用列表推导式和 c.isprintable() 方法过滤掉不可打印的字符
        text = ''.join(c for c in key_sequence.toString() if c.isprintable())
        # 找到发送信号的对象（即触发快捷键的 CustomLineEdit）
        sender = self.sender()
        if isinstance(sender, CustomLineEdit):
            sender.setText(text)
            # print(text)
            self.hide_button.setText(text)

    # 关闭窗口
    def closeEvent(self, event):
        self.cancel_clicked()

    # 按键
    def keyPressEvent(self, event):
        if event.key() == 16777220:  # 判断是否按下回车键
            pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = ShortCutDialog()
    main_window.show()
    sys.exit(app.exec_())
