# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QImage, QPixmap


class CustomListItem(QWidget):
    def __init__(self, text, list_widget, list_item, img=None, seg=None, parent=None):
        super(CustomListItem, self).__init__(parent)
        self.setStyleSheet("QWidget { background-color: rgb(55, 65, 79)};")
        self.list_widget = list_widget  # 存储 QListWidget 的引用
        self.list_item = list_item  # 存储 QListWidgetItem 的引用

        self.progress_text_label = QLabel(" " + text + " ")
        self.progress_text_label.setFont(QFont("微软雅黑", 12))
        self.progress_text_label.setMinimumWidth(50)
        # self.progress_text_label.setWordWrap(True)  # 自动换行
        if img is not None:
            self.img_text_label = QLabel("Origin:")
            self.img_text_label.setFont(QFont("微软雅黑", 12))
            self.img_text_label.setAlignment(Qt.AlignVCenter | Qt.AlignHCenter)
            # self.img_text_label.setWordWrap(True)

            self.img_label = QLabel()
            self.img_label.setMinimumWidth(80)
            self.img_label.setPixmap(self.set_pixmap(img))
        if seg is not None:
            self.seg_text_label = QLabel("Predict:")
            self.seg_text_label.setFont(QFont("微软雅黑", 12))
            self.seg_text_label.setAlignment(Qt.AlignVCenter | Qt.AlignHCenter)
            # self.seg_text_label.setWordWrap(True)

            self.seg_label = QLabel()
            self.seg_label.setMinimumWidth(80)
            self.seg_label.setPixmap(self.set_pixmap(seg))

        self.spacing_label = QLabel()
        if img is not None:
            self.spacing_label.setMinimumWidth(400)

        layout = QHBoxLayout()
        layout.addWidget(self.progress_text_label)
        if img is not None:
            layout.addWidget(self.img_text_label)
            layout.addWidget(self.img_label)
        if seg is not None:
            layout.addWidget(self.seg_text_label)
            layout.addWidget(self.seg_label)
        layout.addWidget(self.spacing_label)
        # layout.addStretch()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

    def set_pixmap(self, image_data):
        # 将 numpy 数组转换为 QImage
        height, width = image_data.shape
        qImg = QImage(image_data.data, width, height, width, QImage.Format_Grayscale8)
        # 将 QImage Convert To QPixmap 并设置到 QLabel
        pixmap = QPixmap.fromImage(qImg)
        # 调整 QPixmap 的尺寸
        scaled_pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio)  # 调整为 100，保持纵横比
        return scaled_pixmap


class CustomListItem2(QWidget):
    def __init__(self, text, list_widget, list_item, img=None, mask=None, pre=None, parent=None):
        super(CustomListItem2, self).__init__(parent)
        self.setStyleSheet("QWidget { background-color: rgb(55, 65, 79)};")
        self.mask_label = None
        self.img_label = None
        self.pre_label = None
        self.list_widget = list_widget  # 存储 QListWidget 的引用
        self.list_item = list_item  # 存储 QListWidgetItem 的引用
        self.progress_text_label = QLabel(" " + text + " ")
        self.progress_text_label.setFont(QFont("微软雅黑", 12))
        self.progress_text_label.setMinimumWidth(50)
        # self.progress_text_label.setWordWrap(True)  # 自动换行

        if img is not None:
            self.img_text_label = QLabel("Origin:")
            self.img_text_label.setFont(QFont("微软雅黑", 12))
            self.img_text_label.setAlignment(Qt.AlignVCenter | Qt.AlignHCenter)
            # self.img_text_label.setWordWrap(True)

            self.img_label = QLabel()
            self.img_label.setMinimumWidth(80)
            self.img_label.setPixmap(self.set_pixmap(img))

        if mask is not None:
            self.mask_text_label = QLabel("Label:")
            self.mask_text_label.setFont(QFont("微软雅黑", 12))
            self.mask_text_label.setAlignment(Qt.AlignVCenter | Qt.AlignHCenter)
            # self.mask_text_label.setWordWrap(True)

            self.mask_label = QLabel()
            self.mask_label.setMinimumWidth(80)
            self.mask_label.setPixmap(self.set_pixmap(mask))

            self.spacing_label = QLabel()
            self.spacing_label.setMinimumWidth(400)

        if pre is not None:
            self.pre_text_label = QLabel("Predict:")
            self.pre_text_label.setFont(QFont("微软雅黑", 12))
            self.pre_text_label.setAlignment(Qt.AlignVCenter | Qt.AlignHCenter)

            self.pre_label = QLabel()
            self.pre_label.setMinimumWidth(80)
            self.pre_label.setPixmap(self.set_pixmap(pre))

        layout = QHBoxLayout()

        if pre is None:
            layout.addWidget(self.progress_text_label)
            if self.img_label is not None:
                layout.addWidget(self.img_text_label)
                layout.addWidget(self.img_label)
            if self.mask_label is not None:
                layout.addWidget(self.mask_text_label)
                layout.addWidget(self.mask_label)
                layout.addWidget(self.spacing_label)
                # layout.addStretch()
        else:
            if self.img_label is not None:
                layout.addWidget(self.img_text_label)
                layout.addWidget(self.img_label)
            if self.mask_label is not None:
                layout.addWidget(self.mask_text_label)
                layout.addWidget(self.mask_label)
                layout.addWidget(self.pre_text_label)
                layout.addWidget(self.pre_label)
                layout.addWidget(self.spacing_label)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

    def set_pixmap(self, image_data):
        # 将 numpy 数组转换为 QImage
        height, width = image_data.shape
        qImg = QImage(image_data.data, width, height, width, QImage.Format_Grayscale8)
        # 将 QImage Convert To QPixmap 并设置到 QLabel
        pixmap = QPixmap.fromImage(qImg)
        # 调整 QPixmap 的尺寸
        scaled_pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio)  # 调整为 80x80，保持纵横比
        return scaled_pixmap
