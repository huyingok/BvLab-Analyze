# -*- coding: utf-8 -*-
button_style = """
    QPushButton {background: rgb(70,80,100); color: rgb(255,255,255); border: 1px solid rgb(70,80,100); 
    padding: 2px 2px; border-radius: 0px; font: 9pt \"微软雅黑\"; height: 20px;}
    QPushButton:hover { background-color: #379eff; border: 1px solid #379eff; border-radius: 0px;}
"""
lineedit_style = """
    QLineEdit {border: 1px solid rgb(70,80,100); height: 24px;}
"""
double_spin_qdarkstyle = """
    QDoubleSpinBox {border-radius: 0px; height: 15px;}
    QDoubleSpinBox:hover { border-radius: 0px; }
    QDoubleSpinBox::up-button {border-radius: 0px; }
    QDoubleSpinBox::up-button:hover {border-radius: 0px; }
    QDoubleSpinBox::down-button {border-radius: 0px; }
    QDoubleSpinBox::down-button:hover {border-radius: 0px; }
"""
spin_qdarkstyle = """
    QSpinBox {border-radius: 0px; height: 15px;}
    QSpinBox:hover { border-radius: 0px; }
    QSpinBox::up-button {border-radius: 0px; }
    QSpinBox::up-button:hover {border-radius: 0px; }
    QSpinBox::down-button {border-radius: 0px; }
    QSpinBox::down-button:hover {border-radius: 0px; }
"""
table_widget_qdarkstyle = """
    QTableWidget { background-color: rgb(25, 35, 45); color: rgb(25, 35, 45); border: rgb(70, 80, 100);
            border-width: 1px; border-style: solid; border-radius: 0px;}
    QTableWidget::item { background-color: rgb(25, 35, 45); color: rgb(255, 255, 255); 
            border: rgb(70, 80, 100); border-width: 1px; border-style: solid; border-radius: 0px;}
    QTableWidget::item:hover { background-color: #379eff; border-radius: 0px;}     
    QTableWidget::item:selected { background-color: #4C9C4C; color: rgb(255, 255, 255); border-radius: 0px;}  
    QHeaderView::section { background-color: rgb(25, 35, 45); color: rgb(255, 255, 255); border-radius: 0px;}
"""
z_list_qdarkstyle = """
    QListWidget {background-color: rgb(25, 35, 45); 
                 color: rgb(255, 255, 255); 
                 border: 1px solid rgb(70, 80, 100);border-radius: 0px;}
    QListWidget::item:selected {border-radius: 0px;background-color: #4C9C4C; 
        color: rgb(255, 255, 255); }
    QListWidget::item:hover{border-radius: 0px; background-color: #379eff; }
"""
color_dialog_style = """
    QDialog {background-color: rgb(70, 80, 100); color: rgb(255, 255, 255); font: 9pt \"微软雅黑\";}
"""
messagebox_style = """
    QMessageBox { background-color: rgb(25, 35, 45); font: 9pt \"微软雅黑\"; color: rgb(255, 255, 255);}
    QLabel {color: rgb(255, 255, 255); font: 9pt \"微软雅黑\";}
    QPushButton { background-color: rgb(70, 80, 100); font: 9pt \"微软雅黑\"; color: rgb(255, 255, 255);
    border: 1px solid rgb(70, 80, 100); padding: 3px 3px; min-width: 80px; height: 15px;}
    QPushButton:hover { background-color: #379eff; border: 1px solid #379eff; }
"""
slider_style = """
QSlider::groove:horizontal {
    height: 8px;
    left: 8px;
    right: 8px;
    background-color: rgb(70, 80, 100);
    border-radius: 4px;
}
QSlider::handle:horizontal {
    width: 20px;
    margin: -6px -6px;
    background-color: #4C9C4C;
    border-radius: 9px;
    border: 1px;
    border-color: rgb(70, 80, 100);
}
QSlider::handle:horizontal:hover {
    width: 20px;
    margin: -6px -6px;
    background-color: #379eff;
    border-radius: 9px;
    border: 1px;
    border-color: rgb(70, 80, 100);
}
"""
tool_button_style = """
    QPushButton {background-color: rgb(70,80,100); color: rgb(255,255,255); 
                 border: 1px solid rgb(70,80,100); width: 28px; height: 28px;}
    QPushButton:hover { background-color: #379eff; border: 1px solid #379eff; border-radius: 0px;}
    QPushButton:pressed {background-color: rgb(90, 100, 120); border: 1px solid rgb(90, 100, 120); }
    QToolTip {background-color: rgb(90, 100, 120); color: rgb(255,255,255); border: 1px solid rgb(90, 100, 120); }
"""
tool_bar_style = """
    QToolBar {background-color: rgb(70,80,100); border-top: 1px solid rgb(25,35,45); 
    border-bottom: 1px solid rgb(25,35,45); spacing: 10px; }
    QToolButton {background-color: rgb(70,80,100); color: rgb(255,255,255); 
    border: 0px solid rgb(70,80,100); }
    QToolButton:hover {background-color: rgb(90, 100, 120); }
    QToolButton:pressed {background-color: rgb(90, 100, 120); padding: 2px; border: 1px solid rgb(90, 100, 120); }
    QToolTip {background-color: rgb(90, 100, 120); color: rgb(255,255,255); border: 1px solid rgb(90, 100, 120); }
"""
tab_widget_style = """
    QTabWidget {font: 8pt "微软雅黑";}
    QTabWidget::pane {background-color: rgb(25, 35, 45); border-radius: 0px;}
    QTabBar {background-color: rgb(25, 35, 45); border-radius: 0px; border: 0px solid rgb(25, 35, 45);}
    QTabBar::tab { background-color: rgb(25, 35, 45); border: 1px solid rgb(25, 35, 45); color: rgb(255, 255, 255); 
    border-radius: 0px;padding: 3px;}
    QTabBar::tab:selected { background-color: #4C9C4C; border: 1px solid #4C9C4C; border-radius: 0px;padding: 3px;}
    QTabBar::tab:hover { background-color: #379eff; border: 1px solid #379eff; border-radius: 0px;padding: 3px;}
"""
layer_list_style = fun_list_style = """
QListWidget {font: 9pt "微软雅黑"; background-color: rgb(25, 35, 45); border-width: 1px; 
border-style: solid; border-color: rgb(70, 80, 100); border-radius: 0px;}
"""
layer_widget_style = fun_widget_style = """
QWidget {background-color: rgb(25, 35, 45);}
"""
images_list_qdarkstyle = """
    QListWidget {background-color: rgb(25, 35, 45); color: rgb(255, 255, 255); border: 1px solid rgb(70, 80, 100);
    border-radius: 0px;padding: 2px;}
    QListWidget::item:selected {background-color: #4C9C4C; border-radius: 0px; color: rgb(255, 255, 255);}
    QListWidget::item:hover {border-radius: 0px; background-color: #379eff; }
"""
dock_widget_qdarkstyle = """
    QDockWidget { font: 9pt "微软雅黑";background: rgb(25, 35, 45); color: rgb(255, 255, 255); border-radius: 0px;
    border:none;}
    QDockWidget::title {background: rgb(70, 80, 100); padding-top: 6px; padding-bottom: 6px; 
    border-radius: 0px;}
    QDockWidget::close-button, QDockWidget::float-button {background: rgb(70, 80, 100); }
"""
menubar_style = """
    QMenuBar { background-color: rgb(70, 80, 100); color: rgb(255, 255, 255); padding: 0px; 
    height:30px;}
    QMenuBar:focus { border: 0px solid rgb(70, 80, 100);}
    QMenuBar::item { padding: 4px; }
    QMenuBar::item:selected { background-color: #379eff; padding: 4px;}
    QMenuBar::item:pressed { padding: 4px; border: 0px solid #379eff; background-color: #1A72BB;
    color: #E0E1E3; margin-bottom: 0px; padding-bottom: 0px;}
"""
menu_style = """
    QMenu { background-color: rgb(70, 80, 100); color: rgb(255, 255, 255); 
            padding-left: 0px; padding-right: 0px;}
    QMenu::item {border:4px solid transparent; }
    QMenu::item:selected { background-color: #379eff; padding-top: 3px;}
"""
titleLabel_style = """
    QLabel {color: rgb(255, 255, 255); font: 9pt "微软雅黑";}
"""
sphere_button_show_style = """
    QPushButton {background-color: red; width: 20px; height: 20px; border-radius: 0px;}
"""
sphere_button_hide_style = """
    QPushButton {background-color: white; width: 20px; height: 20px; border-radius: 0px;}
"""
image_button_show_style = """
    QPushButton {background-color: red; width: 20px; height: 20px; border-radius: 0px;}
"""
image_button_hide_style = """
    QPushButton {background-color: white; width: 20px; height: 20px; border-radius: 0px;}
"""
openColorDialog_button_style = """
    QPushButton{background-color: rgb(255, 255, 255); color: rgb(25, 35, 45);
    border-radius: 0px; width: 20px; height: 20px;}
    QPushButton:hover { background-color: #379eff; }
"""
MarkColorDialog_button_style = """
    QPushButton{background-color: rgb(0, 255, 0); color: rgb(25, 35, 45);
    border-radius: 0px; width: 20px; height: 20px;}
    QPushButton:hover { background-color: #379eff; }
"""
# 下拉框样式
comboBox_style = """
QComboBox {
    font: 8pt \"微软雅黑\";
    border-radius: 0px;
    height: 18px;
    min-width: 80px;
    padding: 1px 3px 1px 3px;
}
"""
# 样式
StyleSheet = """
/*标题栏*/
CustomTitleBar {
    background-color: rgb(70, 80, 100);
}
/*最小化最大化关闭按钮通用默认背景*/
#buttonMinimum,#buttonMaximum,#buttonClose {
    border: none;
    background-color: rgb(70, 80, 100);
}
/*Hover*/
#buttonMinimum:hover,#buttonMaximum:hover {
    background-color: rgb(90, 100, 120);
    color: white;
    padding-top: 5px;
}
#buttonClose:hover {    
    color: white;
    background-color: Firebrick;
    padding-top: 5px;
}
/*鼠标按下不放*/
#buttonMinimum:pressed,#buttonMaximum:pressed {
    color: white;
}
#buttonClose:pressed {
    color: white;
    background-color: Firebrick;
}
"""
