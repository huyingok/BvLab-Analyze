# -*- coding: utf-8 -*-
button_alpha_style = """
    QPushButton {
    background: rgb(70,80,100); 
    color: rgba(255,255,255,0.5); 
    border: 1px solid rgb(70,80,100); 
    border-radius: 0px; 
    padding: 2px 2px; 
    min-width: 60px;
    }
    QPushButton:hover {
    background-color: #379eff; 
    border: 1px solid #379eff;
    }
"""
button_style = """
    QPushButton {
    background: rgb(70,80,100); 
    color: rgb(255,255,255); 
    border: 1px solid rgb(70,80,100); 
    padding: 2px 2px; 
    border-radius: 0px; 
    min-width: 60px;
    }
    QPushButton:hover {
    background-color: #379eff; 
    border: 1px solid #379eff;
    }
"""
lineedit_alpha_style = """
    QLineEdit {
    border: 1px solid rgb(70,80,100); 
    color: rgba(255,255,255,0.5); 
    height: 22px;
    }
"""
lineedit_style = """
    QLineEdit {
    border: 1px solid rgb(70,80,100); 
    color: rgb(255,255,255); 
    height: 22px;
    border-radius: 0px;
    }
"""
checkBox_alpha_style = """
    QCheckBox {
    color: rgba(255,255,255,0.5); 
    }
"""
checkBox_style = """
    QCheckBox {
    color: rgb(255,255,255); 
    }
"""
messagebox_style = """
    QMessageBox {
    background-color: rgb(25, 35, 45); font: 9pt \"微软雅黑\"; color: rgb(255, 255, 255);
    }
    QLabel {
    color: rgb(255, 255, 255); font: 9pt \"微软雅黑\";
    }
    QPushButton { 
    background-color: rgb(70, 80, 100); font: 9pt \"微软雅黑\"; color: rgb(255, 255, 255);
    border: 1px solid rgb(70, 80, 100); padding: 3px 3px; min-width: 80px; height: 15px;
    }
    QPushButton:hover {
    background-color: #379eff; border: 1px solid #379eff;
    }
"""
spin_style = """
    QSpinBox {border-radius: 0px; height: 20px;}
    QSpinBox:hover { border-radius: 0px; }
    QSpinBox::up-button {border-radius: 0px; }
    QSpinBox::up-button:hover {border-radius: 0px; }
    QSpinBox::down-button {border-radius: 0px; }
    QSpinBox::down-button:hover {border-radius: 0px; }
"""
double_spin_qdarkstyle = """
    QDoubleSpinBox {border-radius: 0px; height: 20px;}
    QDoubleSpinBox:hover { border-radius: 0px; }
    QDoubleSpinBox::up-button {border-radius: 0px; }
    QDoubleSpinBox::up-button:hover {border-radius: 0px; }
    QDoubleSpinBox::down-button {border-radius: 0px; }
    QDoubleSpinBox::down-button:hover {border-radius: 0px; }
"""
comboBox_style = """
QComboBox {
    border-radius: 0px;
    height: 20px;
}
"""
label_style = """
    QToolTip {background-color: rgb(70,80,100); color: rgb(255,255,255); border: 1px solid rgb(70,80,100); }
"""
tool_button_style = """
    QPushButton {background-color: rgb(70,80,100); color: rgb(255,255,255); 
                 border: 1px solid rgb(70,80,100); min-width: 28px; height: 26px;}
    QPushButton:hover { background-color: #379eff; border: 1px solid #379eff; border-radius: 0px;}
    QPushButton:pressed {background-color: rgb(90, 100, 120); border: 1px solid rgb(90, 100, 120); }
    QToolTip {background-color: rgb(90, 100, 120); color: rgb(255,255,255); border: 1px solid rgb(90, 100, 120); }
"""
tool_bar_style = """
    QToolBar {background-color: rgb(70,80,100); border: 1px solid rgb(25,35,45); spacing: 10px; font: 11pt \"微软雅黑\";}
    QToolButton {background-color: rgb(70,80,100); color: rgb(255,255,255); border: 0px solid rgb(70,80,100); 
    font: 11pt \"微软雅黑\";}
    QToolButton:hover {background-color: #379eff; border: 1px solid #379eff;}
    QToolButton:pressed {background-color: #379eff; padding: 2px; border: 1px solid rgb(90, 100, 120); }
    QToolTip {background-color: rgb(90, 100, 120); color: rgb(255,255,255); border: 1px solid rgb(90, 100, 120); }
"""
fun_list_style = """
    QListWidget {font: 9pt "微软雅黑"; background-color: rgb(25, 35, 45); border-width: 1px; 
    border-style: solid; border-color: rgb(70, 80, 100); border-radius: 0px}
"""
fun_widget_style = """
    QWidget {background-color: rgb(25, 35, 45)}
"""
listwidget_qdarkstyle = """
    QListWidget {background-color: rgb(25, 35, 45); 
                 color: rgb(255, 255, 255); 
                 border: 1px solid rgb(70, 80, 100);border-radius: 0px;}
    QListWidget::item:selected {border-radius: 0px;background-color: #4C9C4C; 
        color: rgb(255, 255, 255); }
    QListWidget::item:hover{border-radius: 0px; background-color: #379eff; }
"""
tab_widget_style = """
    QTabWidget {font: 8pt \"微软雅黑\";}
    QTabWidget::pane {background-color: rgb(25, 35, 45); border-radius: 0px;}
    QTabBar {background-color: rgb(25, 35, 45); border-radius: 0px; border: 0px solid rgb(25, 35, 45);}
    QTabBar::tab { background-color: rgb(25, 35, 45); border: 1px solid rgb(25, 35, 45); color: rgb(255, 255, 255); 
    border-radius: 0px;padding: 3px;}
    QTabBar::tab:selected { background-color: #4C9C4C; border: 1px solid #4C9C4C; border-radius: 0px;padding: 3px;}
    QTabBar::tab:hover { background-color: #379eff; border: 1px solid #379eff; border-radius: 0px;padding: 3px;}
"""
text_edit_style = """
    QTextEdit {font: 9pt \"微软雅黑\";
    border: 1px solid rgb(70,80,100); 
    color: rgb(255,255,255); 
    border-radius: 0px;}
"""
table_view_style = """
    QTableView {font: 8pt \"微软雅黑\"; 
    background-color: rgb(25, 35, 45); 
    border-radius: 0px; 
    border: 1px solid rgb(70, 80, 100);}
"""
