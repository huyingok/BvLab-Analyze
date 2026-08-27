import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTableView, QComboBox, QLabel, QTextEdit, QPushButton,
    QGroupBox, QRadioButton, QButtonGroup, QFileDialog, QMessageBox,
    QTabWidget, QCheckBox, QLineEdit, QInputDialog, QFrame
)
import qdarkstyle
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QSettings
from PyQt5.QtGui import QFont
from control_style.ControlStyle import (button_style, checkBox_style, comboBox_style,
                                        tab_widget_style, text_edit_style, table_view_style)
from DataStatistics.ui_create_chart import Ui_MainWindow


# Set Global Font to Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.family'] = 'Microsoft YaHei'  # '微软雅黑'


class PandasModel(QAbstractTableModel):
    """将 pandas DataFrame Adapt to Qt Table Model"""
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, parent=QModelIndex()):
        return self._data.shape[0]

    def columnCount(self, parent=QModelIndex()):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            if pd.isna(value):
                return ''
            if isinstance(value, (float, np.float64)):
                return f'{value:.4f}'
            return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._data.columns[section])
        else:
            return str(self._data.index[section])


class DataAnalyzer:
    def __init__(self):
        self.df = None
        self.basic_cols = []       # Basic Metric Column Names（不含File Name）
        self.bin_count_cols = []    # Binning count column names
        self.bin_length_cols = []   # Binning length column names
        self.bin_tort_cols = []     # Binning tortuosity column names
        self.group_col = 'Group'

    def load_excel(self, filepath):
        """加载ExcelFile，Adaptive Column Count"""
        try:
            # Read All Data，No Header，Preserve Original Format
            df_raw = pd.read_excel(filepath, header=None, dtype=str)  # Read All as String First，Avoid Conversion Issues
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Unable to read Excel file:\n{str(e)}")
            return False

        # Remove completely empty rows and columns
        df_raw = df_raw.dropna(how='all').dropna(axis=1, how='all')
        if df_raw.empty:
            QMessageBox.critical(None, "Error", "Excel file is empty")
            return False

        # Show First Rows for Debugging（可注释掉）
        # print("原始数据前5行:")
        # print(df_raw.head())

        # Try to Find Data Start Row（第一列包含"File Name"的行）
        start_row = None
        for i in range(min(10, len(df_raw))):
            cell = str(df_raw.iloc[i, 0]) if pd.notna(df_raw.iloc[i, 0]) else ''
            if 'File Name' in cell:
                start_row = i
                break

        if start_row is None:
            # Manual inquiry
            reply = QMessageBox.question(None, "Cannot locate header",
                                         "Cannot find 'File Name' row automatically. Input data start row manually? (starting from 0)?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                row_str, ok = QInputDialog.getText(None, "Input row number", "Enter Data Start Row (first row is 0):")
                if ok and row_str.isdigit():
                    start_row = int(row_str)
                else:
                    return False
            else:
                return False

        # Header Should Be At start_row 行，This Row Contains Basic Column Names
        header_row = start_row
        # Next Row May Be Unit or Description，但这里我们忽略，Use Directly header_row As Column Names
        # 但原Excel中，第1行是"Main Results"，第2行是子标题，第3行是列名+Data。我们需要根据实际情况调整
        # Actually，提供的文本中第1行是空，第2行有"Main Results"，第3行是"File Name","Volume",...
        # 所以我们假设 start_row 就是列名所在行
        col_names_raw = df_raw.iloc[header_row].tolist()
        # Clean Column Names：Remove Leading and Trailing Spaces，Replace Null Values With'Unnamed'
        col_names = []
        for idx, name in enumerate(col_names_raw):
            if pd.isna(name) or str(name).strip() == '':
                col_names.append(f'Unnamed_{idx}')
            else:
                col_names.append(str(name).strip())

        # Data Starts From header_row+1 Start
        data_rows = df_raw.iloc[header_row+1:].copy()
        # Reset index
        data_rows.reset_index(drop=True, inplace=True)

        # Set column names
        data_rows.columns = col_names

        # Now we need to identify basic metric columns and binning blocks
        # Basic Metric Columns：从 File Name 到 Mean Segment Surface Area，通常是前13Columns
        # But Column Count May Change Due to Merged Cells，We Can Judge Based on Column Name Content
        basic_indicators = [
            'File Name', 'Volume', 'Network Length', 'Surface Area',
            'Branchpoints', 'Endpoints', 'Number of Segments',
            'Segment Partitioning', 'Mean Segment Radius',
            'Mean Segment Length', 'Mean Segment Tortuosity',
            'Mean Segment Volume', 'Mean Segment Surface Area'
        ]

        # Find These Columns In col_names Index In
        basic_indices = []
        for name in basic_indicators:
            found = False
            for i, col in enumerate(col_names):
                if name in col:  # Partial match
                    basic_indices.append(i)
                    found = True
                    break
            if not found:
                # If Not Found，Assume Column Order is Consistent，Take By Position
                # Simple Processing：取前13Columns
                basic_indices = list(range(min(13, len(col_names))))
                break
        if not basic_indices:
            # 取前13Columns
            basic_indices = list(range(min(13, len(col_names))))

        # Ensure basic_indices Are Unique and Continuous
        basic_indices = sorted(set(basic_indices))
        # 如果少于13，Fill With Subsequent Columns？But May Cause Misalignment，Better Let User Choose
        if len(basic_indices) < 13:
            QMessageBox.warning(None, "Warning",
                                f"Only found {len(basic_indices)} basic metric columns, expected 13. Please check Excel format; some columns may be missing. Will process with existing columns.")
            # 补全到13Columns，用None占位
            while len(basic_indices) < 13:
                basic_indices.append(None)

        # Extract basic column names
        actual_basic_names = [col_names[i] if i is not None else f'Missing_{j}' for j, i in enumerate(basic_indices)]

        # Remaining columns as binning related columns
        remaining_indices = [i for i in range(len(col_names)) if i not in basic_indices if i is not None]
        remaining_names = [col_names[i] for i in remaining_indices]

        # Number of Binning Intervals：通常是21（0-1到20+）
        # We Assume Binning Blocks Are Three Consecutive Blocks：计数、Length、扭曲度，每块21Columns
        # But Only Part May Be Available，We Can Dynamically Allocate Based on Actual Column Count
        # Prioritize Identifying Keywords in Binning Column Names（Count, Length, Tortuosity）
        count_indices = []
        length_indices = []
        tort_indices = []
        for idx, name in zip(remaining_indices, remaining_names):
            lower = name.lower()
            if 'count' in lower or 'number' in lower:
                count_indices.append(idx)
            elif 'length' in lower:
                length_indices.append(idx)
            elif 'tort' in lower:
                tort_indices.append(idx)
            else:
                # Unrecognizable，May Have No Keywords，Allocate In Order
                pass

        # If Number of Recognized Blocks is Incorrect，Then Divide Equally In Order（假设每块21Columns）
        bin_size = 21
        if len(remaining_indices) >= 3 * bin_size:
            # Enough for Three Blocks，Split In Order
            count_indices = remaining_indices[:bin_size]
            length_indices = remaining_indices[bin_size:2*bin_size]
            tort_indices = remaining_indices[2*bin_size:3*bin_size]
        elif len(remaining_indices) >= 2 * bin_size:
            # Only Two Blocks，Assume Count and Length
            count_indices = remaining_indices[:bin_size]
            length_indices = remaining_indices[bin_size:2*bin_size]
            tort_indices = []
        elif len(remaining_indices) >= bin_size:
            # Only One Block，Assume Count
            count_indices = remaining_indices[:bin_size]
            length_indices = []
            tort_indices = []
        else:
            # No binning columns
            count_indices = length_indices = tort_indices = []

        # Construct binning interval labels
        bin_labels = [f'{i}-{i+1}' for i in range(20)] + ['20+']

        # Generate binning column names
        bin_count_names = [f'Count_{bin_labels[i]}' for i in range(len(count_indices))]
        bin_length_names = [f'Length_{bin_labels[i]}' for i in range(len(length_indices))]
        bin_tort_names = [f'Tortuosity_{bin_labels[i]}' for i in range(len(tort_indices))]

        # Combine all column names
        all_names = actual_basic_names + bin_count_names + bin_length_names + bin_tort_names
        # If Column Counts Do Not Match，Need to Adjust data_rows Column Count
        if len(all_names) != data_rows.shape[1]:
            QMessageBox.warning(None, "Warning",
                                f"Column Count Mismatch：Expected {len(all_names)} Columns，Actual {data_rows.shape[1]} Columns。"
                                "Will Truncate or Pad to Minimum Column Count。")
            min_cols = min(len(all_names), data_rows.shape[1])
            data_rows = data_rows.iloc[:, :min_cols]
            all_names = all_names[:min_cols]

        data_rows.columns = all_names

        # Convert Basic Metric Columns to Numeric Type（File NameExcluding）
        for col in all_names:
            if col != 'File Name' and col not in bin_count_names+bin_length_names+bin_tort_names:
                data_rows[col] = pd.to_numeric(data_rows[col], errors='coerce')
            elif col in bin_count_names+bin_length_names+bin_tort_names:
                data_rows[col] = pd.to_numeric(data_rows[col], errors='coerce')

        # Ensure File Name Column as String
        if 'File Name' in data_rows.columns:
            data_rows['File Name'] = data_rows['File Name'].astype(str)
        else:
            data_rows.insert(0, 'File Name', 'Unknown')

        # Add group column
        if data_rows['File Name'].str.contains(r'\(.*?\)').all():
            # str.extract(r'\((.*?)\)') Use Regex to Extract Content in Parentheses
            # \( 和 \) Match Parentheses
            # (.*?) Is a Capture Group，Non-greedy Match Any Character in Parentheses
            data_rows[self.group_col] = data_rows['File Name'].str.extract(r'\((.*?)\)')
        else:
            data_rows[self.group_col] = data_rows['File Name'].str[:4]

        self.df = data_rows
        self.basic_cols = [c for c in all_names if c not in bin_count_names+bin_length_names+bin_tort_names and c != 'File Name' and c != self.group_col]
        self.bin_count_cols = bin_count_names
        self.bin_length_cols = bin_length_names
        self.bin_tort_cols = bin_tort_names

        return True

    def get_numeric_columns(self):
        if self.df is None:
            return []
        return [col for col in self.df.columns if col not in ['File Name', self.group_col] and
                pd.api.types.is_numeric_dtype(self.df[col])]

    def describe(self, column, group_by=None):
        if self.df is None or column not in self.df.columns:
            return None
        data = self.df[column].dropna()
        if group_by and group_by in self.df.columns:
            stats = self.df.groupby(group_by)[column].describe(percentiles=[.25, .5, .75])
            return stats
        else:
            stats = data.describe(percentiles=[.25, .5, .75])
            return stats

    def get_grouped_data(self, column, group_col):
        if self.df is None or column not in self.df.columns or group_col not in self.df.columns:
            return [], []
        groups = self.df[group_col].dropna().unique()
        data = []
        labels = []
        for g in sorted(groups):
            vals = self.df[self.df[group_col] == g][column].dropna()
            if not vals.empty:
                data.append(vals.values)
                labels.append(str(g))
        return data, labels


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.analyzer = DataAnalyzer()
        self.qdarkstyle_sheet = qdarkstyle.load_stylesheet(qt_api='pyqt5')
        self.initUI()

    def initUI(self):
        self.setupUi(self)

        # self.setWindowTitle("3D Dataset Statistical Analysis Tool")
        # self.setGeometry(100, 100, 600, 500)
        # self.central = QWidget()
        # self.setCentralWidget(self.central)
        # main_layout = QHBoxLayout(self.central)
        #
        # # Left control panel
        # control_panel = QWidget()
        # control_layout = QVBoxLayout(control_panel)
        # control_panel.setMaximumWidth(300)
        #
        # file_group = QGroupBox("Data file")
        # file_layout = QVBoxLayout()
        # self.load_btn = QPushButton("加载ExcelFile")
        # self.load_btn.setStyleSheet(button_style)
        # self.load_btn.clicked.connect(self.load_file)
        # file_layout.addWidget(self.load_btn)
        # self.file_label = QLabel("No file loaded")
        # self.file_label.setWordWrap(True)
        # file_layout.addWidget(self.file_label)
        # file_group.setLayout(file_layout)
        # control_layout.addWidget(file_group)
        #
        # var_group = QGroupBox("Variable selection")
        # var_layout = QVBoxLayout()
        # var_layout.addWidget(QLabel("Select Numeric Column:"))
        # self.var_combo = QComboBox()
        # self.var_combo.setStyleSheet(self.qdarkstyle_sheet + comboBox_style)
        # self.var_combo.currentTextChanged.connect(self.update_analysis)
        # var_layout.addWidget(self.var_combo)
        # self.group_check = QCheckBox("Group by Group (GroupColumns)")
        # self.group_check.setStyleSheet(checkBox_style)
        # self.group_check.stateChanged.connect(self.update_analysis)
        # var_layout.addWidget(self.group_check)
        # var_group.setLayout(var_layout)
        # control_layout.addWidget(var_group)
        #
        # plot_group = QGroupBox("Plot type")
        # plot_layout = QVBoxLayout()
        # self.plot_type_group = QButtonGroup(self)
        # self.hist_radio = QRadioButton("Histogram")
        # self.hist_radio.setChecked(True)
        # self.box_radio = QRadioButton("Box plot")
        # self.plot_type_group.addButton(self.hist_radio)
        # self.plot_type_group.addButton(self.box_radio)
        # plot_layout.addWidget(self.hist_radio)
        # plot_layout.addWidget(self.box_radio)
        # plot_group.setLayout(plot_layout)
        # control_layout.addWidget(plot_group)
        #
        # self.refresh_btn = QPushButton("Refresh analysis")
        # self.refresh_btn.setStyleSheet(button_style)
        # self.refresh_btn.clicked.connect(self.update_analysis)
        # control_layout.addWidget(self.refresh_btn)
        #
        # control_layout.addStretch()
        #
        # # Right area
        # analyze_view = QWidget()
        # analyze_view_layout = QHBoxLayout(analyze_view)
        # right_splitter = QSplitter(Qt.Vertical)
        # self.table_view = QTableView()
        # right_splitter.addWidget(self.table_view)
        #
        # bottom_tab = QTabWidget()
        # self.figure = Figure()
        # self.canvas = FigureCanvas(self.figure)
        # bottom_tab.addTab(self.canvas, "Graph")
        # self.stats_text = QTextEdit()
        # self.stats_text.setFont(QFont("Courier", 10))
        # self.stats_text.setReadOnly(True)
        # bottom_tab.addTab(self.stats_text, "Statistical results")
        # right_splitter.addWidget(bottom_tab)
        # # right_splitter.setSizes([400, 400])
        # right_splitter.setHandleWidth(5)
        #
        # analyze_view_layout.addWidget(right_splitter)
        #
        # mid_splitter = QSplitter(Qt.Horizontal)
        # mid_splitter.setHandleWidth(2)
        #
        # mid_splitter.addWidget(control_panel)
        # mid_splitter.addWidget(analyze_view)
        #
        # main_layout.addWidget(mid_splitter)

        self.file_label.setWordWrap(True)  # Set word wrap

        self.load_btn.setStyleSheet(button_style)
        self.load_btn.clicked.connect(self.load_file)
        self.var_combo.setStyleSheet(self.qdarkstyle_sheet + comboBox_style)
        self.var_combo.currentTextChanged.connect(self.update_analysis)
        self.group_check.setStyleSheet(checkBox_style)
        self.group_check.setStyleSheet(self.qdarkstyle_sheet)
        self.group_check.stateChanged.connect(self.update_analysis)
        self.refresh_btn.setStyleSheet(button_style)
        self.refresh_btn.clicked.connect(self.update_analysis)

        self.hist_radio.setStyleSheet(self.qdarkstyle_sheet)
        self.box_radio.setStyleSheet(self.qdarkstyle_sheet)

        bottom_tab = QTabWidget()
        bottom_tab.setStyleSheet(self.qdarkstyle_sheet + tab_widget_style)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        bottom_tab.addTab(self.canvas, "Graph")

        self.stats_table = QTableView()
        self.stats_table.setAlternatingRowColors(True)
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        self.stats_table.setStyleSheet(table_view_style)  # If there is style
        bottom_tab.addTab(self.stats_table, "Statistical results")

        # self.stats_text = QTextEdit()
        # self.stats_text.setStyleSheet(text_edit_style)
        # self.stats_text.setReadOnly(True)
        # bottom_tab.addTab(self.stats_text, "Statistical results")

        # right_splitter = QSplitter(Qt.Vertical)
        # right_splitter.setHandleWidth(5)
        # right_splitter.addWidget(self.table_view)
        # right_splitter.addWidget(bottom_tab)
        # self.analyze_view_layout.addWidget(right_splitter)

        self.analyze_view_layout.addWidget(bottom_tab)
        bottom_tab.setCurrentIndex(1)

        mid_splitter = QSplitter(Qt.Horizontal)
        mid_splitter.setHandleWidth(5)
        mid_splitter.addWidget(self.control_panel)
        mid_splitter.addWidget(self.analyze_view)
        # Prevent child windows from being completely hidden
        mid_splitter.setChildrenCollapsible(False)
        self.main_layout.addWidget(mid_splitter)

    def load_file(self):
        # 创建QSettingsObject
        settings = QSettings("MyCompany", "MyApp")
        initial_path = settings.value("VesselXlsxPath", "")
        filepath, _ = QFileDialog.getOpenFileName(self,
                                                  "SelectExcelFile",
                                                  initial_path,
                                                  "ExcelFile (*.xlsx *.xls)")
        if filepath:
            settings.setValue("VesselXlsxPath", str(Path(filepath).parent))
        self.read_analyze_file(filepath)

    def read_analyze_file(self, filepath):
        if filepath:
            if self.analyzer.load_excel(filepath):
                self.file_label.setText(f"Loaded: {Path(filepath).name}")
                self.var_combo.clear()
                num_cols = self.analyzer.get_numeric_columns()
                self.var_combo.addItems(num_cols)
                # model = PandasModel(self.analyzer.df)
                # self.table_view.setModel(model)
                # self.table_view.resizeColumnsToContents()

                from PyQt5.QtCore import QTimer
                QTimer.singleShot(10, self.update_analysis)  # 10ms Delay
                # self.update_analysis()
            else:
                self.file_label.setText("Load failed")

    def update_analysis(self):
        if self.analyzer.df is None:
            return
        col = self.var_combo.currentText()
        if not col:
            return
        grouped = self.group_check.isChecked()
        group_col = self.analyzer.group_col if grouped else None

        stats = self.analyzer.describe(col, group_by=group_col)
        if stats is None:
            return

        # self.stats_text.clear()
        # if grouped:
        #     self.stats_text.append("Grouped Statistical Results:\n")
        #     self.stats_text.append(str(stats))
        # else:
        #     self.stats_text.append("Descriptive Statistics:\n")
        #     self.stats_text.append(str(stats))

        # 将 stats Convert To DataFrame And Display in Table
        if grouped:
            # stats Already Is DataFrame，Use Directly
            df_stats = stats
        else:
            # stats Yes Series，Convert To DataFrame
            # df_stats = stats.to_frame().T  # Transpose to one row multiple columns
            # 或者更友好的格式：df_stats = stats.reset_index().rename(columns={'index':'统计量', 0:'值'})

            # stats Yes Series，Convert To DataFrame (Vertical layout - stats as rows)
            df_stats = stats.reset_index().rename(columns={'index': 'Statistic', 0: 'Value'})

        model = PandasModel(df_stats)
        self.stats_table.setModel(model)
        self.stats_table.resizeColumnsToContents()

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if self.hist_radio.isChecked():
            if grouped:
                data_list, labels = self.analyzer.get_grouped_data(col, group_col)
                if data_list:
                    ax.hist(data_list, bins=20, alpha=0.7, label=labels)
                    ax.legend()
                    ax.set_title(f'{col} Grouped histogram')
                else:
                    ax.text(0.5, 0.5, 'No grouped data', ha='center', va='center')
            else:
                data = self.analyzer.df[col].dropna()
                ax.hist(data, bins=20, edgecolor='black')
                ax.set_title(f'{col} histogram')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
        else:
            if grouped:
                data_list, labels = self.analyzer.get_grouped_data(col, group_col)
                if data_list:
                    ax.boxplot(data_list, tick_labels=labels)
                    ax.set_title(f'{col} Grouped Boxplot')
                else:
                    ax.text(0.5, 0.5, 'No grouped data', ha='center', va='center')
            else:
                data = self.analyzer.df[col].dropna()
                ax.boxplot(data)
                ax.set_title(f'{col} Boxplot')
                ax.set_xticklabels([col])
            ax.set_ylabel(col)

        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())