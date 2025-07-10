import os
import sys
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QComboBox, 
                             QTabWidget, QScrollArea, QMessageBox, QSlider,
                             QGroupBox, QGridLayout, QSplitter, QFrame,
                             QMenuBar, QMenu, QAction, QToolTip)
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor, QIcon, QFont
from PyQt5.QtCore import Qt, QSize
import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 导入对话框
from ui.help_dialog import HelpDialog, AlgorithmInfoDialog
from ui.welcome_dialog import WelcomeDialog

# 导入算法模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms.enhancement.histogram_equalization import HistogramEqualization
from algorithms.enhancement.clahe import CLAHE
from algorithms.enhancement.gamma_correction import GammaCorrection
from algorithms.enhancement.unsharp_masking import UnsharpMasking
from algorithms.enhancement.wavelet_denoising import WaveletDenoising
from algorithms.segmentation.thresholding import Thresholding
from algorithms.segmentation.region_growing import RegionGrowing
from algorithms.segmentation.watershed import Watershed
from algorithms.segmentation.active_contour import ActiveContour
from algorithms.segmentation.unet import UNet

class ImageViewer(QWidget):
    def __init__(self, title=""):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # 标题
        if title:
            self.title_label = QLabel(title)
            self.title_label.setAlignment(Qt.AlignCenter)
            self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
            self.layout.addWidget(self.title_label)
        
        # 图像显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ddd;")
        self.layout.addWidget(self.image_label)
        
        # 直方图
        self.figure = Figure(figsize=(5, 3))  # 增加高度
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        self.image = None
    
    def set_image(self, image):
        self.image = image
        if image is not None:
            h, w = image.shape[:2]
            bytes_per_line = 3 * w
            if len(image.shape) == 2:  # 灰度图
                q_image = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
            else:  # 彩色图
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            # 更新直方图
            self.update_histogram(image)
    
    def update_histogram(self, image):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if len(image.shape) == 2:  # 灰度图
            ax.hist(image.ravel(), 256, [0, 256], color='gray', alpha=0.7)
        else:  # 彩色图
            colors = ('b', 'g', 'r')
            for i, color in enumerate(colors):
                ax.hist(image[:, :, i].ravel(), 256, [0, 256], color=color, alpha=0.5)
        
        ax.set_xlim([0, 256])
        ax.set_title("直方图", fontsize=12, pad=10)  # 增加字体大小和上边距
        
        # 调整图表布局，确保所有元素可见
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LungVision - 胸部X-Ray图像分析系统")
        self.setMinimumSize(1200, 800)
        
        # 设置应用图标
        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "logo.png")
        self.setWindowIcon(QIcon(icon_path))
        
        # 显示欢迎对话框
        self.show_welcome_dialog()
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 设置应用样式
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #f8f9fa;
                color: #212529;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #0069d9;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
            QComboBox {
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 5px;
                background-color: white;
                min-height: 25px;
                margin: 5px 0;
            }
            QSlider {
                border: none;
                height: 25px;
                margin: 5px 0;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #e9ecef;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #007bff;
                border: none;
                width: 16px;
                height: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #0056b3;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 15px;
                padding-top: 20px;
                padding-bottom: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 10px;
                background-color: #f8f9fa;
            }
            QProgressBar {
                border: none;
                border-radius: 7px;
                text-align: center;
                background: #e9ecef;
                height: 14px;
                margin: 5px 0;
                font-size: 10px;
                color: #495057;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #4facfe, stop:1 #00f2fe);
                border-radius: 7px;
            }
            QLabel {
                margin: 2px 0;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        
        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 顶部控制区域
        top_controls = QWidget()
        top_layout = QHBoxLayout(top_controls)
        
        # 加载图像按钮
        self.load_button = QPushButton("加载图像")
        self.load_button.clicked.connect(self.load_image)
        top_layout.addWidget(self.load_button)
        
        # 保存结果按钮
        self.save_button = QPushButton("保存结果")
        self.save_button.clicked.connect(self.save_result)
        self.save_button.setEnabled(False)
        top_layout.addWidget(self.save_button)
        
        # 重置按钮
        self.reset_button = QPushButton("重置")
        self.reset_button.clicked.connect(self.reset)
        self.reset_button.setEnabled(False)
        top_layout.addWidget(self.reset_button)
        
        main_layout.addWidget(top_controls)
        
        # 分割线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line)
        
        # 主要内容区域
        content_widget = QSplitter(Qt.Horizontal)
        
        # 左侧：图像显示区域
        self.display_area = QWidget()
        display_layout = QHBoxLayout(self.display_area)
        
        # 原始图像显示
        self.original_viewer = ImageViewer("原始图像")
        display_layout.addWidget(self.original_viewer)
        
        # 处理后图像显示
        self.processed_viewer = ImageViewer("处理结果")
        display_layout.addWidget(self.processed_viewer)
        
        content_widget.addWidget(self.display_area)
        
        # 右侧：算法控制面板
        self.control_panel = QTabWidget()
        
        # 图像增强选项卡
        enhancement_tab = QWidget()
        scroll_area1 = QScrollArea()
        scroll_area1.setWidgetResizable(True)
        scroll_content1 = QWidget()
        enhancement_layout = QVBoxLayout(scroll_content1)
        enhancement_layout.setContentsMargins(10, 10, 10, 10)
        enhancement_layout.setSpacing(15)
        scroll_area1.setWidget(scroll_content1)
        
        # 增强算法选择
        enhancement_group = QGroupBox("增强算法")
        enhancement_group_layout = QVBoxLayout(enhancement_group)
        enhancement_group_layout.setContentsMargins(15, 20, 15, 15)
        enhancement_group_layout.setSpacing(10)
        
        # 算法选择标签
        algorithm_label = QLabel("选择算法:")
        algorithm_label.setStyleSheet("font-weight: bold;")
        enhancement_group_layout.addWidget(algorithm_label)
        
        # 算法选择区域
        algo_selection_widget = QWidget()
        algo_selection_layout = QHBoxLayout(algo_selection_widget)
        algo_selection_layout.setContentsMargins(0, 0, 0, 0)
        
        # 算法下拉框
        self.enhancement_combo = QComboBox()
        self.enhancement_combo.addItems([
            "直方图均衡化",
            "自适应直方图均衡化(CLAHE)",
            "伽马校正",
            "非锐化掩蔽",
            "小波去噪"
        ])
        self.enhancement_combo.currentIndexChanged.connect(self.update_enhancement_params)
        algo_selection_layout.addWidget(self.enhancement_combo)
        
        # 算法帮助按钮
        help_button = QPushButton("?")
        help_button.setFixedSize(25, 25)
        help_button.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                border-radius: 12px;
                font-weight: bold;
                color: white;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        help_button.setToolTip("查看当前算法的详细说明")
        help_button.clicked.connect(lambda: self.show_algorithm_info("enhancement"))
        algo_selection_layout.addWidget(help_button)
        
        enhancement_group_layout.addWidget(algo_selection_widget)
        
        # 算法参数区域
        params_label = QLabel("参数设置:")
        params_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        enhancement_group_layout.addWidget(params_label)
        
        self.enhancement_params_widget = QWidget()
        self.enhancement_params_layout = QVBoxLayout(self.enhancement_params_widget)
        self.enhancement_params_layout.setContentsMargins(0, 0, 0, 0)
        self.enhancement_params_layout.setSpacing(8)
        enhancement_group_layout.addWidget(self.enhancement_params_widget)
        
        # 应用增强按钮
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 10, 0, 0)
        
        self.apply_enhancement_button = QPushButton("应用增强")
        self.apply_enhancement_button.clicked.connect(self.apply_enhancement)
        self.apply_enhancement_button.setEnabled(False)
        self.apply_enhancement_button.setMinimumHeight(35)
        button_layout.addStretch()
        button_layout.addWidget(self.apply_enhancement_button)
        button_layout.addStretch()
        
        enhancement_group_layout.addWidget(button_container)
        
        enhancement_layout.addWidget(enhancement_group)
        enhancement_layout.addStretch()
        
        # 分割选项卡
        segmentation_tab = QWidget()
        scroll_area2 = QScrollArea()
        scroll_area2.setWidgetResizable(True)
        scroll_content2 = QWidget()
        segmentation_layout = QVBoxLayout(scroll_content2)
        segmentation_layout.setContentsMargins(10, 10, 10, 10)
        segmentation_layout.setSpacing(15)
        scroll_area2.setWidget(scroll_content2)
        segmentation_tab.setLayout(QVBoxLayout())
        segmentation_tab.layout().addWidget(scroll_area2)
        
        # 分割算法选择
        segmentation_group = QGroupBox("分割算法")
        segmentation_group_layout = QVBoxLayout(segmentation_group)
        segmentation_group_layout.setContentsMargins(15, 20, 15, 15)
        segmentation_group_layout.setSpacing(10)
        
        # 算法选择标签
        seg_algorithm_label = QLabel("选择算法:")
        seg_algorithm_label.setStyleSheet("font-weight: bold;")
        segmentation_group_layout.addWidget(seg_algorithm_label)
        
        # 算法选择区域
        seg_algo_selection_widget = QWidget()
        seg_algo_selection_layout = QHBoxLayout(seg_algo_selection_widget)
        seg_algo_selection_layout.setContentsMargins(0, 0, 0, 0)
        
        # 算法下拉框
        self.segmentation_combo = QComboBox()
        self.segmentation_combo.addItems([
            "阈值分割",
            "区域生长",
            "分水岭分割",
            "主动轮廓",
            "U-Net深度学习"
        ])
        self.segmentation_combo.currentIndexChanged.connect(self.update_segmentation_params)
        seg_algo_selection_layout.addWidget(self.segmentation_combo)
        
        # 算法帮助按钮
        seg_help_button = QPushButton("?")
        seg_help_button.setFixedSize(25, 25)
        seg_help_button.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                border-radius: 12px;
                font-weight: bold;
                color: white;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        seg_help_button.setToolTip("查看当前算法的详细说明")
        seg_help_button.clicked.connect(lambda: self.show_algorithm_info("segmentation"))
        seg_algo_selection_layout.addWidget(seg_help_button)
        
        segmentation_group_layout.addWidget(seg_algo_selection_widget)
        
        # 算法参数区域
        seg_params_label = QLabel("参数设置:")
        seg_params_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        segmentation_group_layout.addWidget(seg_params_label)
        
        # 创建一个固定高度的参数容器，避免界面元素叠加
        self.segmentation_params_container = QWidget()
        self.segmentation_params_container.setMinimumHeight(250)  # 设置一个固定的最小高度
        params_container_layout = QVBoxLayout(self.segmentation_params_container)
        params_container_layout.setContentsMargins(0, 0, 0, 0)
        
        self.segmentation_params_widget = QWidget()
        self.segmentation_params_layout = QVBoxLayout(self.segmentation_params_widget)
        self.segmentation_params_layout.setContentsMargins(0, 0, 0, 0)
        self.segmentation_params_layout.setSpacing(8)
        
        params_container_layout.addWidget(self.segmentation_params_widget)
        params_container_layout.addStretch()
        segmentation_group_layout.addWidget(self.segmentation_params_container)
        
        # 应用分割按钮
        seg_button_container = QWidget()
        seg_button_layout = QHBoxLayout(seg_button_container)
        seg_button_layout.setContentsMargins(0, 10, 0, 0)
        
        self.apply_segmentation_button = QPushButton("应用分割")
        self.apply_segmentation_button.clicked.connect(self.apply_segmentation)
        self.apply_segmentation_button.setEnabled(False)
        self.apply_segmentation_button.setMinimumHeight(35)
        seg_button_layout.addStretch()
        seg_button_layout.addWidget(self.apply_segmentation_button)
        seg_button_layout.addStretch()
        
        segmentation_group_layout.addWidget(seg_button_container)
        
        segmentation_layout.addWidget(segmentation_group)
        segmentation_layout.addStretch()
        
        # 添加选项卡
        enhancement_tab.setLayout(QVBoxLayout())
        enhancement_tab.layout().addWidget(scroll_area1)
        self.control_panel.addTab(enhancement_tab, "图像增强")
        self.control_panel.addTab(segmentation_tab, "肺叶分割")
        
        content_widget.addWidget(self.control_panel)
        content_widget.setSizes([700, 500])
        
        main_layout.addWidget(content_widget)
        
        # 状态栏
        self.statusBar().showMessage("就绪")
        
        # 初始化变量
        self.original_image = None
        self.processed_image = None
        self.current_enhancement = None
        self.current_segmentation = None
        
        # 初始化算法实例
        self.enhancement_algorithms = {
            0: HistogramEqualization(),
            1: CLAHE(),
            2: GammaCorrection(),
            3: UnsharpMasking(),
            4: WaveletDenoising()
        }
        
        self.segmentation_algorithms = {
            0: Thresholding(),
            1: RegionGrowing(),
            2: Watershed(),
            3: ActiveContour(),
            4: UNet()
        }
        
        # 初始化参数控件
        self.update_enhancement_params(0)
        self.update_segmentation_params(0)
    
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif)"
        )
        
        if file_path:
            try:
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    raise Exception("无法读取图像文件")
                
                # 转换为灰度图
                if len(self.original_image.shape) == 3:
                    self.original_image_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                else:
                    self.original_image_gray = self.original_image.copy()
                
                # 显示原始图像
                self.original_viewer.set_image(self.original_image)
                self.processed_viewer.set_image(None)
                
                # 启用按钮
                self.apply_enhancement_button.setEnabled(True)
                self.apply_segmentation_button.setEnabled(True)
                self.reset_button.setEnabled(True)
                
                self.processed_image = None
                self.save_button.setEnabled(False)
                
                self.statusBar().showMessage(f"已加载图像: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载图像时出错: {str(e)}")
    
    def update_enhancement_params(self, index):
        # 清除现有参数控件
        while self.enhancement_params_layout.count():
            item = self.enhancement_params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 创建参数表单布局
        form_widget = QWidget()
        form_layout = QGridLayout(form_widget)
        form_layout.setContentsMargins(5, 5, 5, 5)
        form_layout.setSpacing(10)
        row = 0
        
        # 根据选择的算法添加参数控件
        if index == 0:  # 直方图均衡化
            # 无参数
            info_label = QLabel("此算法无需设置参数")
            info_label.setStyleSheet("color: #666; font-style: italic;")
            form_layout.addWidget(info_label, row, 0, 1, 3)
            
        elif index == 1:  # CLAHE
            # 添加clip limit参数
            clip_label = QLabel("对比度限制:")
            self.clahe_clip_slider = QSlider(Qt.Horizontal)
            self.clahe_clip_slider.setRange(1, 50)
            self.clahe_clip_slider.setValue(20)
            self.clahe_clip_value = QLabel("2.0")
            self.clahe_clip_value.setMinimumWidth(40)
            self.clahe_clip_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.clahe_clip_slider.valueChanged.connect(
                lambda v: self.clahe_clip_value.setText(f"{v/10:.1f}")
            )
            form_layout.addWidget(clip_label, row, 0)
            form_layout.addWidget(self.clahe_clip_slider, row, 1)
            form_layout.addWidget(self.clahe_clip_value, row, 2)
            row += 1
            
            # 添加网格大小参数
            grid_label = QLabel("网格大小:")
            self.clahe_grid_slider = QSlider(Qt.Horizontal)
            self.clahe_grid_slider.setRange(2, 16)
            self.clahe_grid_slider.setValue(8)
            self.clahe_grid_value = QLabel("8")
            self.clahe_grid_value.setMinimumWidth(40)
            self.clahe_grid_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.clahe_grid_slider.valueChanged.connect(
                lambda v: self.clahe_grid_value.setText(f"{v}")
            )
            form_layout.addWidget(grid_label, row, 0)
            form_layout.addWidget(self.clahe_grid_slider, row, 1)
            form_layout.addWidget(self.clahe_grid_value, row, 2)
            
        elif index == 2:  # 伽马校正
            gamma_label = QLabel("伽马值:")
            self.gamma_slider = QSlider(Qt.Horizontal)
            self.gamma_slider.setRange(1, 50)
            self.gamma_slider.setValue(10)
            self.gamma_value = QLabel("1.0")
            self.gamma_value.setMinimumWidth(40)
            self.gamma_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.gamma_slider.valueChanged.connect(
                lambda v: self.gamma_value.setText(f"{v/10:.1f}")
            )
            form_layout.addWidget(gamma_label, row, 0)
            form_layout.addWidget(self.gamma_slider, row, 1)
            form_layout.addWidget(self.gamma_value, row, 2)
            
        elif index == 3:  # 非锐化掩蔽
            # 添加半径参数
            radius_label = QLabel("模糊半径:")
            self.unsharp_radius_slider = QSlider(Qt.Horizontal)
            self.unsharp_radius_slider.setRange(1, 20)
            self.unsharp_radius_slider.setValue(5)
            self.unsharp_radius_value = QLabel("5")
            self.unsharp_radius_value.setMinimumWidth(40)
            self.unsharp_radius_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.unsharp_radius_slider.valueChanged.connect(
                lambda v: self.unsharp_radius_value.setText(f"{v}")
            )
            form_layout.addWidget(radius_label, row, 0)
            form_layout.addWidget(self.unsharp_radius_slider, row, 1)
            form_layout.addWidget(self.unsharp_radius_value, row, 2)
            row += 1
            
            # 添加强度参数
            amount_label = QLabel("强度:")
            self.unsharp_amount_slider = QSlider(Qt.Horizontal)
            self.unsharp_amount_slider.setRange(1, 50)
            self.unsharp_amount_slider.setValue(15)
            self.unsharp_amount_value = QLabel("1.5")
            self.unsharp_amount_value.setMinimumWidth(40)
            self.unsharp_amount_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.unsharp_amount_slider.valueChanged.connect(
                lambda v: self.unsharp_amount_value.setText(f"{v/10:.1f}")
            )
            form_layout.addWidget(amount_label, row, 0)
            form_layout.addWidget(self.unsharp_amount_slider, row, 1)
            form_layout.addWidget(self.unsharp_amount_value, row, 2)
            
        elif index == 4:  # 小波去噪
            # 添加阈值参数
            threshold_label = QLabel("阈值:")
            self.wavelet_threshold_slider = QSlider(Qt.Horizontal)
            self.wavelet_threshold_slider.setRange(10, 100)
            self.wavelet_threshold_slider.setValue(30)
            self.wavelet_threshold_value = QLabel("30")
            self.wavelet_threshold_value.setMinimumWidth(40)
            self.wavelet_threshold_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.wavelet_threshold_slider.valueChanged.connect(
                lambda v: self.wavelet_threshold_value.setText(f"{v}")
            )
            form_layout.addWidget(threshold_label, row, 0)
            form_layout.addWidget(self.wavelet_threshold_slider, row, 1)
            form_layout.addWidget(self.wavelet_threshold_value, row, 2)
            row += 1
            
            # 添加小波类型选择
            wavelet_type_label = QLabel("小波类型:")
            self.wavelet_type_combo = QComboBox()
            self.wavelet_type_combo.addItems(["db1", "db2", "sym2", "coif1", "bior1.3"])
            form_layout.addWidget(wavelet_type_label, row, 0)
            form_layout.addWidget(self.wavelet_type_combo, row, 1, 1, 2)
        
        # 添加表单到布局
        self.enhancement_params_layout.addWidget(form_widget)
    
    def update_segmentation_params(self, index):
        # 清除现有参数控件
        while self.segmentation_params_layout.count():
            item = self.segmentation_params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 创建参数表单布局
        form_widget = QWidget()
        form_layout = QGridLayout(form_widget)
        form_layout.setContentsMargins(5, 5, 5, 5)
        form_layout.setSpacing(10)
        row = 0
        
        # 根据选择的算法添加参数控件
        if index == 0:  # 阈值分割
            # 添加阈值参数
            threshold_label = QLabel("阈值:")
            self.threshold_slider = QSlider(Qt.Horizontal)
            self.threshold_slider.setRange(0, 255)
            self.threshold_slider.setValue(127)
            self.threshold_value = QLabel("127")
            self.threshold_value.setMinimumWidth(40)
            self.threshold_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.threshold_slider.valueChanged.connect(
                lambda v: self.threshold_value.setText(f"{v}")
            )
            form_layout.addWidget(threshold_label, row, 0)
            form_layout.addWidget(self.threshold_slider, row, 1)
            form_layout.addWidget(self.threshold_value, row, 2)
            row += 1
            
            # 添加阈值类型选择
            threshold_type_label = QLabel("阈值类型:")
            self.threshold_type_combo = QComboBox()
            self.threshold_type_combo.addItems(["二值化", "反二值化", "截断", "阈值为零", "反阈值为零"])
            form_layout.addWidget(threshold_type_label, row, 0)
            form_layout.addWidget(self.threshold_type_combo, row, 1, 1, 2)
            
        elif index == 1:  # 区域生长
            # 添加种子点X坐标输入
            seed_x_label = QLabel("种子点X坐标:")
            self.seed_x_slider = QSlider(Qt.Horizontal)
            self.seed_x_slider.setRange(0, 100)
            self.seed_x_slider.setValue(50)
            self.seed_x_value = QLabel("50%")
            self.seed_x_value.setMinimumWidth(40)
            self.seed_x_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.seed_x_slider.valueChanged.connect(
                lambda v: self.seed_x_value.setText(f"{v}%")
            )
            form_layout.addWidget(seed_x_label, row, 0)
            form_layout.addWidget(self.seed_x_slider, row, 1)
            form_layout.addWidget(self.seed_x_value, row, 2)
            row += 1
            
            # 添加种子点Y坐标输入
            seed_y_label = QLabel("种子点Y坐标:")
            self.seed_y_slider = QSlider(Qt.Horizontal)
            self.seed_y_slider.setRange(0, 100)
            self.seed_y_slider.setValue(50)
            self.seed_y_value = QLabel("50%")
            self.seed_y_value.setMinimumWidth(40)
            self.seed_y_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.seed_y_slider.valueChanged.connect(
                lambda v: self.seed_y_value.setText(f"{v}%")
            )
            form_layout.addWidget(seed_y_label, row, 0)
            form_layout.addWidget(self.seed_y_slider, row, 1)
            form_layout.addWidget(self.seed_y_value, row, 2)
            row += 1
            
            # 添加种子点坐标说明
            seed_info_label = QLabel("种子点坐标示例: 图像中心(50%, 50%)，左上角(0%, 0%)，右下角(100%, 100%)")
            seed_info_label.setAlignment(Qt.AlignCenter)
            seed_info_label.setStyleSheet("font-style: italic; color: #6c757d; font-size: 11px;")
            seed_info_label.setWordWrap(True)
            form_layout.addWidget(seed_info_label, row, 0, 1, 3)
            row += 1
            
            # 添加阈值参数
            threshold_label = QLabel("生长阈值:")
            self.region_threshold_slider = QSlider(Qt.Horizontal)
            self.region_threshold_slider.setRange(1, 50)
            self.region_threshold_slider.setValue(10)
            self.region_threshold_value = QLabel("10")
            self.region_threshold_value.setMinimumWidth(40)
            self.region_threshold_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.region_threshold_slider.valueChanged.connect(
                lambda v: self.region_threshold_value.setText(f"{v}")
            )
            form_layout.addWidget(threshold_label, row, 0)
            form_layout.addWidget(self.region_threshold_slider, row, 1)
            form_layout.addWidget(self.region_threshold_value, row, 2)
            
        elif index == 2:  # 分水岭算法
            # 添加标记距离参数
            distance_label = QLabel("标记距离:")
            self.watershed_distance_slider = QSlider(Qt.Horizontal)
            self.watershed_distance_slider.setRange(1, 20)
            self.watershed_distance_slider.setValue(9)
            self.watershed_distance_value = QLabel("9")
            self.watershed_distance_value.setMinimumWidth(40)
            self.watershed_distance_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.watershed_distance_slider.valueChanged.connect(
                lambda v: self.watershed_distance_value.setText(f"{v}")
            )
            form_layout.addWidget(distance_label, row, 0)
            form_layout.addWidget(self.watershed_distance_slider, row, 1)
            form_layout.addWidget(self.watershed_distance_value, row, 2)
            
        elif index == 3:  # 主动轮廓
            # 添加迭代次数参数
            iterations_label = QLabel("迭代次数:")
            self.contour_iterations_slider = QSlider(Qt.Horizontal)
            self.contour_iterations_slider.setRange(10, 300)
            self.contour_iterations_slider.setValue(100)
            self.contour_iterations_value = QLabel("100")
            self.contour_iterations_value.setMinimumWidth(40)
            self.contour_iterations_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.contour_iterations_slider.valueChanged.connect(
                lambda v: self.contour_iterations_value.setText(f"{v}")
            )
            form_layout.addWidget(iterations_label, row, 0)
            form_layout.addWidget(self.contour_iterations_slider, row, 1)
            form_layout.addWidget(self.contour_iterations_value, row, 2)
            row += 1
            
            # 添加alpha参数
            alpha_label = QLabel("Alpha:")
            self.contour_alpha_slider = QSlider(Qt.Horizontal)
            self.contour_alpha_slider.setRange(1, 50)
            self.contour_alpha_slider.setValue(15)
            self.contour_alpha_value = QLabel("0.15")
            self.contour_alpha_value.setMinimumWidth(40)
            self.contour_alpha_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.contour_alpha_slider.valueChanged.connect(
                lambda v: self.contour_alpha_value.setText(f"{v/100:.2f}")
            )
            form_layout.addWidget(alpha_label, row, 0)
            form_layout.addWidget(self.contour_alpha_slider, row, 1)
            form_layout.addWidget(self.contour_alpha_value, row, 2)
            row += 1
            
            # 添加beta参数
            beta_label = QLabel("Beta:")
            self.contour_beta_slider = QSlider(Qt.Horizontal)
            self.contour_beta_slider.setRange(1, 50)
            self.contour_beta_slider.setValue(10)
            self.contour_beta_value = QLabel("0.10")
            self.contour_beta_value.setMinimumWidth(40)
            self.contour_beta_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.contour_beta_slider.valueChanged.connect(
                lambda v: self.contour_beta_value.setText(f"{v/100:.2f}")
            )
            form_layout.addWidget(beta_label, row, 0)
            form_layout.addWidget(self.contour_beta_slider, row, 1)
            form_layout.addWidget(self.contour_beta_value, row, 2)
            
        elif index == 4:  # U-Net深度学习
            # 添加模型选择
            model_label = QLabel("模型选择:")
            self.unet_model_combo = QComboBox()
            self.unet_model_combo.addItems(["预训练模型", "自定义训练"])
            form_layout.addWidget(model_label, row, 0)
            form_layout.addWidget(self.unet_model_combo, row, 1, 1, 2)
            row += 1
            
            # 添加置信度阈值
            confidence_label = QLabel("置信度阈值:")
            self.unet_confidence_slider = QSlider(Qt.Horizontal)
            self.unet_confidence_slider.setRange(1, 99)
            self.unet_confidence_slider.setValue(50)
            self.unet_confidence_value = QLabel("0.50")
            self.unet_confidence_value.setMinimumWidth(40)
            self.unet_confidence_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.unet_confidence_slider.valueChanged.connect(
                lambda v: self.unet_confidence_value.setText(f"{v/100:.2f}")
            )
            form_layout.addWidget(confidence_label, row, 0)
            form_layout.addWidget(self.unet_confidence_slider, row, 1)
            form_layout.addWidget(self.unet_confidence_value, row, 2)
        
        # 添加表单到布局
        self.segmentation_params_layout.addWidget(form_widget)
    
    def apply_enhancement(self):
        if self.original_image is None:
            return
        
        try:
            # 获取当前选择的增强算法
            index = self.enhancement_combo.currentIndex()
            algorithm = self.enhancement_algorithms[index]
            
            # 设置算法参数
            if index == 0:  # 直方图均衡化
                params = {}
            elif index == 1:  # CLAHE
                clip_limit = self.clahe_clip_slider.value() / 10.0
                grid_size = self.clahe_grid_slider.value()
                params = {"clip_limit": clip_limit, "grid_size": grid_size}
            elif index == 2:  # 伽马校正
                gamma = self.gamma_slider.value() / 10.0
                params = {"gamma": gamma}
            elif index == 3:  # 非锐化掩蔽
                radius = self.unsharp_radius_slider.value()
                amount = self.unsharp_amount_slider.value() / 10.0
                params = {"radius": radius, "amount": amount}
            elif index == 4:  # 小波去噪
                threshold = self.wavelet_threshold_slider.value()
                wavelet = self.wavelet_type_combo.currentText()
                params = {"threshold": threshold, "wavelet": wavelet}
            
            # 应用增强算法
            self.processed_image = algorithm.process(self.original_image_gray, **params)
            
            # 显示处理后的图像
            self.processed_viewer.set_image(self.processed_image)
            
            # 启用保存按钮和保存菜单项
            self.save_button.setEnabled(True)
            self.save_action.setEnabled(True)
            
            self.statusBar().showMessage(f"已应用{self.enhancement_combo.currentText()}增强")
            
            # 记录当前处理
            self.current_enhancement = index
            self.current_segmentation = None
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"应用增强算法时出错: {str(e)}")
    
    def apply_segmentation(self):
        if self.original_image is None:
            return
        
        try:
            # 获取当前选择的分割算法
            index = self.segmentation_combo.currentIndex()
            algorithm = self.segmentation_algorithms[index]
            
            # 使用原始图像或已增强的图像
            input_image = self.processed_image if self.processed_image is not None else self.original_image_gray
            
            # 设置算法参数
            if index == 0:  # 阈值分割
                threshold = self.threshold_slider.value()
                threshold_type = self.threshold_type_combo.currentIndex()
                params = {"threshold": threshold, "threshold_type": threshold_type}
            elif index == 1:  # 区域生长
                # 使用用户输入的种子点坐标
                h, w = input_image.shape[:2]
                # 将百分比转换为实际坐标
                seed_x_percent = self.seed_x_slider.value() / 100.0
                seed_y_percent = self.seed_y_slider.value() / 100.0
                seed_x = int(w * seed_x_percent)
                seed_y = int(h * seed_y_percent)
                # 确保坐标在图像范围内
                seed_x = max(0, min(seed_x, w - 1))
                seed_y = max(0, min(seed_y, h - 1))
                seed_point = (seed_x, seed_y)
                threshold = self.region_threshold_slider.value()
                params = {"seed_point": seed_point, "threshold": threshold}
            elif index == 2:  # 分水岭算法
                distance = self.watershed_distance_slider.value()
                params = {"distance": distance}
            elif index == 3:  # 主动轮廓
                iterations = self.contour_iterations_slider.value()
                alpha = self.contour_alpha_slider.value() / 100.0
                beta = self.contour_beta_slider.value() / 100.0
                params = {"iterations": iterations, "alpha": alpha, "beta": beta}
            elif index == 4:  # U-Net深度学习
                model_type = self.unet_model_combo.currentIndex()
                confidence = self.unet_confidence_slider.value() / 100.0
                params = {"model_type": model_type, "confidence": confidence}
            
            # 应用分割算法
            self.processed_image = algorithm.process(input_image, **params)
            
            # 显示处理后的图像
            self.processed_viewer.set_image(self.processed_image)
            
            # 启用保存按钮和保存菜单项
            self.save_button.setEnabled(True)
            self.save_action.setEnabled(True)
            
            self.statusBar().showMessage(f"已应用{self.segmentation_combo.currentText()}分割")
            
            # 记录当前处理
            self.current_segmentation = index
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"应用分割算法时出错: {str(e)}")
    
    def save_result(self):
        if self.processed_image is None:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "", "PNG图像 (*.png);;JPEG图像 (*.jpg);;所有文件 (*)"
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.processed_image)
                self.statusBar().showMessage(f"结果已保存至: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存图像时出错: {str(e)}")
    
    def reset(self):
        if self.original_image is not None:
            # 重新显示原图
            self.original_viewer.set_image(self.original_image)
            # 清空处理后的图像
            self.processed_viewer.set_image(None)
            self.processed_image = None
            # 禁用保存按钮和保存菜单项
            self.save_button.setEnabled(False)
            self.save_action.setEnabled(False)
            # 重置当前处理状态
            self.current_enhancement = None
            self.current_segmentation = None
            # 更新状态栏
            self.statusBar().showMessage("已重置")
            # 确保处理后的图像视图被清空
            self.processed_viewer.image = None
            self.processed_viewer.image_label.clear()
            self.processed_viewer.figure.clear()
            self.processed_viewer.canvas.draw()
    
    def create_menu_bar(self):
        """
        创建菜单栏
        """
        # 创建菜单栏
        menu_bar = self.menuBar()
        menu_bar.setStyleSheet("""
            QMenuBar {
                background-color: #f8f9fa;
                border-bottom: 1px solid #ddd;
            }
            QMenuBar::item {
                padding: 5px 10px;
                background-color: transparent;
            }
            QMenuBar::item:selected {
                background-color: #e9ecef;
                border-radius: 4px;
            }
            QMenu {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px 0px;
            }
            QMenu::item {
                padding: 5px 20px;
            }
            QMenu::item:selected {
                background-color: #007bff;
                color: white;
            }
        """)
        
        # 文件菜单
        file_menu = menu_bar.addMenu("文件")
        
        # 加载图像动作
        load_action = QAction("加载图像", self)
        load_action.triggered.connect(self.load_image)
        file_menu.addAction(load_action)
        
        # 保存结果动作
        save_action = QAction("保存结果", self)
        save_action.triggered.connect(self.save_result)
        save_action.setEnabled(False)
        self.save_action = save_action  # 保存引用以便后续启用/禁用
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # 退出动作
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 帮助菜单
        help_menu = menu_bar.addMenu("帮助")
        
        # 查看帮助动作
        help_action = QAction("查看帮助", self)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)
        
        # # 算法说明动作
        # algorithm_info_action = QAction("算法说明", self)
        # algorithm_info_action.triggered.connect(self.show_algorithm_info)
        # help_menu.addAction(algorithm_info_action)
        
        # help_menu.addSeparator()
        
        # 关于动作
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def show_help(self):
        """
        显示帮助对话框
        """
        help_dialog = HelpDialog(self)
        help_dialog.exec_()
    
    def show_algorithm_info(self, algorithm_type=None):
        """
        显示当前选择的算法信息
        
        参数:
            algorithm_type: 算法类型，可以是 "enhancement" 或 "segmentation"，
                          如果为 None，则根据当前选项卡自动判断
        """
        # 如果未指定算法类型，则根据当前选项卡判断
        if algorithm_type is None:
            if self.control_panel.currentIndex() == 0:  # 图像增强选项卡
                algorithm_type = "enhancement"
            else:  # 肺叶分割选项卡
                algorithm_type = "segmentation"
        
        # 根据算法类型获取算法名称
        if algorithm_type == "enhancement":
            algorithm_index = self.enhancement_combo.currentIndex()
            algorithm_name = self.enhancement_combo.currentText()
        else:  # segmentation
            algorithm_index = self.segmentation_combo.currentIndex()
            algorithm_name = self.segmentation_combo.currentText()
        
        # 显示算法信息对话框
        info_dialog = AlgorithmInfoDialog(algorithm_name, algorithm_type, self)
        info_dialog.exec_()
    
    def show_about(self):
        """
        显示关于对话框
        """
        # 创建自定义关于对话框
        from PyQt5.QtWidgets import QDialog
        about_dialog = QDialog(self)
        about_dialog.setWindowTitle("关于 LungVision")
        about_dialog.setMinimumSize(500, 400)
        
        # 设置对话框图标
        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "logo.png")
        about_dialog.setWindowIcon(QIcon(icon_path))
        
        # 设置对话框样式
        about_dialog.setStyleSheet("""
            QDialog {
                background-color: #f8f9fa;
            }
            QLabel {
                color: #212529;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0069d9;
            }
        """)
        
        # 主布局
        layout = QVBoxLayout(about_dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Logo和标题区域
        logo_title_layout = QHBoxLayout()
        
        # Logo
        logo_label = QLabel()
        logo_pixmap = QPixmap(icon_path)
        logo_label.setPixmap(logo_pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        logo_label.setAlignment(Qt.AlignCenter)
        logo_title_layout.addWidget(logo_label)
        
        # 标题和版本信息
        title_version_layout = QVBoxLayout()
        
        title_label = QLabel("LungVision")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #007bff;")
        title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title_version_layout.addWidget(title_label)
        
        subtitle_label = QLabel("胸部X-Ray图像分析系统")
        subtitle_label.setStyleSheet("font-size: 16px; color: #6c757d;")
        subtitle_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title_version_layout.addWidget(subtitle_label)
        
        version_label = QLabel("版本: 1.0.0")
        version_label.setStyleSheet("font-size: 14px; color: #6c757d;")
        version_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title_version_layout.addWidget(version_label)
        
        logo_title_layout.addLayout(title_version_layout)
        layout.addLayout(logo_title_layout)
        
        # 分割线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # 描述文本
        description_label = QLabel()
        description_label.setWordWrap(True)
        description_label.setStyleSheet("font-size: 14px; line-height: 1.5;")
        description_label.setText("""
        <p>LungVision是一款专为胸部X-Ray图像分析设计的软件，提供了强大的图像增强和肺叶分割功能。</p>
        <p>本软件集成了多种图像处理算法，通过直观的用户界面，帮助用户更好地分析和处理胸部X光片，提高诊断效率。</p>
        <p>© 2025 WangHaotian SCU BME</p>
        """)
        layout.addWidget(description_label)
        
        layout.addStretch()
        
        # 底部按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        close_button = QPushButton("关闭")
        close_button.setMinimumWidth(100)
        close_button.clicked.connect(about_dialog.accept)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
        # 显示对话框
        about_dialog.exec_()
    
    def show_welcome_dialog(self):
        """
        显示欢迎对话框
        """
        # 检查是否应该显示欢迎对话框
        # 这里可以添加配置文件检查逻辑，暂时默认显示
        welcome_dialog = WelcomeDialog(self)
        welcome_dialog.exec_()
        
        # 可以在这里保存用户的选择到配置文件
        # 例如：是否在启动时显示欢迎对话框
