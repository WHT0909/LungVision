from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QCheckBox, QFrame)
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt, QSize
import os

class WelcomeDialog(QDialog):
    """
    欢迎对话框，在应用程序启动时显示
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("欢迎使用 LungVision")
        self.setMinimumSize(700, 500)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        # 设置对话框样式
        self.setStyleSheet("""
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
            QCheckBox {
                color: #495057;
            }
        """)
        
        # 主布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # 标题
        title_label = QLabel("欢迎使用 LungVision")
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
            margin-bottom: 10px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 副标题
        subtitle_label = QLabel("胸部X-Ray图像分析系统")
        subtitle_label.setStyleSheet("""
            font-size: 16px;
            color: #6c757d;
            margin-bottom: 20px;
        """)
        subtitle_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle_label)
        
        # 分割线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("margin-bottom: 20px;")
        layout.addWidget(line)
        
        # 内容区域
        content_label = QLabel()
        content_label.setWordWrap(True)
        content_label.setStyleSheet("font-size: 14px; line-height: 1.5;")
        content_label.setText("""
        <p><b>LungVision</b> 是一款专为胸部X-Ray图像分析设计的软件，提供了强大的图像增强和肺叶分割功能。</p>
        
        <p><b>主要功能：</b></p>
        <ul>
            <li><b>图像增强</b> - 提供5种不同的增强算法，改善X-Ray图像的质量和可视性</li>
            <li><b>肺叶分割</b> - 提供5种不同的分割算法，用于识别和分离肺部区域</li>
            <li><b>图像处理</b> - 直观的用户界面，实时直方图显示，处理前后对比</li>
        </ul>
        
        <p><b>快速入门：</b></p>
        <ol>
            <li>点击<b>加载图像</b>按钮，选择一张胸部X-Ray图像</li>
            <li>在右侧面板选择<b>图像增强</b>选项卡，选择增强算法并调整参数</li>
            <li>点击<b>应用增强</b>按钮，查看增强效果</li>
            <li>在右侧面板选择<b>肺叶分割</b>选项卡，选择分割算法并调整参数</li>
            <li>点击<b>应用分割</b>按钮，查看分割结果</li>
            <li>点击<b>保存结果</b>按钮，保存处理后的图像</li>
        </ol>
        
        <p>如需详细帮助，请点击菜单栏中的<b>帮助 > 查看帮助</b>。</p>
        """)
        layout.addWidget(content_label)
        
        # 底部区域
        bottom_layout = QHBoxLayout()
        
        # 不再显示选项
        self.show_again_checkbox = QCheckBox("启动时不再显示此对话框")
        bottom_layout.addWidget(self.show_again_checkbox)
        
        bottom_layout.addStretch()
        
        # 开始使用按钮
        start_button = QPushButton("开始使用")
        start_button.setMinimumWidth(120)
        start_button.clicked.connect(self.accept)
        bottom_layout.addWidget(start_button)
        
        layout.addLayout(bottom_layout)
    
    def should_show_again(self):
        """
        返回是否应该再次显示欢迎对话框
        """
        return not self.show_again_checkbox.isChecked()