import os
import sys
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTextBrowser, 
                             QPushButton, QTreeWidget, QTreeWidgetItem, QSplitter,
                             QWidget, QLabel, QFrame)
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt, QUrl, QSize

class HelpDialog(QDialog):
    """
    帮助对话框，用于显示帮助文档
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LungVision 帮助")
        self.setMinimumSize(900, 600)
        
        # 设置对话框图标
        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "logo.png")
        self.setWindowIcon(QIcon(icon_path))
        
        # 设置对话框样式
        self.setStyleSheet("""
            QDialog {
                background-color: #f8f9fa;
            }
            QTextBrowser {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
                padding: 15px;
                font-size: 14px;
                min-height: 450px;
                line-height: 1.6;
                color: #333333;
            }
            QTextBrowser p {
                margin-top: 8px;
                margin-bottom: 8px;
            }
            QTextBrowser h1 {
                margin-top: 20px;
                margin-bottom: 15px;
                color: #1a73e8;
                font-size: 22px;
            }
            QTextBrowser h2 {
                margin-top: 18px;
                margin-bottom: 12px;
                color: #1967d2;
                font-size: 18px;
            }
            QTextBrowser h3 {
                margin-top: 16px;
                margin-bottom: 10px;
                color: #185abc;
                font-size: 16px;
            }
            QTextBrowser li {
                margin-top: 4px;
                margin-bottom: 4px;
                padding-left: 5px;
            }
            QTextBrowser ul, QTextBrowser ol {
                margin-top: 8px;
                margin-bottom: 8px;
                padding-left: 20px;
            }
            QTextBrowser pre {
                background-color: #f5f7f9;
                border: 1px solid #e1e4e8;
                border-radius: 4px;
                padding: 12px;
                margin: 10px 0;
                font-family: Consolas, Monaco, 'Courier New', monospace;
                overflow-x: auto;
            }
            QTextBrowser table {
                border-collapse: collapse;
                margin: 15px 0;
                width: 100%;
            }
            QTextBrowser th, QTextBrowser td {
                border: 1px solid #ddd;
                padding: 8px 12px;
            }
            QTextBrowser th {
                background-color: #f2f2f2;
                font-weight: bold;
            }
            QTreeWidget {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
                font-size: 14px;
            }
            QTreeWidget::item {
                padding: 5px;
            }
            QTreeWidget::item:selected {
                background-color: #007bff;
                color: white;
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
        layout = QVBoxLayout(self)
        
        # 标题 - 压缩标题空间
        title_label = QLabel("LungVision 帮助文档")
        title_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #212529;
            margin-bottom: 5px;
            padding: 2px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 分割线 - 减少分割线占用空间
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("margin-bottom: 5px; margin-top: 0px;")
        layout.addWidget(line)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：目录树 - 减小目录树宽度
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabel("目录")
        self.tree_widget.setMinimumWidth(180)
        self.tree_widget.setMaximumWidth(250)
        self.populate_tree()
        splitter.addWidget(self.tree_widget)
        
        # 右侧：内容显示
        self.text_browser = QTextBrowser()
        self.text_browser.setOpenExternalLinks(True)
        self.text_browser.setSearchPaths([os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")])
        splitter.addWidget(self.text_browser)
        
        # 设置分割器比例 - 扩大内容区域
        splitter.setSizes([200, 700])
        layout.addWidget(splitter)
        
        # 设置分割器伸缩因子，使内容区域获得更多空间
        splitter.setStretchFactor(0, 1)  # 目录树
        splitter.setStretchFactor(1, 4)  # 内容区域
        
        # 底部按钮
        button_layout = QHBoxLayout()
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.accept)
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)
        
        # 连接信号
        self.tree_widget.itemClicked.connect(self.on_item_clicked)
        
        # 默认显示帮助索引
        self.text_browser.setSource(QUrl("help_index.md"))
    
    def populate_tree(self):
        """
        填充目录树
        """
        # 系统介绍
        system_item = QTreeWidgetItem(self.tree_widget, ["系统介绍"])
        system_item.setData(0, Qt.UserRole, "system_introduction.md")
        
        # 图像增强算法
        enhancement_item = QTreeWidgetItem(self.tree_widget, ["图像增强算法"])
        enhancement_item.setData(0, Qt.UserRole, "enhancement_algorithms.md")
        
        # 增强算法子项
        algorithms = [
            {"name": "直方图均衡化", "anchor": "1-直方图均衡化"},
            {"name": "自适应直方图均衡化(CLAHE)", "anchor": "2-自适应直方图均衡化clahe"},
            {"name": "伽马校正", "anchor": "3-伽马校正"},
            {"name": "非锐化掩蔽", "anchor": "4-非锐化掩蔽"},
            {"name": "小波去噪", "anchor": "5-小波去噪"}
        ]
        
        for algo in algorithms:
            item = QTreeWidgetItem(enhancement_item, [algo["name"]])
            item.setData(0, Qt.UserRole, "enhancement_algorithms.md#" + algo["anchor"])
        
        # 肺叶分割算法
        segmentation_item = QTreeWidgetItem(self.tree_widget, ["肺叶分割算法"])
        segmentation_item.setData(0, Qt.UserRole, "segmentation_algorithms.md")
        
        # 分割算法子项
        seg_algorithms = [
            {"name": "阈值分割", "anchor": "1-阈值分割"},
            {"name": "区域生长", "anchor": "2-区域生长"},
            {"name": "分水岭分割", "anchor": "3-分水岭分割"},
            {"name": "主动轮廓", "anchor": "4-主动轮廓snake算法"},
            {"name": "U-Net深度学习分割", "anchor": "5-u-net深度学习分割"}
        ]
        
        for algo in seg_algorithms:
            item = QTreeWidgetItem(segmentation_item, [algo["name"]])
            item.setData(0, Qt.UserRole, "segmentation_algorithms.md#" + algo["anchor"])
        
        # 常见问题
        faq_item = QTreeWidgetItem(self.tree_widget, ["常见问题"])
        faq_item.setData(0, Qt.UserRole, "faq.md")
        
        # 常见问题子项
        faq_sections = [
            {"name": "软件使用方法", "anchor": "软件使用方法"},
            {"name": "图像加载问题", "anchor": "图像加载问题"},
            {"name": "图像增强问题", "anchor": "图像增强问题"},
            {"name": "肺叶分割问题", "anchor": "肺叶分割问题"},
            {"name": "其他问题", "anchor": "其他问题"},
            {"name": "性能优化建议", "anchor": "性能优化建议"}
        ]
        
        for section in faq_sections:
            item = QTreeWidgetItem(faq_item, [section["name"]])
            item.setData(0, Qt.UserRole, f"faq.md#{section['anchor']}")
        
        # 展开所有项
        self.tree_widget.expandAll()
    
    def on_item_clicked(self, item, column):
        """
        处理目录项点击事件
        """
        url = item.data(0, Qt.UserRole)
        if url:
            # 设置文档源并滚动到指定位置
            self.text_browser.setSource(QUrl(url))
            
            # 如果URL包含锚点，确保滚动到正确位置
            if '#' in url:
                # QTextBrowser会自动处理锚点，但有时可能需要额外的滚动调整
                pass  # QTextBrowser已经自动处理了锚点跳转

class AlgorithmInfoDialog(QDialog):
    """
    算法信息对话框，用于显示单个算法的详细信息
    """
    def __init__(self, algorithm_name, algorithm_type, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{algorithm_name} - 算法说明")
        self.setMinimumSize(600, 400)
        
        # 设置对话框图标
        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "logo.png")
        self.setWindowIcon(QIcon(icon_path))
        
        # 设置对话框样式
        self.setStyleSheet("""
            QDialog {
                background-color: #f8f9fa;
            }
            QTextBrowser {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
                padding: 10px;
                font-size: 14px;
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
        layout = QVBoxLayout(self)
        
        # 标题
        title_label = QLabel(f"{algorithm_name}")
        title_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #212529;
            margin-bottom: 10px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 分割线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("margin-bottom: 10px;")
        layout.addWidget(line)
        
        # 内容显示
        self.text_browser = QTextBrowser()
        self.text_browser.setOpenExternalLinks(True)
        self.text_browser.setSearchPaths([os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")])
        layout.addWidget(self.text_browser)
        
        # 底部按钮
        button_layout = QHBoxLayout()
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.accept)
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)
        
        # 加载算法信息
        if algorithm_type == "enhancement":
            file_name = "enhancement_algorithms.md"
        else:  # segmentation
            file_name = "segmentation_algorithms.md"
        
        # 查找算法在文档中的位置
        docs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
        file_path = os.path.join(docs_path, file_name)
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # 查找算法部分
                algorithm_sections = {}
                current_section = None
                section_content = []
                
                for line in content.split('\n'):
                    if line.startswith('## '):
                        if current_section:
                            algorithm_sections[current_section] = '\n'.join(section_content)
                        current_section = line[3:].strip()
                        section_content = [line]
                    elif current_section:
                        section_content.append(line)
                
                if current_section:
                    algorithm_sections[current_section] = '\n'.join(section_content)
                
                # 查找匹配的算法
                for section, content in algorithm_sections.items():
                    if algorithm_name in section:
                        self.text_browser.setMarkdown(content)
                        break
                else:
                    self.text_browser.setMarkdown(f"# {algorithm_name}\n\n未找到该算法的详细说明。")
        else:
            self.text_browser.setMarkdown(f"# {algorithm_name}\n\n未找到算法文档。")