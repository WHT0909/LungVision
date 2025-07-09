# LungVision 开发者文档

## 1. 系统架构概述

### 1.1 项目结构

LungVision 项目采用模块化设计，清晰地分离了用户界面、算法实现和资源文件，使系统具有良好的可维护性和可扩展性。项目结构如下：

```
LungVision/
├── algorithms/             # 算法实现模块
│   ├── enhancement/        # 图像增强算法
│   │   ├── clahe.py
│   │   ├── gamma_correction.py
│   │   ├── histogram_equalization.py
│   │   ├── unsharp_masking.py
│   │   └── wavelet_denoising.py
│   └── segmentation/        # 肺叶分割算法
│       ├── active_contour.py
│       ├── region_growing.py
│       ├── thresholding.py
│       ├── unet.py
│       └── watershed.py
├── docs/                    # 文档
│   ├── enhancement_algorithms.md
│   ├── faq.md
│   ├── help_index.md
│   ├── segmentation_algorithms.md
│   └── system_introduction.md
├── ui/                      # 用户界面
│   ├── help_dialog.py
│   ├── main_window.py
│   └── welcome_dialog.py
├── x-ray/                   # 示例图像
├── main.py                  # 程序入口
└── requirements.txt         # 依赖包列表
```

### 1.2 系统架构设计

LungVision 采用经典的 MVC（Model-View-Controller）架构模式：

- **Model（模型）**：`algorithms` 目录中的各个算法类，负责数据处理和业务逻辑
- **View（视图）**：`ui` 目录中的界面类，负责用户界面的显示和交互
- **Controller（控制器）**：`main_window.py` 中的 `MainWindow` 类，协调模型和视图之间的交互

这种架构设计使得数据处理逻辑与界面表现分离，便于独立开发和测试，同时也方便后续扩展新的算法或修改界面。

### 1.3 技术栈

- **编程语言**：Python 3.7+
- **GUI 框架**：PyQt5
- **图像处理**：OpenCV, NumPy
- **数据可视化**：Matplotlib
- **高级图像处理**：scikit-image, PyWavelets

## 2. 核心模块详解

### 2.1 用户界面模块

#### 2.1.1 主窗口（MainWindow）

`MainWindow` 类是整个应用的核心控制器，负责协调用户界面与算法模块之间的交互。主要功能包括：

- 初始化用户界面组件
- 处理用户输入事件
- 调用相应的算法处理图像
- 显示处理结果
- 管理应用状态

主窗口界面采用分区设计：

- **顶部控制区**：包含加载图像、保存结果和重置按钮
- **中央显示区**：左侧显示原始图像，右侧显示处理结果，每个图像下方显示对应的直方图
- **右侧控制面板**：包含两个选项卡（图像增强和肺叶分割），每个选项卡中包含算法选择和参数调整控件

#### 2.1.2 图像显示组件（ImageViewer）

`ImageViewer` 类是一个自定义组件，用于显示图像及其直方图。主要功能包括：

- 显示图像（支持灰度和彩色图像）
- 自动调整图像大小以适应显示区域
- 实时显示图像的直方图

#### 2.1.3 辅助对话框

- **WelcomeDialog**：欢迎对话框，在应用启动时显示
- **HelpDialog**：帮助对话框，显示系统使用帮助
- **AlgorithmInfoDialog**：算法信息对话框，显示当前选择的算法的详细说明

### 2.2 图像增强模块

图像增强模块包含 5 种不同的算法，每种算法都被封装为一个独立的类，具有统一的接口（`process` 方法），便于在主程序中统一调用。

#### 2.2.1 直方图均衡化（HistogramEqualization）

**原理**：通过重新分配图像的灰度值分布，使其在整个灰度范围内更加均匀，从而提高图像的整体对比度。

**实现**：使用 OpenCV 的 `equalizeHist` 函数实现。

**特点**：
- 无需参数设置，操作简单
- 全局处理，适合整体对比度较低的图像
- 可能会过度增强噪声

#### 2.2.2 自适应直方图均衡化（CLAHE）

**原理**：将图像分割成多个小块，在每个小块上分别应用直方图均衡化，然后使用双线性插值合并结果。同时，它限制了对比度放大的程度，避免了噪声的过度放大。

**实现**：使用 OpenCV 的 `createCLAHE` 和 `apply` 函数实现。

**参数**：
- `clip_limit`：对比度限制，控制对比度增强的程度
- `grid_size`：网格大小，定义局部区域的大小

**特点**：
- 局部自适应处理，保留更多图像细节
- 通过对比度限制机制有效控制噪声放大
- 对光照不均匀的图像效果特别好

#### 2.2.3 伽马校正（GammaCorrection）

**原理**：通过幂律变换调整图像的亮度和对比度。公式为：输出 = 输入^γ（归一化后）。

**实现**：使用 NumPy 的数学函数实现幂律变换。

**参数**：
- `gamma`：伽马值，控制亮度调整的程度

**特点**：
- 非线性调整，可以有效处理过暗或过亮的图像
- 简单直观，易于理解和实现
- 可以增强特定亮度区域的细节

#### 2.2.4 非锐化掩蔽（UnsharpMasking）

**原理**：通过从原始图像中减去其模糊版本，然后将差异加回原图，从而增强边缘和细节。

**实现**：使用 OpenCV 的 `GaussianBlur` 函数生成模糊图像，然后进行图像运算。

**参数**：
- `radius`：高斯模糊的半径，控制增强的细节尺度
- `amount`：锐化的程度，值越大，边缘增强越明显

**特点**：
- 可以有效增强边缘和细节
- 保留图像整体亮度
- 参数直观易调整

#### 2.2.5 小波去噪（WaveletDenoising）

**原理**：利用小波变换将图像分解为不同频率和尺度的分量，通过阈值处理去除噪声分量，然后重建图像。

**实现**：使用 PyWavelets 库进行小波分解、阈值处理和重建。

**参数**：
- `threshold`：阈值，控制去噪强度
- `wavelet`：小波类型，使用的小波函数类型

**特点**：
- 可以有效去除噪声同时保留边缘
- 多尺度分析能力强
- 适合处理不同类型的噪声

### 2.3 肺叶分割模块

肺叶分割模块包含 5 种不同的算法，每种算法都被封装为一个独立的类，具有统一的接口（`process` 方法），便于在主程序中统一调用。

#### 2.3.1 阈值分割（Thresholding）

**原理**：通过设定一个阈值，将图像中的像素分为两类：高于阈值的像素归为一类，低于阈值的像素归为另一类。

**实现**：使用 OpenCV 的 `threshold` 函数实现。

**参数**：
- `threshold`：阈值，分割的灰度值界限
- `threshold_type`：阈值类型，如二值化、反二值化等

**特点**：
- 实现简单，计算速度快
- 直观易理解
- 适合对比度高的图像

#### 2.3.2 区域生长（RegionGrowing）

**原理**：从种子点开始，逐步将相似的邻近像素添加到区域中，基于区域的同质性原则。

**实现**：使用递归或队列实现区域生长算法。

**参数**：
- `seed_point`：种子点，生长起始位置
- `threshold`：阈值，相似性判断标准

**特点**：
- 考虑像素的空间连通性
- 可以生成连通的区域
- 对噪声有一定的抵抗力

#### 2.3.3 分水岭分割（Watershed）

**原理**：将图像视为地形表面，灰度值代表高度。算法模拟水从最低点开始灌注，当来自不同盆地的水即将汇合时，建立分水岭线（边界）。

**实现**：使用 OpenCV 的 `watershed` 函数实现。

**参数**：
- `distance`：距离阈值，控制分割的精细程度

**特点**：
- 总是产生闭合的边界
- 可以分割相互接触的对象
- 对边缘有良好的响应

#### 2.3.4 主动轮廓（ActiveContour）

**原理**：主动轮廓是一种能量最小化的曲线，它通过内部力（控制曲线的平滑度）和外部力（吸引曲线向图像边缘移动）的平衡，逐步拟合目标边界。

**实现**：使用 scikit-image 的 `active_contour` 函数实现。

**参数**：
- `iterations`：迭代次数，轮廓演化的步数
- `alpha`：控制曲线的弹性（抵抗拉伸）
- `beta`：控制曲线的刚性（抵抗弯曲）

**特点**：
- 可以生成平滑连续的边界
- 对噪声有一定的抵抗力
- 可以整合先验知识

#### 2.3.5 U-Net深度学习分割（UNet）

**原理**：U-Net是一种基于卷积神经网络的图像分割方法，它采用编码器-解码器结构，通过下采样捕获上下文信息，然后通过上采样恢复空间分辨率，同时使用跳跃连接保留细节信息。

**实现**：在当前版本中，使用简单的图像处理操作模拟 U-Net 的行为。在实际应用中，应该使用 TensorFlow 或 PyTorch 实现真正的 U-Net 模型。

**参数**：
- `model_type`：模型类型，预训练或自定义训练
- `confidence`：置信度阈值，控制像素被分类为前景的概率阈值

**特点**：
- 分割精度高
- 可以学习复杂的特征和模式
- 对噪声和变化有较强的鲁棒性

## 3. 系统工作流程

### 3.1 启动流程

1. 用户启动 `main.py`
2. 创建 `QApplication` 实例
3. 创建 `MainWindow` 实例
4. 显示欢迎对话框
5. 显示主窗口
6. 进入事件循环

### 3.2 图像处理流程

#### 3.2.1 图像增强流程

1. 用户加载图像
2. 选择增强算法
3. 调整算法参数
4. 点击"应用增强"按钮
5. `MainWindow` 获取当前选择的算法和参数
6. 调用相应的算法类的 `process` 方法
7. 显示处理结果和直方图
8. 启用保存按钮

#### 3.2.2 肺叶分割流程

1. 用户加载图像（或使用已增强的图像）
2. 选择分割算法
3. 调整算法参数
4. 点击"应用分割"按钮
5. `MainWindow` 获取当前选择的算法和参数
6. 调用相应的算法类的 `process` 方法
7. 显示分割结果
8. 启用保存按钮

### 3.3 数据流

1. 图像数据从文件系统加载到内存
2. 原始图像显示在左侧面板
3. 图像数据传递给算法模块处理
4. 处理后的图像返回给主窗口
5. 处理结果显示在右侧面板
6. 用户可以将结果保存到文件系统

## 4. 开发指南

### 4.1 环境配置

1. 克隆或下载项目到本地
2. 安装 Python 3.7 或更高版本
3. 安装依赖包：
   ```
   pip install -r requirements.txt
   ```

### 4.2 添加新的增强算法

1. 在 `algorithms/enhancement/` 目录下创建新的 Python 文件，如 `new_algorithm.py`
2. 定义一个新的算法类，实现 `process` 方法：
   ```python
   import cv2
   import numpy as np

   class NewAlgorithm:
       def __init__(self):
           pass

       def process(self, image, param1=default1, param2=default2, **kwargs):
           # 实现算法逻辑
           # 处理图像
           # 返回处理后的图像
           return processed_image
   ```
3. 在 `main_window.py` 中导入新算法：
   ```python
   from algorithms.enhancement.new_algorithm import NewAlgorithm
   ```
4. 在 `MainWindow.__init__` 方法中添加算法实例：
   ```python
   self.enhancement_algorithms[5] = NewAlgorithm()
   ```
5. 在 `MainWindow.__init__` 方法中更新算法下拉框：
   ```python
   self.enhancement_combo.addItems([
       "直方图均衡化",
       "自适应直方图均衡化(CLAHE)",
       "伽马校正",
       "非锐化掩蔽",
       "小波去噪",
       "新算法名称"
   ])
   ```
6. 在 `update_enhancement_params` 方法中添加新算法的参数控件
7. 在 `apply_enhancement` 方法中添加新算法的参数获取逻辑
8. 在 `docs/enhancement_algorithms.md` 中添加新算法的说明文档

### 4.3 添加新的分割算法

1. 在 `algorithms/segmentation/` 目录下创建新的 Python 文件，如 `new_segmentation.py`
2. 定义一个新的算法类，实现 `process` 方法
3. 在 `main_window.py` 中导入新算法
4. 在 `MainWindow.__init__` 方法中添加算法实例
5. 在 `MainWindow.__init__` 方法中更新算法下拉框
6. 在 `update_segmentation_params` 方法中添加新算法的参数控件
7. 在 `apply_segmentation` 方法中添加新算法的参数获取逻辑
8. 在 `docs/segmentation_algorithms.md` 中添加新算法的说明文档

### 4.4 修改用户界面

1. 在 `ui/main_window.py` 中修改 `MainWindow` 类的界面初始化代码
2. 添加新的控件和布局
3. 连接信号和槽函数
4. 更新样式表

### 4.5 调试技巧

1. 使用 `print` 语句或日志记录中间结果
2. 使用 Matplotlib 可视化中间处理步骤
3. 使用 OpenCV 的 `imshow` 函数显示中间图像
4. 使用 Python 的调试器（如 pdb）进行断点调试

## 5. 性能优化建议

### 5.1 图像处理优化

1. 对大尺寸图像进行预处理，如缩放到合适的大小
2. 使用 NumPy 的向量化操作代替循环
3. 考虑使用多线程处理耗时的算法
4. 对于 U-Net 等深度学习模型，考虑使用 GPU 加速

### 5.2 界面响应优化

1. 在处理大图像时，显示进度条或加载动画
2. 使用 QThread 将耗时操作放在后台线程中执行
3. 使用信号-槽机制更新界面，避免界面冻结

### 5.3 内存管理

1. 及时释放不再使用的大型数据
2. 避免不必要的深拷贝
3. 对于大型图像，考虑使用内存映射或分块处理

## 6. 未来扩展方向

### 6.1 功能扩展

1. 添加更多图像增强和分割算法
2. 实现真正的 U-Net 深度学习分割
3. 添加批处理功能，支持处理多张图像
4. 添加图像标注功能，支持手动分割
5. 添加测量工具，如面积、周长、密度等
6. 添加 3D 重建功能，支持 CT 序列处理

### 6.2 技术改进

1. 使用 TensorFlow 或 PyTorch 实现深度学习模型
2. 添加模型训练界面，支持用户自定义训练
3. 优化算法性能，支持实时处理
4. 添加插件系统，支持第三方扩展

### 6.3 用户体验改进

1. 添加多语言支持
2. 添加主题切换功能
3. 添加用户配置保存和加载功能
4. 添加快捷键和工具提示
5. 添加撤销/重做功能

## 7. 常见问题与解决方案

### 7.1 安装问题

**问题**：安装依赖包时出错
**解决方案**：
- 确保 Python 版本兼容（3.7+）
- 尝试使用虚拟环境安装
- 对于 OpenCV 安装问题，可以尝试使用预编译的二进制包

### 7.2 运行问题

**问题**：程序启动时报错
**解决方案**：
- 检查是否所有依赖包都已正确安装
- 检查 Python 路径是否正确
- 检查文件权限

**问题**：处理大图像时程序崩溃
**解决方案**：
- 增加系统内存
- 在处理前缩小图像尺寸
- 使用分块处理技术

### 7.3 算法问题

**问题**：分割结果不理想
**解决方案**：
- 尝试不同的算法和参数组合
- 先进行图像增强，再进行分割
- 对于复杂图像，考虑使用 U-Net 等深度学习方法

**问题**：U-Net 模型未加载
**解决方案**：
- 当前版本中 U-Net 为模拟实现，需要开发者自行实现真正的深度学习模型
- 参考 TensorFlow 或 PyTorch 的 U-Net 实现

## 8. 参考资料

### 8.1 技术文档

- [PyQt5 官方文档](https://www.riverbankcomputing.com/static/Docs/PyQt5/)
- [OpenCV 官方文档](https://docs.opencv.org/)
- [NumPy 官方文档](https://numpy.org/doc/)
- [Matplotlib 官方文档](https://matplotlib.org/stable/contents.html)
- [scikit-image 官方文档](https://scikit-image.org/docs/stable/)
- [PyWavelets 官方文档](https://pywavelets.readthedocs.io/)

### 8.2 算法参考

- 直方图均衡化：[OpenCV 直方图均衡化教程](https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html)
- CLAHE：[OpenCV CLAHE 教程](https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html)
- 伽马校正：[图像伽马校正原理](https://en.wikipedia.org/wiki/Gamma_correction)
- 非锐化掩蔽：[非锐化掩蔽原理](https://en.wikipedia.org/wiki/Unsharp_masking)
- 小波去噪：[PyWavelets 去噪示例](https://pywavelets.readthedocs.io/en/latest/ref/thresholding-functions.html)
- 阈值分割：[OpenCV 阈值分割教程](https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html)
- 区域生长：[区域生长算法原理](https://en.wikipedia.org/wiki/Region_growing)
- 分水岭分割：[OpenCV 分水岭分割教程](https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html)
- 主动轮廓：[scikit-image 主动轮廓教程](https://scikit-image.org/docs/dev/auto_examples/edges/plot_active_contours.html)
- U-Net：[U-Net 原始论文](https://arxiv.org/abs/1505.04597)

### 8.3 学习资源

- [Python 图像处理入门](https://www.pyimagesearch.com/)
- [数字图像处理基础](https://www.amazon.com/Digital-Image-Processing-Rafael-Gonzalez/dp/0133356728)
- [医学图像分析教程](https://www.sciencedirect.com/book/9780123735850/medical-image-processing)
- [深度学习与医学图像分析](https://www.springer.com/gp/book/9783319429984)

## 9. 贡献指南

### 9.1 代码风格

- 遵循 PEP 8 Python 代码风格指南
- 使用有意义的变量名和函数名
- 添加适当的注释和文档字符串
- 保持代码简洁清晰

### 9.2 提交流程

1. Fork 项目仓库
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request
5. 等待代码审查和合并

### 9.3 报告问题

- 使用 GitHub Issues 报告问题
- 提供详细的问题描述和复现步骤
- 如果可能，附上截图或错误日志

## 10. 版权和许可

LungVision 是一个用于教育和研究目的的开源项目，欢迎贡献代码和提出改进建议。

© 2025 WangHaotian SCU BME

联系方式：wanghaotian70094 [at] foxmail.com