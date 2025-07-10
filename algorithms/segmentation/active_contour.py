import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
# # 解决窗口中文乱码
# cv.namedWindow("dummy", cv.WINDOW_NORMAL)
# cv.destroyWindow("dummy")
# cv.waitKey(1)
def getGaussianPE(src):
    """ 描述：计算负高斯势能(Negative Gaussian Potential Energy, NGPE) 输入：单通道灰度图src 输出：无符号的浮点型单通道，取值0.0 ~ 255.0 """
    imblur = cv.GaussianBlur(src, ksize=(5, 5), sigmaX=3)
    dx = cv.Sobel(imblur, cv.CV_16S, 1, 0)  # X方向上取一阶导数，16位有符号数，卷积核3x3
    dy = cv.Sobel(imblur, cv.CV_16S, 0, 1)
    E = dx**2 + dy**2
    return E

def getDiagCycleMat(alpha, beta, n):
    """ 计算5对角循环矩阵 """
    a = 2 * alpha + 6 * beta
    b = -(alpha + 4 * beta)
    c = beta
    diag_mat_a = a * np.eye(n)
    diag_mat_b = b * np.roll(np.eye(n), 1, 0) + b * np.roll(np.eye(n), -1, 0)
    diag_mat_c = c * np.roll(np.eye(n), 2, 0) + c * np.roll(np.eye(n), -2, 0)
    return diag_mat_a + diag_mat_b + diag_mat_c

def getCircleContour(centre=(0, 0), radius=(1, 1), N=200):
    """ 以参数方程的形式，获取n个离散点围成的圆形/椭圆形轮廓 输入：中心centre=（x0, y0）, 半轴长radius=(a, b)， 离散点数N 输出：由离散点坐标(x, y)组成的2xN矩阵 """
    t = np.linspace(0, 2 * np.pi, N)
    x = centre[0] + radius[0] * np.cos(t)
    y = centre[1] + radius[1] * np.sin(t)
    return np.array([x, y])

def snake(img, snake, alpha=0.5, beta=0.1, gamma=0.1, max_iter=2500, convergence=0.01):
    """ 根据Snake模型的隐式格式进行迭代 输入：弹力系数alpha，刚性系数beta，迭代步长gamma，最大迭代次数max_iter，收敛阈值convergence 输出：由收敛轮廓坐标(x, y)组成的2xN矩阵， 历次迭代误差list """
    x, y, errs = snake[0].copy(), snake[1].copy(), []
    n = len(x)
    # 计算5对角循环矩阵A，及其相关逆阵
    A = getDiagCycleMat(alpha, beta, n)
    inv = np.linalg.inv(A + gamma * np.eye(n))
    # 初始化
    y_max, x_max = img.shape
    max_px_move = 1.0
    # 计算负高斯势能矩阵，及其梯度
    E_ext = -getGaussianPE(img)
    fx = cv.Sobel(E_ext, cv.CV_16S, 1, 0)
    fy = cv.Sobel(E_ext, cv.CV_16S, 0, 1)
    T = np.max([abs(fx), abs(fy)])
    fx, fy = fx / T, fy / T
    for g in range(max_iter):
        x_pre, y_pre = x.copy(), y.copy()
        i, j = np.uint8(y), np.uint8(x)
        try:
            xn = inv @ (gamma * x + fx[i, j])
            yn = inv @ (gamma * y + fy[i, j])
        except Exception as e:
            print("索引超出范围")
        # 判断收敛
        x, y = xn, yn
        err = np.mean(0.5 * np.abs(x_pre - x) + 0.5 * np.abs(y_pre - y))
        errs.append(err)
        if err < convergence:
            print(f"Snake迭代{g}次后，趋于收敛。\t err = {err:.3f}")
            break
    return x, y, errs


class ActiveContour:
    """
    主动轮廓分割 (Snake算法)
    使用能量最小化的曲线拟合目标边界
    """

    def __init__(self):
        self.selected_points = []
        self.is_selecting = False

    def select_initial_contour(self, image):
        """
        允许用户手动选择初始轮廓的点

        参数:
            image: 输入图像

        返回:
            包含用户选择点的轮廓数组
        """
        # 创建一个窗口用于选择点
        window_name = "Select Initial Contour Points (At least 6 points, Press Enter to confirm)"
        self.selected_points = []
        self.is_selecting = True
        clone = image.copy()

        # 鼠标回调函数
        def mouse_callback(event, x, y, flags, param):
            if not self.is_selecting:
                return
                
            if event == cv.EVENT_LBUTTONDOWN:
                self.selected_points.append((x, y))
                # 在图像上标记选择的点
                cv.circle(clone, (x, y), 3, (0, 255, 0), -1)
                # 如果有多个点，绘制连接线
                if len(self.selected_points) > 1:
                    cv.line(clone, self.selected_points[-2], self.selected_points[-1], (0, 255, 0), 1)
                # 如果至少有3个点，连接第一个点和最后一个点形成闭合轮廓
                if len(self.selected_points) > 2:
                    cv.line(clone, self.selected_points[0], self.selected_points[-1], (0, 255, 0), 1, cv.LINE_AA)
                cv.imshow(window_name, clone)

        # 创建窗口并设置鼠标回调
        cv.namedWindow(window_name)
        cv.setMouseCallback(window_name, mouse_callback)
        cv.imshow(window_name, clone)

        # 等待用户选择点并按Enter确认
        while self.is_selecting:
            key = cv.waitKey(1) & 0xFF
            if key == 13:  # Enter键
                if len(self.selected_points) >= 6:  # 至少需要6个点
                    self.is_selecting = False
                else:
                    print("请至少选择6个点")
            elif key == 27:  # Esc键
                self.selected_points = []
                self.is_selecting = False

        cv.destroyWindow(window_name)

        # 如果用户取消选择或没有选择足够的点，返回None
        if not self.selected_points or len(self.selected_points) < 6:
            return None

        # 将选择的点转换为适合snake算法的格式
        points = np.array(self.selected_points)
        x = points[:, 0]
        y = points[:, 1]

        # 使用插值增加点的数量，使轮廓更平滑
        t = np.arange(len(x))
        t_interp = np.linspace(0, len(x) - 1, 200)
        x_interp = np.interp(t_interp, t, x, period=len(x))
        y_interp = np.interp(t_interp, t, y, period=len(x))

        return np.array([x_interp, y_interp])

    def process(self, image, iterations=100, alpha=0.15, beta=0.10, manual_init=True, **kwargs):
        """
        应用主动轮廓分割

        参数:
            image: 输入图像 (灰度或彩色)
            iterations: 迭代次数
            alpha: 曲线的弹性参数
            beta: 曲线的刚性参数
            manual_init: 是否手动选择初始轮廓

        返回:
            分割后的图像
        """
        # 确保图像是灰度图
        if len(image.shape) > 2:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 初始轮廓
        init = None
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        radius_y, radius_x = h // 3, w // 3

        # 如果启用手动选择，让用户选择初始轮廓
        if manual_init:
            init = self.select_initial_contour(image if len(image.shape) > 2 else cv.cvtColor(gray, cv.COLOR_GRAY2BGR))

        # 如果用户没有选择或取消选择，使用默认椭圆轮廓
        if init is None:
            init = getCircleContour((center_x, center_y), (radius_x, radius_y), N=200)

        # 应用主动轮廓
        try:
            x, y, _ = snake(gray, snake=init, alpha=alpha, beta=beta, gamma=0.1, max_iter=iterations)

            # 创建掩码
            mask = np.zeros_like(gray)
            snake_int = np.rint(np.array([x, y]).T).astype(np.int32)
            cv.fillPoly(mask, [snake_int], 255)

            # 应用掩码到原始图像
            segmented = cv.bitwise_and(gray, gray, mask=mask)

            return segmented
        except Exception as e:
            print(f"主动轮廓算法失败: {str(e)}")
            # 如果主动轮廓失败，返回简单的椭圆分割或用户选择的轮廓
            mask = np.zeros_like(gray)
            if manual_init and self.selected_points and len(self.selected_points) >= 3:
                # 使用用户选择的点创建掩码
                points = np.array(self.selected_points).reshape((-1, 1, 2)).astype(np.int32)
                cv.fillPoly(mask, [points], 255)
            else:
                # 使用默认椭圆
                cv.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, 255, -1)
            
            segmented = cv.bitwise_and(gray, gray, mask=mask)
            return segmented