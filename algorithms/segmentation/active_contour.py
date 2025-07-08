import cv2
import numpy as np
from skimage.segmentation import active_contour

class ActiveContour:
    """
    主动轮廓分割 (Snake算法)
    使用能量最小化的曲线拟合目标边界
    """
    
    def __init__(self):
        pass
    
    def process(self, image, iterations=100, alpha=0.15, beta=0.10, **kwargs):
        """
        应用主动轮廓分割
        
        参数:
            image: 输入图像 (灰度或彩色)
            iterations: 迭代次数
            alpha: 曲线的弹性参数
            beta: 曲线的刚性参数
        
        返回:
            分割后的图像
        """
        # 确保图像是灰度图
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 创建初始轮廓 (椭圆)
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        radius_y, radius_x = h // 3, w // 3
        
        t = np.linspace(0, 2*np.pi, 100)
        init_x = center_x + radius_x * np.cos(t)
        init_y = center_y + radius_y * np.sin(t)
        init = np.array([init_x, init_y]).T
        
        # 应用主动轮廓
        try:
            snake = active_contour(
                gray, 
                init, 
                alpha=alpha, 
                beta=beta, 
                gamma=0.001,
                max_iterations=iterations
            )
            
            # 创建掩码
            mask = np.zeros_like(gray)
            snake_int = np.rint(snake).astype(np.int32)
            cv2.fillPoly(mask, [snake_int], 255)
            
            # 应用掩码到原始图像
            segmented = cv2.bitwise_and(gray, gray, mask=mask)
            
            return segmented
        except Exception as e:
            # 如果主动轮廓失败，返回简单的椭圆分割
            mask = np.zeros_like(gray)
            cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, 255, -1)
            segmented = cv2.bitwise_and(gray, gray, mask=mask)
            return segmented