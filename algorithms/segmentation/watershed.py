import cv2
import numpy as np

class Watershed:
    """
    分水岭分割
    将图像视为地形表面，通过模拟水流分割区域
    """
    
    def __init__(self):
        pass
    
    def process(self, image, distance=9, **kwargs):
        """
        应用分水岭分割
        
        参数:
            image: 输入图像 (灰度或彩色)
            distance: 距离变换阈值
        
        返回:
            分割后的图像
        """
        # 确保图像是灰度图
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 阈值处理
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 噪声去除
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 确定背景区域
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # 距离变换
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        
        # 确定前景区域
        _, sure_fg = cv2.threshold(dist_transform, distance * dist_transform.max() / 100, 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # 查找未知区域
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # 标记
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # 创建彩色图像用于分水岭算法
        if len(image.shape) == 2:
            color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            color_img = image.copy()
        
        # 应用分水岭算法
        markers = cv2.watershed(color_img, markers)
        
        # 创建结果图像
        result = np.zeros_like(gray)
        result[markers > 1] = 255
        
        return result