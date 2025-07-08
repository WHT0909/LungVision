import cv2
import numpy as np

class Thresholding:
    """
    阈值分割
    基于像素强度阈值将图像分割为前景和背景
    """
    
    def __init__(self):
        pass
    
    def process(self, image, threshold=127, threshold_type=0, **kwargs):
        """
        应用阈值分割
        
        参数:
            image: 输入图像 (灰度或彩色)
            threshold: 阈值 (0-255)
            threshold_type: 阈值类型
                0: 二值化 (THRESH_BINARY)
                1: 反二值化 (THRESH_BINARY_INV)
                2: 截断 (THRESH_TRUNC)
                3: 阈值为零 (THRESH_TOZERO)
                4: 反阈值为零 (THRESH_TOZERO_INV)
        
        返回:
            分割后的图像
        """
        # 确保图像是灰度图
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 映射阈值类型
        thresh_types = [
            cv2.THRESH_BINARY,
            cv2.THRESH_BINARY_INV,
            cv2.THRESH_TRUNC,
            cv2.THRESH_TOZERO,
            cv2.THRESH_TOZERO_INV
        ]
        
        # 应用阈值
        _, segmented = cv2.threshold(gray, threshold, 255, thresh_types[threshold_type])
        
        return segmented