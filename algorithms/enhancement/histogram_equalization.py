import cv2
import numpy as np

class HistogramEqualization:
    """
    直方图均衡化算法
    通过重新分布图像的灰度值，提高图像对比度
    """
    
    def __init__(self):
        pass
    
    def process(self, image, **kwargs):
        """
        应用直方图均衡化
        
        参数:
            image: 输入图像 (灰度或彩色)
        
        返回:
            处理后的图像
        """
        # 确保图像是灰度图
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 应用直方图均衡化
        equalized = cv2.equalizeHist(gray)
        
        return equalized