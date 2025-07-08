import cv2
import numpy as np

class CLAHE:
    """
    对比度受限的自适应直方图均衡化 (CLAHE)
    在小区域内应用直方图均衡化，并限制对比度以减少噪声放大
    """
    
    def __init__(self):
        pass
    
    def process(self, image, clip_limit=2.0, grid_size=8, **kwargs):
        """
        应用CLAHE算法
        
        参数:
            image: 输入图像 (灰度或彩色)
            clip_limit: 对比度限制阈值
            grid_size: 网格大小
        
        返回:
            处理后的图像
        """
        # 确保图像是灰度图
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        
        # 应用CLAHE
        enhanced = clahe.apply(gray)
        
        return enhanced