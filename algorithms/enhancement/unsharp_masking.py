import cv2
import numpy as np

class UnsharpMasking:
    """
    非锐化掩蔽
    通过从原始图像中减去模糊版本来增强边缘
    """
    
    def __init__(self):
        pass
    
    def process(self, image, radius=5, amount=1.5, **kwargs):
        """
        应用非锐化掩蔽
        
        参数:
            image: 输入图像 (灰度或彩色)
            radius: 高斯模糊的半径
            amount: 锐化强度
        
        返回:
            处理后的图像
        """
        # 确保图像是灰度图
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 创建模糊版本
        blurred = cv2.GaussianBlur(gray, (0, 0), radius)
        
        # 计算掩蔽 (原始 - 模糊)
        mask = cv2.subtract(gray, blurred)
        
        # 应用掩蔽到原始图像
        sharpened = cv2.addWeighted(gray, 1.0, mask, amount, 0)
        
        return sharpened