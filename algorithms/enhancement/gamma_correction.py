import cv2
import numpy as np

class GammaCorrection:
    """
    伽马校正
    通过非线性变换调整图像亮度和对比度
    """
    
    def __init__(self):
        pass
    
    def process(self, image, gamma=1.0, **kwargs):
        """
        应用伽马校正
        
        参数:
            image: 输入图像 (灰度或彩色)
            gamma: 伽马值 (>1增加亮度, <1降低亮度)
        
        返回:
            处理后的图像
        """
        # 确保图像是灰度图
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 归一化到0-1范围
        normalized = gray / 255.0
        
        # 应用伽马校正
        corrected = np.power(normalized, 1.0/gamma)
        
        # 转换回0-255范围
        corrected = np.uint8(corrected * 255)
        
        return corrected