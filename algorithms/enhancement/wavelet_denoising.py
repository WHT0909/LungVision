import cv2
import numpy as np
import pywt

class WaveletDenoising:
    """
    小波去噪
    使用小波变换去除图像噪声，保留重要特征
    """
    
    def __init__(self):
        pass
    
    def process(self, image, threshold=30, wavelet='db1', **kwargs):
        """
        应用小波去噪
        
        参数:
            image: 输入图像 (灰度或彩色)
            threshold: 阈值，控制去噪强度
            wavelet: 小波类型
        
        返回:
            处理后的图像
        """
        # 确保图像是灰度图
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 小波分解
        coeffs = pywt.wavedec2(gray, wavelet, level=3)
        
        # 阈值处理
        for i in range(1, len(coeffs)):
            for j in range(len(coeffs[i])):
                coeffs[i][j] = pywt.threshold(coeffs[i][j], threshold, 'soft')
        
        # 小波重构
        denoised = pywt.waverec2(coeffs, wavelet)
        
        # 确保尺寸与原图一致
        denoised = denoised[:gray.shape[0], :gray.shape[1]]
        
        # 转换为uint8类型
        denoised = np.uint8(denoised)
        
        return denoised