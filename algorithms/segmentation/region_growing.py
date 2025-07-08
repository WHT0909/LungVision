import cv2
import numpy as np
from queue import Queue

class RegionGrowing:
    """
    区域生长分割
    从种子点开始，逐步将相似像素添加到区域中
    """
    
    def __init__(self):
        pass
    
    def process(self, image, seed_point=(0, 0), threshold=10, **kwargs):
        """
        应用区域生长分割
        
        参数:
            image: 输入图像 (灰度或彩色)
            seed_point: 种子点坐标 (x, y)
            threshold: 区域生长阈值
        
        返回:
            分割后的图像
        """
        # 确保图像是灰度图
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 创建掩码
        h, w = gray.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 获取种子点的灰度值
        x, y = seed_point
        seed_value = gray[y, x]
        
        # 初始化队列和已访问集合
        q = Queue()
        q.put((x, y))
        visited = set([(x, y)])
        
        # 定义4连通邻域
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # 区域生长
        while not q.empty():
            x, y = q.get()
            
            # 将当前点添加到区域
            mask[y, x] = 255
            
            # 检查邻域
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                
                # 检查边界
                if nx < 0 or ny < 0 or nx >= w or ny >= h:
                    continue
                
                # 检查是否已访问
                if (nx, ny) in visited:
                    continue
                
                # 检查相似性
                if abs(int(gray[ny, nx]) - int(seed_value)) <= threshold:
                    q.put((nx, ny))
                    visited.add((nx, ny))
        
        # 应用掩码到原始图像
        segmented = cv2.bitwise_and(gray, gray, mask=mask)
        
        return segmented