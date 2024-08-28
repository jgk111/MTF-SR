import cv2
import numpy as np

# 下采样
def downsample_dem(high_res_dem, scale=2):
    lr_dem = cv2.resize(high_res_dem,
                        (high_res_dem.shape[1] // scale, high_res_dem.shape[0] // scale),
                        interpolation=cv2.INTER_CUBIC)
    return lr_dem
# 归一化
def normalize_dem(dem):
    min_value = np.min(dem)
    max_value = np.max(dem)
    normalized_dem = (dem - min_value) / (max_value - min_value)
    return normalized_dem, min_value, max_value
# 反归一化
def denormalize_dem(normalized_dem, min_value, max_value):
    original_dem = normalized_dem * (max_value - min_value) + min_value
    return original_dem
# 坡度和坡向的计算
def calculate_slope_aspect(dem, cellsize):
    dzdx = (np.roll(dem, -1, axis=1) - np.roll(dem, 1, axis=1)) / (2 * cellsize)
    dzdy = (np.roll(dem, -1, axis=0) - np.roll(dem, 1, axis=0)) / (2 * cellsize)
    
    # Calculate slope
    slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2))

    # Calculate aspect
    aspect = np.arctan2(dzdy, dzdx)
    aspect = np.degrees(aspect)
    aspect = np.where(aspect < 0, 360 + aspect, aspect)
    
    return slope, aspect