import cv2

# 下采样
def downsample_dem(high_res_dem, scale=2):
    lr_dem = cv2.resize(high_res_dem,
                        (high_res_dem.shape[1] // scale, high_res_dem.shape[0] // scale),
                        interpolation=cv2.INTER_LINEAR)
    return lr_dem