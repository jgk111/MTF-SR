import numpy as np

# IDW插值函数
def idw_interpolation(x, y, z, xi, yi, power=2):
    dist = np.sqrt((x - xi) ** 2 + (y - yi) ** 2)
    dist[dist == 0] = 1e-10
    weights = 1 / dist**power
    zi = np.sum(weights * z) / np.sum(weights)
    return zi

# IDW局部插值优化
def local_interpolation_optimization_with_idw(predicted_dem, low_res_dem):
    hr_height, hr_width = predicted_dem.shape
    lr_height, lr_width = low_res_dem.shape
    optimized_dem = np.zeros_like(predicted_dem)

    for i in range(hr_height):
        for j in range(hr_width):
            x_lr = i * (lr_height / hr_height)
            y_lr = j * (lr_width / hr_width)

            x0, x1 = int(np.floor(x_lr)), int(np.ceil(x_lr))
            y0, y1 = int(np.floor(y_lr)), int(np.ceil(y_lr))

            x_points = np.array([x0, x0, x1, x1])
            y_points = np.array([y0, y1, y0, y1])
            z_points = np.array([
                low_res_dem[x0, y0],
                low_res_dem[x0, y1],
                low_res_dem[x1, y0],
                low_res_dem[x1, y1]
            ])

            optimized_value = idw_interpolation(x_points, y_points, z_points, x_lr, y_lr, power=2)
            optimized_dem[i, j] = (optimized_value + predicted_dem[i, j]) / 2

    return optimized_dem
