from utils import *
from model import *
from loss import *
from interpolation import *
import torch
import torch.optim as optim
import rasterio
import numpy as np

# 打开DEM文件并读取高分辨率DEM数据
with rasterio.open('C:\\Users\\Lenovo\\Desktop\\tif3.tif') as src:
    high_res_dem = src.read(1)  # 读取第一通道的数据
    cellsize = src.res[0]  # 取像元大小

# 切割DEM数据的一小块区域 (例如：100x100的区域)
# 假设DEM数据较大，你可以根据需要调整起始坐标和块的大小
x_start, y_start = 500, 500  # 起始位置
x_size, y_size = 100, 100  # 块大小
high_res_dem = high_res_dem[y_start:y_start + y_size, x_start:x_start + x_size]

# 数据归一化
high_res_dem, min_value, max_value = normalize_dem(high_res_dem)

# 数据准备
low_res_dem = downsample_dem(high_res_dem, scale=2)
slope, aspect = calculate_slope_aspect(low_res_dem, cellsize)
input_data = np.stack([low_res_dem, slope, aspect], axis=0)  # 三通道输入

# 将数据转换为Tensor
input_data_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # (1, 3, H, W)
high_res_dem_tensor = torch.tensor(high_res_dem, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

# 模型
model = EDSR(scale=2, num_res_blocks=32, num_filters=256)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    output = model(input_data_tensor)

    # 计算损失
    loss = combined_loss(high_res_dem_tensor, output,cellsize)

    # 反向传播与优化
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# 预测并优化
predicted_dem = model(input_data_tensor).detach().numpy().squeeze()
# 反归一化
predicted_dem = denormalize_dem(predicted_dem, min_value, max_value)

optimized_dem = local_interpolation_optimization_with_idw(predicted_dem, low_res_dem)

print("优化后的高分辨率DEM:", optimized_dem)
