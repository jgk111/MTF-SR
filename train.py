"""
第二版
"""
from utils import *
from model import *
from loss import *
from interpolation import *
import torch
import torch.optim as optim
import rasterio  
  
# 打开DEM文件  
with rasterio.open('C:\\Users\\Administrator\\Desktop\\遥感资料\\tif3.tif') as src:  
    # 直接从元数据中获取像元大小  
    cellsize = src.res[0]  # 通常res是一个包含(x_cellsize, y_cellsize)的元组  
# 数据准备
high_res_dem = ...  # 高分辨率DEM
low_res_dem = downsample_dem(high_res_dem, scale=2)
slope,aspect = calculate_slope_aspect(low_res_dem,cellsize)
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
optimized_dem = local_interpolation_optimization_with_idw(predicted_dem, low_res_dem)

print("优化后的高分辨率DEM:", optimized_dem)

