import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


# 梯度损失
def gradient_loss(y_true, y_pred, cellsize, alpha=0.5, beta=0.5):
    # 确保输入张量不在计算图中，并转为 NumPy 数组用于计算坡度和坡向
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()

    # 计算坡度和坡向
    true_slope, true_aspect = calculate_slope_aspect(y_true_np, cellsize)
    pred_slope, pred_aspect = calculate_slope_aspect(y_pred_np, cellsize)

    # 转换回 PyTorch 张量
    true_slope = torch.tensor(true_slope, dtype=torch.float32, device=y_true.device)
    true_aspect = torch.tensor(true_aspect, dtype=torch.float32, device=y_true.device)
    pred_slope = torch.tensor(pred_slope, dtype=torch.float32, device=y_true.device)
    pred_aspect = torch.tensor(pred_aspect, dtype=torch.float32, device=y_true.device)

    # 计算损失
    slope_diff = nn.L1Loss()(pred_slope, true_slope)
    aspect_diff = nn.L1Loss()(pred_aspect, true_aspect)

    return alpha * slope_diff + beta * aspect_diff


# 综合损失
def combined_loss(y_true, y_pred, cellsize):
    mse_loss = nn.L1Loss()(y_true, y_pred)  # MSE损失
    grad_loss = gradient_loss(y_true, y_pred, cellsize)  # 梯度损失
    return mse_loss + grad_loss
