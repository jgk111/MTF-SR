import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

# 梯度损失
def gradient_loss(y_true, y_pred,cellsize,alpha=0.5,beta=0.5):
    true_slope, true_aspect = calculate_slope_aspect(y_true,cellsize)
    pred_slope, pred_aspect = calculate_slope_aspect(y_pred,cellsize)
    slope_diff = nn.L1Loss()(true_slope, pred_slope)
    aspect_diff = nn.L1Loss()(true_aspect, pred_aspect)
    return alpha * slope_diff + beta * aspect_diff

# 综合损失
def combined_loss(y_true, y_pred):
    mse_loss = nn.L1Loss()(y_true, y_pred)  # MSE损失
    grad_loss = gradient_loss(y_true, y_pred)  # 梯度损失
    return mse_loss +  grad_loss
