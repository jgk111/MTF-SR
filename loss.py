import torch
import torch.nn as nn
import torch.nn.functional as F

# SSIM 损失
def ssim_loss(y_true, y_pred):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_true = F.avg_pool2d(y_true, 3, 1, 1)
    mu_pred = F.avg_pool2d(y_pred, 3, 1, 1)
    sigma_true = F.avg_pool2d(y_true ** 2, 3, 1, 1) - mu_true ** 2
    sigma_pred = F.avg_pool2d(y_pred ** 2, 3, 1, 1) - mu_pred ** 2
    sigma_true_pred = F.avg_pool2d(y_true * y_pred, 3, 1, 1) - mu_true * mu_pred
    ssim_map = ((2 * mu_true * mu_pred + C1) * (2 * sigma_true_pred + C2)) / \
               ((mu_true ** 2 + mu_pred ** 2 + C1) * (sigma_true + sigma_pred + C2))
    return torch.clamp((1 - ssim_map) / 2, 0, 1).mean()

# 梯度损失
def gradient_loss(y_true, y_pred):
    dy_true, dx_true = torch.gradient(y_true)
    dy_pred, dx_pred = torch.gradient(y_pred)
    grad_diff_y = torch.mean(torch.abs(dy_true - dy_pred))
    grad_diff_x = torch.mean(torch.abs(dx_true - dx_pred))
    return grad_diff_x + grad_diff_y

# 综合损失
def combined_loss(y_true, y_pred, alpha=0.5, beta=0.5):
    mse_loss = nn.MSELoss()(y_true, y_pred)  # MSE损失
    ssim = ssim_loss(y_true, y_pred)  # SSIM损失
    grad_loss = gradient_loss(y_true, y_pred)  # 梯度损失
    return mse_loss + alpha * ssim + beta * grad_loss
