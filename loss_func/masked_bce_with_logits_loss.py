import torch
import torch.nn as nn


def masked_bce_with_logits_loss(logits, targets, mask):
    """
    计算带掩码的BCE损失

    参数:
        logits: 模型预测的logits，形状为 [batch_size]
        targets: 真实标签，形状为 [batch_size]
        mask: 掩码矩阵，形状为 [batch_size]
    """
    # 使用BCEWithLogitsLoss，但只计算掩码位置的损失
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    loss = loss_fn(logits, targets)

    # 应用掩码
    masked_loss = loss * mask

    # 计算平均损失，只考虑掩码为1的位置
    return masked_loss.sum() / mask.sum().clamp(min=1e-8)
