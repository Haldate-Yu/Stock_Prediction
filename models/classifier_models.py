import torch
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64,
                 num_layers=1, dropout=0.2,
                 threshold=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, 1)
        self.threshold = threshold

    def forward(self, x, mask=None):
        x = x.permute(0, 2, 1)  # [B, T, D]

        out, _ = self.lstm(x)  # out: [B, T, H]
        # out = out.permute(0, 2, 1)  # [B, H, T]

        # 如果提供了掩码，应用掩码
        if mask is not None:
            # 扩展掩码维度以匹配LSTM输出
            mask_expanded = mask.unsqueeze(-1).expand_as(out)

            # 将填充位置的值设为0
            out = out * mask_expanded

            # 计算每个样本的有效长度
            seq_lengths = mask.sum(dim=1).long()

            # 使用掩码进行加权平均池化
            # 对有效时间步取平均，而不是使用AdaptiveAvgPool1d
            batch_size = x.size(0)
            indices = (seq_lengths - 1).unsqueeze(1).unsqueeze(2).expand(batch_size, 1, out.size(2))
            pooled = out.gather(1, indices).squeeze(1)  # [B, H]
        else:
            # 如果没有提供掩码，使用原始的自适应平均池化
            out = out.permute(0, 2, 1)  # [B, H, T]
            pooled = self.pool(out).squeeze(-1)  # [B, H]

        logits = self.fc(pooled).squeeze(-1)  # [B]
        probs = torch.sigmoid(logits)
        preds = (probs >= self.threshold).float()

        return {
            'logits': logits,
            'probs': probs,
            'preds': preds
        }
