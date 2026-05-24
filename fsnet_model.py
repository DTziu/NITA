# fsnet_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FSNet(nn.Module):
    """FS-Net: Flow Sequence Network for Encrypted Traffic Classification"""

    def __init__(self, num_classes, seq_len=200, vocab_size=2049, embed_dim=256, hidden_dim=256, num_layers=2):
        super(FSNet, self).__init__()
        self.seq_len = seq_len

        # 包长嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 多层 CNN 提取局部流量特征
        self.conv1 = nn.Conv1d(embed_dim, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.3)

        # LSTM 层
        self.lstm = nn.LSTM(256, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len)

        # 嵌入
        x = self.embedding(x)  # (batch, seq_len, embed_dim)

        # 转置为 CNN 输入格式
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)

        # CNN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(F.relu(self.conv3(x)))

        # 转置为 LSTM 输入格式
        x = x.transpose(1, 2)  # (batch, seq_len//2, 256)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len//2, hidden_dim*2)

        # 取最后一个时间步
        x = lstm_out[:, -1, :]  # (batch, hidden_dim*2)

        # 分类
        out = self.classifier(x)
        return out


def create_model(num_classes, seq_len=200, vocab_size=2049):
    """创建 FS-Net 模型"""
    model = FSNet(num_classes, seq_len, vocab_size=vocab_size)
    return model
