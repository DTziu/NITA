#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 dataset.csv 转换为 FS-Net 模型输入格式
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import pickle


def load_dataset(csv_file='dataset.csv'):
    """加载数据集"""
    df = pd.read_csv(csv_file, encoding='utf-8')
    return df


def parse_packet_sequence(seq_str):
    """解析包长序列字符串"""
    try:
        return list(map(int, seq_str.split(',')))
    except:
        return []


def pad_sequence(seq, max_len=200, pad_value=0):
    """填充/截断序列到固定长度"""
    if len(seq) > max_len:
        return seq[:max_len]
    else:
        return seq + [pad_value] * (max_len - len(seq))


def create_fsnet_dataset(csv_file='dataset.csv', max_len=200):
    """创建 FS-Net 训练数据集"""
    df = load_dataset(csv_file)

    # 解析包长序列 - 从多个特征列中提取
    seq_cols = [col for col in df.columns if col.startswith('特征')]
    df['packet_lengths'] = df[seq_cols].values.tolist()

    # 获取应用标签
    apps = sorted(df['所属应用'].unique())
    app_to_idx = {app: idx for idx, app in enumerate(apps)}

    # 准备数据
    X = []
    y = []

    for _, row in df.iterrows():
        seq = [abs(int(x)) for x in row['packet_lengths'] if x != 0]  # 取绝对包长并移除填充0
        seq = pad_sequence(seq, max_len=max_len)
        X.append(seq)
        y.append(app_to_idx[row['所属应用']])

    X = np.array(X, dtype=np.int32)
    y = np.array(y, dtype=np.int64)

    # 保存标签映射
    with open('app_labels.pkl', 'wb') as f:
        pickle.dump(app_to_idx, f)

    print(f"数据集形状: X={X.shape}, y={y.shape}")
    print(f"应用类别: {apps}")

    return X, y, app_to_idx


def save_dataset(X, y, output_file='fsnet_dataset.npz'):
    """保存数据集"""
    np.savez(output_file, X=X, y=y)
    print(f"数据集已保存至: {output_file}")


if __name__ == '__main__':
    # 创建数据集
    X, y, labels = create_fsnet_dataset(max_len=200)

    # 保存
    save_dataset(X, y, 'fsnet_dataset.npz')

    print("\n数据预处理完成!")