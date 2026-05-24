# fsnet_train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import pickle
import os
import time


class FlowDataset(Dataset):
    """FS-Net 数据集"""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(data_file='fsnet_dataset.npz', labels_file='app_labels.pkl'):
    """加载预处理后的数据"""
    data = np.load(data_file)
    X = data['X']
    y = data['y']

    with open(labels_file, 'rb') as f:
        labels = pickle.load(f)

    return X, y, labels


def train_model(model, train_loader, val_loader, num_epochs=30, device=None, patience=5):
    """训练模型"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)

    best_acc = 0.0
    no_improve = 0
    model.to(device)

    for epoch in range(num_epochs):
        start_time = time.time()

        # 训练
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)

        # 验证 (使用测试集作为验证)
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        scheduler.step(val_acc)

        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{num_epochs} | Time {epoch_time:.1f}s | '
              f'Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_fsnet_model.pth')
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}, best val acc {best_acc:.4f}')
            break

    return model


def evaluate_model(model, test_loader, device=None):
    """评估模型"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    print("\nTest Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))

    return acc, precision, recall


def main():
    # 加载数据
    print("Loading data...")
    X, y, labels = load_data()
    num_classes = len(labels)
    print(f"Data shape: X={X.shape}, y={y.shape}, num_classes={num_classes}")

    # 创建数据集
    dataset = FlowDataset(X, y)

    # 切分数据集：7:3 (70%训练，30%测试)，保持类别平衡
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    train_dataset = FlowDataset(X_train, y_train)
    test_dataset = FlowDataset(X_test, y_test)
    print(f"Split sizes: train={len(train_dataset)}, test={len(test_dataset)}")

    # 数据加载器
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 创建模型
    from fsnet_model import create_model
    model = create_model(num_classes, seq_len=X.shape[1])

    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 训练模型 (使用测试集作为验证集进行早停)
    print("Training model...")
    trained_model = train_model(model, train_loader, test_loader, num_epochs=30, device=device, patience=5)

    # 加载最佳模型
    trained_model.load_state_dict(torch.load('best_fsnet_model.pth', map_location=device, weights_only=True))

    # 最终评估
    print("Evaluating model...")
    evaluate_model(trained_model, test_loader, device=device)


if __name__ == '__main__':
    main()
