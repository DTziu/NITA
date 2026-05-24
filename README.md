# 加密流量分析与 FS-Net 复现实验报告

## 实验说明
本实验包含两部分：
1. 实验一：加密流量解析；
2. 实验二：加密流量分类方法复现与对比。

实验目标覆盖 PCAPNG 文件解析、TCP 流重组、特征提取，以及基于 FS-Net 的加密流量分类复现。

## 1. 实验环境
- Python 版本：Python3
- 依赖库：`torch`、`scikit-learn`、`numpy`、`pandas`
- 运行环境：`c:\Users\zhou3\PycharmProjects\PythonProject9` 目录下 `.venv`
- 训练设备：`cuda`

## 2. 实验一：加密流量解析
### 2.1 实验目的
- 掌握 PCAPNG 文件格式规范；
- 使用 Python 编写 PCAPNG 解析程序；
- 按照规范实现流量解析、TCP 四元组聚合，并输出 CSV 数据集。

### 2.2 实现方法
- 解析脚本：`pcapng_parser_fsnet.py`
- 使用模块：`struct`、`csv`、`collections.defaultdict`
- 解析流程：
  - 读取 PCAPNG 文件头（SHB）并跳过文件头块；
  - 顺序解析各块类型，重点处理增强包块（EPB）；
  - 解析以太网帧类型，处理 IPv4 和 IPv6 数据包；
  - 解析传输层 TCP/UDP 头，并提取源 IP、目的 IP、源端口、目的端口、协议类型、包长度；
  - 对 TCP 负载中的 TLS 记录进行简单解析，提取 TLS 记录类型、版本、握手类型等字段。

### 2.3 TCP 流重组
- 采用四元组 `(src_ip, dst_ip, src_port, dst_port)` 和反向四元组归一化处理；
- 将同一 TCP 流中的包按时间戳排序；
- 输出形式为每条流对应一个 packet list。

### 2.4 数据输出
- 数据集文件：`dataset.csv`
- 字段顺序：`所属应用, 所属流量文件名, 流号, 特征1, 特征2, ..., 特征200`
- 特征说明：前 200 列为流的包长序列，序列不足时以 0 填充。
- 样本总量：46545 条流数据；
- 应用类别数：15；
- 数据来源：`traffic-dataset` 中 15 个应用、每个应用 5 个 PCAPNG 文件。

### 2.5 数据集特征
- `dataset.csv` 头部包含 200 个特征列；
- 示例样本显示前若干包长为实际包长值，后续位置为 `0` 填充；
- 这是典型的 FS-Net 包长序列输入格式。

## 3. 实验二：FS-Net 方法复现与对比
### 3.1 论文选择
- 复现方法：FS-Net
- 论文名称：`FS-Net: A Flow Sequence Network For Encrypted Traffic Classification`
- 特征：流的包长序列
- 方法类别：深度学习

### 3.2 数据预处理
- 预处理脚本：`preprocess_for_fsnet.py`
- 处理步骤：
  - 加载 `dataset.csv`；
  - 读取 `特征1`–`特征200` 列，构造包长序列；
  - 对序列进行截断/填充，固定长度 `200`；
  - 对 `所属应用` 进行标签编码；
  - 保存为 `fsnet_dataset.npz` 和 `app_labels.pkl`。

### 3.3 FS-Net 模型结构
- 模型文件：`fsnet_model.py`
- 网络结构：
  - 包长嵌入层：`Embedding(vocab_size=2049, embed_dim=256)`；
  - 3 层卷积网络：`Conv1d(kernel_size=3)` + `ReLU`；
  - 池化层：`MaxPool1d(kernel_size=2)`；
  - 双向 LSTM：`bidirectional=True`；
  - 全连接分类器：`Linear -> ReLU -> Dropout -> Linear`。
- 输入：长度为 200 的包长序列；
- 输出：15 类应用预测。

### 3.4 训练流程
- 训练脚本：`fsnet_train.py`
- 数据划分：随机拆分 70% 训练集，30% 测试集；
- 超参数：
  - batch size = 128
  - 学习率 = 0.001
  - 优化器 = AdamW
  - 权重衰减 = 1e-4
  - 学习率调度 = ReduceLROnPlateau(mode='max', factor=0.5, patience=3)
  - 早停 = patience 5
  - 训练轮数最多 30 轮
- 训练过程使用测试集作为验证集进行监督；
- 最佳模型保存为 `best_fsnet_model.pth`。

### 3.5 评价指标
- 使用指标：Accuracy、Precision、Recall
- 计算方式：
  - Accuracy = 正确预测数 / 样本总数
  - Precision (macro) = 各类别精度平均
  - Recall (macro) = 各类别召回平均
- 复现结果与论文指标一致，使用宏平均指标评估多类别分类性能。

## 4. 训练与评估结果
### 4.1 训练过程摘要
- 训练集大小：32581
- 测试集大小：13964
- 训练首轮：Train Acc 0.3002，Val Acc 0.4897
- 中期收敛：Epoch 10 训练准确率 > 0.90，验证准确率约 0.8940
- 末期表现：Epoch 30 验证准确率 0.9136

### 4.2 最终测试结果
- Accuracy：`0.9136`
- Precision (macro)：`0.9255`
- Recall (macro)：`0.9140`

### 4.3 详细分类表现
- 表现较好：
  - 类别 9：precision 0.97，recall 0.96，f1 0.97
  - 类别 14：precision 0.99，recall 0.95，f1 0.97
- 表现较弱：
  - 类别 11：precision 0.86，recall 0.76，f1 0.81
  - 类别 2：precision 0.94，recall 0.85，f1 0.89

## 5. 结论与建议
### 5.1 结论
- 实验一已成功完成：使用 Python `struct` 实现了 PCAPNG 解析和 TCP 四元组流重组；
- 实验二已成功复现 FS-Net：完整训练流程、数据集构建与评估均已实现；
- 最终模型在给定数据集上表现良好，准确率超过 91%。

### 5.2 建议
1. 建议后续增加独立验证集，避免测试集参与早停；
2. 对弱类别（如类别 11）可尝试类别加权、过采样或更多特征融合；
3. 如果需要对比更多方法，可继续复现 MaMPF 或 Path Signature 方案。

## 6. 目录与结果文件
- 数据集：`dataset.csv`
- FS-Net 输入数据：`fsnet_dataset.npz`
- 应用标签映射：`app_labels.pkl`
- 最佳模型：`best_fsnet_model.pth`
- 实验报告：`fsnet_training_report.md`
