import os
import pickle
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split


class StockPredictionDataset(Dataset):
    """
    用于股票数据的自定义 PyTorch 数据集。
    """

    def __init__(self, feature_data_path, label_data_path):
        self.data_path = feature_data_path
        self.grouped_data_path = os.path.join(os.path.dirname(self.data_path), 'grouped_data.csv')
        self.label_path = label_data_path
        self.features = []
        self.labels = []

    def load_data(self):
        def parse_string(string):
            """手动解析字符串形式的列表"""
            # 去除方括号并分割
            elements = string.strip('[]').split(',')
            # 转换为浮点数
            return [float(x.strip()) for x in elements if x.strip()]

        # features
        if os.path.exists(self.grouped_data_path):
            grouped_data = pd.read_csv(self.grouped_data_path)
        else:
            feature_df = pd.read_csv(self.data_path)
            # 将交易时间列转换为日期时间类型
            feature_df['交易时间'] = pd.to_datetime(feature_df['交易时间'])

            # 提取月份信息并转换为字符串类型
            feature_df['月份'] = feature_df['交易时间'].dt.to_period('M').astype(str)
            grouped_data = feature_df.groupby(['股票代码', '月份']).agg(list).reset_index().sort_values(
                ['股票代码', '月份'])
            grouped_data.to_csv(os.path.join(os.path.dirname(self.data_path), 'grouped_data.csv'), index=False)

        # labels
        label_df = pd.read_excel(self.label_path)

        for i in tqdm(range(len(grouped_data))):
            stock_code = grouped_data.iloc[i]['股票代码']
            month = grouped_data.iloc[i]['月份']
            label_value = label_df.loc[(label_df['CODE'] == stock_code) & (label_df['MONTH'] == month)][
                'lottery_label_weighted']
            if label_value.empty:
                continue
            else:
                label = label_value.values[0]
            features = grouped_data.iloc[i].drop(['股票代码', '月份', '公司简称', '交易时间']).values.tolist()
            # 归一化
            features_np = np.array([parse_string(feature) for feature in features])
            feature_max = np.max(features_np, axis=0)
            feature_min = np.min(features_np, axis=0)
            features_norm = (features_np - feature_min) / (feature_max - feature_min)
            self.features.append(features_norm)
            self.labels.append(label)

    def __len__(self):
        """
        返回数据集中的样本总数。
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        根据索引获取一个样本（特征和标签）。
        """
        return self.features[idx], self.labels[idx]


def save_dataset(dataset, save_path):
    """
    保存 PyTorch 数据集。
    """
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f'Dataset saved to {save_path}.')


def load_dataset(save_path):
    """
    加载 PyTorch 数据集。
    """
    with open(save_path, 'rb') as f:
        dataset = pickle.load(f)
    print(f'Dataset loaded from {save_path}.')
    return dataset


def padding_seq(batch, max_length=31):
    # 分离数据和标签
    sequences, labels = zip(*batch)

    # 创建掩码和填充后的数据
    padded_sequences = torch.zeros(len(sequences), 7, max_length)
    mask = torch.zeros(len(sequences), max_length)

    for i, seq in enumerate(sequences):
        # 截断到最大长度
        seq = seq[:, :max_length] if seq.shape[1] > max_length else seq

        # 填充数据
        padded_sequences[i, :, :seq.shape[1]] = torch.FloatTensor(seq)

        # 设置掩码
        mask[i, :seq.shape[1]] = 1.0

    # 标签转换
    labels = torch.FloatTensor(labels)

    return padded_sequences, labels, mask


def split_dataset(dataset, batch_size=16, max_length=31,
                  train_ratio=0.8,
                  val_ratio=0.1,
                  test_ratio=0.1):
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    # 创建数据加载器，添加collate_fn参数并设置固定最大长度
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: padding_seq(x, max_length=max_length)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=lambda x: padding_seq(x, max_length=max_length)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=lambda x: padding_seq(x, max_length=max_length)
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # dataset = StockPredictionDataset('./combined_data.csv', './lottery_label.xlsx')
    # dataset.load_data()
    # print(f'Dataset size: {len(dataset)}.')
    # print(f'First sample: {dataset[0]}.')

    # 保存数据集
    # save_dataset(dataset, './stock_prediction_dataset.pkl')

    # 加载数据集
    loaded_dataset = load_dataset('./stock_prediction_dataset.pkl')
    print(f'Loaded dataset size: {len(loaded_dataset)}.')
    print(f'First loaded sample: {loaded_dataset[0]}.')
    train_dataset, val_dataset, test_dataset = split_dataset(loaded_dataset)
    magic_number = 42
