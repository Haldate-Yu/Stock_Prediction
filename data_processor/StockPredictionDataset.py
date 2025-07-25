import os
import pickle
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split

FULL_MONTH_DAYS = 31


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
        self.data_masks = []
        self.stock_codes = []
        self.month_list = []

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
            # process with mask
            single_month_df, single_month_mask = self.process_single_month_file(grouped_data.iloc[i])

            stock_code = grouped_data.iloc[i]['股票代码']
            month = grouped_data.iloc[i]['月份']
            label_value = label_df.loc[(label_df['CODE'] == stock_code) & (label_df['MONTH'] == month)][
                'lottery_label_weighted']
            if label_value.empty:
                # 没有标签的情况下，直接跳过整个月份的数据
                continue
            else:
                label = label_value.values[0]
            # features = grouped_data.iloc[i].drop(['股票代码', '月份', '公司简称', '交易时间']).values.tolist()
            # process with mask
            features = single_month_df.drop(columns='交易时间').values.tolist()
            # 归一化
            # features_np = np.array([parse_string(feature) for feature in features])
            # process with mask
            features_np = np.array([feature for feature in features])
            feature_max = np.max(features_np, axis=0)
            feature_min = np.min(features_np, axis=0)
            features_norm = (features_np - feature_min) / (feature_max - feature_min + 1e-10)
            self.features.append(features_norm)
            # process with mask
            self.data_masks.append(single_month_mask)
            self.labels.append(label)
            self.stock_codes.append(stock_code)
            self.month_list.append(month)

    def process_single_month_file(self, grouped_data_i):
        time_list = grouped_data_i['交易时间'].replace("['", "").replace("']", "").replace("Timestamp(", "").replace(
            ")", "").split(
            "', '")
        date_list = [pd.to_datetime(i.replace("['", "").replace("']", "")) for i in time_list]

        open_price_list = eval(grouped_data_i['日开盘价'])
        high_price_list = eval(grouped_data_i['日最高价'])
        low_price_list = eval(grouped_data_i['日最低价'])
        close_price_list = eval(grouped_data_i['日收盘价'])
        prev_close_price_list = eval(grouped_data_i['前日收盘价'])
        change_rate_list = eval(grouped_data_i['日涨跌幅(%)'])
        turnover_rate_list = eval(grouped_data_i['日换手率(%)'])
        data_new = {
            '交易时间': date_list,
            '日开盘价': open_price_list,
            '日最高价': high_price_list,
            '日最低价': low_price_list,
            '日收盘价': close_price_list,
            '前日收盘价': prev_close_price_list,
            '日涨跌幅(%)': change_rate_list,
            '日换手率(%)': turnover_rate_list
        }
        df_new = pd.DataFrame(data_new)

        month = grouped_data_i['月份']
        # 提取完整月份
        start_date = pd.to_datetime(month)
        end_date = start_date + pd.offsets.MonthEnd(0)
        full_time_axis = pd.date_range(start=start_date, end=end_date)
        full_time_list = [i for i in full_time_axis]
        # 将其补充至31条数据，不足的在后边补齐
        while len(full_time_list) < FULL_MONTH_DAYS:
            full_time_list.append(full_time_axis[len(full_time_axis) - 1])

        single_frame_mask = []
        result = []
        for date in full_time_list:
            if date in df_new['交易时间'].values:
                # 匹配上则直接取值
                row = df_new[df_new['交易时间'] == date].iloc[0].to_dict()
                row['交易时间'] = date
                single_frame_mask.append(1)
            elif date < df_new['交易时间'].min():
                # 时间在最小时间之前沿用最小时间记录
                row = df_new[df_new['交易时间'] == df_new['交易时间'].min()].iloc[0].to_dict()
                row['交易时间'] = date
                single_frame_mask.append(0)
            elif date > df_new['交易时间'].max():
                # 时间在最大时间之后沿用最大时间记录
                row = df_new[df_new['交易时间'] == df_new['交易时间'].max()].iloc[0].to_dict()
                row['交易时间'] = date
                single_frame_mask.append(0)
            else:
                # 时间在表格时间中间按比例求指标值
                prev_date = df_new[df_new['交易时间'] < date]['交易时间'].max()
                next_date = df_new[df_new['交易时间'] > date]['交易时间'].min()

                prev_row = df_new[df_new['交易时间'] == prev_date].iloc[0]
                next_row = df_new[df_new['交易时间'] == next_date].iloc[0]

                ratio = (date - prev_date) / (next_date - prev_date)

                row = {}
                for col in df_new.columns[1:]:
                    row[col] = prev_row[col] + ratio * (next_row[col] - prev_row[col])
                row['交易时间'] = date
                single_frame_mask.append(0)

            result.append(row)
        return pd.DataFrame(result), single_frame_mask

    def __len__(self):
        """
        返回数据集中的样本总数。
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        根据索引获取一个样本（特征和标签）。
        """
        if hasattr(self, 'data_masks') and self.data_masks[idx] is not None:
            if hasattr(self, 'stock_codes') and hasattr(self, 'month_list'):
                return self.features[idx].astype(np.float32), self.labels[idx].astype(np.float32), np.array(
                    self.data_masks[idx]), self.stock_codes[idx], self.month_list[idx]
            else:
                return self.features[idx].astype(np.float32), self.labels[idx].astype(np.float32), np.array(
                    self.data_masks[idx])

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


def downsample_dataset(dataset, use_padding=True):
    class_0_indices = [i for i, label in enumerate(dataset.labels) if label == 0]
    class_1_indices = [i for i, label in enumerate(dataset.labels) if label == 1]

    num_class_1 = len(class_1_indices)
    selected_class_0_indices = np.random.choice(class_0_indices, num_class_1, replace=False)

    selected_indices = np.concatenate([selected_class_0_indices, class_1_indices])
    np.random.shuffle(selected_indices)

    new_features = [dataset.features[i] for i in selected_indices]
    new_labels = [dataset.labels[i] for i in selected_indices]
    if not use_padding:
        new_data_masks = [dataset.data_masks[i] for i in selected_indices]

    dataset.features = new_features
    dataset.labels = new_labels
    if not use_padding:
        dataset.data_masks = new_data_masks

    return dataset


def upsample_dataset(dataset, use_padding=True):
    class_0_indices = [i for i, label in enumerate(dataset.labels) if label == 0]
    class_1_indices = [i for i, label in enumerate(dataset.labels) if label == 1]

    num_class_0 = len(class_0_indices)
    num_class_1 = len(class_1_indices)

    if num_class_0 > num_class_1:
        minority_indices = class_1_indices
        majority_count = num_class_0
    else:
        minority_indices = class_0_indices
        majority_count = num_class_1

    # 复制少数类样本直到数量和多数类相同
    new_minority_indices = []
    while len(new_minority_indices) < majority_count:
        new_minority_indices.extend(minority_indices)

    new_minority_indices = new_minority_indices[:majority_count]

    all_indices = (class_0_indices if minority_indices == class_1_indices else class_1_indices) + new_minority_indices
    np.random.shuffle(all_indices)

    new_features = [dataset.features[i] for i in all_indices]
    new_labels = [dataset.labels[i] for i in all_indices]
    new_stock_codes = [dataset.stock_codes[i] for i in all_indices]
    new_month_list = [dataset.month_list[i] for i in all_indices]

    if not use_padding:
        new_data_masks = [dataset.data_masks[i] for i in all_indices]

    dataset.features = new_features
    dataset.labels = new_labels
    dataset.stock_codes = new_stock_codes
    dataset.month_list = new_month_list
    if not use_padding:
        dataset.data_masks = new_data_masks

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
                  test_ratio=0.1,
                  use_padding=True,
                  sample_type='None',
                  k_fold=10,
                  fold_idx=0):
    if sample_type == 'down':
        dataset = downsample_dataset(dataset, use_padding=use_padding)
    elif sample_type == 'up':
        dataset = upsample_dataset(dataset, use_padding=use_padding)

    # train_size = int(train_ratio * len(dataset))
    # val_size = int(val_ratio * len(dataset))
    # test_size = len(dataset) - train_size - val_size
    # train_dataset, val_dataset, test_dataset = random_split(
    #     dataset, [train_size, val_size, test_size]
    # )

    # 修改为K折交叉验证
    fold_size = len(dataset) // k_fold
    val_start = (fold_idx % k_fold) * fold_size
    val_end = val_start + fold_size
    val_dataset = [dataset[i] for i in range(val_start, val_end)]

    test_start = (fold_idx + 1) % k_fold * fold_size
    test_end = test_start + fold_size
    test_dataset = [dataset[i] for i in range(test_start, test_end)]

    val_indices = set(range(val_start, val_end))
    test_indices = set(range(test_start, test_end))

    train_dataset = [dataset[i] for i in range(len(dataset))
                     if i not in val_indices
                     and i not in test_indices]

    # 创建数据加载器，添加collate_fn参数并设置固定最大长度
    if use_padding:
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
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size
        )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # dataset = StockPredictionDataset('./combined_data.csv', './lottery_label.xlsx')
    # dataset.load_data()
    # print(f'Dataset size: {len(dataset)}.')
    # print(f'First sample: {dataset[0]}.')

    # 保存数据集
    # save_dataset(dataset, './stock_prediction_dataset_full.pkl')

    # 加载数据集
    loaded_dataset = load_dataset('./stock_prediction_dataset_full.pkl')
    print(f'Loaded dataset size: {len(loaded_dataset)}.')
    print(f'First loaded sample: {loaded_dataset[0]}.')
    train_dataset, val_dataset, test_dataset = split_dataset(loaded_dataset, use_padding=False)
    magic_number = 42
