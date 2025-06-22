import torch
from torch.utils.data import Dataset, DataLoader


class StockPredictionDataset(Dataset):
    """
    用于股票数据的自定义 PyTorch 数据集。
    """

    def __init__(self, dataframe):
        """
        Args:
            dataframe (pandas.DataFrame): 包含特征和标签的数据。
                                          假设最后一列是标签，其余是特征。
        """
        if dataframe.empty:
            self.features = torch.empty(0)
            self.labels = torch.empty(0)
            return

        # 将特征和标签分离
        # .values 会将 dataframe 转换为 numpy array
        features_np = dataframe.iloc[:, :-1].values
        labels_np = dataframe.iloc[:, -1].values

        # 将 numpy array 转换为 PyTorch tensors
        self.features = torch.tensor(features_np, dtype=torch.float32)
        self.labels = torch.tensor(labels_np, dtype=torch.float32)

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
