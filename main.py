import os
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.log_utils import seed_everything, save_results
from data_processor.StockPredictionDataset import load_dataset, split_dataset, StockPredictionDataset
from models.classifier_models import LSTMClassifier
from trainer import train_model, evaluate_model
import warnings

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="训练时空图卷积网络模型进行多步预测")

    parser.add_argument("--sample_type", type=str, default='None', help="采样方式")
    parser.add_argument("--padding_type", type=str, default='timely', help="填充方式")
    parser.add_argument("--batch_size", type=int, default=128, help="批大小")
    parser.add_argument("--max_length", type=int, default=31, help="序列最大长度")
    parser.add_argument("--epochs", type=int, default=300, help="训练轮数")
    parser.add_argument("--patience", type=int, default=10, help="早停耐心值")
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    parser.add_argument("--lstm_hidden_size", type=int, default=64, help="LSTM隐藏层大小")
    parser.add_argument("--lstm_num_layers", type=int, default=2, help="LSTM层数")
    parser.add_argument("--lstm_dropout", type=float, default=0.5, help="LSTM的dropout率")

    parser.add_argument("--k_fold", type=int, default=10, help="k折交叉验证")
    parser.add_argument("--fold_idx", type=int, default=0, help="当前折的索引")

    parser.add_argument("--checkpoint_file", type=str, default='checkpoint.pt', help="结果保存路径")

    args = parser.parse_args()

    # 设置随机种子以确保可重复性
    seed_everything(args.seed)

    # 数据处理
    print("正在加载和处理数据...")
    global dataset
    if args.padding_type == 'timely':
        dataset = load_dataset('./data_processor/stock_prediction_dataset_full.pkl')
    elif args.padding_type =='sequence':
        dataset = load_dataset('./data_processor/stock_prediction_dataset.pkl')

    for i in range(args.k_fold):
        print(f"正在划分第{i + 1}折数据...")
        args.checkpoint_file = f'LSTM_lr{args.lr}_wd{args.weight_decay}_{args.sample_type}_sample_{args.padding_type}_fold{i}_model.pt'
        args.fold_idx = i

        train_dataset, val_dataset, test_dataset = split_dataset(dataset,
                                                                 batch_size=args.batch_size,
                                                                 max_length=args.max_length,
                                                                 use_padding=args.padding_type == 'sequence',
                                                                 sample_type=args.sample_type,
                                                                 k_fold=args.k_fold,
                                                                 fold_idx=i)
        print("数据加载完成！")
        # 初始化模型
        print("正在初始化模型...")
        model = LSTMClassifier(input_dim=7,
                               hidden_dim=args.lstm_hidden_size,
                               num_layers=args.lstm_num_layers,
                               dropout=args.lstm_dropout)

        trained_model = train_model(model, train_dataset, val_dataset,
                                    epochs=args.epochs, patient=args.patience,
                                    lr=args.lr, wd=args.weight_decay,
                                    save_path=args.checkpoint_file)

        results = evaluate_model(trained_model, test_dataset)
        save_results(args, results)


if __name__ == '__main__':
    main()
