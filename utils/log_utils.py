import os
import csv
import random
import torch
import numpy as np


def num_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def num_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def seed_everything(seed):
    r"""Sets the seed for generating random numbers in PyTorch, numpy and
        Python.

        Args:
            seed (int): The desired seed.
        """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 固定CuDNN算法
    torch.backends.cudnn.benchmark = False  # 禁用自动优化


def save_results(args,
                 metrics):
    r"""Saves the results of the experiment in a CSV file.

        Args:
            args (argparse.Namespace): The command-line arguments.
            metrics (dict): The metrics to be saved.
        """
    if not os.path.exists('./results'):
        print("=" * 20)
        print("Creating Results File !!!")

        os.makedirs('./results')

    file_name = f'./results/result.csv'

    header_list = ['Model', 'Epoch', 'Learning_Rate', 'Weight_Decay',
                   'Lstm_Seq_Length', 'Padding_type', 'Sample_Type', 'Fold_Id',
                   'Acc', 'Auc']
    with open(file_name, 'a+') as file:
        file.seek(0)
        header = file.read(5)
        if header != 'Model':
            dw = csv.DictWriter(file, delimiter=',', fieldnames=header_list)
            dw.writeheader()
        line = ("{}, {}, {}, {}, "
                "{}, {}, {}, {}, "
                "{}, {}\n").format(
            "LSTM", args.epochs, args.lr, args.weight_decay,
            31, args.padding_type, args.sample_type, args.fold_idx,
            metrics['accuracy'], metrics['auc']
        )
        file.write(line)

    # Save the best result
    if not os.path.exists('./pred_results'):
        print("=" * 20)
        print("Creating Results File !!!")

        os.makedirs('./pred_results')
    file_name = f'./pred_results/LSTM_lr{args.lr}_wd{args.weight_decay}_{args.sample_type}_sample_{args.padding_type}_fold{args.fold_idx}_result.npz'
    np.savez(
        file_name,
        accuracy=metrics['accuracy'],
        auc=metrics['auc'],
        predictions=metrics['predictions'],
        probabilities=metrics['probabilities'],
        true_labels=metrics['true_labels'],
        stock_codes=metrics['stock_codes'],
        months=metrics['months']
    )
    print("=" * 20)
    print("Results saved successfully!!!")
