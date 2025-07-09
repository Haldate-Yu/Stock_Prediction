import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
from utils.train_utils import EarlyStopping
from loss_func.masked_bce_with_logits_loss import masked_bce_with_logits_loss


def train_model(model, train_loader, val_loader,
                epochs=300, patient=10,
                lr=5e-4, wd=1e-6,
                save_path = 'checkpoint.pt'):
    # 记录训练开始时间和模型参数
    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 使用BCEWithLogitsLoss，它结合了Sigmoid和BCE Loss，数值更稳定
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_auc = 0.0
    best_model = None

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )
    early_stopping = EarlyStopping(
        patience=patient, verbose=False, path=save_path
    )

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        for data, target, mask in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"):
            data, target, mask = data.to(device), target.to(device), mask.to(device)

            optimizer.zero_grad()
            output = model(data, mask)
            loss = criterion(output['logits'], target)
            # loss = masked_bce_with_logits_loss(output['logits'], target, mask[:, -1])
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(output['preds'].cpu().numpy())
            train_targets.extend(target.cpu().numpy())

        # 计算训练准确率
        train_acc = accuracy_score(train_targets, train_preds)

        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds = []
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for input, target, mask in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]'):
                input, target, mask = input.to(device), target.to(device), mask.to(device)

                output = model(input, mask)
                loss = criterion(output['logits'], target)
                # loss = masked_bce_with_logits_loss(output['logits'], target, mask[:, -1])

                val_loss += loss.item()
                val_preds.extend(output['preds'].cpu().numpy())
                val_probs.extend(output['probs'].cpu().numpy())
                val_labels.extend(target.cpu().numpy())

        # 计算验证准确率和AUC
        val_acc = accuracy_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_probs)
        val_loss_avg = val_loss / len(val_loader)

        # 更新学习率
        scheduler.step(val_loss_avg)

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_val_acc = val_acc
            best_val_auc = val_auc
            best_model = model.state_dict().copy()
            print(f'Saving best model with val loss: {best_val_loss:.4f}')

        early_stopping(val_loss_avg, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # 计算训练时长
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f}s")
    print(f"Best val loss: {best_val_loss:.4f}, acc: {best_val_acc:.4f}, auc: {best_val_auc:.4f}")
    # 加载最佳模型
    model.load_state_dict(best_model)
    return model


def evaluate_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    test_preds = []
    test_probs = []
    test_labels = []

    with torch.no_grad():
        for input, target, mask in tqdm(test_loader, desc='Evaluating'):
            input, target, mask = input.to(device), target.to(device), mask.to(device)

            output = model(input, mask)
            test_preds.extend(output['preds'].cpu().numpy())
            test_probs.extend(output['probs'].cpu().numpy())
            test_labels.extend(target.cpu().numpy())

    # 计算测试准确率和AUC
    test_acc = accuracy_score(test_labels, test_preds)
    test_auc = roc_auc_score(test_labels, test_probs)

    print(f'Test Accuracy: {test_acc:.2f}%')
    print(f'Test AUC: {test_auc:.4f}')

    return {
        'accuracy': test_acc,
        'auc': test_auc,
        'predictions': test_preds,
        'probabilities': test_probs,
        'true_labels': test_labels
    }
