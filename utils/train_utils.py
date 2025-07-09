import torch


class EarlyStopping:
    """早停机制，避免过拟合"""

    def __init__(self, patience=10, verbose=False, delta=1e-10, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''保存模型当验证损失减小时'''
        if self.verbose:
            print(f'验证损失减小 ({self.val_loss_min:.6f} --> {val_loss:.6f})。保存模型...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
