import time
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Huấn luyện mô hình trong 1 epoch.
    Trả về: (loss trung bình, accuracy, thời gian thực thi epoch tính bằng giây)
    """
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    start_time = time.time()

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)

    elapsed_time = time.time() - start_time
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, elapsed_time


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Đánh giá mô hình trên tập validation hoặc test.
    Tính toán 4 độ đo yêu cầu:
      - Accuracy
      - Macro Precision
      - Macro Recall
      - Macro F1
    Trả về: (loss, acc, precision, recall, f1, y_true, y_pred)
    """
    model.eval()
    total, correct, running_loss = 0, 0, 0.0
    y_true, y_pred = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        running_loss += criterion(logits, yb).item() * yb.size(0)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
        y_true.extend(yb.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())

    eval_loss = running_loss / total
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return eval_loss, acc, prec, rec, macro_f1, y_true, y_pred
