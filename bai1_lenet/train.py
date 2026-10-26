import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn

from data import build_train_val_loaders
from engine import evaluate, train_one_epoch
from model import LeNet5, layer_table, n_params


def get_args():
    p = argparse.ArgumentParser(description="Bài 1: Huấn luyện LeNet-5 trên VinaFood21")
    p.add_argument("--data_root", type=str, required=True,
                   help="Đường dẫn đến thư mục VinaFood21 (chứa train/ và test/)")
    p.add_argument("--img_size", type=int, default=64,
                   help="Kích thước ảnh vuông resize (mặc định: 64)")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Batch size (mặc định: 32)")
    p.add_argument("--epochs", type=int, default=20,
                   help="Số epoch huấn luyện tối đa (mặc định: 20)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate ban đầu cho Adam (mặc định: 0.001)")
    p.add_argument("--patience_lr", type=int, default=2,
                   help="Số epoch chờ trước khi giảm LR qua ReduceLROnPlateau (mặc định: 2)")
    p.add_argument("--patience_es", type=int, default=5,
                   help="Số epoch chờ trước khi dừng sớm Early Stopping (mặc định: 5)")
    p.add_argument("--lr_factor", type=float, default=0.5,
                   help="Hệ số giảm learning rate (mặc định: 0.5)")
    p.add_argument("--min_lr", type=float, default=1e-6,
                   help="Learning rate tối thiểu (mặc định: 1e-6)")
    p.add_argument("--pool", type=str, default="avg", choices=["avg", "max"],
                   help="Loại pooling: 'avg' (AvgPool2d) hoặc 'max' (MaxPool2d)")
    p.add_argument("--num_workers", type=int, default=2,
                   help="Số luồng nạp dữ liệu (mặc định: 2)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed ngẫu nhiên (mặc định: 42)")
    p.add_argument("--out_dir", type=str, default="outputs",
                   help="Thư mục lưu mô hình và kết quả")
    return p.parse_args()


def save_metrics_plot(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Đồ thị Loss
    ax1.plot(epochs, [h["train_loss"] for h in history], label="Train Loss", color="blue")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="Val Loss", color="orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Đường cong Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Đồ thị Macro-F1 & Accuracy
    ax2.plot(epochs, [h["val_f1"] for h in history], label="Val Macro-F1", color="green")
    ax2.plot(epochs, [h["val_acc"] for h in history], label="Val Accuracy", color="red")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Độ chính xác và Macro-F1 trên Validation")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print(f" BÀI 1: HUẤN LUYỆN LENET-5 TRÊN VINAFOOD21 (POOLING: {args.pool.upper()})")
    print("==================================================================")
    print(f"Device      : {device}")
    print(f"Image Size  : {args.img_size}x{args.img_size}")
    print(f"Batch Size  : {args.batch_size}")
    print(f"Epochs      : {args.epochs}")
    print(f"Optimizer   : Adam (lr = {args.lr})")
    print(f"Random Seed : {args.seed}")

    train_dl, val_dl, classes = build_train_val_loaders(
        args.data_root, img_size=args.img_size, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed)
    print(f"Số lớp: {len(classes)} | Tập train: {len(train_dl.dataset)} ảnh | Tập val: {len(val_dl.dataset)} ảnh")

    # Khởi tạo mô hình
    model = LeNet5(in_ch=3, img=args.img_size, num_classes=len(classes), pool=args.pool).to(device)
    total_params = n_params(model)
    print(f"Tổng số tham số mô hình: {total_params:,}")

    # In bảng chi tiết từng layer (B.3 Bước 2)
    rows, _ = layer_table(model, img=args.img_size)
    print("\n--- BẢNG THAM SỐ VÀ OUTPUT SHAPE TỪNG LAYER ---")
    print(f"{'Layer':<10}{'Output Shape':<18}{'Tham số':>12}")
    print("-" * 40)
    for name, shape, p in rows:
        print(f"{name:<10}{shape:<18}{p:>12,}")
    print("-" * 40)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.patience_lr,
        min_lr=args.min_lr, verbose=True
    )

    history = []
    best_f1 = -1.0
    es_counter = 0
    total_train_time = 0.0

    print("\n--- BẮT ĐẦU HUẤN LUYỆN (CÓ REDUCE LR & EARLY STOPPING) ---")
    for ep in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc, ep_time = train_one_epoch(model, train_dl, criterion, optimizer, device)
        va_loss, va_acc, va_prec, va_rec, va_f1, _, _ = evaluate(model, val_dl, criterion, device)
        total_train_time += ep_time

        # Bước cập nhật Learning Rate Scheduler dựa trên val_loss
        scheduler.step(va_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        lr_changed = (new_lr != current_lr)

        history.append({
            "epoch": ep,
            "lr": current_lr,
            "train_loss": round(tr_loss, 4),
            "train_acc": round(tr_acc, 4),
            "val_loss": round(va_loss, 4),
            "val_acc": round(va_acc, 4),
            "val_precision": round(va_prec, 4),
            "val_recall": round(va_rec, 4),
            "val_f1": round(va_f1, 4),
            "time_sec": round(ep_time, 2)
        })

        mark = ""
        if va_f1 > best_f1:
            best_f1 = va_f1
            es_counter = 0
            torch.save({
                "model": model.state_dict(),
                "classes": classes,
                "img_size": args.img_size,
                "pool": args.pool,
                "epoch": ep,
                "val_f1": best_f1
            }, out_dir / "best.pt")
            mark = " [BEST]"
        else:
            es_counter += 1

        lr_info = f" | LR: {new_lr:.2e}" if lr_changed else ""
        print(f"Epoch {ep:>2}/{args.epochs} [{ep_time:.1f}s] "
              f"Train Loss: {tr_loss:.4f} | Acc: {tr_acc:.4f} || "
              f"Val Loss: {va_loss:.4f} | Acc: {va_acc:.4f} | Prec: {va_prec:.4f} | Rec: {va_rec:.4f} | F1: {va_f1:.4f}{mark}{lr_info}")

        # Kiểm tra Early Stopping
        if es_counter >= args.patience_es:
            print(f"\n[EARLY STOPPING] Kích hoạt tại Epoch {ep} vì Val Macro-F1 không cải thiện trong {args.patience_es} epochs liên tiếp.")
            break

    avg_time_per_epoch = total_train_time / args.epochs
    print("\n==================================================================")
    print(f"HUẤN LUYỆN HOÀN TẤT!")
    print(f"Best Val Macro-F1      : {best_f1:.4f}")
    print(f"Thời gian trung bình   : {avg_time_per_epoch:.2f} giây/epoch")
    print(f"Checkpoint tốt nhất lưu tại: {out_dir / 'best.pt'}")
    print("==================================================================")

    # Lưu lịch sử ra file CSV
    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    save_metrics_plot(history, out_dir / "loss_curve.png")
    print(f"Đã lưu biểu đồ huấn luyện tại: {out_dir / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
