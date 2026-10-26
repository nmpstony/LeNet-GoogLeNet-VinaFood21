import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn

from data import build_train_val_loaders
from engine import evaluate, train_one_epoch
from model import MiniGoogLeNet, n_params, print_block_channel_analysis


def get_args():
    p = argparse.ArgumentParser(description="Bài 2: Huấn luyện GoogLeNet rút gọn trên VinaFood21")
    p.add_argument("--data_root", type=str, required=True,
                   help="Đường dẫn đến thư mục VinaFood21 (chứa train/ và test/)")
    p.add_argument("--img_size", type=int, default=64,
                   help="Kích thước ảnh vuông resize (mặc định: 64)")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Batch size (mặc định: 32)")
    p.add_argument("--epochs", type=int, default=20,
                   help="Số epoch huấn luyện (mặc định: 20)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate cho Adam (mặc định: 0.001)")
    p.add_argument("--num_blocks", type=int, default=4, choices=[3, 4],
                   help="Số lượng Inception block: 3 hoặc 4 (mặc định: 4)")
    p.add_argument("--lr_patience", type=int, default=2,
                   help="Số epoch không cải thiện val loss trước khi giảm LR (mặc định: 2)")
    p.add_argument("--lr_factor", type=float, default=0.5,
                   help="Hệ số giảm learning rate (mặc định: 0.5)")
    p.add_argument("--early_stop", type=int, default=5,
                   help="Patience cho Early Stopping (mặc định: 5, đặt 0 để tắt)")
    p.add_argument("--num_workers", type=int, default=2,
                   help="Số luồng nạp dữ liệu (mặc định: 2)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed ngẫu nhiên (mặc định: 42)")
    p.add_argument("--out_dir", type=str, default="outputs",
                   help="Thư mục lưu checkpoint và kết quả")
    return p.parse_args()


def save_metrics_plot(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    ax1.plot(epochs, [h["train_loss"] for h in history], label="Train Loss", color="blue")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="Val Loss", color="orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Đường cong Loss (GoogLeNet)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Macro-F1 & Accuracy
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
    print(f" BÀI 2: HUẤN LUYỆN GOOGLENET RÚT GỌN ({args.num_blocks} INCEPTION BLOCKS)")
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
    model = MiniGoogLeNet(in_ch=3, num_classes=len(classes), num_blocks=args.num_blocks).to(device)
    total_params = n_params(model)

    # In shape và đối chiếu channel của từng block trước khi huấn luyện (yêu cầu đề bài)
    print("\n--- ĐỐI CHIẾU SỐ CHANNEL OUTPUT TỪNG BLOCK (TÍNH TAY VS CODE) ---")
    print_block_channel_analysis(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=1e-6
    )

    history = []
    best_f1 = -1.0
    epochs_no_improve = 0
    total_train_time = 0.0

    print("\n--- BẮT ĐẦU HUẤN LUYỆN (CÓ REDUCE LR VÀ EARLY STOPPING) ---")
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc, ep_time = train_one_epoch(model, train_dl, criterion, optimizer, device)
        va_loss, va_acc, va_prec, va_rec, va_f1, _, _ = evaluate(model, val_dl, criterion, device)
        total_train_time += ep_time

        # Giảm LR nếu val loss không cải thiện
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(va_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        lr_changed_msg = f" | [LR giảm: {old_lr:.6f} -> {current_lr:.6f}]" if current_lr < old_lr else ""

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
            epochs_no_improve = 0
            torch.save({
                "model": model.state_dict(),
                "classes": classes,
                "img_size": args.img_size,
                "num_blocks": args.num_blocks,
                "epoch": ep,
                "val_f1": best_f1
            }, out_dir / "best.pt")
            mark = " [BEST]"
        else:
            epochs_no_improve += 1

        print(f"Epoch {ep:>2}/{args.epochs} [{ep_time:.1f}s | lr: {current_lr:.6f}] "
              f"Train Loss: {tr_loss:.4f} | Acc: {tr_acc:.4f} || "
              f"Val Loss: {va_loss:.4f} | Acc: {va_acc:.4f} | F1: {va_f1:.4f}{mark}{lr_changed_msg}")

        # Kiểm tra Early Stopping
        if args.early_stop > 0 and epochs_no_improve >= args.early_stop:
            print(f"\n[EARLY STOPPING] Kích hoạt sau {ep} epoch vì Val Macro-F1 không tăng trong {args.early_stop} epoch liên tiếp.")
            break

    avg_time_per_epoch = total_train_time / args.epochs
    print("\n==================================================================")
    print(f"HUẤN LUYỆN HOÀN TẤT!")
    print(f"Tổng tham số           : {total_params:,}")
    print(f"Best Val Macro-F1      : {best_f1:.4f}")
    print(f"Thời gian trung bình   : {avg_time_per_epoch:.2f} giây/epoch")
    print(f"Checkpoint tốt nhất lưu tại: {out_dir / 'best.pt'}")
    print("==================================================================")

    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    save_metrics_plot(history, out_dir / "loss_curve.png")
    print(f"Đã lưu biểu đồ huấn luyện tại: {out_dir / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
