import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from data import build_test_loader
from engine import evaluate
from model import LeNet5, n_params


def get_args():
    p = argparse.ArgumentParser(description="Bài 1: Đánh giá LeNet-5 trên tập Test")
    p.add_argument("--data_root", type=str, required=True,
                   help="Đường dẫn đến thư mục VinaFood21 (chứa test/)")
    p.add_argument("--ckpt", type=str, default="outputs/best.pt",
                   help="Đường dẫn tới file checkpoint .pt")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def save_confusion_matrix(cm, classes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=9)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=9)

    ax.set_xlabel("Nhãn dự đoán (Predicted)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Nhãn thực tế (Ground Truth)", fontsize=11, fontweight="bold")
    ax.set_title("Confusion Matrix (21x21) - LeNet-5", fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt)

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint tại: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt["classes"]
    img_size = ckpt["img_size"]
    pool_type = ckpt.get("pool", "avg")

    print("==================================================================")
    print(f" BÀI 1: ĐÁNH GIÁ MÔ HÌNH LENET-5 (POOL: {pool_type.upper()})")
    print("==================================================================")
    print(f"Checkpoint : {ckpt_path} (epoch {ckpt['epoch']})")
    print(f"Device     : {device}")
    print(f"Số lớp     : {len(classes)}")

    model = LeNet5(in_ch=3, img=img_size, num_classes=len(classes), pool=pool_type).to(device)
    model.load_state_dict(ckpt["model"])
    total_params = n_params(model)
    print(f"Số tham số : {total_params:,}")

    test_dl, test_classes = build_test_loader(
        args.data_root, img_size=img_size, batch_size=args.batch_size, num_workers=args.num_workers)
    assert test_classes == classes, "Danh sách lớp của tập test không khớp với model checkpoint!"

    criterion = nn.CrossEntropyLoss()
    loss, acc, prec, rec, macro_f1, y_true, y_pred = evaluate(model, test_dl, criterion, device)

    print("\n------------------------------------------------------------------")
    print(" BẢNG 4 ĐỘ ĐO ĐÁNH GIÁ TRÊN TẬP TEST:")
    print("------------------------------------------------------------------")
    print(f"1. Test Loss        : {loss:.4f}")
    print(f"2. Test Accuracy    : {acc * 100:.2f}% ({acc:.4f})")
    print(f"3. Macro Precision  : {prec * 100:.2f}% ({prec:.4f})")
    print(f"4. Macro Recall     : {rec * 100:.2f}% ({rec:.4f})")
    print(f"5. Macro F1-Score   : {macro_f1 * 100:.2f}% ({macro_f1:.4f})")
    print("------------------------------------------------------------------\n")

    print("--- BẢNG F1 VÀ CHỈ SỐ TỪNG LỚP (CLASSIFICATION REPORT) ---")
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print(report)

    # Lưu confusion matrix
    cm_path = ckpt_path.parent / f"confusion_matrix_{pool_type}.png"
    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(cm, classes, cm_path)
    print(f"\nĐã lưu ma trận nhầm lẫn tại: {cm_path}")

    # Lưu kết quả tổng hợp ra file JSON
    results = {
        "model": "LeNet5",
        "pooling": pool_type,
        "checkpoint": str(ckpt_path),
        "epoch": ckpt["epoch"],
        "num_params": total_params,
        "test_loss": round(loss, 4),
        "test_accuracy": round(acc, 4),
        "macro_precision": round(prec, 4),
        "macro_recall": round(rec, 4),
        "macro_f1": round(macro_f1, 4)
    }
    json_path = ckpt_path.parent / f"test_metrics_{pool_type}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Đã lưu kết quả tóm tắt tại : {json_path}")


if __name__ == "__main__":
    main()
