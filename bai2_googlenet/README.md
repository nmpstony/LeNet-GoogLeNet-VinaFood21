# Bài 2: Xây dựng và Huấn luyện mô hình GoogLeNet rút gọn trên VinaFood21

Dự án này triển khai kiến trúc GoogLeNet rút gọn với 3 đến 4 Inception Blocks (theo hướng dẫn B.4 trong tài liệu DS201 Lab 2) để phân loại 21 món ăn Việt Nam từ bộ dữ liệu VinaFood21.

## 1. Cấu trúc thư mục

```
bai2_googlenet/
├── data.py          # Data augmentation, DataLoader và phân chia Train/Val/Test
├── model.py         # InceptionBlock, MiniGoogLeNet và hàm đối chiếu channel lý thuyết vs code
├── engine.py        # train_one_epoch và evaluate tính toán 4 độ đo (Acc, Prec, Rec, F1)
├── train.py         # Huấn luyện mô hình, lưu checkpoint tốt nhất và biểu đồ loss
├── evaluate.py      # Đánh giá trên tập test, in bảng 4 độ đo, bảng F1 và ma trận nhầm lẫn
├── requirements.txt # Danh sách thư viện cần thiết
└── README.md        # Hướng dẫn chi tiết
```

## 2. Đặc tả kiến trúc GoogLeNet rút gọn

- **Ảnh đầu vào**: 64 × 64 × 3 (đồng bộ với Bài 1 để đảm bảo tính so sánh công bằng).
- **Stem**: Conv2d(3, 64, kernel=3, padding=1) + ReLU -> Output: 64 × 64 × 64.
- **Inception Block 1**: `(64, 16, 16, 24, 4, 8, 8)`
  - Nhánh 1 (1×1): 16 channels
  - Nhánh 2 (1×1 -> 3×3): 24 channels
  - Nhánh 3 (1×1 -> 5×5): 8 channels
  - Nhánh 4 (pool -> 1×1): 8 channels
  - **Output channel**: 16 + 24 + 8 + 8 = **56 channels** (theo công thức lý thuyết $f_1 + f_3 + f_5 + f_p$).
  - MaxPool 3×3 (stride 2) -> Output: 32 × 32 × 56.
- **Inception Block 2**: `(56, 32, 32, 48, 8, 16, 16)`
  - Nhánh 1 (1×1): 32 channels
  - Nhánh 2 (1×1 -> 3×3): 48 channels
  - Nhánh 3 (1×1 -> 5×5): 16 channels
  - Nhánh 4 (pool -> 1×1): 16 channels
  - **Output channel**: 32 + 48 + 16 + 16 = **112 channels** (22,104 tham số, khớp mục B.4 Bước 3).
  - MaxPool 3×3 (stride 2) -> Output: 16 × 16 × 112.
- **Inception Block 3**: `(112, 64, 64, 96, 16, 32, 32)`
  - **Output channel**: 64 + 96 + 32 + 32 = **224 channels**.
- **Inception Block 4**: `(224, 64, 64, 96, 16, 32, 32)`
  - **Output channel**: 64 + 96 + 32 + 32 = **224 channels**.
  - MaxPool 3×3 (stride 2) -> Output: 8 × 8 × 224.
- **Phân loại**:
  - Dùng `nn.AdaptiveAvgPool2d((1, 1))` (GAP) thay thế `nn.Flatten()` để giảm mạnh số tham số và tránh ghi nhớ dữ liệu.
  - Dropout(0.4) -> Linear(224, 21).

## 3. Cài đặt và Hướng dẫn chạy

### Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### Kiểm tra mô hình và đối chiếu channel
```bash
python model.py
```

### Huấn luyện mô hình (mặc định 4 blocks)
```bash
python train.py --data_root /path/to/VinaFood21 --epochs 20 --batch_size 32 --lr 1e-3 --num_blocks 4 --out_dir outputs
```

*(Hoặc chạy với cấu hình 3 Inception blocks: `--num_blocks 3`)*

### Đánh giá trên tập test
```bash
python evaluate.py --data_root /path/to/VinaFood21 --ckpt outputs/best.pt
```

## 4. Báo cáo kết quả yêu cầu

Sau khi chạy xong, kết quả lưu trong thư mục `outputs/` gồm:
- `best.pt`: Trọng số mô hình có Macro-F1 tập val cao nhất.
- `history.csv`: Bảng ghi chép chi tiết qua từng epoch: Loss, Accuracy, Precision, Recall, F1 và thời gian huấn luyện.
- `loss_curve.png`: Biểu đồ đường cong Loss và Macro-F1.
- `confusion_matrix_googlenet.png`: Ma trận nhầm lẫn kích thước 21×21.
- `test_metrics_googlenet.json`: Tóm tắt các độ đo chính trên tập test.
