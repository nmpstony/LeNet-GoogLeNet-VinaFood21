# Bài 1: Xây dựng và Huấn luyện mô hình LeNet-5 trên VinaFood21

Dự án này triển khai kiến trúc LeNet-5 điều chỉnh (theo hướng dẫn B.3 trong tài liệu DS201 Lab 2) để phân loại 21 món ăn Việt Nam từ bộ dữ liệu VinaFood21.

## 1. Cấu trúc thư mục

```
bai1_lenet/
├── data.py          # Xử lý transforms (Augmentation) và dựng DataLoaders (Train/Val/Test)
├── model.py         # Kiến trúc LeNet5 (hỗ trợ cả AvgPool2d và MaxPool2d), bảng đếm tham số
├── engine.py        # Các hàm train_one_epoch và evaluate tính toán 4 độ đo
├── train.py         # Huấn luyện mô hình, lưu checkpoint tốt nhất và biểu đồ loss
├── evaluate.py      # Đánh giá trên tập test, in 4 độ đo, bảng F1 và ma trận nhầm lẫn
├── requirements.txt # Danh sách thư viện cần thiết
└── README.md        # Hướng dẫn sử dụng chi tiết
```

## 2. Đặc tả kiến trúc LeNet-5 điều chỉnh

- **Ảnh đầu vào**: 64 × 64 × 3 (ảnh màu).
- **C1**: Conv2d(3, 6, kernel=5), ReLU -> output: 60 × 60 × 6 (456 tham số)
- **S2**: AvgPool2d(2, 2) hoặc MaxPool2d(2, 2) -> output: 30 × 30 × 6 (0 tham số)
- **C3**: Conv2d(6, 16, kernel=5), ReLU -> output: 26 × 26 × 16 (2,416 tham số)
- **S4**: AvgPool2d(2, 2) hoặc MaxPool2d(2, 2) -> output: 13 × 13 × 16 (0 tham số)
- **Flatten**: 13 × 13 × 16 = 2,704 chiều
- **F5**: Linear(2704, 120), ReLU -> output: 120 (324,600 tham số)
- **F6**: Linear(120, 84), ReLU -> output: 84 (10,164 tham số)
- **OUT**: Linear(84, 21) -> output: 21 (1,785 tham số)
- **Tổng số tham số**: **339,421** tham số (khớp chính xác lý thuyết).

## 3. Cài đặt và Hướng dẫn chạy

### Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### Kiểm tra mô hình và số tham số
```bash
python model.py
```

### Huấn luyện mô hình

**1. Phiên bản mặc định với AvgPool2d:**
```bash
python train.py --data_root /path/to/VinaFood21 --epochs 20 --batch_size 32 --lr 1e-3 --pool avg --out_dir outputs_avg
```

**2. Phiên bản so sánh với MaxPool2d:**
```bash
python train.py --data_root /path/to/VinaFood21 --epochs 20 --batch_size 32 --lr 1e-3 --pool max --out_dir outputs_max
```

### Đánh giá trên tập test

**Đánh giá bản AvgPool:**
```bash
python evaluate.py --data_root /path/to/VinaFood21 --ckpt outputs_avg/best.pt
```

**Đánh giá bản MaxPool:**
```bash
python evaluate.py --data_root /path/to/VinaFood21 --ckpt outputs_max/best.pt
```

## 4. Báo cáo kết quả yêu cầu

Sau khi chạy xong, kết quả lưu trong thư mục outputs gồm:
- `best.pt`: Checkpoint trọng số mô hình tốt nhất (dựa trên Macro-F1 tập val).
- `history.csv`: Lịch sử loss, accuracy, precision, recall, f1 và thời gian từng epoch.
- `loss_curve.png`: Biểu đồ trực quan hoá loss và val F1.
- `confusion_matrix_avg.png` / `confusion_matrix_max.png`: Ma trận nhầm lẫn kích thước 21×21.
- `test_metrics_avg.json` / `test_metrics_max.json`: Kết quả 4 độ đo trên tập test.
