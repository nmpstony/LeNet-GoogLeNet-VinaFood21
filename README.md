# DS201 - Lab 2: Mạng Neural Tích Chập (CNN) trên bộ dữ liệu VinaFood21

Kho lưu trữ này chứa mã nguồn hoàn chỉnh cho **Bài 1** và **Bài 2** của Lab 2 môn **DS201 – Deep Learning trong Khoa học Dữ liệu**, được tổ chức thành 2 dự án độc lập nằm trong cùng 1 repository để tiện quản lý và nộp bài lên GitHub.

---

## 📁 1. Cấu trúc Repository

```
cnn_vinafood21/
├── .gitignore               # Loại bỏ các file tạm, checkpoint .pt nặng và thư mục outputs
├── requirements.txt         # Thư viện yêu cầu dùng chung
├── README.md                # Tài liệu tổng quan toàn bộ repository và so sánh kết quả
│
├── bai1_lenet/              # BÀI 1: LeNet-5 điều chỉnh (AvgPool & MaxPool)
│   ├── data.py              # Xử lý transforms (Augmentation) và dựng DataLoaders (Train/Val/Test)
│   ├── model.py             # Kiến trúc LeNet5, hỗ trợ AvgPool2d / MaxPool2d, bảng tham số chi tiết
│   ├── engine.py            # Hàm train_one_epoch và evaluate tính toán 4 độ đo (Acc, Prec, Rec, F1)
│   ├── train.py             # Huấn luyện mô hình, lưu checkpoint tốt nhất và biểu đồ loss
│   ├── evaluate.py          # Đánh giá trên test, in 4 độ đo, bảng F1 từng lớp và Confusion Matrix
│   ├── requirements.txt     # Phụ thuộc của Bài 1
│   └── README.md            # Hướng dẫn chi tiết Bài 1
│
└── bai2_googlenet/          # BÀI 2: GoogLeNet rút gọn (Inception Blocks)
    ├── data.py              # Xử lý transforms và DataLoaders đồng bộ với Bài 1 (cùng seed 42)
    ├── model.py             # InceptionBlock, MiniGoogLeNet, bảng đối chiếu channel tính tay vs code
    ├── engine.py            # Hàm train_one_epoch và evaluate tính toán 4 độ đo
    ├── train.py             # Huấn luyện mô hình, in shape trước khi train, lưu history và checkpoint
    ├── evaluate.py          # Đánh giá trên test, in 4 độ đo, bảng F1 từng lớp và Confusion Matrix
    ├── requirements.txt     # Phụ thuộc của Bài 2
    └── README.md            # Hướng dẫn chi tiết Bài 2
```

---

## 🍽️ 2. Bộ dữ liệu VinaFood21

- **Số lớp**: 21 món ăn Việt Nam (`banh-can`, `banh-mi-chao`, `pho`, `bun-bo-hue`, ...).
- **Kích thước ảnh**: Thống nhất resize về `64 × 64` cho cả hai bài để đảm bảo tính so sánh công bằng.
- **Phân chia tập dữ liệu**:
  - Tập train: 10,044 ảnh (khi huấn luyện tự động tách 10% làm tập validation với cùng một `seed = 42`).
  - Tập test: 6,682 ảnh dùng để đánh giá mô hình.
- **Chuẩn hoá**: Mean `[0.485, 0.456, 0.406]` và Std `[0.229, 0.224, 0.225]`.
- **Data Augmentation (chỉ áp dụng tập train)**: RandomResizedCrop (scale 0.7-1.0), RandomHorizontalFlip, RandomRotation(15°), ColorJitter(0.2).

---

## 🚀 3. Hướng dẫn chạy nhanh

### A. Chạy trên Kaggle (GPU miễn phí)
Dự án đã được tích hợp sẵn Kaggle Kernel và đang chạy tại:
🔗 **[Kaggle Notebook: DS201 Lab02 VinaFood21 CNN](https://www.kaggle.com/code/nmpstony/ds201-lab02-vinafood21-cnn)**
- GPU: Đã bật GPU (T4 / P100)
- Tự động mount bộ dữ liệu `sonbui25/vinafood21`
- Tự động chạy Bài 1 (AvgPool, MaxPool) và Bài 2 (GoogLeNet)
- Đầy đủ `ReduceLROnPlateau` và `EarlyStopping`

---

### B. Chạy trên máy cá nhân (Local với GPU)

```bash
pip install -r requirements.txt
```

#### Chạy Bài 1: LeNet-5
```bash
cd bai1_lenet

# 1. Kiểm tra shape và bảng tham số
python model.py

# 2. Huấn luyện bản AvgPool (với ReduceLROnPlateau & EarlyStopping)
python train.py --data_root ../../VinaFood21 --epochs 20 --batch_size 64 --num_workers 4 --pool avg --patience_lr 2 --patience_es 5 --out_dir outputs_avg

# 3. Huấn luyện bản MaxPool để so sánh
python train.py --data_root ../../VinaFood21 --epochs 20 --batch_size 64 --num_workers 4 --pool max --patience_lr 2 --patience_es 5 --out_dir outputs_max

# 4. Đánh giá trên tập test
python evaluate.py --data_root ../../VinaFood21 --ckpt outputs_avg/best.pt
python evaluate.py --data_root ../../VinaFood21 --ckpt outputs_max/best.pt
```

#### Chạy Bài 2: GoogLeNet rút gọn
```bash
cd ../bai2_googlenet

# 1. Kiểm tra kiến trúc và đối chiếu số channel output từng block
python model.py

# 2. Huấn luyện mô hình (4 Inception blocks, Global Average Pooling)
python train.py --data_root ../../VinaFood21 --epochs 20 --batch_size 64 --num_workers 4 --num_blocks 4 --patience_lr 2 --patience_es 5 --out_dir outputs

# 3. Đánh giá trên tập test
python evaluate.py --data_root ../../VinaFood21 --ckpt outputs/best.pt
```

---

## 📊 4. Đối chiếu số Channel Output từng Inception Block (Bài 2)

Theo nguyên lý Inception, công thức tính số channel đầu ra của một block là:
$$\text{Output Channels} = f_1 + f_3 + f_5 + f_p$$
*(Trong đó $f_{3r}$ và $f_{5r}$ là số channel giảm chiều trung gian qua conv 1×1, không tính vào output)*.

| Khối / Tầng | Cấu hình tham số $(in\_ch, f_1, f_{3r}, f_3, f_{5r}, f_5, f_p)$ | Channel ra (Tính tay) | Channel ra (In từ code) | Số tham số |
|---|---|:---:|:---:|---:|
| **Stem Conv** | Conv2d(3, 64, kernel=3, pad=1) | 64 | 64 | 1,792 |
| **Inception 1** | $(64, 16, 16, 24, 4, 8, 8)$ | $16 + 24 + 8 + 8 = 56$ | **56** | 7,148 |
| **Inception 2** | $(56, 32, 32, 48, 8, 16, 16)$ | $32 + 48 + 16 + 16 = 112$ | **112** | **22,104** *(khớp B.4 Bước 3)* |
| **Inception 3** | $(112, 64, 64, 96, 16, 32, 32)$ | $64 + 96 + 32 + 32 = 224$ | **224** | 88,112 |
| **Inception 4** | $(224, 64, 64, 96, 16, 32, 32)$ | $64 + 96 + 32 + 32 = 224$ | **224** | 107,824 |
| **Classifier** | GAP + Dropout(0.4) + Linear(224, 21) | 21 | 21 | 4,725 |
| **TỔNG CỘNG** | **Toàn bộ mạng MiniGoogLeNet** | | | **231,705** |

---

## 📈 5. Bảng tổng hợp so sánh giữa Bài 1 và Bài 2

| Tiêu chí so sánh | Bài 1: LeNet-5 (AvgPool) | Bài 1: LeNet-5 (MaxPool) | Bài 2: GoogLeNet rút gọn (4 Blocks) |
|---|:---:|:---:|:---:|
| **Số tham số (Parameters)** | **339,421** | **339,421** | **231,705** |
| **Đặc điểm kích thước mô hình** | Lớp F5 sau Flatten chiếm ~95.6% tham số | Lớp F5 sau Flatten chiếm ~95.6% tham số | Dùng **Global Average Pooling** giảm mạnh tham số |
| **Thời gian huấn luyện / epoch** | Nhanh (~1.0x) | Nhanh (~1.0x) | Khoảng 1.5x - 2.0x (nhiều nhánh conv song song) |
| **Trích xuất đặc trưng** | Tuần tự (filter 5×5) | Tuần tự (filter 5×5) | Đa tỉ lệ (1×1, 3×3, 5×5 đồng thời) |
| **Macro-Precision (Test)** | Đo sau khi huấn luyện | Đo sau khi huấn luyện | Đo sau khi huấn luyện |
| **Macro-Recall (Test)** | Đo sau khi huấn luyện | Đo sau khi huấn luyện | Đo sau khi huấn luyện |
| **Macro-F1 (Test)** | Độ đo chính | Độ đo chính | Độ đo chính |
| **Accuracy (Test)** | Đo sau khi huấn luyện | Đo sau khi huấn luyện | Đo sau khi huấn luyện |

---

## 💡 6. Nhận xét và Kết luận

1. **Hiệu quả của Bottleneck 1×1 và Global Average Pooling (GAP) ở GoogLeNet**:
   - Ở LeNet-5, mặc dù mạng chỉ có 2 lớp Conv nhưng do dùng `Flatten` từ tensor kích thước $13 \times 13 \times 16 = 2,704$ vào lớp `Linear(2704, 120)`, riêng lớp `F5` đã tiêu tốn tới **324,600** tham số (chiếm tới 95.6% toàn bộ mạng).
   - Ở GoogLeNet, nhờ cơ chế **Global Average Pooling** gom feature map về vector 1 chiều trước khi đưa vào classifier, mạng sâu hơn nhiều (4 Inception blocks đa tỉ lệ) nhưng tổng số tham số chỉ là **279,549** (ít hơn cả LeNet-5), giúp hạn chế hiện tượng overfitting và ghi nhớ dữ liệu.
2. **Khả năng nắm bắt đặc trưng**:
   - GoogLeNet trích xuất đặc trưng đa quy mô (multi-scale) nhờ kết hợp filter 1×1, 3×3 và 5×5 cùng lúc, giúp nhận diện các món ăn Việt Nam với nhiều chi tiết phức tạp tốt hơn cấu trúc tuần tự của LeNet-5.
3. **AvgPool vs MaxPool (Bài 1)**:
   - MaxPool giữ lại các đặc trưng cục bộ có cường độ kích hoạt cao nhất (biên cạnh, đường nét món ăn), thường đem lại độ chính xác và macro-F1 cải thiện hơn so với AvgPool vốn làm mịn các giá trị kích hoạt.
