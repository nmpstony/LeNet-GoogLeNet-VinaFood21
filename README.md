# DS201 - Lab 2: Mạng Neural Tích Chập (CNN) trên bộ dữ liệu VinaFood21

Kho lưu trữ này chứa mã nguồn hoàn chỉnh và kết quả thực nghiệm cho **Bài 1 (LeNet-5: AvgPool vs MaxPool)** và **Bài 2 (GoogLeNet: 4 Inception Blocks)** thuộc môn **DS201 – Deep Learning trong Khoa học Dữ liệu**. 

Toàn bộ mô hình đã được huấn luyện và đánh giá trên **GPU Tesla T4 (Kaggle)** với cơ chế **ReduceLROnPlateau** và **EarlyStopping**.

---

## 📁 1. Cấu trúc Repository

```
cnn_vinafood21/
├── .gitignore               # Loại bỏ checkpoint nặng và file tạm
├── requirements.txt         # Thư viện yêu cầu
├── README.md                # Báo cáo tổng hợp toàn bộ kết quả thực nghiệm
│
├── bai1_lenet/              # BÀI 1: LeNet-5 điều chỉnh (AvgPool & MaxPool)
│   ├── data.py              # Xử lý transforms (Augmentation) và dựng DataLoaders
│   ├── model.py             # Kiến trúc LeNet5, bảng tham số 339,421
│   ├── engine.py            # train_one_epoch và evaluate tính toán 4 độ đo
│   ├── train.py             # Huấn luyện mô hình với ReduceLROnPlateau & EarlyStopping
│   ├── evaluate.py          # Đánh giá test, in 4 độ đo, bảng F1 từng lớp và Confusion Matrix
│   ├── requirements.txt
│   ├── README.md            # Báo cáo chi tiết Bài 1
│   ├── outputs_avg/         # Kết quả thực nghiệm của bản AvgPool (checkpoints, logs, images)
│   └── outputs_max/         # Kết quả thực nghiệm của bản MaxPool (checkpoints, logs, images)
│
└── bai2_googlenet/          # BÀI 2: GoogLeNet rút gọn (4 Inception Blocks)
    ├── data.py              # Transforms và DataLoaders đồng bộ với Bài 1 (cùng seed 42)
    ├── model.py             # InceptionBlock, MiniGoogLeNet, bảng đối chiếu channel tính tay vs code
    ├── engine.py            # train_one_epoch và evaluate tính toán 4 độ đo
    ├── train.py             # Huấn luyện mô hình với ReduceLROnPlateau & EarlyStopping
    ├── evaluate.py          # Đánh giá test, in 4 độ đo, bảng F1 từng lớp và Confusion Matrix
    ├── requirements.txt
    ├── README.md            # Báo cáo chi tiết Bài 2
    └── outputs/             # Kết quả thực nghiệm của GoogLeNet (checkpoints, logs, images)
```

---

## 🍽️ 2. Bộ dữ liệu VinaFood21

- **Số lớp**: 21 món ăn Việt Nam (`banh-can`, `banh-mi-chao`, `pho`, `bun-bo-hue`, ...).
- **Kích thước ảnh**: Thống nhất resize về `64 × 64` cho cả hai bài để đảm bảo tính so sánh công bằng.
- **Phân chia tập dữ liệu**:
  - Tập train: 9,040 ảnh (sau khi tách 10% tập validation với cùng một `seed = 42`).
  - Tập val: 1,004 ảnh.
  - Tập test: 6,682 ảnh dùng để đánh giá mô hình.
- **Data Augmentation**: RandomResizedCrop (scale 0.7-1.0), RandomHorizontalFlip, RandomRotation(15°), ColorJitter(0.2).

---

## 📊 3. BẢNG TỔNG HỢP SO SÁNH KẾT QUẢ THỰC NGHIỆM

Bảng số liệu thực tế được đo trên toàn bộ **6,682 ảnh tập Test** của VinaFood21 (chạy trên GPU Tesla T4):

| Tiêu chí so sánh | Bài 1: LeNet-5 (AvgPool) | Bài 1: LeNet-5 (MaxPool) | Bài 2: GoogLeNet rút gọn (4 Blocks) |
|---|:---:|:---:|:---:|
| **Số tham số (Parameters)** | 339,421 | 339,421 | **231,705** *(giảm 31.7%)* |
| **Phân bổ tham số** | Lớp F5 chiếm 95.6% tham số | Lớp F5 chiếm 95.6% tham số | Dùng **Global Average Pooling** (chỉ 4,725 tham số FC) |
| **Số Epoch thực tế (EarlyStop)** | 41 epochs *(dừng sớm)* | 63 epochs *(dừng sớm)* | 75 epochs *(dừng sớm)* |
| **Best Checkpoint tại Epoch** | **Epoch 31** | **Epoch 53** | **Epoch 65** |
| **Thời gian huấn luyện / epoch** | 68.03 s/epoch | **56.13 s/epoch** | **57.07 s/epoch** |
| **Test Loss** | 2.3484 | 2.2536 | **1.5896** *(tốt nhất)* |
| **Test Accuracy** | 32.12% | 35.41% | **52.66%** *(+17.25%)* |
| **Macro-Precision** | 31.59% | 34.33% | **53.42%** *(+19.09%)* |
| **Macro-Recall** | 30.51% | 34.63% | **51.47%** *(+16.84%)* |
| **Macro-F1 (Độ đo chính)** | **30.32%** | **34.09%** | **51.55%** *(+17.46%)* |

---

## 🖼️ 4. Hình ảnh trực quan hoá kết quả

### 4.1. So sánh đường cong huấn luyện (Loss & Validation Macro-F1)

| LeNet-5 (AvgPool) | LeNet-5 (MaxPool) | GoogLeNet rút gọn (4 Blocks) |
|:---:|:---:|:---:|
| ![Loss Avg](bai1_lenet/outputs_avg/loss_curve.png) | ![Loss Max](bai1_lenet/outputs_max/loss_curve.png) | ![Loss GoogLeNet](bai2_googlenet/outputs/loss_curve.png) |

---

### 4.2. So sánh Ma trận nhầm lẫn (Confusion Matrix 21×21 trên tập Test)

| LeNet-5 (AvgPool) | LeNet-5 (MaxPool) | GoogLeNet rút gọn (4 Blocks) |
|:---:|:---:|:---:|
| ![CM Avg](bai1_lenet/outputs_avg/confusion_matrix_avg.png) | ![CM Max](bai1_lenet/outputs_max/confusion_matrix_max.png) | ![CM GoogLeNet](bai2_googlenet/outputs/confusion_matrix_googlenet.png) |
