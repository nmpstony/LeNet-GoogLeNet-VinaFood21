import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """
    LeNet-5 điều chỉnh cho ảnh màu 64x64x3 trên tập dữ liệu VinaFood21 (21 lớp).
    Hỗ trợ cả 2 chế độ pooling:
      - 'avg': nn.AvgPool2d(2, 2) (bản gốc điều chỉnh theo B.3)
      - 'max': nn.MaxPool2d(2, 2) (dùng để so sánh hiệu năng theo yêu cầu Bài 1)
    """
    def __init__(self, in_ch=3, img=64, num_classes=21, pool="avg"):
        super().__init__()
        self.pool_type = pool.lower()
        PoolLayer = nn.AvgPool2d if self.pool_type == "avg" else nn.MaxPool2d

        # 7 layer chính (2 Conv, 2 Pool, 3 Linear)
        self.c1 = nn.Conv2d(in_ch, 6, kernel_size=5)
        self.s2 = PoolLayer(2, 2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.s4 = PoolLayer(2, 2)

        self.n_flat = self._flat_dim(in_ch, img)
        self.f5 = nn.Linear(self.n_flat, 120)
        self.f6 = nn.Linear(120, 84)
        self.out = nn.Linear(84, num_classes)

    def _flat_dim(self, in_ch, img):
        # Đo số chiều sau S4 bằng 1 lần forward giả lập
        with torch.no_grad():
            z = torch.zeros(1, in_ch, img, img)
            z = self.s4(self.c3(self.s2(self.c1(z))))
        return int(z.flatten(1).shape[1])

    def forward(self, x):
        x = self.s2(torch.relu(self.c1(x)))
        x = self.s4(torch.relu(self.c3(x)))
        x = x.flatten(1)
        x = torch.relu(self.f5(x))
        x = torch.relu(self.f6(x))
        return self.out(x)


def n_params(module):
    """Tính tổng số tham số có thể huấn luyện của module."""
    return sum(p.numel() for p in module.parameters())


def layer_table(model, img=64, in_ch=3):
    """Dựng bảng kích thước output và số tham số của từng layer (theo B.3 Bước 2)."""
    device = next(model.parameters()).device
    x = torch.zeros(1, in_ch, img, img, device=device)
    rows, total = [], 0
    for name, layer in [("C1", model.c1), ("S2", model.s2),
                        ("C3", model.c3), ("S4", model.s4)]:
        x = layer(x)
        p = n_params(layer)
        total += p
        rows.append((name, "x".join(map(str, tuple(x.shape)[1:])), p))

    x = x.flatten(1)
    rows.append(("flatten", str(x.shape[1]), 0))

    for name, layer in [("F5", model.f5), ("F6", model.f6), ("OUT", model.out)]:
        x = layer(x)
        p = n_params(layer)
        total += p
        rows.append((name, str(x.shape[1]), p))

    return rows, total


if __name__ == "__main__":
    print("=== KIỂM TRA MÔ HÌNH LENET-5 (AVG-POOL) ===")
    m_avg = LeNet5(in_ch=3, img=64, num_classes=21, pool="avg")
    x = torch.rand(2, 3, 64, 64)
    out_avg = m_avg(x)
    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(out_avg.shape))
    print("Số chiều sau flatten:", m_avg.n_flat)
    print("Tổng tham số        :", f"{n_params(m_avg):,}")

    rows, total = layer_table(m_avg, img=64)
    print("\nBảng tham số chi tiết từng layer (AvgPool):")
    print(f"{'Layer':<12}{'Output Shape':<18}{'Tham số':>12}")
    print("-" * 42)
    for name, shape, p in rows:
        print(f"{name:<12}{shape:<18}{p:>12,}")
    print("-" * 42)
    print(f"{'TỔNG':<12}{'':<18}{total:>12,}")

    print("\n=== KIỂM TRA MÔ HÌNH LENET-5 (MAX-POOL) ===")
    m_max = LeNet5(in_ch=3, img=64, num_classes=21, pool="max")
    out_max = m_max(x)
    print("Output shape:", tuple(out_max.shape))
    print("Tổng tham số:", f"{n_params(m_max):,}")
