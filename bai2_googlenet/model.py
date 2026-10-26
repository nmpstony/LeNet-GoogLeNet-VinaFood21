import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    """
    Inception block tiêu chuẩn với 4 nhánh song song (theo mục B.4 / A.6):
      - Nhánh 1 (1x1): Conv 1x1 -> f1
      - Nhánh 2 (1x1 -> 3x3): Conv 1x1 -> f3r, Conv 3x3 (pad 1) -> f3
      - Nhánh 3 (1x1 -> 5x5): Conv 1x1 -> f5r, Conv 5x5 (pad 2) -> f5
      - Nhánh 4 (pool -> 1x1): MaxPool 3x3 (stride 1, pad 1), Conv 1x1 -> fp
    Nối các nhánh theo chiều channel (dim=1).
    Số channel output = f1 + f3 + f5 + fp (f3r và f5r là số channel trung gian).
    """
    def __init__(self, in_ch, f1, f3r, f3, f5r, f5, fp):
        super().__init__()
        self.in_ch = in_ch
        self.f1 = f1
        self.f3r = f3r
        self.f3 = f3
        self.f5r = f5r
        self.f5 = f5
        self.fp = fp
        self.expected_out_channels = f1 + f3 + f5 + fp

        # Nhánh 1: conv 1x1
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, f1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        # Nhánh 2: conv 1x1 -> conv 3x3
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, f3r, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f3r, f3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Nhánh 3: conv 1x1 -> conv 5x5
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, f5r, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f5r, f5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        # Nhánh 4: maxpool 3x3 stride 1 -> conv 1x1
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_ch, fp, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Nối 4 nhánh theo chiều channel (dim=1)
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class MiniGoogLeNet(nn.Module):
    """
    Mô hình GoogLeNet rút gọn cho ảnh 64x64x3 trên VinaFood21:
      - 3 đến 4 Inception blocks
      - Không dùng auxiliary classifier
      - Nối nhánh bằng torch.cat([b1, b2, b3, b4], dim=1)
      - nn.AdaptiveAvgPool2d(1) thay cho nn.Flatten() trước lớp phân loại
    """
    def __init__(self, in_ch=3, num_classes=21, num_blocks=4):
        super().__init__()
        assert num_blocks in [3, 4], "Chỉ hỗ trợ cấu hình 3 hoặc 4 Inception block theo yêu cầu đề bài!"
        self.num_blocks = num_blocks

        # Stem: Conv 3x3 chuyển từ 3 channels -> 64 channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Block 1: in_ch=64 -> out_ch=16+24+8+8=56
        self.block1 = InceptionBlock(64, f1=16, f3r=16, f3=24, f5r=4, f5=8, fp=8)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Block 2: in_ch=56 -> out_ch=32+48+16+16=112 (Khớp chính xác B.4 Bước 3)
        self.block2 = InceptionBlock(56, f1=32, f3r=32, f3=48, f5r=8, f5=16, fp=16)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Block 3: in_ch=112 -> out_ch=64+96+32+32=224
        self.block3 = InceptionBlock(112, f1=64, f3r=64, f3=96, f5r=16, f5=32, fp=32)

        if num_blocks == 4:
            # Block 4: in_ch=224 -> out_ch=64+96+32+32=224
            self.block4 = InceptionBlock(224, f1=64, f3r=64, f3=96, f5r=16, f5=32, fp=32)
            self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            final_channels = 224
        else:
            self.block4 = None
            self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            final_channels = 224

        # Phân loại: Dùng Global Average Pooling thay vì Flatten
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(final_channels, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)
        if self.block4 is not None:
            x = self.block4(x)
        x = self.pool3(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)

    def get_block_info(self, sample_input=None):
        """
        Trích xuất thông tin output shape và số channel lý thuyết vs thực tế của từng block.
        """
        device = next(self.parameters()).device
        if sample_input is None:
            sample_input = torch.zeros(1, 3, 64, 64, device=device)
        else:
            sample_input = sample_input.to(device)

        info = []
        with torch.no_grad():
            x = self.stem(sample_input)
            info.append(("Stem Conv (3->64)", tuple(x.shape), 64, 64, n_params(self.stem)))

            # Block 1
            x = self.block1(x)
            info.append(("Inception Block 1 (64->56)", tuple(x.shape),
                         self.block1.expected_out_channels, x.shape[1], n_params(self.block1)))
            x = self.pool1(x)

            # Block 2
            x = self.block2(x)
            info.append(("Inception Block 2 (56->112)", tuple(x.shape),
                         self.block2.expected_out_channels, x.shape[1], n_params(self.block2)))
            x = self.pool2(x)

            # Block 3
            x = self.block3(x)
            info.append(("Inception Block 3 (112->224)", tuple(x.shape),
                         self.block3.expected_out_channels, x.shape[1], n_params(self.block3)))

            # Block 4 nếu có
            if self.block4 is not None:
                x = self.block4(x)
                info.append(("Inception Block 4 (224->224)", tuple(x.shape),
                             self.block4.expected_out_channels, x.shape[1], n_params(self.block4)))
            x = self.pool3(x)

            x = self.gap(x)
            info.append(("GlobalAvgPool2d(1)", tuple(x.shape), x.shape[1], x.shape[1], 0))

            x = torch.flatten(x, 1)
            logits = self.classifier(self.dropout(x))
            info.append(("Classifier (FC)", tuple(logits.shape), logits.shape[1], logits.shape[1], n_params(self.classifier)))

        return info


def n_params(module):
    """Tính tổng số tham số có thể huấn luyện của module."""
    return sum(p.numel() for p in module.parameters())


def print_block_channel_analysis(model):
    """In bảng phân tích đối chiếu channel lý thuyết (tính tay) và code."""
    info = model.get_block_info()
    print("=" * 80)
    print(f"{'Tên tầng / Khối':<30}{'Output Shape':<18}{'C_lý_thuyết':>12}{'C_thực_tế':>10}{'Tham số':>10}")
    print("=" * 80)
    for name, shape, c_theory, c_actual, params in info:
        print(f"{name:<30}{str(shape):<18}{c_theory:>12}{c_actual:>10}{params:>10,}")
    print("=" * 80)
    print(f"Tổng số tham số toàn mạng: {n_params(model):,}")


if __name__ == "__main__":
    print("=== KIỂM TRA MÔ HÌNH GOOGLENET RÚT GỌN (4 INCEPTION BLOCKS) ===")
    model = MiniGoogLeNet(in_ch=3, num_classes=21, num_blocks=4)
    print_block_channel_analysis(model)

    print("\nKiểm tra forward pass với dummy batch:")
    x = torch.rand(2, 3, 64, 64)
    out = model(x)
    print(f"Input shape : {tuple(x.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
    assert out.shape == (2, 21), "Lỗi output shape không phải (2, 21)!"
    print("Kiểm tra kiến trúc GoogLeNet thành công!")
