import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# MODEL MIMARISI (Sadece Sınıflar)
# ==============================================================================

class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1),
            nn.InstanceNorm2d(c),
            nn.ReLU(True),
            nn.Conv2d(c, c, 3, 1, 1),
            nn.InstanceNorm2d(c)
        )
    def forward(self, x): return x + self.block(x)

class StegoGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(3, 32, 9, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(35, 64, 3, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )
        self.res = nn.Sequential(*[Residual(128) for _ in range(5)])
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(True)
        )
        self.dec3 = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(32, 3, 9, 1),
            nn.Sigmoid()
        )

    def forward(self, content, msg):
        x = self.enc1(content)
        # Boyut uyuşmazlığı varsa mesajı content boyutuna getir
        if msg.shape[2:] != x.shape[2:]:
            msg = F.interpolate(msg, size=x.shape[2:], mode="nearest")
        
        # Kanal birleştirme (Concatenation)
        x = torch.cat([x, msg], dim=1)
        
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.res(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        return x

class PureGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(3, 32, 9, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )
        self.res = nn.Sequential(*[Residual(128) for _ in range(5)])
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(True)
        )
        self.dec3 = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(32, 3, 9, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.res(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        return x

class Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(3, 32, 9, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            *[Residual(128) for _ in range(5)],
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
            nn.ReflectionPad2d(4),
            nn.Conv2d(32, 3, 9, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.model(x)

    # ==============================================================================
# DISCRIMINATOR (SRNet) VE YARDIMCILARI (models.py'nin EN ALTINA EKLE)
# ==============================================================================
class LayerType1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
    def forward(self, x): return F.relu(self.bn(self.conv(x)))

class LayerType2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False); self.b1 = nn.BatchNorm2d(out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        res = x
        out = F.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        return out + res

class LayerType3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False); self.b1 = nn.BatchNorm2d(out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(out_ch)
        self.pool = nn.AvgPool2d(3, 2, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, 2, bias=False)
    def forward(self, x):
        res = self.skip(x)
        out = F.relu(self.b1(self.c1(x)))
        out = self.pool(self.b2(self.c2(out)))
        return out + res

class SRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            LayerType1(3, 64),
            LayerType1(64, 16),
            *[LayerType2(16, 16) for _ in range(5)],
            LayerType3(16, 16),
            LayerType3(16, 64),
            LayerType3(64, 128),
            LayerType3(128, 256),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.fc = nn.Linear(256, 2)
    def forward(self, x):
        return self.fc(self.layers(x))