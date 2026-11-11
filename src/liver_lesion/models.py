"""3D U-Net and gated attention U-Net architectures."""
import torch
from torch import nn
from torch.nn import functional as F
from .config import CHANNELS, ATTN_LEVELS

class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout, stride=1):
        super().__init__(
            nn.Conv3d(cin, cout, 3, stride, 1, bias=False), nn.InstanceNorm3d(cout, affine=True), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(cout, cout, 3, 1, 1, bias=False), nn.InstanceNorm3d(cout, affine=True), nn.LeakyReLU(0.01, inplace=True))

class AttentionGate(nn.Module):
    def __init__(self, c_x, c_g, c_int):
        super().__init__()
        self.wx = nn.Conv3d(c_x, c_int, 1, bias=False)
        self.wg = nn.Conv3d(c_g, c_int, 1)
        self.psi = nn.Conv3d(c_int, 1, 1)
        nn.init.constant_(self.psi.bias, 3.0)      # sigmoid(3) ~ 0.95: gate starts open
        self.keep_map, self.last_map = False, None

    def forward(self, x, g):
        alpha = torch.sigmoid(self.psi(F.relu(self.wx(x) + self.wg(g))))
        if self.keep_map:
            self.last_map = alpha.detach().float().cpu()
        return x * alpha

class UNet3D(nn.Module):
    def __init__(self, in_ch=1, n_classes=3, channels=CHANNELS, attention=False, attn_levels=ATTN_LEVELS):
        super().__init__()
        ch = list(channels)
        self.enc = nn.ModuleList([ConvBlock(in_ch, ch[0])] + [ConvBlock(ch[i - 1], ch[i], 2) for i in range(1, len(ch))])
        self.up, self.dec, self.gates = nn.ModuleList(), nn.ModuleList(), nn.ModuleDict()
        self.levels = list(reversed(range(len(ch) - 1)))
        for i in self.levels:
            self.up.append(nn.ConvTranspose3d(ch[i + 1], ch[i], 2, 2))
            self.dec.append(ConvBlock(2 * ch[i], ch[i]))
            if attention and i in attn_levels:
                self.gates[str(i)] = AttentionGate(ch[i], ch[i], max(ch[i] // 2, 8))
        self.head = nn.Conv3d(ch[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < len(self.enc) - 1:
                skips.append(x)
        for k, i in enumerate(self.levels):
            x = self.up[k](x)
            s = skips[i]
            if str(i) in self.gates:
                s = self.gates[str(i)](s, x)
            x = self.dec[k](torch.cat([s, x], dim=1))
        return self.head(x)

def build_model(arch):
    assert arch in ("unet", "attn")
    return UNet3D(attention=(arch == "attn"))
