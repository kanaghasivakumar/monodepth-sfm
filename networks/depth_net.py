import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


def conv_bn_relu(in_ch, out_ch, kernel=3, stride=1, padding=1):
    """Standard Conv -> BatchNorm -> ELU block used throughout the decoder."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ELU(inplace=True)
    )


class DepthDecoder(nn.Module):
    """
    Multi-scale decoder with skip connections from the ResNet18 encoder.

    Encoder feature map channels (ResNet18):
        layer0 (stem):  64  @ H/2  x W/2
        layer1:         64  @ H/4  x W/4
        layer2:         128 @ H/8  x W/8
        layer3:         256 @ H/16 x W/16
        layer4:         512 @ H/32 x W/32

    We decode upward from layer4, merging skip connections at each scale,
    and emit a disparity map at 4 scales (full res, 1/2, 1/4, 1/8).
    """

    def __init__(self, num_ch_enc=None):
        super().__init__()

        # Default ResNet18 encoder output channels per stage
        if num_ch_enc is None:
            num_ch_enc = [64, 64, 128, 256, 512]

        # Decoder channel widths (top-down)
        self.num_ch_dec = [16, 32, 64, 128, 256]

        # Upconv layers: each upsamples by 2x then refines with skip
        # upconv[i] takes encoder[i+1] channels + encoder[i] skip channels
        self.upconv4 = conv_bn_relu(num_ch_enc[4],              self.num_ch_dec[4])
        self.upconv3 = conv_bn_relu(self.num_ch_dec[4] + num_ch_enc[3], self.num_ch_dec[3])
        self.upconv2 = conv_bn_relu(self.num_ch_dec[3] + num_ch_enc[2], self.num_ch_dec[2])
        self.upconv1 = conv_bn_relu(self.num_ch_dec[2] + num_ch_enc[1], self.num_ch_dec[1])
        self.upconv0 = conv_bn_relu(self.num_ch_dec[1] + num_ch_enc[0], self.num_ch_dec[0])

        # Disparity heads at 4 scales (sigmoid output → [0, 1] inverse depth)
        # Scale 3: H/8 x W/8
        self.disp3 = nn.Sequential(
            nn.Conv2d(self.num_ch_dec[3], 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        # Scale 2: H/4 x W/4
        self.disp2 = nn.Sequential(
            nn.Conv2d(self.num_ch_dec[2], 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        # Scale 1: H/2 x W/2
        self.disp1 = nn.Sequential(
            nn.Conv2d(self.num_ch_dec[1], 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        # Scale 0: H x W (full resolution)
        self.disp0 = nn.Sequential(
            nn.Conv2d(self.num_ch_dec[0], 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, encoder_features):
        """
        Args:
            encoder_features: list of 5 tensors from ResNet18 encoder
                [feat0, feat1, feat2, feat3, feat4]
                shapes: [B,64,H/2,W/2], [B,64,H/4,W/4], [B,128,H/8,W/8],
                        [B,256,H/16,W/16], [B,512,H/32,W/32]

        Returns:
            disps: dict with keys 0,1,2,3 mapping to disparity maps
                   ALL upsampled to full [B, 1, H, W] resolution
        """
        f0, f1, f2, f3, f4 = encoder_features

        # --- Decode upward with skip connections ---
        # Each step: upsample 2x, concat skip, refine with conv

        x = F.interpolate(f4, scale_factor=2, mode='nearest')
        x = self.upconv4(x)                             # [B, 256, H/16, W/16]

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, f3], dim=1)
        x = self.upconv3(x)                             # [B, 128, H/8, W/8]
        disp3_raw = self.disp3(x)                       # [B, 1, H/8, W/8]

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, f2], dim=1)
        x = self.upconv2(x)                             # [B, 64, H/4, W/4]
        disp2_raw = self.disp2(x)                       # [B, 1, H/4, W/4]

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, f1], dim=1)
        x = self.upconv1(x)                             # [B, 32, H/2, W/2]
        disp1_raw = self.disp1(x)                       # [B, 1, H/2, W/2]

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, f0], dim=1)
        x = self.upconv0(x)                             # [B, 16, H, W]
        disp0_raw = self.disp0(x)                       # [B, 1, H, W]

        # --- Upsample ALL scales to full resolution before loss computation ---
        # This avoids texture-copy artifacts from computing loss on downsampled targets
        H, W = disp0_raw.shape[2], disp0_raw.shape[3]
        disps = {
            0: disp0_raw,
            1: F.interpolate(disp1_raw, size=(H, W), mode='bilinear', align_corners=True),
            2: F.interpolate(disp2_raw, size=(H, W), mode='bilinear', align_corners=True),
            3: F.interpolate(disp3_raw, size=(H, W), mode='bilinear', align_corners=True),
        }

        return disps


class DepthNet(nn.Module):
    """
    Full depth estimation network: ResNet18 encoder + multi-scale decoder.

    Takes a single RGB frame I_t and outputs 4 disparity maps at full resolution.

    Disparity → Depth conversion:
        depth = 1 / (min_disp + (max_disp - min_disp) * disparity)
    where min_disp = 1/max_depth, max_disp = 1/min_depth (e.g. 0.01 to 100 meters).
    """

    def __init__(self, pretrained=True, min_depth=0.1, max_depth=100.0):
        super().__init__()

        self.min_disp = 1.0 / max_depth
        self.max_disp = 1.0 / min_depth

        # --- Encoder: ResNet18 pretrained on ImageNet ---
        resnet = models.resnet18(weights=None)
        ckpt = torch.load(
            '/home/omb8654/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth',
            map_location='cpu'
        )
        resnet.load_state_dict(ckpt)

        # Extract feature stages individually to capture skip connections
        self.encoder = nn.ModuleDict({
            'layer0': nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu),  # stride 2
            'pool':   resnet.maxpool,                                          # stride 2
            'layer1': resnet.layer1,                                           # stride 1
            'layer2': resnet.layer2,                                           # stride 2
            'layer3': resnet.layer3,                                           # stride 2
            'layer4': resnet.layer4,                                           # stride 2
        })

        self.decoder = DepthDecoder()

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] — normalized input frame

        Returns:
            disps: dict {0: [B,1,H,W], 1: [B,1,H,W], 2: [B,1,H,W], 3: [B,1,H,W]}
            depths: dict {0: [B,1,H,W], ...} — converted depth maps
        """
        # --- Encode ---
        f0 = self.encoder['layer0'](x)         # [B, 64,  H/2,  W/2]
        p  = self.encoder['pool'](f0)           # [B, 64,  H/4,  W/4]
        f1 = self.encoder['layer1'](p)          # [B, 64,  H/4,  W/4]
        f2 = self.encoder['layer2'](f1)         # [B, 128, H/8,  W/8]
        f3 = self.encoder['layer3'](f2)         # [B, 256, H/16, W/16]
        f4 = self.encoder['layer4'](f3)         # [B, 512, H/32, W/32]

        # --- Decode ---
        disps = self.decoder([f0, f1, f2, f3, f4])

        # --- Convert disparity to metric depth ---
        depths = {}
        for s, d in disps.items():
            depths[s] = 1.0 / (self.min_disp + (self.max_disp - self.min_disp) * d)

        return disps, depths