import torch
import torch.nn as nn


class PoseNet(nn.Module):
    """
    Pose Estimation Network.

    Accepts two concatenated RGB frames (source + target) and regresses a
    6-DoF relative pose vector: [tx, ty, tz, rx, ry, rz].

    Rotation is represented as axis-angle (Rodrigues) rather than Euler angles
    to avoid gimbal lock. The magnitude of [rx, ry, rz] encodes the rotation
    angle in radians; the direction encodes the rotation axis.

    The final 6-vector is converted to a 4x4 SE(3) transformation matrix
    via rodriguez_to_matrix() before being passed to the warp engine.

    Input:
        frames: [B, 6, H, W]  — two concatenated RGB frames (3+3 channels)

    Output:
        T:      [B, 4, 4]     — rigid transformation matrix (source -> target)
        axisangle: [B, 1, 3]  — raw rotation output (before normalization)
        translation: [B, 1, 3]
    """

    def __init__(self, num_input_frames=2, num_ch=256):
        super().__init__()

        in_channels = 3 * num_input_frames  # 6 for a pair

        self.encoder = nn.Sequential(
            # Each conv halves spatial dims via stride=2
            self._conv_block(in_channels, 16,  stride=2),  # H/2
            self._conv_block(16,          32,  stride=2),  # H/4
            self._conv_block(32,          64,  stride=2),  # H/8
            self._conv_block(64,          128, stride=2),  # H/16
            self._conv_block(128,         256, stride=2),  # H/32
            self._conv_block(256,         256, stride=2),  # H/64
        )

        # Global average pool collapses spatial dims entirely
        self.gap = nn.AdaptiveAvgPool2d(1)   # [B, 256, 1, 1]

        # Separate heads for translation and rotation
        # Scale factor 0.01: keeps initial pose predictions near identity,
        # critical for stable early training
        self.translation_head = nn.Sequential(
            nn.Linear(256, 3)
        )
        self.rotation_head = nn.Sequential(
            nn.Linear(256, 3)
        )

        # Initialize output heads to produce near-zero pose at the start
        for m in [self.translation_head, self.rotation_head]:
            nn.init.normal_(m[0].weight, mean=0.0, std=1e-4)
            nn.init.zeros_(m[0].bias)

    @staticmethod
    def _conv_block(in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, frame_t, frame_s):
        """
        Args:
            frame_t: [B, 3, H, W] — target frame I_t
            frame_s: [B, 3, H, W] — source frame (I_t-1 or I_t+1)

        Returns:
            T:           [B, 4, 4]  — SE(3) transform from target to source
            axisangle:   [B, 1, 3]  — raw axis-angle rotation vector
            translation: [B, 1, 3]  — raw translation vector
        """
        # Concatenate along channel dim: [B, 6, H, W]
        x = torch.cat([frame_t, frame_s], dim=1)

        # Encode: [B, 256, H/64, W/64]
        x = self.encoder(x)

        # Global average pool: [B, 256, 1, 1] -> [B, 256]
        x = self.gap(x).flatten(1)

        # Regress 6-DoF
        # Scale by 0.01 to keep initial predictions near identity transform
        translation = self.translation_head(x).unsqueeze(1) * 0.01  # [B, 1, 3]
        axisangle   = self.rotation_head(x).unsqueeze(1)   * 0.01   # [B, 1, 3]

        # Convert axis-angle to 4x4 SE(3) matrix
        T = axisangle_to_matrix(axisangle, translation)              # [B, 4, 4]

        return T, axisangle, translation


def axisangle_to_matrix(axisangle, translation):
    """
    Converts axis-angle rotation + translation vector to a 4x4 SE(3) matrix.

    Uses Rodrigues' rotation formula:
        R = I + sin(theta)*K + (1 - cos(theta))*K^2
    where K is the skew-symmetric matrix of the unit axis and theta = ||axisangle||.

    Args:
        axisangle:   [B, 1, 3]
        translation: [B, 1, 3]

    Returns:
        T: [B, 4, 4]
    """
    B = axisangle.shape[0]
    device = axisangle.device
    dtype  = axisangle.dtype

    # Squeeze to [B, 3]
    aa = axisangle.squeeze(1)    # [B, 3]
    t  = translation.squeeze(1)  # [B, 3]

    # Rotation angle (L2 norm of axis-angle vector): [B]
    theta = aa.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)  # [B, 1]

    # Unit axis: [B, 3]
    axis = aa / theta  # [B, 3]

    # Skew-symmetric matrix K from unit axis [kx, ky, kz]:
    # K = [[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]]
    kx = axis[:, 0]
    ky = axis[:, 1]
    kz = axis[:, 2]
    zeros = torch.zeros(B, device=device, dtype=dtype)

    # K: [B, 3, 3]
    K = torch.stack([
        torch.stack([zeros,  -kz,    ky],  dim=1),
        torch.stack([kz,     zeros,  -kx], dim=1),
        torch.stack([-ky,    kx,     zeros], dim=1),
    ], dim=1)

    # Rodrigues formula
    theta = theta.unsqueeze(2)  # [B, 1, 1] for broadcasting
    I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)

    R = (I
         + torch.sin(theta) * K
         + (1 - torch.cos(theta)) * torch.bmm(K, K))  # [B, 3, 3]

    # Assemble 4x4 SE(3) matrix
    T = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()
    T[:, :3, :3] = R
    T[:, :3,  3] = t

    return T  # [B, 4, 4]