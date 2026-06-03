import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIM(nn.Module):
    """
    Differentiable Structural Similarity Index (SSIM) for photometric loss.

    Computes a local SSIM map using average-pooling as an approximation
    of Gaussian filtering. Returns a per-pixel dissimilarity map (1 - SSIM) / 2
    normalized to [0, 1].

    Args:
        window_size: int — size of the pooling kernel (default 3, not 11,
                     to keep gradients sharp and computation cheap)

    Input:
        x, y: [B, 3, H, W] — two image tensors in [0, 1]

    Output:
        ssim_map: [B, 3, H, W] — per-pixel dissimilarity in [0, 1]
    """

    def __init__(self, window_size=3):
        super().__init__()
        self.window_size = window_size
        self.pad = window_size // 2
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        # Reflect-pad to preserve spatial dimensions after pooling
        x = F.pad(x, [self.pad] * 4, mode='reflect')
        y = F.pad(y, [self.pad] * 4, mode='reflect')

        # Local means via average pooling
        # mu_x, mu_y: [B, 3, H, W]
        mu_x = F.avg_pool2d(x, self.window_size, stride=1, padding=0)
        mu_y = F.avg_pool2d(y, self.window_size, stride=1, padding=0)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy   = mu_x * mu_y

        # Local variances and covariance
        sigma_x_sq  = F.avg_pool2d(x ** 2, self.window_size, stride=1, padding=0) - mu_x_sq
        sigma_y_sq  = F.avg_pool2d(y ** 2, self.window_size, stride=1, padding=0) - mu_y_sq
        sigma_xy    = F.avg_pool2d(x * y,  self.window_size, stride=1, padding=0) - mu_xy

        # SSIM numerator and denominator
        numerator   = (2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)
        denominator = (mu_x_sq + mu_y_sq + self.C1) * (sigma_x_sq + sigma_y_sq + self.C2)

        ssim_map = numerator / denominator.clamp(min=1e-8)  # [B, 3, H, W]

        # Convert similarity [−1,1] → dissimilarity [0,1]
        return ((1 - ssim_map) / 2).clamp(0, 1)


class PhotometricLoss(nn.Module):
    """
    Per-pixel photometric reconstruction loss.

    Combines:
        - 85% SSIM dissimilarity
        - 15% L1 absolute difference

    This blend was established in Monodepth2 (Godard et al., 2019) and is
    now standard. Pure L1 is too sensitive to lighting; pure SSIM is
    insensitive to absolute brightness shifts.

    Input:
        pred:   [B, 3, H, W] — synthesized (warped) frame
        target: [B, 3, H, W] — ground-truth target frame I_t

    Output:
        loss_map: [B, 1, H, W] — per-pixel photometric error (mean over channels)
    """

    def __init__(self, ssim_weight=0.85, l1_weight=0.15):
        super().__init__()
        assert abs(ssim_weight + l1_weight - 1.0) < 1e-6, \
            "SSIM and L1 weights must sum to 1.0"
        self.ssim = SSIM()
        self.w_ssim = ssim_weight
        self.w_l1   = l1_weight

    def forward(self, pred, target):
        # SSIM dissimilarity: [B, 3, H, W]
        ssim_map = self.ssim(pred, target)

        # L1 map: [B, 3, H, W]
        l1_map = (pred - target).abs()

        # Weighted combination, averaged over RGB channels → [B, 1, H, W]
        loss_map = (self.w_ssim * ssim_map + self.w_l1 * l1_map).mean(dim=1, keepdim=True)

        return loss_map


class AutoMask(nn.Module):
    """
    Auto-masking to suppress two failure modes:

    1. Dynamic Objects: independently moving objects (cars, pedestrians) violate
       the static-world assumption. Their pixels produce lower warped error than
       the unwarped baseline ONLY when the object moves consistently with camera
       motion — but for truly independent motion, warped error > unwarped error.

    2. Static Camera: when the vehicle is stationary, the synthesized frame is
       identical to the source, so the photometric loss drops to near-zero regardless
       of the depth prediction. This allows the pose network weights to degrade.

    Solution (Monodepth2):
        Compute pe_unwarped = photometric_loss(I_t+1, I_t)  — no warp at all.
        Compute pe_warped   = photometric_loss(I_t_prime, I_t)  — with warp.

        Keep pixel if pe_warped < pe_unwarped.
        Discard (mask=0) otherwise.

    The mask is computed per-pixel, per-source-frame, then combined across
    source frames by taking the minimum error across frames before masking.

    Input:
        pe_warped:   list of [B, 1, H, W] — warped photometric errors
                     (one per source frame, i.e. [I_t-1→I_t, I_t+1→I_t])
        pe_unwarped: list of [B, 1, H, W] — unwarped photometric errors
                     (same pairing, no warp applied)

    Output:
        loss:        scalar — mean photometric loss over valid pixels
        mask:        [B, 1, H, W] — binary validity mask (1=valid, 0=ignored)
    """

    def __init__(self):
        super().__init__()

    def forward(self, pe_warped, pe_unwarped):
        """
        Args:
            pe_warped:   list of K tensors, each [B, 1, H, W]
            pe_unwarped: list of K tensors, each [B, 1, H, W]

        Returns:
            loss: scalar
            mask: [B, 1, H, W]
        """
        # Stack along a new 'source' dimension → [B, 1, H, W, K]
        # Then take the per-pixel minimum across source frames.
        # This follows Monodepth2: use the best-reconstructed source at each pixel.

        # pe_warped_stack:   [B, K, H, W]
        pe_warped_stack   = torch.cat(pe_warped,   dim=1)
        pe_unwarped_stack = torch.cat(pe_unwarped, dim=1)

        # Add a small uniform noise to break ties at perfectly static regions
        # (prevents degenerate all-zero masks on completely textureless frames)
        pe_unwarped_stack = pe_unwarped_stack + torch.randn_like(pe_unwarped_stack) * 1e-5

        # Per-pixel minimum warped error across source frames: [B, 1, H, W]
        min_pe_warped,   _ = torch.min(pe_warped_stack,   dim=1, keepdim=True)
        min_pe_unwarped, _ = torch.min(pe_unwarped_stack, dim=1, keepdim=True)

        # Binary mask: keep pixels where the warp actually helped
        # mask: [B, 1, H, W], dtype=bool
        mask = (min_pe_warped < min_pe_unwarped).float()

        # Mean loss over valid pixels only
        # Clamp denominator to avoid NaN when mask is all zeros (degenerate batch)
        valid_pixels = mask.sum().clamp(min=1.0)
        loss = (min_pe_warped * mask).sum() / valid_pixels

        return loss, mask


class SmoothnessLoss(nn.Module):
    """
    Edge-aware disparity smoothness loss.

    Penalizes large disparity gradients, but downweights the penalty
    exponentially where image gradients are large (i.e. at true object edges).
    This preserves depth discontinuities at object boundaries while
    aggressively regularizing flat textureless regions (sky, walls).

    Formula:
        L_smooth = |∂d/∂x| * exp(-|∂I/∂x|) + |∂d/∂y| * exp(-|∂I/∂y|)

    where d is the mean-normalized disparity and I is the RGB image.

    Mean-normalizing disparity removes the scale ambiguity that would otherwise
    cause the network to drive all disparities toward zero to minimize this loss.

    Input:
        disp:  [B, 1, H, W] — predicted disparity map
        image: [B, 3, H, W] — corresponding RGB frame (used for edge detection)

    Output:
        loss: scalar — mean edge-aware smoothness penalty
    """

    def __init__(self):
        super().__init__()

    def forward(self, disp, image):
        # Mean-normalize disparity to remove scale dependency
        # mean_disp: [B, 1, 1, 1] via mean over H, W
        mean_disp = disp.mean(dim=[2, 3], keepdim=True).clamp(min=1e-7)
        disp_norm = disp / mean_disp   # [B, 1, H, W]

        # Disparity gradients: finite differences along x and y
        # grad_disp_x: [B, 1, H, W-1], grad_disp_y: [B, 1, H-1, W]
        grad_disp_x = (disp_norm[:, :, :, 1:] - disp_norm[:, :, :, :-1]).abs()
        grad_disp_y = (disp_norm[:, :, 1:, :] - disp_norm[:, :, :-1, :]).abs()

        # Image gradients: mean over RGB channels first → [B, 1, H, W]
        image_gray = image.mean(dim=1, keepdim=True)
        grad_img_x = (image_gray[:, :, :, 1:] - image_gray[:, :, :, :-1]).abs()
        grad_img_y = (image_gray[:, :, 1:, :] - image_gray[:, :, :-1, :]).abs()

        # Edge-aware weights: decay smoothness penalty at image edges
        weight_x = torch.exp(-grad_img_x)  # [B, 1, H, W-1]
        weight_y = torch.exp(-grad_img_y)  # [B, 1, H-1, W]

        # Weighted smoothness penalty
        smooth_x = (grad_disp_x * weight_x).mean()
        smooth_y = (grad_disp_y * weight_y).mean()

        return smooth_x + smooth_y


class SfMLoss(nn.Module):
    """
    Top-level loss module combining all three components.

    Total loss per scale s:
        L_s = L_photometric_s + lambda_smooth * L_smooth_s / (2^s)

    The smoothness weight is divided by 2^s because lower-resolution depth maps
    already have less spatial variation and need proportionally less regularization.

    Args:
        lambda_smooth: float — weight on smoothness loss (default 1e-3,
                       standard for KITTI-scale scenes)
        num_scales:    int   — number of decoder output scales (default 4)
    """

    def __init__(self, lambda_smooth=1e-3, num_scales=4):
        super().__init__()
        self.photometric = PhotometricLoss()
        self.automask    = AutoMask()
        self.smoothness  = SmoothnessLoss()
        self.lambda_smooth = lambda_smooth
        self.num_scales    = num_scales

    def forward(self, target, warped_frames, source_frames, disps):
        total_loss   = 0.0
        total_photo  = 0.0
        total_smooth = 0.0

        for s in range(self.num_scales):
            disp = disps[s]
            
            # Fetch warped frames specifically computed from scale s depth
            warped_frames_s = warped_frames[s]

            pe_warped   = []
            pe_unwarped = []

            for warped, source in zip(warped_frames_s, source_frames):
                pe_w = self.photometric(warped, target)
                pe_u = self.photometric(source, target)

                pe_warped.append(pe_w)
                pe_unwarped.append(pe_u)

            photo_loss, mask = self.automask(pe_warped, pe_unwarped)
            smooth_loss = self.smoothness(disp, target) / (2 ** s)

            scale_loss   = photo_loss + self.lambda_smooth * smooth_loss
            total_loss   = total_loss + scale_loss
            total_photo  = total_photo  + photo_loss.item()
            total_smooth = total_smooth + smooth_loss.item()

        total_loss = total_loss / self.num_scales

        loss_breakdown = {
            'photo':  total_photo  / self.num_scales,
            'smooth': total_smooth / self.num_scales,
            'mask':   mask,
        }

        return total_loss, loss_breakdown