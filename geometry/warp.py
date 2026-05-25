import torch
import torch.nn.functional as F


class BackprojectDepth(torch.nn.Module):
    """
    Unprojects a 2D pixel grid into 3D camera coordinates using predicted depth
    and the inverse intrinsic matrix K_inv.

    For each pixel (u, v) with depth d:
        P_cam = d * K_inv * [u, v, 1]^T

    Input:
        depth:  [B, 1, H, W]  — predicted inverse depth (disparity), converted to depth
        K_inv:  [B, 3, 3]     — inverse of camera intrinsic matrix

    Output:
        cam_points: [B, 4, H*W] — homogeneous 3D points in camera space
    """

    def __init__(self, batch_size, height, width):
        super().__init__()
        self.B = batch_size
        self.H = height
        self.W = width

        # Build the static pixel coordinate grid once, register as buffer
        # so it moves with .to(device) automatically
        meshgrid = torch.meshgrid(
            torch.arange(width, dtype=torch.float32),
            torch.arange(height, dtype=torch.float32),
            indexing='xy'
        )
        # u_coords: [H, W], v_coords: [H, W]
        u_coords = meshgrid[0].reshape(1, -1)   # [1, H*W]
        v_coords = meshgrid[1].reshape(1, -1)   # [1, H*W]
        ones     = torch.ones_like(u_coords)     # [1, H*W]

        # pixel_coords: [1, 3, H*W] — homogeneous pixel coordinates
        pixel_coords = torch.stack([u_coords, v_coords, ones], dim=1)
        # Expand to [B, 3, H*W]
        self.register_buffer(
            'pixel_coords',
            pixel_coords.expand(batch_size, -1, -1)
        )

    def forward(self, depth, K_inv):
        B = depth.shape[0]
        
        # Flatten depth: [B, 1, H*W]
        depth_flat = depth.view(B, 1, -1)

        # Use the registered buffer but slice to actual batch size
        pixel_coords = self.pixel_coords[:B]  # [B, 3, H*W]

        # Unproject
        cam_points = torch.bmm(K_inv, pixel_coords)  # [B, 3, H*W]
        cam_points = cam_points * depth_flat          # [B, 3, H*W]

        ones = torch.ones(B, 1, self.H * self.W,
                        dtype=cam_points.dtype,
                        device=cam_points.device)
        cam_points = torch.cat([cam_points, ones], dim=1)  # [B, 4, H*W]

        return cam_points


class Project3DPoints(torch.nn.Module):
    """
    Projects 3D camera-space points into 2D pixel coordinates of the target frame
    using the intrinsic matrix K and a 6-DoF rigid transformation T.

    Pipeline:
        1. Apply T (4x4 SE3 matrix) to rotate/translate points into target frame
        2. Project via K: x = K @ P[:3] / P[2]
        3. Normalize to [-1, 1] for use with grid_sample

    Input:
        cam_points: [B, 4, H*W] — homogeneous 3D points in source camera space
        K:          [B, 3, 3]   — camera intrinsic matrix
        T:          [B, 4, 4]   — rigid body transformation (source -> target)

    Output:
        pix_coords: [B, H, W, 2] — normalized 2D sampling grid for grid_sample
        depth_proj: [B, 1, H, W] — projected depth (Z) in target frame
    """

    def __init__(self, batch_size, height, width, eps=1e-7):
        super().__init__()
        self.B   = batch_size
        self.H   = height
        self.W   = width
        self.eps = eps  # guards against division by near-zero depth

    def forward(self, cam_points, K, T):
        B = cam_points.shape[0]
        
        P = torch.bmm(T, cam_points)
        X = P[:, 0]
        Y = P[:, 1]
        Z = P[:, 2].clamp(min=self.eps)

        cam_points_norm = torch.stack([X/Z, Y/Z, torch.ones_like(X)], dim=1)
        pix_homogeneous = torch.bmm(K, cam_points_norm)

        u = pix_homogeneous[:, 0]
        v = pix_homogeneous[:, 1]

        u_norm = (u / (self.W - 1)) * 2.0 - 1.0
        v_norm = (v / (self.H - 1)) * 2.0 - 1.0

        pix_coords = torch.stack([u_norm, v_norm], dim=2)
        pix_coords = pix_coords.view(B, self.H, self.W, 2)

        depth_proj = Z.view(B, 1, self.H, self.W)

        return pix_coords, depth_proj


def inverse_warp(source_frame, depth, K, K_inv, T,
                 backproject: BackprojectDepth,
                 project: Project3DPoints):
    """
    Full differentiable inverse warp: synthesizes the source view from the
    target frame using predicted depth and relative pose.

    Args:
        source_frame: [B, 3, H, W]  — the frame to sample FROM (e.g. I_t+1)
        depth:        [B, 1, H, W]  — predicted depth of the TARGET frame (I_t)
        K:            [B, 3, 3]     — camera intrinsics
        K_inv:        [B, 3, 3]     — inverse intrinsics
        T:            [B, 4, 4]     — rigid transform from target to source frame
        backproject:  BackprojectDepth instance
        project:      Project3DPoints instance

    Returns:
        warped:       [B, 3, H, W]  — synthesized I_t_prime
        pix_coords:   [B, H, W, 2]  — sampling grid (useful for mask computation)
        depth_proj:   [B, 1, H, W]  — projected depth
    """
    # Step 1: Lift I_t depth map to 3D
    cam_points = backproject(depth, K_inv)          # [B, 4, H*W]

    # Step 2: Transform to source frame and project to 2D
    pix_coords, depth_proj = project(cam_points, K, T)  # [B, H, W, 2], [B, 1, H, W]

    # Step 3: Sample source frame at projected coordinates
    # align_corners=True: pixel centers are at -1 and +1 (consistent with normalize above)
    # padding_mode='zeros': out-of-bounds pixels become black (masked out later)
    warped = F.grid_sample(
        source_frame,
        pix_coords,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )  # [B, 3, H, W]

    return warped, pix_coords, depth_proj