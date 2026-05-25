import os
import numpy as np
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def parse_intrinsics(calib_path, cam_id='02'):
    """
    Parses the camera intrinsic matrix K from KITTI's calib_cam_to_cam.txt.

    KITTI calibration files store the 3x4 projection matrix P_rect_0X.
    We extract the 3x3 intrinsic block (first 3 columns) directly.

    Format in file:
        P_rect_02: fx 0 cx 0  0 fy cy 0  0 0 1 0

    Args:
        calib_path: str — path to calib_cam_to_cam.txt
        cam_id:     str — camera identifier, '02' for left color camera

    Returns:
        K: np.ndarray [3, 3] — intrinsic matrix
    """
    with open(calib_path, 'r') as f:
        lines = f.readlines()

    key = f'P_rect_{cam_id}:'
    for line in lines:
        if line.startswith(key):
            vals = list(map(float, line.strip().split()[1:]))
            # vals is a flat 12-element 3x4 projection matrix row-major
            P = np.array(vals).reshape(3, 4)
            K = P[:3, :3]   # intrinsic block: [3, 3]
            return K.astype(np.float32)

    raise ValueError(f"Key {key} not found in {calib_path}")


def scale_intrinsics(K, orig_w, orig_h, new_w, new_h):
    """
    Adjusts the intrinsic matrix K when the image is resized.

    Scaling factors applied to focal lengths and principal point:
        fx_new = fx * (new_w / orig_w)
        fy_new = fy * (new_h / orig_h)
        cx_new = cx * (new_w / orig_w)
        cy_new = cy * (new_h / orig_h)

    Args:
        K:      np.ndarray [3, 3]
        orig_w, orig_h: int — original image dimensions
        new_w,  new_h:  int — resized image dimensions

    Returns:
        K_scaled: np.ndarray [3, 3]
    """
    K_scaled = K.copy()
    K_scaled[0, 0] *= new_w / orig_w   # fx
    K_scaled[1, 1] *= new_h / orig_h   # fy
    K_scaled[0, 2] *= new_w / orig_w   # cx
    K_scaled[1, 2] *= new_h / orig_h   # cy
    return K_scaled


class KITTIDepthDataset(Dataset):
    """
    PyTorch Dataset for self-supervised monocular depth training on KITTI.

    Loads temporal triplets (I_t-1, I_t, I_t+1) from continuous video sequences.
    The middle frame I_t is the target; the flanking frames are the sources used
    for view synthesis supervision.

    Expected directory structure:
        root/
        └── split/                        # 'train' or 'test'
            └── <sequence_name>/
                ├── image_02/
                │   └── data/
                │       ├── 0000000000.png
                │       └── ...
                └── calib_cam_to_cam.txt

    Args:
        root_dir:   str  — path to kitti_data root
        split:      str  — 'train' or 'test'
        height:     int  — target image height (default 192)
        width:      int  — target image width  (default 640)
        augment:    bool — apply color jitter and horizontal flip during training
    """

    # ImageNet normalization — used because encoder is pretrained on ImageNet
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, root_dir, split='train', height=192, width=640, augment=True):
        self.root   = Path(root_dir) / split
        self.split  = split
        self.H      = height
        self.W      = width
        self.augment = augment and (split == 'train')

        self.to_tensor   = T.ToTensor()
        self.normalize   = T.Normalize(mean=self.MEAN, std=self.STD)
        self.color_jitter = T.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )

        # --- Index all valid triplets across all sequences ---
        # A valid triplet requires frames at indices [i-1, i, i+1] to exist
        self.triplets = []   # list of (frame_paths[3], calib_path)

        for seq_dir in sorted(self.root.iterdir()):
            if not seq_dir.is_dir():
                continue

            img_dir   = seq_dir / 'image_02' / 'data'
            calib_path = seq_dir / 'calib_cam_to_cam.txt'

            if not img_dir.exists() or not calib_path.exists():
                continue

            frames = sorted(img_dir.glob('*.png'))

            # Skip first and last — they cannot form complete triplets
            for i in range(1, len(frames) - 1):
                self.triplets.append((
                    [frames[i - 1], frames[i], frames[i + 1]],
                    calib_path
                ))

        if len(self.triplets) == 0:
            raise RuntimeError(
                f"No valid triplets found under {self.root}. "
                "Check directory structure."
            )

        # Cache one intrinsic per sequence to avoid re-parsing the same file
        # repeatedly across adjacent triplets
        self._calib_cache = {}

    def __len__(self):
        return len(self.triplets)

    def _load_image(self, path):
        """Loads a PNG as a PIL Image in RGB mode."""
        return Image.open(str(path)).convert('RGB')

    def _get_intrinsics(self, calib_path):
        """Returns cached and scaled intrinsics for a given calibration file."""
        key = str(calib_path)
        if key not in self._calib_cache:
            # Parse at original resolution first
            K_orig = parse_intrinsics(key)

            # We need original image size to scale correctly
            # KITTI raw images are 1242 x 375 (approximately)
            # Read from the first frame instead of hardcoding
            orig_w, orig_h = 1242, 375   # KITTI standard; override if needed

            K_scaled = scale_intrinsics(K_orig, orig_w, orig_h, self.W, self.H)
            self._calib_cache[key] = K_scaled

        return self._calib_cache[key]

    def _augment_images(self, images):
        """
        Applies consistent augmentation across all three frames in a triplet.

        Color jitter is applied independently per frame to simulate
        auto-exposure variation between frames (realistic for real cameras).
        Horizontal flip is applied consistently to avoid breaking the geometry.

        Args:
            images: list of 3 PIL Images

        Returns:
            images: list of 3 augmented PIL Images
            flipped: bool
        """
        # Random horizontal flip — applied identically to all frames
        flipped = False
        if torch.rand(1).item() > 0.5:
            images = [TF.hflip(img) for img in images]
            flipped = True

        # Color jitter applied independently per frame
        images = [self.color_jitter(img) for img in images]

        return images, flipped

    def __getitem__(self, idx):
        """
        Returns a single training sample as a dict:

            'target':        [3, H, W]  — normalized I_t
            'source_prev':   [3, H, W]  — normalized I_t-1
            'source_next':   [3, H, W]  — normalized I_t+1
            'K':             [3, 3]     — scaled intrinsic matrix
            'K_inv':         [3, 3]     — inverse intrinsic matrix
            'target_raw':    [3, H, W]  — unnormalized I_t (for smoothness loss)
        """
        frame_paths, calib_path = self.triplets[idx]

        # --- Load raw PIL images ---
        images = [self._load_image(p) for p in frame_paths]
        # images[0]: I_t-1, images[1]: I_t, images[2]: I_t+1

        # Actual image dimensions before resize (for intrinsic scaling)
        orig_w, orig_h = images[1].size   # PIL size: (W, H)

        # --- Resize all frames to training resolution ---
        resize = T.Resize((self.H, self.W), interpolation=T.InterpolationMode.LANCZOS)
        images = [resize(img) for img in images]

        # --- Augmentation ---
        flipped = False
        if self.augment:
            images, flipped = self._augment_images(images)

        # --- Convert to tensors and normalize ---
        tensors = [self.to_tensor(img) for img in images]    # list of [3, H, W] in [0,1]

        # Keep unnormalized target for edge-aware smoothness loss
        target_raw = tensors[1].clone()

        # Normalize all frames with ImageNet stats
        tensors = [self.normalize(t) for t in tensors]

        # --- Intrinsics ---
        K = self._get_intrinsics(calib_path)
        # Re-scale from actual orig dims (not hardcoded) if they differ
        if (orig_w, orig_h) != (1242, 375):
            K = scale_intrinsics(
                parse_intrinsics(str(calib_path)),
                orig_w, orig_h, self.W, self.H
            )

        # If horizontally flipped, adjust cx: cx_new = W - cx
        if flipped:
            K = K.copy()
            K[0, 2] = self.W - K[0, 2]

        K_tensor     = torch.from_numpy(K)
        K_inv_tensor = torch.from_numpy(np.linalg.inv(K))

        return {
            'target':      tensors[1],      # [3, H, W]
            'source_prev': tensors[0],      # [3, H, W]
            'source_next': tensors[2],      # [3, H, W]
            'K':           K_tensor,        # [3, 3]
            'K_inv':       K_inv_tensor,    # [3, 3]
            'target_raw':  target_raw,      # [3, H, W] unnormalized
        }


def build_dataloader(root_dir, split, height, width, batch_size,
                     num_workers=4, augment=True):
    """
    Convenience constructor for train/val dataloaders.

    Args:
        root_dir:    str
        split:       'train' or 'test'
        height:      int
        width:       int
        batch_size:  int
        num_workers: int
        augment:     bool

    Returns:
        DataLoader
    """
    dataset = KITTIDepthDataset(
        root_dir=root_dir,
        split=split,
        height=height,
        width=width,
        augment=augment
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True    # ensures fixed batch size for BackprojectDepth buffers
    )