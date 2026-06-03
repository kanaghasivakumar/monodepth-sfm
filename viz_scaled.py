import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from networks.depth_net import DepthNet
from train import CONFIG
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load('checkpoints/best_checkpoint.pth', map_location=device)
depth_net = DepthNet(
    pretrained=False,
    min_depth=CONFIG['min_depth'],
    max_depth=CONFIG['max_depth']
).to(device)
depth_net.load_state_dict(ckpt['depth_net'])
depth_net.eval()

normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor  = T.ToTensor()

# Median scale factors observed during evaluation (~33x average)
MEDIAN_SCALE = 33.0

test_images = [
    'kitti_data/test/2011_09_26_drive_0009_sync/image_02/data/0000000050.png',
    'kitti_data/test/2011_09_26_drive_0009_sync/image_02/data/0000000100.png',
    'kitti_data/test/2011_09_26_drive_0009_sync/image_02/data/0000000150.png',
    'kitti_data/test/2011_09_26_drive_0009_sync/image_02/data/0000000200.png',
    'kitti_data/test/2011_09_26_drive_0057_sync/image_02/data/0000000050.png',
    'kitti_data/test/2011_09_26_drive_0057_sync/image_02/data/0000000150.png',
]

os.makedirs('viz/scaled', exist_ok=True)

for i, path in enumerate(test_images):
    if not os.path.exists(path):
        print(f'Missing: {path}')
        continue

    img    = Image.open(path).convert('RGB')
    img_r  = img.resize((640, 192))
    x      = normalize(to_tensor(img_r)).unsqueeze(0).to(device)

    with torch.no_grad():
        _, depths = depth_net(x)
        d = depths[0].squeeze().cpu().numpy()

    # Apply median scale to get approximate metric depth
    d_scaled = d * MEDIAN_SCALE
    d_scaled = np.clip(d_scaled, CONFIG['min_depth'], CONFIG['max_depth'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].imshow(img_r)
    axes[0].set_title('Input RGB', fontsize=13)
    axes[0].axis('off')

    im = axes[1].imshow(
        d_scaled, cmap='plasma',
        vmin=0, vmax=50
    )
    axes[1].set_title('Predicted Depth (meters)', fontsize=13)
    axes[1].axis('off')
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Depth (m)', fontsize=11)

    plt.tight_layout()
    out = f'viz/scaled/depth_scaled_{i:02d}.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {out} — scaled range: {d_scaled.min():.1f}m to {d_scaled.max():.1f}m')

print('Done.')