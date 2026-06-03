import os
import urllib.request
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

from train import CONFIG
from networks.depth_net import DepthNet

# ---------------------------------------------------------------------------
# Eigen Split File Download
# ---------------------------------------------------------------------------

def get_eigen_split(utils_dir="utils"):
    """Downloads the standard Eigen test split text file if missing."""
    os.makedirs(utils_dir, exist_ok=True)
    split_path = os.path.join(utils_dir, 'eigen_test_files.txt')
    
    if not os.path.exists(split_path):
        print(f"Downloading Eigen test split to {split_path}...")
        url = "https://raw.githubusercontent.com/nianticlabs/monodepth2/master/splits/eigen/test_files.txt"
        urllib.request.urlretrieve(url, split_path)
    
    return split_path

# ---------------------------------------------------------------------------
# KITTI Evaluation Metrics
# ---------------------------------------------------------------------------

def compute_errors(gt, pred):
    """Computes standard Eigen split depth estimation metrics."""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

# ---------------------------------------------------------------------------
# Velodyne to Depth Map Projection
# ---------------------------------------------------------------------------

def read_calib_file(filepath):
    """Reads KITTI calibration file into a dictionary of numpy arrays."""
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line or line == '':
                continue
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data

def generate_depth_map(calib_dir, velo_filename, im_shape):
    """
    Projects Velodyne sparse point cloud into the camera image plane.
    Handles both KITTI Raw (R, T) and KITTI Odometry (Tr) calib formats.
    """
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))

    P_rect = cam2cam['P_rect_02'].reshape(3, 4)
    
    R_rect = np.eye(4)
    R_rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    
    # Construct the Velodyne to Camera rigid transformation
    Tr_velo_to_cam = np.eye(4)
    if 'Tr' in velo2cam:
        Tr_velo_to_cam[:3, :4] = velo2cam['Tr'].reshape(3, 4)
    elif 'R' in velo2cam and 'T' in velo2cam:
        Tr_velo_to_cam[:3, :3] = velo2cam['R'].reshape(3, 3)
        Tr_velo_to_cam[:3, 3] = velo2cam['T']
    else:
        raise KeyError(f"Could not find 'Tr' or 'R'/'T' keys in {os.path.join(calib_dir, 'calib_velo_to_cam.txt')}")

    scan = np.fromfile(velo_filename, dtype=np.float32).reshape(-1, 4)
    pts_3d = scan[:, :3]
    
    pts_3d = pts_3d[pts_3d[:, 0] >= 0, :]
    pts_3d_homo = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
    
    pts_2d_homo = P_rect @ R_rect @ Tr_velo_to_cam @ pts_3d_homo.T
    pts_2d = (pts_2d_homo[:2, :] / pts_2d_homo[2, :]).T
    depths = pts_2d_homo[2, :]

    h, w = im_shape
    pts_2d = np.int32(np.round(pts_2d))

    val_inds = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & \
               (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
    
    pts_2d = pts_2d[val_inds, :]
    depths = depths[val_inds]

    depth_map = np.zeros((h, w))
    
    for i in range(len(pts_2d)):
        u, v = pts_2d[i, 0], pts_2d[i, 1]
        z = depths[i]
        if depth_map[v, u] == 0 or z < depth_map[v, u]:
            depth_map[v, u] = z
            
    return depth_map

# ---------------------------------------------------------------------------
# Evaluation Pipeline
# ---------------------------------------------------------------------------

def evaluate(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on: {device}")

    print(f"Loading checkpoint: {config['resume_ckpt']}")
    ckpt = torch.load(config['resume_ckpt'], map_location=device)
    
    depth_net = DepthNet(
        pretrained=False, 
        min_depth=config['min_depth'],
        max_depth=config['max_depth']
    ).to(device)
    
    depth_net.load_state_dict(ckpt['depth_net'])
    depth_net.train()

    to_tensor = T.ToTensor()

    split_file = get_eigen_split()
    test_files = [] 
    
    with open(split_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
                
            folder_path = parts[0]
            frame_id = parts[1]
            sequence_name = folder_path.split('/')[1]
            
            base_dir = os.path.join(config['root_dir'], 'test', sequence_name)
            
            rgb_path = os.path.join(base_dir, 'image_02', 'data', f"{int(frame_id):010d}.png")
            velo_path = os.path.join(base_dir, 'velodyne_points', 'data', f"{int(frame_id):010d}.bin")
            calib_dir = base_dir 
            
            if os.path.exists(rgb_path) and os.path.exists(velo_path):
                test_files.append((rgb_path, velo_path, calib_dir))

    if not test_files:
        raise RuntimeError("No test files found. Check your KITTI test directory structure.")

    errors = []
    print(f"Beginning evaluation on {len(test_files)} frames...")
    
    with torch.no_grad():
        for rgb_path, velo_path, calib_dir in tqdm(test_files):
            img = Image.open(rgb_path).convert('RGB')
            orig_w, orig_h = img.size
            
            img_resized = img.resize((config['width'], config['height']), Image.LANCZOS)
            input_tensor = to_tensor(img_resized).unsqueeze(0).to(device)
            
            _, depths = depth_net(input_tensor)
            pred_depth = depths[0]
            
            pred_depth_resized = F.interpolate(
                pred_depth, size=(orig_h, orig_w), mode='bilinear', align_corners=False
            ).squeeze().cpu().numpy()
            
            gt_depth = generate_depth_map(calib_dir, velo_path, (orig_h, orig_w))

            gt_nonzero = gt_depth[gt_depth > 0]
            if len(gt_nonzero) > 0 and len(errors) < 3:  # only print first 3 frames
                print(f"GT depth — min: {gt_nonzero.min():.2f}, max: {gt_nonzero.max():.2f}, "
                    f"median: {np.median(gt_nonzero):.2f}, points: {len(gt_nonzero)}")
                print(f"Pred depth — min: {pred_depth_resized.min():.4f}, "
                    f"max: {pred_depth_resized.max():.4f}, "
                    f"median: {np.median(pred_depth_resized):.4f}")
                print(f"Scale ratio: {np.median(gt_nonzero) / np.median(pred_depth_resized):.2f}")
            
            crop = np.array([0.40810811 * orig_h, 0.99189189 * orig_h,
                             0.03594771 * orig_w, 0.96405229 * orig_w]).astype(np.int32)
            
            gt_mask = (gt_depth > config['min_depth']) & (gt_depth < config['max_depth'])
            crop_mask = np.zeros_like(gt_mask)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            
            valid_mask = gt_mask & crop_mask
            
            pred_valid = pred_depth_resized[valid_mask]
            gt_valid = gt_depth[valid_mask]
            
            if len(gt_valid) > 0:
                ratio = np.median(gt_valid) / np.median(pred_valid)
                ratio = np.clip(ratio, 0.1, 100.0)
                pred_valid *= ratio

                pred_valid = np.clip(pred_valid, config['min_depth'], config['max_depth'])
                errors.append(compute_errors(gt_valid, pred_valid))

    mean_errors = np.array(errors).mean(0)
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = mean_errors

    print("\n--- Final Eigen Split Metrics ---")
    print(f"AbsRel:  {abs_rel:.4f}")
    print(f"SqRel:   {sq_rel:.4f}")
    print(f"RMSE:    {rmse:.4f}")
    print(f"RMSElog: {rmse_log:.4f}")
    print(f"δ < 1.25:   {a1:.4f}")
    print(f"δ < 1.25²:  {a2:.4f}")
    print(f"δ < 1.25³:  {a3:.4f}")

if __name__ == '__main__':
    evaluate(CONFIG)