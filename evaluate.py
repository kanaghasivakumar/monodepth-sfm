import os
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import wandb
from tqdm import tqdm

from networks.depth_net import DepthNet

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
            # Handle standard KITTI calib float formatting
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data

def generate_depth_map(calib_dir, velo_filename, im_shape):
    """
    Projects Velodyne sparse point cloud into the camera image plane.
    Requires both cam_to_cam and velo_to_cam calibration files.
    """
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))

    # P_rect_02: 3x4 projection matrix after rectification
    P_rect = cam2cam['P_rect_02'].reshape(3, 4)
    
    # R_rect_00: 3x3 rectifying rotation matrix
    R_rect = np.eye(4)
    R_rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    
    # Tr_velo_to_cam: 3x4 rigid transformation
    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[:3, :4] = velo2cam['Tr'].reshape(3, 4)

    # Load Velodyne points [N, 4] (x, y, z, reflectance)
    scan = np.fromfile(velo_filename, dtype=np.float32).reshape(-1, 4)
    pts_3d = scan[:, :3]
    
    # Filter out points behind the camera
    pts_3d = pts_3d[pts_3d[:, 0] >= 0, :]
    
    # Convert to homogeneous coordinates [N, 4]
    pts_3d_homo = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
    
    # Project: P_rect * R_rect * Tr_velo_to_cam * X
    pts_2d_homo = P_rect @ R_rect @ Tr_velo_to_cam @ pts_3d_homo.T
    pts_2d = (pts_2d_homo[:2, :] / pts_2d_homo[2, :]).T
    depths = pts_2d_homo[2, :]

    # Filter points outside image bounds
    h, w = im_shape
    val_inds = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & \
               (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
    
    pts_2d = pts_2d[val_inds, :]
    depths = depths[val_inds]

    # Create dense depth map (sparse projection)
    depth_map = np.zeros((h, w))
    pts_2d = np.int32(np.round(pts_2d))
    
    # Handle multiple points hitting the same pixel by keeping the closest one
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

    # Initialize W&B from existing checkpoint data if possible
    wandb.init(
        project="monodepth-sfm",
        name="eigen_split_eval",
        config=config,
        job_type="evaluation"
    )

    # 1. Load Model
    print(f"Loading checkpoint: {config['resume_ckpt']}")
    ckpt = torch.load(config['resume_ckpt'], map_location=device)
    
    depth_net = DepthNet(
        pretrained=False, # We are loading trained weights
        min_depth=config['min_depth'],
        max_depth=config['max_depth']
    ).to(device)
    
    depth_net.load_state_dict(ckpt['depth_net'])
    depth_net.eval()

    # ImageNet normalization used during training
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = T.ToTensor()

    # 2. Setup Evaluation Arrays
    errors = []
    
    # Here you would load the standard Eigen test split text file that contains
    # the exact 697 test frames. For this script, we assume a list of tuples:
    # (rgb_path, velodyne_path, calib_dir)
    # You will need to populate `test_files` based on your exact split text file.
    test_files = [] 
    
    # NOTE: In practice, read from 'eigen_test_files.txt'
    # Example format: 2011_09_26/2011_09_26_drive_0002_sync 0000000069
    
    print("Beginning Eigen Split evaluation...")
    with torch.no_grad():
        for rgb_path, velo_path, calib_dir in tqdm(test_files):
            # Load RGB
            img = Image.open(rgb_path).convert('RGB')
            orig_w, orig_h = img.size
            
            # Resize and normalize identically to train.py
            img_resized = img.resize((config['width'], config['height']), Image.LANCZOS)
            input_tensor = normalize(to_tensor(img_resized)).unsqueeze(0).to(device)
            
            # Predict
            _, depths = depth_net(input_tensor)
            pred_depth = depths[0].squeeze().cpu().numpy()
            
            # Upsample prediction back to original resolution for evaluation
            import cv2
            pred_depth_resized = cv2.resize(pred_depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            # Generate Ground Truth from Velodyne
            gt_depth = generate_depth_map(calib_dir, velo_path, (orig_h, orig_w))
            
            # Eigen split standard cropping (Garg crop)
            # Evaluates only on the valid interior region of the KITTI image
            crop = np.array([0.40810811 * orig_h, 0.99189189 * orig_h,
                             0.03594771 * orig_w, 0.96405229 * orig_w]).astype(np.int32)
            
            gt_mask = (gt_depth > config['min_depth']) & (gt_depth < config['max_depth'])
            crop_mask = np.zeros_like(gt_mask)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            
            valid_mask = gt_mask & crop_mask
            
            # Standard median scaling (since monocular scale is ambiguous)
            pred_valid = pred_depth_resized[valid_mask]
            gt_valid = gt_depth[valid_mask]
            
            if len(gt_valid) > 0:
                ratio = np.median(gt_valid) / np.median(pred_valid)
                pred_valid *= ratio
                
                # Clamp max depth for metric stability
                pred_valid[pred_valid < config['min_depth']] = config['min_depth']
                pred_valid[pred_valid > config['max_depth']] = config['max_depth']
                
                errors.append(compute_errors(gt_valid, pred_valid))

    # 3. Aggregate and Log Metrics
    mean_errors = np.array(errors).mean(0)
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = mean_errors

    print(f"\n--- Final Eigen Split Metrics ---")
    print(f"AbsRel:  {abs_rel:.4f}")
    print(f"SqRel:   {sq_rel:.4f}")
    print(f"RMSE:    {rmse:.4f}")
    print(f"RMSElog: {rmse_log:.4f}")
    print(f"δ < 1.25:   {a1:.4f}")
    print(f"δ < 1.25²:  {a2:.4f}")
    print(f"δ < 1.25³:  {a3:.4f}")

    wandb.log({
        "eval/AbsRel": abs_rel,
        "eval/SqRel": sq_rel,
        "eval/RMSE": rmse,
        "eval/RMSE_log": rmse_log,
        "eval/delta_1.25": a1,
        "eval/delta_1.25_sq": a2,
        "eval/delta_1.25_cube": a3,
    })
    
    wandb.finish()

if __name__ == '__main__':
    # You can import CONFIG from train.py directly or redefine it here
    from train import CONFIG
    evaluate(CONFIG)