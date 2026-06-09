# MonoDepth: Self-Supervised Monocular Depth and Ego-Motion Estimation

A PyTorch implementation of a self-supervised monocular depth estimation framework
trained on the KITTI Vision Benchmark Suite. The system learns to predict dense
depth maps from single RGB images without any ground-truth depth supervision,
using multi-frame view synthesis as the training signal.

Evaluated on the Eigen split test set:

| Metric     | Score  |
|------------|--------|
| AbsRel     | 0.137  |
| SqRel      | 1.39   |
| RMSE       | 6.01m  |
| RMSElog    | 0.235  |
| δ < 1.25   | 83.7%  |


## How it works

The core idea is that if you know how a camera moved between two video frames,
and you know how far away everything is in the current frame, you can
mathematically synthesize what the current frame should look like by warping
the neighboring frame into it. If your depth and motion predictions are correct,
the synthesized frame will be pixel-for-pixel identical to the real frame. The
difference between them is the training signal.

This eliminates the need for LiDAR, stereo rigs, or any form of depth annotation.
The network learns geometry purely from watching a camera move through the world.

Three components work together during training:

**Depth network.** Takes a single RGB frame and outputs inverse depth (disparity)
maps at four decoder scales, all upsampled to full input resolution before loss
computation. Built on a ResNet18 encoder pretrained on ImageNet with a custom
multi-scale decoder using skip connections.

**Pose network.** Takes two consecutive RGB frames concatenated along the channel
dimension and regresses a 6-DoF relative camera transformation represented as
axis-angle rotation and translation. The output is converted to a 4x4 SE(3)
rigid body transformation matrix via Rodrigues rotation formula.

**Inverse warp engine.** Pure differentiable geometry. Unprojects the target
frame to 3D using predicted depth and camera intrinsics, applies the pose
transformation, reprojects into the source frame coordinate system, and
bilinearly samples pixel values. Fully differentiable so gradients flow back
through the sampling operation to both networks.


## Repository structure

```
monodepth-sfm/
├── geometry/
│   ├── __init__.py
│   └── warp.py              # BackprojectDepth, Project3DPoints, inverse_warp
├── networks/
│   ├── __init__.py
│   ├── depth_net.py         # DepthNet: ResNet18 encoder + multi-scale decoder
│   └── pose_net.py          # PoseNet: CNN + axis-angle regression
├── losses/
│   ├── __init__.py
│   └── loss.py              # SSIM, PhotometricLoss, AutoMask, SmoothnessLoss, SfMLoss
├── data/
│   ├── __init__.py
│   └── kitti_dataset.py     # KITTIDepthDataset, build_dataloader
├── train.py                 # Training loop with resume, early stopping, visualization
├── evaluate.py              # Eigen split evaluation against LiDAR ground truth
├── viz_scaled.py            # Generate metric-scaled depth visualizations
└── requirements.txt
```


## Setup

Clone the repository and create a Python environment. Python 3.11 is recommended.

```bash
git clone https://github.com/kanaghasivakumar/monodepth-sfm.git
cd monodepth-sfm
```

Install dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install tensorboard pillow numpy matplotlib
```

The training script expects PyTorch 2.x with CUDA. CPU training is supported
but will be extremely slow given the dataset size.


Don't add them. They contain Quest-specific paths like `/gpfs/projects/e32706/omb8654/` hardcoded throughout, which would immediately reveal the HPC infrastructure. Anyone reading the scripts would see exactly where they were run.

Replace that section with something neutral that describes what to do manually:


## Data preparation

Download the KITTI Raw dataset from the official source:
https://www.cvlibs.net/datasets/kitti/raw_data.php

For each sequence, download the synced and rectified data zip and the
corresponding date-level calibration zip. Extract only the `image_02` folder
from each sequence zip and place the `calib_cam_to_cam.txt` file from the
calibration zip directly inside the sequence folder.

For evaluation, additionally download the Velodyne point cloud data for the
Eigen test sequences and place `calib_velo_to_cam.txt` alongside
`calib_cam_to_cam.txt` in each test sequence folder.

The expected structure is:

```
kitti_data/
├── train/
│   ├── 2011_09_26_drive_0001_sync/
│   │   ├── image_02/data/
│   │   └── calib_cam_to_cam.txt
│   └── ...
└── test/
    ├── 2011_09_26_drive_0009_sync/
    │   ├── image_02/
    │   ├── velodyne_points/
    │   ├── calib_cam_to_cam.txt
    │   └── calib_velo_to_cam.txt
    └── ...
```

The dataset loader indexes all valid temporal triplets automatically.
35,436 triplets are available across the full Eigen training split.



## Training

All hyperparameters are configured in the `CONFIG` dictionary at the top of
`train.py`. The most important ones:

```python
CONFIG = {
    'root_dir':            'kitti_data',
    'height':              192,
    'width':               640,
    'batch_size':          12,
    'lr':                  1e-4,
    'num_epochs':          150,
    'lambda_smooth':       1e-3,
    'lr_patience':         3,
    'early_stop_patience': 8,
    'resume_ckpt':         'checkpoints/best_checkpoint.pth',
}
```

Start training:

```bash
python train.py
```

Training saves a checkpoint after every epoch under `checkpoints/` and keeps
a separate `best_checkpoint.pth` tracking the lowest validation loss. Loss
curves are written to `curves/` as PNGs after every epoch. Depth visualizations
on a fixed reference batch are written to `viz/` after every epoch.

To resume from a checkpoint, set `resume_ckpt` to the path of the checkpoint
you want to restore. The training loop restores model weights, optimizer state,
scheduler state, loss history, and early stopping counter so training continues
seamlessly from where it left off.

TensorBoard logs are written to `runs/`. To view them:

```bash
tensorboard --logdir runs/
```

On a single NVIDIA A100 with batch size 12 and 35,436 training triplets,
each epoch takes roughly 17 minutes. The presented model was trained for
70 epochs, reaching a best photometric loss of 0.296 before early stopping
criteria were met.


## Evaluation

Run the Eigen split evaluation against LiDAR ground truth:

```bash
python evaluate.py
```

This loads `checkpoints/best_checkpoint.pth` by default. The evaluation script
downloads the standard Eigen test split file from the Monodepth2 repository on
first run, projects the Velodyne point clouds into the camera image plane to
produce sparse ground-truth depth maps, and computes the standard seven metrics:
AbsRel, SqRel, RMSE, RMSElog, and the three threshold accuracy metrics.

Per-image median scaling is applied at test time, which is standard protocol
for self-supervised monocular depth evaluation. The network predicts depth up
to an unknown scale factor since the photometric loss is scale-invariant, and
median scaling recovers the metric scale for each test image independently.


## Generating depth visualizations

To generate metric-scaled depth map visualizations on test images:

```bash
python viz_scaled.py
```

Edit the `test_images` list in the script to point at the frames you want to
visualize. Output PNGs are written to `viz/scaled/`. The predicted depth values
are scaled by the median scale factor observed during evaluation to produce
approximate metric depth in meters.


## Loss function

The total loss at each decoder scale s is:

```
L_s = L_photo + lambda_smooth * L_smooth / (2^s)
```

The photometric loss combines SSIM and L1:

```
L_photo = 0.85 * SSIM(I_t_prime, I_t) + 0.15 * |I_t_prime - I_t|
```

Auto-masking suppresses pixels where the photometric error of the unwarped
source frame against the target is lower than the warped error. This handles
two failure modes: independently moving objects that violate the static world
assumption, and static camera frames where the view synthesis loss collapses
to near-zero regardless of depth predictions.

The edge-aware smoothness loss penalizes large depth gradients, weighted
exponentially by image gradients so the penalty is suppressed at true object
edges:

```
L_smooth = |d_disp/dx| * exp(-|dI/dx|) + |d_disp/dy| * exp(-|dI/dy|)
```

Disparity is mean-normalized before computing this loss to prevent the network
from driving all values toward zero to minimize the penalty.


## Known limitations

Scale ambiguity is inherent to self-supervised monocular depth. The network
learns depth up to an unknown scale factor and metric scale is recovered at
evaluation time via median scaling against LiDAR. This means the model cannot
produce metric depth estimates at inference time without an external reference.

Performance is sensitive to scene diversity in the training data. The Eigen
split used here covers primarily urban and residential driving in Germany.
Generalization to other environments, lighting conditions, or camera setups
will require fine-tuning or retraining.

The static world assumption underlying structure-from-motion is violated by
independently moving objects. Auto-masking mitigates this but does not fully
solve the problem. Pixels on moving objects often have degraded depth predictions.


## References

Godard, C., Mac Aodha, O., Firman, M., and Brostow, G. (2019). Digging into
self-supervised monocular depth estimation. ICCV 2019.

Geiger, A., Lenz, P., Stiller, C., and Urtasun, R. (2013). Vision meets
robotics: The KITTI dataset. International Journal of Robotics Research, 32(11).

Zhou, T., Brown, M., Snavely, N., and Lowe, D. (2017). Unsupervised learning
of depth and ego-motion from video. CVPR 2017.