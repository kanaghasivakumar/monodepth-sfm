import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from data.kitti_dataset import build_dataloader
from networks.depth_net import DepthNet
from networks.pose_net import PoseNet
from geometry.warp import BackprojectDepth, Project3DPoints, inverse_warp
from losses.loss import SfMLoss


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    # Data
    'root_dir':    'kitti_data',
    'height':      192,
    'width':       640,
    'batch_size':  12,
    'num_workers': 4,

    # Model
    'pretrained_encoder': True,
    'min_depth':   0.1,
    'max_depth':   100.0,

    # Loss
    'lambda_smooth': 1e-3,
    'num_scales':    4,

    # Optimization
    'lr':           1e-4,
    'weight_decay': 1e-5,
    'num_epochs':   20,
    'scheduler_step_size': 15,   # drop LR by 0.1 at this epoch

    # Logging
    'log_dir':      'runs/sfm_learner',
    'save_dir':     'checkpoints',
    'log_freq':     100,          # log every N steps
}


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------

def build_models(config, device):
    """
    Instantiates and returns all trainable modules on the target device.

    Returns:
        depth_net:   DepthNet
        pose_net:    PoseNet
        backproject: BackprojectDepth
        project:     Project3DPoints
    """
    depth_net = DepthNet(
        pretrained=config['pretrained_encoder'],
        min_depth=config['min_depth'],
        max_depth=config['max_depth']
    ).to(device)

    pose_net = PoseNet().to(device)

    # Geometry modules: stateless except for the static pixel grid buffer
    # in BackprojectDepth — must match batch size and spatial dims exactly
    backproject = BackprojectDepth(
        batch_size=config['batch_size'],
        height=config['height'],
        width=config['width']
    ).to(device)

    project = Project3DPoints(
        batch_size=config['batch_size'],
        height=config['height'],
        width=config['width']
    ).to(device)

    return depth_net, pose_net, backproject, project


# ---------------------------------------------------------------------------
# Single training step
# ---------------------------------------------------------------------------

def training_step(batch, depth_net, pose_net, backproject, project,
                  loss_fn, device):
    """
    Executes a single forward pass and loss computation for one batch.

    Full pipeline:
        1. Predict multi-scale depth maps for I_t
        2. Predict relative poses T(t→t-1) and T(t→t+1)
        3. Warp I_t-1 and I_t+1 into I_t's coordinate space
        4. Compute photometric + smoothness loss with auto-masking
        5. Return total loss scalar and breakdown dict for logging

    Args:
        batch:      dict from KITTIDepthDataset.__getitem__
        depth_net:  DepthNet
        pose_net:   PoseNet
        backproject: BackprojectDepth
        project:    Project3DPoints
        loss_fn:    SfMLoss
        device:     torch.device

    Returns:
        loss:           scalar tensor (differentiable)
        loss_breakdown: dict of float scalars for logging
    """
    # --- Unpack batch and move to device ---
    target      = batch['target'].to(device)       # [B, 3, H, W]
    source_prev = batch['source_prev'].to(device)  # [B, 3, H, W]
    source_next = batch['source_next'].to(device)  # [B, 3, H, W]
    K           = batch['K'].to(device)            # [B, 3, 3]
    K_inv       = batch['K_inv'].to(device)        # [B, 3, 3]
    target_raw  = batch['target_raw'].to(device)   # [B, 3, H, W]

    # --- Step 1: Predict depth of target frame I_t ---
    # disps: {0..3: [B, 1, H, W]}  all at full resolution
    # depths: {0..3: [B, 1, H, W]} converted from disparity
    disps, depths = depth_net(target)

    # Use full-resolution depth for warping
    depth_full = depths[0]   # [B, 1, H, W]
    print("depth_full shape:", depth_full.shape)

    # --- Step 2: Predict relative poses ---
    # Two forward passes: (I_t, I_t-1) and (I_t, I_t+1)
    # pose_net returns T: [B, 4, 4] — rigid transform from target to source
    T_prev, _, _ = pose_net(target, source_prev)  # [B, 4, 4]
    T_next, _, _ = pose_net(target, source_next)  # [B, 4, 4]

    # --- Step 3: Inverse warp both source frames into I_t space ---
    warped_prev, _, _ = inverse_warp(
        source_frame=source_prev,
        depth=depth_full,
        K=K,
        K_inv=K_inv,
        T=T_prev,
        backproject=backproject,
        project=project
    )  # [B, 3, H, W]

    warped_next, _, _ = inverse_warp(
        source_frame=source_next,
        depth=depth_full,
        K=K,
        K_inv=K_inv,
        T=T_next,
        backproject=backproject,
        project=project
    )  # [B, 3, H, W]

    # --- Step 4: Compute loss ---
    # SfMLoss expects lists of warped and source frames
    loss, loss_breakdown = loss_fn(
        target=target_raw,             # unnormalized target for loss computation
        warped_frames=[warped_prev, warped_next],
        source_frames=[source_prev, source_next],
        disps=disps
    )

    return loss, loss_breakdown


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config):
    """
    Full training loop with:
        - AdamW optimizer with cosine-ish step LR decay
        - TensorBoard logging of loss components
        - Checkpoint saving every epoch
        - Gradient clipping for stability
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # --- Data ---
    train_loader = build_dataloader(
        root_dir=config['root_dir'],
        split='train',
        height=config['height'],
        width=config['width'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        augment=True
    )

    # --- Models ---
    depth_net, pose_net, backproject, project = build_models(config, device)

    # --- Loss ---
    loss_fn = SfMLoss(
        lambda_smooth=config['lambda_smooth'],
        num_scales=config['num_scales']
    ).to(device)

    # --- Optimizer: joint over both networks ---
    params = list(depth_net.parameters()) + list(pose_net.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # Step LR: drop by factor 10 at scheduler_step_size epochs
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['scheduler_step_size'],
        gamma=0.1
    )

    # --- Logging ---
    writer = SummaryWriter(log_dir=config['log_dir'])
    import os
    os.makedirs(config['save_dir'], exist_ok=True)

    global_step = 0

    for epoch in range(config['num_epochs']):
        depth_net.train()
        pose_net.train()

        for batch_idx, batch in enumerate(train_loader):

            optimizer.zero_grad()

            loss, breakdown = training_step(
                batch=batch,
                depth_net=depth_net,
                pose_net=pose_net,
                backproject=backproject,
                project=project,
                loss_fn=loss_fn,
                device=device
            )

            loss.backward()

            # Gradient clipping: prevents exploding gradients in pose network
            # especially during early training when poses are far from identity
            nn.utils.clip_grad_norm_(params, max_norm=1.0)

            optimizer.step()

            # --- Logging ---
            if global_step % config['log_freq'] == 0:
                writer.add_scalar('loss/total',  loss.item(),             global_step)
                writer.add_scalar('loss/photo',  breakdown['photo'],      global_step)
                writer.add_scalar('loss/smooth', breakdown['smooth'],     global_step)
                writer.add_scalar('train/lr',    scheduler.get_last_lr()[0], global_step)

                print(
                    f"Epoch [{epoch+1:02d}/{config['num_epochs']}] "
                    f"Step [{batch_idx+1:04d}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Photo: {breakdown['photo']:.4f} "
                    f"Smooth: {breakdown['smooth']:.6f}"
                )

            global_step += 1

        scheduler.step()

        # --- Save checkpoint every epoch ---
        checkpoint = {
            'epoch':          epoch + 1,
            'depth_net':      depth_net.state_dict(),
            'pose_net':       pose_net.state_dict(),
            'optimizer':      optimizer.state_dict(),
            'scheduler':      scheduler.state_dict(),
            'config':         config,
        }
        ckpt_path = os.path.join(
            config['save_dir'], f'checkpoint_epoch_{epoch+1:02d}.pth'
        )
        torch.save(checkpoint, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    writer.close()
    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    train(CONFIG)