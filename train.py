import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    'lr':               1e-4,
    'weight_decay':     1e-5,
    'num_epochs':       50,

    # ReduceLROnPlateau
    'lr_patience':      3,       # epochs with no improvement before LR drop
    'lr_factor':        0.5,     # multiply LR by this on plateau
    'min_lr':           1e-6,    # floor on LR

    # Early stopping
    'early_stop_patience': 8,    # epochs with no improvement before stopping

    # Logging
    'log_dir':      'runs/sfm_learner',
    'save_dir':     'checkpoints',
    'curves_dir':   'curves',
    'viz_dir':      'viz',
    'log_freq':     100,
}


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------

def build_models(config, device):
    depth_net = DepthNet(
        pretrained=config['pretrained_encoder'],
        min_depth=config['min_depth'],
        max_depth=config['max_depth']
    ).to(device)

    pose_net = PoseNet().to(device)

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
    target      = batch['target'].to(device)
    source_prev = batch['source_prev'].to(device)
    source_next = batch['source_next'].to(device)
    K           = batch['K'].to(device)
    K_inv       = batch['K_inv'].to(device)
    target_raw  = batch['target_raw'].to(device)

    disps, depths = depth_net(target)
    depth_full = depths[0]

    T_prev, _, _ = pose_net(target, source_prev)
    T_next, _, _ = pose_net(target, source_next)

    warped_prev, _, _ = inverse_warp(
        source_frame=source_prev,
        depth=depth_full, K=K, K_inv=K_inv, T=T_prev,
        backproject=backproject, project=project
    )
    warped_next, _, _ = inverse_warp(
        source_frame=source_next,
        depth=depth_full, K=K, K_inv=K_inv, T=T_next,
        backproject=backproject, project=project
    )

    loss, loss_breakdown = loss_fn(
        target=target_raw,
        warped_frames=[warped_prev, warped_next],
        source_frames=[source_prev, source_next],
        disps=disps
    )

    return loss, loss_breakdown, depth_full, target_raw


# ---------------------------------------------------------------------------
# Depth visualization
# ---------------------------------------------------------------------------

def save_depth_viz(depth_tensor, target_tensor, epoch, viz_dir):
    """
    Saves a side-by-side PNG of the input RGB frame and predicted depth map
    for the first sample in the batch.

    Args:
        depth_tensor:  [B, 1, H, W]
        target_tensor: [B, 3, H, W] unnormalized RGB in [0, 1]
        epoch:         int
        viz_dir:       str
    """
    os.makedirs(viz_dir, exist_ok=True)

    # Take first sample in batch, move to CPU numpy
    rgb   = target_tensor[0].cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
    depth = depth_tensor[0, 0].cpu().detach().numpy()          # [H, W]

    # Clip RGB to [0,1] in case of floating point noise
    rgb = np.clip(rgb, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].imshow(rgb)
    axes[0].set_title(f'Input RGB — Epoch {epoch}')
    axes[0].axis('off')

    im = axes[1].imshow(depth, cmap='plasma', vmin=0, vmax=80)
    axes[1].set_title(f'Predicted Depth — Epoch {epoch}')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = os.path.join(viz_dir, f'depth_epoch_{epoch:03d}.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Curve saving
# ---------------------------------------------------------------------------

def save_curves(history, curves_dir):
    """
    Saves loss and LR curves as PNGs.

    Args:
        history:    dict with keys 'total', 'photo', 'smooth', 'lr'
                    each mapping to a list of per-epoch floats
        curves_dir: str
    """
    os.makedirs(curves_dir, exist_ok=True)
    epochs = range(1, len(history['total']) + 1)

    # Loss curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history['total'], 'b-o', markersize=3)
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)

    axes[1].plot(epochs, history['photo'], 'r-o', markersize=3)
    axes[1].set_title('Photometric Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].grid(True)

    axes[2].plot(epochs, history['smooth'], 'g-o', markersize=3)
    axes[2].set_title('Smoothness Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(curves_dir, 'loss_curves.png'), dpi=100)
    plt.close()

    # LR curve
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history['lr'], 'm-o', markersize=3)
    ax.set_title('Learning Rate')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LR')
    ax.set_yscale('log')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(curves_dir, 'lr_curve.png'), dpi=100)
    plt.close()

    # Also save raw history as numpy for later use in evaluation
    np.save(os.path.join(curves_dir, 'history.npy'), history)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    os.makedirs(config['save_dir'],   exist_ok=True)
    os.makedirs(config['curves_dir'], exist_ok=True)
    os.makedirs(config['viz_dir'],    exist_ok=True)

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
    print(f"Training on {len(train_loader.dataset)} triplets, "
          f"{len(train_loader)} steps/epoch")

    # --- Models ---
    depth_net, pose_net, backproject, project = build_models(config, device)

    # --- Loss ---
    loss_fn = SfMLoss(
        lambda_smooth=config['lambda_smooth'],
        num_scales=config['num_scales']
    ).to(device)

    # --- Optimizer ---
    params = list(depth_net.parameters()) + list(pose_net.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # ReduceLROnPlateau: monitors epoch-mean total loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['lr_factor'],
        patience=config['lr_patience'],
        min_lr=config['min_lr'],
        verbose=True
    )

    # --- Logging ---
    writer = SummaryWriter(log_dir=config['log_dir'])

    # History for curve saving
    history = {'total': [], 'photo': [], 'smooth': [], 'lr': []}

    # Early stopping state
    best_loss         = float('inf')
    early_stop_counter = 0
    best_ckpt_path    = None

    global_step = 0

    # Keep one fixed batch for consistent epoch visualizations
    viz_batch = None

    for epoch in range(1, config['num_epochs'] + 1):
        depth_net.train()
        pose_net.train()

        epoch_losses  = []
        epoch_photo   = []
        epoch_smooth  = []

        for batch_idx, batch in enumerate(train_loader):

            # Save the first batch of epoch 1 as the fixed viz batch
            if viz_batch is None:
                viz_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}

            optimizer.zero_grad()

            loss, breakdown, depth_full, target_raw = training_step(
                batch=batch,
                depth_net=depth_net,
                pose_net=pose_net,
                backproject=backproject,
                project=project,
                loss_fn=loss_fn,
                device=device
            )

            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_photo.append(breakdown['photo'])
            epoch_smooth.append(breakdown['smooth'])

            # Step-level TensorBoard logging
            if global_step % config['log_freq'] == 0:
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('step/loss_total',  loss.item(),       global_step)
                writer.add_scalar('step/loss_photo',  breakdown['photo'],global_step)
                writer.add_scalar('step/loss_smooth', breakdown['smooth'],global_step)
                writer.add_scalar('step/lr',          current_lr,        global_step)

                print(
                    f"Epoch [{epoch:02d}/{config['num_epochs']}] "
                    f"Step [{batch_idx+1:04d}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}  "
                    f"Photo: {breakdown['photo']:.4f}  "
                    f"Smooth: {breakdown['smooth']:.6f}  "
                    f"LR: {current_lr:.2e}"
                )

            global_step += 1

        # --- Epoch-level metrics ---
        mean_loss   = np.mean(epoch_losses)
        mean_photo  = np.mean(epoch_photo)
        mean_smooth = np.mean(epoch_smooth)
        current_lr  = optimizer.param_groups[0]['lr']

        history['total'].append(float(mean_loss))
        history['photo'].append(float(mean_photo))
        history['smooth'].append(float(mean_smooth))
        history['lr'].append(float(current_lr))

        writer.add_scalar('epoch/loss_total',  mean_loss,   epoch)
        writer.add_scalar('epoch/loss_photo',  mean_photo,  epoch)
        writer.add_scalar('epoch/loss_smooth', mean_smooth, epoch)
        writer.add_scalar('epoch/lr',          current_lr,  epoch)

        print(f"\n--- Epoch {epoch:02d} Summary ---")
        print(f"  Mean Loss:   {mean_loss:.4f}")
        print(f"  Mean Photo:  {mean_photo:.4f}")
        print(f"  Mean Smooth: {mean_smooth:.6f}")
        print(f"  LR:          {current_lr:.2e}")

        # --- LR scheduler step ---
        scheduler.step(mean_loss)

        # --- Depth visualization on fixed batch ---
        depth_net.eval()
        with torch.no_grad():
            _, viz_depths = depth_net(viz_batch['target'].to(device))
            save_depth_viz(
                depth_tensor=viz_depths[0],
                target_tensor=viz_batch['target_raw'].to(device),
                epoch=epoch,
                viz_dir=config['viz_dir']
            )
        depth_net.train()

        # --- Save loss curves after every epoch ---
        save_curves(history, config['curves_dir'])

        # --- Checkpoint: save every epoch, keep best separately ---
        ckpt = {
            'epoch':      epoch,
            'depth_net':  depth_net.state_dict(),
            'pose_net':   pose_net.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'scheduler':  scheduler.state_dict(),
            'config':     config,
            'history':    history,
        }
        ckpt_path = os.path.join(config['save_dir'],
                                 f'checkpoint_epoch_{epoch:03d}.pth')
        torch.save(ckpt, ckpt_path)

        if mean_loss < best_loss:
            best_loss = mean_loss
            early_stop_counter = 0
            best_ckpt_path = os.path.join(config['save_dir'], 'best_checkpoint.pth')
            torch.save(ckpt, best_ckpt_path)
            print(f"  New best model saved (loss={best_loss:.4f})")
        else:
            early_stop_counter += 1
            print(f"  No improvement for {early_stop_counter}/"
                  f"{config['early_stop_patience']} epochs")

        # --- Early stopping ---
        if early_stop_counter >= config['early_stop_patience']:
            print(f"\nEarly stopping triggered at epoch {epoch}. "
                  f"Best loss: {best_loss:.4f}")
            break

        print()

    writer.close()
    save_curves(history, config['curves_dir'])
    print(f"\nTraining complete. Best checkpoint: {best_ckpt_path}")
    print(f"Best loss: {best_loss:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    train(CONFIG)