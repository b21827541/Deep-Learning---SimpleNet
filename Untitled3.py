#!/usr/bin/env python
# coding: utf-8

import argparse
# Import libraries
import os
import time
from pathlib import Path

import numpy as np
import torch
# Add DDP imports
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import models, transforms
from torchvision.models import Wide_ResNet50_2_Weights
from tqdm import tqdm

# Set seed for reproducability
def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# Define weight initialization function
def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)


class FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.patch_size = 3  # as per paper

    # Get neighbourhood feats as stated in paper Eq 1 & 2
    # Return tensor shape B, C, H, W
    def get_neighborhood_features(self, feature_map):
        B, C, H, W = feature_map.shape
        pad = self.patch_size // 2
        padded = F.pad(feature_map, (pad, pad, pad, pad), mode='reflect')
        # Unfold to get local patches: shape [B, C*patch_size*patch_size, H*W]
        patches = F.unfold(padded, kernel_size=self.patch_size, stride=1)
        patches = patches.view(B, C, self.patch_size * self.patch_size, H, W)
        # Compute mean over the patch dimension -> [B, C, H, W]
        local_features = patches.mean(dim=2)
        return local_features

    # Forward pass
    # Get only layer 2 and 3 feats as per the paper
    def forward(self, x):
        # Pass through the backbone layers.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        # Get layer2 features (e.g. 512 channels)
        layer2_out = self.layer2(x)
        layer2_features = self.get_neighborhood_features(layer2_out)

        # Get layer3 features (e.g. 1024 channels)
        layer3_out = self.layer3(layer2_out)
        layer3_features = self.get_neighborhood_features(layer3_out)

        # Match spatial dimensions: resize layer3 features to layer2's HxW.
        _, _, H2, W2 = layer2_features.shape
        layer3_features = F.interpolate(layer3_features, size=(H2, W2),
                                        mode='bilinear', align_corners=False)

        # Return the unflattened features as a tuple.
        return (layer2_features, layer3_features)


# A simple adapter (projection) that is a single linear layer without bias.
class FeatureAdapter(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.fc = nn.Linear(feature_dim, feature_dim, bias=False)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        # x shape: [B*H*W, C] or [B, H*W, C]
        if x.dim() == 3:
            B, HW, C = x.shape
            x = x.reshape(-1, C)
        return self.fc(x)


# A discriminator with two fully-connected layers (with LeakyReLU)
class Discriminator(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # Use feature_dim for both layers
        self.fc1 = nn.Linear(feature_dim, feature_dim)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(feature_dim, 1, bias=True)

        # Proper initialization
        nn.init.xavier_normal_(self.fc1.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.fc2(x)
        return x


# A helper to initialize models and freeze the feature extractor parameters for training.
def initialize_models(device):
    # Load pre-trained Wide ResNet-50-2 from torchvision
    backbone = models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)

    # Freeze backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False
    backbone = backbone.to(device)

    # Wrap the backbone in our FeatureExtractor
    feature_extractor = FeatureExtractor(backbone).to(device)

    # Feature dimension: 512 (layer2) + 1024 (layer3) = 1536
    feature_dim = 1536
    adapter = FeatureAdapter(feature_dim).to(device)
    discriminator = Discriminator(feature_dim).to(device)

    # Initialize adapter and discriminator weights
    adapter.apply(init_weight)
    discriminator.apply(init_weight)

    return feature_extractor, adapter, discriminator


# Get Image transforms pipeline
def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# Dataset classes for DDP
class MVTecDataset(Dataset):
    def __init__(self, base_path, category, split='train', transform=None):
        """
        Args:
            base_path (Path) : Root directory of MVTec dataset
            category (str) : Category name (e.g., 'bottle')
            split (str) : 'train' or 'test'
            transform : Image transformations
        """
        self.base_path = base_path
        self.category = category
        self.split = split
        self.transform = transform if transform else get_transform()

        # Get image paths and labels
        self.samples = self._load_dataset()

    # Load the dataset and split
    def _load_dataset(self):
        path = self.base_path / self.category / self.split

        # Get all image paths
        if self.split == 'train':
            # Training set only contains good images
            image_paths = list(path.glob('good/*.png'))
            labels = [0] * len(image_paths)  # 0 for normal
        else:
            # Test set contains both good and defective images
            image_paths = []
            labels = []
            for defect_type in path.iterdir():
                if defect_type.is_dir():  # Skip any non-directory files
                    curr_paths = list(defect_type.glob('*.png'))
                    image_paths.extend(curr_paths)
                    # 0 for normal (good), 1 for anomaly
                    curr_labels = [0 if defect_type.name == 'good' else 1] * len(curr_paths)
                    labels.extend(curr_labels)

        return list(zip(image_paths, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        with Image.open(image_path) as img:
            img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)

        return img, label

# Create train and test dataloaders for a category with DDP support
def get_dataloaders(base_path, category, batch_size=4, world_size=1, rank=0):
    """
    base_path (Path) : Root directory of MVTec dataset
    category (str) : Category name (e.g., 'bottle')
    batch_size (int) : Batch size
    world_size (int) : World size
    rank (int) : Rank
    """
    transform = get_transform()

    # Create datasets
    train_dataset = MVTecDataset(base_path, category, 'train', transform)
    test_dataset = MVTecDataset(base_path, category, 'test', transform)

    # Create samplers for DDP
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    ) if world_size > 1 else None

    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    ) if world_size > 1 else None

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader, train_sampler

# Pool and combine features from two layers
def pool_and_combine_features(feats):
    """
    Expects feats to be a tuple: (layer2_features, layer3_features),
    each of shape [B, C, H, W]
    """
    layer2_feat = feats[0]
    layer3_feat = feats[1]

    # Apply average pooling.
    pooled_layer2 = F.avg_pool2d(layer2_feat, kernel_size=3, stride=1, padding=1)
    pooled_layer3 = F.avg_pool2d(layer3_feat, kernel_size=3, stride=1, padding=1)

    # Ensure spatial dimensions match.
    target_size = pooled_layer2.shape[-2:]
    resized_layer3 = F.interpolate(pooled_layer3, size=target_size,
                                   mode='bilinear', align_corners=False)

    # Concatenate along the channel dimension.
    combined = torch.cat([pooled_layer2, resized_layer3], dim=1)
    return combined

# Flatten a feature map from [B, C, H, W] to [B*H*W, C]
def flatten_features(feat):
    B, C, H, W = feat.shape
    return feat.permute(0, 2, 3, 1).reshape(-1, C)

# Compute truncated L1 loss with correct thresholds
def compute_loss(normal_pred, anomalous_pred, th_pos=0.5, th_neg=-0.5):
    """
    normal_pred : (B, C, H, W)
    anomalous_pred : (B, C, H, W)
    th_pos : float
    th_neg : float
    """
    loss_normal = torch.clamp(th_pos - normal_pred, min=0)
    loss_anomalous = torch.clamp(-th_neg + anomalous_pred, min=0)
    return (loss_normal.mean() + loss_anomalous.mean())

# Add Gaussian noise to embeddings
def inject_noise(embedding, noise_std=0.015):
    noise = torch.randn_like(embedding) * noise_std
    return embedding + noise


class AnomalyEvaluator:
    def __init__(self, feature_extractor, adapter, discriminator, device):
        """Initialize evaluator with models."""
        self.feature_extractor = feature_extractor
        self.adapter = adapter
        self.discriminator = discriminator
        self.device = device

        # Switch to eval mode
        self.feature_extractor.eval()
        self.adapter.eval()
        self.discriminator.eval()

    def generate_anomaly_maps(self, test_loader, rank):
        """Generate anomaly maps for test images."""
        image_scores = []
        true_labels = []

        # Unwrap models if DDP is used
        def unwrap(model):
            return model.module if hasattr(model, "module") else model

        feat_model = unwrap(self.feature_extractor).to(self.device)
        adapter_model = unwrap(self.adapter).to(self.device)
        discrim_model = unwrap(self.discriminator).to(self.device)

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"[GPU:{rank}] Generating anomaly maps"):
                images = images.to(self.device)
                feats = feat_model(images)
                combined_feats = pool_and_combine_features(feats)
                embedding = flatten_features(combined_feats)
                adapted = adapter_model(embedding)
                # Get anomaly scores
                scores = discrim_model(adapted)
                # Define anomaly score as -D(q)
                scores = -scores
                B = images.shape[0]
                scores = scores.view(B, -1)
                image_scores_batch = scores.max(dim=1)[0]
                image_scores.append(image_scores_batch.cpu().numpy())
                true_labels.append(labels.cpu().numpy())

        all_scores = np.concatenate(image_scores)
        all_labels = np.concatenate(true_labels)

        # Only compute AUROC on rank 0 if distributed
        if dist.is_initialized():
            # Gather all scores and labels from all ranks
            gathered_scores = [None for _ in range(dist.get_world_size())]
            gathered_labels = [None for _ in range(dist.get_world_size())]

            dist.gather_object(all_scores, gathered_scores if rank == 0 else None, dst=0)
            dist.gather_object(all_labels, gathered_labels if rank == 0 else None, dst=0)

            if rank == 0:
                all_scores = np.concatenate([s for s in gathered_scores if s is not None])
                all_labels = np.concatenate([l for l in gathered_labels if l is not None])
                auroc = roc_auc_score(all_labels, all_scores)
                print(f"Image-level AUROC: {auroc:.4f}")
                return auroc
            return None
        else:
            auroc = roc_auc_score(all_labels, all_scores)
            print(f"Image-level AUROC: {auroc:.4f}")
            return auroc

    def evaluate(self, test_loader, rank=0):
        """Evaluate model performance on test set."""
        auroc = self.generate_anomaly_maps(test_loader, rank)

        if auroc is not None:
            return {'image_auroc': auroc}
        else:
            return {'image_auroc': 0.0}  # Return placeholder for non-rank-0 processes


def save_checkpoint(adapter, discriminator, optimizer, epoch, auroc, category, rank, is_best=False):
    """
    Save model checkpoints during training.
    """
    if rank != 0:
        return  # Only save on rank 0

    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = Path('checkpoints') / category
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint filename
    checkpoint_filename = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
    best_checkpoint_filename = checkpoint_dir / 'best_model.pth'

    # Create checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'adapter_state_dict': adapter.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'auroc': auroc
    }

    # Save current checkpoint
    torch.save(checkpoint, checkpoint_filename)

    # Save best model if current model is the best
    if is_best:
        torch.save(checkpoint, best_checkpoint_filename)
        print(f"New best model saved with AUROC: {auroc:.4f}")


def train_model_ddp(rank, world_size, args):
    """Main training function with DDP"""
    # Initialize distribution process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # Set device for this process
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    seed_everything(args.seed + rank)  # Different seed per process

    if rank == 0:
        print(f"\nTraining category: {args.category}")
        print("=" * 50)
        print(f"World size: {world_size}, Using DDP")

    # Create dataset in mvtec_anomaly_detection folder relative to current file
    ad_path = Path(os.path.dirname(os.path.abspath(__file__))) / "mvtec_anomaly_detection"

    # Initialize models
    feature_extractor, adapter, discriminator = initialize_models(device)

    # Only wrap trainable models in DDP (not feature_extractor since it's frozen)
    # feature_extractor doesn't need DDP since all params require_grad=False
    adapter = DDP(adapter, device_ids=[rank])
    discriminator = DDP(discriminator, device_ids=[rank])

    # Create dataloaders with appropriate samplers
    train_loader, test_loader, train_sampler = get_dataloaders(
        ad_path,
        args.category,
        args.batch_size,
        world_size,
        rank
    )

    # Optimizer with appropriate learning rates
    optimizer = torch.optim.Adam([
        {'params': adapter.parameters(), 'lr': args.learning_rate},
        {'params': discriminator.parameters(), 'lr': 2e-4}
    ], weight_decay=1e-5)

    best_auroc = 0
    noise_std = 0.015

    # Training loop
    for epoch in range(args.meta_epochs):
        # Set epoch for the sampler
        if train_sampler:
            train_sampler.set_epoch(epoch)

        feature_extractor.eval()  # frozen
        adapter.train()
        discriminator.train()

        total_loss = 0

        for gan_epoch in range(4):
            desc = f'Epoch {epoch + 1}/{args.meta_epochs} (GAN Epoch {gan_epoch + 1}/4)'
            progress_bar = tqdm(train_loader, desc=desc) if rank == 0 else train_loader

            for batch_idx, (images, _) in enumerate(progress_bar):
                images = images.to(device)

                with torch.no_grad():
                    features = feature_extractor(images)  # list of feature maps

                # Pool and combine features
                combined_feats = pool_and_combine_features(features)  # [B, C, H, W]
                B, C, H, W = combined_feats.shape

                # Flatten to [B*H*W, C]
                embedding = combined_feats.permute(0, 2, 3, 1).reshape(-1, C)

                # Adapt features
                adapted_features = adapter(embedding)

                # Generate anomalous features
                anomalous_features = inject_noise(adapted_features, noise_std=noise_std)

                # Get discriminator outputs
                normal_pred = discriminator(adapted_features)
                anomalous_pred = discriminator(anomalous_features)

                # Compute loss
                loss = compute_loss(normal_pred, anomalous_pred)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if rank == 0 and batch_idx % 10 == 0 and isinstance(progress_bar, tqdm):
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                    })

        # Calculate average loss across all processes
        avg_loss = total_loss / (len(train_loader) * 4)

        if rank == 0:
            print(f"Epoch {epoch + 1}/{args.meta_epochs}: Avg Loss = {avg_loss:.4f}")

        # Evaluation
        if (epoch + 1) % 5 == 0 or epoch == args.meta_epochs - 1:
            # Switch to eval mode
            feature_extractor.eval()
            adapter.eval()
            discriminator.eval()

            # Create evaluator
            evaluator = AnomalyEvaluator(feature_extractor, adapter, discriminator, device)
            auroc_results = evaluator.evaluate(test_loader, rank)

            # Only rank 0 gets a meaningful AUROC
            auroc = auroc_results['image_auroc']

            if rank == 0:
                print(f"Rank {rank}: Image-level AUROC: {auroc:.4f}")

            # Synchronize processes
            dist.barrier()

            # Broadcast AUROC from rank 0 to all processes
            auroc_tensor = torch.tensor([auroc], device=device)
            dist.broadcast(auroc_tensor, src=0)
            auroc = auroc_tensor.item()

            if auroc > best_auroc:
                best_auroc = auroc
                save_checkpoint(adapter, discriminator, optimizer, epoch, auroc,
                                args.category, rank, is_best=True)

    # Clean up
    dist.destroy_process_group()

    if rank == 0:
        print(f"\nTraining completed. Best AUROC: {best_auroc:.4f}")

    return best_auroc


def train_all_categories_ddp(world_size, args):
    """Train all categories with DDP"""
    categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
                  'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
                  'wood', 'zipper']

    results = {}
    for category in categories:
        args.category = category
        mp.spawn(
            train_model_ddp,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )


def main():
    parser = argparse.ArgumentParser(description='Train anomaly detection model with DDP')
    parser.add_argument('--category', type=str, default='bottle',
                        help='MVTec category to train on')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size per GPU')
    parser.add_argument('--meta_epochs', type=int, default=40,
                        help='Number of meta epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--all', action='store_true',
                        help='Train all categories')
    parser.add_argument('--port', type=str, default='12355',
                        help='Port for DDP communication')
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='Number of GPUs to use (default: all available)')

    args = parser.parse_args()

    # Set random seed for reproducibility
    seed_everything(args.seed)

    # Get world size (number of GPUs)
    available_gpus = torch.cuda.device_count()
    world_size = args.num_gpus if args.num_gpus is not None else available_gpus

    # Limit world_size to available GPUs
    world_size = min(world_size, available_gpus)

    print(f"Number of GPUs available: {available_gpus}")
    print(f"Using {world_size} GPUs for training")

    if world_size < 1:
        raise ValueError("No GPU available for training")

    # Train all categories or just one
    if args.all:
        train_all_categories_ddp(world_size, args)
    else:
        # Start DDP training
        mp.spawn(
            train_model_ddp,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")