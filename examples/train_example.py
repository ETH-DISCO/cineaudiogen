"""
Example Training Script for Stem Separation with Auxiliary Targets

This demonstrates how to integrate auxiliary targets into a training loop.
Replace the placeholder encoder with your actual model architecture.

Usage:
    python train_example.py /path/to/dataset --epochs 10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from cineaudiogen.auxiliary_targets import (
    StemSeparationDataset,
    AuxiliaryHeads,
    MusicTag,
    ReverbType
)


# =============================================================================
# PLACEHOLDER MODEL (Replace with your actual architecture)
# =============================================================================

class PlaceholderEncoder(nn.Module):
    """
    Placeholder encoder - replace with your actual model.

    Real options:
    - Conv-TasNet encoder
    - Transformer encoder
    - U-Net encoder
    - HDemucs encoder
    """

    def __init__(self, in_channels: int = 2, encoder_dim: int = 512):
        super().__init__()
        self.encoder_dim = encoder_dim

        # Simple conv encoder (placeholder)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=16, stride=8, padding=4),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, encoder_dim, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) input audio
        Returns:
            features: (B, T', D) encoded features
        """
        # (B, C, T) -> (B, D, T')
        features = self.conv(x)
        # (B, D, T') -> (B, T', D)
        return features.transpose(1, 2)


class PlaceholderDecoder(nn.Module):
    """
    Placeholder decoder - outputs stem masks.
    """

    def __init__(self, encoder_dim: int = 512, n_stems: int = 4):
        super().__init__()
        self.n_stems = n_stems

        self.decoder = nn.Sequential(
            nn.Linear(encoder_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_stems),
            nn.Sigmoid()  # Mask in [0, 1]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T', D) encoder features
        Returns:
            masks: (B, T', n_stems) stem masks
        """
        return self.decoder(features)


class StemSeparationModel(nn.Module):
    """
    Full model with auxiliary heads.
    """

    def __init__(
        self,
        in_channels: int = 2,
        encoder_dim: int = 512,
        n_stems: int = 4,
        aux_config: dict = None
    ):
        super().__init__()

        self.encoder = PlaceholderEncoder(in_channels, encoder_dim)
        self.decoder = PlaceholderDecoder(encoder_dim, n_stems)

        # Auxiliary heads
        aux_config = aux_config or {}
        self.aux_heads = AuxiliaryHeads(
            encoder_dim=encoder_dim,
            enable_ducking=aux_config.get('ducking', True),
            enable_reverb=aux_config.get('reverb', True),
            enable_music_tag=aux_config.get('music_tag', True),
            ducking_weight=aux_config.get('ducking_weight', 0.1),
            reverb_weight=aux_config.get('reverb_weight', 0.05),
            music_tag_weight=aux_config.get('music_tag_weight', 0.05),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, T) input mix
        Returns:
            masks: (B, T', n_stems) stem masks
            aux_outputs: dict of auxiliary predictions
        """
        features = self.encoder(x)
        masks = self.decoder(features)
        aux_outputs = self.aux_heads(features)

        return masks, aux_outputs, features


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_main_loss = 0.0
    total_aux_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        mix = batch['mix'].to(device)
        stems = {k: v.to(device) for k, v in batch['stems'].items()}

        # Forward pass
        masks, aux_outputs, features = model(mix)

        # Main loss (placeholder - replace with your actual loss)
        # For real training, you'd apply masks to mix and compare with stems
        # Here we just use a dummy L1 loss on masks
        main_loss = masks.mean()  # Placeholder!

        # Auxiliary loss
        aux_targets = {}

        # Ducking envelope target
        if 'ducking_envelope' in batch:
            env = batch['ducking_envelope'].to(device)
            # Align envelope length with feature length
            feat_len = features.shape[1]
            if env.shape[1] != feat_len:
                env = torch.nn.functional.interpolate(
                    env.unsqueeze(1), size=feat_len, mode='linear', align_corners=False
                ).squeeze(1)
            aux_targets['ducking_envelope'] = env

        # Reverb target
        if 'reverb_label' in batch:
            aux_targets['reverb_label'] = batch['reverb_label'].to(device)

        # Music tag target
        if 'music_tag' in batch:
            aux_targets['music_tag'] = batch['music_tag'].to(device)

        # Compute auxiliary loss
        aux_loss, aux_loss_dict = model.aux_heads.compute_loss(aux_outputs, aux_targets)

        # Total loss
        loss = main_loss + aux_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        total_loss += loss.item()
        total_main_loss += main_loss.item()
        total_aux_loss += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
        n_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'main': f'{main_loss.item():.4f}',
            'aux': f'{aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss:.4f}'
        })

    return {
        'total_loss': total_loss / n_batches,
        'main_loss': total_main_loss / n_batches,
        'aux_loss': total_aux_loss / n_batches,
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
):
    """Validate model."""
    model.eval()

    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            mix = batch['mix'].to(device)

            masks, aux_outputs, features = model(mix)

            # Placeholder loss
            loss = masks.mean()

            total_loss += loss.item()
            n_batches += 1

    return {'val_loss': total_loss / n_batches}


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train stem separation with auxiliary targets")
    parser.add_argument('data_dir', type=str, help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--chunk-sec', type=float, default=5.0, help='Chunk duration in seconds')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("STEM SEPARATION TRAINING WITH AUXILIARY TARGETS")
    print(f"{'='*60}\n")

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = StemSeparationDataset(
        args.data_dir,
        chunk_duration_sec=args.chunk_sec,
        aux_targets=['ducking_envelope', 'reverb_class', 'music_tag'],
        split='train'
    )

    val_dataset = StemSeparationDataset(
        args.data_dir,
        chunk_duration_sec=args.chunk_sec,
        aux_targets=['ducking_envelope', 'reverb_class', 'music_tag'],
        split='val'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    print("\nCreating model...")
    model = StemSeparationModel(
        in_channels=2,
        encoder_dim=512,
        n_stems=4,
        aux_config={
            'ducking': True,
            'reverb': True,
            'music_tag': True,
            'ducking_weight': 0.1,
            'reverb_weight': 0.05,
            'music_tag_weight': 0.05,
        }
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)

        if len(val_loader) > 0:
            val_metrics = validate(model, val_loader, device)
        else:
            val_metrics = {}

        print(f"\nEpoch {epoch}: train_loss={train_metrics['total_loss']:.4f}, "
              f"main={train_metrics['main_loss']:.4f}, aux={train_metrics['aux_loss']:.4f}")
        if val_metrics:
            print(f"         val_loss={val_metrics['val_loss']:.4f}")
        print("-" * 60)

    print("\nTraining complete!")

    # Save model
    save_path = "stem_sep_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
