"""
Auxiliary Training Targets for Stem Separation Models

This module provides utilities for extracting and using auxiliary training targets
that can improve stem separation model performance through multi-task learning.

Auxiliary targets available:
1. Ducking Envelope - Time-varying gain curve showing speech activity
2. Reverb Classification - Per-stem reverb preset classification
3. Music Tag Classification - Music genre/mood classification

Usage:
    from auxiliary_targets import StemSeparationDataset, AuxiliaryHeads

    dataset = StemSeparationDataset(
        data_dir="/path/to/dataset",
        aux_targets=['ducking_envelope', 'reverb_class', 'music_tag']
    )

    # In your model
    aux_heads = AuxiliaryHeads(encoder_dim=512)
"""

import os
import json
import glob
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Union
from pathlib import Path

# Optional PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[auxiliary_targets] PyTorch not available. Dataset classes disabled.")

# Optional audio imports
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class ReverbType(Enum):
    """Reverb preset classification labels."""
    NONE = 0
    ROOM = 1
    SMALL_ROOM = 2
    MEDIUM_ROOM = 3
    LARGE_HALL = 4
    BATHROOM = 5
    PLATE = 6
    SPRING = 7
    UNKNOWN = 8


class MusicTag(Enum):
    """Music tag classification labels (subset of common tags)."""
    NONE = 0  # No music in scene
    CINEMATIC = 1
    TENSE = 2
    AMBIENT = 3
    DRAMATIC = 4
    PEACEFUL = 5
    SUSPENSE = 6
    EMOTIONAL = 7
    EPIC = 8
    DARK = 9
    UPLIFTING = 10
    MYSTERIOUS = 11
    ACTION = 12
    ROMANTIC = 13
    MELANCHOLIC = 14
    OTHER = 15  # Fallback for unknown tags


# Mapping from string to enum
REVERB_TYPE_MAP = {
    None: ReverbType.NONE,
    "none": ReverbType.NONE,
    "room": ReverbType.ROOM,
    "small_room": ReverbType.SMALL_ROOM,
    "medium_room": ReverbType.MEDIUM_ROOM,
    "large_hall": ReverbType.LARGE_HALL,
    "bathroom": ReverbType.BATHROOM,
    "plate": ReverbType.PLATE,
    "spring": ReverbType.SPRING,
}

MUSIC_TAG_MAP = {
    None: MusicTag.NONE,
    "cinematic": MusicTag.CINEMATIC,
    "tense": MusicTag.TENSE,
    "ambient": MusicTag.AMBIENT,
    "dramatic": MusicTag.DRAMATIC,
    "peaceful": MusicTag.PEACEFUL,
    "suspense": MusicTag.SUSPENSE,
    "emotional": MusicTag.EMOTIONAL,
    "epic": MusicTag.EPIC,
    "dark": MusicTag.DARK,
    "uplifting": MusicTag.UPLIFTING,
    "mysterious": MusicTag.MYSTERIOUS,
    "action": MusicTag.ACTION,
    "romantic": MusicTag.ROMANTIC,
    "melancholic": MusicTag.MELANCHOLIC,
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DuckingEnvelope:
    """Represents a ducking envelope for a single stem."""
    stem_name: str
    sample_rate: int
    duration_sec: float
    keyframes: List[Dict[str, float]]  # [{time, gain, gain_db}, ...]
    ducking_params: Dict[str, float] = field(default_factory=dict)

    def to_array(self, target_length: int, hop_size: int = 512) -> np.ndarray:
        """
        Convert keyframe envelope to fixed-length numpy array.

        Args:
            target_length: Number of frames in output
            hop_size: Samples per frame (for time alignment)

        Returns:
            np.ndarray of shape (target_length,) with gain values [0, 1]
        """
        if not self.keyframes:
            return np.ones(target_length, dtype=np.float32)

        # Create time array for target frames
        frame_times = np.arange(target_length) * hop_size / self.sample_rate

        # Extract keyframe times and gains
        kf_times = np.array([kf['time'] for kf in self.keyframes])
        kf_gains = np.array([kf['gain'] for kf in self.keyframes])

        # Interpolate to target frames
        envelope = np.interp(frame_times, kf_times, kf_gains)

        return envelope.astype(np.float32)

    def to_binary_vad(self, threshold: float = 0.7) -> np.ndarray:
        """
        Convert to binary voice activity detection target.

        Args:
            threshold: Gain below this indicates speech is active

        Returns:
            Binary array where 1 = speech active, 0 = no speech
        """
        # Lower gain = speech present (ducking active)
        envelope = self.to_array(target_length=1000)  # Arbitrary, will be resampled
        return (envelope < threshold).astype(np.float32)


@dataclass
class AuxiliaryTargets:
    """Container for all auxiliary targets for a scene."""
    scene_name: str

    # Ducking envelopes (one per ducked stem)
    ducking_envelopes: Dict[str, DuckingEnvelope] = field(default_factory=dict)

    # Reverb classifications (one per stem)
    reverb_types: Dict[str, ReverbType] = field(default_factory=dict)

    # Music tag
    music_tag: MusicTag = MusicTag.NONE

    # Scene type (for stratified sampling)
    scene_type: str = "unknown"

    # Additional metadata
    duration_sec: float = 0.0
    sample_rate: int = 48000


# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================

def extract_reverb_type(reverb_settings: Optional[Dict]) -> ReverbType:
    """Extract reverb type from settings dict."""
    if not reverb_settings or not isinstance(reverb_settings, dict):
        return ReverbType.NONE

    reverb_type_str = reverb_settings.get('type', '').lower()
    return REVERB_TYPE_MAP.get(reverb_type_str, ReverbType.UNKNOWN)


def extract_music_tag(tag_str: Optional[str]) -> MusicTag:
    """Extract music tag enum from string."""
    if not tag_str:
        return MusicTag.NONE

    tag_lower = tag_str.lower().strip()

    # Direct match
    if tag_lower in MUSIC_TAG_MAP:
        return MUSIC_TAG_MAP[tag_lower]

    # Fuzzy match (tag contains known keyword)
    for keyword, tag_enum in MUSIC_TAG_MAP.items():
        if keyword and keyword in tag_lower:
            return tag_enum

    return MusicTag.OTHER


def load_gain_envelope(envelope_path: str) -> Dict[str, DuckingEnvelope]:
    """
    Load gain envelope JSON and parse into DuckingEnvelope objects.

    Returns:
        Dict mapping stem name to DuckingEnvelope
    """
    if not os.path.exists(envelope_path):
        return {}

    with open(envelope_path, 'r') as f:
        data = json.load(f)

    envelopes = {}
    sample_rate = data.get('sample_rate', 48000)
    duration_sec = data.get('duration_sec', 0)

    for stem_name, env_data in data.get('envelopes', {}).items():
        if not isinstance(env_data, dict):
            continue

        envelopes[stem_name] = DuckingEnvelope(
            stem_name=stem_name,
            sample_rate=sample_rate,
            duration_sec=duration_sec,
            keyframes=env_data.get('keyframes', []),
            ducking_params=env_data.get('ducking_params', {})
        )

    return envelopes


def load_auxiliary_targets(scene_dir: str) -> Optional[AuxiliaryTargets]:
    """
    Load all auxiliary targets from a scene directory.

    Args:
        scene_dir: Path to scene directory containing metadata.json and linear/gain_envelope.json

    Returns:
        AuxiliaryTargets object or None if metadata missing
    """
    metadata_path = os.path.join(scene_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        return None

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    scene_name = os.path.basename(scene_dir)

    # Extract ducking envelopes
    envelope_path = os.path.join(scene_dir, 'linear', 'gain_envelope.json')
    ducking_envelopes = load_gain_envelope(envelope_path)

    # Extract reverb types from critic_adjustments
    reverb_types = {}
    critic_adj = metadata.get('critic_adjustments', {})
    for stem in ['speech', 'music', 'ambience', 'sfx']:
        stem_settings = critic_adj.get(stem, {})
        reverb_settings = stem_settings.get('reverb')
        reverb_types[stem] = extract_reverb_type(reverb_settings)

    # Extract music tag
    scene_plan = metadata.get('scene_plan', {})
    gemini_context = scene_plan.get('gemini_context', {})
    music_tag_str = gemini_context.get('music_tag')
    music_tag = extract_music_tag(music_tag_str)

    # Get scene type and duration
    scene_type = metadata.get('scene_type', 'unknown')

    # Try to get duration from envelope or outputs
    duration_sec = 0.0
    outputs = metadata.get('outputs', {})
    linear_out = outputs.get('linear', {})
    if ducking_envelopes:
        first_env = next(iter(ducking_envelopes.values()))
        duration_sec = first_env.duration_sec

    return AuxiliaryTargets(
        scene_name=scene_name,
        ducking_envelopes=ducking_envelopes,
        reverb_types=reverb_types,
        music_tag=music_tag,
        scene_type=scene_type,
        duration_sec=duration_sec,
        sample_rate=48000
    )


# =============================================================================
# PYTORCH DATASET
# =============================================================================

if TORCH_AVAILABLE:

    class StemSeparationDataset(Dataset):
        """
        PyTorch Dataset for stem separation with auxiliary targets.

        Loads audio chunks and their corresponding auxiliary targets for training.

        Args:
            data_dir: Root directory containing scene subdirectories
            chunk_duration_sec: Duration of audio chunks to load
            chunk_hop_sec: Hop between chunks (for overlap)
            sample_rate: Target sample rate
            aux_targets: List of auxiliary targets to include
                         Options: 'ducking_envelope', 'reverb_class', 'music_tag', 'scene_type'
            split: 'train', 'val', or 'test' (uses hash-based splitting)
            split_ratio: Tuple of (train, val, test) ratios
        """

        def __init__(
            self,
            data_dir: str,
            chunk_duration_sec: float = 5.0,
            chunk_hop_sec: float = 2.5,
            sample_rate: int = 48000,
            aux_targets: List[str] = None,
            split: str = 'train',
            split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
            use_linear: bool = True,  # Use linear stems (sum = mix) vs release
        ):
            self.data_dir = Path(data_dir)
            self.chunk_duration_sec = chunk_duration_sec
            self.chunk_hop_sec = chunk_hop_sec
            self.sample_rate = sample_rate
            self.aux_targets = aux_targets or ['ducking_envelope']
            self.split = split
            self.split_ratio = split_ratio
            self.use_linear = use_linear

            self.chunk_samples = int(chunk_duration_sec * sample_rate)
            self.hop_samples = int(chunk_hop_sec * sample_rate)

            # Scan for valid scenes
            self.scenes = self._scan_scenes()

            # Build chunk index
            self.chunks = self._build_chunk_index()

            print(f"[Dataset] {split}: {len(self.scenes)} scenes, {len(self.chunks)} chunks")

        def _scene_to_split(self, scene_name: str) -> str:
            """Deterministic hash-based split assignment."""
            h = hash(scene_name) % 1000 / 1000.0
            train_end = self.split_ratio[0]
            val_end = train_end + self.split_ratio[1]

            if h < train_end:
                return 'train'
            elif h < val_end:
                return 'val'
            else:
                return 'test'

        def _scan_scenes(self) -> List[Dict]:
            """Scan data directory for valid scenes."""
            scenes = []

            for scene_dir in sorted(self.data_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue

                metadata_path = scene_dir / 'metadata.json'
                if not metadata_path.exists():
                    continue

                scene_name = scene_dir.name

                # Check split
                if self._scene_to_split(scene_name) != self.split:
                    continue

                # Check for required files
                stem_dir = 'linear' if self.use_linear else 'release'
                mix_path = scene_dir / stem_dir / 'mix.wav'
                if not mix_path.exists():
                    continue

                # Load auxiliary targets
                aux = load_auxiliary_targets(str(scene_dir))
                if aux is None:
                    continue

                scenes.append({
                    'name': scene_name,
                    'dir': scene_dir,
                    'stem_dir': stem_dir,
                    'aux_targets': aux,
                    'duration_sec': aux.duration_sec
                })

            return scenes

        def _build_chunk_index(self) -> List[Dict]:
            """Build index of all chunks across all scenes."""
            chunks = []

            for scene in self.scenes:
                duration = scene['duration_sec']
                if duration <= 0:
                    continue

                # Calculate number of chunks
                total_samples = int(duration * self.sample_rate)

                offset = 0
                while offset + self.chunk_samples <= total_samples:
                    chunks.append({
                        'scene_idx': len(chunks),
                        'scene': scene,
                        'offset_samples': offset,
                        'offset_sec': offset / self.sample_rate
                    })
                    offset += self.hop_samples

            return chunks

        def __len__(self) -> int:
            return len(self.chunks)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            chunk_info = self.chunks[idx]
            scene = chunk_info['scene']
            offset = chunk_info['offset_samples']

            stem_dir = scene['dir'] / scene['stem_dir']

            # Load audio stems
            stems = {}
            for stem_name in ['speech', 'music', 'ambience', 'sfx']:
                stem_path = stem_dir / f'{stem_name}.wav'
                if stem_path.exists():
                    audio, sr = sf.read(str(stem_path), start=offset, stop=offset + self.chunk_samples)
                    if len(audio.shape) == 1:
                        audio = audio[np.newaxis, :]  # (1, T)
                    else:
                        audio = audio.T  # (C, T)
                    stems[stem_name] = torch.from_numpy(audio.astype(np.float32))
                else:
                    stems[stem_name] = torch.zeros(1, self.chunk_samples)

            # Load mix
            mix_path = stem_dir / 'mix.wav'
            mix_audio, _ = sf.read(str(mix_path), start=offset, stop=offset + self.chunk_samples)
            if len(mix_audio.shape) == 1:
                mix_audio = mix_audio[np.newaxis, :]
            else:
                mix_audio = mix_audio.T
            mix = torch.from_numpy(mix_audio.astype(np.float32))

            # Build output dict
            output = {
                'mix': mix,
                'stems': stems,
                'scene_name': scene['name'],
            }

            # Add auxiliary targets
            aux = scene['aux_targets']

            if 'ducking_envelope' in self.aux_targets:
                # Get envelope for this chunk (frame-level)
                hop_size = 512  # Standard STFT hop
                n_frames = self.chunk_samples // hop_size

                # Use combined envelope (prefer music, fallback to sfx, else ones)
                combined_env = np.ones(n_frames, dtype=np.float32)

                for stem_name in ['music', 'sfx']:
                    if stem_name in aux.ducking_envelopes:
                        env = aux.ducking_envelopes[stem_name]
                        full_envelope = env.to_array(
                            target_length=int(aux.duration_sec * self.sample_rate / hop_size),
                            hop_size=hop_size
                        )
                        # Extract chunk
                        start_frame = offset // hop_size
                        end_frame = start_frame + n_frames
                        if end_frame <= len(full_envelope):
                            combined_env = full_envelope[start_frame:end_frame]
                        break  # Use first available

                output['ducking_envelope'] = torch.from_numpy(combined_env)

            if 'reverb_class' in self.aux_targets:
                # Use speech reverb as the classification target (most consistent)
                speech_reverb = aux.reverb_types.get('speech', ReverbType.NONE)
                output['reverb_label'] = torch.tensor(speech_reverb.value, dtype=torch.long)

            if 'music_tag' in self.aux_targets:
                output['music_tag'] = torch.tensor(aux.music_tag.value, dtype=torch.long)

            if 'scene_type' in self.aux_targets:
                # Convert scene type to index
                scene_types = ['dialogue_heavy', 'music_montage', 'ambient', 'action_sfx', 'sparse', 'balanced']
                try:
                    scene_type_idx = scene_types.index(aux.scene_type)
                except ValueError:
                    scene_type_idx = 0
                output['scene_type'] = torch.tensor(scene_type_idx, dtype=torch.long)

            return output


# =============================================================================
# AUXILIARY HEAD ARCHITECTURES
# =============================================================================

if TORCH_AVAILABLE:

    class DuckingEnvelopeHead(nn.Module):
        """
        Auxiliary head for predicting ducking envelope from encoder features.

        Input: (B, T, D) encoder features
        Output: (B, T) gain envelope [0, 1]
        """

        def __init__(self, encoder_dim: int = 512, hidden_dim: int = 256):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(encoder_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()  # Output in [0, 1]
            )

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            """
            Args:
                features: (B, T, D) encoder features
            Returns:
                envelope: (B, T) predicted gain envelope
            """
            return self.proj(features).squeeze(-1)


    class ReverbClassificationHead(nn.Module):
        """
        Auxiliary head for classifying reverb type from encoder features.

        Uses global pooling for scene-level classification.
        """

        def __init__(self, encoder_dim: int = 512, num_classes: int = 9):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Sequential(
                nn.Linear(encoder_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            """
            Args:
                features: (B, T, D) encoder features
            Returns:
                logits: (B, num_classes) classification logits
            """
            # (B, T, D) -> (B, D, T) -> (B, D, 1) -> (B, D)
            pooled = self.pool(features.transpose(1, 2)).squeeze(-1)
            return self.classifier(pooled)


    class MusicTagHead(nn.Module):
        """
        Auxiliary head for classifying music tag from encoder features.
        """

        def __init__(self, encoder_dim: int = 512, num_classes: int = 16):
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(encoder_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )
            self.classifier = nn.Sequential(
                nn.Linear(encoder_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            """
            Args:
                features: (B, T, D) encoder features
            Returns:
                logits: (B, num_classes) classification logits
            """
            # Attention-weighted pooling
            attn_weights = F.softmax(self.attention(features), dim=1)  # (B, T, 1)
            pooled = (features * attn_weights).sum(dim=1)  # (B, D)
            return self.classifier(pooled)


    class AuxiliaryHeads(nn.Module):
        """
        Combined auxiliary heads for multi-task learning.

        Usage:
            aux_heads = AuxiliaryHeads(encoder_dim=512)
            aux_outputs = aux_heads(encoder_features)
            aux_loss = aux_heads.compute_loss(aux_outputs, targets)
        """

        def __init__(
            self,
            encoder_dim: int = 512,
            enable_ducking: bool = True,
            enable_reverb: bool = True,
            enable_music_tag: bool = True,
            ducking_weight: float = 0.1,
            reverb_weight: float = 0.05,
            music_tag_weight: float = 0.05,
        ):
            super().__init__()

            self.enable_ducking = enable_ducking
            self.enable_reverb = enable_reverb
            self.enable_music_tag = enable_music_tag

            self.ducking_weight = ducking_weight
            self.reverb_weight = reverb_weight
            self.music_tag_weight = music_tag_weight

            if enable_ducking:
                self.ducking_head = DuckingEnvelopeHead(encoder_dim)

            if enable_reverb:
                self.reverb_head = ReverbClassificationHead(encoder_dim, num_classes=len(ReverbType))

            if enable_music_tag:
                self.music_tag_head = MusicTagHead(encoder_dim, num_classes=len(MusicTag))

        def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            Args:
                features: (B, T, D) encoder features
            Returns:
                Dict with predictions for each enabled head
            """
            outputs = {}

            if self.enable_ducking:
                outputs['ducking_envelope'] = self.ducking_head(features)

            if self.enable_reverb:
                outputs['reverb_logits'] = self.reverb_head(features)

            if self.enable_music_tag:
                outputs['music_tag_logits'] = self.music_tag_head(features)

            return outputs

        def compute_loss(
            self,
            predictions: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor]
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
            """
            Compute weighted auxiliary loss.

            Args:
                predictions: Output from forward()
                targets: Dict with 'ducking_envelope', 'reverb_label', 'music_tag'

            Returns:
                total_loss: Weighted sum of all auxiliary losses
                loss_dict: Individual losses for logging
            """
            total_loss = 0.0
            loss_dict = {}

            if self.enable_ducking and 'ducking_envelope' in predictions and 'ducking_envelope' in targets:
                ducking_loss = F.mse_loss(predictions['ducking_envelope'], targets['ducking_envelope'])
                total_loss = total_loss + self.ducking_weight * ducking_loss
                loss_dict['aux_ducking'] = ducking_loss.item()

            if self.enable_reverb and 'reverb_logits' in predictions and 'reverb_label' in targets:
                reverb_loss = F.cross_entropy(predictions['reverb_logits'], targets['reverb_label'])
                total_loss = total_loss + self.reverb_weight * reverb_loss
                loss_dict['aux_reverb'] = reverb_loss.item()

            if self.enable_music_tag and 'music_tag_logits' in predictions and 'music_tag' in targets:
                music_loss = F.cross_entropy(predictions['music_tag_logits'], targets['music_tag'])
                total_loss = total_loss + self.music_tag_weight * music_loss
                loss_dict['aux_music_tag'] = music_loss.item()

            return total_loss, loss_dict


# =============================================================================
# UTILITIES
# =============================================================================

def analyze_dataset_aux_targets(data_dir: str) -> Dict[str, Any]:
    """
    Analyze auxiliary target distribution across dataset.

    Returns statistics useful for:
    - Class balancing
    - Sanity checking
    - Understanding data composition
    """
    data_dir = Path(data_dir)

    stats = {
        'total_scenes': 0,
        'scenes_with_ducking': 0,
        'reverb_distribution': {rt.name: 0 for rt in ReverbType},
        'music_tag_distribution': {mt.name: 0 for mt in MusicTag},
        'scene_type_distribution': {},
        'ducking_enabled': {'music': 0, 'sfx': 0},
        'total_duration_hours': 0.0,
    }

    for scene_dir in sorted(data_dir.iterdir()):
        if not scene_dir.is_dir():
            continue

        aux = load_auxiliary_targets(str(scene_dir))
        if aux is None:
            continue

        stats['total_scenes'] += 1
        stats['total_duration_hours'] += aux.duration_sec / 3600.0

        # Ducking
        if aux.ducking_envelopes:
            stats['scenes_with_ducking'] += 1
            for stem in ['music', 'sfx']:
                if stem in aux.ducking_envelopes:
                    stats['ducking_enabled'][stem] += 1

        # Reverb
        for stem, reverb_type in aux.reverb_types.items():
            stats['reverb_distribution'][reverb_type.name] += 1

        # Music tag
        stats['music_tag_distribution'][aux.music_tag.name] += 1

        # Scene type
        st = aux.scene_type
        stats['scene_type_distribution'][st] = stats['scene_type_distribution'].get(st, 0) + 1

    return stats


def print_dataset_stats(stats: Dict[str, Any]):
    """Pretty print dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET AUXILIARY TARGET STATISTICS")
    print("=" * 60)

    print(f"\nTotal Scenes: {stats['total_scenes']}")
    print(f"Total Duration: {stats['total_duration_hours']:.2f} hours")
    print(f"Scenes with Ducking: {stats['scenes_with_ducking']}")

    print("\n--- Ducking Enabled ---")
    for stem, count in stats['ducking_enabled'].items():
        print(f"  {stem}: {count}")

    print("\n--- Reverb Distribution ---")
    for reverb_type, count in sorted(stats['reverb_distribution'].items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {reverb_type}: {count}")

    print("\n--- Music Tag Distribution ---")
    for tag, count in sorted(stats['music_tag_distribution'].items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {tag}: {count}")

    print("\n--- Scene Type Distribution ---")
    for scene_type, count in sorted(stats['scene_type_distribution'].items(), key=lambda x: -x[1]):
        print(f"  {scene_type}: {count}")

    print("=" * 60 + "\n")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze auxiliary training targets in dataset")
    parser.add_argument('data_dir', type=str, help='Path to dataset directory')
    parser.add_argument('--test-load', action='store_true', help='Test loading a few samples')

    args = parser.parse_args()

    # Analyze dataset
    print(f"Analyzing {args.data_dir}...")
    stats = analyze_dataset_aux_targets(args.data_dir)
    print_dataset_stats(stats)

    # Test loading
    if args.test_load and TORCH_AVAILABLE:
        print("\nTesting dataset loading...")
        dataset = StemSeparationDataset(
            args.data_dir,
            aux_targets=['ducking_envelope', 'reverb_class', 'music_tag'],
            split='train'
        )

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample keys: {list(sample.keys())}")
            print(f"Mix shape: {sample['mix'].shape}")
            print(f"Stem shapes: {[(k, v.shape) for k, v in sample['stems'].items()]}")
            if 'ducking_envelope' in sample:
                print(f"Ducking envelope shape: {sample['ducking_envelope'].shape}")
            if 'reverb_label' in sample:
                print(f"Reverb label: {sample['reverb_label'].item()} ({ReverbType(sample['reverb_label'].item()).name})")
            if 'music_tag' in sample:
                print(f"Music tag: {sample['music_tag'].item()} ({MusicTag(sample['music_tag'].item()).name})")
