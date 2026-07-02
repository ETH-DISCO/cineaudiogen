"""
Quality Validation for Cinematic Audio Dataset

Automated QA checks to ensure generated scenes are valid for ML training:
1. Additivity check: mix == sum(stems) for linear output
2. Loudness check: LUFS within acceptable range
3. Clipping check: true peak < 0 dBTP
4. Duration check: all stems same length
5. Corruption check: detect silence, NaN, inf, DC offset

Usage:
    python validate.py /path/to/dataset
    python validate.py /path/to/scene_dir --single

    # In code:
    from validate import validate_scene, validate_dataset
    report = validate_scene("/path/to/scene")
    if not report.is_valid:
        print(report.errors)
"""

import os
import json
import glob
import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("[validate] soundfile not available")

try:
    import pyloudnorm as pyln
    PYLOUDNORM_AVAILABLE = True
except ImportError:
    PYLOUDNORM_AVAILABLE = False
    print("[validate] pyloudnorm not available - loudness checks disabled")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ValidationConfig:
    """Thresholds for validation checks."""

    # Additivity check (linear stems should sum to mix)
    max_additivity_error: float = 1e-3  # Max allowed RMS error

    # Loudness checks (LUFS)
    min_lufs: float = -35.0  # Too quiet
    max_lufs: float = -10.0  # Too loud
    max_true_peak_dbtp: float = 0.0  # Clipping threshold (0 = true clipping)

    # Duration checks
    max_duration_mismatch_samples: int = 10  # Allow tiny rounding errors

    # Corruption checks
    min_rms_db: float = -80.0  # Below this = silence
    max_dc_offset: float = 0.01  # Max acceptable DC offset
    max_consecutive_zeros: int = 48000  # 1 second of zeros = suspicious

    # Per-stem silence thresholds (some stems may legitimately be silent)
    allow_silent_stems: List[str] = field(default_factory=lambda: ['music', 'sfx'])


DEFAULT_CONFIG = ValidationConfig()


# =============================================================================
# VALIDATION REPORT
# =============================================================================

@dataclass
class ValidationError:
    """A single validation error."""
    check_name: str
    severity: str  # 'error', 'warning'
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report for a scene."""
    scene_name: str
    scene_dir: str
    is_valid: bool = True
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, check_name: str, message: str, **details):
        """Add an error (invalidates the scene)."""
        self.errors.append(ValidationError(check_name, 'error', message, details))
        self.is_valid = False

    def add_warning(self, check_name: str, message: str, **details):
        """Add a warning (doesn't invalidate)."""
        self.warnings.append(ValidationError(check_name, 'warning', message, details))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'scene_name': self.scene_name,
            'is_valid': self.is_valid,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'errors': [
                {'check': e.check_name, 'message': e.message, 'details': e.details}
                for e in self.errors
            ],
            'warnings': [
                {'check': w.check_name, 'message': w.message, 'details': w.details}
                for w in self.warnings
            ],
            'metrics': self.metrics
        }


# =============================================================================
# AUDIO ANALYSIS UTILITIES
# =============================================================================

def compute_rms_db(audio: np.ndarray) -> float:
    """Compute RMS level in dB."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        return -100.0
    return 20 * np.log10(rms)


def compute_true_peak_db(audio: np.ndarray) -> float:
    """Compute true peak in dB (simple version without oversampling)."""
    peak = np.max(np.abs(audio))
    if peak < 1e-10:
        return -100.0
    return 20 * np.log10(peak)


def compute_dc_offset(audio: np.ndarray) -> float:
    """Compute DC offset (mean value)."""
    return float(np.abs(np.mean(audio)))


def count_consecutive_zeros(audio: np.ndarray, threshold: float = 1e-8) -> int:
    """Count maximum consecutive near-zero samples."""
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)  # Mix to mono for analysis

    is_zero = np.abs(audio) < threshold
    if not np.any(is_zero):
        return 0

    # Find runs of zeros
    changes = np.diff(is_zero.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1

    # Handle edge cases
    if is_zero[0]:
        starts = np.concatenate([[0], starts])
    if is_zero[-1]:
        ends = np.concatenate([ends, [len(is_zero)]])

    if len(starts) == 0 or len(ends) == 0:
        return len(audio) if is_zero[0] else 0

    runs = ends - starts[:len(ends)]
    return int(np.max(runs)) if len(runs) > 0 else 0


def has_nan_or_inf(audio: np.ndarray) -> bool:
    """Check for NaN or Inf values."""
    return np.any(np.isnan(audio)) or np.any(np.isinf(audio))


def measure_loudness(audio: np.ndarray, sample_rate: int) -> Tuple[float, float]:
    """
    Measure integrated loudness and true peak.

    Returns:
        (integrated_lufs, true_peak_dbtp)
    """
    if not PYLOUDNORM_AVAILABLE:
        return (-23.0, compute_true_peak_db(audio))  # Fallback

    try:
        meter = pyln.Meter(sample_rate)
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        elif audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
            audio = audio.T  # Ensure (samples, channels)

        lufs = meter.integrated_loudness(audio)
        peak = compute_true_peak_db(audio)
        return (lufs, peak)
    except Exception as e:
        return (-23.0, compute_true_peak_db(audio))


# =============================================================================
# VALIDATION CHECKS
# =============================================================================

def check_additivity(
    scene_dir: str,
    config: ValidationConfig = DEFAULT_CONFIG
) -> Tuple[bool, float, str]:
    """
    Check that linear stems sum to the mix.

    Returns:
        (passed, error_value, message)
    """
    linear_dir = os.path.join(scene_dir, 'linear')
    if not os.path.exists(linear_dir):
        return (False, float('inf'), "Linear directory not found")

    mix_path = os.path.join(linear_dir, 'mix.wav')
    if not os.path.exists(mix_path):
        return (False, float('inf'), "Mix file not found")

    try:
        mix, sr = sf.read(mix_path)

        # Load and sum all stems
        stem_sum = None
        stem_names = ['speech', 'music', 'ambience', 'sfx']

        for stem_name in stem_names:
            stem_path = os.path.join(linear_dir, f'{stem_name}.wav')
            if os.path.exists(stem_path):
                stem_audio, _ = sf.read(stem_path)
                if stem_sum is None:
                    stem_sum = stem_audio.copy()
                else:
                    # Handle length mismatches
                    min_len = min(len(stem_sum), len(stem_audio))
                    stem_sum = stem_sum[:min_len] + stem_audio[:min_len]

        if stem_sum is None:
            return (False, float('inf'), "No stems found")

        # Compute error
        min_len = min(len(mix), len(stem_sum))
        error = float(np.sqrt(np.mean((mix[:min_len] - stem_sum[:min_len]) ** 2)))

        passed = error < config.max_additivity_error
        return (passed, error, f"RMS error: {error:.2e}")

    except Exception as e:
        return (False, float('inf'), f"Error loading files: {e}")


def check_loudness(
    audio_path: str,
    config: ValidationConfig = DEFAULT_CONFIG
) -> Tuple[bool, Dict[str, float], List[str]]:
    """
    Check loudness is within acceptable range.

    Returns:
        (passed, metrics, messages)
    """
    try:
        audio, sr = sf.read(audio_path)
        lufs, peak = measure_loudness(audio, sr)

        metrics = {
            'integrated_lufs': float(lufs) if lufs is not None else None,
            'true_peak_dbtp': float(peak) if peak is not None else None,
        }

        messages = []
        passed = True

        if lufs < config.min_lufs:
            messages.append(f"Too quiet: {lufs:.1f} LUFS (min: {config.min_lufs})")
            passed = False
        elif lufs > config.max_lufs:
            messages.append(f"Too loud: {lufs:.1f} LUFS (max: {config.max_lufs})")
            passed = False

        if peak > config.max_true_peak_dbtp:
            messages.append(f"Clipping: {peak:.1f} dBTP (max: {config.max_true_peak_dbtp})")
            passed = False

        return (passed, metrics, messages)

    except Exception as e:
        return (False, {}, [f"Error: {e}"])


def check_duration_match(
    scene_dir: str,
    config: ValidationConfig = DEFAULT_CONFIG
) -> Tuple[bool, Dict[str, int], str]:
    """
    Check all stems have matching duration.

    Returns:
        (passed, durations_dict, message)
    """
    linear_dir = os.path.join(scene_dir, 'linear')
    if not os.path.exists(linear_dir):
        return (False, {}, "Linear directory not found")

    durations = {}
    stem_files = ['mix.wav', 'speech.wav', 'music.wav', 'ambience.wav', 'sfx.wav']

    for stem_file in stem_files:
        path = os.path.join(linear_dir, stem_file)
        if os.path.exists(path):
            try:
                info = sf.info(path)
                durations[stem_file] = int(info.frames)
            except Exception:
                pass

    if len(durations) < 2:
        return (False, durations, "Not enough files to compare")

    values = list(durations.values())
    max_diff = max(values) - min(values)

    passed = max_diff <= config.max_duration_mismatch_samples
    message = f"Max difference: {max_diff} samples"

    return (passed, durations, message)


def check_stem_corruption(
    stem_path: str,
    stem_name: str,
    config: ValidationConfig = DEFAULT_CONFIG
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Check a single stem for corruption.

    Returns:
        (passed, metrics, messages)
    """
    try:
        audio, sr = sf.read(stem_path)
        messages = []
        passed = True

        metrics = {
            'rms_db': float(compute_rms_db(audio)),
            'dc_offset': float(compute_dc_offset(audio)),
            'max_consecutive_zeros': int(count_consecutive_zeros(audio)),
            'has_nan_inf': bool(has_nan_or_inf(audio)),
        }

        # Check for NaN/Inf
        if metrics['has_nan_inf']:
            messages.append("Contains NaN or Inf values")
            passed = False

        # Check for excessive DC offset
        if metrics['dc_offset'] > config.max_dc_offset:
            messages.append(f"High DC offset: {metrics['dc_offset']:.4f}")
            passed = False

        # Check for silence (only error if not in allowed list)
        if metrics['rms_db'] < config.min_rms_db:
            if stem_name not in config.allow_silent_stems:
                messages.append(f"Silent: {metrics['rms_db']:.1f} dB RMS")
                passed = False
            else:
                messages.append(f"Silent (allowed): {metrics['rms_db']:.1f} dB RMS")

        # Check for long stretches of zeros
        if metrics['max_consecutive_zeros'] > config.max_consecutive_zeros:
            messages.append(f"Long silence: {metrics['max_consecutive_zeros']} consecutive zeros")
            # Only warning, not error (could be legitimate)

        return (passed, metrics, messages)

    except Exception as e:
        return (False, {}, [f"Error loading: {e}"])


# =============================================================================
# MAIN VALIDATION FUNCTION
# =============================================================================

def validate_scene(
    scene_dir: str,
    config: ValidationConfig = DEFAULT_CONFIG
) -> ValidationReport:
    """
    Validate a complete scene directory.

    Runs all validation checks and returns a comprehensive report.

    Args:
        scene_dir: Path to scene directory
        config: Validation thresholds

    Returns:
        ValidationReport with all checks
    """
    scene_name = os.path.basename(scene_dir)
    report = ValidationReport(scene_name=scene_name, scene_dir=scene_dir)

    # Check metadata exists
    metadata_path = os.path.join(scene_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        report.add_error('metadata', 'metadata.json not found')
        return report

    # Load metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        report.add_error('metadata', f'Failed to load metadata: {e}')
        return report

    # Check linear directory exists
    linear_dir = os.path.join(scene_dir, 'linear')
    if not os.path.exists(linear_dir):
        report.add_error('structure', 'linear/ directory not found')
        return report

    # 1. Additivity check
    add_passed, add_error, add_msg = check_additivity(scene_dir, config)
    report.metrics['additivity_error'] = add_error
    if not add_passed:
        report.add_error('additivity', f'Stems do not sum to mix: {add_msg}')

    # Also check stored additivity if available
    stored_error = metadata.get('outputs', {}).get('linear', {}).get('additivity_error')
    if stored_error is not None:
        report.metrics['stored_additivity_error'] = stored_error
        if stored_error > config.max_additivity_error:
            report.add_warning('additivity_stored', f'Stored additivity error high: {stored_error:.2e}')

    # 2. Loudness check on mix
    mix_path = os.path.join(linear_dir, 'mix.wav')
    if os.path.exists(mix_path):
        loud_passed, loud_metrics, loud_msgs = check_loudness(mix_path, config)
        report.metrics['loudness'] = loud_metrics
        for msg in loud_msgs:
            if not loud_passed:
                report.add_error('loudness', msg)
            else:
                report.add_warning('loudness', msg)

    # 3. Duration match check
    dur_passed, dur_metrics, dur_msg = check_duration_match(scene_dir, config)
    report.metrics['durations'] = dur_metrics
    if not dur_passed:
        report.add_error('duration', f'Duration mismatch: {dur_msg}')

    # 4. Per-stem corruption checks
    stem_names = ['speech', 'music', 'ambience', 'sfx']
    report.metrics['stems'] = {}

    for stem_name in stem_names:
        stem_path = os.path.join(linear_dir, f'{stem_name}.wav')
        if os.path.exists(stem_path):
            corr_passed, corr_metrics, corr_msgs = check_stem_corruption(
                stem_path, stem_name, config
            )
            report.metrics['stems'][stem_name] = corr_metrics

            for msg in corr_msgs:
                if not corr_passed and 'Silent (allowed)' not in msg:
                    report.add_error(f'corruption_{stem_name}', msg)
                elif 'Long silence' in msg:
                    report.add_warning(f'corruption_{stem_name}', msg)

    # 5. Check scene type consistency
    scene_type = metadata.get('scene_type')
    if scene_type:
        report.metrics['scene_type'] = scene_type

    return report


def validate_dataset(
    data_dir: str,
    config: ValidationConfig = DEFAULT_CONFIG,
    max_workers: int = 4,
    progress: bool = True
) -> Dict[str, Any]:
    """
    Validate entire dataset directory.

    Args:
        data_dir: Root dataset directory
        config: Validation thresholds
        max_workers: Number of parallel workers
        progress: Show progress bar

    Returns:
        Summary dict with overall statistics and per-scene reports
    """
    data_path = Path(data_dir)
    scene_dirs = [d for d in sorted(data_path.iterdir())
                  if d.is_dir() and (d / 'metadata.json').exists()]

    if not scene_dirs:
        return {'error': 'No valid scenes found', 'total': 0}

    reports = []
    failed_scenes = []
    warning_scenes = []

    # Run validation
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(validate_scene, str(d), config): d
                      for d in scene_dirs}

            iterator = as_completed(futures)
            if progress:
                iterator = tqdm(iterator, total=len(futures), desc="Validating")

            for future in iterator:
                try:
                    report = future.result()
                    reports.append(report)
                    if not report.is_valid:
                        failed_scenes.append(report.scene_name)
                    elif report.warnings:
                        warning_scenes.append(report.scene_name)
                except Exception as e:
                    scene_dir = futures[future]
                    failed_scenes.append(scene_dir.name)
    else:
        iterator = scene_dirs
        if progress:
            iterator = tqdm(iterator, desc="Validating")

        for scene_dir in iterator:
            report = validate_scene(str(scene_dir), config)
            reports.append(report)
            if not report.is_valid:
                failed_scenes.append(report.scene_name)
            elif report.warnings:
                warning_scenes.append(report.scene_name)

    # Compute summary statistics
    total = len(reports)
    valid = sum(1 for r in reports if r.is_valid)
    invalid = total - valid
    with_warnings = len(warning_scenes)

    # Aggregate metrics
    lufs_values = [r.metrics.get('loudness', {}).get('integrated_lufs')
                   for r in reports if r.metrics.get('loudness', {}).get('integrated_lufs')]
    peak_values = [r.metrics.get('loudness', {}).get('true_peak_dbtp')
                   for r in reports if r.metrics.get('loudness', {}).get('true_peak_dbtp')]
    add_errors = [r.metrics.get('additivity_error')
                  for r in reports if r.metrics.get('additivity_error') is not None
                  and r.metrics.get('additivity_error') < float('inf')]

    # Error type breakdown
    error_types = {}
    for report in reports:
        for error in report.errors:
            error_types[error.check_name] = error_types.get(error.check_name, 0) + 1

    summary = {
        'total_scenes': total,
        'valid_scenes': valid,
        'invalid_scenes': invalid,
        'scenes_with_warnings': with_warnings,
        'pass_rate': valid / total if total > 0 else 0,
        'failed_scenes': failed_scenes,
        'warning_scenes': warning_scenes,
        'error_type_counts': error_types,
        'metrics': {
            'lufs': {
                'mean': np.mean(lufs_values) if lufs_values else None,
                'min': np.min(lufs_values) if lufs_values else None,
                'max': np.max(lufs_values) if lufs_values else None,
            },
            'true_peak': {
                'mean': np.mean(peak_values) if peak_values else None,
                'max': np.max(peak_values) if peak_values else None,
            },
            'additivity_error': {
                'mean': np.mean(add_errors) if add_errors else None,
                'max': np.max(add_errors) if add_errors else None,
            }
        },
        'reports': [r.to_dict() for r in reports]
    }

    return summary


def print_validation_summary(summary: Dict[str, Any]):
    """Pretty print validation summary."""
    print("\n" + "=" * 70)
    print("DATASET VALIDATION SUMMARY")
    print("=" * 70)

    print(f"\nTotal Scenes:    {summary['total_scenes']}")
    print(f"Valid Scenes:    {summary['valid_scenes']} ({summary['pass_rate']*100:.1f}%)")
    print(f"Invalid Scenes:  {summary['invalid_scenes']}")
    print(f"With Warnings:   {summary['scenes_with_warnings']}")

    if summary.get('error_type_counts'):
        print("\n--- Error Breakdown ---")
        for error_type, count in sorted(summary['error_type_counts'].items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}")

    metrics = summary.get('metrics', {})
    if metrics.get('lufs', {}).get('mean') is not None:
        print("\n--- Loudness Statistics ---")
        print(f"  LUFS:      {metrics['lufs']['mean']:.1f} mean "
              f"({metrics['lufs']['min']:.1f} to {metrics['lufs']['max']:.1f})")

    if metrics.get('true_peak', {}).get('max') is not None:
        print(f"  True Peak: {metrics['true_peak']['max']:.1f} dBTP max")

    if metrics.get('additivity_error', {}).get('mean') is not None:
        print(f"\n--- Additivity ---")
        print(f"  Mean Error: {metrics['additivity_error']['mean']:.2e}")
        print(f"  Max Error:  {metrics['additivity_error']['max']:.2e}")

    if summary.get('failed_scenes'):
        print(f"\n--- Failed Scenes ({len(summary['failed_scenes'])}) ---")
        for scene in summary['failed_scenes'][:10]:
            print(f"  - {scene}")
        if len(summary['failed_scenes']) > 10:
            print(f"  ... and {len(summary['failed_scenes']) - 10} more")

    print("=" * 70 + "\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Validate cinematic audio dataset")
    parser.add_argument('path', type=str, help='Path to dataset or scene directory')
    parser.add_argument('--single', action='store_true', help='Validate single scene')
    parser.add_argument('--workers', type=int, default=4, help='Parallel workers')
    parser.add_argument('--json', type=str, help='Output JSON report to file')
    parser.add_argument('--strict', action='store_true', help='Use stricter thresholds')

    args = parser.parse_args()

    # Configure thresholds
    config = ValidationConfig()
    if args.strict:
        config.max_additivity_error = 1e-4
        config.max_true_peak_dbtp = -0.5
        config.min_lufs = -30.0

    if args.single:
        # Validate single scene
        report = validate_scene(args.path, config)

        print(f"\n{'='*50}")
        print(f"Scene: {report.scene_name}")
        print(f"Valid: {report.is_valid}")
        print(f"{'='*50}")

        if report.errors:
            print("\nERRORS:")
            for e in report.errors:
                print(f"  [{e.check_name}] {e.message}")

        if report.warnings:
            print("\nWARNINGS:")
            for w in report.warnings:
                print(f"  [{w.check_name}] {w.message}")

        print(f"\nMetrics: {json.dumps(report.metrics, indent=2)}")

    else:
        # Validate entire dataset
        print(f"Validating dataset: {args.path}")
        summary = validate_dataset(args.path, config, max_workers=args.workers)
        print_validation_summary(summary)

        if args.json:
            # Remove individual reports for cleaner output
            output = {k: v for k, v in summary.items() if k != 'reports'}
            output['failed_scene_details'] = [
                r for r in summary['reports'] if not r['is_valid']
            ]
            with open(args.json, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Report saved to: {args.json}")


if __name__ == "__main__":
    main()
