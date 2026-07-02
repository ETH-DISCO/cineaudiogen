import os
import json
import random
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from scipy import signal
from pedalboard import (
    Pedalboard,
    Reverb,
    Compressor,
    HighShelfFilter,
    Gain,
    Limiter,
    HighpassFilter
)
from pedalboard.io import AudioFile

# Perceptual quality metrics
try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False

try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False


# Sidechain-ducking gain envelope: a 1st-order recursive smoother of a per-sample
# target gain, with attack vs release coefficient chosen by a threshold. The
# recurrence is sequential; numba compiles it to a tight loop (fast + operates on the
# array in place, no Python-list blow-up). Falls back to a memory-light chunked pure
# -Python loop if numba is unavailable. Both are bit-identical to a plain numpy loop.
def _ducking_envelope_py(sc, thr, red, ac, rc):
    n = sc.shape[0]
    env = np.empty(n, dtype=np.float32)
    gain = 1.0
    CH = 2_000_000
    for s in range(0, n, CH):
        ab = (sc[s:s + CH] > thr).tolist()
        eb = [0.0] * len(ab)
        for j, a in enumerate(ab):
            if a:
                c = ac; t = red
            else:
                c = rc; t = 1.0
            gain = c * gain + (1.0 - c) * t
            eb[j] = gain
        env[s:s + len(ab)] = eb
    return env


try:
    from numba import njit as _njit

    @_njit(cache=True)
    def _ducking_envelope(sc, thr, red, ac, rc):
        n = sc.shape[0]
        env = np.empty(n, dtype=np.float32)
        gain = 1.0
        for i in range(n):
            if sc[i] > thr:
                c = ac; t = red
            else:
                c = rc; t = 1.0
            gain = c * gain + (1.0 - c) * t
            env[i] = gain
        return env
except Exception:
    _ducking_envelope = _ducking_envelope_py


class AudioEngine:
    def __init__(self, sample_rate=48000):
        self.sr = sample_rate

    def _get_reverb_presets(self):
        """Return all available reverb presets."""
        return {
            # Indoor spaces
            "dry": {"room_size": 0.1, "damping": 0.9, "wet": 0.1},
            "small_room": {"room_size": 0.2, "damping": 0.7, "wet": 0.2},
            "medium_room": {"room_size": 0.4, "damping": 0.5, "wet": 0.3},
            "room": {"room_size": 0.4, "damping": 0.5, "wet": 0.3},  # Alias for medium_room
            "large_room": {"room_size": 0.6, "damping": 0.4, "wet": 0.35},
            "bathroom": {"room_size": 0.25, "damping": 0.8, "wet": 0.4},
            # Large venues
            "large_hall": {"room_size": 0.8, "damping": 0.3, "wet": 0.4},
            "cathedral": {"room_size": 0.95, "damping": 0.2, "wet": 0.5},
            # Industrial
            "garage": {"room_size": 0.5, "damping": 0.6, "wet": 0.35},
            "parking_garage": {"room_size": 0.7, "damping": 0.4, "wet": 0.45},
            "warehouse": {"room_size": 0.75, "damping": 0.35, "wet": 0.4},
            "sewer": {"room_size": 0.9, "damping": 0.1, "wet": 0.6},
            "tunnel": {"room_size": 0.85, "damping": 0.15, "wet": 0.55},
            # Outdoor
            "forest": {"room_size": 0.3, "damping": 0.8, "wet": 0.15},
            "open_field": {"room_size": 0.15, "damping": 0.9, "wet": 0.1},
        }

    def _envelope_to_keyframes(self, envelope, keyframe_rate_hz=100, threshold_db=0.5):
        """
        Convert a sample-rate envelope to sparse keyframes.
        Only stores points where the gain changes significantly.

        Args:
            envelope: numpy array of gain values (one per sample)
            keyframe_rate_hz: Base sampling rate for keyframes (default 100 Hz)
            threshold_db: Minimum change in dB to trigger a new keyframe

        Returns:
            List of keyframes: [{"time": float, "gain": float, "gain_db": float}, ...]
        """
        if envelope is None or len(envelope) == 0:
            return []

        # Downsample to keyframe rate first
        samples_per_keyframe = max(1, int(self.sr / keyframe_rate_hz))
        num_keyframes = len(envelope) // samples_per_keyframe

        keyframes = []
        last_gain_db = None

        for i in range(num_keyframes):
            start_idx = i * samples_per_keyframe
            end_idx = start_idx + samples_per_keyframe
            # Use mean gain in this window
            gain = float(np.mean(envelope[start_idx:end_idx]))
            gain = max(gain, 1e-6)  # Avoid log(0)
            gain_db = 20 * np.log10(gain)
            time_sec = start_idx / self.sr

            # Only add keyframe if gain changed significantly
            if last_gain_db is None or abs(gain_db - last_gain_db) >= threshold_db:
                keyframes.append({
                    "time": round(time_sec, 4),
                    "gain": round(gain, 6),
                    "gain_db": round(gain_db, 2)
                })
                last_gain_db = gain_db

        # Always include final point
        if len(envelope) > 0:
            final_gain = float(envelope[-1])
            final_gain = max(final_gain, 1e-6)
            final_gain_db = 20 * np.log10(final_gain)
            final_time = (len(envelope) - 1) / self.sr
            if not keyframes or keyframes[-1]["time"] < final_time - 0.01:
                keyframes.append({
                    "time": round(final_time, 4),
                    "gain": round(final_gain, 6),
                    "gain_db": round(final_gain_db, 2)
                })

        return keyframes

    def load_audio(self, path, target_duration_sec=None, max_duration_sec=None, force_mono=False):
        """
        Loads audio into a float32 numpy array (Channels, Samples).
        Handles resampling and length matching.

        Args:
            path: Path to audio file
            target_duration_sec: Target duration (trims or loops to fit)
            max_duration_sec: Maximum duration (trims only, no looping)
            force_mono: Downmix to centered mono
        """
        if not path or not os.path.exists(path):
            return None

        # Use Pedalboard's resampling - open file once with target sample rate
        try:
            with AudioFile(path).resampled_to(self.sr) as f:
                audio = f.read(f.frames)

            # Optionally downmix to mono and re-center (dual-mono stereo)
            # Useful for dialogue that is hard-panned L/R.
            if force_mono and audio is not None and audio.ndim == 2:
                if audio.shape[0] >= 2:
                    mono = np.mean(audio[:2, :], axis=0, keepdims=True)
                    audio = np.vstack([mono, mono])
                elif audio.shape[0] == 1:
                    audio = np.vstack([audio, audio])

            # Ensure Stereo (2, N)
            if audio.shape[0] == 1:
                audio = np.vstack([audio, audio]) # Mono to Stereo
            elif audio.shape[0] > 2:
                audio = audio[:2, :] # Drop extra channels

            # Apply max duration limit (trim only, no looping)
            if max_duration_sec is not None and max_duration_sec > 0:
                max_samples = int(max_duration_sec * self.sr)
                if audio.shape[1] > max_samples:
                    audio = audio[:, :max_samples]
                    # Short fade-out (50ms) to avoid clicks
                    fade_len = min(int(0.05 * self.sr), max_samples // 4)
                    if fade_len > 0:
                        fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                        audio[:, -fade_len:] *= fade

            # Match target duration if specified (trim or loop)
            if target_duration_sec is not None and target_duration_sec > 0:
                target_samples = int(target_duration_sec * self.sr)
                current_samples = audio.shape[1]
                
                if current_samples > target_samples:
                    # Trim with a small fade-out at the end
                    audio = audio[:, :target_samples]
                    fade_len = min(int(0.5 * self.sr), target_samples // 4)  # 0.5s or 25% max
                    if fade_len > 0:
                        fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                        audio[:, -fade_len:] *= fade
                        
                elif current_samples < target_samples:
                    # Loop the audio to fill the duration (with crossfade)
                    loops_needed = (target_samples // current_samples) + 1
                    looped = np.tile(audio, (1, loops_needed))
                    audio = looped[:, :target_samples]
                    # Apply fade out at the very end
                    fade_len = min(int(0.5 * self.sr), target_samples // 4)
                    if fade_len > 0:
                        fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                        audio[:, -fade_len:] *= fade

            return audio
            
        except Exception as e:
            print(f"    [Engine Error] Failed to load {path}: {e}")
            return None

    def build_chain(self, settings):
        """
        Translates JSON settings (from Gemini) into a DSP Processing Chain.
        """
        board = Pedalboard()
        
        if not settings:
            return board

        # 1. EQ / Filters
        if "low_cut_hz" in settings:
            board.append(HighpassFilter(cutoff_frequency_hz=float(settings["low_cut_hz"])))
            
        if "high_shelf_db" in settings and "high_shelf_hz" in settings:
            board.append(HighShelfFilter(
                cutoff_frequency_hz=float(settings["high_shelf_hz"]), 
                gain_db=float(settings["high_shelf_db"])
            ))

        # 2. Dynamics (Compression)
        if "compressor" in settings:
            c = settings["compressor"]
            board.append(Compressor(
                threshold_db=c.get("threshold", -20),
                ratio=c.get("ratio", 2.0),
                attack_ms=c.get("attack", 10),
                release_ms=c.get("release", 100)
            ))

        # 3. Reverb (The most critical part for "Worldizing")
        if "reverb" in settings and settings["reverb"] is not None:
            r = settings["reverb"]
            reverb_presets = self._get_reverb_presets()
            reverb_type = r.get("type", "medium_room")
            preset = reverb_presets.get(reverb_type, reverb_presets["medium_room"])
            
            board.append(Reverb(
                room_size=preset["room_size"],
                damping=preset.get("damping", 0.5),
                wet_level=r.get("wet_amount", preset["wet"]),
                dry_level=1.0
            ))

        # 4. Volume / Gain (Last in chain)
        if "gain_db" in settings:
            board.append(Gain(gain_db=float(settings["gain_db"])))

        return board

    def process_stem(self, audio_array, settings):
        """Runs audio through the chain."""
        if audio_array is None: return None
        
        board = self.build_chain(settings)
        # Pedalboard expects (Channels, Samples)
        processed = board(audio_array, self.sr)
        return processed

    def _apply_ducking(self, target_audio, sidechain_audio, threshold_db=-20.0,
                        reduction_db=-12.0, attack_ms=20, release_ms=225,
                        return_envelope=False):
        """
        Apply sidechain ducking: reduce target_audio when sidechain_audio is loud.
        Used to duck music under dialogue.

        Args:
            target_audio: Audio to duck (e.g., music) - shape (2, N)
            sidechain_audio: Audio triggering the duck (e.g., dialogue) - shape (2, N)
            threshold_db: Level above which ducking kicks in
            reduction_db: How much to reduce (negative value)
            attack_ms: How fast ducking engages
            release_ms: How fast ducking releases
            return_envelope: If True, return (audio, envelope) tuple

        Returns:
            If return_envelope is False: processed audio
            If return_envelope is True: (processed audio, gain envelope)
        """
        if target_audio is None or sidechain_audio is None:
            if return_envelope:
                return target_audio, None
            return target_audio

        # Ensure same length
        min_len = min(target_audio.shape[1], sidechain_audio.shape[1])
        target_audio = target_audio[:, :min_len].copy()
        sidechain_audio = sidechain_audio[:, :min_len]

        # Convert to mono envelope for sidechain detection
        sidechain_mono = np.mean(np.abs(sidechain_audio), axis=0)

        # Convert threshold to linear
        threshold_lin = 10 ** (threshold_db / 20.0)
        reduction_lin = 10 ** (reduction_db / 20.0)

        # Calculate attack/release coefficients
        attack_coef = np.exp(-1.0 / (self.sr * attack_ms / 1000.0))
        release_coef = np.exp(-1.0 / (self.sr * release_ms / 1000.0))

        # Generate gain envelope (numba-accelerated recurrence; see _ducking_envelope).
        envelope = _ducking_envelope(
            np.ascontiguousarray(sidechain_mono, dtype=np.float32),
            float(threshold_lin), float(reduction_lin),
            float(attack_coef), float(release_coef),
        )

        # Apply envelope to both channels
        target_audio[0, :] *= envelope
        target_audio[1, :] *= envelope

        if return_envelope:
            return target_audio, envelope
        return target_audio

    def _compute_gain_reduction_envelope(self, input_audio, output_audio):
        """
        Compute gain reduction envelope by comparing input vs output.
        Used to capture compressor/limiter effect for stem reconstruction.

        Args:
            input_audio: Audio before processing - shape (2, N)
            output_audio: Audio after processing - shape (2, N)

        Returns:
            envelope: Gain reduction curve - shape (N,)
        """
        # Use mono sum for envelope calculation (more stable)
        input_mono = np.mean(np.abs(input_audio), axis=0)
        output_mono = np.mean(np.abs(output_audio), axis=0)

        # Avoid division by zero
        epsilon = 1e-10
        envelope = np.where(
            input_mono > epsilon,
            output_mono / (input_mono + epsilon),
            1.0  # No reduction where input is silent
        )

        # Clip to valid range (no expansion expected)
        envelope = np.clip(envelope, 0.0, 1.0).astype(np.float32)
        return envelope

    def _measure_loudness(self, audio):
        """
        Measure integrated loudness (LUFS) and true peak (dBTP).

        Args:
            audio: Audio array - shape (2, N)

        Returns:
            dict: {'loudness_lufs': float, 'true_peak_dbtp': float}
        """
        meter = pyln.Meter(self.sr)
        # pyloudnorm expects (samples, channels)
        audio_t = audio.T

        loudness = meter.integrated_loudness(audio_t)

        # True peak measurement via 4x oversampling (ITU-R BS.1770)
        upsample_factor = 4
        upsampled = signal.resample_poly(audio, upsample_factor, 1, axis=1)
        peak_linear = np.max(np.abs(upsampled))
        true_peak_dbtp = 20 * np.log10(peak_linear + 1e-10)

        return {
            'loudness_lufs': loudness,
            'true_peak_dbtp': true_peak_dbtp
        }

    def measure_perceptual_quality(self, reference, degraded, metric='all'):
        """
        Measure perceptual audio quality using PESQ and/or STOI.

        PESQ: Perceptual Evaluation of Speech Quality (-0.5 to 4.5, higher is better)
        STOI: Short-Time Objective Intelligibility (0 to 1, higher is better)

        Args:
            reference: Reference/clean audio - shape (2, N) or (N,)
            degraded: Degraded/processed audio - shape (2, N) or (N,)
            metric: 'pesq', 'stoi', or 'all'

        Returns:
            dict with perceptual quality metrics
        """
        results = {}

        # Convert to mono if stereo
        if len(reference.shape) > 1 and reference.shape[0] == 2:
            ref_mono = reference.mean(axis=0)
        else:
            ref_mono = reference.flatten()

        if len(degraded.shape) > 1 and degraded.shape[0] == 2:
            deg_mono = degraded.mean(axis=0)
        else:
            deg_mono = degraded.flatten()

        # Ensure same length
        min_len = min(len(ref_mono), len(deg_mono))
        ref_mono = ref_mono[:min_len]
        deg_mono = deg_mono[:min_len]

        # PESQ requires 8kHz or 16kHz
        if PESQ_AVAILABLE and metric in ('pesq', 'all'):
            try:
                # Resample to 16kHz for PESQ
                target_sr = 16000
                if self.sr != target_sr:
                    ref_16k = signal.resample(ref_mono, int(len(ref_mono) * target_sr / self.sr))
                    deg_16k = signal.resample(deg_mono, int(len(deg_mono) * target_sr / self.sr))
                else:
                    ref_16k, deg_16k = ref_mono, deg_mono

                # PESQ score (wideband mode)
                pesq_score = pesq(target_sr, ref_16k, deg_16k, 'wb')
                results['pesq'] = round(pesq_score, 3)
            except Exception as e:
                results['pesq'] = None
                results['pesq_error'] = str(e)

        # STOI works at any sample rate
        if STOI_AVAILABLE and metric in ('stoi', 'all'):
            try:
                stoi_score = stoi(ref_mono, deg_mono, self.sr, extended=False)
                results['stoi'] = round(stoi_score, 4)
            except Exception as e:
                results['stoi'] = None
                results['stoi_error'] = str(e)

        return results

    def measure_ducking_quality(self, speech, music_original, music_ducked):
        """
        Measure perceptual quality impact of ducking on music.

        Compares original music vs ducked music, using speech as reference
        for intelligibility.

        Args:
            speech: Speech/dialogue audio - shape (2, N)
            music_original: Music before ducking - shape (2, N)
            music_ducked: Music after ducking - shape (2, N)

        Returns:
            dict with ducking quality metrics
        """
        results = {}

        # Music degradation (how much did ducking hurt the music)
        music_quality = self.measure_perceptual_quality(music_original, music_ducked)
        results['music_pesq'] = music_quality.get('pesq')
        results['music_stoi'] = music_quality.get('stoi')

        # Speech intelligibility in mix
        if speech is not None:
            # Mix speech with original music
            min_len = min(speech.shape[1], music_original.shape[1])
            mix_original = speech[:, :min_len] + music_original[:, :min_len] * 0.5
            mix_ducked = speech[:, :min_len] + music_ducked[:, :min_len] * 0.5

            # How intelligible is speech in the ducked mix vs original?
            speech_in_original = self.measure_perceptual_quality(speech[:, :min_len], mix_original)
            speech_in_ducked = self.measure_perceptual_quality(speech[:, :min_len], mix_ducked)

            results['speech_stoi_original_mix'] = speech_in_original.get('stoi')
            results['speech_stoi_ducked_mix'] = speech_in_ducked.get('stoi')

            # STOI improvement from ducking (positive = better speech clarity)
            if results['speech_stoi_original_mix'] and results['speech_stoi_ducked_mix']:
                results['stoi_improvement'] = round(
                    results['speech_stoi_ducked_mix'] - results['speech_stoi_original_mix'], 4
                )

        return results

    def analyze_stem(self, audio, name="stem"):
        """
        Comprehensive audio analysis for a stem.

        Returns metrics useful for mixing decisions:
        - Integrated loudness (LUFS)
        - Loudness range (LRA) - dynamic range
        - True peak (dBTP)
        - RMS level (dB)
        - Crest factor (peak-to-RMS ratio in dB)
        - Duration (seconds)

        Args:
            audio: Audio array - shape (2, N) or (N,)
            name: Name of the stem for logging

        Returns:
            dict with audio metrics
        """
        if audio is None:
            return None

        # Ensure stereo
        if audio.ndim == 1:
            audio = np.stack([audio, audio])

        audio_t = audio.T  # pyloudnorm expects (samples, channels)
        duration_sec = audio.shape[1] / self.sr

        meter = pyln.Meter(self.sr)

        # Integrated loudness
        try:
            loudness_lufs = meter.integrated_loudness(audio_t)
        except Exception:
            loudness_lufs = -70.0  # Very quiet / silence

        # Loudness range (LRA) - measures dynamic variation
        try:
            # Block-based loudness range calculation (simplified LRA)
            block_size = int(3.0 * self.sr)  # 3 second blocks
            hop_size = int(1.0 * self.sr)    # 1 second hop

            if audio.shape[1] > block_size:
                block_loudnesses = []
                for start in range(0, audio.shape[1] - block_size, hop_size):
                    block = audio[:, start:start + block_size].T
                    try:
                        block_lufs = meter.integrated_loudness(block)
                        if block_lufs > -70:  # Ignore very quiet blocks
                            block_loudnesses.append(block_lufs)
                    except Exception:
                        pass

                if len(block_loudnesses) >= 2:
                    # LRA is difference between 95th and 10th percentile
                    sorted_lufs = np.sort(block_loudnesses)
                    p10_idx = max(0, int(len(sorted_lufs) * 0.10))
                    p95_idx = min(len(sorted_lufs) - 1, int(len(sorted_lufs) * 0.95))
                    loudness_range_lu = sorted_lufs[p95_idx] - sorted_lufs[p10_idx]
                else:
                    loudness_range_lu = 0.0
            else:
                loudness_range_lu = 0.0
        except Exception:
            loudness_range_lu = 0.0

        # True peak
        upsample_factor = 4
        upsampled = signal.resample_poly(audio, upsample_factor, 1, axis=1)
        peak_linear = np.max(np.abs(upsampled))
        true_peak_dbtp = 20 * np.log10(peak_linear + 1e-10)

        # RMS level
        rms_linear = np.sqrt(np.mean(audio ** 2))
        rms_db = 20 * np.log10(rms_linear + 1e-10)

        # Crest factor (peak-to-RMS ratio) - indicates transient content
        crest_factor_db = true_peak_dbtp - rms_db

        return {
            'name': name,
            'duration_sec': round(duration_sec, 2),
            'loudness_lufs': round(loudness_lufs, 1),
            'loudness_range_lu': round(loudness_range_lu, 1),
            'true_peak_dbtp': round(true_peak_dbtp, 1),
            'rms_db': round(rms_db, 1),
            'crest_factor_db': round(crest_factor_db, 1),
        }

    def _normalize_loudness(self, audio, target_lufs=-27.0, true_peak_ceiling=-2.0,
                            randomize_lu=1.0):
        """
        Normalize audio to target loudness while respecting true peak ceiling.

        Args:
            audio: Audio array - shape (2, N)
            target_lufs: Target integrated loudness in LUFS
            true_peak_ceiling: Maximum true peak in dBTP
            randomize_lu: Random offset range in LU (±)

        Returns:
            tuple: (normalized_audio, loudness_gain_linear, final_measurements)
        """
        # Add randomization to target
        if randomize_lu > 0:
            target_lufs += random.uniform(-randomize_lu, randomize_lu)

        # Measure current loudness
        measurements = self._measure_loudness(audio)
        current_lufs = measurements['loudness_lufs']

        # Handle silent/very quiet audio
        if current_lufs == float('-inf') or np.isnan(current_lufs):
            return audio, 1.0, measurements

        # Calculate gain needed
        gain_db = target_lufs - current_lufs
        gain_linear = 10 ** (gain_db / 20.0)

        # Apply gain
        normalized = audio * gain_linear

        # Check true peak after gain
        post_measurements = self._measure_loudness(normalized)
        if post_measurements['true_peak_dbtp'] > true_peak_ceiling:
            # Need to limit - apply true peak limiter
            limiter = Limiter(threshold_db=true_peak_ceiling)
            normalized = limiter(normalized, self.sr)
            post_measurements = self._measure_loudness(normalized)

        return normalized, gain_linear, post_measurements

    def mix_and_master(self, stems_dict, output_path, events=None,
                       stem_gains=None, duck_music=True):
        """
        Summing engine with per-stem gain, event placement, ducking, and limiting.
        
        Args:
            stems_dict: {'speech': audio_arr, 'music': audio_arr, 'ambience': audio_arr}
            output_path: Where to write the final mix
            events: List of event dicts with 'audio', 'timestamp' (seconds)
            stem_gains: Dict of gain values per stem, e.g. {'music': 0.7, 'ambience': 0.5}
            duck_music: Whether to duck music under dialogue
        """
        # Default stem gains (dialogue loudest, music/ambience lower)
        default_gains = {
            'speech': 1.0,
            'dialogue': 1.0,
            'music': 0.35,      # Music sits behind dialogue
            'ambience': 0.25,   # Ambience is subtle
            'sfx': 0.8,         # SFX prominent but not overpowering
        }
        if stem_gains:
            default_gains.update(stem_gains)
        
        # 1. Determine max length from stems
        max_len = 0
        for name, audio in stems_dict.items():
            if audio is not None:
                max_len = max(max_len, audio.shape[1])
        
        # Account for events that might extend beyond stems
        if events:
            for ev in events:
                if ev.get('audio') is not None:
                    ts_samples = int(ev.get('timestamp', 0) * self.sr)
                    ev_len = ev['audio'].shape[1]
                    max_len = max(max_len, ts_samples + ev_len)
        
        if max_len == 0: 
            return False

        # 2. Create blank canvas
        mix_bus = np.zeros((2, max_len), dtype=np.float32)
        
        # 3. Get dialogue for ducking reference (before mixing)
        dialogue_audio = stems_dict.get('speech')
        if dialogue_audio is None:
            dialogue_audio = stems_dict.get('dialogue')
        
        # 4. Summing Loop for main stems
        for name, audio in stems_dict.items():
            if audio is not None:
                audio = audio.copy()  # Don't modify original
                
                # Apply per-stem gain
                gain = default_gains.get(name, 0.8)
                audio *= gain
                
                # Apply ducking to music stem
                if duck_music and name == 'music' and dialogue_audio is not None:
                    audio = self._apply_ducking(audio, dialogue_audio)
                
                # Pad if too short
                current_len = audio.shape[1]
                if current_len < max_len:
                    padding = np.zeros((2, max_len - current_len), dtype=np.float32)
                    audio = np.concatenate([audio, padding], axis=1)
                
                # Add to mix
                mix_bus += audio
        
        # 5. Place time-aligned events (SFX) with ducking
        if events:
            for ev in events:
                ev_audio = ev.get('audio')
                if ev_audio is None:
                    continue

                timestamp_sec = ev.get('timestamp', 0)
                start_sample = int(timestamp_sec * self.sr)

                # Apply SFX gain
                ev_audio = ev_audio.copy() * default_gains.get('sfx', 0.8)

                # Apply ducking to SFX under dialogue
                if duck_music and dialogue_audio is not None:
                    # Extract the dialogue segment that overlaps with this SFX
                    ev_len = ev_audio.shape[1]
                    end_sample = start_sample + ev_len
                    dialogue_segment_end = min(end_sample, dialogue_audio.shape[1])

                    if start_sample < dialogue_audio.shape[1]:
                        # Create a dialogue segment matching the SFX length
                        dialogue_segment = np.zeros((2, ev_len), dtype=np.float32)
                        overlap_len = dialogue_segment_end - start_sample
                        if overlap_len > 0:
                            dialogue_segment[:, :overlap_len] = dialogue_audio[:, start_sample:dialogue_segment_end]

                        # Duck the SFX under dialogue
                        ev_audio = self._apply_ducking(
                            ev_audio, dialogue_segment,
                            threshold_db=-25.0,  # Slightly more sensitive for SFX
                            reduction_db=-8.0,   # Less aggressive reduction than music
                            attack_ms=5,
                            release_ms=100
                        )

                ev_len = ev_audio.shape[1]
                end_sample = start_sample + ev_len

                # Ensure we don't write past the buffer
                if end_sample > max_len:
                    ev_audio = ev_audio[:, :max_len - start_sample]
                    end_sample = max_len

                if start_sample < max_len:
                    mix_bus[:, start_sample:end_sample] += ev_audio

        # 6. Mastering Chain
        master_board = Pedalboard([
            Gain(gain_db=-3.0),  # Headroom
            Compressor(threshold_db=-12, ratio=3.0, attack_ms=20, release_ms=200),  # Glue compression
            Limiter(threshold_db=-2.0)  # True peak ceiling at -2 dBTP
        ])
        final_mix = master_board(mix_bus, self.sr)

        # 7. Export
        sf.write(output_path, final_mix.T, self.sr)  # Soundfile expects (Samples, Channels)
        return True

    def mix_and_master_with_stems(self, stems_dict, output_path, events=None,
                                   stem_gains=None, duck_music=True, export_stems=True,
                                   sfx_event_settings=None):
        """
        Summing engine that also exports ground truth stems.
        The exported stems sum exactly to the final mix.

        Args:
            stems_dict: {'speech': audio_arr, 'music': audio_arr, 'ambience': audio_arr}
            output_path: Where to write the final mix
            events: List of event dicts with 'audio', 'timestamp' (seconds)
            stem_gains: Dict of gain values per stem
            duck_music: Whether to duck music/sfx under dialogue
            export_stems: Whether to export ground truth stems
            sfx_event_settings: Dict of per-SFX event mixing params from Critic
                               e.g. {'sfx_0': {'gain_db': -6.0, 'reverb': {...}}, ...}

        Returns:
            dict: {
                'success': bool,
                'stem_paths': {name: file_path} if export_stems else None,
                'additivity_error': float (max error between sum of stems and mix)
            }
        """
        # Default stem gains
        default_gains = {
            'speech': 1.0,
            'dialogue': 1.0,
            'music': 0.35,
            'ambience': 0.25,
            'sfx': 0.8,
        }
        if stem_gains:
            default_gains.update(stem_gains)

        # 1. Determine max length
        max_len = 0
        for name, audio in stems_dict.items():
            if audio is not None:
                max_len = max(max_len, audio.shape[1])

        if events:
            for ev in events:
                if ev.get('audio') is not None:
                    ts_samples = int(ev.get('timestamp', 0) * self.sr)
                    ev_len = ev['audio'].shape[1]
                    max_len = max(max_len, ts_samples + ev_len)

        if max_len == 0:
            return {'success': False, 'stem_paths': None, 'additivity_error': None}

        # 2. Get dialogue for ducking reference
        dialogue_audio = stems_dict.get('speech')
        if dialogue_audio is None:
            dialogue_audio = stems_dict.get('dialogue')

        # 3. Process stems and track pre-master versions
        mix_bus = np.zeros((2, max_len), dtype=np.float32)
        pre_master_stems = {}

        for name, audio in stems_dict.items():
            if audio is not None:
                audio = audio.copy()
                gain = default_gains.get(name, 0.8)
                audio *= gain

                # Apply ducking to music
                if duck_music and name == 'music' and dialogue_audio is not None:
                    audio = self._apply_ducking(audio, dialogue_audio)

                # Pad if needed
                if audio.shape[1] < max_len:
                    padding = np.zeros((2, max_len - audio.shape[1]), dtype=np.float32)
                    audio = np.concatenate([audio, padding], axis=1)

                pre_master_stems[name] = audio.copy()
                mix_bus += audio

        # 4. Process SFX events and combine into single stem
        sfx_combined = np.zeros((2, max_len), dtype=np.float32)
        sfx_event_settings = sfx_event_settings or {}

        if events:
            for idx, ev in enumerate(events):
                ev_audio = ev.get('audio')
                if ev_audio is None:
                    continue

                timestamp_sec = ev.get('timestamp', 0)
                start_sample = int(timestamp_sec * self.sr)
                ev_audio = ev_audio.copy()

                # Get per-event settings from Critic (if available)
                sfx_key = f"sfx_{idx}"
                ev_settings = sfx_event_settings.get(sfx_key, {})

                # Apply per-event gain (absolute dB value, or fallback to default)
                ev_gain_db = ev_settings.get('gain_db', None)
                if ev_gain_db is not None:
                    ev_gain_linear = 10 ** (float(ev_gain_db) / 20.0)
                    ev_audio *= ev_gain_linear
                else:
                    # Fallback to default SFX gain
                    ev_audio *= default_gains.get('sfx', 0.8)

                # Apply per-event reverb (if specified by Critic)
                ev_reverb = ev_settings.get('reverb', None)
                if ev_reverb and isinstance(ev_reverb, dict):
                    reverb_presets = self._get_reverb_presets()
                    reverb_type = ev_reverb.get('type', 'medium_room')
                    preset = reverb_presets.get(reverb_type, reverb_presets['medium_room'])
                    wet_amount = ev_reverb.get('wet_amount', preset['wet'])

                    reverb = Reverb(
                        room_size=preset['room_size'],
                        damping=preset.get('damping', 0.5),
                        wet_level=wet_amount,
                        dry_level=1.0
                    )
                    ev_audio = reverb(ev_audio, self.sr)

                # Apply ducking to SFX
                if duck_music and dialogue_audio is not None:
                    ev_len = ev_audio.shape[1]
                    end_sample = start_sample + ev_len
                    dialogue_segment_end = min(end_sample, dialogue_audio.shape[1])

                    if start_sample < dialogue_audio.shape[1]:
                        dialogue_segment = np.zeros((2, ev_len), dtype=np.float32)
                        overlap_len = dialogue_segment_end - start_sample
                        if overlap_len > 0:
                            dialogue_segment[:, :overlap_len] = dialogue_audio[:, start_sample:dialogue_segment_end]

                        ev_audio = self._apply_ducking(
                            ev_audio, dialogue_segment,
                            threshold_db=-25.0, reduction_db=-8.0,
                            attack_ms=10, release_ms=150
                        )

                ev_len = ev_audio.shape[1]
                end_sample = start_sample + ev_len

                if end_sample > max_len:
                    ev_audio = ev_audio[:, :max_len - start_sample]
                    end_sample = max_len

                if start_sample < max_len:
                    sfx_combined[:, start_sample:end_sample] += ev_audio
                    mix_bus[:, start_sample:end_sample] += ev_audio

        # Add SFX to pre_master_stems if any events
        if np.any(sfx_combined != 0):
            pre_master_stems['sfx'] = sfx_combined

        # 5. Apply mastering chain step-by-step to capture gain reduction
        headroom_gain = 10 ** (-3.0 / 20.0)
        post_headroom = mix_bus * headroom_gain

        # Compression (glue)
        compressor = Compressor(threshold_db=-12, ratio=3.0, attack_ms=20, release_ms=200)
        post_compression = compressor(post_headroom, self.sr)
        compression_envelope = self._compute_gain_reduction_envelope(post_headroom, post_compression)

        # Limiting (-2 dBTP true peak ceiling)
        limiter = Limiter(threshold_db=-2.0)
        post_limiter = limiter(post_compression, self.sr)
        limiter_envelope = self._compute_gain_reduction_envelope(post_compression, post_limiter)

        # Combined master envelope (before loudness normalization)
        master_envelope = compression_envelope * limiter_envelope

        # 6. Loudness normalization (-27 LKFS ±1 LU, -2 dBTP ceiling)
        final_mix, loudness_gain, loudness_measurements = self._normalize_loudness(
            post_limiter,
            target_lufs=-27.0,
            true_peak_ceiling=-2.0,
            randomize_lu=1.0
        )

        # Compute total gain from pre-master to final (includes all processing)
        # This avoids accumulating errors from separate envelope multiplications
        pre_master_mix = mix_bus * headroom_gain
        epsilon = 1e-10

        # Compute per-sample gain as ratio of final to pre-master (per channel)
        total_gain_L = np.where(
            np.abs(pre_master_mix[0, :]) > epsilon,
            final_mix[0, :] / (pre_master_mix[0, :] + epsilon * np.sign(pre_master_mix[0, :] + epsilon)),
            1.0
        )
        total_gain_R = np.where(
            np.abs(pre_master_mix[1, :]) > epsilon,
            final_mix[1, :] / (pre_master_mix[1, :] + epsilon * np.sign(pre_master_mix[1, :] + epsilon)),
            1.0
        )

        # 7. Create ground truth stems by applying the exact gain ratio
        ground_truth_stems = {}
        for name, stem_audio in pre_master_stems.items():
            processed = stem_audio * headroom_gain
            processed[0, :] *= total_gain_L
            processed[1, :] *= total_gain_R
            ground_truth_stems[name] = processed

        # 8. Verify additivity
        reconstructed = np.zeros((2, max_len), dtype=np.float32)
        for stem in ground_truth_stems.values():
            reconstructed += stem
        max_error = float(np.max(np.abs(final_mix - reconstructed)))

        # 9. Export final mix
        sf.write(output_path, final_mix.T, self.sr)

        # 10. Export ground truth stems
        stem_paths = None
        if export_stems:
            output_dir = os.path.dirname(output_path)
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            stem_paths = {}

            for name, stem in ground_truth_stems.items():
                stem_path = os.path.join(output_dir, f"{base_name}_{name}_stem.wav")
                sf.write(stem_path, stem.T, self.sr)
                stem_paths[name] = stem_path

        return {
            'success': True,
            'stem_paths': stem_paths,
            'additivity_error': max_error,
            'loudness': {
                'integrated_lufs': loudness_measurements['loudness_lufs'],
                'true_peak_dbtp': loudness_measurements['true_peak_dbtp']
            }
        }

    # =========================================================================
    # NEW: Unified Scene Rendering (raw / rough / linear / release)
    # =========================================================================

    def render_scene(self, raw_stems, output_dir, events,
                     rough_settings, critic_settings, sfx_event_settings,
                     scene_type_config=None):
        """
        Renders all four output types for a scene.

        Args:
            raw_stems: Dict of raw loaded audio arrays
                       {'speech': arr, 'music': arr, 'ambience': arr}
            output_dir: Base scene folder (e.g., ./data/scene_001/)
            events: List of SFX event dicts with 'audio', 'timestamp', 'description'
            rough_settings: Pre-Critic DSP settings per stem
            critic_settings: Post-Critic DSP settings per stem
            sfx_event_settings: Per-event mixing from Critic
                               e.g. {'sfx_0': {'gain_db': -6.0, 'reverb': {...}}, ...}
            scene_type_config: Dict with scene type configuration including:
                              - gain_offsets: {speech, music, ambience, sfx}
                              - duck_music: bool
                              - duck_sfx: bool

        Returns:
            dict with paths and metadata for all outputs
        """
        # Create subfolders (skip 'raw' for speed - not needed for training)
        for subdir in ['rough', 'linear', 'release']:
            os.makedirs(f"{output_dir}/{subdir}", exist_ok=True)

        # Determine scene length from speech stem
        speech = raw_stems.get('speech')
        if speech is None:
            return {'success': False, 'error': 'No speech stem provided'}
        max_len = speech.shape[1]

        # Extract scene type config settings
        stc = scene_type_config or {}
        gain_offsets = stc.get('gain_offsets', {})
        duck_music = stc.get('duck_music', True)
        duck_sfx = stc.get('duck_sfx', True)

        # Apply gain offsets to settings
        rough_settings = self._apply_gain_offsets(rough_settings, gain_offsets)
        critic_settings = self._apply_gain_offsets(critic_settings, gain_offsets)

        # Create ducking config
        ducking_config = {
            'duck_music': duck_music,
            'duck_sfx': duck_sfx,
        }

        # 1. Skip raw/ export (not needed for training, saves time/disk)
        # raw_result = self._export_raw(raw_stems, events, output_dir, max_len)

        # 2. Export rough/ (pre-Critic, clean stems, mixbus sidechain)
        rough_result = self._export_rough(raw_stems, events, rough_settings,
                                          output_dir, max_len, ducking_config)

        # 3. Export linear/ (post-Critic, sidechain IN stems)
        linear_result = self._export_linear(raw_stems, events, critic_settings,
                                            sfx_event_settings, output_dir, max_len,
                                            ducking_config)

        # 4. Export release/ (post-Critic, clean stems, mixbus sidechain)
        release_result = self._export_release(raw_stems, events, critic_settings,
                                              sfx_event_settings, output_dir, max_len,
                                              ducking_config)

        return {
            'success': True,
            # 'raw': raw_result,  # Skipped for performance
            'rough': rough_result,
            'linear': linear_result,
            'release': release_result,
            'scene_type_config': stc,
        }

    def _apply_gain_offsets(self, settings, gain_offsets):
        """Apply scene type gain offsets to stem settings."""
        if not gain_offsets:
            return settings

        adjusted = {}
        for stem_name, stem_settings in settings.items():
            offset = gain_offsets.get(stem_name, 0.0)
            if offset == 0.0:
                adjusted[stem_name] = stem_settings
                continue

            # Copy settings and add offset to gain_db
            new_settings = dict(stem_settings) if stem_settings else {}
            current_gain = new_settings.get('gain_db', 0.0)
            new_settings['gain_db'] = current_gain + offset
            adjusted[stem_name] = new_settings

        return adjusted

    def _build_sfx_stem(self, events, max_len, sfx_event_settings=None,
                        apply_per_event_fx=False, speech_for_ducking=None):
        """
        Build combined SFX stem from events.

        Args:
            events: List of event dicts with 'audio', 'timestamp'
            max_len: Target length in samples
            sfx_event_settings: Per-event mixing params (gain, reverb)
            apply_per_event_fx: If True, apply per-event gain/reverb from Critic
            speech_for_ducking: If provided, apply sidechain ducking to each event

        Returns:
            np.array: Combined SFX stem (2, max_len)
        """
        sfx_combined = np.zeros((2, max_len), dtype=np.float32)
        sfx_event_settings = sfx_event_settings or {}

        if not events:
            return sfx_combined

        for idx, ev in enumerate(events):
            ev_audio = ev.get('audio')
            if ev_audio is None:
                continue

            timestamp_sec = ev.get('timestamp', 0)
            start_sample = int(timestamp_sec * self.sr)
            ev_audio = ev_audio.copy()

            # Apply per-event processing if enabled
            if apply_per_event_fx:
                sfx_key = f"sfx_{idx}"
                ev_settings = sfx_event_settings.get(sfx_key, {})

                # Per-event gain
                ev_gain_db = ev_settings.get('gain_db', None)
                if ev_gain_db is not None:
                    ev_gain_linear = 10 ** (float(ev_gain_db) / 20.0)
                    ev_audio *= ev_gain_linear

                # Per-event reverb
                ev_reverb = ev_settings.get('reverb', None)
                if ev_reverb and isinstance(ev_reverb, dict):
                    reverb_presets = self._get_reverb_presets()
                    reverb_type = ev_reverb.get('type', 'medium_room')
                    preset = reverb_presets.get(reverb_type, reverb_presets['medium_room'])
                    wet_amount = ev_reverb.get('wet_amount', preset['wet'])

                    reverb = Reverb(
                        room_size=preset['room_size'],
                        damping=preset.get('damping', 0.5),
                        wet_level=wet_amount,
                        dry_level=1.0
                    )
                    ev_audio = reverb(ev_audio, self.sr)

            # Apply ducking if speech provided
            if speech_for_ducking is not None:
                ev_len = ev_audio.shape[1]
                end_sample = start_sample + ev_len
                dialogue_segment_end = min(end_sample, speech_for_ducking.shape[1])

                if start_sample < speech_for_ducking.shape[1]:
                    dialogue_segment = np.zeros((2, ev_len), dtype=np.float32)
                    overlap_len = dialogue_segment_end - start_sample
                    if overlap_len > 0:
                        dialogue_segment[:, :overlap_len] = speech_for_ducking[:, start_sample:dialogue_segment_end]

                    ev_audio = self._apply_ducking(
                        ev_audio, dialogue_segment,
                        threshold_db=-25.0, reduction_db=-8.0,
                        attack_ms=10, release_ms=150
                    )

            # Place in timeline
            ev_len = ev_audio.shape[1]
            end_sample = start_sample + ev_len

            if end_sample > max_len:
                ev_audio = ev_audio[:, :max_len - start_sample]
                end_sample = max_len

            if start_sample < max_len:
                sfx_combined[:, start_sample:end_sample] += ev_audio

        return sfx_combined

    def _pad_to_length(self, audio, max_len):
        """Pad audio to specified length."""
        if audio is None:
            return np.zeros((2, max_len), dtype=np.float32)
        if audio.shape[1] < max_len:
            padding = np.zeros((2, max_len - audio.shape[1]), dtype=np.float32)
            return np.concatenate([audio, padding], axis=1)
        return audio[:, :max_len]

    def _export_raw(self, raw_stems, events, output_dir, max_len):
        """
        Export unprocessed source stems.
        No processing at all - just raw loaded audio.
        NO mix file for raw (just stems).
        """
        raw_dir = f"{output_dir}/raw"
        stem_paths = {}

        # Export main stems (no processing)
        for name in ['speech', 'music', 'ambience']:
            audio = raw_stems.get(name)
            if audio is not None:
                audio = self._pad_to_length(audio.copy(), max_len)
                path = f"{raw_dir}/{name}.wav"
                sf.write(path, audio.T, self.sr)
                stem_paths[name] = path

        # Build raw SFX (events at timestamps, no effects)
        sfx = self._build_sfx_stem(events, max_len,
                                   apply_per_event_fx=False,
                                   speech_for_ducking=None)
        if np.any(sfx != 0):
            path = f"{raw_dir}/sfx.wav"
            sf.write(path, sfx.T, self.sr)
            stem_paths['sfx'] = path

        return {'stem_paths': stem_paths}

    def _export_rough(self, raw_stems, events, rough_settings, output_dir, max_len,
                      ducking_config=None):
        """
        Export pre-Critic stems (clean, no ducking) + mastered mix.
        Sidechain is applied ON MIXBUS only.
        """
        rough_dir = f"{output_dir}/rough"
        stem_paths = {}
        ducking_config = ducking_config or {'duck_music': True, 'duck_sfx': True}

        # Get speech for sidechain reference
        speech = raw_stems.get('speech')
        speech_processed = self.process_stem(speech.copy(), rough_settings.get('speech', {}))
        speech_processed = self._pad_to_length(speech_processed, max_len)

        # Process and export stems (CLEAN - no ducking)
        processed_stems = {}

        # Speech
        processed_stems['speech'] = speech_processed
        sf.write(f"{rough_dir}/speech.wav", speech_processed.T, self.sr)
        stem_paths['speech'] = f"{rough_dir}/speech.wav"

        # Music (clean, no ducking in stem) - handle None gracefully
        music = raw_stems.get('music')
        if music is not None:
            music_processed = self.process_stem(music.copy(), rough_settings.get('music', {}))
            music_processed = self._pad_to_length(music_processed, max_len)
            processed_stems['music'] = music_processed
            sf.write(f"{rough_dir}/music.wav", music_processed.T, self.sr)
            stem_paths['music'] = f"{rough_dir}/music.wav"
        else:
            # Export silence stem for training (model learns absence)
            silence = np.zeros((2, max_len), dtype=np.float32)
            processed_stems['music'] = silence
            sf.write(f"{rough_dir}/music.wav", silence.T, self.sr)
            stem_paths['music'] = f"{rough_dir}/music.wav"

        # Ambience - handle None gracefully
        ambience = raw_stems.get('ambience')
        if ambience is not None:
            ambience_processed = self.process_stem(ambience.copy(), rough_settings.get('ambience', {}))
            ambience_processed = self._pad_to_length(ambience_processed, max_len)
            processed_stems['ambience'] = ambience_processed
            sf.write(f"{rough_dir}/ambience.wav", ambience_processed.T, self.sr)
            stem_paths['ambience'] = f"{rough_dir}/ambience.wav"
        else:
            silence = np.zeros((2, max_len), dtype=np.float32)
            processed_stems['ambience'] = silence
            sf.write(f"{rough_dir}/ambience.wav", silence.T, self.sr)
            stem_paths['ambience'] = f"{rough_dir}/ambience.wav"

        # SFX (no per-event processing, no ducking in stem)
        sfx = self._build_sfx_stem(events, max_len,
                                   apply_per_event_fx=False,
                                   speech_for_ducking=None)
        # Apply global SFX settings from rough_settings
        sfx = self.process_stem(sfx, rough_settings.get('sfx', {}))
        processed_stems['sfx'] = sfx
        sf.write(f"{rough_dir}/sfx.wav", sfx.T, self.sr)
        stem_paths['sfx'] = f"{rough_dir}/sfx.wav"

        # Build mix with sidechain ON MIXBUS (respecting ducking config)
        mix = self._build_mixbus_with_sidechain(processed_stems, speech_processed,
                                                ducking_config)

        # Apply mastering chain
        mix, loudness_info = self._apply_mastering(mix)

        sf.write(f"{rough_dir}/mix.wav", mix.T, self.sr)

        return {
            'stem_paths': stem_paths,
            'mix_path': f"{rough_dir}/mix.wav",
            'loudness': loudness_info
        }

    def _export_linear(self, raw_stems, events, critic_settings,
                       sfx_event_settings, output_dir, max_len,
                       ducking_config=None):
        """
        Export post-Critic stems with sidechain BAKED INTO stems.
        mix = sum(stems) exactly.
        Also exports gain_envelope.json with ducking automation curves.
        """
        linear_dir = f"{output_dir}/linear"
        stem_paths = {}
        gain_envelopes = {}
        ducking_config = ducking_config or {'duck_music': True, 'duck_sfx': True}

        # Get speech for sidechain reference
        speech = raw_stems.get('speech')
        speech_processed = self.process_stem(speech.copy(), critic_settings.get('speech', {}))
        speech_processed = self._pad_to_length(speech_processed, max_len)

        # Process stems with ducking BAKED IN (if enabled)
        stems = {}

        # Speech (no ducking on speech itself)
        stems['speech'] = speech_processed
        sf.write(f"{linear_dir}/speech.wav", speech_processed.T, self.sr)
        stem_paths['speech'] = f"{linear_dir}/speech.wav"

        # Music (ducking BAKED IN if enabled) - capture envelope
        music = raw_stems.get('music')
        music_envelope = None
        if music is not None:
            music_processed = self.process_stem(music.copy(), critic_settings.get('music', {}))
            music_processed = self._pad_to_length(music_processed, max_len)

            # Apply ducking only if enabled for this scene type
            if ducking_config.get('duck_music', True):
                music_ducked, music_envelope = self._apply_ducking(
                    music_processed, speech_processed, return_envelope=True
                )
                stems['music'] = music_ducked
            else:
                stems['music'] = music_processed

            sf.write(f"{linear_dir}/music.wav", stems['music'].T, self.sr)
            stem_paths['music'] = f"{linear_dir}/music.wav"
        else:
            # Export silence stem for training (model learns absence)
            silence = np.zeros((2, max_len), dtype=np.float32)
            stems['music'] = silence
            sf.write(f"{linear_dir}/music.wav", silence.T, self.sr)
            stem_paths['music'] = f"{linear_dir}/music.wav"

        # Ambience (no ducking typically, but pad to length)
        ambience = raw_stems.get('ambience')
        if ambience is not None:
            ambience_processed = self.process_stem(ambience.copy(), critic_settings.get('ambience', {}))
            ambience_processed = self._pad_to_length(ambience_processed, max_len)
            stems['ambience'] = ambience_processed
            sf.write(f"{linear_dir}/ambience.wav", ambience_processed.T, self.sr)
            stem_paths['ambience'] = f"{linear_dir}/ambience.wav"
        else:
            silence = np.zeros((2, max_len), dtype=np.float32)
            stems['ambience'] = silence
            sf.write(f"{linear_dir}/ambience.wav", silence.T, self.sr)
            stem_paths['ambience'] = f"{linear_dir}/ambience.wav"

        # SFX: Build without ducking first, then apply ducking to capture envelope
        sfx_no_duck = self._build_sfx_stem(events, max_len,
                                           sfx_event_settings=sfx_event_settings,
                                           apply_per_event_fx=True,
                                           speech_for_ducking=None)
        # Apply global SFX settings
        sfx_no_duck = self.process_stem(sfx_no_duck, critic_settings.get('sfx', {}))

        # Apply ducking only if enabled for this scene type
        sfx_envelope = None
        if ducking_config.get('duck_sfx', True):
            sfx_ducked, sfx_envelope = self._apply_ducking(
                sfx_no_duck, speech_processed,
                threshold_db=-25.0, reduction_db=-8.0,
                attack_ms=10, release_ms=150,
                return_envelope=True
            )
            stems['sfx'] = sfx_ducked
        else:
            stems['sfx'] = sfx_no_duck

        sf.write(f"{linear_dir}/sfx.wav", stems['sfx'].T, self.sr)
        stem_paths['sfx'] = f"{linear_dir}/sfx.wav"

        # Convert envelopes to keyframes and save
        if music_envelope is not None:
            gain_envelopes['music'] = {
                'description': 'Sidechain ducking envelope (triggered by speech)',
                'ducking_params': {
                    'threshold_db': -20.0,
                    'reduction_db': -12.0,
                    'attack_ms': 20,
                    'release_ms': 225
                },
                'keyframes': self._envelope_to_keyframes(music_envelope)
            }

        if sfx_envelope is not None:
            gain_envelopes['sfx'] = {
                'description': 'Sidechain ducking envelope (triggered by speech)',
                'ducking_params': {
                    'threshold_db': -25.0,
                    'reduction_db': -8.0,
                    'attack_ms': 15,
                    'release_ms': 200
                },
                'keyframes': self._envelope_to_keyframes(sfx_envelope)
            }

        # Save gain envelopes to JSON (includes ducking config info)
        envelope_path = f"{linear_dir}/gain_envelope.json"
        with open(envelope_path, 'w') as f:
            json.dump({
                'sample_rate': self.sr,
                'duration_sec': max_len / self.sr,
                'ducking_enabled': {
                    'music': ducking_config.get('duck_music', True),
                    'sfx': ducking_config.get('duck_sfx', True),
                },
                'envelopes': gain_envelopes
            }, f, indent=2)

        # Mix = simple sum of stems (exact additivity)
        mix = np.zeros((2, max_len), dtype=np.float32)
        for stem in stems.values():
            mix += stem

        # Prevent 16-bit clipping AND make additivity bit-exact in the stored files:
        # 1) if the mix (or any stem) peaks above -1 dBFS, apply ONE shared gain to
        #    all stems (a scalar distributes over the sum, preserving additivity);
        # 2) quantize stems to int16 (rounded) and write the mix as their EXACT
        #    integer sum, so stored mix == sum(stored stems) with error 0.
        target_peak = 10 ** (-1.0 / 20.0)  # -1 dBFS
        peak = max(float(np.max(np.abs(mix))),
                   *(float(np.max(np.abs(s))) for s in stems.values()))
        norm_gain = target_peak / peak if peak > target_peak else 1.0
        mix_i = np.zeros(mix.shape, dtype=np.int32)
        for name in stems:
            k = np.round(stems[name] * (norm_gain * 32768.0)).clip(-32768, 32767).astype(np.int32)
            sf.write(stem_paths[name], k.astype(np.int16).T, self.sr, subtype='PCM_16')
            mix_i += k
        assert int(np.max(np.abs(mix_i))) <= 32767, "int16 overflow in linear mix"
        sf.write(f"{linear_dir}/mix.wav", mix_i.astype(np.int16).T, self.sr, subtype='PCM_16')

        # mix.wav is the exact integer sum of the stored stems
        additivity_error = 0.0

        # Measure loudness on what was actually written
        loudness_info = self._measure_loudness(mix_i.astype(np.float32) / 32768.0)

        return {
            'stem_paths': stem_paths,
            'mix_path': f"{linear_dir}/mix.wav",
            'envelope_path': envelope_path,
            'additivity_error': additivity_error,
            'peak_normalization': {
                'applied': norm_gain != 1.0,
                'gain_db': float(20 * np.log10(norm_gain)) if norm_gain != 1.0 else 0.0,
                'target_peak_dbfs': -1.0,
            },
            'loudness': loudness_info
        }

    def _export_release(self, raw_stems, events, critic_settings,
                        sfx_event_settings, output_dir, max_len,
                        ducking_config=None):
        """
        Export post-Critic stems (clean, no ducking) + mastered mix.
        Sidechain is applied ON MIXBUS only.
        """
        release_dir = f"{output_dir}/release"
        stem_paths = {}
        ducking_config = ducking_config or {'duck_music': True, 'duck_sfx': True}

        # Get speech for sidechain reference
        speech = raw_stems.get('speech')
        speech_processed = self.process_stem(speech.copy(), critic_settings.get('speech', {}))
        speech_processed = self._pad_to_length(speech_processed, max_len)

        # Process and export stems (CLEAN - no ducking)
        processed_stems = {}

        # Speech
        processed_stems['speech'] = speech_processed
        sf.write(f"{release_dir}/speech.wav", speech_processed.T, self.sr)
        stem_paths['speech'] = f"{release_dir}/speech.wav"

        # Music (clean, no ducking in stem) - handle None gracefully
        music = raw_stems.get('music')
        if music is not None:
            music_processed = self.process_stem(music.copy(), critic_settings.get('music', {}))
            music_processed = self._pad_to_length(music_processed, max_len)
            processed_stems['music'] = music_processed
            sf.write(f"{release_dir}/music.wav", music_processed.T, self.sr)
            stem_paths['music'] = f"{release_dir}/music.wav"
        else:
            silence = np.zeros((2, max_len), dtype=np.float32)
            processed_stems['music'] = silence
            sf.write(f"{release_dir}/music.wav", silence.T, self.sr)
            stem_paths['music'] = f"{release_dir}/music.wav"

        # Ambience - handle None gracefully
        ambience = raw_stems.get('ambience')
        if ambience is not None:
            ambience_processed = self.process_stem(ambience.copy(), critic_settings.get('ambience', {}))
            ambience_processed = self._pad_to_length(ambience_processed, max_len)
            processed_stems['ambience'] = ambience_processed
            sf.write(f"{release_dir}/ambience.wav", ambience_processed.T, self.sr)
            stem_paths['ambience'] = f"{release_dir}/ambience.wav"
        else:
            silence = np.zeros((2, max_len), dtype=np.float32)
            processed_stems['ambience'] = silence
            sf.write(f"{release_dir}/ambience.wav", silence.T, self.sr)
            stem_paths['ambience'] = f"{release_dir}/ambience.wav"

        # SFX (per-event processing, NO ducking in stem)
        sfx = self._build_sfx_stem(events, max_len,
                                   sfx_event_settings=sfx_event_settings,
                                   apply_per_event_fx=True,
                                   speech_for_ducking=None)
        # Apply global SFX settings
        sfx = self.process_stem(sfx, critic_settings.get('sfx', {}))
        processed_stems['sfx'] = sfx
        sf.write(f"{release_dir}/sfx.wav", sfx.T, self.sr)
        stem_paths['sfx'] = f"{release_dir}/sfx.wav"

        # Build mix with sidechain ON MIXBUS (respecting ducking config)
        mix = self._build_mixbus_with_sidechain(processed_stems, speech_processed,
                                                ducking_config)

        # Apply mastering chain
        mix, loudness_info = self._apply_mastering(mix)

        sf.write(f"{release_dir}/mix.wav", mix.T, self.sr)

        return {
            'stem_paths': stem_paths,
            'mix_path': f"{release_dir}/mix.wav",
            'loudness': loudness_info
        }

    def _build_mixbus_with_sidechain(self, stems_dict, speech_for_sidechain,
                                     ducking_config=None):
        """
        Build mix with sidechain applied ON MIXBUS (not in stems).
        Stems remain clean, only the summed mix gets ducking.

        Args:
            stems_dict: Dict of processed stems
            speech_for_sidechain: Speech audio for ducking trigger
            ducking_config: Dict with 'duck_music' and 'duck_sfx' bools
        """
        ducking_config = ducking_config or {'duck_music': True, 'duck_sfx': True}
        max_len = speech_for_sidechain.shape[1]
        mix = np.zeros((2, max_len), dtype=np.float32)

        # Speech goes through unaffected
        mix += stems_dict.get('speech', np.zeros((2, max_len), dtype=np.float32))

        # Music gets ducked based on speech (on mixbus) - if enabled
        music = stems_dict.get('music')
        if music is not None:
            if ducking_config.get('duck_music', True):
                ducked_music = self._apply_ducking(music.copy(), speech_for_sidechain)
                mix += ducked_music
            else:
                mix += music

        # Ambience - typically not ducked
        ambience = stems_dict.get('ambience')
        if ambience is not None:
            mix += ambience

        # SFX gets ducked based on speech (on mixbus) - if enabled
        sfx = stems_dict.get('sfx')
        if sfx is not None:
            if ducking_config.get('duck_sfx', True):
                ducked_sfx = self._apply_ducking(sfx.copy(), speech_for_sidechain,
                                                 threshold_db=-25.0, reduction_db=-8.0,
                                                 attack_ms=10, release_ms=150)
                mix += ducked_sfx
            else:
                mix += sfx

        return mix

    def _apply_mastering(self, mix):
        """
        Apply mastering chain: headroom, compression, limiter, loudness normalization.

        Returns:
            tuple: (mastered_mix, loudness_info)
        """
        # Headroom
        headroom_gain = 10 ** (-3.0 / 20.0)
        mix = mix * headroom_gain

        # Compression (glue)
        compressor = Compressor(threshold_db=-12, ratio=3.0, attack_ms=20, release_ms=200)
        mix = compressor(mix, self.sr)

        # Limiter
        limiter = Limiter(threshold_db=-2.0)
        mix = limiter(mix, self.sr)

        # Loudness normalization
        mix, _, loudness_measurements = self._normalize_loudness(
            mix,
            target_lufs=-27.0,
            true_peak_ceiling=-2.0,
            randomize_lu=1.0
        )

        loudness_info = {
            'integrated_lufs': loudness_measurements['loudness_lufs'],
            'true_peak_dbtp': loudness_measurements['true_peak_dbtp']
        }

        return mix, loudness_info