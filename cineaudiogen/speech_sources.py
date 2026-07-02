"""
Speech Source Manager for Cinematic Audio Pipeline

Provides unified access to multiple speech datasets:
- Expresso: Emotional conversational speech (long clips)
- RAVDESS: Professional actors, 8 emotions, studio quality 48kHz (short clips, concatenatable)
- ASED: Amharic Speech Emotional Dataset, 5 emotions (short clips, 4 concatenated per scene)
- NonverbalTTS: Short clips with nonverbal vocalizations (laughs, coughs, sighs)

Note: MELD is disabled (contains TV production audio with baked-in SFX)

Usage:
    manager = SpeechSourceManager(data_root="./data")

    # Get random dialogue from any source
    dialogue = manager.get_random_dialogue()

    # Get dialogue by emotion
    dialogue = manager.get_dialogue_by_emotion("angry")

    # Get dialogue from specific source
    dialogue = manager.get_random_dialogue(source="ravdess")
    dialogue = manager.get_random_dialogue(source="ased")

    # Get nonverbal vocalization (laugh, cough, sigh, etc.)
    nv_clip = manager.get_nonverbal_clip(nv_type="laugh")
"""

import os
import json
import glob
import random
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from pathlib import Path


@dataclass
class DialogueInfo:
    """Information about a dialogue clip."""
    path: str                    # Path to audio file (may be concatenated temp file)
    source: str                  # "expresso", "meld", or "nonverbal"
    emotion: Optional[str]       # Primary emotion (if available)
    style: str                   # Style/context string
    duration_sec: float          # Duration in seconds
    speakers: List[str]          # List of speakers
    utterances: List[str]        # List of utterance texts (if available)
    metadata: Dict               # Additional metadata


@dataclass
class NonverbalClipInfo:
    """Information about a nonverbal vocalization clip."""
    path: str                    # Path to audio file
    nv_type: str                 # Type: laugh, cough, sigh, sneeze, grunt, etc.
    emotion: Optional[str]       # Emotion label
    duration_sec: float          # Duration in seconds
    text: str                    # Transcript with emoji annotations
    speaker_id: str              # Speaker identifier
    gender: str                  # m/f
    metadata: Dict               # Additional metadata


class SpeechSourceManager:
    """Unified manager for multiple speech datasets."""

    def __init__(self, data_root: str, cache_dir: Optional[str] = None):
        self.data_root = data_root
        self.cache_dir = cache_dir or os.path.join(data_root, "speech_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize sources
        self.expresso_files = []
        self.meld_index = None
        self.meld_dialogues = {}  # (split, dialogue_id) -> list of entries
        self.nonverbal_index = None
        self.nonverbal_by_type = {}  # nv_type -> list of entries
        self.ravdess_index = None
        self.ravdess_by_emotion = {}  # emotion -> list of entries
        self.ased_index = None
        self.ased_by_emotion = {}  # emotion -> list of entries

        self._load_expresso()
        self._load_meld()
        self._load_nonverbal()
        self._load_ravdess()
        self._load_ased()

        print(f"[SpeechSources] Loaded {len(self.expresso_files)} Expresso files")
        print(f"[SpeechSources] Loaded {len(self.meld_dialogues)} MELD dialogues (disabled)")
        print(f"[SpeechSources] Loaded {sum(len(v) for v in self.ravdess_by_emotion.values())} RAVDESS clips")
        print(f"[SpeechSources] Loaded {sum(len(v) for v in self.ased_by_emotion.values())} ASED clips")
        print(f"[SpeechSources] Loaded {sum(len(v) for v in self.nonverbal_by_type.values())} NonverbalTTS clips")

    def _load_expresso(self):
        """Load Expresso dataset file list."""
        expresso_pattern = os.path.join(
            self.data_root,
            "speech_expresso/expresso/audio_48khz/conversational/**/**/*.wav"
        )
        self.expresso_files = glob.glob(expresso_pattern, recursive=True)

    def _load_meld(self):
        """Load MELD index and organize by dialogue."""
        meld_index_path = os.path.join(
            self.data_root, "meld/MELD.Raw/meld_index.json"
        )

        if not os.path.exists(meld_index_path):
            print(f"[SpeechSources] MELD index not found at {meld_index_path}")
            return

        with open(meld_index_path, 'r') as f:
            self.meld_index = json.load(f)

        # Organize by dialogue
        meld_audio_root = os.path.join(self.data_root, "meld/MELD.Raw")
        for entry in self.meld_index['entries']:
            key = (entry['split'], entry['dialogue_id'])
            if key not in self.meld_dialogues:
                self.meld_dialogues[key] = []

            # Convert relative path to absolute
            entry['abs_path'] = os.path.join(meld_audio_root, entry['file'])
            self.meld_dialogues[key].append(entry)

        # Sort each dialogue by utterance_id
        for key in self.meld_dialogues:
            self.meld_dialogues[key].sort(key=lambda x: x['utterance_id'])

    def _load_nonverbal(self):
        """Load NonverbalTTS index and organize by NV type."""
        nonverbal_index_path = os.path.join(
            self.data_root, "nonverbal_tts/nonverbal_index.json"
        )

        if not os.path.exists(nonverbal_index_path):
            print(f"[SpeechSources] NonverbalTTS index not found at {nonverbal_index_path}")
            return

        with open(nonverbal_index_path, 'r') as f:
            self.nonverbal_index = json.load(f)

        # Organize by NV type
        nonverbal_root = os.path.join(self.data_root, "nonverbal_tts")
        for entry in self.nonverbal_index['entries']:
            # Get absolute path
            entry['abs_path'] = os.path.join(nonverbal_root, entry['file'])

            # Add to each NV type bucket
            for nv_type in entry.get('nv_types', ['unknown']):
                if nv_type not in self.nonverbal_by_type:
                    self.nonverbal_by_type[nv_type] = []
                self.nonverbal_by_type[nv_type].append(entry)

    def _load_ravdess(self):
        """Load RAVDESS emotional speech dataset."""
        ravdess_index_path = os.path.join(
            self.data_root, "ravdess/ravdess_index.json"
        )

        if not os.path.exists(ravdess_index_path):
            print(f"[SpeechSources] RAVDESS index not found at {ravdess_index_path}")
            return

        with open(ravdess_index_path, 'r') as f:
            self.ravdess_index = json.load(f)

        # Organize by emotion
        ravdess_root = os.path.join(self.data_root, "ravdess")
        for entry in self.ravdess_index['entries']:
            # Get absolute path
            entry['abs_path'] = os.path.join(ravdess_root, entry['file'])

            emotion = entry.get('emotion', 'unknown')
            if emotion not in self.ravdess_by_emotion:
                self.ravdess_by_emotion[emotion] = []
            self.ravdess_by_emotion[emotion].append(entry)

    def _load_ased(self):
        """Load ASED (Amharic Speech Emotional Dataset)."""
        ased_index_path = os.path.join(
            self.data_root, "ased/ased_index.json"
        )

        if not os.path.exists(ased_index_path):
            print(f"[SpeechSources] ASED index not found at {ased_index_path}")
            return

        with open(ased_index_path, 'r') as f:
            self.ased_index = json.load(f)

        # Organize by emotion
        ased_root = os.path.join(self.data_root, "ased")
        for entry in self.ased_index['entries']:
            # Get absolute path
            entry['abs_path'] = os.path.join(ased_root, entry['file'])

            emotion = entry.get('emotion', 'unknown')
            if emotion not in self.ased_by_emotion:
                self.ased_by_emotion[emotion] = []
            self.ased_by_emotion[emotion].append(entry)

    def _build_ased_dialogue(
        self,
        emotion: str,
        num_clips: int = 4,
        min_duration: float = 8.0,
        max_duration: float = 20.0
    ) -> Optional[Tuple[str, float, str, List[str]]]:
        """
        Build an ASED dialogue by concatenating multiple clips.

        Concatenates clips from different speakers of the same emotion.
        ASED clips are ~3 seconds each, so 4 clips = ~12 seconds.

        Args:
            emotion: Target emotion
            num_clips: Number of clips to concatenate (default 4)
            min_duration: Minimum total duration
            max_duration: Maximum total duration

        Returns: (path, duration, emotion, [speaker_ids])
        """
        if emotion not in self.ased_by_emotion:
            return None

        entries = self.ased_by_emotion[emotion].copy()
        random.shuffle(entries)

        # Collect clips from different speakers for variety
        audio_chunks = []
        target_sr = 48000  # Upsample from 16kHz to 48kHz
        silence_samples = int(0.15 * target_sr)  # 0.15s gap between clips
        silence = np.zeros(silence_samples, dtype=np.float32)

        # Loudness normalization
        meter = pyln.Meter(target_sr)
        target_lufs = -23.0

        total_duration = 0.0
        used_clips = []
        used_speakers = set()

        for entry in entries:
            if len(used_clips) >= num_clips:
                break

            # Prefer different speakers for variety
            speaker = entry.get('speaker_id', 'unknown')
            if speaker in used_speakers and len(entries) > num_clips * 2:
                continue

            path = entry['abs_path']
            if not os.path.exists(path):
                continue

            try:
                audio, sr = sf.read(path, dtype='float32')

                # Resample from 16kHz to 48kHz
                if sr != target_sr:
                    from scipy import signal
                    audio = signal.resample(audio, int(len(audio) * target_sr / sr))

                # Ensure mono
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)

                # Normalize loudness
                try:
                    current_lufs = meter.integrated_loudness(audio)
                    if current_lufs > -70.0:
                        gain_db = target_lufs - current_lufs
                        gain_linear = 10 ** (gain_db / 20.0)
                        audio = audio * gain_linear
                except:
                    pass

                # Add silence before (except first)
                if audio_chunks:
                    audio_chunks.append(silence)
                    total_duration += 0.15

                audio_chunks.append(audio)
                total_duration += len(audio) / target_sr
                used_clips.append(entry['file'])
                used_speakers.add(speaker)

            except Exception:
                continue

        if total_duration < min_duration or len(used_clips) < 2:
            return None

        # Concatenate and save
        concatenated = np.concatenate(audio_chunks)

        # Generate cache filename
        cache_hash = hash(tuple(used_clips)) & 0xFFFFFFFF
        cache_filename = f"ased_{emotion}_{len(used_clips)}clips_{cache_hash:08x}.wav"
        cache_path = os.path.join(self.cache_dir, cache_filename)

        sf.write(cache_path, concatenated, target_sr)

        return (cache_path, total_duration, emotion, list(used_speakers))

    def _get_ased_dialogue(
        self,
        min_duration: float = 8.0,
        max_duration: float = 20.0,
        emotion: Optional[str] = None
    ) -> Optional[DialogueInfo]:
        """Get an ASED dialogue (4 clips concatenated, same emotion)."""
        if not self.ased_by_emotion:
            return None

        # Select emotion
        if emotion and emotion in self.ased_by_emotion:
            target_emotion = emotion
        else:
            target_emotion = random.choice(list(self.ased_by_emotion.keys()))

        # Build dialogue with 4 clips
        result = self._build_ased_dialogue(
            target_emotion, num_clips=4,
            min_duration=min_duration, max_duration=max_duration
        )

        if result:
            path, duration, em, speakers = result
            return DialogueInfo(
                path=path,
                source="ased",
                emotion=em,
                style=f"ased_{em}",
                duration_sec=duration,
                speakers=[f"speaker_{s}" for s in speakers],
                utterances=[],
                metadata={
                    "language": "amharic",
                    "num_clips": 4,
                    "speakers": speakers
                }
            )

        return None

    def _build_ravdess_dialogue(
        self,
        emotion: str,
        min_duration: float = 10.0,
        max_duration: float = 30.0
    ) -> Optional[Tuple[str, float, str, List[int]]]:
        """
        Build a RAVDESS dialogue using two actors of the same emotion.

        Simulates a conversation by interleaving clips from two different actors.

        Returns: (path, duration, emotion, [actor1, actor2])
        """
        if emotion not in self.ravdess_by_emotion:
            return None

        entries = self.ravdess_by_emotion[emotion]

        # Group by actor
        by_actor = {}
        for e in entries:
            actor = e.get('actor', 0)
            if actor not in by_actor:
                by_actor[actor] = []
            by_actor[actor].append(e)

        # Need at least 2 actors
        actors = list(by_actor.keys())
        if len(actors) < 2:
            return None

        # Pick 2 random actors
        actor1, actor2 = random.sample(actors, 2)
        clips1 = by_actor[actor1].copy()
        clips2 = by_actor[actor2].copy()
        random.shuffle(clips1)
        random.shuffle(clips2)

        # Interleave clips to simulate dialogue
        audio_chunks = []
        sample_rate = 48000
        silence_samples = int(0.3 * sample_rate)  # 0.3s gap between turns
        silence = np.zeros(silence_samples, dtype=np.float32)

        # Loudness normalization - target -23 LUFS for dialogue
        meter = pyln.Meter(sample_rate)
        target_lufs = -23.0

        total_duration = 0.0
        clip_idx = 0
        used_clips = []

        while total_duration < max_duration:
            # Alternate between actors
            if clip_idx % 2 == 0:
                if not clips1:
                    break
                entry = clips1.pop(0)
            else:
                if not clips2:
                    break
                entry = clips2.pop(0)

            path = entry['abs_path']
            if not os.path.exists(path):
                continue

            try:
                audio, sr = sf.read(path, dtype='float32')

                if sr != sample_rate:
                    from scipy import signal
                    audio = signal.resample(audio, int(len(audio) * sample_rate / sr))

                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)

                # Normalize loudness to target LUFS
                try:
                    current_lufs = meter.integrated_loudness(audio)
                    if current_lufs > -70.0:  # Only normalize if not silence
                        gain_db = target_lufs - current_lufs
                        gain_linear = 10 ** (gain_db / 20.0)
                        audio = audio * gain_linear
                except:
                    pass  # Skip normalization if it fails

                # Add silence before (except first)
                if audio_chunks:
                    audio_chunks.append(silence)
                    total_duration += 0.3

                audio_chunks.append(audio)
                total_duration += len(audio) / sample_rate
                used_clips.append(entry['file'])
                clip_idx += 1

            except Exception:
                continue

        if total_duration < min_duration or not audio_chunks:
            return None

        # Concatenate and save
        concatenated = np.concatenate(audio_chunks)

        # Generate cache filename
        cache_hash = hash(tuple(used_clips[:4])) & 0xFFFFFFFF
        cache_filename = f"ravdess_{emotion}_a{actor1}_a{actor2}_{cache_hash:08x}.wav"
        cache_path = os.path.join(self.cache_dir, cache_filename)

        sf.write(cache_path, concatenated, sample_rate)

        return (cache_path, total_duration, emotion, [actor1, actor2])

    def _get_ravdess_dialogue(
        self,
        min_duration: float = 10.0,
        max_duration: float = 30.0,
        emotion: Optional[str] = None
    ) -> Optional[DialogueInfo]:
        """Get a RAVDESS dialogue (two actors, same emotion, interleaved)."""
        if not self.ravdess_by_emotion:
            return None

        # Select emotion
        if emotion and emotion in self.ravdess_by_emotion:
            target_emotion = emotion
        else:
            target_emotion = random.choice(list(self.ravdess_by_emotion.keys()))

        # Build dialogue with two actors
        result = self._build_ravdess_dialogue(
            target_emotion, min_duration, max_duration
        )

        if result:
            path, duration, em, actors = result
            return DialogueInfo(
                path=path,
                source="ravdess",
                emotion=em,
                style=f"ravdess_{em}",
                duration_sec=duration,
                speakers=[f"actor_{a}" for a in actors],
                utterances=[],
                metadata={
                    "actors": actors,
                    "intensity": "mixed"
                }
            )

        return None

    def _get_audio_duration(self, path: str) -> float:
        """Get audio duration in seconds."""
        try:
            info = sf.info(path)
            return info.frames / info.samplerate
        except:
            return 0.0

    def _get_expresso_style(self, path: str) -> str:
        """Extract style from Expresso path."""
        parts = path.split(os.sep)
        return parts[-2] if len(parts) > 2 else "default"

    def _concatenate_meld_dialogue(
        self,
        entries: List[Dict],
        min_duration: float = 10.0,
        max_duration: float = 120.0
    ) -> Optional[Tuple[str, float, List[str], List[str], str]]:
        """
        Concatenate MELD utterances into a single audio file.

        Returns: (path, duration, speakers, utterances, dominant_emotion)
        """
        if not entries:
            return None

        # Collect audio data
        audio_chunks = []
        speakers = []
        utterances = []
        emotions = []
        sample_rate = 48000

        # Add small silence between utterances (0.3s)
        silence_samples = int(0.3 * sample_rate)
        silence = np.zeros(silence_samples, dtype=np.float32)

        total_duration = 0.0

        for entry in entries:
            path = entry['abs_path']
            if not os.path.exists(path):
                continue

            try:
                audio, sr = sf.read(path, dtype='float32')

                # Resample if needed
                if sr != sample_rate:
                    from scipy import signal
                    audio = signal.resample(audio, int(len(audio) * sample_rate / sr))

                # Ensure mono
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)

                # Add silence before (except first)
                if audio_chunks:
                    audio_chunks.append(silence)
                    total_duration += 0.3

                audio_chunks.append(audio)
                total_duration += len(audio) / sample_rate

                speakers.append(entry['speaker'])
                utterances.append(entry['utterance'])
                emotions.append(entry['emotion'])

                if total_duration >= max_duration:
                    break

            except Exception as e:
                continue

        if total_duration < min_duration or not audio_chunks:
            return None

        # Concatenate and save
        concatenated = np.concatenate(audio_chunks)

        # Determine dominant emotion
        emotion_counts = {}
        for e in emotions:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)

        # Generate cache filename
        dialogue_id = entries[0]['dialogue_id']
        split = entries[0]['split']
        cache_filename = f"meld_{split}_dia{dialogue_id}.wav"
        cache_path = os.path.join(self.cache_dir, cache_filename)

        # Save to cache
        sf.write(cache_path, concatenated, sample_rate)

        return (cache_path, total_duration, list(set(speakers)), utterances, dominant_emotion)

    def get_random_dialogue(
        self,
        source: Optional[str] = None,
        min_duration: float = 10.0,
        max_duration: float = 300.0
    ) -> Optional[DialogueInfo]:
        """
        Get a random dialogue from available sources.

        Args:
            source: "expresso", "ravdess", "ased", or None for random
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds

        Returns:
            DialogueInfo or None if no suitable dialogue found
        """
        if source is None:
            # Weight by available content (MELD disabled - has baked-in SFX)
            sources = []
            if self.expresso_files:
                sources.append("expresso")
            if self.ravdess_by_emotion:
                sources.append("ravdess")
            if self.ased_by_emotion:
                sources.append("ased")

            if not sources:
                return None

            source = random.choice(sources)

        if source == "expresso":
            return self._get_expresso_dialogue(min_duration, max_duration)
        elif source == "ravdess":
            return self._get_ravdess_dialogue(min_duration, max_duration)
        elif source == "ased":
            return self._get_ased_dialogue(min_duration, max_duration)
        elif source == "meld":
            # MELD disabled but keep method for backwards compatibility
            return self._get_meld_dialogue(min_duration, max_duration)

        return None

    def _get_expresso_dialogue(
        self,
        min_duration: float,
        max_duration: float
    ) -> Optional[DialogueInfo]:
        """Get a random Expresso dialogue."""
        if not self.expresso_files:
            return None

        # Try up to 10 random files
        for _ in range(10):
            path = random.choice(self.expresso_files)
            duration = self._get_audio_duration(path)

            if min_duration <= duration <= max_duration:
                style = self._get_expresso_style(path)
                return DialogueInfo(
                    path=path,
                    source="expresso",
                    emotion=style,  # Expresso uses style as emotion proxy
                    style=style,
                    duration_sec=duration,
                    speakers=["speaker"],
                    utterances=[],
                    metadata={"original_path": path}
                )

        return None

    def _get_meld_dialogue(
        self,
        min_duration: float,
        max_duration: float
    ) -> Optional[DialogueInfo]:
        """Get a random MELD dialogue (concatenated)."""
        if not self.meld_dialogues:
            return None

        # Try up to 10 random dialogues
        dialogue_keys = list(self.meld_dialogues.keys())
        random.shuffle(dialogue_keys)

        for key in dialogue_keys[:10]:
            entries = self.meld_dialogues[key]
            result = self._concatenate_meld_dialogue(
                entries, min_duration, max_duration
            )

            if result:
                path, duration, speakers, utterances, emotion = result
                return DialogueInfo(
                    path=path,
                    source="meld",
                    emotion=emotion,
                    style=f"meld_{emotion}",
                    duration_sec=duration,
                    speakers=speakers,
                    utterances=utterances,
                    metadata={
                        "dialogue_id": key[1],
                        "split": key[0],
                        "num_utterances": len(entries)
                    }
                )

        return None

    def get_dialogue_by_emotion(
        self,
        emotion: str,
        source: Optional[str] = None,
        min_duration: float = 10.0,
        max_duration: float = 300.0
    ) -> Optional[DialogueInfo]:
        """
        Get a dialogue with a specific dominant emotion.

        Args:
            emotion: Target emotion (angry, happy, sad, fearful, surprised, disgust, neutral, calm)
            source: "expresso", "ravdess", "ased", or None
            min_duration: Minimum duration
            max_duration: Maximum duration

        Returns:
            DialogueInfo or None
        """
        emotion = emotion.lower()

        # RAVDESS has explicit emotion labels (preferred)
        if source in (None, "ravdess") and self.ravdess_by_emotion:
            # Map common emotion names to RAVDESS labels
            emotion_map = {
                "anger": "angry",
                "joy": "happy",
                "happiness": "happy",
                "sadness": "sad",
                "fear": "fearful",
                "surprise": "surprised",
            }
            ravdess_emotion = emotion_map.get(emotion, emotion)

            if ravdess_emotion in self.ravdess_by_emotion:
                result = self._get_ravdess_dialogue(min_duration, max_duration, emotion=ravdess_emotion)
                if result:
                    return result

        # ASED has explicit emotion labels (5 emotions)
        if source in (None, "ased") and self.ased_by_emotion:
            # Map common emotion names to ASED labels
            ased_emotion_map = {
                "anger": "angry",
                "joy": "happy",
                "happiness": "happy",
                "sadness": "sad",
                "fear": "fearful",
            }
            ased_emotion = ased_emotion_map.get(emotion, emotion)

            if ased_emotion in self.ased_by_emotion:
                result = self._get_ased_dialogue(min_duration, max_duration, emotion=ased_emotion)
                if result:
                    return result

        # Fallback to Expresso (map emotion to style)
        if source in (None, "expresso") and self.expresso_files:
            # Expresso styles that might match emotions
            emotion_style_map = {
                "anger": ["angry", "frustrated"],
                "joy": ["happy", "laughing", "excited"],
                "sadness": ["sad", "sympathetic"],
                "fear": ["scared", "whisper"],
                "surprise": ["surprised", "confused"],
                "disgust": ["disgusted", "sarcastic"],
                "neutral": ["default", "neutral", "projected"],
            }

            target_styles = emotion_style_map.get(emotion, [emotion])
            matching_files = [
                f for f in self.expresso_files
                if any(s in f.lower() for s in target_styles)
            ]

            if matching_files:
                for _ in range(10):
                    path = random.choice(matching_files)
                    duration = self._get_audio_duration(path)
                    if min_duration <= duration <= max_duration:
                        style = self._get_expresso_style(path)
                        return DialogueInfo(
                            path=path,
                            source="expresso",
                            emotion=emotion,
                            style=style,
                            duration_sec=duration,
                            speakers=["speaker"],
                            utterances=[],
                            metadata={"target_emotion": emotion}
                        )

        return None

    def get_meld_emotions(self) -> Dict[str, int]:
        """Get available MELD emotions and their counts."""
        if self.meld_index:
            return self.meld_index.get('emotions', {})
        return {}

    def get_nonverbal_types(self) -> Dict[str, int]:
        """Get available NonverbalTTS NV types and their counts."""
        return {nv_type: len(entries) for nv_type, entries in self.nonverbal_by_type.items()}

    def get_nonverbal_clip(
        self,
        nv_type: Optional[str] = None,
        emotion: Optional[str] = None,
        min_duration: float = 0.5,
        max_duration: float = 10.0
    ) -> Optional[NonverbalClipInfo]:
        """
        Get a nonverbal vocalization clip.

        Args:
            nv_type: Type of vocalization (laugh, cough, sigh, sneeze, grunt, etc.)
                     If None, returns any type.
            emotion: Filter by emotion (happy, sad, angry, etc.)
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds

        Returns:
            NonverbalClipInfo or None
        """
        if not self.nonverbal_by_type:
            return None

        # Get candidate entries
        if nv_type and nv_type in self.nonverbal_by_type:
            candidates = self.nonverbal_by_type[nv_type]
        elif nv_type:
            # Try fuzzy match
            candidates = []
            for key, entries in self.nonverbal_by_type.items():
                if nv_type in key or key in nv_type:
                    candidates.extend(entries)
        else:
            # Any type
            candidates = [e for entries in self.nonverbal_by_type.values() for e in entries]

        if not candidates:
            return None

        # Filter by emotion if specified
        if emotion:
            emotion = emotion.lower()
            candidates = [e for e in candidates if e.get('emotion', '').lower() == emotion]

        if not candidates:
            return None

        # Filter by duration and pick random
        random.shuffle(candidates)
        for entry in candidates[:20]:
            duration = entry.get('duration', 0)
            if min_duration <= duration <= max_duration:
                path = entry['abs_path']
                if os.path.exists(path):
                    return NonverbalClipInfo(
                        path=path,
                        nv_type=entry.get('nv_types', ['unknown'])[0],
                        emotion=entry.get('emotion'),
                        duration_sec=duration,
                        text=entry.get('text', ''),
                        speaker_id=entry.get('speaker_id', 'unknown'),
                        gender=entry.get('gender', 'unknown'),
                        metadata={
                            "split": entry.get('split'),
                            "dnsmos": entry.get('dnsmos'),
                            "source": entry.get('source')
                        }
                    )

        return None

    def get_random_nonverbal_dialogue(
        self,
        min_duration: float = 10.0,
        max_duration: float = 60.0
    ) -> Optional[DialogueInfo]:
        """
        Get a random NonverbalTTS clip as a DialogueInfo (for use as speech source).
        NonverbalTTS clips with speech + nonverbal sounds.

        Args:
            min_duration: Minimum duration
            max_duration: Maximum duration

        Returns:
            DialogueInfo or None
        """
        if not self.nonverbal_index:
            return None

        entries = self.nonverbal_index.get('entries', [])
        if not entries:
            return None

        # Filter by duration
        valid_entries = [
            e for e in entries
            if min_duration <= e.get('duration', 0) <= max_duration
        ]

        if not valid_entries:
            return None

        # Pick random
        entry = random.choice(valid_entries)
        path = entry['abs_path']

        if not os.path.exists(path):
            return None

        return DialogueInfo(
            path=path,
            source="nonverbal",
            emotion=entry.get('emotion'),
            style=f"nonverbal_{entry.get('nv_types', ['unknown'])[0]}",
            duration_sec=entry.get('duration', 0),
            speakers=[entry.get('speaker_id', 'unknown')],
            utterances=[entry.get('text', '')],
            metadata={
                "nv_types": entry.get('nv_types'),
                "gender": entry.get('gender'),
                "split": entry.get('split')
            }
        )

    def get_ravdess_emotions(self) -> Dict[str, int]:
        """Get available RAVDESS emotions and their counts."""
        return {emotion: len(entries) for emotion, entries in self.ravdess_by_emotion.items()}

    def get_ased_emotions(self) -> Dict[str, int]:
        """Get available ASED emotions and their counts."""
        return {emotion: len(entries) for emotion, entries in self.ased_by_emotion.items()}

    def get_stats(self) -> Dict:
        """Get statistics about available speech sources."""
        return {
            "expresso": {
                "total_files": len(self.expresso_files),
                "available": len(self.expresso_files) > 0
            },
            "ravdess": {
                "total_clips": sum(len(v) for v in self.ravdess_by_emotion.values()),
                "emotions": self.get_ravdess_emotions(),
                "available": len(self.ravdess_by_emotion) > 0
            },
            "ased": {
                "total_clips": sum(len(v) for v in self.ased_by_emotion.values()),
                "emotions": self.get_ased_emotions(),
                "language": "amharic",
                "clips_per_scene": 4,
                "available": len(self.ased_by_emotion) > 0
            },
            "meld": {
                "total_dialogues": len(self.meld_dialogues),
                "total_utterances": sum(len(v) for v in self.meld_dialogues.values()),
                "emotions": self.get_meld_emotions(),
                "available": False,  # Disabled - has baked-in SFX
                "note": "disabled - contains TV production audio"
            },
            "nonverbal": {
                "total_clips": self.nonverbal_index['total_files'] if self.nonverbal_index else 0,
                "nv_types": self.get_nonverbal_types(),
                "available": len(self.nonverbal_by_type) > 0
            }
        }


# Quick test
if __name__ == "__main__":
    from . import config
    manager = SpeechSourceManager(str(config.data_root()))
    print("\nStats:", json.dumps(manager.get_stats(), indent=2))

    print("\n--- Testing random RAVDESS dialogue ---")
    dialogue = manager.get_random_dialogue(source="ravdess", min_duration=10.0)
    if dialogue:
        print(f"Source: {dialogue.source}")
        print(f"Path: {dialogue.path}")
        print(f"Duration: {dialogue.duration_sec:.1f}s")
        print(f"Emotion: {dialogue.emotion}")
        print(f"Gender: {dialogue.metadata.get('gender')}")

    print("\n--- Testing random ASED dialogue ---")
    dialogue = manager.get_random_dialogue(source="ased", min_duration=8.0)
    if dialogue:
        print(f"Source: {dialogue.source}")
        print(f"Path: {dialogue.path}")
        print(f"Duration: {dialogue.duration_sec:.1f}s")
        print(f"Emotion: {dialogue.emotion}")
        print(f"Language: {dialogue.metadata.get('language')}")
        print(f"Num clips: {dialogue.metadata.get('num_clips')}")

    print("\n--- Testing emotion-based dialogue (angry) ---")
    dialogue = manager.get_dialogue_by_emotion("angry", min_duration=10.0)
    if dialogue:
        print(f"Source: {dialogue.source}")
        print(f"Emotion: {dialogue.emotion}")
        print(f"Duration: {dialogue.duration_sec:.1f}s")

    print("\n--- Testing nonverbal clip (laugh) ---")
    nv_clip = manager.get_nonverbal_clip(nv_type="laugh")
    if nv_clip:
        print(f"Type: {nv_clip.nv_type}")
        print(f"Emotion: {nv_clip.emotion}")
        print(f"Duration: {nv_clip.duration_sec:.1f}s")
        print(f"Text: {nv_clip.text[:80]}...")

    print("\n--- Testing nonverbal clip (cough) ---")
    nv_clip = manager.get_nonverbal_clip(nv_type="cough")
    if nv_clip:
        print(f"Type: {nv_clip.nv_type}")
        print(f"Duration: {nv_clip.duration_sec:.1f}s")

    print("\n--- Testing nonverbal dialogue ---")
    dialogue = manager.get_random_nonverbal_dialogue(min_duration=3.0, max_duration=30.0)
    if dialogue:
        print(f"Source: {dialogue.source}")
        print(f"Duration: {dialogue.duration_sec:.1f}s")
        print(f"Style: {dialogue.style}")
