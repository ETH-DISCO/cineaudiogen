import os
import json
import random
import numpy as np
import scipy.spatial.distance
import soundfile as sf
import laion_clap
import torch

from .scene_types import (
    SceneType, SceneTypeConfig, get_scene_type_config,
    get_director_prompt_additions, get_sfx_event_count,
    should_include_stem, get_scene_type_summary
)

from . import config
from .llm import get_backend, LLMError


class CinematicDirector:
    def __init__(self, backend=None):
        """Args:
            backend: An LLMBackend (see llm.py). None builds one from the
                     CINEAUDIOGEN_LLM_* environment configuration.
        """
        print("  [Director] Initializing Creative Agent...")
        self.backend = backend or get_backend()

        # 1. Load CLAP (Heavy Model)
        print("  [Director] Loading CLAP for SFX retrieval...")
        self.clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
        self.clap_model.load_ckpt()
        self.clap_model.eval()
        
        if torch.cuda.is_available():
            self.clap_model = self.clap_model.cuda()

        # 2. Load Indices
        self._load_indices()

    def _resolve_asset_path(self, p):
        """Resolve relative paths from indices (e.g. './data/...') to absolute paths.

        Indices may have been built with paths relative to the project root or to
        the data root, and the pipeline may run from a different working directory.
        """
        if not p:
            return None

        p = str(p)
        if os.path.isabs(p):
            return p if os.path.exists(p) else None

        # Normalize leading './'
        if p.startswith("./"):
            p = p[2:]

        data_root = config.data_root()
        candidates = []
        # Most common: 'data/...' relative to the project root (data_root's parent)
        if p.startswith("data/"):
            candidates.append(str(data_root.parent / p))

        # Sometimes we might get paths relative to the data root
        candidates.append(str(data_root / p))
        candidates.append(str(data_root.parent / p))

        for c in candidates:
            if os.path.exists(c):
                return c

        return None

    def _load_indices(self):
        """Internal method to load lookup tables."""
        index_dir = config.index_dir()

        # Load SFX Embeddings
        sfx_path = index_dir / "sfx_index.npz"
        if sfx_path.exists():
            data = np.load(sfx_path)
            self.sfx_paths = data['paths']
            self.sfx_embeds = data['embeddings']
        else:
            print(f"[Error] SFX index not found at {sfx_path}. "
                  "Run `cineaudiogen-build-indices` first.")
            self.sfx_paths = None

        # Load Music Metadata
        music_path = index_dir / "music_lookup.json"
        if music_path.exists():
            with open(music_path, 'r') as f:
                self.music_db = json.load(f)
            self.valid_music_tags = list(self.music_db.keys())
        else:
            print(f"[Error] Music lookup not found at {music_path}. "
                  "Run `cineaudiogen-build-indices` first.")
            self.music_db = {}
            self.valid_music_tags = []

    def _get_sfx_match(self, text_prompt, relevance_threshold=0.75):
        """Retrieves best SFX file path using Vector Search.
        
        Args:
            text_prompt: Description of the sound to search for
            relevance_threshold: Maximum cosine distance to accept (0=identical, 2=opposite)
                                 Default 0.75 filters out poor matches.
        
        Returns:
            Path to best matching SFX file, or None if no good match found.
        """
        if not text_prompt or self.sfx_paths is None: 
            return None
            
        with torch.no_grad():
            text_embed = self.clap_model.get_text_embedding([text_prompt])
        
        distances = scipy.spatial.distance.cdist(text_embed, self.sfx_embeds, metric='cosine')
        best_idx = np.argmin(distances)
        best_distance = distances[0, best_idx]
        
        # Reject matches that are too dissimilar
        if best_distance > relevance_threshold:
            print(f"    [SFX] No good match for '{text_prompt}' (best distance: {best_distance:.3f})")
            return None
            
        return self._resolve_asset_path(self.sfx_paths[best_idx])

    def _get_music_match(self, tag_prompt):
        """Retrieves Music file path using Metadata Tags."""
        tag = tag_prompt.lower()
        filepath = None
        
        # 1. Exact Match
        if tag in self.music_db:
            filepath = random.choice(self.music_db[tag])
        
        # 2. Fuzzy Match (tag contains known tag or vice versa)
        if filepath is None:
            for known_tag in self.music_db.keys():
                if known_tag in tag or tag in known_tag:
                    filepath = random.choice(self.music_db[known_tag])
                    break
        
        # 3. Fallback
        if filepath is None and self.valid_music_tags:
            fallback_tag = "cinematic" if "cinematic" in self.music_db else self.valid_music_tags[0]
            filepath = random.choice(self.music_db[fallback_tag])
        
        # The index stores absolute paths
        if filepath and os.path.exists(filepath):
            return filepath

        # Fallback for indices built with data-root-relative paths
        if filepath and not os.path.isabs(filepath):
            resolved = self._resolve_asset_path(filepath)
            if resolved:
                return resolved

        return None

    def _get_audio_duration_sec(self, path):
        """Best-effort audio duration lookup (seconds)."""
        try:
            info = sf.info(path)
            if info.frames and info.samplerate:
                return float(info.frames) / float(info.samplerate)
        except Exception:
            pass
        return None

    def _normalize_event_timestamps(self, events, duration_sec):
        """Ensure timestamps fit within the dialogue duration.

        If Gemini returns timestamps beyond the duration, we scale them down
        to preserve relative placement, then clamp.
        """
        if not events or not duration_sec or duration_sec <= 0:
            return events

        timestamps = []
        for ev in events:
            if not isinstance(ev, dict):
                continue
            ts = ev.get("timestamp")
            if isinstance(ts, (int, float)):
                timestamps.append(float(ts))

        if not timestamps:
            return events

        max_ts = max(timestamps)
        if max_ts <= duration_sec:
            # Still clamp just in case of tiny overshoots
            clamped = []
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                ts = ev.get("timestamp", 0.0)
                ts = float(ts) if isinstance(ts, (int, float)) else 0.0
                ts = max(0.0, min(ts, max(0.0, duration_sec - 0.05)))
                new_ev = dict(ev)
                new_ev["timestamp"] = ts
                clamped.append(new_ev)
            return clamped

        # Scale all timestamps to fit into ~90% of the file, then clamp
        target_max = max(0.0, duration_sec * 0.9)
        scale = (target_max / max_ts) if max_ts > 0 else 0.0

        normalized = []
        for ev in events:
            if not isinstance(ev, dict):
                continue
            ts = ev.get("timestamp", 0.0)
            ts = float(ts) if isinstance(ts, (int, float)) else 0.0
            ts = ts * scale
            ts = max(0.0, min(ts, max(0.0, duration_sec - 0.05)))
            new_ev = dict(ev)
            new_ev["timestamp"] = ts
            normalized.append(new_ev)

        return normalized

    def plan_scene(self, dialogue_path, style_context, scene_type: SceneType = None):
        """
        The Public API called by main.py.
        Returns a dictionary of paths and instructions.

        Args:
            dialogue_path: Path to dialogue audio file
            style_context: Style/mood context string
            scene_type: SceneType enum for varied generation (default: DIALOGUE_HEAVY)
        """
        # Default to dialogue-heavy if not specified
        if scene_type is None:
            scene_type = SceneType.DIALOGUE_HEAVY

        config = get_scene_type_config(scene_type)
        scene_summary = get_scene_type_summary(scene_type)

        print(f"  [Director] Analyzing scene: {os.path.basename(dialogue_path)} ({style_context}) [{scene_summary}]")

        # Determine which stems to include based on scene type
        include_music = should_include_stem(config, 'music')
        include_ambience = should_include_stem(config, 'ambience')
        sfx_max_events = get_sfx_event_count(config)

        # 1. Build prompt based on scene type
        max_tags = min(len(self.valid_music_tags), 100)
        tags_list_str = ", ".join(self.valid_music_tags[:max_tags])

        dialogue_duration_sec = self._get_audio_duration_sec(dialogue_path)
        duration_line = (
            f"Dialogue Duration: {dialogue_duration_sec:.2f} seconds."
            if dialogue_duration_sec is not None
            else "Dialogue Duration: unknown."
        )

        # Scene type specific instructions
        scene_type_guidance = get_director_prompt_additions(config)

        # Build music instruction based on whether we want music
        if include_music:
            music_instruction = "1. Music: Select ONE tag from the list above that fits the scene mood."
        else:
            music_instruction = "1. Music: Set to null (this scene has no music)."

        # Build ambience instruction
        if include_ambience:
            ambience_instruction = "2. Ambience: Describe a continuous background loop appropriate for the scene."
        else:
            ambience_instruction = "2. Ambience: Set to null (this scene has no ambience)."

        # Build SFX instruction with scene-appropriate count
        if sfx_max_events > 0:
            sfx_instruction = f"""3. Events: List {sfx_max_events} or fewer distinct sound events (Foley or World sounds).
           - Keep descriptions concise (2-5 words).
           - IMPORTANT: All timestamps must be within the dialogue duration (0.0 to {dialogue_duration_sec:.1f} seconds)."""
        else:
            sfx_instruction = "3. Events: Set to empty array (this scene has no SFX events)."

        prompt = f"""
        Role: World Class Cinema Sound Supervisor.
        Task: Design audio for a cinematic scene. Make it realistic and immersive.

        Scene Type: {scene_type.value.upper()}
        Style Context: "{style_context.upper()}"
        {duration_line}

        {scene_type_guidance}

        Available Music Tags: [{tags_list_str}]

        Instructions:
        {music_instruction}
        {ambience_instruction}
        {sfx_instruction}

        Take into account what the people are actually saying and the emotional tone.

        Output JSON:
        {{
            "music_tag": "string or null",
            "ambience_cue": "string or null",
            "events": [
                {{"sound": "string description", "timestamp": float}}
            ]
        }}
        """

        # 2. Call the LLM backend (retries, audio handling, and JSON parsing
        # live in llm.py; metrics-only backends never receive the audio).
        if not self.backend.supports_audio:
            prompt += ("\n        Note: You cannot hear the dialogue audio. Base your "
                       "decisions on the style context, scene type, and duration above.\n")
        try:
            cue_sheet, token_usage = self.backend.generate_json(
                prompt, audio_path=dialogue_path, temperature=0.65,
            )
        except LLMError as e:
            print(f"  [Director Error] {e}")
            return None

        # Safety: validate cue_sheet is a dict (Gemini sometimes returns arrays)
        if not isinstance(cue_sheet, dict):
            print(f"  [Director Error] Gemini returned invalid format (expected dict, got {type(cue_sheet).__name__})")
            return None

        # Safety: normalize timestamps to the actual dialogue duration
        if dialogue_duration_sec is not None:
            if isinstance(cue_sheet.get("events"), list):
                cue_sheet["events"] = self._normalize_event_timestamps(
                    cue_sheet["events"],
                    dialogue_duration_sec,
                )

        # 3. Retrieve Assets (respecting null values from the model)
        music_tag = cue_sheet.get('music_tag')
        ambience_cue = cue_sheet.get('ambience_cue')

        # Only retrieve music if tag is provided and not null
        music_path = None
        if music_tag and music_tag.lower() != 'null':
            music_path = self._get_music_match(music_tag)

        # Only retrieve ambience if cue is provided and not null
        ambience_path = None
        if ambience_cue and ambience_cue.lower() != 'null':
            ambience_path = self._get_sfx_match(ambience_cue)

        # Retrieve SFX events
        retrieved_events = []
        events_list = cue_sheet.get('events', [])
        if events_list and isinstance(events_list, list):
            for ev in events_list:
                if not isinstance(ev, dict):
                    continue
                sound_desc = ev.get('sound')
                if not sound_desc:
                    continue
                path = self._get_sfx_match(sound_desc)
                if path:
                    timestamp = round(float(ev.get('timestamp', 0)), 2)
                    retrieved_events.append({
                        "file_path": path,
                        "timestamp": timestamp,
                        "description": sound_desc
                    })

        # 4. Return Manifesto with scene type info
        return {
            "dialogue": dialogue_path,
            "music": music_path,
            "ambience": ambience_path,
            "events": retrieved_events,
            "gemini_context": cue_sheet,
            "dialogue_duration_sec": dialogue_duration_sec,
            "token_usage": token_usage,
            "scene_type": scene_type.value,
            "scene_type_config": {
                "speech_prob": config.speech_prob,
                "music_prob": config.music_prob,
                "ambience_prob": config.ambience_prob,
                "sfx_range": [config.sfx_min_events, config.sfx_max_events],
                "duck_music": config.duck_music_under_speech,
                "duck_sfx": config.duck_sfx_under_speech,
                "gain_offsets": {
                    "speech": config.speech_gain_offset,
                    "music": config.music_gain_offset,
                    "ambience": config.ambience_gain_offset,
                    "sfx": config.sfx_gain_offset,
                },
            },
        }