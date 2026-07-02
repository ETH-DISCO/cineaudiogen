"""Generate a single cinematic scene (Director -> Critic -> Engine).

Usage:
    cineaudiogen-scene [-s SCENE_TYPE] [--dialogue path.wav]
    python -m cineaudiogen.generate ...
"""
import os
import glob
import random
import json
import argparse
import gc
from .engine import AudioEngine
from .director import CinematicDirector
from .critic import AudioCritic
from .scene_types import (
    SceneType, select_random_scene_type, get_scene_type_summary,
    get_scene_type_config
)
from . import config

SAMPLE_RATE = config.SAMPLE_RATE


def _json_safe(obj, _seen=None):
    """Make a value JSON-serializable: break reference cycles and coerce numpy /
    other objects to plain types. Prevents 'Circular reference detected' on dump."""
    if _seen is None:
        _seen = set()
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (dict, list, tuple)):
        if id(obj) in _seen:
            return "<circular>"
        _seen = _seen | {id(obj)}
    if isinstance(obj, dict):
        return {str(k): _json_safe(v, _seen) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v, _seen) for v in obj]
    try:
        import numpy as _np
        if isinstance(obj, _np.generic):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return f"<ndarray shape={obj.shape}>"
    except Exception:
        pass
    return str(obj)


def generate_scene(scene_type=None, dialogue_path=None, scene_name=None, style=None):
    """
    Generate one scene end-to-end.

    Args:
        scene_type: Optional SceneType enum. If None, selects randomly.
        dialogue_path: Optional path to a specific dialogue WAV. If None, a random
                       clip is chosen from the speech library.
        scene_name: Optional output folder name. Defaults to the dialogue basename.
        style: Optional style/mood context. Defaults to the dialogue's parent dir.

    Returns:
        Path to the generated scene directory, or None on failure.
    """
    print("\n=== CINEAUDIOGEN: SINGLE SCENE ===\n")

    api_key = config.gemini_api_key()

    # Select scene type
    if scene_type is None:
        scene_type = select_random_scene_type()
    scene_summary = get_scene_type_summary(scene_type)
    print(f"[CONFIG] Scene Type: {scene_summary}\n")

    # 1. Initialize Agents
    print("[1/6] Initializing Virtual Studio...")
    director = CinematicDirector(api_key)
    engine = AudioEngine(sample_rate=SAMPLE_RATE)
    critic = AudioCritic(api_key)

    # 2. Pick a Dialogue (random unless one was provided)
    print("[2/6] Selecting Scene...")

    if dialogue_path is None:
        speech_glob = str(config.data_root() / "speech_expresso" / "expresso"
                          / "audio_48khz" / "conversational" / "**" / "*.wav")
        all_speech = glob.glob(speech_glob, recursive=True)
        if not all_speech:
            print(f"Error: No speech files found under {config.data_root()}. "
                  "Set CINEAUDIOGEN_DATA_ROOT or pass --dialogue.")
            return
        dialogue_path = random.choice(all_speech)

    if scene_name is None:
        scene_name = os.path.splitext(os.path.basename(dialogue_path))[0]

    # Create per-run output folder (named after dialogue file)
    scene_dir = str(config.output_dir() / scene_name)
    os.makedirs(scene_dir, exist_ok=True)

    # Extract Context (Folder Name) unless style was provided
    if style is None:
        parts = dialogue_path.split(os.sep)
        style = parts[-2] if len(parts) > 2 else "default"
    print(f"Target: {scene_name} (Style: {style})")

    # 3. Director Phase (Planning) - pass scene type
    print("[3/6] Director is planning the scene...")
    scene_plan = director.plan_scene(dialogue_path, style, scene_type=scene_type)

    if not scene_plan:
        print("Director failed to generate a plan.")
        return

    print(f"    - Music Tag: {scene_plan['gemini_context'].get('music_tag')}")
    if scene_plan.get('ambience'):
        print(f"    - Ambience: {os.path.basename(scene_plan['ambience'])}")
    else:
        print("    - Ambience: None (scene type excludes)")

    # Planning done — free the Director (CLAP model ~2.5GB) before the heavy render.
    del director
    gc.collect()

    # 4. Load Raw Stems
    print("[4/6] Loading raw stems...")

    # Speech (The Anchor) - downmix to centered mono
    raw_speech = engine.load_audio(scene_plan['dialogue'], force_mono=True)
    if raw_speech is None:
        print("Error: Failed to load speech.")
        return
    scene_len = raw_speech.shape[1]
    dialogue_duration_sec = scene_len / SAMPLE_RATE

    # Music (trim/loop to fit scene length) - may be None for some scene types
    raw_music = None
    if scene_plan.get('music'):
        raw_music = engine.load_audio(scene_plan['music'], target_duration_sec=dialogue_duration_sec)

    # Ambience (loop to fit scene length) - may be None for some scene types
    raw_ambience = None
    if scene_plan.get('ambience'):
        raw_ambience = engine.load_audio(scene_plan['ambience'], target_duration_sec=dialogue_duration_sec)

    # Raw stems dict
    raw_stems = {
        "speech": raw_speech,
        "music": raw_music,
        "ambience": raw_ambience
    }

    # Show what we loaded
    stems_loaded = [k for k, v in raw_stems.items() if v is not None]
    print(f"    - Loaded stems: {', '.join(stems_loaded)}")

    # Load SFX events with audio
    events = []
    if scene_plan.get('events'):
        print(f"    - Loading {len(scene_plan['events'])} SFX events...")
        for ev in scene_plan['events']:
            if not isinstance(ev, dict):
                continue
            ev_audio = engine.load_audio(ev.get('file_path'), max_duration_sec=5.0)
            if ev_audio is not None:
                events.append({
                    'audio': ev_audio,
                    'timestamp': ev.get('timestamp', 0),
                    'description': ev.get('description', '')
                })

    # 5. Define Settings & Get Critic Feedback
    print("[5/6] Building rough mix and getting Critic feedback...")

    def _subtle_random_reverb():
        """Very subtle default reverb (baseline)."""
        return {
            "type": "room",
            "wet_amount": round(random.uniform(0.01, 0.03), 3),
        }

    # Pre-Critic baseline settings
    rough_settings = {
        "speech": {
            "compressor": {"threshold": -15, "ratio": 3.0},
            "reverb": _subtle_random_reverb(),
        },
        "music": {"gain_db": -18.0},
        "ambience": {"gain_db": -6.0},
        "sfx": {
            "gain_db": -12.0,
            "reverb": _subtle_random_reverb(),
        },
    }

    # Build rough mix for Critic to analyze
    rough_dir = f"{scene_dir}/rough"
    os.makedirs(rough_dir, exist_ok=True)

    # Process stems with rough settings for the rough mix
    processed_rough = {}
    for name, audio in raw_stems.items():
        if audio is not None:
            processed_rough[name] = engine.process_stem(audio.copy(), rough_settings.get(name, {}))

    # Build rough SFX (no per-event processing)
    sfx_rough = engine._build_sfx_stem(events, scene_len, apply_per_event_fx=False)
    sfx_rough = engine.process_stem(sfx_rough, rough_settings.get('sfx', {}))
    processed_rough['sfx'] = sfx_rough

    # Build rough mix with sidechain on mixbus (using default ducking for rough)
    speech_processed = engine._pad_to_length(processed_rough['speech'], scene_len)
    rough_mix = engine._build_mixbus_with_sidechain(
        {k: engine._pad_to_length(v, scene_len) for k, v in processed_rough.items()},
        speech_processed
    )
    rough_mix, _ = engine._apply_mastering(rough_mix)

    import soundfile as sf
    rough_mix_path = f"{rough_dir}/mix.wav"
    sf.write(rough_mix_path, rough_mix.T, SAMPLE_RATE)

    # Free the rough-mix working arrays; render_scene rebuilds from raw_stems.
    del processed_rough, rough_mix, speech_processed, sfx_rough
    gc.collect()

    # Build events list for Critic context
    events_for_critic = []
    for idx, ev in enumerate(events):
        events_for_critic.append({
            "idx": idx,
            "timestamp": ev.get('timestamp', 0),
            "description": ev.get('description', '')
        })

    context_info = {
        "style": style,
        "mood": scene_plan['gemini_context'].get('music_tag'),
        "rough_settings": rough_settings,
        "events": events_for_critic,
        "scene_type": scene_type.value,  # Include scene type in context
    }

    # Get Critic feedback
    adjustments = critic.critique_mix(rough_mix_path, context_info)

    # Pull token usage out for metadata
    critic_token_usage = None
    if isinstance(adjustments, dict) and "token_usage" in adjustments:
        critic_token_usage = adjustments.pop("token_usage")

    # Extract per-SFX event settings
    sfx_event_settings = adjustments.pop('sfx_events', {}) if isinstance(adjustments, dict) else {}

    print("    - Applying Producer Notes...")

    def merge_stem_settings(base_settings, delta_settings):
        """Merge base DSP settings with Critic deltas."""
        base = (base_settings or {}).copy()
        mods = (delta_settings or {})

        if isinstance(mods, dict) and "gain_db" in mods:
            base_gain = base.get("gain_db", 0.0)
            try:
                base_gain = float(base_gain)
            except Exception:
                base_gain = 0.0
            try:
                delta_gain = float(mods.get("gain_db", 0.0))
            except Exception:
                delta_gain = 0.0
            base["gain_db"] = base_gain + delta_gain

        if isinstance(mods, dict):
            for k, v in mods.items():
                if k == "gain_db":
                    continue
                base[k] = v

        return base

    # Merge rough settings with Critic adjustments to get final critic_settings
    critic_settings = {}
    for stem_name in ['speech', 'music', 'ambience', 'sfx']:
        critic_settings[stem_name] = merge_stem_settings(
            rough_settings.get(stem_name, {}),
            adjustments.get(stem_name, {})
        )

    # 6. Render All Outputs
    print(f"[6/6] Rendering all outputs to {scene_dir}...")

    # Get scene type config for engine
    scene_type_config = scene_plan.get('scene_type_config', {})

    result = engine.render_scene(
        raw_stems=raw_stems,
        output_dir=scene_dir,
        events=events,
        rough_settings=rough_settings,
        critic_settings=critic_settings,
        sfx_event_settings=sfx_event_settings,
        scene_type_config=scene_type_config
    )

    # Save Metadata
    meta_path = f"{scene_dir}/metadata.json"

    director_token_usage = scene_plan.get("token_usage") if isinstance(scene_plan, dict) else None

    def _total_from_usage(usage_obj):
        if not isinstance(usage_obj, dict):
            return None
        um = usage_obj.get("usage_metadata")
        if not isinstance(um, dict):
            return None
        if isinstance(um.get("total_token_count"), int):
            return um.get("total_token_count")
        p = um.get("prompt_token_count") or 0
        c = um.get("candidates_token_count") or 0
        try:
            return int(p) + int(c)
        except Exception:
            return None

    director_total = _total_from_usage(director_token_usage)
    critic_total = _total_from_usage(critic_token_usage)
    total_tokens = None
    if director_total is not None or critic_total is not None:
        total_tokens = int((director_total or 0) + (critic_total or 0))

    full_metadata = {
        "scene_type": scene_plan.get('scene_type'),
        "scene_type_config": scene_plan.get('scene_type_config'),
        "scene_plan": scene_plan,
        "critic_adjustments": adjustments,
        "sfx_event_settings": sfx_event_settings,
        "token_usage": {
            "director": director_token_usage,
            "critic": critic_token_usage,
            "total_token_count": total_tokens,
        },
        "outputs": {
            "raw": result.get('raw'),
            "rough": result.get('rough'),
            "linear": result.get('linear'),
            "release": result.get('release'),
        },
        "linear_additivity_verified": result.get('linear', {}).get('additivity_error', 1) < 1e-4,
    }
    with open(meta_path, 'w') as f:
        json.dump(_json_safe(full_metadata), f, indent=2)

    print("\n[SUCCESS] Scene complete.")
    print(f"Scene Type: {scene_type.value}")
    print(f"Output structure:")
    print(f"  {scene_dir}/")
    print(f"    rough/   - Pre-Critic stems + mastered mix")
    print(f"    linear/  - Post-Critic stems (ducking baked in), mix = sum(stems)")
    print(f"    release/ - Post-Critic stems (clean), mastered mix")
    print(f"    metadata.json")

    return scene_dir


def parse_scene_type(value):
    """Parse scene type from string."""
    try:
        return SceneType(value)
    except ValueError:
        valid = [st.value for st in SceneType]
        raise argparse.ArgumentTypeError(
            f"Invalid scene type: {value}. Valid options: {', '.join(valid)}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate a single cinematic scene",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scene Types:
  dialogue_heavy  - Speech-centric (default behavior)
  music_montage   - Music-focused, minimal/no speech
  ambient         - Atmospheric, ambience-focused
  action_sfx      - Dense SFX sequences
  sparse          - Minimal elements (train on absence)
  balanced        - Equal stem prominence

Examples:
  cineaudiogen-scene                           # Random scene type
  cineaudiogen-scene --scene-type sparse       # Specific scene type
  cineaudiogen-scene -s music_montage --dialogue my_dialogue.wav
        """
    )
    parser.add_argument(
        '-s', '--scene-type',
        type=parse_scene_type,
        default=None,
        help='Scene type to generate (default: random)'
    )
    parser.add_argument(
        '--dialogue',
        default=None,
        help='Path to a specific dialogue WAV (default: random clip from the speech library)'
    )
    parser.add_argument(
        '--data-root',
        default=None,
        help='Asset library root (overrides CINEAUDIOGEN_DATA_ROOT)'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory (overrides CINEAUDIOGEN_OUTPUT_DIR)'
    )
    parser.add_argument(
        '--list-types',
        action='store_true',
        help='List available scene types and exit'
    )

    args = parser.parse_args()

    if args.data_root:
        os.environ["CINEAUDIOGEN_DATA_ROOT"] = args.data_root
    if args.output_dir:
        os.environ["CINEAUDIOGEN_OUTPUT_DIR"] = args.output_dir

    if args.list_types:
        print("Available scene types:")
        for st in SceneType:
            summary = get_scene_type_summary(st)
            print(f"  {summary}")
        return

    generate_scene(scene_type=args.scene_type, dialogue_path=args.dialogue)


if __name__ == "__main__":
    main()
