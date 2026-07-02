"""
Parallelized batch production pipeline.

Optimizations:
1. ThreadPoolExecutor for Director/Critic (I/O bound - Gemini API)
2. ProcessPoolExecutor for Engine rendering (CPU bound - audio DSP)
3. Batch processing with progress tracking, resumable via per-scene plan.json

Usage:
    cineaudiogen-batch -n 50 -p 4 -r 2
    python -m cineaudiogen.batch ...
"""

import os
import json
import random
import argparse
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Dict, Any
from tqdm import tqdm
import soundfile as sf

from .engine import AudioEngine
from .critic import AudioCritic
from .director import CinematicDirector
from .scene_types import (
    SceneType, select_random_scene_type, get_scene_type_summary,
    DEFAULT_SCENE_TYPE_WEIGHTS
)
from .speech_sources import SpeechSourceManager
from .s3_uploader import S3StreamingUploader
from . import config
from .llm import get_backend

SAMPLE_RATE = config.SAMPLE_RATE


def generate_scene_id() -> str:
    """Generate a short unique ID for a scene (8 chars)."""
    return uuid.uuid4().hex[:8]


@dataclass
class SceneJob:
    """A scene to be processed."""
    dialogue_path: str
    scene_name: str
    scene_dir: str
    style: str
    scene_type: SceneType
    speech_source: str = "expresso"  # "expresso", "ravdess", "ased"
    emotion: Optional[str] = None    # Emotion label (if available from MELD)


@dataclass
class ScenePlan:
    """Output from Director phase - saved to disk for workers to load."""
    job: SceneJob
    plan: Dict[str, Any]
    # Note: raw_stems and events are NOT stored here - too large to pickle
    # Workers reload audio from paths in plan
    rough_settings: Dict
    critic_settings: Dict
    sfx_event_settings: Dict
    critic_token_usage: Optional[Dict]
    # Raw AI responses for debugging
    critic_raw_response: Optional[Dict] = None  # {critique_text, raw_adjustments}
    stem_analysis: Optional[Dict] = None  # Audio metrics sent to critic


def _subtle_random_reverb(dry_probability=0.15):
    """Very subtle default reverb, with chance of completely dry.

    Args:
        dry_probability: Chance of returning None (no reverb). Default 15%.
    """
    if random.random() < dry_probability:
        return None  # Completely dry - no reverb applied
    return {
        "type": "room",
        "wet_amount": round(random.uniform(0.01, 0.03), 3),
    }


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


def save_plan_to_disk(scene_plan: ScenePlan) -> bool:
    """
    Save ScenePlan to disk immediately after planning completes.
    This allows resuming from plan.json if rendering fails.
    """
    try:
        plan_data = {
            "job": {
                "dialogue_path": scene_plan.job.dialogue_path,
                "scene_name": scene_plan.job.scene_name,
                "scene_dir": scene_plan.job.scene_dir,
                "style": scene_plan.job.style,
                "scene_type": scene_plan.job.scene_type.value,
                "speech_source": scene_plan.job.speech_source,
                "emotion": scene_plan.job.emotion,
            },
            "plan": scene_plan.plan,
            "rough_settings": scene_plan.rough_settings,
            "critic_settings": scene_plan.critic_settings,
            "sfx_event_settings": scene_plan.sfx_event_settings,
            "critic_token_usage": scene_plan.critic_token_usage,
            # Raw AI responses for debugging
            "critic_raw_response": scene_plan.critic_raw_response,
            "stem_analysis": scene_plan.stem_analysis,
        }
        plan_path = f"{scene_plan.job.scene_dir}/plan.json"
        with open(plan_path, 'w') as f:
            json.dump(plan_data, f, indent=2)
        return True
    except Exception as e:
        print(f"  [WARN] Failed to save plan.json for {scene_plan.job.scene_name}: {e}")
        return False


def load_plan_from_disk(scene_dir: str) -> Optional[ScenePlan]:
    """
    Load ScenePlan from disk if plan.json exists.
    Returns None if file doesn't exist or is invalid.
    """
    plan_path = f"{scene_dir}/plan.json"
    if not os.path.exists(plan_path):
        return None

    try:
        with open(plan_path, 'r') as f:
            plan_data = json.load(f)

        # Reconstruct SceneJob
        job_data = plan_data["job"]
        job = SceneJob(
            dialogue_path=job_data["dialogue_path"],
            scene_name=job_data["scene_name"],
            scene_dir=job_data["scene_dir"],
            style=job_data["style"],
            scene_type=SceneType(job_data["scene_type"]),
            speech_source=job_data.get("speech_source", "expresso"),
            emotion=job_data.get("emotion"),
        )

        # Reconstruct ScenePlan
        scene_plan = ScenePlan(
            job=job,
            plan=plan_data["plan"],
            rough_settings=plan_data["rough_settings"],
            critic_settings=plan_data["critic_settings"],
            sfx_event_settings=plan_data["sfx_event_settings"],
            critic_token_usage=plan_data.get("critic_token_usage"),
            critic_raw_response=plan_data.get("critic_raw_response"),
            stem_analysis=plan_data.get("stem_analysis"),
        )
        return scene_plan
    except Exception as e:
        print(f"  [WARN] Failed to load plan.json from {scene_dir}: {e}")
        return None


def plan_scene_worker(job: SceneJob, director: CinematicDirector,
                      critic: AudioCritic, engine: AudioEngine) -> Optional[ScenePlan]:
    """
    Worker function for planning phase (Director + Critic).
    Runs in main thread (shares CLAP model).
    """
    try:
        # Director phase
        scene_plan = director.plan_scene(job.dialogue_path, job.style, scene_type=job.scene_type)
        if not scene_plan:
            return None

        # Load raw stems temporarily for rough mix
        dialogue_duration_sec = scene_plan.get('dialogue_duration_sec', 60)

        raw_speech = engine.load_audio(scene_plan['dialogue'], force_mono=True)
        if raw_speech is None:
            return None

        scene_len = raw_speech.shape[1]

        raw_music = None
        if scene_plan.get('music'):
            raw_music = engine.load_audio(scene_plan['music'], target_duration_sec=dialogue_duration_sec)

        raw_ambience = None
        if scene_plan.get('ambience'):
            raw_ambience = engine.load_audio(scene_plan['ambience'], target_duration_sec=dialogue_duration_sec)

        raw_stems = {
            "speech": raw_speech,
            "music": raw_music,
            "ambience": raw_ambience
        }

        # Load SFX events temporarily
        events = []
        for ev in scene_plan.get('events', []):
            if not isinstance(ev, dict):
                continue
            ev_audio = engine.load_audio(ev.get('file_path'), max_duration_sec=5.0)
            if ev_audio is not None:
                events.append({
                    'audio': ev_audio,
                    'timestamp': ev.get('timestamp', 0),
                    'description': ev.get('description', '')
                })

        # Rough settings
        rough_settings = {
            "speech": {
                "compressor": {"threshold": -15, "ratio": 3.0},
                "reverb": _subtle_random_reverb(),
            },
            "music": {"gain_db": -9.0},  # Was -18.0 (too quiet) - now more audible
            "ambience": {"gain_db": -6.0},
            "sfx": {
                "gain_db": -9.0,  # Was -12.0 - now matches music level
                "reverb": _subtle_random_reverb(),
            },
        }

        # Build rough mix for Critic
        rough_dir = f"{job.scene_dir}/rough"
        os.makedirs(rough_dir, exist_ok=True)

        processed_rough = {}
        for name, audio in raw_stems.items():
            if audio is not None:
                processed_rough[name] = engine.process_stem(audio.copy(), rough_settings.get(name, {}))

        sfx_rough = engine._build_sfx_stem(events, scene_len, apply_per_event_fx=False)
        sfx_rough = engine.process_stem(sfx_rough, rough_settings.get('sfx', {}))
        processed_rough['sfx'] = sfx_rough

        speech_processed = engine._pad_to_length(processed_rough['speech'], scene_len)
        rough_mix = engine._build_mixbus_with_sidechain(
            {k: engine._pad_to_length(v, scene_len) for k, v in processed_rough.items()},
            speech_processed
        )
        rough_mix, _ = engine._apply_mastering(rough_mix)

        rough_mix_path = f"{rough_dir}/mix.wav"
        sf.write(rough_mix_path, rough_mix.T, SAMPLE_RATE)

        # Analyze stems for critic context
        stem_analysis = {}
        for stem_name, audio in raw_stems.items():
            if audio is not None:
                stem_analysis[stem_name] = engine.analyze_stem(audio, name=stem_name)
        # Analyze the rough mix too
        stem_analysis["rough_mix"] = engine.analyze_stem(rough_mix, name="rough_mix")

        # Critic phase
        events_for_critic = [
            {"idx": idx, "timestamp": ev.get('timestamp', 0), "description": ev.get('description', '')}
            for idx, ev in enumerate(events)
        ]
        context_info = {
            "style": job.style,
            "mood": scene_plan['gemini_context'].get('music_tag'),
            "rough_settings": rough_settings,
            "events": events_for_critic,
            "stem_analysis": stem_analysis,  # Audio metrics for each stem
        }

        adjustments = critic.critique_mix(rough_mix_path, context_info)

        critic_token_usage = None
        critic_raw_response = None
        if isinstance(adjustments, dict):
            critic_token_usage = adjustments.pop("token_usage", None)
            # Capture raw AI response for debugging
            critic_raw_response = {
                "critique_text": adjustments.pop("critique_text", None),
                "raw_adjustments": adjustments.pop("raw_adjustments", None),
            }

        sfx_event_settings = adjustments.pop('sfx_events', {}) if isinstance(adjustments, dict) else {}

        # Merge settings
        critic_settings = {}
        for stem_name in ['speech', 'music', 'ambience', 'sfx']:
            critic_settings[stem_name] = merge_stem_settings(
                rough_settings.get(stem_name, {}),
                adjustments.get(stem_name, {})
            )

        # Create ScenePlan WITHOUT raw audio - workers will reload from paths
        result = ScenePlan(
            job=job,
            plan=scene_plan,
            rough_settings=rough_settings,
            critic_settings=critic_settings,
            sfx_event_settings=sfx_event_settings,
            critic_token_usage=critic_token_usage,
            critic_raw_response=critic_raw_response,
            stem_analysis=stem_analysis
        )

        # Save plan to disk immediately (allows resume if rendering fails)
        save_plan_to_disk(result)

        return result

    except Exception as e:
        print(f"  [ERROR] Planning {job.scene_name}: {e}")
        return None


def render_scene_worker(scene_plan: ScenePlan) -> Optional[Dict]:
    """
    Worker function for rendering phase (Engine).
    Runs in process pool (CPU bound).

    Reloads audio from disk - avoids pickling large numpy arrays.
    """
    try:
        engine = AudioEngine(sample_rate=SAMPLE_RATE)
        plan = scene_plan.plan

        # Reload raw stems from paths in plan
        dialogue_duration_sec = plan.get('dialogue_duration_sec', 60)

        raw_speech = engine.load_audio(plan['dialogue'], force_mono=True)
        if raw_speech is None:
            return {'scene_name': scene_plan.job.scene_name, 'success': False, 'error': 'Failed to load speech'}

        raw_music = None
        if plan.get('music'):
            raw_music = engine.load_audio(plan['music'], target_duration_sec=dialogue_duration_sec)

        raw_ambience = None
        if plan.get('ambience'):
            raw_ambience = engine.load_audio(plan['ambience'], target_duration_sec=dialogue_duration_sec)

        raw_stems = {
            "speech": raw_speech,
            "music": raw_music,
            "ambience": raw_ambience
        }

        # Reload SFX events
        events = []
        for ev in plan.get('events', []):
            if not isinstance(ev, dict):
                continue
            ev_audio = engine.load_audio(ev.get('file_path'), max_duration_sec=5.0)
            if ev_audio is not None:
                events.append({
                    'audio': ev_audio,
                    'timestamp': ev.get('timestamp', 0),
                    'description': ev.get('description', '')
                })

        scene_type_config = plan.get('scene_type_config', {})

        result = engine.render_scene(
            raw_stems=raw_stems,
            output_dir=scene_plan.job.scene_dir,
            events=events,
            rough_settings=scene_plan.rough_settings,
            critic_settings=scene_plan.critic_settings,
            sfx_event_settings=scene_plan.sfx_event_settings,
            scene_type_config=scene_type_config
        )

        # Save metadata
        director_token_usage = scene_plan.plan.get("token_usage")

        director_total = _total_from_usage(director_token_usage)
        critic_total = _total_from_usage(scene_plan.critic_token_usage)
        total_tokens = None
        if director_total is not None or critic_total is not None:
            total_tokens = int((director_total or 0) + (critic_total or 0))

        metadata = {
            "scene_type": scene_plan.plan.get('scene_type'),
            "scene_type_config": scene_plan.plan.get('scene_type_config'),
            "scene_plan": scene_plan.plan,
            "critic_adjustments": scene_plan.critic_settings,
            "sfx_event_settings": scene_plan.sfx_event_settings,
            "token_usage": {
                "director": director_token_usage,
                "critic": scene_plan.critic_token_usage,
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

        with open(f"{scene_plan.job.scene_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        return {
            'scene_name': scene_plan.job.scene_name,
            'scene_type': scene_plan.job.scene_type,
            'success': True
        }

    except Exception as e:
        print(f"  [ERROR] Rendering {scene_plan.job.scene_name}: {e}")
        return {
            'scene_name': scene_plan.job.scene_name,
            'scene_type': scene_plan.job.scene_type,
            'success': False,
            'error': str(e)
        }


def run_parallel_production(
    max_scenes: Optional[int] = None,
    planning_workers: int = 4,
    rendering_workers: int = 4,
    scene_type_weights: Optional[Dict[SceneType, float]] = None,
    s3_bucket: Optional[str] = None,
    s3_prefix: str = "cinematic_audio_scenes"
):
    """
    Run parallelized production pipeline.

    Args:
        max_scenes: Maximum number of scenes to process (None = all)
        planning_workers: Number of threads for Director/Critic (I/O bound)
        rendering_workers: Number of processes for Engine (CPU bound)
        scene_type_weights: Custom scene type distribution
    """
    print("\n=== CINEAUDIOGEN: PARALLEL PRODUCTION ===\n")

    backend = get_backend()
    output_root = str(config.output_dir())
    os.makedirs(output_root, exist_ok=True)

    weights = scene_type_weights or DEFAULT_SCENE_TYPE_WEIGHTS
    print("[CONFIG] Scene Type Distribution:")
    for st, w in weights.items():
        print(f"         {st.value}: {w*100:.0f}%")
    print(f"\n[CONFIG] Planning workers: {planning_workers} (threads)")
    print(f"[CONFIG] Rendering workers: {rendering_workers} (processes)")

    # Initialize S3 uploader if configured
    s3_uploader = None
    if s3_bucket:
        print(f"[CONFIG] S3 streaming enabled: s3://{s3_bucket}/{s3_prefix}")
        print(f"[CONFIG] Local files will be deleted after upload\n")
        s3_uploader = S3StreamingUploader(s3_bucket, s3_prefix)
    else:
        print(f"[CONFIG] S3 streaming disabled - files will remain local\n")

    # Initialize shared resources (Director has heavy CLAP model)
    print("[INIT] Initializing Virtual Studio...")
    director = CinematicDirector(backend)
    engine = AudioEngine(sample_rate=SAMPLE_RATE)  # For loading audio in planning phase
    critic = AudioCritic(backend)

    # Initialize speech source manager
    print("[INIT] Loading speech sources...")
    speech_manager = SpeechSourceManager(str(config.data_root()))
    speech_stats = speech_manager.get_stats()

    # Categorize scenes: complete, has_plan (needs render only), needs_planning
    jobs_needing_planning = []
    plans_needing_render = []
    skipped_complete = 0

    # First, scan existing scene folders for plan.json files to resume
    print("[SCAN] Scanning for existing scenes...")
    existing_scene_dirs = [d for d in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, d))]
    for scene_folder in existing_scene_dirs:
        scene_dir = os.path.join(output_root, scene_folder)

        # Skip if fully complete
        if os.path.exists(f"{scene_dir}/metadata.json"):
            skipped_complete += 1
            continue

        # Check if plan.json exists (planning done, rendering failed)
        existing_plan = load_plan_from_disk(scene_dir)
        if existing_plan:
            plans_needing_render.append(existing_plan)
            if max_scenes and len(plans_needing_render) >= max_scenes:
                break

    # Now create new scenes from speech sources (50% Expresso, 50% MELD)
    print("[SCAN] Generating new scene jobs...")
    scenes_needed = max_scenes - len(plans_needing_render) if max_scenes else 100

    for i in range(scenes_needed):
        if max_scenes and (len(jobs_needing_planning) + len(plans_needing_render)) >= max_scenes:
            break

        # Use Expresso only
        source = "expresso"
        min_dur, max_dur = 15.0, 300.0

        dialogue_info = speech_manager.get_random_dialogue(
            source=source,
            min_duration=min_dur,
            max_duration=max_dur
        )

        if dialogue_info is None:
            continue

        # Generate unique scene name
        source_prefix = {"expresso": "expr", "ravdess": "ravd", "ased": "ased", "meld": "meld"}.get(dialogue_info.source, "unk")
        emotion_tag = dialogue_info.emotion or "default"
        scene_id = generate_scene_id()
        scene_name = f"{source_prefix}_{emotion_tag}_{scene_id}"
        scene_dir = os.path.join(output_root, scene_name)

        # Create new scene directory
        os.makedirs(scene_dir, exist_ok=True)

        scene_type = select_random_scene_type(weights)

        jobs_needing_planning.append(SceneJob(
            dialogue_path=dialogue_info.path,
            scene_name=scene_name,
            scene_dir=scene_dir,
            style=dialogue_info.style,
            scene_type=scene_type,
            speech_source=dialogue_info.source,
            emotion=dialogue_info.emotion
        ))

    total_to_process = len(jobs_needing_planning) + len(plans_needing_render)
    expresso_count = speech_stats['expresso']['total_files']
    print(f"       Speech source: {expresso_count} Expresso files (Expresso-only mode)")
    print(f"       Skipped {skipped_complete} complete scenes")
    print(f"       {len(plans_needing_render)} have plan.json (render only)")
    print(f"       {len(jobs_needing_planning)} need planning (API calls)\n")

    if total_to_process == 0:
        print("[DONE] No scenes to process.")
        return

    # Track statistics
    scene_type_counts = {st: 0 for st in SceneType}
    successful = 0
    failed = 0

    # Start with plans that already exist (no API calls needed)
    scene_plans = list(plans_needing_render)
    for plan in scene_plans:
        scene_type_counts[plan.job.scene_type] += 1

    if plans_needing_render:
        print(f"[RESUME] Loaded {len(plans_needing_render)} existing plans from disk")

    # Phase 1: Planning (Director + Critic) - only for scenes without plan.json
    if jobs_needing_planning:
        print(f"[PHASE 1] Planning {len(jobs_needing_planning)} scenes (API calls)...")

        # Parallelize planning - CLAP model and API client are thread-safe
        with ThreadPoolExecutor(max_workers=planning_workers) as executor:
            futures = {executor.submit(plan_scene_worker, job, director, critic, engine): job for job in jobs_needing_planning}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Planning"):
                plan = future.result()
                if plan:
                    scene_plans.append(plan)
                    scene_type_counts[plan.job.scene_type] += 1
                else:
                    failed += 1

        print(f"       Planned {len(scene_plans) - len(plans_needing_render)} new scenes, {failed} failed\n")
    else:
        print("[PHASE 1] No new planning needed (all have plan.json)\n")

    if not scene_plans:
        print("[DONE] No scenes to render.")
        return

    # Phase 2: Rendering (Engine) - Process pool for CPU bound
    print(f"[PHASE 2] Rendering {len(scene_plans)} scenes with {rendering_workers} workers...")

    # Track S3 upload statistics
    s3_uploaded_count = 0
    s3_failed_count = 0
    s3_total_size_mb = 0

    with ProcessPoolExecutor(max_workers=rendering_workers) as executor:
        futures = {executor.submit(render_scene_worker, sp): sp for sp in scene_plans}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Rendering"):
            result = future.result()
            if result and result.get('success'):
                successful += 1

                # Upload to S3 and delete local files if configured
                if s3_uploader:
                    scene_plan = futures[future]
                    upload_result = s3_uploader.upload_scene(
                        scene_plan.job.scene_dir,
                        delete_local=True
                    )
                    if upload_result['success']:
                        s3_uploaded_count += 1
                        s3_total_size_mb += upload_result['total_size_mb']
                    else:
                        s3_failed_count += 1
                        print(f"  [S3 Upload] Failed for {upload_result['scene_name']}")
            else:
                failed += 1

    # Print statistics
    print(f"\n[STATS] Results:")
    print(f"        Successful: {successful}")
    print(f"        Failed: {failed}")

    if s3_uploader:
        print(f"\n[STATS] S3 Upload:")
        print(f"        Uploaded: {s3_uploaded_count} scenes ({s3_total_size_mb:.1f} MB)")
        print(f"        Failed: {s3_failed_count}")
        print(f"        S3 location: s3://{s3_bucket}/{s3_prefix}")

    print(f"\n[STATS] Scene Type Distribution:")
    total = sum(scene_type_counts.values())
    for st, count in scene_type_counts.items():
        if count > 0:
            pct = (count / total) * 100 if total > 0 else 0
            print(f"        {st.value}: {count} ({pct:.1f}%)")

    print("\n[DONE] Parallel production complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Parallelized cinematic audio pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-n', '--max-scenes',
        type=int,
        default=None,
        help='Maximum scenes to process (default: all)'
    )
    parser.add_argument(
        '-p', '--planning-workers',
        type=int,
        default=4,
        help='Threads for planning phase (default: 4, Gemini API I/O bound)'
    )
    parser.add_argument(
        '-r', '--rendering-workers',
        type=int,
        default=2,
        help='Processes for rendering phase (default: 2)'
    )
    parser.add_argument(
        '--s3-bucket',
        type=str,
        default=None,
        help='S3 bucket name for streaming upload (enables auto-delete of local files)'
    )
    parser.add_argument(
        '--s3-prefix',
        type=str,
        default='cinematic_audio_scenes',
        help='S3 key prefix/folder (default: cinematic_audio_scenes)'
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

    args = parser.parse_args()

    if args.data_root:
        os.environ["CINEAUDIOGEN_DATA_ROOT"] = args.data_root
    if args.output_dir:
        os.environ["CINEAUDIOGEN_OUTPUT_DIR"] = args.output_dir

    run_parallel_production(
        max_scenes=args.max_scenes,
        planning_workers=args.planning_workers,
        rendering_workers=args.rendering_workers,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix
    )


if __name__ == "__main__":
    main()
