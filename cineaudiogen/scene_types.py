"""
Scene Type Definitions for Cinematic Audio Pipeline

Scene types create diversity in training data by varying:
- Which stems are present/absent
- Relative prominence of each stem
- Number and density of SFX events
- Mixing parameters and ducking behavior

This is critical for training robust stem separation models that can handle
real-world scenarios beyond simple "all stems present" cases.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict
import random


class SceneType(Enum):
    """
    Available scene types for data generation diversity.

    For ML training, variety in scene composition is essential:
    - Models must learn what's NOT present (silent stems)
    - Models must handle varying stem prominence
    - Models must generalize across mixing styles
    """
    DIALOGUE_HEAVY = "dialogue_heavy"      # Speech-centric (current default)
    MUSIC_MONTAGE = "music_montage"        # Music-focused, minimal/no speech
    AMBIENT_ESTABLISHING = "ambient"       # Atmospheric, ambience-focused
    ACTION_SFX = "action_sfx"              # Dense SFX sequences
    SPARSE = "sparse"                       # Minimal elements (train on absence)
    BALANCED = "balanced"                   # Equal stem prominence


@dataclass
class SceneTypeConfig:
    """Configuration for a scene type."""

    # Stem presence probabilities (0.0 = never, 1.0 = always)
    speech_prob: float = 1.0
    music_prob: float = 1.0
    ambience_prob: float = 1.0

    # SFX event count range
    sfx_min_events: int = 0
    sfx_max_events: int = 5

    # Relative gain adjustments (dB offset from baseline)
    speech_gain_offset: float = 0.0
    music_gain_offset: float = 0.0
    ambience_gain_offset: float = 0.0
    sfx_gain_offset: float = 0.0

    # Ducking behavior
    duck_music_under_speech: bool = True
    duck_sfx_under_speech: bool = True
    music_ducking_reduction_db: float = -12.0
    sfx_ducking_reduction_db: float = -8.0

    # Director prompt guidance
    director_focus: str = "balanced"  # What to emphasize in Gemini prompt

    # Special flags
    use_silence_as_speech: bool = False  # For music_montage: generate silent speech
    loop_short_speech: bool = False       # For sparse: use very short speech clips


# Scene type configurations
SCENE_TYPE_CONFIGS: Dict[SceneType, SceneTypeConfig] = {

    SceneType.DIALOGUE_HEAVY: SceneTypeConfig(
        speech_prob=1.0,
        music_prob=0.9,
        ambience_prob=0.95,
        sfx_min_events=1,
        sfx_max_events=5,
        speech_gain_offset=0.0,
        music_gain_offset=0.0,       # Was -3.0 - no extra suppression needed
        ambience_gain_offset=-3.0,   # Ambience lower
        sfx_gain_offset=0.0,
        duck_music_under_speech=True,
        duck_sfx_under_speech=True,
        music_ducking_reduction_db=-6.0,  # Was -12.0 - gentler ducking
        sfx_ducking_reduction_db=-6.0,    # Was -8.0 - gentler ducking
        director_focus="dialogue_clarity",
    ),

    SceneType.MUSIC_MONTAGE: SceneTypeConfig(
        speech_prob=0.2,             # 20% chance of having any speech
        music_prob=1.0,              # Always have music
        ambience_prob=0.7,           # Often have ambience
        sfx_min_events=0,
        sfx_max_events=3,            # Sparse SFX
        speech_gain_offset=-6.0,     # Speech quieter when present
        music_gain_offset=6.0,       # Music prominent
        ambience_gain_offset=0.0,
        sfx_gain_offset=-3.0,
        duck_music_under_speech=False,  # Don't duck music - it's the focus
        duck_sfx_under_speech=False,
        music_ducking_reduction_db=0.0,
        sfx_ducking_reduction_db=0.0,
        director_focus="music_emotion",
        use_silence_as_speech=True,  # Generate silent speech stem
    ),

    SceneType.AMBIENT_ESTABLISHING: SceneTypeConfig(
        speech_prob=0.3,             # Occasional speech
        music_prob=0.5,              # Sometimes music
        ambience_prob=1.0,           # Always ambience
        sfx_min_events=2,
        sfx_max_events=8,            # More environmental SFX
        speech_gain_offset=-3.0,
        music_gain_offset=-3.0,      # Was -6.0 - less suppression
        ambience_gain_offset=6.0,    # Ambience prominent
        sfx_gain_offset=3.0,         # SFX more audible
        duck_music_under_speech=True,
        duck_sfx_under_speech=False,  # SFX are environmental, not ducked
        music_ducking_reduction_db=-3.0,  # Was -6.0 - even gentler
        sfx_ducking_reduction_db=0.0,
        director_focus="atmosphere",
    ),

    SceneType.ACTION_SFX: SceneTypeConfig(
        speech_prob=0.6,             # Often have speech (action dialogue)
        music_prob=0.8,              # Usually have intense music
        ambience_prob=0.7,
        sfx_min_events=5,
        sfx_max_events=12,           # Dense SFX
        speech_gain_offset=3.0,      # Speech cuts through
        music_gain_offset=0.0,
        ambience_gain_offset=-6.0,   # Ambience lower
        sfx_gain_offset=6.0,         # SFX prominent
        duck_music_under_speech=True,
        duck_sfx_under_speech=True,
        music_ducking_reduction_db=-8.0,  # Was -15.0 - less aggressive
        sfx_ducking_reduction_db=-6.0,    # Was -10.0 - less aggressive
        director_focus="action_intensity",
    ),

    SceneType.SPARSE: SceneTypeConfig(
        speech_prob=0.5,             # Sometimes speech
        music_prob=0.3,              # Rarely music
        ambience_prob=0.4,           # Sometimes ambience
        sfx_min_events=0,
        sfx_max_events=2,            # Very sparse SFX
        speech_gain_offset=0.0,
        music_gain_offset=-3.0,      # Was -12.0 - way too quiet, unlearnable
        ambience_gain_offset=-3.0,   # Was -6.0 - reduce ambience suppression too
        sfx_gain_offset=-3.0,        # Was -6.0
        duck_music_under_speech=True,
        duck_sfx_under_speech=True,
        music_ducking_reduction_db=-6.0,  # Was -12.0
        sfx_ducking_reduction_db=-6.0,    # Was -8.0
        director_focus="minimal",
    ),

    SceneType.BALANCED: SceneTypeConfig(
        speech_prob=1.0,
        music_prob=1.0,
        ambience_prob=1.0,
        sfx_min_events=2,
        sfx_max_events=6,
        speech_gain_offset=0.0,
        music_gain_offset=0.0,
        ambience_gain_offset=0.0,
        sfx_gain_offset=0.0,
        duck_music_under_speech=True,
        duck_sfx_under_speech=True,
        music_ducking_reduction_db=-4.0,  # Was -9.0 - much gentler ducking
        sfx_ducking_reduction_db=-4.0,    # Was -6.0
        director_focus="balanced",
    ),
}


# Default weights for random scene type selection
# Rebalanced to ensure ~55% of clips have prominent music (avoid catastrophic forgetting)
DEFAULT_SCENE_TYPE_WEIGHTS: Dict[SceneType, float] = {
    SceneType.DIALOGUE_HEAVY: 0.20,      # Reduced from 0.35 - music was too suppressed
    SceneType.MUSIC_MONTAGE: 0.25,       # Increased from 0.15 - music is primary element
    SceneType.AMBIENT_ESTABLISHING: 0.10, # Reduced from 0.15
    SceneType.ACTION_SFX: 0.10,          # Reduced from 0.15
    SceneType.SPARSE: 0.05,              # Reduced from 0.10 - avoid silent stems
    SceneType.BALANCED: 0.30,            # Increased from 0.10 - all stems equal prominence
}


def get_scene_type_config(scene_type: SceneType) -> SceneTypeConfig:
    """Get configuration for a scene type."""
    return SCENE_TYPE_CONFIGS.get(scene_type, SCENE_TYPE_CONFIGS[SceneType.DIALOGUE_HEAVY])


def select_random_scene_type(weights: Optional[Dict[SceneType, float]] = None) -> SceneType:
    """
    Select a random scene type based on weights.

    Args:
        weights: Optional custom weights. Uses DEFAULT_SCENE_TYPE_WEIGHTS if None.

    Returns:
        Selected SceneType
    """
    weights = weights or DEFAULT_SCENE_TYPE_WEIGHTS

    scene_types = list(weights.keys())
    probs = list(weights.values())

    # Normalize probabilities
    total = sum(probs)
    probs = [p / total for p in probs]

    return random.choices(scene_types, weights=probs, k=1)[0]


def should_include_stem(config: SceneTypeConfig, stem_name: str) -> bool:
    """
    Determine if a stem should be included based on scene type config.

    Args:
        config: Scene type configuration
        stem_name: One of 'speech', 'music', 'ambience'

    Returns:
        True if stem should be included, False otherwise
    """
    prob_map = {
        'speech': config.speech_prob,
        'music': config.music_prob,
        'ambience': config.ambience_prob,
    }

    prob = prob_map.get(stem_name, 1.0)
    return random.random() < prob


def get_sfx_event_count(config: SceneTypeConfig) -> int:
    """Get random SFX event count based on scene type config."""
    return random.randint(config.sfx_min_events, config.sfx_max_events)


def get_gain_adjustments(config: SceneTypeConfig) -> Dict[str, float]:
    """
    Get gain adjustments for all stems.

    Returns dict with gain offsets in dB for each stem.
    """
    return {
        'speech': config.speech_gain_offset,
        'music': config.music_gain_offset,
        'ambience': config.ambience_gain_offset,
        'sfx': config.sfx_gain_offset,
    }


def get_director_prompt_additions(config: SceneTypeConfig) -> str:
    """
    Get additional prompt text for the Director based on scene type focus.

    Returns string to append to Director prompt.
    """
    focus_prompts = {
        "dialogue_clarity": """
        PRIORITY: Dialogue clarity is paramount. Select music and ambience that won't compete
        with speech frequencies. SFX should support the narrative without masking dialogue.
        Choose subtle, cinematic underscore rather than prominent music.""",

        "music_emotion": """
        PRIORITY: This is a music-focused scene (montage/transition). Music is the primary
        element - select emotionally resonant tracks. Minimize SFX events. If speech exists,
        it should be brief or atmospheric. The music carries the emotional weight.""",

        "atmosphere": """
        PRIORITY: This is an atmospheric/establishing scene. Ambience is the foundation -
        select rich, immersive environmental sounds. SFX should be environmental (birds,
        wind, distant sounds). Music should be subtle drones or absent. Create a sense of place.""",

        "action_intensity": """
        PRIORITY: This is a high-intensity action scene. Select many SFX events - impacts,
        movements, environmental reactions. Music should be driving/tense. Speech cuts through
        chaos. Layer dense sound design that creates urgency and movement.""",

        "minimal": """
        PRIORITY: This is a sparse, quiet scene. Less is more. Select minimal elements -
        perhaps just subtle ambience, occasional SFX. Create tension through silence and
        restraint. Music should be absent or barely perceptible.""",

        "balanced": """
        PRIORITY: Create a well-balanced mix where all elements have appropriate presence.
        No single element dominates. Music supports dialogue, ambience creates space,
        SFX punctuate key moments.""",
    }

    return focus_prompts.get(config.director_focus, "")


def get_scene_type_summary(scene_type: SceneType) -> str:
    """Get a human-readable summary of a scene type for logging."""
    config = get_scene_type_config(scene_type)

    stem_status = []
    if config.speech_prob >= 0.9:
        stem_status.append("speech:always")
    elif config.speech_prob >= 0.5:
        stem_status.append("speech:likely")
    elif config.speech_prob > 0:
        stem_status.append("speech:rare")
    else:
        stem_status.append("speech:never")

    if config.music_prob >= 0.9:
        stem_status.append("music:always")
    elif config.music_prob >= 0.5:
        stem_status.append("music:likely")
    elif config.music_prob > 0:
        stem_status.append("music:rare")
    else:
        stem_status.append("music:never")

    sfx_desc = f"sfx:{config.sfx_min_events}-{config.sfx_max_events}"

    return f"{scene_type.value} [{', '.join(stem_status)}, {sfx_desc}]"
