# Architecture

Deep-dive into the CineAudioGen pipeline. For install/quickstart see the
[README](../README.md).

## Pipeline flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        PIPELINE FLOW                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐               │
│   │ Director │────▶│  Engine  │────▶│  Critic  │               │
│   │ (Gemini) │     │  (DSP)   │     │ (Gemini) │               │
│   └──────────┘     └──────────┘     └──────────┘               │
│        │                │                │                      │
│        ▼                ▼                ▼                      │
│   Scene Plan       Rough Mix      DSP Adjustments              │
│   - Music tag      - Basic FX     - Per-stem gains             │
│   - Ambience       - Sidechain    - Reverb settings            │
│   - SFX events     - Mastering    - Per-SFX mixing             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

| Component | Module | Role |
|-----------|--------|------|
| **Director** | `cineaudiogen/director.py` | LLM agent: analyzes dialogue audio (on audio-capable backends), selects a music tag, describes ambience, places timestamped SFX events. Retrieves SFX/ambience via CLAP cosine similarity and music via tag lookup. |
| **Engine** | `cineaudiogen/engine.py` | All DSP: loading/resampling, per-stem chains (gain, low-cut, compression, 14 reverb presets), envelope-following sidechain ducking (numba-JIT), mastering, stem export. |
| **Critic** | `cineaudiogen/critic.py` | LLM agent: listens to the rough mix (on audio-capable backends) alongside per-stem measurements (LUFS, LRA, true peak, RMS, crest) and returns **delta** adjustments per stem and per SFX event. |
| **LLM backends** | `cineaudiogen/llm.py` | Provider abstraction for the two agents: Gemini (default, native audio), OpenAI (`gpt-4o-audio`), any OpenAI-compatible endpoint (local models), Anthropic Claude (metrics-only). Backends declare `supports_audio`; agents adapt prompts so text-only models mix from the measurements. |
| **Scene types** | `cineaudiogen/scene_types.py` | Six weighted archetypes controlling stem presence, SFX density, gain offsets, ducking. |
| **Speech sources** | `cineaudiogen/speech_sources.py` | Unified loader for dialogue corpora (Expresso; loaders included for RAVDESS, ASED, NonverbalTTS). Short clips are concatenated into pseudo-dialogues. |
| **Validation** | `cineaudiogen/validate.py` | Dataset QA: additivity, loudness ranges, duration match, NaN/silence/corruption checks. |
| **Aux targets** | `cineaudiogen/auxiliary_targets.py` | PyTorch dataset + heads for multi-task training (ducking envelope, reverb class, music tag). |

## Batch orchestration (`cineaudiogen/batch.py`)

Two-phase design, resumable at scene granularity:

1. **Planning** (ThreadPoolExecutor — I/O bound): Director → rough mix → Critic per
   scene. The CLAP model (~2.5 GB) and the Gemini client are shared across threads.
   The result is saved immediately as `plan.json` in the scene directory.
2. **Rendering** (ProcessPoolExecutor — CPU bound): each worker reloads audio from
   the paths in `plan.json` (no large arrays are pickled) and renders all outputs.

If a run crashes, re-running skips scenes with `metadata.json` (complete) and
renders scenes that have `plan.json` but no metadata — without re-spending API calls.
With `--s3-bucket`, each finished scene is uploaded and deleted locally, keeping
disk usage flat during large runs.

## Output modes & the sidechain problem

Sidechain ducking (music/SFX attenuate under dialogue) is essential for realistic
cinematic mixes but breaks the additivity assumption (`mix = Σ stems`) that
mask-based separation models rely on. CineAudioGen renders each scene in three
modes that resolve this tension differently:

```
scene/
├── rough/         # Pre-Critic baseline (stems + mastered mix)
├── linear/        # ML TRAINING TARGET — additive
│   ├── mix.wav    # == Σ stems, bit-exact
│   └── gain_envelope.json
├── release/       # Production reference — non-additive
└── metadata.json
```

| Processing step | rough | linear | release |
|-----------------|:-----:|:------:|:-------:|
| Per-stem gain/EQ/reverb | ✓ | ✓ | ✓ |
| Critic adjustments | | ✓ | ✓ |
| Per-SFX event mixing | | ✓ | ✓ |
| **Sidechain baked into stems** | | **✓** | |
| **Sidechain on mix bus** | ✓ | | **✓** |
| Mixbus compression + limiter | ✓ | | ✓ |
| Loudness norm (−27 LUFS) | ✓ | | ✓ |
| Shared-gain peak norm (−1 dBFS) | | ✓ | |

```
Linear (training):
  music_ducked = duck(music, speech)
  sfx_ducked   = duck(sfx, speech)
  mix          = speech + music_ducked + ambience + sfx_ducked   # exact

Release (production):
  mix = master(speech + duck(music, speech) + ambience + duck(sfx, speech))
  # stems exported clean — they intentionally do NOT sum to the mix
```

### Bit-exact additivity

Float-domain additivity is not enough: a hot mix hard-clips when quantized to
16-bit PCM, silently destroying the ground truth. The linear exporter therefore:

1. applies **one shared scalar gain** to the mix and all stems so the peak lands at
   −1 dBFS (a shared scalar distributes over the sum, preserving additivity), then
2. quantizes stems to int16 and writes `mix.wav` as their **exact integer sum**.

Result: `max|mix − Σ stems| = 0` in the stored files, verified per scene and
recorded in `metadata.json` (`additivity_error`, `peak_normalization`).

### Sidechain ducking

Envelope follower with per-sample attack/release smoothing (numba-JIT compiled;
~0.6 s for 25 minutes of 48 kHz audio, with a pure-Python fallback):

| Target | Threshold | Reduction | Attack | Release |
|--------|-----------|-----------|--------|---------|
| Music | −20 dB | −12 dB | 20 ms | 225 ms |
| SFX | −25 dB | −8 dB | 10–15 ms | 150–200 ms |

`linear/gain_envelope.json` stores the resulting automation as sparse keyframes
(only ≥0.5 dB changes), so you can reconstruct un-ducked stems
(`music_original = music_ducked / gain`), visualize the automation, or train models
to predict ducking from speech:

```json
{
  "sample_rate": 48000,
  "duration_sec": 414.14,
  "ducking_enabled": {"music": true, "sfx": true},
  "envelopes": {
    "music": {
      "ducking_params": {"threshold_db": -20.0, "reduction_db": -12.0,
                          "attack_ms": 20, "release_ms": 225},
      "keyframes": [
        {"time": 0.0,  "gain": 1.0,   "gain_db": 0.0},
        {"time": 0.62, "gain": 0.887, "gain_db": -1.04},
        {"time": 0.65, "gain": 0.382, "gain_db": -8.35}
      ]
    }
  }
}
```

## Scene types

Weighted archetypes (`scene_types.py`) vary stem presence and prominence so models
also learn *absence* and unusual balances:

| Type | Weight | Music | SFX events | Ducking |
|------|-------:|:-----:|:----------:|:-------:|
| `balanced` | 30% | always | 2–6 | moderate |
| `music_montage` | 25% | always (prominent) | 0–3 | none |
| `dialogue_heavy` | 20% | 90% | 1–5 | strong |
| `ambient` | 10% | 50% | 2–8 | moderate |
| `action_sfx` | 10% | 80% | 5–12 | aggressive |
| `sparse` | 5% | 30% | 0–2 | moderate |

## Reverb presets

14 presets across environments, applied per stem or per SFX event:

| Indoor | Venue | Industrial | Outdoor |
|---|---|---|---|
| dry, small_room, medium_room, large_room, bathroom | large_hall, cathedral | garage, parking_garage, warehouse, sewer, tunnel | forest, open_field |

## Mastering chain (rough & release mixes)

```
Input → Headroom (−3 dB) → Compressor (−12 dB, 3:1) → Limiter (−2 dBTP)
      → Loudness normalization (−27 LUFS ± 1 LU, −2 dBTP ceiling)
```

## Metadata schema (per scene)

```json
{
  "scene_type": "balanced",
  "scene_plan": {
    "dialogue": ".../dialogue.wav",
    "music": ".../track.mp3",
    "ambience": ".../clip.wav",
    "events": [{"file_path": "...", "timestamp": 5.2, "description": "Door slam"}],
    "gemini_context": {"music_tag": "tense", "ambience_cue": "Urban night"},
    "dialogue_duration_sec": 45.5
  },
  "critic_adjustments": {
    "music": {"gain_db": -3.0, "low_cut_hz": 200},
    "speech": {"compressor": {"threshold": -20, "ratio": 4.0}}
  },
  "sfx_event_settings": {"sfx_0": {"gain_db": -6.0, "reverb": {"type": "large_hall", "wet_amount": 0.3}}},
  "token_usage": {"director": {"...": "..."}, "critic": {"...": "..."}, "total_token_count": 17054},
  "outputs": {
    "rough":   {"stem_paths": {"...": "..."}, "loudness": {"...": "..."}},
    "linear":  {"stem_paths": {"...": "..."}, "additivity_error": 0.0,
                 "peak_normalization": {"applied": true, "gain_db": -3.45}},
    "release": {"stem_paths": {"...": "..."}, "loudness": {"...": "..."}}
  },
  "linear_additivity_verified": true
}
```

## Training notes

Use `linear/` for training — `mix.wav` as input, stems as targets:

- Additivity enables a **mixture-consistency loss**: `loss += ||mix − Σ predicted||`.
- Ducking baked into stems exposes models to real gain interactions between sources.
- `release/` is the domain-shift evaluation: mastering makes it non-additive, so the
  gap between linear and release scores measures robustness to real mixing.
- Auxiliary targets (ducking envelope regression, reverb classification, music-tag
  classification) are wired up in `auxiliary_targets.py`;
  `examples/train_example.py` shows a full multi-task setup.

## Validation

`cineaudiogen-validate <dir>` checks every scene for: linear additivity, loudness
in range, true peak below ceiling, equal stem durations, NaN/Inf, DC offset,
silence/corruption, and structural completeness. Use `--strict` for tighter
thresholds and `--json report.json` for a machine-readable report.
