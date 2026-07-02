# CineAudioGen

**An agentic pipeline for generating cinematic audio source-separation training data.**

CineAudioGen builds realistic movie-style scenes — dialogue, music, ambience, and sound
effects — and renders them as isolated stems plus mixes. Unlike existing synthetic
datasets that simply sum stems, it applies real production techniques (sidechain
ducking, compression, reverb, mastering) while still guaranteeing a **bit-exact
additive ground truth** for supervised learning: in the `linear` render of every
scene, `mix.wav` is the *exact* sample-wise sum of the four stems.

Two LLM agents drive the creative decisions:

```
 dialogue.wav ──► Director ──► scene plan ──► Engine ──► rough mix ──► Critic
                  (Gemini)     music tag      (DSP)                    (Gemini)
                    +CLAP      ambience cue                               │
                               SFX events                    mixing notes │
                                                 ┌────────────────────────┘
                                                 ▼
                                     Engine renders 3 outputs:
                                     rough/  linear/  release/
```

- **Director** (Gemini, multimodal): listens to the dialogue, picks a music mood tag,
  describes an ambience bed, and places timestamped SFX events. Music is retrieved by
  tag lookup; SFX/ambience by CLAP embedding similarity over your SFX library.
- **Engine** (pedalboard + numpy DSP): loads stems, applies per-stem processing,
  envelope-following sidechain ducking (music/SFX duck under speech), and mastering.
- **Critic** (Gemini, multimodal): listens to the rough mix with per-stem loudness
  measurements and returns delta adjustments (gains, EQ, reverb, compression,
  per-SFX-event settings) that shape the final render.

## Output

Each scene directory contains three renders of the *same* creative content:

```
scene_xyz/
├── rough/      # pre-Critic baseline (stems + mastered mix)
├── linear/     # POST-Critic, ducking baked into stems → mix.wav == Σ stems (bit-exact)
│   └── gain_envelope.json   # the exact ducking automation curves
├── release/    # POST-Critic, clean stems, ducking + mastering on the mix bus
├── plan.json      # resumable scene plan (batch mode)
└── metadata.json  # scene type, DSP settings, SFX timestamps, loudness, additivity
```

`linear/` is the ML training target: stems are peak-normalized with a single shared
gain (so nothing clips at 16-bit) and the mix is written as the exact integer sum of
the stems — `max|mix − Σstems| = 0`. `release/` is the perceptual reference: the same
scene mixed the way a real dub stage would deliver it, where stems intentionally do
*not* sum to the mix. Training on `linear` and evaluating on `release` lets you study
how separation models degrade as mixes move away from additivity.

Scene variety comes from six weighted scene types (`dialogue_heavy`, `music_montage`,
`ambient`, `action_sfx`, `sparse`, `balanced`) that control stem presence, SFX
density, and ducking aggressiveness.

## Installation

```bash
git clone https://github.com/toofaloof/cineaudiogen
cd cineaudiogen
pip install -e .          # or: pip install -e ".[s3]" for streaming uploads
```

Python ≥ 3.12. `ffmpeg` is recommended for mp3 decoding during indexing.

## Setup

**1. LLM provider.** The Director and Critic run on a pluggable LLM backend
(default: Gemini `gemini-3-flash-preview`):

```bash
cp .env.example .env      # edit it, then:
source .env
```

| `CINEAUDIOGEN_LLM_PROVIDER` | Can *listen* to the audio? | Notes |
|---|---|---|
| `gemini` (default) | ✅ native, any length | best default; cheap flash tier |
| `openai` | ✅ up to ~10 min/clip | `gpt-4o-audio` family; audio sent in-request at 16 kHz |
| `openai-compatible` | model-dependent (`CINEAUDIOGEN_LLM_AUDIO=on`) | any OpenAI-style endpoint: OpenRouter, vLLM, Ollama, LM Studio — including local audio models like Qwen2-Audio |
| `anthropic` | ❌ metrics-only | Claude mixes "by the meters": the Critic reasons from per-stem LUFS/LRA/peak/RMS/crest measurements instead of listening |

Backends that can't listen still work — agents adapt their prompts, and the
Critic always receives full loudness measurements. Install provider extras as
needed: `pip install -e ".[openai]"` or `".[anthropic]"`. Override the model
with `CINEAUDIOGEN_LLM_MODEL`.

**2. Asset library.** Point `CINEAUDIOGEN_DATA_ROOT` (default `./data`) at:

```
data/
├── speech_expresso/expresso/audio_48khz/conversational/   # dialogue clips
├── sfx_fsd50k/FSD50K.dev_audio/                           # SFX + ambience pool
└── music/                                                 # any music library
    ├── downloads/*.mp3
    └── attributions.json     # [{"filename": "...", "tags": ["tense", ...]}, ...]
```

| Asset | Suggested source | License |
|---|---|---|
| Dialogue | [Expresso](https://speechbot.github.io/expresso/) | CC BY-NC 4.0 |
| SFX / ambience | [FSD50K](https://zenodo.org/records/4060432) | CC BY 4.0 (per-clip licenses vary) |
| Music | [FMA](https://github.com/mdeff/fma) or any tagged royalty-free library | per-track |

Any 48 kHz dialogue corpus works (`speech_sources.py` also ships loaders for RAVDESS,
ASED, and NonverbalTTS); any folder of audio works as the SFX pool. Music needs the
small `attributions.json` with per-track `tags` so the Director can retrieve by mood.

**3. Build retrieval indices** (CLAP embeddings for SFX, tag lookup for music):

```bash
cineaudiogen-build-indices          # add --skip-sfx to rebuild only the music lookup
```

## Usage

```bash
# One scene, random type
cineaudiogen-scene

# One scene, specific type + your own dialogue file
cineaudiogen-scene -s action_sfx --dialogue my_dialogue.wav

# Batch production: 50 scenes, 4 planning threads, 2 render processes.
# Fully resumable — plans are saved per scene, crashed runs pick up where they left off.
cineaudiogen-batch -n 50 -p 4 -r 2

# Stream scenes to S3-compatible storage and delete local copies as you go
cineaudiogen-batch -n 500 --s3-bucket my-bucket --s3-prefix scenes

# A single 25-minute scene (concatenates same-speaker dialogue clips first)
cineaudiogen-long-scene --minutes 25 -s balanced

# Validate a generated dataset (additivity, loudness, corruption, structure)
cineaudiogen-validate output/ --workers 4
```

Every command accepts `--data-root` / `--output-dir`, or set
`CINEAUDIOGEN_DATA_ROOT` / `CINEAUDIOGEN_OUTPUT_DIR`.

**Cost & performance.** A scene costs roughly 15–20k Gemini tokens (two multimodal
calls). Rendering is CPU-bound; a ~3-minute scene renders in well under a minute, and
the ducking envelope is numba-JIT compiled (~0.6 s for 25 minutes of audio). Batch
mode separates I/O-bound planning (threads) from CPU-bound rendering (processes).
Rendering peaks at roughly 0.5 GB RAM per minute of scene — long scenes need
correspondingly more (see `cineaudiogen-long-scene --help`).

For component details — the sidechain problem, processing matrix, reverb presets,
gain-envelope format, metadata schema, training tips — see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). An example multi-task training setup
(separation + auxiliary ducking/reverb/tag heads) is in
[examples/train_example.py](examples/train_example.py).

## License

The **code** is MIT-licensed (see [LICENSE](LICENSE)).

The **audio you generate** is a derivative of your source assets and inherits their
licenses — e.g. scenes built from Expresso speech are CC BY-NC (non-commercial).
Track the per-asset licenses recorded in each scene's `metadata.json` before
distributing generated data.

## Citation

If you use CineAudioGen in your research, please cite the accompanying paper:

```bibtex
@inproceedings{cineaudiogen2026,
  title  = {Realistic Data Generation and Evaluation for Cinematic Audio Source Separation},
  author = {[authors]},
  year   = {2026},
  note   = {To appear}
}
```
