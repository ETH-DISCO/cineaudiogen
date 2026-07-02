"""Generate one long (e.g. 25-minute) cinematic scene through the normal pipeline.

Scenes are as long as their dialogue, and individual dialogue clips are usually
a few minutes at most, so this module first concatenates clips from a single
directory (same speakers / same style keeps the result coherent) into one long
dialogue, then runs the standard Director -> Critic -> Engine flow on it.

Usage:
    cineaudiogen-long-scene --minutes 25 --scene-type balanced \\
        --speech-dir data/speech_expresso/expresso/audio_48khz/conversational/ex04-ex03/default
    python -m cineaudiogen.long_scene ...

Note: long scenes are memory-hungry (a 25-minute render peaks around ~13 GB
RSS); make sure the machine has enough RAM (or swap) before going very long.
"""
import argparse
import glob
import os

import numpy as np
import soundfile as sf

from . import config
from .scene_types import SceneType
from .generate import generate_scene, parse_scene_type

SR = config.SAMPLE_RATE


def build_dialogue(speech_dir, out_path, target_sec, gap_sec=0.4):
    """Concatenate clips from one directory with short gaps to >= target_sec,
    then trim to exactly target_sec. Mono, 48 kHz, 16-bit."""
    clips = sorted(glob.glob(os.path.join(speech_dir, "*.wav")))
    if not clips:
        raise SystemExit(f"no .wav clips found in {speech_dir}")
    target_samples = int(target_sec * SR)
    gap = np.zeros(int(gap_sec * SR), dtype=np.float32)
    parts, total, used = [], 0, 0
    for c in clips:
        x, sr = sf.read(c, dtype="float32")
        if sr != SR:
            raise SystemExit(f"unexpected sample rate {sr} in {c} (expected {SR})")
        if x.ndim == 2:
            x = x.mean(axis=1)
        parts.append(x)
        parts.append(gap)
        total += len(x) + len(gap)
        used += 1
        if total >= target_samples:
            break
    if total < target_samples:
        raise SystemExit(
            f"only {total / SR:.0f}s of speech available in {speech_dir}, "
            f"need {target_sec:.0f}s — point --speech-dir at a larger clip set"
        )
    sig = np.concatenate(parts)[:target_samples]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, sig, SR, subtype="PCM_16")
    return len(sig) / SR, used


def main():
    parser = argparse.ArgumentParser(
        description="Generate one long scene by concatenating dialogue clips"
    )
    parser.add_argument('--minutes', type=float, default=25.0,
                        help='Target scene length in minutes (default: 25)')
    parser.add_argument('-s', '--scene-type', type=parse_scene_type,
                        default=SceneType.BALANCED,
                        help='Scene type (default: balanced)')
    parser.add_argument('--speech-dir', default=None,
                        help='Directory of same-speaker/style .wav clips to concatenate '
                             '(default: first style dir found in the Expresso library)')
    parser.add_argument('--gap-sec', type=float, default=0.4,
                        help='Silence between concatenated clips (default: 0.4s)')
    parser.add_argument('--data-root', default=None,
                        help='Asset library root (overrides CINEAUDIOGEN_DATA_ROOT)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (overrides CINEAUDIOGEN_OUTPUT_DIR)')
    args = parser.parse_args()

    if args.data_root:
        os.environ["CINEAUDIOGEN_DATA_ROOT"] = args.data_root
    if args.output_dir:
        os.environ["CINEAUDIOGEN_OUTPUT_DIR"] = args.output_dir

    speech_dir = args.speech_dir
    if speech_dir is None:
        # Default: first speaker-pair/style directory in the Expresso library
        pattern = str(config.data_root() / "speech_expresso" / "expresso"
                      / "audio_48khz" / "conversational" / "*" / "*")
        candidates = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
        if not candidates:
            raise SystemExit(
                f"no speech directories found under {config.data_root()}; "
                "pass --speech-dir explicitly"
            )
        speech_dir = candidates[0]
        print(f"[long-scene] using speech dir: {speech_dir}")

    dialogue_path = str(config.output_dir() / "_long_dialogue.wav")
    dur, n = build_dialogue(speech_dir, dialogue_path, args.minutes * 60.0, args.gap_sec)
    print(f"[long-scene] built dialogue {dur:.1f}s ({dur / 60:.2f} min) from {n} clips "
          f"-> {dialogue_path}", flush=True)

    scene_dir = generate_scene(
        scene_type=args.scene_type,
        dialogue_path=dialogue_path,
        scene_name=f"long{int(args.minutes)}min",
        style=os.path.basename(speech_dir) or "default",
    )
    if not scene_dir:
        raise SystemExit("[long-scene] pipeline failed (see output above)")

    # Quick verification: bit-exact linear additivity
    lin = os.path.join(scene_dir, "linear")
    stems = {}
    for f in ("speech", "music", "ambience", "sfx", "mix"):
        p = os.path.join(lin, f + ".wav")
        if os.path.exists(p):
            stems[f] = sf.read(p, dtype="int16")[0].astype(np.int32)
    if {"speech", "music", "ambience", "sfx", "mix"} <= stems.keys():
        err = int(np.max(np.abs(
            stems["mix"] - (stems["speech"] + stems["music"]
                            + stems["ambience"] + stems["sfx"]))))
        print(f"[long-scene] linear additivity max|mix-Σstems| = {err} LSB "
              f"({'bit-exact' if err == 0 else 'NOT additive'})")
    print(f"[long-scene] DONE scene_dir={scene_dir}")


if __name__ == "__main__":
    main()
