"""Build the retrieval indices the Director needs.

Two artifacts are written to <data_root>/indices/:

1. sfx_index.npz     — CLAP embeddings for every SFX/ambience file, used for
                       semantic search ("impatient car horn" -> closest clip).
2. music_lookup.json — tag -> [file paths] lookup for music retrieval, built
                       from per-library attributions.json files that carry a
                       "tags" list per track (see README for the format).

Usage:
    cineaudiogen-build-indices                 # index SFX + music
    cineaudiogen-build-indices --skip-sfx      # music lookup only (fast)
    python -m cineaudiogen.build_indices ...

Expected data layout (override roots with --sfx-dir / --music-lib):
    <data_root>/sfx_fsd50k/FSD50K.dev_audio/   SFX pool (any folder of wav/mp3 works)
    <data_root>/<music-lib>/downloads/*.mp3    music files
    <data_root>/<music-lib>/attributions.json  [{"filename": ..., "tags": [...]}, ...]
"""
import argparse
import glob
import json
import os

import numpy as np
from tqdm import tqdm

from . import config


def build_sfx_index(sfx_dirs, out_path, clip_seconds=10.0):
    """Embed every audio file in `sfx_dirs` with CLAP and save paths+embeddings."""
    import torch
    import laion_clap
    import librosa

    torch.set_num_threads(4)  # prevent thrashing on small machines

    files = []
    for d in sfx_dirs:
        files += glob.glob(os.path.join(d, "**", "*.wav"), recursive=True)
        files += glob.glob(os.path.join(d, "**", "*.mp3"), recursive=True)
    files = sorted(set(files))
    if not files:
        print(f"[sfx] WARNING: no audio files found under {sfx_dirs}; skipping SFX index")
        return

    print(f"[sfx] embedding {len(files)} files with CLAP (HTSAT-tiny)...")
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
    model.load_ckpt()  # downloads the checkpoint on first run
    model.eval()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    paths, embeds = [], []
    for f in tqdm(files, desc="Indexing SFX"):
        try:
            # CLAP expects 48 kHz; the first N seconds capture the texture
            audio, _ = librosa.load(f, sr=48000, duration=clip_seconds)
            if len(audio) == 0:
                continue
            with torch.no_grad():
                t = torch.from_numpy(audio.reshape(1, -1)).float()
                if use_cuda:
                    t = t.cuda()
                emb = model.get_audio_embedding_from_data(x=t, use_tensor=True)
            paths.append(f)
            embeds.append(emb.cpu().numpy()[0])
        except Exception:
            continue  # unreadable file — skip

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, paths=np.array(paths), embeddings=np.array(embeds))
    print(f"[sfx] saved {len(paths)} embeddings -> {out_path}")


def build_music_lookup(music_libs, out_path):
    """Merge per-library attributions.json files into one tag -> [files] lookup."""
    tag_to_files = {}
    for lib in music_libs:
        attr_path = os.path.join(lib, "attributions.json")
        downloads = os.path.join(lib, "downloads")
        if not os.path.exists(attr_path):
            print(f"[music] WARNING: skipping {lib} (no attributions.json)")
            continue
        with open(attr_path, 'r', encoding='utf-8') as f:
            attributions = json.load(f)
        added = 0
        for entry in attributions:
            filename = entry.get("filename")
            tags = entry.get("tags", [])
            if not filename:
                continue
            filepath = os.path.join(downloads, filename)
            if not os.path.exists(filepath):
                continue
            for tag in tags:
                tag = tag.lower().strip()
                if tag:
                    tag_to_files.setdefault(tag, set()).add(filepath)
            added += 1
        print(f"[music] {lib}: {added} tracks")

    lookup = {tag: sorted(files) for tag, files in sorted(tag_to_files.items())}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(lookup, f, indent=2)
    print(f"[music] saved {len(lookup)} tags -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Build CineAudioGen retrieval indices")
    parser.add_argument('--data-root', default=None,
                        help='Asset library root (overrides CINEAUDIOGEN_DATA_ROOT)')
    parser.add_argument('--sfx-dir', action='append', default=None,
                        help='SFX pool directory (repeatable; default: '
                             '<data_root>/sfx_fsd50k/FSD50K.dev_audio)')
    parser.add_argument('--music-lib', action='append', default=None,
                        help='Music library dir containing downloads/ + attributions.json '
                             '(repeatable; default: <data_root>/music)')
    parser.add_argument('--skip-sfx', action='store_true', help='Skip the (slow) CLAP SFX index')
    parser.add_argument('--skip-music', action='store_true', help='Skip the music lookup')
    args = parser.parse_args()

    if args.data_root:
        os.environ["CINEAUDIOGEN_DATA_ROOT"] = args.data_root

    root = config.data_root()
    index_dir = config.index_dir()

    if not args.skip_sfx:
        sfx_dirs = args.sfx_dir or [str(root / "sfx_fsd50k" / "FSD50K.dev_audio")]
        build_sfx_index(sfx_dirs, str(index_dir / "sfx_index.npz"))

    if not args.skip_music:
        music_libs = args.music_lib or [str(root / "music")]
        build_music_lookup(music_libs, str(index_dir / "music_lookup.json"))


if __name__ == "__main__":
    main()
