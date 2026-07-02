"""CineAudioGen — an agentic pipeline for generating cinematic source-separation
training data (speech / music / ambience / SFX stems with bit-exact additive mixes).

Heavy submodules (director loads a ~2.5 GB CLAP model, engine pulls in DSP
libraries) are intentionally NOT imported here; import them explicitly:

    from cineaudiogen.engine import AudioEngine
    from cineaudiogen.director import CinematicDirector
    from cineaudiogen.critic import AudioCritic
"""

__version__ = "0.1.0"
