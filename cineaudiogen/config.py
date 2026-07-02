"""Central configuration for CineAudioGen.

Everything is overridable via environment variables so the package never
hardcodes machine-specific paths or secrets:

    GEMINI_API_KEY             Google Gemini API key (default LLM provider)
    CINEAUDIOGEN_DATA_ROOT     Root of the asset library      (default: ./data)
    CINEAUDIOGEN_OUTPUT_DIR    Where scenes are written       (default: ./output)
    CINEAUDIOGEN_LLM_PROVIDER  gemini | openai | openai-compatible | anthropic
    CINEAUDIOGEN_LLM_MODEL     model override for the chosen provider
    CINEAUDIOGEN_LLM_BASE_URL  endpoint for openai-compatible servers
    CINEAUDIOGEN_LLM_AUDIO     auto | on | off — whether the LLM gets the audio

Paths are resolved lazily (at call time), so setting an env var or changing
the working directory before invoking the pipeline behaves as expected.
"""
import os
from pathlib import Path

SAMPLE_RATE = 48000


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def data_root() -> Path:
    """Root directory of the asset library (speech / music / SFX / indices)."""
    return _env_path("CINEAUDIOGEN_DATA_ROOT", "data")


def output_dir() -> Path:
    """Directory where generated scenes are written."""
    return _env_path("CINEAUDIOGEN_OUTPUT_DIR", "output")


def index_dir() -> Path:
    """Directory holding the CLAP SFX index and the music tag lookup."""
    return data_root() / "indices"


def gemini_api_key() -> str:
    """Return the Gemini API key, or raise with setup instructions."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Create a key at "
            "https://aistudio.google.com/apikey and export it, e.g.\n"
            "    export GEMINI_API_KEY=your-key-here\n"
            "(or put it in a .env file and `source` it — see .env.example).\n"
            "To use a different LLM provider instead, set "
            "CINEAUDIOGEN_LLM_PROVIDER (openai | openai-compatible | anthropic)."
        )
    return key


def llm_provider() -> str:
    """Which LLM backend drives the Director/Critic (default: gemini)."""
    return os.environ.get("CINEAUDIOGEN_LLM_PROVIDER", "gemini").strip().lower()


def llm_model():
    """Optional model override for the chosen provider (None = provider default)."""
    return os.environ.get("CINEAUDIOGEN_LLM_MODEL", "").strip() or None


def llm_base_url():
    """Endpoint for openai-compatible servers (OpenRouter, vLLM, Ollama, ...)."""
    return os.environ.get("CINEAUDIOGEN_LLM_BASE_URL", "").strip() or None


def llm_audio_mode() -> str:
    """auto (provider decides) | on | off — whether the LLM receives audio."""
    return os.environ.get("CINEAUDIOGEN_LLM_AUDIO", "auto").strip().lower()
