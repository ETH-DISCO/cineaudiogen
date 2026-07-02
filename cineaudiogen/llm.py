"""Multi-provider LLM backends for the Director and Critic agents.

The pipeline's agents do two multimodal calls per scene: the Director listens
to the dialogue, the Critic listens to the rough mix. Not every provider can
listen — backends declare `supports_audio`, and agents adapt their prompts
(the Critic already computes per-stem loudness metrics, so a text-only model
can mix "by the meters" instead of by ear).

Select a backend via environment variables (see `.env.example`):

    CINEAUDIOGEN_LLM_PROVIDER   gemini (default) | openai | openai-compatible | anthropic
    CINEAUDIOGEN_LLM_MODEL      model override (each provider has a sane default)
    CINEAUDIOGEN_LLM_BASE_URL   endpoint for openai-compatible (OpenRouter/vLLM/Ollama/...)
    CINEAUDIOGEN_LLM_AUDIO      auto (default) | off  — force metrics-only mode

Every backend returns parsed JSON plus a normalized token-usage dict, so the
rest of the pipeline is provider-agnostic.
"""
import io
import json
import os
import time
from abc import ABC, abstractmethod

from . import config


class LLMError(RuntimeError):
    """An LLM call failed after retries (or returned unusable output)."""


def _strip_fences(text):
    """Extract JSON from a response that may be wrapped in markdown fences."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[: -3]
    # Fall back to the outermost {...} if there's leading/trailing prose
    if not t.lstrip().startswith(("{", "[")):
        start, end = t.find("{"), t.rfind("}")
        if start != -1 and end > start:
            t = t[start:end + 1]
    return t.strip()


def _usage(model, prompt_tokens, completion_tokens, total_tokens=None):
    """Normalize token usage into the shape the pipeline stores in metadata."""
    meta = {}
    if prompt_tokens is not None:
        meta["prompt_token_count"] = int(prompt_tokens)
    if completion_tokens is not None:
        meta["candidates_token_count"] = int(completion_tokens)
    if total_tokens is None and meta:
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
    if total_tokens is not None:
        meta["total_token_count"] = int(total_tokens)
    return {"model": model, "usage_metadata": meta or None}


class LLMBackend(ABC):
    """One JSON-producing LLM call, with retries. Subclasses implement _call."""

    name = "base"
    supports_audio = False

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def _call(self, prompt, audio_path, temperature):
        """Return (response_text, usage_dict). audio_path may be None."""

    def generate_json(self, prompt, audio_path=None, temperature=0.5,
                      max_retries=3):
        """Run the call and parse a JSON object from the response.

        Backends that can't listen (`supports_audio` is False) silently ignore
        `audio_path` — callers adapt the prompt for that case.
        """
        if not self.supports_audio:
            audio_path = None
        delay = 2.0
        last_err = None
        for attempt in range(max_retries):
            try:
                text, usage = self._call(prompt, audio_path, temperature)
                return json.loads(_strip_fences(text)), usage
            except Exception as e:  # includes malformed-JSON parses — retry those too
                last_err = e
                # Retrying won't fix configuration problems (missing/invalid
                # credentials, bad arguments) — fail fast on those.
                msg = str(e).lower()
                if (isinstance(e, TypeError)
                        or any(s in type(e).__name__ for s in
                               ("Authentication", "PermissionDenied", "Forbidden"))
                        or "authentication" in msg or "api key" in msg
                        or "credentials" in msg):
                    break
                if attempt < max_retries - 1:
                    print(f"  [LLM:{self.name}] attempt {attempt + 1} failed: {e}. "
                          f"Retrying in {delay:.0f}s...")
                    time.sleep(delay)
                    delay *= 2
        raise LLMError(f"{self.name} call failed: {last_err}")


class GeminiBackend(LLMBackend):
    """Google Gemini (default). Native audio input via the Files API — handles
    even very long scenes, and the flash tier is cheap. Needs GEMINI_API_KEY."""

    name = "gemini"
    supports_audio = True
    DEFAULT_MODEL = "gemini-3-flash-preview"

    def __init__(self, model=None):
        super().__init__(model or self.DEFAULT_MODEL)
        from google import genai
        from google.genai import types
        self._types = types
        self.client = genai.Client(api_key=config.gemini_api_key())

    def _call(self, prompt, audio_path, temperature):
        uploaded = None
        contents = [prompt]
        try:
            if audio_path:
                uploaded = self.client.files.upload(file=audio_path)
                contents = [uploaded, prompt]
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=self._types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=temperature,
                ),
            )
        finally:
            if uploaded is not None:  # free Gemini storage quota
                try:
                    self.client.files.delete(name=uploaded.name)
                except Exception:
                    pass
        um = getattr(response, "usage_metadata", None)
        usage = _usage(
            self.model,
            getattr(um, "prompt_token_count", None),
            getattr(um, "candidates_token_count", None),
            getattr(um, "total_token_count", None),
        )
        return response.text, usage


class OpenAIBackend(LLMBackend):
    """OpenAI audio-capable chat models (gpt-4o-audio family).

    Audio is sent base64-in-chat, which caps request size — we transcode a
    16 kHz mono copy for the model's ears (the pipeline audio stays 48 kHz)
    and reject clips longer than MAX_AUDIO_SECONDS with a clear error.
    Needs OPENAI_API_KEY. Install with `pip install cineaudiogen[openai]`.
    """

    name = "openai"
    supports_audio = True
    DEFAULT_MODEL = "gpt-4o-audio-preview"
    MAX_AUDIO_SECONDS = 600  # ~19 MB base64 at 16 kHz mono PCM16

    def __init__(self, model=None, base_url=None, api_key=None):
        super().__init__(model or self.DEFAULT_MODEL)
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                f"the '{self.name}' provider needs the openai package: "
                "pip install 'cineaudiogen[openai]'"
            ) from e
        key = api_key or os.environ.get("OPENAI_API_KEY", "").strip()
        if not key and base_url is None:
            raise RuntimeError(
                "OPENAI_API_KEY is not set (required for the openai provider)."
            )
        self.client = OpenAI(api_key=key or "not-needed", base_url=base_url)
        if config.llm_audio_mode() == "off":
            self.supports_audio = False

    def _encode_audio_b64(self, path):
        import base64
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly

        x, sr = sf.read(path, dtype="float32")
        if x.ndim == 2:
            x = x.mean(axis=1)
        duration = len(x) / sr
        if duration > self.MAX_AUDIO_SECONDS:
            raise LLMError(
                f"audio is {duration:.0f}s; the {self.name} provider caps at "
                f"{self.MAX_AUDIO_SECONDS}s per request — use the gemini provider "
                "for long scenes"
            )
        if sr != 16000:
            from math import gcd
            g = gcd(16000, sr)
            x = resample_poly(x, 16000 // g, sr // g)
        buf = io.BytesIO()
        sf.write(buf, x.astype(np.float32), 16000, format="WAV", subtype="PCM_16")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _call(self, prompt, audio_path, temperature):
        content = [{"type": "text", "text": prompt}]
        if audio_path:
            content.append({
                "type": "input_audio",
                "input_audio": {"data": self._encode_audio_b64(audio_path),
                                "format": "wav"},
            })
        # No response_format=json_object: the audio models and many local
        # OpenAI-compatible servers don't support it; the fence-stripping
        # parser + retry in the base class covers JSON reliability instead.
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
        )
        text = response.choices[0].message.content
        u = getattr(response, "usage", None)
        usage = _usage(
            self.model,
            getattr(u, "prompt_tokens", None),
            getattr(u, "completion_tokens", None),
            getattr(u, "total_tokens", None),
        )
        return text, usage


class OpenAICompatibleBackend(OpenAIBackend):
    """Any OpenAI-compatible endpoint: OpenRouter, vLLM, LM Studio, Ollama...

    Set CINEAUDIOGEN_LLM_BASE_URL (and CINEAUDIOGEN_LLM_API_KEY if the server
    wants one). Audio input defaults OFF (most local models are text-only);
    set CINEAUDIOGEN_LLM_AUDIO=on if your model can listen (e.g. Qwen2-Audio).
    """

    name = "openai-compatible"
    DEFAULT_MODEL = "local-model"

    def __init__(self, model=None, base_url=None):
        base_url = base_url or config.llm_base_url()
        if not base_url:
            raise RuntimeError(
                "CINEAUDIOGEN_LLM_BASE_URL is not set (required for the "
                "openai-compatible provider), e.g. http://localhost:11434/v1"
            )
        key = os.environ.get("CINEAUDIOGEN_LLM_API_KEY", "").strip() or "not-needed"
        super().__init__(model=model, base_url=base_url, api_key=key)
        self.supports_audio = config.llm_audio_mode() == "on"


class AnthropicBackend(LLMBackend):
    """Anthropic Claude — metrics-only (Claude has no audio input).

    The Critic works surprisingly well here because it already receives
    per-stem LUFS/LRA/peak/RMS/crest measurements: mixing by the meters.
    Auth resolves like any Anthropic SDK app (ANTHROPIC_API_KEY, or an
    `ant auth login` profile). Install with `pip install cineaudiogen[anthropic]`.
    """

    name = "anthropic"
    supports_audio = False
    DEFAULT_MODEL = "claude-opus-4-8"

    def __init__(self, model=None):
        super().__init__(model or self.DEFAULT_MODEL)
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "the 'anthropic' provider needs the anthropic package: "
                "pip install 'cineaudiogen[anthropic]'"
            ) from e
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()

    def _call(self, prompt, audio_path, temperature):
        # `temperature` is intentionally ignored: sampling params are removed
        # on Opus 4.7+ and would 400. Prompt wording carries the intent.
        # No cache_control either — these prompts (~1-2k tokens) sit below
        # Opus 4.8's 4096-token minimum cacheable prefix, so it would be a no-op.
        response = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            raise LLMError("the model declined the request (stop_reason=refusal)")
        text = next((b.text for b in response.content if b.type == "text"), "")
        u = response.usage
        usage = _usage(self.model, getattr(u, "input_tokens", None),
                       getattr(u, "output_tokens", None))
        return text, usage


_PROVIDERS = {
    "gemini": GeminiBackend,
    "openai": OpenAIBackend,
    "openai-compatible": OpenAICompatibleBackend,
    "anthropic": AnthropicBackend,
}


def get_backend():
    """Build the LLM backend selected by CINEAUDIOGEN_LLM_PROVIDER."""
    provider = config.llm_provider()
    cls = _PROVIDERS.get(provider)
    if cls is None:
        raise ValueError(
            f"unknown CINEAUDIOGEN_LLM_PROVIDER={provider!r}; "
            f"choose one of: {', '.join(sorted(_PROVIDERS))}"
        )
    backend = cls(model=config.llm_model())
    ears = "listens to audio" if backend.supports_audio else "metrics-only (no audio input)"
    print(f"  [LLM] provider={backend.name} model={backend.model} ({ears})")
    return backend
