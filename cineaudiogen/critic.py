import os

from .llm import get_backend, LLMError


class AudioCritic:
    def __init__(self, backend=None):
        """Args:
            backend: An LLMBackend (see llm.py). None builds one from the
                     CINEAUDIOGEN_LLM_* environment configuration.
        """
        self.backend = backend or get_backend()

    def critique_mix(self, mix_path, scene_context):
        """
        Listens to the rough mix (or, on metrics-only backends, reads the
        per-stem measurements) and returns a JSON of DSP adjustments.
        Includes per-SFX event mixing parameters.
        """
        print(f"  [Critic] Analyzing rough mix: {os.path.basename(mix_path)}...")

        # 1. Construct the "Mixing Console" Prompt
        # We teach the model how to control our engine.py
        rough = scene_context.get("rough_settings", {}) if isinstance(scene_context, dict) else {}
        rough_music_gain = None
        rough_speech_gain = None
        rough_sfx_gain = None
        if isinstance(rough, dict):
            rough_music_gain = rough.get("music", {}).get("gain_db") if isinstance(rough.get("music"), dict) else None
            rough_speech_gain = rough.get("speech", {}).get("gain_db") if isinstance(rough.get("speech"), dict) else None
            rough_sfx_gain = rough.get("sfx", {}).get("gain_db") if isinstance(rough.get("sfx"), dict) else None

        # Build events section for per-SFX mixing
        events = scene_context.get("events", []) if isinstance(scene_context, dict) else []
        events_section = ""
        if events:
            events_lines = []
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                idx = ev.get("idx", 0)
                ts = ev.get("timestamp", 0)
                desc = ev.get("description", "unknown sound")
                events_lines.append(f"  - sfx_{idx} ({ts:.1f}s): \"{desc}\"")
            events_section = f"""
        SFX Events in this mix (each needs individual mixing):
{chr(10).join(events_lines)}
"""

        # Build audio analysis section
        stem_analysis = scene_context.get("stem_analysis", {}) if isinstance(scene_context, dict) else {}
        analysis_section = ""
        if stem_analysis:
            analysis_lines = []
            for stem_name, metrics in stem_analysis.items():
                if metrics is None:
                    continue
                line = (f"  - {stem_name}: "
                       f"loudness={metrics.get('loudness_lufs', 'N/A')} LUFS, "
                       f"LRA={metrics.get('loudness_range_lu', 'N/A')} LU, "
                       f"peak={metrics.get('true_peak_dbtp', 'N/A')} dBTP, "
                       f"RMS={metrics.get('rms_db', 'N/A')} dB, "
                       f"crest={metrics.get('crest_factor_db', 'N/A')} dB")
                analysis_lines.append(line)
            if analysis_lines:
                analysis_section = f"""
        Audio Measurements (use these to make informed mixing decisions):
{chr(10).join(analysis_lines)}

        Metric guide:
        - loudness (LUFS): Overall perceived loudness. Speech should typically be -18 to -24 LUFS in the mix.
        - LRA (Loudness Range): Dynamic variation. High LRA = very dynamic, low LRA = compressed/constant.
        - peak (dBTP): True peak level. Watch for clipping (> -1 dBTP).
        - RMS: Average signal level.
        - crest: Peak-to-RMS ratio. High crest = transient-heavy (drums, impacts). Low crest = sustained (pads, voice).
"""

        prompt = f"""
        Role: Senior Re-recording Mixer.
        Task: Listen to this 'Rough Mix' and fix the mixing mistakes.

        Scene Context:
        - Dialogue Style: "{scene_context.get('style', 'neutral')}"
        - Setting/Mood: "{scene_context.get('mood', 'cinematic')}"
        - Active Elements: Speech, Music, Ambience, SFX Events.

        Current Rough Settings (if provided):
        - Music gain_db: {rough_music_gain}
        - Speech gain_db: {rough_speech_gain}
        - SFX gain_db: {rough_sfx_gain}
{events_section}{analysis_section}
        Listening Goals:
        1. Dialogue Clarity: Is the dialogue SEVERELY masked by music? (Only reduce music if it's making speech unintelligible. Modern cinematic mixes have audible music - don't crush it!)
        2. Music Preservation: Music should remain AUDIBLE and contribute to the emotional tone. Avoid reducing music gain more than -6 dB unless absolutely necessary. For music-focused scenes, music can be LOUDER than dialogue.
        3. Space: Does the SFX sound too dry/fake? (If yes, add reverb).
        4. Glue: Does the dialogue sit IN the mix? (If no, add compression to speech rather than lowering other elements).
        5. Overall Balance: Aim for a full, rich mix. Not everything needs to be quiet - modern mixes have presence across all stems.
        6. Per-SFX Balance: Each SFX event may need different treatment based on its nature and timing.

        IMPORTANT MIXING PHILOSOPHY:
        - Be CONSERVATIVE with gain reductions. It's better to boost speech than to crush music/SFX.
        - Music should be audible in most scenes - it carries emotional weight.
        - If the scene type suggests music prominence (montage, emotional beats), preserve or even boost music.
        - Only aggressively reduce music (-10 dB or more) if dialogue is completely unintelligible.

        Cinematic Artistic Choices:
        - Big dramatic scenes: larger reverb on speech, fuller music presence.
        - Intimate scenes: drier mix, but music should still be present.
        - Music montages/emotional moments: Music is the STAR - keep it prominent.

        For SFX events: Consider each sound's nature - a footstep needs less reverb than a door slam.
        Subtle sounds like fabric rustle should be quieter, impactful sounds like slaps can be louder.


        Output JSON Instructions for the DSP Engine:
        You must return a JSON object with keys for 'music', 'speech', 'sfx' (global SFX/ambience), and 'sfx_events' (per-event adjustments).

        Allowed parameters per stem (music, speech, sfx):
        - "gain_db": float (-24.0 to +6.0) -> DELTA adjustment in dB relative to the current rough settings.
            (negative = quieter, positive = louder)
        - "low_cut_hz": int (0 to 500) -> Remove muddy bass.
        - "reverb": {{"type": "<type>", "wet_amount": 0.0 to 0.5}}
            Available reverb types (choose based on scene context):
            - "dry": Minimal reverb, very close/intimate
            - "small_room": Small space, office, closet
            - "medium_room": Standard room, living room
            - "large_room": Large room, conference room
            - "large_hall": Concert hall, theater
            - "cathedral": Massive reverberant space
            - "sewer": Dark, wet, metallic reflections
            - "tunnel": Long echoey tube
            - "bathroom": Bright, tiled reflections
            - "garage": Medium industrial space
            - "parking_garage": Large concrete space
            - "warehouse": Big open industrial
            - "forest": Outdoor with soft diffusion
            - "open_field": Outdoor, minimal reflections
        - "compressor": {{"threshold": -20, "ratio": 3.0}} -> Only for speech.

        For 'sfx_events', provide per-event mixing using keys like "sfx_0", "sfx_1", etc:
        - "gain_db": float (-24.0 to +6.0) -> Absolute gain for this event.
        - "reverb": {{"type": "<type>", "wet_amount": 0.0 to 0.5}} -> Same reverb types as above.

        Example Output (showing CONSERVATIVE mixing - preserve music!):
        {{
            "critique": "Speech needs compression for presence. Music is supporting nicely but could use a subtle low-cut to clear dialogue frequencies. Door slam at 5s needs more space.",
            "adjustments": {{
                "music": {{"gain_db": -3.0, "low_cut_hz": 200}},
                "speech": {{"gain_db": 2.0, "compressor": {{"threshold": -20, "ratio": 3.0}}}},
                "sfx": {{"gain_db": 0.0}},
                "sfx_events": {{
                    "sfx_0": {{"gain_db": -3.0, "reverb": {{"type": "room", "wet_amount": 0.2}}}},
                    "sfx_1": {{"gain_db": 0.0}}
                }}
            }}
        }}

        Remember: Boost speech rather than crushing music. Small adjustments (-3 to -6 dB) are usually enough.
        """

        # 2. Call the LLM backend (retries, audio handling, and JSON parsing
        # live in llm.py; low temperature for precise engineering numbers).
        if not self.backend.supports_audio:
            prompt += ("\n        Note: You cannot listen to the mix audio. Base your "
                       "adjustments on the audio measurements and context above.\n")
        try:
            result, token_usage = self.backend.generate_json(
                prompt, audio_path=mix_path, temperature=0.4,
            )
        except LLMError as e:
            print(f"  [Critic Error] {e}")
            return self._get_fallback_settings(scene_context)

        if not isinstance(result, dict):
            return self._get_fallback_settings(scene_context)
        critique_text = result.get('critique', 'No notes.')
        print(f"  [Critic] Notes: {critique_text}")
        adjustments = result.get("adjustments", self._get_fallback_settings())
        if not isinstance(adjustments, dict):
            adjustments = self._get_fallback_settings()

        # Attach extra keys for bookkeeping (ignored by mix merge).
        adjustments["token_usage"] = token_usage
        adjustments["critique_text"] = critique_text  # Raw AI reasoning for debugging
        adjustments["raw_adjustments"] = result.get("adjustments", {})  # Original before any processing
        return adjustments

    def _get_fallback_settings(self, scene_context=None):
        """Safe defaults if AI fails.

        These are DELTA adjustments (relative to the rough settings).
        Includes per-SFX fallback defaults.
        """
        fallback = {
            "music": {"gain_db": 0.0},
            "speech": {},
            "sfx": {"gain_db": 0.0},
            "sfx_events": {}
        }
        # Add per-event fallback defaults
        if scene_context and isinstance(scene_context, dict):
            events = scene_context.get("events", [])
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                idx = ev.get("idx", 0)
                fallback["sfx_events"][f"sfx_{idx}"] = {"gain_db": 0.0}
        return fallback