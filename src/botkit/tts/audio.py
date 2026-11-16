"""WAV/OGG → OGG/Opus in memory (needs the optional ``audio`` extra: soundfile + numpy).

Messengers play a voice bubble only for OGG/Opus. libsndfile (bundled with the
``soundfile`` wheel) encodes Opus, so neither ffmpeg nor temp files are needed.

Audio is resampled before encoding. In a measured upload test against VK's voice-message
server, 16 kHz Opus was accepted while the same speech at 24 and 48 kHz got "The page is
temporarily unavailable". So 16 kHz (plenty for speech, smaller file) is the default.
"""

from __future__ import annotations

import io

OPUS_RATE = 16000
OPUS_RATES = (8000, 12000, 16000, 24000, 48000)  # the rates Opus supports natively
MIN_RATE, MAX_RATE = 8000, 96000


def _libs():
    try:
        import numpy
        import soundfile
    except ImportError:
        raise RuntimeError(
            "Audio conversion needs the optional dependencies: pip install 'botkit[audio]'"
        ) from None
    return numpy, soundfile


def resample(samples, rate: int, target: int):
    """Linear-interpolation resampling of mono/stereo float samples (adequate for speech)."""
    numpy, _ = _libs()
    if rate == target:
        return samples
    count = max(1, round(len(samples) * target / rate))
    positions = numpy.linspace(0, len(samples) - 1, count)
    if samples.ndim == 1:
        return numpy.interp(positions, numpy.arange(len(samples)), samples)
    return numpy.stack(
        [numpy.interp(positions, numpy.arange(len(samples)), samples[:, c]) for c in range(samples.shape[1])],
        axis=1,
    )


def to_ogg_opus(data: bytes, *, target_rate: int = OPUS_RATE) -> bytes:
    numpy, soundfile = _libs()
    try:
        samples, rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except (soundfile.LibsndfileError, RuntimeError, ValueError):
        raise ValueError("The audio is not a valid WAV or OGG file") from None
    if not MIN_RATE <= rate <= MAX_RATE:
        raise ValueError(f"Unsupported sample rate: {rate}")
    if len(samples) == 0:
        raise ValueError("The audio is empty")
    samples = numpy.clip(resample(samples, rate, target_rate), -1.0, 1.0)
    out = io.BytesIO()
    soundfile.write(out, (samples * 32767).astype("int16"), target_rate, format="OGG", subtype="OPUS")
    return out.getvalue()


def audio_seconds(data: bytes) -> float | None:
    """Playing time of a WAV/OGG, or None if it cannot be read."""
    try:
        _, soundfile = _libs()
        return round(soundfile.info(io.BytesIO(data)).duration, 3)
    except Exception:
        return None
