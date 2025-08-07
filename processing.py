import io
import math
from dataclasses import dataclass
from typing import Optional, Tuple
from pydub import AudioSegment
import numpy as np
import pyloudnorm as pyln
import librosa
import soundfile as sf

# Try to import pedalboard, fallback to simple processing if not available
try:
    from pedalboard import Pedalboard, Reverb, HighpassFilter, HighShelfFilter, Compressor
    HAS_PEDALBOARD = True
except ImportError:
    HAS_PEDALBOARD = False
    print("Warning: pedalboard not available, using simplified audio processing")

@dataclass
class SegmentSpec:
    start: float  # seconds
    end: float    # seconds


def load_audiosegment(file_bytes: bytes) -> AudioSegment:
    return AudioSegment.from_file(io.BytesIO(file_bytes))


def audiosegment_to_wav_bytes(seg: AudioSegment) -> bytes:
    buf = io.BytesIO()
    seg.export(buf, format='wav')
    return buf.getvalue()


def compute_loudness(wav_bytes: bytes) -> float:
    data, sr = sf.read(io.BytesIO(wav_bytes))
    if data.ndim > 1:
        data = data.mean(axis=1)
    meter = pyln.Meter(sr)
    return meter.integrated_loudness(data)


def match_loudness(target_lufs: float, wav_bytes: bytes) -> bytes:
    data, sr = sf.read(io.BytesIO(wav_bytes))
    mono = data
    if mono.ndim > 1:
        mono = mono.mean(axis=1)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(mono)
    diff = target_lufs - loudness
    gain = 10 ** (diff / 20)
    data_out = data * gain
    buf = io.BytesIO()
    sf.write(buf, data_out, sr, format='WAV')
    return buf.getvalue()


def apply_serene_processing(wav_bytes: bytes, brightness: float = 0.6) -> bytes:
    data, sr = sf.read(io.BytesIO(wav_bytes))
    if data.ndim == 1:
        data = data[:, None]
    
    if HAS_PEDALBOARD:
        # Use pedalboard for advanced audio processing
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=80.0),
            Compressor(threshold_db=-24, ratio=2.0, attack_ms=30, release_ms=150),
            Reverb(room_size=0.35 + 0.25*brightness, damping=0.4, wet_level=0.12*brightness, dry_level=1.0, width=0.9, freeze_mode=0.0),
            HighShelfFilter(cutoff_frequency_hz=6000, gain_db=3.0*brightness)
        ])
        processed = board(data.T, sr).T
    else:
        # Fallback: simple processing using basic filtering
        # Apply a simple high-pass filter equivalent and basic gain adjustment
        processed = data.copy()
        # Simple brightness adjustment (boost higher frequencies)
        if brightness > 0.5:
            gain_factor = 1.0 + (brightness - 0.5) * 0.3
            processed = processed * gain_factor
        # Ensure we don't clip
        if np.max(np.abs(processed)) > 1.0:
            processed = processed / np.max(np.abs(processed))
    
    buf = io.BytesIO()
    sf.write(buf, processed, sr, format='WAV')
    return buf.getvalue()


def loop_to_duration(seg: AudioSegment, target_ms: int) -> AudioSegment:
    """Loop (repeat) an AudioSegment until it reaches at least target_ms, then trim."""
    if len(seg) == 0:
        return AudioSegment.silent(duration=target_ms)
    out = AudioSegment.empty()
    while len(out) < target_ms:
        out += seg
    if len(out) > target_ms:
        out = out[:target_ms]
    return out


def replace_segment(base: AudioSegment, replacement: AudioSegment, spec: SegmentSpec, crossfade_ms: int = 300, loop_fill: bool = True) -> AudioSegment:
    start_ms = int(spec.start * 1000)
    end_ms = int(spec.end * 1000)
    pre = base[:start_ms]
    post = base[end_ms:]
    window_len = end_ms - start_ms
    # Loop or trim replacement to window length
    if loop_fill:
        replacement = loop_to_duration(replacement, window_len)
    else:
        if len(replacement) < window_len:
            replacement += AudioSegment.silent(duration=window_len - len(replacement))
        else:
            replacement = replacement[:window_len]
    # Safe crossfade limits cannot exceed either segment length
    def safe_append(a: AudioSegment, b: AudioSegment, cf: int) -> AudioSegment:
        if len(a) == 0 or len(b) == 0:
            # Nothing to crossfade with -> no crossfade
            return a + b
        real_cf = min(cf, len(a), len(b))
        if real_cf < 0:
            real_cf = 0
        return a.append(b, crossfade=real_cf)

    combined = safe_append(pre, replacement, crossfade_ms)
    combined = safe_append(combined, post, crossfade_ms)
    return combined


def seconds_from_timestamp(ts: str) -> float:
    # Format: HH:MM:SS(.ms)
    parts = ts.split(':')
    if len(parts) != 3:
        raise ValueError('Timestamp must be HH:MM:SS or HH:MM:SS.mmm')
    h, m, s = parts
    sec = float(s)
    return int(h)*3600 + int(m)*60 + sec

