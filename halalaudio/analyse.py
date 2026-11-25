"""Find music in a soundtrack.

The checker looks for the two things that make music music and that speech
and ordinary sound effects rarely have together:

  * tonal persistence: strong spectral peaks that hold steady across frames
    (sustained notes and chords), measured as the share of peak energy that
    stays put from one frame to the next;
  * a beat: a regular pulse in the onset envelope between 60 and 180 BPM.

It also measures syllable-rate modulation (around 4 Hz), which is high in
speech, to avoid flagging a voice as a melody.

Each window gets a music score from 0 to 1. Windows above the threshold are
merged into flagged segments with timestamps. It is a screening tool: it
points you at the seconds to listen to, it doesn't replace your ears.
"""
from __future__ import annotations

import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 16000
FRAME = 1024
HOP = 256


@dataclass
class Window:
    start: float
    end: float
    score: float
    tonal: float
    beat: float
    speech: float
    loudness: float
    gaps: float = 0.0


@dataclass
class Segment:
    start: float
    end: float
    peak: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Report:
    duration: float
    windows: list[Window]
    segments: list[Segment]
    threshold: float

    @property
    def clean(self) -> bool:
        return not self.segments

    @property
    def music_seconds(self) -> float:
        return sum(s.duration for s in self.segments)


# ---------------------------------------------------------------- audio in

def load_audio(path: str | Path) -> np.ndarray:
    """Mono float32 at 16 kHz. WAV is read directly; anything else goes through ffmpeg."""
    path = Path(path)
    if path.suffix.lower() == ".wav":
        try:
            return _read_wav(path)
        except (wave.Error, ValueError):
            pass
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is needed for non-WAV files: install it from ffmpeg.org or your package manager")
    out = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype="<i2").astype(np.float32) / 32768.0


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width != 2:
        raise ValueError("16-bit WAV only")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if rate != SR:
        x = _resample(x, rate, SR)
    return x


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    n = int(round(len(x) * dst / src))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


# ---------------------------------------------------------------- features

def _stft(x: np.ndarray) -> np.ndarray:
    if len(x) < FRAME:
        x = np.pad(x, (0, FRAME - len(x)))
    n = 1 + (len(x) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    frames = x[idx] * np.hanning(FRAME)[None, :]
    return np.abs(np.fft.rfft(frames, axis=1))


TONAL_FRAME = 4096  # ~4 Hz bins: fine enough to tell a held note from a gliding voice
TONAL_LAG = 32      # frames (~0.5 s): notes and chords hold their pitch this long, a voice almost never does


def _tonal_persistence(x: np.ndarray) -> np.ndarray:
    """Per frame: share of strong-peak energy still at the same pitch half a second later."""
    if len(x) < TONAL_FRAME + TONAL_LAG * HOP:
        return np.zeros(1)
    n = 1 + (len(x) - TONAL_FRAME) // HOP
    idx = np.arange(TONAL_FRAME)[None, :] + HOP * np.arange(n)[:, None]
    mag = np.abs(np.fft.rfft(x[idx] * np.hanning(TONAL_FRAME)[None, :], axis=1))
    lo, hi = int(100 * TONAL_FRAME / SR), int(3000 * TONAL_FRAME / SR)
    m = mag[:, lo:hi]
    floor = np.median(m, axis=1, keepdims=True) * 8 + 1e-9
    peaks = (m > np.roll(m, 1, axis=1)) & (m >= np.roll(m, -1, axis=1)) & (m > floor)
    # a held pitch must be there at every checkpoint across the half second, not just at both ends
    L = TONAL_LAG
    now = peaks[:-L]
    held_mask = now.copy()
    for k in (L // 4, L // 2, 3 * L // 4, L):
        later = peaks[k : k + len(now)]
        held_mask &= later | np.roll(later, 1, axis=1) | np.roll(later, -1, axis=1)
    held = (held_mask * m[:-L]).sum(axis=1)
    total = (now * m[:-L]).sum(axis=1)
    voiced = total > 0
    return held[voiced] / total[voiced] if voiced.any() else np.zeros(1)


def _onset_envelope(mag: np.ndarray) -> np.ndarray:
    logm = np.log1p(mag * 10)
    flux = np.maximum(0, np.diff(logm, axis=0)).sum(axis=1)
    return np.concatenate([[0], flux])


def _beat_strength(env: np.ndarray) -> float:
    """Peak of the normalised autocorrelation of the onset envelope at 60-180 BPM."""
    e = env - env.mean()
    if np.allclose(e, 0):
        return 0.0
    ac = np.correlate(e, e, mode="full")[len(e) - 1 :]
    ac /= ac[0] + 1e-9
    fps = SR / HOP
    lo, hi = int(fps * 60 / 180), int(fps * 60 / 60)
    if hi >= len(ac):
        return 0.0
    seg = ac[lo:hi]
    k = int(np.argmax(seg)) + lo
    # a real beat repeats: check the double lag too
    double = ac[2 * k] if 2 * k < len(ac) else 0.0
    return float(max(0.0, min(1.0, 0.6 * ac[k] + 0.4 * max(0.0, double))))


def _gap_peakiness(x: np.ndarray) -> float:
    """Sharp spectral peaks in the quietest quarter of the window.

    A voice pauses between words; a music bed doesn't. In the gaps, music shows up
    as narrow steady peaks while silence, room tone and breath stay flat.
    """
    F, H = 2048, 256
    if len(x) < F * 2:
        return 0.0
    n = 1 + (len(x) - F) // H
    idx = np.arange(F)[None, :] + H * np.arange(n)[:, None]
    mag = np.abs(np.fft.rfft(x[idx] * np.hanning(F)[None, :], axis=1))
    energy = mag.sum(axis=1)
    quiet = energy <= np.percentile(energy, 25)
    lo, hi = int(100 * F / SR), int(3000 * F / SR)
    spec = np.median(mag[quiet, lo:hi], axis=0)
    base = np.convolve(spec, np.ones(21) / 21, mode="same")
    return float(np.maximum(0, spec - 2.5 * base).sum() / (spec.sum() + 1e-9))


def _speech_modulation(x: np.ndarray) -> float:
    """Share of envelope modulation energy at 2-8 Hz (syllable rate)."""
    hop = 160  # 10 ms
    n = len(x) // hop
    if n < 50:
        return 0.0
    env = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-12)
    env = env - env.mean()
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    freqs = np.fft.rfftfreq(len(env), d=hop / SR)
    band = spec[(freqs >= 2) & (freqs <= 8)].sum()
    total = spec[(freqs >= 0.5) & (freqs <= 30)].sum() + 1e-12
    return float(band / total)


def features(x: np.ndarray) -> tuple[float, float, float, float, float]:
    """(tonal, beat, speech, gaps, loudness_db) for one window."""
    loud_db = float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))
    if loud_db < -50:
        return 0.0, 0.0, 0.0, 0.0, loud_db
    tonal = float(np.median(_tonal_persistence(x)))
    beat = _beat_strength(_onset_envelope(_stft(x)))
    return tonal, beat, _speech_modulation(x), _gap_peakiness(x), loud_db


def combine(tonal: float, beat: float, speech: float, gaps: float) -> float:
    """Music score 0-1. `gaps` should already be smoothed over neighbouring windows.

    Pitches held for half a second, a steady beat, or steady tones under the pauses
    in a voice all point to music. Speech glides and stops, so a voice alone scores low.
    """
    raw = 4.0 * max(0.0, tonal - 0.1) + 1.6 * max(0.0, beat - 0.3) + 8.0 * max(0.0, gaps - 0.03)
    if tonal < 0.15:
        raw -= 0.5 * max(0.0, speech - 0.5)
    return float(1 / (1 + np.exp(-10 * (raw - 0.25))))


# ---------------------------------------------------------------- report

def analyse(x: np.ndarray, window: float = 3.0, step: float = 1.0, threshold: float = 0.6,
            min_segment: float = 2.0) -> Report:
    w, s = int(window * SR), int(step * SR)
    windows = []
    if len(x) < w:
        x = np.pad(x, (0, w - len(x)))
    feats = []
    for start in range(0, len(x) - w + 1, s):
        feats.append((start, features(x[start : start + w])))
    gaps = np.array([f[3] for _, f in feats])
    # a music bed runs on: smooth the gap evidence over five windows so one odd pause can't trigger it
    smooth = np.array([np.median(gaps[max(0, i - 2) : i + 3]) for i in range(len(gaps))])
    for (start, (tonal, beat, speech, _g, loud)), g in zip(feats, smooth):
        sc = 0.0 if loud < -50 else combine(tonal, beat, speech, float(g))
        windows.append(Window(start / SR, (start + w) / SR, sc, tonal, beat, speech, loud, float(g)))
    segments: list[Segment] = []
    for win in windows:
        if win.score < threshold:
            continue
        if segments and win.start <= segments[-1].end:
            segments[-1].end = max(segments[-1].end, win.end)
            segments[-1].peak = max(segments[-1].peak, win.score)
        else:
            segments.append(Segment(win.start, win.end, win.score))
    segments = [g for g in segments if g.duration >= min_segment]
    return Report(len(x) / SR, windows, segments, threshold)


def check_file(path: str | Path, **kw) -> Report:
    return analyse(load_audio(path), **kw)


def timestamp(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
