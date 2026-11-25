"""Synthetic test signals: speech-like voice, music, sound effects. 16 kHz mono."""
import numpy as np

SR = 16000


def _t(sec):
    return np.arange(int(sec * SR)) / SR


def _tone(freq, sec, harmonics=5, decay=0.0):
    t = _t(sec)
    x = sum((0.6 ** h) * np.sin(2 * np.pi * freq * (h + 1) * t) for h in range(harmonics))
    env = np.minimum(1, t / 0.02) * np.exp(-decay * t)
    return x * env


def music(sec=12, seed=0, beat=True, bpm=120):
    rng = np.random.default_rng(seed)
    chords = [[261.6, 329.6, 392.0], [220.0, 261.6, 329.6], [174.6, 220.0, 261.6], [196.0, 246.9, 293.7]]
    out = np.zeros(int(sec * SR))
    bar = 2.0
    for i in range(int(sec / bar)):
        c = chords[i % 4]
        seg = sum(_tone(f, bar, decay=0.3) for f in c)
        mel = np.concatenate([_tone(rng.choice(c) * 2, bar / 4, decay=1.5) for _ in range(4)])
        s = int(i * bar * SR)
        out[s : s + len(seg)] += seg[: len(out) - s] + 0.5 * mel[: len(out) - s]
    if beat:
        period = 60 / bpm
        kick_t = _t(0.15)
        kick = np.sin(2 * np.pi * (60 + 80 * np.exp(-30 * kick_t)) * kick_t) * np.exp(-25 * kick_t)
        for k in range(int(sec / period)):
            s = int(k * period * SR)
            out[s : s + len(kick)] += 2.5 * kick[: len(out) - s]
    return (out / (np.abs(out).max() + 1e-9) * 0.5).astype(np.float32)


def speech(sec=12, seed=0):
    """Pitch-varying glottal buzz through moving formants, chopped into syllables and phrases."""
    rng = np.random.default_rng(seed)
    n = int(sec * SR)
    # intonation: slow phrase contour + a step per syllable + glide, like a real voice
    t = _t(sec)
    steps = np.repeat(rng.normal(0, 18, int(sec * 5) + 1), SR // 5)[:n]
    glide = 12 * np.sin(2 * np.pi * rng.uniform(2.5, 4.5) * t + rng.uniform(0, 6))
    f0 = np.clip(140 + 25 * np.sin(2 * np.pi * 0.35 * t + rng.uniform(0, 6)) + steps + glide, 60, 320)
    phase = np.cumsum(2 * np.pi * f0 / SR)
    buzz = np.sign(np.sin(phase)) * 0.3 + 0.05 * rng.normal(size=n)
    out = np.zeros(n)
    i = 0
    vowels = [(730, 1090), (270, 2290), (300, 870), (530, 1840), (640, 1190)]
    while i < n:
        dur = int(rng.uniform(0.12, 0.28) * SR)
        f1, f2 = vowels[rng.integers(len(vowels))]
        seg = buzz[i : i + dur]
        spec = np.fft.rfft(seg)
        fr = np.fft.rfftfreq(len(seg), 1 / SR)
        filt = np.exp(-((fr - f1) / 120) ** 2) + 0.7 * np.exp(-((fr - f2) / 180) ** 2)
        y = np.fft.irfft(spec * filt, len(seg))
        y *= np.hanning(len(y)) ** 0.7
        out[i : i + len(y)] += y
        i += dur + int(rng.uniform(0.02, 0.08) * SR)
        if rng.random() < 0.12:
            i += int(rng.uniform(0.25, 0.6) * SR)  # phrase pause
    return (out / (np.abs(out).max() + 1e-9) * 0.5).astype(np.float32)


def sfx(sec=12, seed=0):
    """Whooshes, clicks and impacts at irregular times: non-musical sound effects."""
    rng = np.random.default_rng(seed)
    n = int(sec * SR)
    out = 0.01 * rng.normal(size=n)
    t = 0.3
    while t < sec - 1:
        kind = rng.integers(3)
        s = int(t * SR)
        if kind == 0:  # whoosh: band-swept noise
            L = int(0.6 * SR)
            noise = rng.normal(size=L)
            env = np.hanning(L)
            out[s : s + L] += 0.4 * noise * env
        elif kind == 1:  # click
            L = int(0.01 * SR)
            out[s : s + L] += rng.normal(size=L)
        else:  # impact
            L = int(0.3 * SR)
            tt = np.arange(L) / SR
            out[s : s + L] += 0.8 * rng.normal(size=L) * np.exp(-20 * tt)
        t += rng.uniform(0.4, 1.7)
    return (out / (np.abs(out).max() + 1e-9) * 0.5).astype(np.float32)


def mix(a, b, gain_b=0.25):
    n = min(len(a), len(b))
    return (a[:n] + gain_b * b[:n]).astype(np.float32)
