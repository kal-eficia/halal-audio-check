import unittest

import numpy as np

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import synth  # noqa: E402
from halalaudio import analyse  # noqa: E402


def flagged_share(x):
    r = analyse(x)
    return r.music_seconds / r.duration


class Clean(unittest.TestCase):
    """Speech and sound effects must not be flagged."""

    def test_speech_is_clean(self):
        for seed in range(1, 26):
            with self.subTest(seed=seed):
                self.assertTrue(analyse(synth.speech(30, seed)).clean)

    def test_sound_effects_are_clean(self):
        for seed in range(1, 16):
            with self.subTest(seed=seed):
                self.assertTrue(analyse(synth.sfx(30, seed)).clean)

    def test_speech_with_sound_effects_is_clean(self):
        for seed in range(1, 8):
            with self.subTest(seed=seed):
                self.assertTrue(analyse(synth.mix(synth.speech(30, seed), synth.sfx(30, seed + 50), 0.5)).clean)

    def test_silence(self):
        self.assertTrue(analyse(np.zeros(16000 * 10, dtype=np.float32)).clean)


class Music(unittest.TestCase):
    def test_music_with_beat(self):
        self.assertGreater(flagged_share(synth.music(20, 1)), 0.95)

    def test_melody_without_beat(self):
        self.assertGreater(flagged_share(synth.music(20, 2, beat=False)), 0.95)

    def test_drums_only(self):
        kick = np.zeros(16000 * 12, np.float32)
        t = np.arange(int(0.15 * 16000)) / 16000
        k = np.sin(2 * np.pi * (60 + 80 * np.exp(-30 * t)) * t) * np.exp(-25 * t)
        for i in range(24):
            s = int(i * 0.5 * 16000)
            kick[s : s + len(k)] += k
        self.assertGreater(flagged_share(kick * 0.5), 0.9)

    def test_music_bed_under_voice(self):
        for seed in range(1, 6):
            x = synth.mix(synth.speech(30, seed), synth.music(30, seed + 20, beat=seed % 2 == 0), 0.15)
            with self.subTest(seed=seed):
                self.assertGreater(flagged_share(x), 0.9)

    def test_short_stinger_is_found_and_located(self):
        x = synth.speech(20, 3)
        x[8 * 16000 : 10 * 16000] += synth.music(2, 8, beat=False) * 0.8
        segs = analyse(x).segments
        self.assertEqual(len(segs), 1)
        self.assertLessEqual(segs[0].start, 8)
        self.assertGreaterEqual(segs[0].end, 10)


if __name__ == "__main__":
    unittest.main()
