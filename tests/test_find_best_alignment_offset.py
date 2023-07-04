import os
from pathlib import Path

import librosa
import numpy as np

import fast_align_audio

DEMO_DIR = Path(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
TEST_FIXTURES_DIR = DEMO_DIR / "test_fixtures"


class TestFindBestAlignmentOffset:
    def test_simple_padded_array(self):
        reference = np.random.uniform(size=10_000).astype("float32")
        delayed = np.pad(reference, (121, 0))
        offset = fast_align_audio.find_best_alignment_offset(
            reference, delayed, max_offset_samples=1000, lookahead_samples=5000
        )
        assert offset == 121

    def test_multi_mic1(self):
        main, sr = librosa.load(TEST_FIXTURES_DIR / "multi_mic1" / "main.flac", sr=None)
        other1, _ = librosa.load(
            TEST_FIXTURES_DIR / "multi_mic1" / "other1.flac", sr=None
        )
        other2, _ = librosa.load(
            TEST_FIXTURES_DIR / "multi_mic1" / "other2.flac", sr=None
        )
        other3, _ = librosa.load(
            TEST_FIXTURES_DIR / "multi_mic1" / "other3.flac", sr=None
        )

        max_offset_samples = int(0.05 * sr)

        offset1 = fast_align_audio.find_best_alignment_offset(
            main, other1, max_offset_samples=max_offset_samples
        )
        print(offset1, "offset1")

        offset2 = fast_align_audio.find_best_alignment_offset(
            main, other2, max_offset_samples=max_offset_samples
        )
        print(offset2, "offset2")

        offset3 = fast_align_audio.find_best_alignment_offset(
            main, other3, max_offset_samples=max_offset_samples
        )
        print(offset3, "offset3")

        # TODO: Assert that the offsets are sane

    # TODO
    # def test_multi_mic2(self):
