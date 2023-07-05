import os
from pathlib import Path

import librosa
import numpy as np
import pytest

import fast_align_audio

DEMO_DIR = Path(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
TEST_FIXTURES_DIR = DEMO_DIR / "test_fixtures"


class TestFindBestAlignmentOffset:
    def test_simple_padded_array(self):
        reference = np.random.uniform(size=10_000).astype("float32")
        delayed = np.pad(reference, (121, 0))
        offset, mse = fast_align_audio.find_best_alignment_offset(
            reference, delayed, max_offset_samples=1000, lookahead_samples=5000
        )
        assert offset == 121

    @pytest.mark.parametrize("folder_name", ["multi_mic1", "multi_mic2"])
    def test_multi_mic(self, folder_name):
        main, sr = librosa.load(TEST_FIXTURES_DIR / folder_name / "main.flac", sr=None)

        other_filenames = ["other1.flac", "other2.flac", "other3.flac"]
        others = []
        for other_filename in other_filenames:
            other, _ = librosa.load(
                TEST_FIXTURES_DIR / folder_name / other_filename, sr=None
            )
            others.append(other)

        max_offset_samples = int(0.05 * sr)

        for i, other in enumerate(others):
            # Test that MSE and corr give similar results
            offset_mse, mse = fast_align_audio.find_best_alignment_offset(
                main,
                other,
                max_offset_samples=max_offset_samples,
                method="mse",
                consider_both_polarities=True,
            )
            offset_corr, corr = fast_align_audio.find_best_alignment_offset(
                main,
                other,
                max_offset_samples=max_offset_samples,
                method="corr",
                consider_both_polarities=True,
            )
            # print(
            #     f"{other_filenames[i]} mse ({mse:.6f}) vs. corr ({corr:.4f}):",
            #     offset_mse,
            #     offset_corr,
            # )
            assert offset_mse == pytest.approx(offset_corr, abs=1)
