import os
from pathlib import Path

import librosa
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.io.wavfile import write

import fast_align_audio

DEMO_DIR = Path(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
TEST_FIXTURES_DIR = DEMO_DIR / "test_fixtures"

DEBUG = False


class TestFindBestAlignmentOffset:
    def test_simple_alignment_positive_offset(self):
        reference = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        delayed = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        offset, mse = fast_align_audio.find_best_alignment_offset(
            reference, delayed, max_offset_samples=4
        )
        assert offset == 1

        aligned = fast_align_audio.align_delayed_signal_with_reference(
            reference, delayed, offset
        )
        assert_array_almost_equal(aligned, reference)

    def test_simple_alignment_negative_offset(self):
        reference = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        delayed = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        offset, mse = fast_align_audio.find_best_alignment_offset(
            reference, delayed, max_offset_samples=4
        )
        assert offset == -1

        aligned = fast_align_audio.align_delayed_signal_with_reference(
            reference, delayed, offset
        )
        assert_array_almost_equal(aligned, reference)

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
            if DEBUG:
                print(
                    f"{other_filenames[i]} mse ({mse:.6f}) vs. corr ({corr:.4f}):",
                    offset_mse,
                    offset_corr,
                )
            assert offset_mse == pytest.approx(offset_corr, abs=1)

            aligned = fast_align_audio.align_delayed_signal_with_reference(
                main, other, offset_corr
            )
            if DEBUG:
                write(
                    TEST_FIXTURES_DIR
                    / folder_name
                    / f"{Path(other_filenames[i]).stem}_aligned.wav",
                    int(sr),
                    aligned,
                )
