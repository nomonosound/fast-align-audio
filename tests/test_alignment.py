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


def convert_decibels_to_amplitude_ratio(decibels):
    return 10 ** (decibels / 20)


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

    def test_align_pair(self):
        reference = np.random.uniform(size=1000).astype("float32")
        arr1, arr2 = fast_align_audio.align_pair(
            reference, np.pad(reference, (121, 0)), offset=121, align_mode="crop"
        )
        assert np.array_equal(reference, arr1) and np.array_equal(reference, arr2)

        arr1, arr2 = fast_align_audio.align_pair(
            reference, reference[121:], offset=-121, align_mode="crop"
        )
        assert np.array_equal(reference[400:500], arr1[400:500]) and np.array_equal(
            reference[400:500], arr2[400:500]
        )

    def test_robustness_to_gain_differences(self):
        folder_name = "multi_mic1"
        main, sr = librosa.load(TEST_FIXTURES_DIR / folder_name / "main.flac", sr=None)
        other, _ = librosa.load(
            TEST_FIXTURES_DIR / folder_name / "other1.flac", sr=None
        )

        max_offset_samples = int(0.01 * sr)

        offset_reference, corr = fast_align_audio.find_best_alignment_offset(
            main,
            other,
            max_offset_samples=max_offset_samples,
            method="corr",
        )

        for gain_db in (-10, -20, -30, -40):
            gain = convert_decibels_to_amplitude_ratio(gain_db)
            other_gained = other * gain
            offset_mse, mse = fast_align_audio.find_best_alignment_offset(
                main,
                other_gained,
                max_offset_samples=max_offset_samples,
                method="mse",
            )
            assert offset_mse == offset_reference
            offset_corr, corr = fast_align_audio.find_best_alignment_offset(
                main,
                other_gained,
                max_offset_samples=max_offset_samples,
                method="corr",
            )
            assert offset_corr == offset_reference

    def test_robustness_to_polarity_difference(self):
        folder_name = "multi_mic1"
        main, sr = librosa.load(TEST_FIXTURES_DIR / folder_name / "main.flac", sr=None)
        other, _ = librosa.load(
            TEST_FIXTURES_DIR / folder_name / "other1.flac", sr=None
        )

        max_offset_samples = int(0.01 * sr)

        offset_reference, corr = fast_align_audio.find_best_alignment_offset(
            main,
            other,
            max_offset_samples=max_offset_samples,
            method="corr",
        )

        other_polarity_inversed = -other
        offset_mse, mse = fast_align_audio.find_best_alignment_offset(
            main,
            other_polarity_inversed,
            max_offset_samples=max_offset_samples,
            method="mse",
            consider_both_polarities=True,
        )
        assert offset_mse == offset_reference
        offset_corr, corr = fast_align_audio.find_best_alignment_offset(
            main,
            other_polarity_inversed,
            max_offset_samples=max_offset_samples,
            method="corr",
            consider_both_polarities=True,
        )
        assert offset_corr == offset_reference

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

        max_offset_samples = int(0.01 * sr)

        for i, other in enumerate(others):
            # Test that mse and corr give similar results
            offset_mse, mse = fast_align_audio.find_best_alignment_offset(
                main,
                other,
                max_offset_samples=max_offset_samples,
                method="mse",
            )
            offset_corr, corr = fast_align_audio.find_best_alignment_offset(
                main,
                other,
                max_offset_samples=max_offset_samples,
                method="corr",
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
