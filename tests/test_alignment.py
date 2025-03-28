import os
from pathlib import Path

import soundfile as sf
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import fast_align_audio

DEMO_DIR = Path(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
TEST_FIXTURES_DIR = DEMO_DIR / "test_fixtures"

DEBUG = False


def convert_decibels_to_amplitude_ratio(decibels):
    return 10 ** (decibels / 20)


@pytest.fixture
def load_audio_file():
    def _load_audio_file(folder_name, filename):
        data, sr = sf.read(TEST_FIXTURES_DIR / folder_name / filename)
        data = data.astype(np.float32)
        return data, sr

    return _load_audio_file


def test_simple_alignment_positive_offset():
    reference = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    delayed = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    offset, mse = fast_align_audio.find_best_alignment_offset(
        reference, delayed, max_offset_samples=4
    )
    assert offset == 1

    aligned, _ = fast_align_audio.align_delayed_signal_with_reference(
        reference.shape[-1], delayed, offset
    )
    assert_array_almost_equal(aligned, reference)


def test_simple_alignment_negative_offset():
    reference = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    delayed = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    offset, mse = fast_align_audio.find_best_alignment_offset(
        reference, delayed, max_offset_samples=4
    )
    assert offset == -1

    aligned, _ = fast_align_audio.align_delayed_signal_with_reference(
        reference.shape[-1], delayed, offset
    )
    assert_array_almost_equal(aligned, reference)


def test_robustness_to_gain_differences(load_audio_file):
    folder_name = "multi_mic1"
    main, sr = load_audio_file(folder_name, "main.flac")
    other, _ = load_audio_file(folder_name, "other1.flac")

    max_offset_samples = int(0.01 * sr)

    offset_reference, corr = fast_align_audio.find_best_alignment_offset(
        main,
        other,
        max_offset_samples=max_offset_samples,
        method="corr",
    )

    for gain_db in (-10, -20, -30):  # TODO: Make it able to go down to -50
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


def test_robustness_to_polarity_difference(load_audio_file):
    folder_name = "multi_mic1"
    main, sr = load_audio_file(folder_name, "main.flac")
    other, _ = load_audio_file(folder_name, "other1.flac")

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
def test_multi_mic(folder_name, load_audio_file):
    main, sr = load_audio_file(folder_name, "main.flac")

    other_filenames = ["other1.flac", "other2.flac", "other3.flac"]
    others = []
    for other_filename in other_filenames:
        other, _ = load_audio_file(folder_name, other_filename)
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

        aligned, _ = fast_align_audio.align_delayed_signal_with_reference(
            main.shape[-1], other, offset_corr
        )
        if DEBUG:
            sf.write(
                TEST_FIXTURES_DIR
                / folder_name
                / f"{Path(other_filenames[i]).stem}_aligned.wav",
                aligned,
                int(sr),
            )


def test_shorter_delayed_audio_alignment_zero_offset():
    reference_audio = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    delayed_audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    offset = 0
    aligned_audio, gaps = fast_align_audio.align_delayed_signal_with_reference(
        reference_audio.shape[-1], delayed_audio, offset
    )
    assert_array_almost_equal(
        aligned_audio, np.array([1.0, 2.0, 3.0, 0.0, 0.0], dtype=np.float32)
    )
    assert gaps == [(3, 5)]


def test_longer_delayed_audio_alignment_zero_offset():
    reference_audio = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    delayed_audio = np.array([1.0, 2.0, 3.0, 4.1, 5.1, 6.0, 7.1], dtype=np.float32)
    offset = 0
    aligned_audio, gaps = fast_align_audio.align_delayed_signal_with_reference(
        reference_audio.shape[-1], delayed_audio, offset
    )
    assert_array_almost_equal(
        aligned_audio, np.array([1.0, 2.0, 3.0, 4.1, 5.1], dtype=np.float32)
    )
    assert gaps == []


def test_shorter_delayed_audio_alignment_positive_offset():
    reference_audio = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    delayed_audio = np.array([0.5, 1.0, 2.0, 3.1], dtype=np.float32)
    offset = 1
    aligned_audio, gaps = fast_align_audio.align_delayed_signal_with_reference(
        reference_audio.shape[-1], delayed_audio, offset
    )
    assert_array_almost_equal(
        aligned_audio, np.array([1.0, 2.0, 3.1, 0.0, 0.0], dtype=np.float32)
    )
    assert gaps == [(3, 5)]


def test_shorter_delayed_audio_alignment_positive_offset_2d():
    reference_audio = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    delayed_audio = np.array([[0.5, 1.0, 2.0, 3.1]], dtype=np.float32)
    offset = 1
    aligned_audio, gaps = fast_align_audio.align_delayed_signal_with_reference(
        reference_audio.shape[-1], delayed_audio, offset
    )
    assert_array_almost_equal(
        aligned_audio, np.array([[1.0, 2.0, 3.1, 0.0, 0.0]], dtype=np.float32)
    )
    assert gaps == [(3, 5)]


def test_longer_delayed_audio_alignment_positive_offset():
    reference_audio = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    delayed_audio = np.array([0.5, 1.0, 2.0, 3.1, 4.1, 5.0, 6.0], dtype=np.float32)
    offset = 1
    aligned_audio, gaps = fast_align_audio.align_delayed_signal_with_reference(
        reference_audio.shape[-1], delayed_audio, offset
    )
    assert_array_almost_equal(
        aligned_audio, np.array([1.0, 2.0, 3.1, 4.1, 5.0], dtype=np.float32)
    )
    assert gaps == []


def test_shorter_delayed_audio_alignment_negative_offset():
    reference_audio = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    delayed_audio = np.array([3.0, 4.0, 5.1], dtype=np.float32)
    offset = -2
    aligned_audio, gaps = fast_align_audio.align_delayed_signal_with_reference(
        reference_audio.shape[-1], delayed_audio, offset
    )
    assert_array_almost_equal(
        aligned_audio, np.array([0.0, 0.0, 3.0, 4.0, 5.1], dtype=np.float32)
    )
    assert gaps == [(0, 2)]


def test_audio_alignment_negative_offset_gap_at_both_ends():
    reference_audio = np.array([1.0, 2.0, 3.0, 4.1, 5.0, 6.0, 7.0], dtype=np.float32)
    delayed_audio = np.array([3.0, 4.11, 5.0], dtype=np.float32)
    offset = -2
    aligned_audio, gaps = fast_align_audio.align_delayed_signal_with_reference(
        reference_audio.shape[-1], delayed_audio, offset
    )
    assert_array_almost_equal(
        aligned_audio, np.array([0.0, 0.0, 3.0, 4.11, 5.0, 0.0, 0.0], dtype=np.float32)
    )
    assert gaps == [(0, 2), (5, 7)]


def test_longer_delayed_audio_alignment_negative_offset():
    reference_audio = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    delayed_audio = np.array([3.1, 4.0, 5.0, 6.0, 7.1, 8.0], dtype=np.float32)
    offset = -2
    aligned_audio, gaps = fast_align_audio.align_delayed_signal_with_reference(
        reference_audio.shape[-1], delayed_audio, offset
    )
    assert_array_almost_equal(
        aligned_audio, np.array([0.0, 0.0, 3.1, 4.0, 5.0], dtype=np.float32)
    )
    assert gaps == [(0, 2)]


def test_longer_delayed_audio_alignment_negative_offset_2d():
    reference_audio = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    delayed_audio = np.array([[3.1, 4.0, 5.0, 6.0, 7.1, 8.0]], dtype=np.float32)
    offset = -2
    aligned_audio, gaps = fast_align_audio.align_delayed_signal_with_reference(
        reference_audio.shape[-1], delayed_audio, offset
    )
    assert_array_almost_equal(
        aligned_audio, np.array([[0.0, 0.0, 3.1, 4.0, 5.0]], dtype=np.float32)
    )
    assert gaps == [(0, 2)]
