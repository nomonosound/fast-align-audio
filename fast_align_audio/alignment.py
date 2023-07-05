from typing import Optional, Tuple

import numpy as np
import _fast_align_audio
from numpy.typing import NDArray


def fast_autocorr(original, delayed, t=1):
    """Only every 4th sample is considered in order to improve execution time"""
    if t == 0:
        return np.corrcoef([original[::4], delayed[::4]])[1, 0]
    elif t < 0:
        return np.corrcoef([original[-t::4], delayed[:t:4]])[1, 0]
    else:
        return np.corrcoef([original[:-t:4], delayed[t::4]])[1, 0]


def find_best_alignment_offset_with_corr_coef(
    reference_signal: NDArray[np.float32],
    delayed_signal: NDArray[np.float32],
    min_offset_samples: int,
    max_offset_samples: int,
    lookahead_samples: Optional[int] = None,
    consider_both_polarities: bool = True,
):
    """
    Returns the estimated delay (in samples) between the original and delayed signal,
    calculated using correlation coefficients. The delay is optimized to maximize the
    correlation between the signals.

    Args:
        reference_signal (NDArray[np.float32]): The original signal array.
        delayed_signal (NDArray[np.float32]): The delayed signal array.
        min_offset_samples (int): The minimum delay offset to consider, in samples.
                                  Can be negative.
        max_offset_samples (int): The maximum delay offset to consider, in samples.
        lookahead_samples (Optional[int]): The number of samples to look at
                                           while estimating the delay. If None, the
                                           whole delayed signal is considered.
        consider_both_polarities (bool): If True, the function will consider both positive
                                         and negative correlations, which corresponds to
                                         the same or opposite polarities in signals,
                                         respectively. Defaults to True.

    Returns:
        tuple: Estimated delay (int) and correlation coefficient (float).
    """
    if lookahead_samples is not None and len(reference_signal) > lookahead_samples:
        middle_of_signal_index = int(np.floor(len(reference_signal) / 2))
        original_signal_slice = reference_signal[
            middle_of_signal_index : middle_of_signal_index + lookahead_samples
        ]
        delayed_signal_slice = delayed_signal[
            middle_of_signal_index : middle_of_signal_index + lookahead_samples
        ]
    else:
        original_signal_slice = reference_signal
        delayed_signal_slice = delayed_signal

    coefs = []
    for lag in range(min_offset_samples, max_offset_samples):
        correlation_coef = fast_autocorr(
            original_signal_slice, delayed_signal_slice, t=lag
        )
        coefs.append(correlation_coef)

    if consider_both_polarities:
        # In this mode we aim to find the correlation coefficient of highest magnitude.
        # We do this to consider the possibility that the delayed signal has opposite
        # polarity compared to the original signal, in which case the correlation
        # coefficient would be negative.
        most_extreme_coef_index = int(np.argmax(np.abs(coefs)))
    else:
        most_extreme_coef_index = int(np.argmax(coefs))
    most_extreme_coef = coefs[most_extreme_coef_index]
    offset = most_extreme_coef_index + min_offset_samples
    return offset, most_extreme_coef


def find_best_alignment_offset(
    reference_signal: NDArray[np.float32],
    delayed_signal: NDArray[np.float32],
    max_offset_samples: int,
    lookahead_samples: Optional[int] = None,
    method: str = "mse",
    consider_both_polarities: bool = False,
) -> Tuple[int, float]:
    """
    Find best offset of `delayed_audio` w.r.t. `reference_audio`.

    Best = smallest mean squared error (mse).

    Args:
        reference_audio, delayed_audio (float32 C-contiguous NumPy arrays):
            The arrays to compare
        max_offset_samples (int > 0):
            Maximum expected offset. It will not find any larger offsets.
        lookahead_samples (int > 0, optional):
            Maximum number of array elements to use for each mse computation.
            If `None` (the default), there is no maximum.
        method: "mse" (fast) or "corr" (slow)
        consider_both_polarities: If set to true, also consider the possibility that the
            delayed signal could possibly have opposite polarity compared to the
            reference_signal. This doubles the execution time when method=="mse".

    Return offset and metric (mse or corr coef)
    """
    assert {reference_signal.dtype, delayed_signal.dtype} == {
        np.dtype("float32")
    }, "Arrays must be float32"
    assert reference_signal.ndim == 1
    assert delayed_signal.ndim == 1

    if method == "mse":
        assert (
            reference_signal.flags["C_CONTIGUOUS"]
            and delayed_signal.flags["C_CONTIGUOUS"]
        ), "Arrays must be C-contiguous"
        if lookahead_samples is None:
            lookahead_samples = max(len(reference_signal), len(delayed_signal))

        result = _fast_align_audio.lib.fast_find_alignment(
            len(delayed_signal),
            _fast_align_audio.ffi.cast("float *", delayed_signal.ctypes.data),
            len(reference_signal),
            _fast_align_audio.ffi.cast("float *", reference_signal.ctypes.data),
            max_offset_samples,
            lookahead_samples,
        )
        if consider_both_polarities:
            inverse_polarity_result = _fast_align_audio.lib.fast_find_alignment(
                len(delayed_signal),
                _fast_align_audio.ffi.cast("float *", (-delayed_signal).ctypes.data),
                len(reference_signal),
                _fast_align_audio.ffi.cast("float *", reference_signal.ctypes.data),
                max_offset_samples,
                lookahead_samples,
            )
            if inverse_polarity_result.min_val < result.min_val:
                return inverse_polarity_result.min_idx, inverse_polarity_result.min_val
        return result.min_idx, result.min_val
    elif method == "corr":
        return find_best_alignment_offset_with_corr_coef(
            reference_signal=reference_signal,
            delayed_signal=delayed_signal,
            min_offset_samples=-max_offset_samples,
            max_offset_samples=max_offset_samples,
            consider_both_polarities=consider_both_polarities,
        )
    else:
        raise ValueError("Unknown method")


def align(a, b, max_offset, max_lookahead=None, *, align_mode, fix_length=None):
    # TODO: Fix the implementation and test it? Or remove the function
    """
    Align `a` and `b`. See the documentation of `find_best_alignment_offset` for most of the args.

    Args:
        align_mode (Either `"crop"` or `"pad"`): How to align `a` and `b`.
            If `crop`, "best_offset" number of elements are removed from the
            front of the "too-long" array. If `pad`, "best_offset" number of
            elements are padding to the front of the "too-short" array.
        fix_length (Either `"shortest"`, `"longest"` or `None`): How to fix the
            lengths of `a` and `b` after alignment. If `shortest`, the longer
            array is cropped (at the end/right) to the length of the shorter one.
            If `longest`, the shorter array is padded (to the end/right) to the
            length of the longest one.  If `None`, lengths are not changed.
    """
    offset, _ = find_best_alignment_offset(b, a, max_offset, max_lookahead)
    if offset > 0:
        # mse(a[offset:], b) = min
        a, b = _align(a, b, offset, align_mode)
    else:
        # mse(a, b[offset:]) = min
        b, a = _align(b, a, -offset, align_mode)
    a, b = _fix(a, b, fix_length)
    return a, b


def align_delayed_signal_with_reference(
    reference_signal:NDArray[np.float32] , delayed_signal, offset
):
    """
    Return delayed_signal padded with zeros in the start or the end,
    depending on whether the offset is negative or positive.
    """
    placeholder = np.zeros_like(reference_signal)
    if offset < 0:
        offset = -offset
        placeholder[offset:] = delayed_signal[0:-offset]
    else:
        placeholder[0:-offset] = delayed_signal[offset:]
    return placeholder


def _align(x, y, offset, align_mode):
    if align_mode == "crop":
        x = x[offset:]
    elif align_mode == "pad":
        y = np.pad(y, (offset, 0))
    return x, y


def _fix(x, y, fix_mode):
    if fix_mode is None:
        return x, y
    elif fix_mode == "shortest":
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]
        return x, y
    elif fix_mode == "longest":
        max_len = max(len(x), len(y))
        x = np.pad(x, (0, max_len - len(x)))
        y = np.pad(y, (0, max_len - len(y)))
        return x, y
    else:
        raise ValueError(f"fix_length={fix_mode!r} not understood")
