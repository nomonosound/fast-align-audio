from typing import Optional, Tuple, List

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


def align_delayed_signal_with_reference(
    reference_signal: NDArray[np.float32],
    delayed_signal: NDArray[np.float32],
    offset: int,
) -> Tuple[NDArray[np.float32], List[Tuple[int, int]]]:
    """
    Align delayed_signal with the reference signal, given the offset.

    The offset denotes the amount of samples the delayed_signal is delayed compared to
    the reference_signal.

    The start or end is filled with `offset` amount of zeros.

    reference_signal and delayed_signal can have different length, but the returned
    array will have the same length as the reference signal.

    This function returns a tuple. The first entry in the tuple is the aligned signal,
    and the second is a list of tuples (start index, end index) that denote any gaps at
    the edges.
    """
    assert type(offset) == int, "offset must be an int"
    placeholder = np.zeros_like(reference_signal)
    gaps = []
    if offset < 0:
        abs_offset = -offset
        insert_length = placeholder.shape[-1] - abs_offset
        gaps.append((0, abs_offset))
        if delayed_signal.shape[-1] > insert_length:
            placeholder[..., abs_offset:] = delayed_signal[..., 0:insert_length]
        else:
            placeholder[..., abs_offset : abs_offset + delayed_signal.shape[-1]] = (
                delayed_signal
            )
            if abs_offset + delayed_signal.shape[-1] < placeholder.shape[-1]:
                gaps.append(
                    (abs_offset + delayed_signal.shape[-1], placeholder.shape[-1])
                )
    elif offset == 0:
        if delayed_signal.shape[-1] > placeholder.shape[-1]:
            placeholder[..., :] = delayed_signal[..., : placeholder.shape[-1]]
        else:
            placeholder[..., 0 : delayed_signal.shape[-1]] = delayed_signal
            if delayed_signal.shape[-1] < placeholder.shape[-1]:
                gaps.append((delayed_signal.shape[-1], placeholder.shape[-1]))
    else:
        # positive offset
        aligned = delayed_signal[..., offset:]
        if aligned.shape[-1] > placeholder.shape[-1]:
            aligned = aligned[..., : placeholder.shape[-1]]
        elif aligned.shape[-1] < placeholder.shape[-1]:
            gaps.append((aligned.shape[-1], placeholder.shape[-1]))
        placeholder[..., : aligned.shape[-1]] = aligned
    return placeholder, gaps


def fill_any_edge_gaps_in_aligned_signal_with_reference(
    reference_signal: NDArray[np.float32],
    aligned_signal: NDArray[np.float32],
    gaps: List[Tuple[int, int]],
    sample_rate: int,
    adjust_reference_gain: bool = True,
    crossfade_duration: float = 0.25,
) -> NDArray[np.float32]:
    raise NotImplementedError()
