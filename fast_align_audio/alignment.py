import numpy as np
import _fast_align_audio


def best_offset(a, b, max_offset, max_lookahead=None):
    """
    Find best offset of `a` w.r.t. `b`.

    Best = smallest mean squared error (mse).

    If the returned offset is positive, it means the smallest mse is:

        ((a[n:...] - b[:...])**2).mean()

    If the returned is negative, it means the smallest mse is:

        ((a[:...] - b[n:...])**2).mean()

    (Here, `...` means that you have to account for different array lengths for
    this computation to actually work.)

    Args:
        a, b (float32 C-contiguous NumPy arrays):
            The arrays to compare
        max_offset (int > 0):
            Maximum expected offset. It will not find any larger offsets.
        max_lookahead (int > 0, optional):
            Maximum number of array elements to use for each mse computation.
            If `None` (the default), there is no maximum.
    """
    assert {a.dtype, b.dtype} == {np.dtype("float32")}, "Arrays must be float32"
    assert (
        a.flags["C_CONTIGUOUS"] and b.flags["C_CONTIGUOUS"]
    ), "Arrays must be C-contiguous"
    if max_lookahead is None:
        max_lookahead = max(len(a), len(b))
    return _fast_align_audio.lib.fast_find_alignment(
        len(a),
        _fast_align_audio.ffi.cast("float *", a.ctypes.data),
        len(b),
        _fast_align_audio.ffi.cast("float *", b.ctypes.data),
        max_offset,
        max_lookahead,
    )


def align(a, b, max_offset, max_lookahead=None, *, align_mode, fix_length=None):
    """
    Align `a` and `b`. See the documentation of `best_offset` for most of the args.

    Args:
        align_mode (Either `"crop"` or `"pad"`): How to align `a` and `b`.
            If `crop`, `best_offset` number of elements are removed from the
            front of the "too-long" array. If `pad`, `best_offset` number of
            elements are padding to the front of the "too-short" array.
        fix_length (Either `"shortest"`, `"longest"` or `None`): How to fix the
            lengths of `a` and `b` after alignment. If `shortest`, the longer
            array is cropped (at the end/right) to the length of the shorter one.
            If `longest`, the shorter array is padded (to the end/right) to the
            length of the longest one.  If `None`, lengths are not changed.
    """
    offset = best_offset(a, b, max_offset, max_lookahead)
    if offset > 0:
        # mse(a[offset:], b) = min
        a, b = _align(a, b, offset, align_mode)
    else:
        # mse(a, b[offset:]) = min
        b, a = _align(b, a, -offset, align_mode)
    a, b = _fix(a, b, fix_length)
    return a, b


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
