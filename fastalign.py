import numpy as np
import _fastalign


def fastalign(a, b, max_offset, max_lookahead):
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
            Maximum expected offset. `fastalign` will not find any larger offsets.
        max_lookahead (int > 0):
            Maximum number of array elements to use for each mse computation.
    """
    assert {a.dtype, b.dtype} == {np.dtype("float32")}, "Arrays must be float32"
    assert a.flags["C_CONTIGUOUS"] and b.flags["C_CONTIGUOUS"], "Arrays must be C-contiguous"
    return _fastalign.lib.fast_find_alignment(
        len(a), _fastalign.ffi.cast("float *", a.ctypes.data),
        len(b), _fastalign.ffi.cast("float *", b.ctypes.data),
        max_offset, max_lookahead,
    )
