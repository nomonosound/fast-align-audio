# fast-align-audio: high-speed NumPy audio alignment

fast-align-audio is designed to swiftly align two similar 1-dimensional NumPy arrays — a common need
in various fields including audio signal processing. If you have two arrays where one
"lags behind" the other due to factors such as different capture sensors (microphones),
propagation delays, or post-processing like reverberation or MP3 compression,
fast-align-audio is here to help.

The package employs a "brute force" alignment approach, leveraging a C-based algorithm
for maximum speed while providing a user-friendly Python API for easy integration.

While this library was initially developed with audio ndarrays in mind, it could also be
used to align other kinds of time-series data that are represented as 1D NumPy arrays.

# Installation

[![PyPI version](https://img.shields.io/pypi/v/fast-align-audio.svg?style=flat)](https://pypi.org/project/fast-align-audio/)
![python 3.9, 3.10, 3.11](https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11-blue)
![os: Linux, Windows](https://img.shields.io/badge/OS-Linux%20%28x86--64%29%20|%20Windows%20%28x86--64%29-blue)

```
$ pip install fast-align-audio
```

# Usage

Here is a basic usage example:

```py
import fast_align_audio
import numpy as np

# Create a random NumPy array
reference = np.random.uniform(size=10_000).astype("float32")
delayed = np.pad(reference, (121, 0))[0:10_000]

# Find the best offset for aligning two arrays
offset, mse = fast_align_audio.find_best_alignment_offset(
    reference_signal=reference,
    delayed_signal=delayed,
    max_offset_samples=1000,
    lookahead_samples=5000,
)
print(offset)  # 121

negative_offset, mse2 = fast_align_audio.find_best_alignment_offset(
    reference_signal=reference,
    delayed_signal=reference[121:],
    max_offset_samples=1000,
    lookahead_samples=5000,
)
print(negative_offset)  # -121

# Align two arrays and confirm they're equal post alignment
aligned_audio, gaps = fast_align_audio.align_delayed_signal_with_reference(
    reference.shape[-1], delayed, offset=offset
)
print(np.array_equal(reference[500:600], aligned_audio[500:600]))  # True
```

In this example, we first create a random numpy array. We then call the `find_best_alignment_offset`
method to find the best offset to align two arrays, and we use the align method to align
the arrays. The np.array_equal method checks if two arrays are equal, demonstrating the
successful alignment of the two original arrays.

# Tips

* For more reliable alignments, filter out unwanted/unrelated sounds before passing the audio snippets to fast-align-audio. E.g. if you are aligning two speech recordings, you could band-pass filter and/or denoise them first.
* This library assumes that the delay is fixed throughout the audio snippet. If you need something that aligns audio tracks in a dynamic way (e.g. due to distance between microphones changing over time), look elsewhere.
* The `"mse"` method is sensitive to loudness differences. If you use this method, make sure the two input audio snippets have roughly the same loudness
* This lib only works well for small offsets, like up to 500 ms, and suitable audio file durations, like for example between 3 and 45 seconds. If you have large audio files with large offsets between them, a different algorithm may be required to solve the problem well.

# Changelog

## [0.4.0] - 2025-03-18

### Added

* Distribute source (.tar.gz) on PyPI in addition to wheels

### Changed

* **Breaking change**: The first argument of `align_delayed_signal_with_reference`, is now `reference_length` (`int`) instead of `reference_signal` (`NDArray[np.float32]`)
* Target numpy 2.x instead of 1.x. If you still depend on numpy 1.x, you need an older version of fast-align-audio.

### Removed

* Remove support for Python 3.8
* Remove musllinux from the build matrix

For the complete changelog, go to [CHANGELOG.md](CHANGELOG.md)

# Development

* Install dev/build/test dependencies as denoted in setup.py
* `python setup.py develop`
* `pytest`

# Acknowledgements

Original C implementation by jonashaag. Now maintained/backed by [Nomono](https://nomono.co/).
