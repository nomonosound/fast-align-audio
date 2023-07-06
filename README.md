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
aligned = fast_align_audio.align_delayed_signal_with_reference(
    reference, delayed, offset=offset
)
print(np.array_equal(reference[500:600], aligned[500:600]))  # True
```

In this example, we first create a random numpy array. We then call the `find_best_alignment_offset`
method to find the best offset to align two arrays, and we use the align method to align
the arrays. The np.array_equal method checks if two arrays are equal, demonstrating the
successful alignment of the two original arrays.

# Tips

* For more reliable alignments, filter out unwanted/unrelated sounds before passing the audio snippets to fast-align-audio. E.g. if you are aligning two speech recordings, you could band-pass filter and/or denoise them first.
* This library assumes that the delay is fixed throughout the audio snippet. If you need something that aligns audio tracks in a dynamic way (e.g. due to distance between microphones changing over time), look elsewhere.
* The `"mse"` method is sensitive to loudness differences. If you use this method, make sure the two input audio snippets have roughly the same loudness

# Development

* Install dev/build/test dependencies as denoted in setup.py
* `python setup.py develop`
* `pytest`

# Acknowledgements

Original C implementation by jonashaag. Now maintained/backed by [Nomono](https://nomono.co/).
