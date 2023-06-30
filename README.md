# fast-align-audio: high-speed NumPy audio alignment

fast-align-audio is designed to swiftly align two similar NumPy arrays — a common need
in various fields including audio signal processing. If you have two arrays where one
"lags behind" the other due to factors such as different capture sensors (microphones),
propagation delays, or post-processing like reverberation or MP3 compression,
fast-align-audio is here to help.

The package employs a "brute force" alignment approach, leveraging a C-based algorithm
for maximum speed while providing a user-friendly Python API for easy integration.

While this library was initially developed with audio ndarrays in mind, it could also be
used to align other kinds of time-series data that are represented as NumPy arrays.

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
arr = np.random.uniform(size=10_000).astype("float32")

# Find the best offset for aligning two arrays
print(fast_align_audio.best_offset(arr, np.pad(arr, (121, 0)), 1_000, 5_000))
# Output: -121

print(fast_align_audio.best_offset(arr, arr[121:], 1_000, 5_000))
# Output: 121

# Align two arrays and confirm they're equal post alignment
arr1, arr2 = fast_align_audio.align(arr, np.pad(arr, (121, 0)), 1_000, 5_000, align_mode="crop")
np.array_equal(arr, arr1) and np.array_equal(arr, arr2)
# Output: True
```

In this example, we first create a random numpy array. We then call the best_offset
method to find the best offset to align two arrays, and we use the align method to align
the arrays. The np.array_equal method checks if two arrays are equal, demonstrating the
successful alignment of the two original arrays.
