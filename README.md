Stupid but fast "brute force" method of finding best alignment offset of two
NumPy arrays. Useful if you have two similar (but not identical) arrays where
one of the arrays "lags behind" the other and you want to align them.

Compile Numba extension:

```
$ python _fastalign_numba.py
```

Use:

```py
import numpy as np
import fastalign

arr = np.random.uniform(size=10_000).astype("float32")

print(fastalign.best_offset(arr, np.pad(arr, (121, 0)), 1_000, 5_000))
# => -121
print(fastalign.best_offset(arr, arr[121:], 1_000, 5_000))
# => 121

arr1, arr2 = fastalign.align(arr, np.pad(arr, (121, 0)), 1_000, 5_000, align_mode="crop")
np.array_equal(arr, arr1) and np.array_equal(arr, arr2)
# => True
```
