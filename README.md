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

a = np.random.uniform(size=10_000).astype("float32")
print(fastalign(a, np.pad(a, (121, 0)), 1_000, 5_000))
# => -121
print(fastalign(a, a[121:], 1_000, 5_000))
# => 121
```
