# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2025-06-28

### Added

* Add support for Python 3.13

## [0.5.0] - 2025-06-28

### Added

* Add support for Python 3.12

## [0.4.0] - 2025-03-18

### Added

* Distribute source (.tar.gz) on PyPI in addition to built wheels

### Changed

* **Breaking change**: The first argument of `align_delayed_signal_with_reference`, is now `reference_length` (int) instead of `reference_signal` (NDArray[np.float32])
* Target numpy 2.x instead of 1.x. If you still depend on numpy 1.x, you need an older version of fast-align-audio.

### Removed

* Remove support for Python 3.8
* Remove musllinux from the build matrix

## [0.3.0] - 2024-01-09

### Changed

* **Breaking change**: `align_delayed_signal_with_reference` now returns a list of gaps in addition to the aligned signal. So it now returns a `Tuple[NDArray[np.float32], List[Tuple[int, int]]]` instead of a `NDArray[np.float32]`.
* Add support for python 3.10 and 3.11

### Fixed

* Fix a bug where `align_delayed_signal_with_reference` didn't work when the input was 2D and the offset was positive

## [0.2.1] - 2023-09-13

### Changed

* Add support for different length arrays in `align_delayed_signal_with_reference`. The API remains unchanged.

## [0.2.0] - 2023-07-05

### Added

* Implement an additional alignment method based on correlation coefficients. Set `method="corr"` in `find_best_alignment_offset` to use it.
* Add support for inverse polarity in `find_best_alignment_offset` (with `consider_both_polarities=True`)
* Add function `align_delayed_signal_with_reference`
* Explicitly list numpy as a dependency
* Set up a test system

### Changed

* Rename `best_offset` to `find_best_alignment_offset`. Rename and reorder its arguments.
* `find_best_alignment_offset` now returns a tuple with offset (int) and metric (MSE or correlation coefficient, depending on method)

### Removed

* Remove `align` function

## [0.1.2] - 2023-07-04

Rename fast_align_audio module to alignment to make it less ambiguous how to do imports

## [0.1.1] - 2023-06-30

Initial release
