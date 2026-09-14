import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - the deployed environment includes OpenCV.
    cv2 = None


_GAUSSIAN_1D = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0


def _interpolate_lines(values, valid, axis, wrap):
    """Interpolate all rows/columns and return their nearest support distance."""
    if axis == 0:
        values = values.T
        valid = valid.T

    _, line_length = values.shape
    positions = np.arange(line_length, dtype=np.int32)[None, :]
    has_support = valid.any(axis=1)

    left = np.maximum.accumulate(
        np.where(valid, positions, -1), axis=1)
    right = np.minimum.accumulate(
        np.where(valid, positions, line_length)[:, ::-1], axis=1)[:, ::-1]

    if wrap:
        first = right[:, 0]
        last = left[:, -1]
        left = np.where(left < 0, last[:, None] - line_length, left)
        right = np.where(right >= line_length, first[:, None] + line_length, right)
        left_values = np.take_along_axis(values, left % line_length, axis=1)
        right_values = np.take_along_axis(values, right % line_length, axis=1)
        span = np.maximum(right - left, 1)
        interpolated = np.where(
            valid, values,
            left_values + (right_values - left_values) * (positions - left) / span)
        support_distance = np.minimum(
            positions - left, right - positions).astype(np.float32)
    else:
        left_values = np.take_along_axis(values, np.maximum(left, 0), axis=1)
        right_values = np.take_along_axis(
            values, np.minimum(right, line_length - 1), axis=1)
        between = (left >= 0) & (right < line_length)
        span = np.maximum(right - left, 1)
        interpolated = np.where(
            between,
            left_values + (right_values - left_values) * (positions - left) / span,
            np.where(left >= 0, left_values, right_values))
        support_distance = np.minimum(
            np.where(left >= 0, positions - left, np.inf),
            np.where(right < line_length, right - positions, np.inf)).astype(np.float32)

    interpolated = np.asarray(interpolated, dtype=np.float32)
    interpolated[~has_support] = np.nan
    support_distance[~has_support] = np.inf

    if axis == 0:
        return interpolated.T, support_distance.T
    return interpolated, support_distance


def _gaussian_smooth(values):
    """Apply the fixed separable kernel with panorama-aware border handling."""
    horizontal_pad = np.pad(values, ((0, 0), (2, 2)), mode='wrap')
    if cv2 is not None:
        return cv2.sepFilter2D(
            horizontal_pad, -1, _GAUSSIAN_1D, _GAUSSIAN_1D,
            borderType=cv2.BORDER_REPLICATE)[:, 2:-2]

    horizontal_windows = np.lib.stride_tricks.sliding_window_view(
        horizontal_pad, 5, axis=1)
    horizontal = np.tensordot(
        horizontal_windows, _GAUSSIAN_1D, axes=([-1], [0]))
    vertical_pad = np.pad(horizontal, ((2, 2), (0, 0)), mode='edge')
    vertical_windows = np.lib.stride_tricks.sliding_window_view(
        vertical_pad, 5, axis=0)
    return np.tensordot(vertical_windows, _GAUSSIAN_1D, axes=([-1], [0]))


def complete_depth(depth, invalid_mask, empty_value=None):
    """Fill only invalid range-image holes with directional push-pull interpolation.

    Horizontal interpolation wraps across the 360-degree panorama seam. Vertical
    interpolation does not wrap. A small Gaussian pass smooths only the pixels that
    were invalid on input; original finite measurements are copied back unchanged.
    ``empty_value`` supplies the range sentinel for an entirely non-finite image.
    """
    source = np.asarray(depth)
    invalid = np.asarray(invalid_mask, dtype=bool)
    if source.ndim != 2 or invalid.shape != source.shape:
        raise ValueError("depth and invalid_mask must be equally shaped 2D arrays")

    invalid = invalid | ~np.isfinite(source)
    valid = ~invalid
    result = source.copy()
    if not invalid.any():
        return result
    if not valid.any():
        if not np.isfinite(result).all():
            finite_invalid = result[np.isfinite(result)]
            dtype_max = (np.finfo(result.dtype).max
                         if np.issubdtype(result.dtype, np.floating)
                         else np.finfo(np.float32).max)
            fill_value = (float(np.max(finite_invalid)) if finite_invalid.size else
                          dtype_max)
            if empty_value is not None:
                fill_value = float(empty_value)
            if not np.isfinite(fill_value):
                raise ValueError("empty_value must be finite")
            result[~np.isfinite(result)] = fill_value
        return result

    values = source.astype(np.float32, copy=True)
    known = valid.copy()

    # Two push-pull rounds are sufficient: the first extends rows/columns that
    # contain measurements; the second reaches rows/columns containing only holes.
    for _ in range(2):
        horizontal, horizontal_distance = _interpolate_lines(
            values, known, axis=1, wrap=True)
        vertical, vertical_distance = _interpolate_lines(
            values, known, axis=0, wrap=False)
        fillable = invalid & ~known & (
            np.isfinite(horizontal) | np.isfinite(vertical))
        if not fillable.any():
            break

        use_horizontal = horizontal_distance < vertical_distance
        tie = horizontal_distance == vertical_distance
        candidates = np.where(use_horizontal, horizontal, vertical)
        candidates[tie] = np.fmin(horizontal[tie], vertical[tie])
        values[fillable] = candidates[fillable]
        known[fillable] = True

    # Use wrapped horizontal padding for the panorama and replicated vertical
    # padding for the non-periodic elevation boundary.
    smoothed = _gaussian_smooth(values)
    completed = invalid & known
    result[completed] = smoothed[completed]
    return result
