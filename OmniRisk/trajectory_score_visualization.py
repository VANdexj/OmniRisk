import numpy as np


_COLOR_STOPS = np.asarray([
    [0.08, 0.82, 0.32],  # low score: green
    [1.00, 0.76, 0.08],  # middle score: amber
    [0.93, 0.18, 0.16],  # high score: red
], dtype=np.float32)


def normalize_scores(scores):
    """Normalize one inference frame to [0, 1]; equal scores use the green endpoint."""
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    score_min = float(scores.min())
    score_max = float(scores.max())
    if np.isclose(score_min, score_max):
        return np.zeros_like(scores), score_min, score_max
    return (scores - score_min) / (score_max - score_min), score_min, score_max


def score_colors(normalized_scores):
    """Map normalized scores to an intuitive green -> amber -> red RGB ramp."""
    values = np.clip(np.asarray(normalized_scores, dtype=np.float32), 0.0, 1.0)
    section = np.minimum((values * 2.0).astype(np.int32), 1)
    fraction = values * 2.0 - section
    return (_COLOR_STOPS[section] * (1.0 - fraction[:, None])
            + _COLOR_STOPS[section + 1] * fraction[:, None])
