import numpy as np


def runtime_state_geometry(mode, center, radius):
    """Return the lightweight RViz label, color, and closed threat circle."""
    label = mode.value if hasattr(mode, 'value') else str(mode)
    color = {
        'HOVER': (0.0, 1.0, 0.0),
        'CRUISE': (0.0, 0.7, 1.0),
        'DODGE': (1.0, 0.0, 0.0),
    }[label]
    center = np.asarray(center, dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * np.pi, 37)
    points = np.column_stack((
        center[0] + float(radius) * np.cos(angles),
        center[1] + float(radius) * np.sin(angles),
        np.full(angles.shape, center[2]),
    ))
    return label, color, points


def pack_xyz32(points):
    """Pack an ``N x >=3`` array as a compact little-endian XYZ32 payload."""
    array = np.asarray(points)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError('points must have shape (N, >=3)')
    xyz = np.ascontiguousarray(array[:, :3], dtype='<f4')
    return xyz.shape[0], xyz.tobytes(), bool(np.isfinite(xyz).all())
