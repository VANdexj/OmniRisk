"""ROS-independent spherical range-grid and dynamic sphere compositing."""

from functools import lru_cache

import numpy as np


def make_ray_directions(height, panorama_width, elevation_min, elevation_max):
    """Return planning-frame unit rays on the one canonical panorama grid."""
    rows = np.arange(height, dtype=np.float32)
    cols = np.arange(panorama_width, dtype=np.float32)
    elevation = elevation_max - rows / (height - 1) * (elevation_max - elevation_min)
    azimuth = cols / panorama_width * (2.0 * np.pi) - np.pi
    cos_elevation = np.cos(elevation)[:, None]
    return np.stack((
        cos_elevation * np.cos(azimuth)[None, :],
        cos_elevation * np.sin(azimuth)[None, :],
        np.broadcast_to(np.sin(elevation)[:, None], (height, panorama_width)),
    ), axis=-1).astype(np.float32)


def project_points_to_range_image(points, height, panorama_width, elevation_min,
                                  elevation_max, min_distance, max_distance):
    """Project planning-frame points to their nearest canonical-grid ray."""
    range_image = np.full((height, panorama_width), max_distance, dtype=np.float32)
    if points is None or len(points) == 0:
        return range_image

    points = np.asarray(points, dtype=np.float32)
    points = points[np.isfinite(points).all(axis=1)]
    if points.size == 0:
        return range_image
    x, y, z = points.T
    distance = np.linalg.norm(points, axis=1)
    elevation = np.arctan2(z, np.hypot(x, y))
    elevation_tolerance = np.finfo(np.float32).eps
    valid = ((distance > min_distance) & (distance < max_distance) &
             (elevation >= elevation_min - elevation_tolerance) &
             (elevation <= elevation_max + elevation_tolerance))
    if not valid.any():
        return range_image

    # Use float64 only for azimuth binning so CPU and CUDA math libraries do
    # not straddle a one-degree half-pixel boundary by float32 ULPs.
    azimuth = np.arctan2(
        y[valid].astype(np.float64), x[valid].astype(np.float64))
    row = np.rint((elevation_max - elevation[valid]) /
                  (elevation_max - elevation_min) * (height - 1)).astype(np.int32)
    col = np.rint((azimuth + np.pi) / (2.0 * np.pi) * panorama_width).astype(np.int32)
    row = np.clip(row, 0, height - 1)
    col %= panorama_width
    np.minimum.at(range_image, (row, col), distance[valid])
    return range_image


@lru_cache(maxsize=8)
def source_columns(panorama_width, network_width):
    """Return the sole 360-degree panorama-to-network column LUT."""
    columns = np.minimum(
        np.floor(np.arange(network_width) * panorama_width / network_width).astype(np.int32),
        panorama_width - 1)
    columns.flags.writeable = False
    return columns


def map_panorama_to_network(values, panorama_width, network_width, omnidirectional):
    """Map an HxW[...] panorama to the network with the shared column policy."""
    if values.shape[1] != panorama_width:
        raise ValueError('panorama width does not match values')
    if omnidirectional:
        return values[:, source_columns(panorama_width, network_width)]
    if network_width > panorama_width:
        raise ValueError('network width exceeds panorama width in non-omnidirectional mode')
    start = panorama_width // 2 - network_width // 2
    return values[:, start:start + network_width]


def _sphere_intersection_depth(rays, center, radius, min_distance):
    """Return exact near (or inside-sphere far) intersections for a ray subset."""
    oc = -center
    b = rays @ oc
    c = float(oc @ oc - radius * radius)
    discriminant = b * b - c
    hit = discriminant >= 0.0
    root = np.sqrt(np.maximum(discriminant, 0.0))
    near = -b - root
    far = -b + root
    return np.where(hit, np.where(near > min_distance, near,
                                  np.where(far > min_distance, far, np.inf)), np.inf)


def _sphere_roi(center, radius, row_elevations, column_azimuths):
    """Return a conservative row slice and one or two wrapped column slices."""
    center64 = np.asarray(center, dtype=np.float64)
    radius64 = float(radius)
    horizontal_distance = float(np.hypot(center64[0], center64[1]))
    distance = float(np.linalg.norm(center64))
    height = row_elevations.size
    width = column_azimuths.size

    if distance <= radius64:
        return slice(0, height), (slice(0, width),)

    angular_radius = np.arcsin(np.clip(radius64 / distance, 0.0, 1.0))
    center_elevation = np.arctan2(center64[2], horizontal_distance)
    elevation_keep = ((row_elevations >= center_elevation - angular_radius) &
                      (row_elevations <= center_elevation + angular_radius))
    rows = np.flatnonzero(elevation_keep)
    if rows.size == 0:
        return None
    # The continuous angular bound is expanded by one canonical ray in each
    # direction. This protects tangent rays from float32 cancellation without
    # changing the exact hit test inside the ROI.
    row_start = max(0, int(rows[0]) - 1)
    row_stop = min(height, int(rows[-1]) + 2)

    if horizontal_distance <= radius64:
        return slice(row_start, row_stop), (slice(0, width),)

    azimuth_radius = np.arcsin(np.clip(radius64 / horizontal_distance, 0.0, 1.0))
    center_azimuth = np.arctan2(center64[1], center64[0])
    delta = np.arctan2(np.sin(column_azimuths - center_azimuth),
                       np.cos(column_azimuths - center_azimuth))
    column_keep = np.abs(delta) <= azimuth_radius
    column_keep |= np.roll(column_keep, 1) | np.roll(column_keep, -1)
    columns = np.flatnonzero(column_keep)
    if columns.size == 0:
        return None
    breaks = np.flatnonzero(np.diff(columns) > 1) + 1
    groups = np.split(columns, breaks)
    column_slices = tuple(slice(int(group[0]), int(group[-1]) + 1)
                          for group in groups)
    return slice(row_start, row_stop), column_slices


def _validate_raster_inputs(base_range, ray_directions, visibility_limit):
    base_range = np.asarray(base_range, dtype=np.float32)
    rays = np.asarray(ray_directions, dtype=np.float32)
    if rays.ndim != 3 or rays.shape[2] != 3 or base_range.shape != rays.shape[:2]:
        raise ValueError('base_range and ray_directions shapes are incompatible')
    if visibility_limit is not None and np.asarray(visibility_limit).shape != base_range.shape:
        raise ValueError('visibility_limit must match base_range')
    return base_range, rays


def _valid_track_geometry(track, origin, rotation, radius_min, radius_max):
    center_world = np.asarray(track['centroid'], dtype=np.float32)
    velocity_world = np.asarray(track['v'], dtype=np.float32)
    dimensions = np.asarray(track['size'], dtype=np.float32)
    if (center_world.shape != (3,) or velocity_world.shape != (3,) or
            dimensions.shape != (3,) or not np.isfinite(center_world).all() or
            not np.isfinite(velocity_world).all() or not np.isfinite(dimensions).all() or
            (dimensions <= 0.0).any()):
        return None
    center = (center_world - origin) @ rotation
    radius = np.clip(0.5 * dimensions.max(), radius_min, radius_max)
    velocity = velocity_world @ rotation
    uid = int(track.get('track_uid', track.get('id', 0)))
    return center, radius, velocity, uid


def _composite_dynamic_result(base_range, best_depth, best_velocity, owner,
                              max_distance, occlusion_tolerance,
                              visibility_limit, return_owner):
    visibility_base = base_range if visibility_limit is None else np.minimum(
        base_range, np.asarray(visibility_limit, dtype=np.float32))
    visible = ((best_depth < max_distance) &
               (best_depth <= visibility_base + occlusion_tolerance))
    velocity = np.zeros_like(best_velocity)
    velocity[visible] = best_velocity[visible]
    range_with_dynamic = base_range.copy()
    range_with_dynamic[visible] = np.minimum(
        range_with_dynamic[visible], best_depth[visible])
    if return_owner:
        owner[~visible] = -1
        return range_with_dynamic, visible.astype(np.float32), velocity, owner
    return range_with_dynamic, visible.astype(np.float32), velocity


def rasterize_dynamic_spheres_full_grid(
        tracks, t_wb, rotation_world_from_planning, base_range, ray_directions,
        min_distance, max_distance, radius_min=0.10, radius_max=0.20,
        occlusion_tolerance=0.15, visibility_limit=None, return_owner=False):
    """Reference implementation that intersects every track with every ray."""
    base_range, rays = _validate_raster_inputs(
        base_range, ray_directions, visibility_limit)

    best_depth = np.full(base_range.shape, max_distance, dtype=np.float32)
    best_velocity = np.zeros((*base_range.shape, 3), dtype=np.float32)
    owner = np.full(base_range.shape, -1, dtype=np.int64)
    origin = np.asarray(t_wb, dtype=np.float32)
    rotation = np.asarray(rotation_world_from_planning, dtype=np.float32)

    for track in sorted(tracks, key=lambda item: int(item.get('track_uid', item.get('id', 0)))):
        geometry = _valid_track_geometry(
            track, origin, rotation, radius_min, radius_max)
        if geometry is None:
            continue
        center, radius, velocity, uid = geometry
        oc = -center
        b = rays @ oc
        c = float(oc @ oc - radius * radius)
        discriminant = b * b - c
        hit = discriminant >= 0.0
        near = np.full(base_range.shape, np.inf, dtype=np.float32)
        far = np.full(base_range.shape, np.inf, dtype=np.float32)
        root = np.sqrt(np.maximum(discriminant, 0.0))
        near[hit] = -b[hit] - root[hit]
        far[hit] = -b[hit] + root[hit]
        depth = np.where(near > min_distance, near,
                         np.where(far > min_distance, far, np.inf))
        update = depth < best_depth
        if update.any():
            best_depth[update] = depth[update]
            best_velocity[update] = velocity
            owner[update] = uid

    return _composite_dynamic_result(
        base_range, best_depth, best_velocity, owner, max_distance,
        occlusion_tolerance, visibility_limit, return_owner)


def rasterize_dynamic_spheres(tracks, t_wb, rotation_world_from_planning, base_range,
                              ray_directions, min_distance, max_distance,
                              radius_min=0.10, radius_max=0.20,
                              occlusion_tolerance=0.15, visibility_limit=None,
                              return_owner=False):
    """Composite confirmed tracker spheres using conservative angular ROIs.

    Tracks contain world-frame ``centroid``, ``v``, ``size`` and stable ``track_uid``.
    The returned velocity is in the planning frame and is zero outside the visible mask.
    """
    base_range, rays = _validate_raster_inputs(
        base_range, ray_directions, visibility_limit)
    best_depth = np.full(base_range.shape, max_distance, dtype=np.float32)
    best_velocity = np.zeros((*base_range.shape, 3), dtype=np.float32)
    owner = (np.full(base_range.shape, -1, dtype=np.int64)
             if return_owner else None)
    origin = np.asarray(t_wb, dtype=np.float32)
    rotation = np.asarray(rotation_world_from_planning, dtype=np.float32)
    row_elevations = np.arctan2(
        rays[:, 0, 2], np.hypot(rays[:, 0, 0], rays[:, 0, 1])).astype(np.float64)
    column_azimuths = np.arctan2(rays[0, :, 1], rays[0, :, 0]).astype(np.float64)
    full_grid_size = base_range.size

    for track in sorted(tracks, key=lambda item: int(item.get('track_uid', item.get('id', 0)))):
        geometry = _valid_track_geometry(
            track, origin, rotation, radius_min, radius_max)
        if geometry is None:
            continue
        center, radius, velocity, uid = geometry
        roi = _sphere_roi(center, radius, row_elevations, column_azimuths)
        if roi is None:
            continue
        row_slice, column_slices = roi
        roi_size = ((row_slice.stop - row_slice.start) *
                    sum(item.stop - item.start for item in column_slices))
        if roi_size >= 0.6 * full_grid_size:
            row_slice = slice(0, base_range.shape[0])
            column_slices = (slice(0, base_range.shape[1]),)

        for column_slice in column_slices:
            index = (row_slice, column_slice)
            depth = _sphere_intersection_depth(
                rays[index], center, radius, min_distance)
            best_depth_roi = best_depth[index]
            update = depth < best_depth_roi
            if not update.any():
                continue
            best_depth_roi[update] = depth[update]
            best_velocity_roi = best_velocity[index]
            best_velocity_roi[update] = velocity
            if owner is not None:
                owner_roi = owner[index]
                owner_roi[update] = uid

    return _composite_dynamic_result(
        base_range, best_depth, best_velocity, owner, max_distance,
        occlusion_tolerance, visibility_limit, return_owner)

