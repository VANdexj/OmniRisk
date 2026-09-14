"""Validation and decoding for the M-detector static range contract."""

import numpy as np


def _message_stamp_ns(message):
    stamp = message.header.stamp
    return int(stamp.to_nsec() if hasattr(stamp, 'to_nsec') else stamp)


def decode_static_range_image(message, height, width, max_distance,
                              expected_frame_id, expected_stamp_ns=None):
    if message.header.frame_id != expected_frame_id:
        raise ValueError(
            f'static range frame_id {message.header.frame_id!r} does not match '
            f'{expected_frame_id!r}')
    if (expected_stamp_ns is not None and
            _message_stamp_ns(message) != int(expected_stamp_ns)):
        raise ValueError('static range must exactly match LiDAR stamp')
    if int(message.height) != int(height) or int(message.width) != int(width):
        raise ValueError('static range dimensions do not match network configuration')
    if str(message.encoding).upper() != '32FC1':
        raise ValueError('static range encoding must be 32FC1')

    row_bytes = int(width) * np.dtype(np.float32).itemsize
    step = int(message.step)
    if step < row_bytes:
        raise ValueError('static range row step is too small')
    payload = memoryview(message.data)
    required_bytes = step * int(height)
    if len(payload) < required_bytes:
        raise ValueError('static range payload is truncated')

    byte_rows = np.frombuffer(payload[:required_bytes], dtype=np.uint8).reshape(
        int(height), step)
    dtype = np.dtype('>f4' if bool(message.is_bigendian) else '<f4')
    image = byte_rows[:, :row_bytes].copy().view(dtype).reshape(
        int(height), int(width)).astype(np.float32, copy=False)
    if (not np.isfinite(image).all() or (image <= 0.0).any() or
            (image > float(max_distance)).any()):
        raise ValueError('static range contains invalid depth values')
    return np.ascontiguousarray(image, dtype=np.float32)


def decode_static_range_pose(message, expected_stamp_ns, expected_frame_id):
    if message.header.frame_id != expected_frame_id:
        raise ValueError('static range pose frame_id does not match range contract')
    if _message_stamp_ns(message) != int(expected_stamp_ns):
        raise ValueError('static range pose must exactly match LiDAR stamp')
    position = message.pose.pose.position
    orientation = message.pose.pose.orientation
    velocity = message.twist.twist.linear
    state = {
        'position': np.array(
            [position.x, position.y, position.z], dtype=np.float64),
        'quaternion': np.array(
            [orientation.x, orientation.y, orientation.z, orientation.w],
            dtype=np.float64),
        'velocity': np.array(
            [velocity.x, velocity.y, velocity.z], dtype=np.float64),
        'stamp': message.header.stamp,
    }
    values = np.concatenate((
        state['position'], state['quaternion'], state['velocity']))
    if not np.isfinite(values).all() or np.linalg.norm(
            state['quaternion']) < 1e-6:
        raise ValueError('static range pose contains invalid values')
    return state
