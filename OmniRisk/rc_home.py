import math


def update_stick_active(axis_x, axis_y, was_active,
                        neutral_deadzone, reactivate_deadzone):
    """Apply radial Schmitt-trigger hysteresis to the normalized stick."""
    if not 0.0 <= neutral_deadzone < reactivate_deadzone <= 1.0:
        raise ValueError(
            "require 0 <= neutral_deadzone < reactivate_deadzone <= 1")
    magnitude = math.hypot(axis_x, axis_y)
    if was_active:
        return magnitude > neutral_deadzone
    return magnitude >= reactivate_deadzone


def advance_height_reference(current, target, max_speed, dt):
    """Move a height reference toward target without exceeding max_speed."""
    if max_speed <= 0.0 or dt <= 0.0:
        raise ValueError("max_speed and dt must be positive")
    error = target - current
    max_step = max_speed * dt
    if abs(error) <= max_step:
        return target, 0.0
    velocity = max_speed if error > 0.0 else -max_speed
    return current + velocity * dt, velocity
