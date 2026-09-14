from dataclasses import dataclass
import math

from runtime_modes import RuntimeMode, mode_allows_trajectory_control


@dataclass(frozen=True)
class HoverTarget:
    position: tuple
    yaw: float


def latch_hover_target(current, position, yaw):
    if current is not None:
        return current
    values = tuple(float(value) for value in position)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("hover position must contain three finite values")
    yaw = float(yaw)
    if not math.isfinite(yaw):
        raise ValueError("hover yaw must be finite")
    return HoverTarget(values, yaw)


def next_trajectory_sample(ctrl_time, ctrl_dt, repeat_first_sample):
    sample_time = ctrl_time + ctrl_dt
    next_ctrl_time = ctrl_time if repeat_first_sample else sample_time
    return sample_time, next_ctrl_time


def trajectory_is_executable(mode, mode_version, trajectory_version,
                             evaluation_time, trajectory_duration,
                             last_commit_mono, now_mono, refresh_timeout):
    if not mode_allows_trajectory_control(RuntimeMode(mode)):
        return False
    values = (evaluation_time, trajectory_duration, last_commit_mono,
              now_mono, refresh_timeout)
    if trajectory_version is None or any(value is None for value in values):
        return False
    values = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in values):
        return False
    evaluation_time, trajectory_duration, last_commit_mono, now_mono, refresh_timeout = values
    refresh_age = now_mono - last_commit_mono
    return (int(trajectory_version) == int(mode_version)
            and refresh_timeout > 0.0
            and 0.0 <= evaluation_time <= trajectory_duration
            and 0.0 <= refresh_age <= refresh_timeout)
