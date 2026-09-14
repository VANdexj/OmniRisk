from dataclasses import dataclass, replace
from enum import Enum


class RuntimeMode(str, Enum):
    HOVER = "HOVER"
    CRUISE = "CRUISE"
    DODGE = "DODGE"


@dataclass(frozen=True)
class RuntimeConfig:
    mode: RuntimeMode
    velocity: float
    radio_range: float
    fusion_lambda: float
    vel_max: float
    acc_max: float
    trajectory_time: float
    goal_length: float


@dataclass(frozen=True)
class RuntimeSnapshot:
    mode: RuntimeMode
    version: int
    config: object


def build_runtime_config(mode, values, vel_max_train, acc_max_train):
    velocity = float(values["velocity"])
    radio_range = float(values["radio_range"])
    fusion_lambda = float(values["fusion_lambda"])
    vel_max_train = float(vel_max_train)
    acc_max_train = float(acc_max_train)
    if velocity <= 0.0 or radio_range <= 0.0:
        raise ValueError("runtime velocity and radio_range must be positive")
    if fusion_lambda < 0.0 or vel_max_train <= 0.0 or acc_max_train <= 0.0:
        raise ValueError(
            "fusion_lambda must be non-negative and training limits must be positive")
    ratio = velocity / vel_max_train
    return RuntimeConfig(
        mode=RuntimeMode(mode),
        velocity=velocity,
        radio_range=radio_range,
        fusion_lambda=fusion_lambda,
        vel_max=velocity,
        acc_max=ratio * ratio * acc_max_train,
        trajectory_time=2.0 * radio_range / velocity,
        goal_length=2.0 * radio_range,
    )


def build_runtime_profiles(values, vel_max_train, acc_max_train):
    cruise = build_runtime_config(
        RuntimeMode.CRUISE, values["cruise"], vel_max_train, acc_max_train)
    return {
        RuntimeMode.HOVER: replace(cruise, mode=RuntimeMode.HOVER),
        RuntimeMode.CRUISE: cruise,
        RuntimeMode.DODGE: build_runtime_config(
            RuntimeMode.DODGE, values["dodge"], vel_max_train, acc_max_train),
    }


def select_runtime_mode(threat_active, navigation_active):
    if threat_active:
        return RuntimeMode.DODGE
    if navigation_active:
        return RuntimeMode.CRUISE
    return RuntimeMode.HOVER


def mode_allows_trajectory_control(mode):
    return RuntimeMode(mode) != RuntimeMode.HOVER


class RuntimeModeManager:
    def __init__(self, profiles):
        self._profiles = dict(profiles)
        self._mode = RuntimeMode.HOVER
        self._version = 0

    @property
    def mode(self):
        return self._mode

    @property
    def version(self):
        return self._version

    def select(self, threat_active, navigation_active):
        selected = select_runtime_mode(threat_active, navigation_active)
        if selected != self._mode:
            self._mode = selected
            self._version += 1
        return RuntimeSnapshot(selected, self._version, self._profiles[selected])

    def is_current(self, snapshot):
        return (snapshot.mode == self._mode and
                int(snapshot.version) == self._version)
