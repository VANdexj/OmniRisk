"""Small, ROS-independent helpers for latest-only perception synchronization."""

from collections import deque
from dataclasses import dataclass
from threading import Condition
import time

import numpy as np


@dataclass(frozen=True)
class StampedValue:
    stamp_ns: int
    value: object
    arrival_mono: float


class StampedCache:
    """Short stamp-ordered cache. It stores data, not pending work."""

    def __init__(self, duration_sec):
        self.duration_ns = int(float(duration_sec) * 1e9)
        self._items = deque()

    def clear(self):
        self._items.clear()

    def append(self, stamp_ns, value, arrival_mono=None):
        stamp_ns = int(stamp_ns)
        arrival_mono = time.monotonic() if arrival_mono is None else float(arrival_mono)
        moved_backward = bool(self._items and stamp_ns < self._items[-1].stamp_ns)
        if moved_backward:
            self.clear()
        entry = StampedValue(stamp_ns, value, arrival_mono)
        if self._items and stamp_ns == self._items[-1].stamp_ns:
            self._items[-1] = entry
        else:
            self._items.append(entry)
        cutoff = stamp_ns - self.duration_ns
        while self._items and self._items[0].stamp_ns < cutoff:
            self._items.popleft()
        return moved_backward

    def exact(self, stamp_ns):
        stamp_ns = int(stamp_ns)
        for item in reversed(self._items):
            if item.stamp_ns == stamp_ns:
                return item
            if item.stamp_ns < stamp_ns:
                break
        return None

    def latest_before(self, stamp_ns, max_age_ns=None, inclusive=True):
        stamp_ns = int(stamp_ns)
        for item in reversed(self._items):
            if item.stamp_ns < stamp_ns or (inclusive and item.stamp_ns == stamp_ns):
                if max_age_ns is None or stamp_ns - item.stamp_ns <= int(max_age_ns):
                    return item
                return None
        return None

    def bracket(self, stamp_ns):
        stamp_ns = int(stamp_ns)
        before = None
        for item in self._items:
            if item.stamp_ns <= stamp_ns:
                before = item
            if item.stamp_ns >= stamp_ns:
                return before, item
        return before, None

    def neighbors(self, stamp_ns):
        """Return the strict previous, exact and strict next stamped entries."""
        stamp_ns = int(stamp_ns)
        previous = exact = following = None
        for item in self._items:
            if item.stamp_ns < stamp_ns:
                previous = item
            elif item.stamp_ns == stamp_ns:
                exact = item
            else:
                following = item
                break
        return previous, exact, following

    def newest_first(self):
        return reversed(self._items)

    def __len__(self):
        return len(self._items)


@dataclass(frozen=True)
class SlotValue:
    version: int
    payload: object


class LatestOnlySlot:
    """Capacity-one overwrite slot with race-free condition notification."""

    def __init__(self):
        self._condition = Condition()
        self._payload = None
        self._version = 0

    def publish(self, payload):
        """Publish payload and return whether an unstarted payload was replaced."""
        with self._condition:
            overwritten = self._payload is not None
            self._version += 1
            self._payload = payload
            self._condition.notify()
            return overwritten

    def take_after(self, last_started_version, timeout=None):
        """Take the newest payload newer than ``last_started_version``."""
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            while (self._payload is None or
                   self._version <= int(last_started_version)):
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(remaining)
            item = SlotValue(self._version, self._payload)
            self._payload = None
            return item

    def clear(self):
        """Discard pending work and wake consumers so they can observe shutdown/reset."""
        with self._condition:
            self._payload = None
            self._version += 1
            self._condition.notify_all()

    def has_pending(self):
        with self._condition:
            return self._payload is not None


def elapsed_since_arrival_ms(arrival_mono, now_mono=None):
    """Return local processing latency without relying on ROS timestamp epochs."""
    now_mono = time.monotonic() if now_mono is None else float(now_mono)
    return max(0.0, now_mono - float(arrival_mono)) * 1000.0


def source_age_within_limit(source_arrival_mono, max_age_ms, now_mono=None):
    """Return whether work can still pass the end-to-end age limit."""
    return elapsed_since_arrival_ms(
        source_arrival_mono, now_mono) <= float(max_age_ms) + 1e-9


@dataclass(frozen=True)
class SyncSelection:
    stamp_ns: int
    lidar: StampedValue
    required: dict
    static: StampedValue
    static_degraded: bool


@dataclass(frozen=True)
class PerceptionFrame:
    sequence: int
    stamp_ns: int
    lidar: StampedValue
    tracks_at_ref: object
    static: StampedValue
    odom_state: object
    odom_delta_ms: float
    static_delta_ms: float
    static_degraded: bool
    odom_degraded: bool
    generation: int
    runtime_snapshot: object
    threat_detected: bool
    lidar_arrival_mono: float


@dataclass(frozen=True)
class PreparedInference:
    sequence: int
    stamp_ns: int
    stamp: object
    generation: int
    runtime_snapshot: object
    depth_input: object
    mask_input: object
    velocity_input: object
    obs_input: object
    odom_state_at_ref: object
    rotation_wc: object
    t_wb: object
    threat_detected: bool
    range_img: object
    current_range: object
    range_img_full: object
    depth_norm: object
    current_valid: object
    final_valid: object
    static_delta_ms: float
    odom_delta_ms: float
    lidar_arrival_mono: float
    preprocess_started_mono: float
    preprocess_completed_mono: float


def select_latest_complete(lidar_cache, required_caches, static_cache,
                           last_assembled_ns, static_tolerance_ns,
                           fallback_deadline_sec=None,
                           allow_static_fallback=True):
    """Select the newest complete frame without consuming older entries as jobs."""
    for lidar in lidar_cache.newest_first():
        stamp_ns = lidar.stamp_ns
        if last_assembled_ns is not None:
            if stamp_ns <= last_assembled_ns:
                break

        required = {}
        for name, cache in required_caches.items():
            item = cache.exact(stamp_ns)
            if item is None:
                break
            required[name] = item
        else:
            static = static_cache.exact(stamp_ns)
            if fallback_deadline_sec is not None:
                deadline = lidar.arrival_mono + fallback_deadline_sec
                if any(item.arrival_mono > deadline for item in required.values()):
                    continue
                if static is not None and static.arrival_mono > deadline:
                    static = None
            degraded = False
            if static is None and allow_static_fallback:
                static = static_cache.latest_before(
                    stamp_ns, static_tolerance_ns, inclusive=False)
                if (static is not None and fallback_deadline_sec is not None and
                        static.arrival_mono > deadline):
                    static = None
                degraded = static is not None
            if static is not None:
                return SyncSelection(stamp_ns, lidar, required, static, degraded)
    return None


def result_is_committable(result_generation, current_generation, sequence,
                          last_committed_sequence, source_arrival_mono,
                          max_age_ms, now_mono=None, shutting_down=False):
    """Pure commit gate shared by runtime code and scheduler unit tests."""
    if shutting_down or int(result_generation) != int(current_generation):
        return False
    if int(sequence) <= int(last_committed_sequence):
        return False
    return source_age_within_limit(source_arrival_mono, max_age_ms, now_mono)


def interpolate_quaternion(q0, q1, ratio):
    """Shortest-path normalized quaternion interpolation in [x, y, z, w] order."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    q = q0 + float(ratio) * (q1 - q0)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return q0.copy()
    return q / norm


def interpolate_kinematic_state(state0, state1, ratio):
    """Interpolate position, orientation and linear velocity at one reference time."""
    ratio = float(ratio)
    return {
        'position': (np.asarray(state0['position'])
                     + ratio * (np.asarray(state1['position']) - np.asarray(state0['position']))),
        'quaternion': interpolate_quaternion(
            state0['quaternion'], state1['quaternion'], ratio),
        'velocity': (np.asarray(state0['velocity'])
                     + ratio * (np.asarray(state1['velocity']) - np.asarray(state0['velocity']))),
    }


def predict_kinematics(position, velocity, dt, max_horizon, acceleration):
    """Bounded constant-acceleration prediction used to mirror tracker semantics."""
    dt = max(0.0, float(dt))
    if dt > float(max_horizon) + 1e-9:
        return None
    position = np.asarray(position, dtype=np.float32)
    velocity = np.asarray(velocity, dtype=np.float32)
    acceleration = np.asarray(acceleration, dtype=np.float32)
    return (position + velocity * dt + 0.5 * acceleration * dt * dt,
            velocity + acceleration * dt)
