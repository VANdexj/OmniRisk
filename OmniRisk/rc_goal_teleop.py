#!/usr/bin/env python3
"""Publish a RadioMaster right-stick direction as a nearby world-frame planner goal.

Supported inputs:
  usb    -> Linux joystick device (RadioMaster SIM)
  mavros -> wireless receiver connected to PX4, exposed as /mavros/rc/in

AETR mapping:
  axis 0 (right-stick horizontal) -> world Y
  axis 1 (right-stick vertical)   -> world X
The horizontal axis is inverted by default for both inputs; the vertical axis
is not. The goal altitude remains fixed at the planner's hover height; vehicle yaw is
intentionally ignored.
"""
import math
import os
import struct
import threading
import time

import rospy
from geometry_msgs.msg import Point, PoseStamped
from mavros_msgs.msg import RCIn
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker

from rc_home import update_stick_active


_JS_EVENT = struct.Struct("IhBB")
_JS_AXIS = 0x02


def normalize_axis(value, axis_max, invert):
    value = max(-1.0, min(1.0, value / axis_max))
    return -value if invert else value


def normalize_rc_axis(value, rc_min, rc_center, rc_max, invert):
    span = rc_max - rc_center if value >= rc_center else rc_center - rc_min
    value = max(-1.0, min(1.0, (value - rc_center) / span))
    return -value if invert else value


def joystick_offset(axis_x, axis_y, radius, deadzone, swap_xy):
    x, y = (axis_y, axis_x) if swap_xy else (axis_x, axis_y)
    magnitude = math.hypot(x, y)
    if magnitude <= deadzone:
        return 0.0, 0.0
    normalized_magnitude = min(1.0, (magnitude - deadzone) /
                               (1.0 - deadzone))
    scale = radius * normalized_magnitude / magnitude
    return x * scale, y * scale


def filter_goal_offset(raw_offset, low_pass_offset, output_offset,
                       dt, tau, slew_rate):
    if low_pass_offset is None or output_offset is None:
        low_pass_offset = (0.0, 0.0)
        output_offset = (0.0, 0.0)

    dt = max(0.0, dt)
    alpha = 1.0 if tau == 0.0 else dt / (tau + dt)
    low_pass_offset = (
        low_pass_offset[0] + alpha * (raw_offset[0] - low_pass_offset[0]),
        low_pass_offset[1] + alpha * (raw_offset[1] - low_pass_offset[1]),
    )

    dx = low_pass_offset[0] - output_offset[0]
    dy = low_pass_offset[1] - output_offset[1]
    distance = math.hypot(dx, dy)
    max_step = slew_rate * dt
    if distance > max_step and distance > 0.0:
        scale = max_step / distance
        output_offset = (output_offset[0] + dx * scale,
                         output_offset[1] + dy * scale)
    else:
        output_offset = low_pass_offset
    return low_pass_offset, output_offset


class RCGoalTeleop:
    def __init__(self):
        rospy.init_node("rc_goal_teleop")
        param = rospy.get_param
        self.input_mode = str(param("~input_mode", "usb")).lower()
        self.js_device = param("~js_device", "/dev/input/js0")
        self.axis_x = int(param("~axis_x", 0))
        self.axis_y = int(param("~axis_y", 1))
        self.axis_max = float(param("~axis_max", 32767.0))
        self.invert_x = bool(param("~invert_x", True))
        self.invert_y = bool(param("~invert_y", False))
        self.rc_topic = param("~rc_topic", "/mavros/rc/in")
        self.rc_min = float(param("~rc_min", 1000.0))
        self.rc_center = float(param("~rc_center", 1500.0))
        self.rc_max = float(param("~rc_max", 2000.0))
        self.rc_timeout = float(param("~rc_timeout", 0.5))
        self.swap_xy = bool(param("~swap_xy", True))
        self.deadzone = float(param("~deadzone", 0.08))
        self.reactivate_deadzone = float(
            param("~reactivate_deadzone", 0.12))
        self.radius = float(param("~radius", 10.0))
        self.filter_tau = float(param("~filter_tau", 0.10))
        self.goal_slew_rate = float(param("~goal_slew_rate", 20.0))
        self.goal_publish_epsilon = float(
            param("~goal_publish_epsilon", 0.30))
        self.rearm_center_frames = int(param("~rearm_center_frames", 3))
        self.fixed_height = float(param("~fixed_height", 0.6))
        self.odom_topic = param("~odom_topic", "/ekf_quat/ekf_odom")
        self.goal_topic = param("~goal_topic", "/move_base_simple/goal")
        self.active_topic = param("~active_topic", "/rc_goal_active")
        self.marker_topic = param("~marker_topic", "/rc_goal_marker")
        self.world_frame = param("~world_frame", "world")
        self.rate_hz = float(param("~rate", 20.0))
        self.calibrate = bool(param("~calibrate", False))

        if self.input_mode not in ("usb", "mavros"):
            raise ValueError("~input_mode must be 'usb' or 'mavros'")
        if self.axis_x < 0 or self.axis_y < 0:
            raise ValueError("~axis_x and ~axis_y must be non-negative")
        if self.axis_x == self.axis_y:
            raise ValueError("~axis_x and ~axis_y must be different")
        if self.axis_max <= 0.0:
            raise ValueError("~axis_max must be positive")
        if not self.rc_min < self.rc_center < self.rc_max:
            raise ValueError("require ~rc_min < ~rc_center < ~rc_max")
        if self.rc_timeout <= 0.0:
            raise ValueError("~rc_timeout must be positive")
        if not 0.0 <= self.deadzone < 1.0:
            raise ValueError("~deadzone must be in [0, 1)")
        if not self.deadzone < self.reactivate_deadzone <= 1.0:
            raise ValueError(
                "require ~deadzone < ~reactivate_deadzone <= 1")
        if self.radius <= 0.0:
            raise ValueError("~radius must be positive")
        if self.filter_tau < 0.0:
            raise ValueError("~filter_tau must be non-negative")
        if self.goal_slew_rate <= 0.0:
            raise ValueError("~goal_slew_rate must be positive")
        if self.goal_publish_epsilon < 0.0:
            raise ValueError("~goal_publish_epsilon must be non-negative")
        if self.rearm_center_frames <= 0:
            raise ValueError("~rearm_center_frames must be positive")
        if self.rate_hz <= 0.0:
            raise ValueError("~rate must be positive")

        self._lock = threading.Lock()
        self._position = None
        self._axes = {}
        self._last_rc_monotonic = None
        self._input_valid = self.input_mode == "usb"
        self._input_sequence = 0
        self._rearm_required = False
        self._rearm_center_count = 0
        self._last_rearm_sequence = None
        self._stick_active = False
        self._low_pass_offset = None
        self._filtered_offset = None
        self._filter_monotonic = None
        self._last_published_offset = None

        rospy.Subscriber(
            self.odom_topic, Odometry, self._odom_cb,
            queue_size=1, tcp_nodelay=True)
        if self.input_mode == "usb":
            try:
                self._fd = os.open(self.js_device, os.O_RDONLY)
            except OSError as error:
                rospy.logfatal("Cannot open %s: %s", self.js_device, error)
                raise
            threading.Thread(target=self._reader, daemon=True).start()
        else:
            rospy.Subscriber(
                self.rc_topic, RCIn, self._rc_cb,
                queue_size=1, tcp_nodelay=True)
        self._goal_pub = rospy.Publisher(
            self.goal_topic, PoseStamped, queue_size=1)
        self._active_pub = rospy.Publisher(
            self.active_topic, Bool, queue_size=1)
        self._marker_pub = rospy.Publisher(
            self.marker_topic, Marker, queue_size=1)

        rospy.loginfo(
            "rc_goal_teleop: mode=%s source=%s axes=(%d,%d) invert=(%s,%s) "
            "swap_xy=%s radius=%.1fm odom=%s goal=%s active=%s%s",
            self.input_mode,
            self.js_device if self.input_mode == "usb" else self.rc_topic,
            self.axis_x, self.axis_y,
            self.invert_x, self.invert_y, self.swap_xy, self.radius,
            self.odom_topic, self.goal_topic, self.active_topic,
            " [CALIBRATE]" if self.calibrate else "")

    def _odom_cb(self, msg):
        position = msg.pose.pose.position
        with self._lock:
            self._position = (position.x, position.y)

    def _clear_axes(self):
        with self._lock:
            self._axes.clear()
            self._last_rc_monotonic = None
            self._input_valid = False
            self._rearm_required = True
            self._rearm_center_count = 0
            self._last_rearm_sequence = None

    def _rc_cb(self, msg):
        required_channels = max(self.axis_x, self.axis_y) + 1
        if len(msg.channels) < required_channels:
            self._clear_axes()
            rospy.logwarn_throttle(
                1.0, "RCIn has %d channels; need at least %d",
                len(msg.channels), required_channels)
            return

        raw_x = msg.channels[self.axis_x]
        raw_y = msg.channels[self.axis_y]
        if not 0 < raw_x < 65535 or not 0 < raw_y < 65535:
            self._clear_axes()
            rospy.logwarn_throttle(
                1.0, "RCIn selected channels are invalid: (%s, %s)",
                raw_x, raw_y)
            return

        with self._lock:
            self._axes[self.axis_x] = raw_x
            self._axes[self.axis_y] = raw_y
            self._last_rc_monotonic = time.monotonic()
            self._input_valid = True
            self._input_sequence += 1

        if self.calibrate:
            rospy.loginfo_throttle(
                0.15, "RC channels[%d]=%d channels[%d]=%d",
                self.axis_x, raw_x, self.axis_y, raw_y)

    def _reader(self):
        buffer = b""
        while not rospy.is_shutdown():
            try:
                data = os.read(self._fd, 64)
            except OSError as error:
                buffer = b""
                self._clear_axes()
                rospy.logerr_throttle(
                    1.0, "Cannot read %s; joystick neutralized: %s",
                    self.js_device, error)
                rospy.sleep(0.2)
                continue
            if not data:
                buffer = b""
                self._clear_axes()
                rospy.logerr_throttle(
                    1.0, "No data from %s; joystick neutralized",
                    self.js_device)
                rospy.sleep(0.2)
                continue

            buffer += data
            while len(buffer) >= _JS_EVENT.size:
                _time, value, event_type, number = _JS_EVENT.unpack(
                    buffer[:_JS_EVENT.size])
                buffer = buffer[_JS_EVENT.size:]
                if event_type & _JS_AXIS:
                    with self._lock:
                        self._axes[number] = value
                        if (self.axis_x in self._axes and
                                self.axis_y in self._axes):
                            self._input_valid = True
                            self._input_sequence += 1

            if self.calibrate:
                with self._lock:
                    axes = dict(self._axes)
                rospy.loginfo_throttle(
                    0.15, "axes=%s" % " ".join(
                        "%d:%d" % (axis, axes[axis]) for axis in sorted(axes)))

    def _normalized_axes(self):
        with self._lock:
            raw_x = self._axes.get(self.axis_x, 0)
            raw_y = self._axes.get(self.axis_y, 0)
            last_rc_monotonic = self._last_rc_monotonic
            input_valid = self._input_valid
            input_sequence = self._input_sequence

        if self.input_mode == "usb":
            return (normalize_axis(raw_x, self.axis_max, self.invert_x),
                    normalize_axis(raw_y, self.axis_max, self.invert_y),
                    input_valid, input_sequence)

        rc_fresh = (last_rc_monotonic is not None and
                    time.monotonic() - last_rc_monotonic <= self.rc_timeout)
        if not rc_fresh:
            rospy.logwarn_throttle(
                1.0, "No fresh data from %s; RC neutralized", self.rc_topic)
            return 0.0, 0.0, False, input_sequence
        return (normalize_rc_axis(
                    raw_x, self.rc_min, self.rc_center,
                    self.rc_max, self.invert_x),
                normalize_rc_axis(
                    raw_y, self.rc_min, self.rc_center,
                    self.rc_max, self.invert_y),
                input_valid, input_sequence)

    def _activation_is_armed(self, axis_x, axis_y,
                             input_valid, input_sequence):
        with self._lock:
            if not input_valid:
                self._rearm_required = True
                self._rearm_center_count = 0
                self._last_rearm_sequence = None
                return False
            if not self._rearm_required:
                return True

            centered = math.hypot(axis_x, axis_y) <= self.deadzone
            if not centered:
                self._rearm_center_count = 0
                self._last_rearm_sequence = None
                return False

            is_new_frame = (self.input_mode == "usb" or
                            input_sequence != self._last_rearm_sequence)
            if is_new_frame:
                self._rearm_center_count += 1
                self._last_rearm_sequence = input_sequence
            if self._rearm_center_count >= self.rearm_center_frames:
                self._rearm_required = False
                self._rearm_center_count = 0
                self._last_rearm_sequence = None
            return False

    def _reset_goal_filter(self):
        self._low_pass_offset = None
        self._filtered_offset = None
        self._filter_monotonic = None
        self._last_published_offset = None

    def _update_goal_filter(self, raw_offset, now):
        dt = (1.0 / self.rate_hz if self._filter_monotonic is None else
              now - self._filter_monotonic)
        self._low_pass_offset, self._filtered_offset = filter_goal_offset(
            raw_offset, self._low_pass_offset, self._filtered_offset,
            dt, self.filter_tau, self.goal_slew_rate)
        self._filter_monotonic = now
        return self._filtered_offset

    def _goal_changed(self, offset):
        if self._last_published_offset is None:
            self._last_published_offset = offset
            return True
        distance = math.hypot(
            offset[0] - self._last_published_offset[0],
            offset[1] - self._last_published_offset[1])
        if distance < self.goal_publish_epsilon:
            return False
        self._last_published_offset = offset
        return True

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            with self._lock:
                position = self._position

            if not self.calibrate:
                axis_x, axis_y, input_valid, input_sequence = (
                    self._normalized_axes())
                if self._activation_is_armed(
                        axis_x, axis_y, input_valid, input_sequence):
                    self._stick_active = update_stick_active(
                        axis_x, axis_y, self._stick_active,
                        self.deadzone, self.reactivate_deadzone)
                else:
                    self._stick_active = False
                if not self._stick_active:
                    self._reset_goal_filter()
                active = Bool()
                active.data = self._stick_active
                self._active_pub.publish(active)

                if position is None:
                    rate.sleep()
                    continue

                stamp = rospy.Time.now()
                goal_x, goal_y = position
                if self._stick_active:
                    raw_offset = joystick_offset(
                        axis_x, axis_y, self.radius,
                        self.deadzone, self.swap_xy)
                    offset_x, offset_y = self._update_goal_filter(
                        raw_offset, time.monotonic())
                    goal_x += offset_x
                    goal_y += offset_y
                    if self._goal_changed((offset_x, offset_y)):
                        self._publish_goal(goal_x, goal_y, stamp)
                self._publish_marker(position, (goal_x, goal_y), stamp)
            rate.sleep()

    def _publish_goal(self, x, y, stamp):
        goal = PoseStamped()
        goal.header.stamp = stamp
        goal.header.frame_id = self.world_frame
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = self.fixed_height
        goal.pose.orientation.w = 1.0
        self._goal_pub.publish(goal)

    def _publish_marker(self, start, goal, stamp):
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        distance = math.hypot(dx, dy)

        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.world_frame
        marker.ns = "rc_goal"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.pose.orientation.w = 1.0

        if distance <= 1e-6:
            marker.action = Marker.DELETE
            self._marker_pub.publish(marker)
            return

        marker.action = Marker.ADD
        arrow_length = 1.0
        tail = (goal[0] - arrow_length * dx / distance,
                goal[1] - arrow_length * dy / distance)
        marker_height = self.fixed_height + 0.12
        marker.points = [
            Point(x=tail[0], y=tail[1], z=marker_height),
            Point(x=goal[0], y=goal[1], z=marker_height),
        ]
        marker.scale.x = 0.10
        marker.scale.y = 0.30
        marker.scale.z = 0.38
        marker.color.r = 0.0
        marker.color.g = 0.75
        marker.color.b = 1.0
        marker.color.a = 0.8
        self._marker_pub.publish(marker)


if __name__ == "__main__":
    try:
        RCGoalTeleop().spin()
    except rospy.ROSInterruptException:
        pass
