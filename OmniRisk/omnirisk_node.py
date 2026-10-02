import rospy
import std_msgs.msg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from threading import Condition, Lock, Thread
from collections import deque
from sensor_msgs.msg import PointCloud2, PointField, Image
from sensor_msgs import point_cloud2

import cv2
import os
import time
import torch
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

from config.config import cfg
from control_msg import PositionCommand
from policy.network import OmniRiskNetwork
from policy.poly_solver import *
from policy.state_transform import *
from perception_sync import (
    LatestOnlySlot,
    PerceptionFrame,
    PreparedInference,
    StampedCache,
    elapsed_since_arrival_ms,
    interpolate_kinematic_state,
    predict_kinematics,
    result_is_committable,
    select_latest_complete,
    source_age_within_limit,
)
from depth_completion import complete_depth
from inference_inputs import forward_with_contiguous_inputs
from static_range_input import decode_static_range_image, decode_static_range_pose
from dynamic_inputs import (
    make_ray_directions,
    map_panorama_to_network,
    project_points_to_range_image,
    rasterize_dynamic_spheres,
)
from trajectory_score_visualization import normalize_scores, score_colors
from trajectory_selection import (
    add_endpoint_continuity_cost,
    select_action_with_hysteresis,
)
from rc_home import advance_height_reference
from control_state import (
    latch_hover_target,
    next_trajectory_sample,
    trajectory_is_executable,
)
from runtime_modes import (
    RuntimeMode,
    RuntimeModeManager,
    build_runtime_profiles,
    mode_allows_trajectory_control,
)
from visualization_pipeline import pack_xyz32, runtime_state_geometry

# v6: optional TrackedObjectArray topic from M-detector integrated tracker.
try:
    from m_detector.msg import TrackedObjectArray
    HAVE_TRACKS_MSG = True
except Exception:
    HAVE_TRACKS_MSG = False

# Dodge demo: optional controller service for auto takeoff
try:
    from quadrotor_msgs.srv import SetTakeoffLand
    HAVE_TAKEOFF_SRV = True
except Exception:
    HAVE_TAKEOFF_SRV = False

try:
    from torch2trt import TRTModule
except ImportError:
    print("tensorrt not found.")


class OmniRiskNode:
    def __init__(self, config, weight):
        self.config = config
        rospy.init_node('omnirisk', anonymous=False)
        # load params
        cfg["train"] = False
        self.height = cfg['image_height']
        self.width = cfg['image_width']
        # Omnidirectional + world-azimuth frame. range/mask/vel use the full panorama (world azimuth, not rotated
        # with heading yaw), Rotation_wc=I (endstate already in world-azimuth frame), yaw is free (does not follow velocity).
        self.omnidirectional = bool(cfg['omnidirectional'])
        self.min_dis, self.max_dis = 0.3, 20.0
        self.goal = np.array(self.config['goal'])
        # Flight mode: dodge = in-place dodge demo (hover at home by default, hand over to the planner only for nearby dynamic threats) | nav = navigation with avoidance (cruise to goal)
        self.nav_mode         = (self.config.get('flight_mode', 'dodge') == 'nav')
        self.arrive_radius    = self.config.get('arrive_radius', 2.0 if self.nav_mode else 0.0)
        self.external_goal_received = False
        self.external_goal_arrive_radius = self.config.get('external_goal_arrive_radius', 2.0)
        self.home_set         = False
        self.home             = None
        self.threat_radius    = self.config.get('threat_radius', 3.0)    # a confirmed track within this distance (m) counts as a threat
        self.hold_after       = self.config.get('dodge_hold_after', 1.5) # keep dodging for this long (s) after the threat is gone, then hover
        self.last_threat_time = -1e9                                     # no threat by default → hover
        self.takeoff_alt_min  = self.config.get('takeoff_alt_min', 0.5)  # altitude threshold (m) for takeoff complete
        self.vz_settle        = self.config.get('vz_settle', 0.1)        # vz threshold (m/s) for stable hover
        self.auto_takeoff     = self.config.get('auto_takeoff', True)    # auto arm + takeoff after startup
        self.hover_height     = self.config.get('hover_height', 1.0)     # common flight height (m) for takeoff/hover/return after dodge
        self.takeoff_service  = self.config.get('takeoff_service', '/network_controller_node/takeoff_land')
        self.rc_active_topic  = self.config.get('rc_active_topic', '/rc_goal_active')
        self.rc_state_timeout = float(self.config.get('rc_state_timeout', 0.6))
        self.rc_return_speed  = float(self.config.get('rc_return_speed', 0.2))
        if self.rc_state_timeout <= 0.0 or self.rc_return_speed <= 0.0:
            raise ValueError('rc_state_timeout and rc_return_speed must be positive')
        self.plan_from_reference = self.config['plan_from_reference']
        self.use_trt = self.config['use_tensorrt']
        self.verbose = self.config['verbose']
        self.visualize = self.config['visualize']
        self.Rotation_bc = R.from_euler('ZYX', [0, self.config['pitch_angle_deg'], 0], degrees=True).as_matrix()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # camera intrinsics for depth point cloud visualization
        self.cam_fx = self.config.get('cam_fx', 161.0)
        self.cam_fy = self.config.get('cam_fy', 161.0)
        self.cam_cx = self.config.get('cam_cx', 80.0)
        self.cam_cy = self.config.get('cam_cy', 48.0)
        self.depth_vis_stride = self.config.get('depth_vis_stride', 4)  # subsample stride for performance

        # variables
        self.odom = Odometry()
        self.odom_init = False
        self.last_yaw = 0.0
        self.ctrl_dt = 0.02
        self.ctrl_time = None
        self.repeat_first_trajectory_sample = False
        self.desire_init = False
        self.arrive = False
        self.desire_pos = None
        self.desire_vel = None
        self.desire_acc = None
        self.optimal_poly_x = None
        self.optimal_poly_y = None
        self.optimal_poly_z = None
        self.lock = Lock()
        self.odom_lock = Lock()
        self.rc_lock = Lock()
        self.last_control_msg = None
        self.rc_state_received = False
        self.rc_active = False
        self.rc_resume_pending = False
        self.rc_state_last_monotonic = None
        self.rc_home = None
        self.rc_home_z_ref = None
        self.state_transform = StateTransform()
        self.lattice_primitive = LatticePrimitive.get_instance()
        self.runtime_profiles = build_runtime_profiles(
            cfg['runtime_modes'], cfg['vel_max_train'], cfg['acc_max_train'])
        self.runtime_modes = RuntimeModeManager(self.runtime_profiles)
        self.active_trajectory_time = None
        self.active_trajectory_version = None
        self.active_trajectory_commit_mono = None
        self.trajectory_refresh_timeout = float(
            cfg['trajectory_refresh_timeout'])
        if (not np.isfinite(self.trajectory_refresh_timeout) or
                self.trajectory_refresh_timeout <= 0.0):
            raise ValueError('trajectory_refresh_timeout must be positive and finite')
        self.fallback_hover_target = None
        self.control_mode = RuntimeMode.HOVER

        self.trajectory_switch_margin = float(cfg['trajectory_switch_margin'])
        self.trajectory_continuity_weight = float(
            cfg['trajectory_continuity_weight'])
        self.last_action_id = None

        # eval
        self.time_forward = 0.0
        self.time_process = 0.0
        self.time_prepare = 0.0
        self.time_interpolation = 0.0
        self.time_visualize = 0.0
        self.count = 0
        self.depth_fps = 10 if self.config.get('use_lidar', False) else 30

        # Load Network
        if self.use_trt:
            self.policy = TRTModule()
            self.policy.load_state_dict(torch.load(weight))
        else:
            state_dict = torch.load(weight, weights_only=True)
            self.policy = OmniRiskNetwork()
            self.policy.load_state_dict(state_dict)
            self.policy = self.policy.to(self.device)
            self.policy.eval()

        # lidar params and state (must be set before warm_up uses them)
        self.lidar_H            = self.config.get('lidar_H', 96)
        self.lidar_W            = self.config.get('lidar_W', 360)
        self.lidar_elev_min     = np.radians(self.config.get('lidar_elev_min_deg', -7.0))
        self.lidar_elev_max     = np.radians(self.config.get('lidar_elev_max_deg', 52.0))
        self.lidar_max_dist     = self.config.get('lidar_max_dist', 20.0)
        self.lidar_depth_clip   = self.config.get('lidar_depth_clip', self.lidar_max_dist)
        self.static_input_mode = self.config.get(
            'static_input_mode', 'pointcloud')
        if self.static_input_mode not in ('range_image', 'pointcloud'):
            raise ValueError('static_input_mode must be range_image or pointcloud')
        self.static_range_frame_id = self.config.get(
            'static_range_frame_id', 'world')
        if not self.static_range_frame_id:
            raise ValueError('static_range_frame_id must not be empty')
        if (self.config.get('use_lidar', False) and
                self.static_input_mode == 'range_image'):
            if not self.omnidirectional:
                raise RuntimeError(
                    'first-version static range input requires omnidirectional planning')
            upstream_odom_topic = rospy.get_param('/dyn_obj/odom_topic', None)
            if upstream_odom_topic != self.config['odom_topic']:
                raise RuntimeError(
                    'static range pose source mismatch: M-detector uses %r, planner uses %r' %
                    (upstream_odom_topic, self.config['odom_topic']))
            contract = {
                '/dyn_obj/static_range_height': self.lidar_H,
                '/dyn_obj/static_range_width': self.lidar_W,
                '/dyn_obj/static_range_elevation_min_deg': float(
                    self.config.get('lidar_elev_min_deg', -7.0)),
                '/dyn_obj/static_range_elevation_max_deg': float(
                    self.config.get('lidar_elev_max_deg', 52.0)),
                '/dyn_obj/static_range_min_distance': self.min_dis,
                '/dyn_obj/static_range_max_distance': self.lidar_max_dist,
                '/dyn_obj/static_range_frame_id': self.static_range_frame_id,
                '/dyn_obj/static_range_world_aligned': True,
            }
            for parameter, expected in contract.items():
                actual = rospy.get_param(parameter, None)
                if isinstance(expected, float) and actual is not None:
                    try:
                        matches = bool(np.isclose(float(actual), expected))
                    except (TypeError, ValueError):
                        matches = False
                else:
                    matches = actual == expected
                if not matches:
                    raise RuntimeError(
                        'static range contract mismatch for %s: expected %r, got %r' %
                        (parameter, expected, actual))
        # 20 Hz inputs only enter a short sync cache. It is not a task queue; each stage always catches up to the latest complete frame.
        self.sync_cache_duration = self.config.get('sync_cache_duration', 0.25)
        self.sync_timeout = self.config.get('sync_timeout', 0.15)
        self.static_fallback_tolerance = self.config.get('static_fallback_tolerance', 0.06)
        self.odom_sync_tolerance = self.config.get('odom_sync_tolerance', 0.03)
        self.output_age_warn_ms = self.config.get('output_age_warn_ms', 150.0)
        self.output_max_age_ms = self.config.get('output_max_age_ms', 120.0)
        self.visualization_max_rate_hz = float(
            self.config.get('visualization_max_rate_hz', 15.0))
        if self.visualization_max_rate_hz <= 0.0:
            raise ValueError('visualization_max_rate_hz must be positive')
        self.track_prediction_horizon = None
        if self.config.get('use_lidar', False):
            self.track_prediction_horizon = float(
                rospy.get_param('/dyn_obj/tracker/max_prediction_horizon'))
        self.track_ballistic_model = bool(self.config.get('track_ballistic_model', True))
        self.track_gravity = np.asarray(
            self.config.get('track_gravity_world', [0.0, 0.0, -9.81]), dtype=np.float32)
        self.dynamic_sphere_radius_min = float(
            self.config.get('dynamic_sphere_radius_min', 0.10))
        self.dynamic_sphere_radius_max = float(
            self.config.get('dynamic_sphere_radius_max', 0.20))
        self.dynamic_occlusion_tolerance = float(
            self.config.get('dynamic_occlusion_tolerance', 0.15))
        self.sync_condition = Condition()
        self.lidar_cache = StampedCache(self.sync_cache_duration)
        self.odom_cache = StampedCache(self.sync_cache_duration)
        self.tracks_cache = StampedCache(self.sync_cache_duration)
        self.static_cache = StampedCache(self.sync_cache_duration)
        self.static_pose_cache = StampedCache(self.sync_cache_duration)
        self.perception_slot = LatestOnlySlot()
        self.inference_slot = LatestOnlySlot()
        self.last_assembled_stamp_ns = None
        self.last_sync_miss_stamp_ns = None
        self.sync_generation = 0
        self.frame_sequence = 0
        self.last_committed_sequence = -1
        self.sync_timeout_count = 0
        self.static_fallback_count = 0
        self.odom_fallback_count = 0
        self.preprocess_overwrite_count = 0
        self.inference_overwrite_count = 0
        self.stale_work_skip_count = 0
        self.output_stale_drop_count = 0
        self.visualization_overwrite_count = 0
        self.output_latency_ms = deque(maxlen=200)
        self.sync_metric_count = 0
        self.pipeline_metric_lock = Lock()
        self.pipeline_metrics_started_mono = time.monotonic()
        self.pipeline_metrics_period = float(
            self.config.get('pipeline_metrics_period', 5.0))
        self.pipeline_event_times = {
            name: deque(maxlen=1000) for name in (
                'lidar_input', 'tracks_input', 'static_input',
                'static_pose_input', 'complete_frame',
                'preprocess_started', 'preprocess_completed', 'inference_started',
                'inference_completed', 'trajectory_commit',
                'visualization_publish')
        }
        self.pipeline_metric_values = {
            name: deque(maxlen=200) for name in (
                'source_age_at_preprocess_ms', 'source_age_at_inference_start_ms',
                'source_age_at_commit_ms', 'preprocess_time_ms',
                'inference_time_ms', 'postprocess_time_ms',
                'pointcloud_decode_ms', 'current_transform_ms',
                'static_transform_ms', 'current_projection_ms',
                'static_projection_ms', 'static_range_decode_ms',
                'dynamic_rasterization_ms',
                'depth_completion_ms', 'tensor_prep_h2d_ms')
        }
        self.prev_current_valid = None
        self.prev_final_valid = None
        self.visualization_condition = Condition()
        self.visualization_pending = None
        # Single spherical grid: point projection, dynamic spheres, back-projection and network input share the H-1 row mapping.
        self.ray_directions = make_ray_directions(
            self.lidar_H, self.lidar_W, self.lidar_elev_min, self.lidar_elev_max)
        self._bp_sin_el = self.ray_directions[:, 0, 2]
        # Virtual ceiling: after inpainting, overlay a world z=ceiling plane on the equirectangular range image to fill the space above
        # and discourage climbing. The range image is built in the gravity-aligned planning frame (z = world z − body z), so per-row ceiling distance = dz/sin(elev),
        # applied only to upward rays (sin_el>0); constant per row, independent of azimuth.
        self.virtual_ceiling_enable = bool(self.config.get('virtual_ceiling_enable', False))
        self.virtual_ceiling_z      = float(self.config.get('virtual_ceiling_z', 2.0))
        self._ceil_t_floor          = max(self.min_dis * 1.1, self.min_dis + 0.05)

        self.warm_up()

        # ros publisher
        self.lattice_traj_pub = rospy.Publisher("/omnirisk/lattice_trajs_visual", PointCloud2, queue_size=1)
        self.best_traj_pub = rospy.Publisher("/omnirisk/best_traj_visual", PointCloud2, queue_size=1)
        self.all_trajs_pub = rospy.Publisher("/omnirisk/trajs_visual", PointCloud2, queue_size=1)
        self.dynamic_score_trajs_pub = rospy.Publisher(
            "/omnirisk/dynamic_score_trajs_visual", MarkerArray, queue_size=1)
        self.depth_cloud_pub = rospy.Publisher("/omnirisk/depth_cloud_world", PointCloud2, queue_size=1)
        self.ctrl_pub = rospy.Publisher(self.config["ctrl_topic"], PositionCommand, queue_size=1)
        # lidar range image publishers (for visualization)
        self.range_image_single_pub = rospy.Publisher("/omnirisk/range_image_single",  Image,       queue_size=1)
        self.range_image_pub        = rospy.Publisher("/omnirisk/range_image",         Image,       queue_size=1)
        self.range_image_inpaint_pub= rospy.Publisher("/omnirisk/range_image_inpaint", Image,       queue_size=1)
        self.range_coverage_pub     = rospy.Publisher("/omnirisk/range_coverage",      Image,       queue_size=1)
        self.range_cloud_pub        = rospy.Publisher("/omnirisk/range_cloud_body",    PointCloud2, queue_size=1)
        # Final LiDAR depth tensor fed to the network: 96×384, single channel, float32, normalized to [0, 1].
        # Unlike range_image_inpaint, this image already has the omnidirectional resize and network-side invalid-value filling.
        self.depth_input_pub        = rospy.Publisher("/omnirisk/depth_input",         Image,       queue_size=1)
        self.dodge_state_pub        = rospy.Publisher("/omnirisk/dodge_state",         MarkerArray, queue_size=1)
        self.mid360_fov_pub         = rospy.Publisher("/omnirisk/mid360_fov",           MarkerArray, queue_size=1)
        # ros subscriber
        self.odom_sub = rospy.Subscriber(self.config['odom_topic'], Odometry, self.callback_odometry, queue_size=1, tcp_nodelay=True)
        self.depth_sub = rospy.Subscriber(self.config['depth_topic'], Image, self.callback_depth, queue_size=1, tcp_nodelay=True)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.callback_set_goal, queue_size=1)
        self.rc_active_sub = rospy.Subscriber(
            self.rc_active_topic, std_msgs.msg.Bool,
            self.callback_rc_active, queue_size=1, tcp_nodelay=True)

        # lidar subscriber (mutually exclusive with depth_sub, selected by config['use_lidar'])
        if self.config.get('use_lidar', False):
            self.lidar_sub = rospy.Subscriber(
                self.config.get('lidar_topic', '/cloud_registered_body'),
                PointCloud2, self.callback_lidar, queue_size=1, tcp_nodelay=True)
            if self.static_input_mode == 'range_image':
                self.static_sub = rospy.Subscriber(
                    self.config.get('static_range_topic', '/m_detector/static_range'),
                    Image, self.callback_static, queue_size=1, tcp_nodelay=True)
                self.static_pose_sub = rospy.Subscriber(
                    self.config.get(
                        'static_range_pose_topic', '/m_detector/static_range_pose'),
                    Odometry, self.callback_static_pose,
                    queue_size=1, tcp_nodelay=True)
            else:
                self.static_sub = rospy.Subscriber(
                    self.config.get('static_topic', '/m_detector/std_points'),
                    PointCloud2, self.callback_static, queue_size=1, tcp_nodelay=True)

        # tracker output is required to form a complete synchronized LiDAR perception frame.
        if HAVE_TRACKS_MSG:
            self.tracks_sub = rospy.Subscriber(
                self.config.get('tracks_topic', '/m_detector/confirmed_tracks'),
                TrackedObjectArray, self.callback_tracks, queue_size=1, tcp_nodelay=True)
        else:
            print("[OmniRisk] m_detector TrackedObjectArray not available — "
                  "complete LiDAR perception frames cannot be formed; hover only.")
        # ros timer / workers
        rospy.on_shutdown(self._notify_workers_shutdown)
        if self.config.get('use_lidar', False):
            Thread(target=self._frame_assembly_worker, daemon=True).start()
            Thread(target=self._preprocessing_worker, daemon=True).start()
            Thread(target=self._inference_worker, daemon=True).start()
            Thread(target=self._visualization_worker, daemon=True).start()
        rospy.sleep(1.0)  # wait connection...
        self.timer_ctrl = rospy.Timer(rospy.Duration(self.ctrl_dt), self.control_pub)
        self.timer_runtime_state = rospy.Timer(
            rospy.Duration(0.1), self._publish_runtime_state)
        if self.config.get('use_lidar', False):
            self.timer_pipeline_metrics = rospy.Timer(
                rospy.Duration(self.pipeline_metrics_period),
                self._log_pipeline_metrics)
        if self.auto_takeoff and HAVE_TAKEOFF_SRV:
            Thread(target=self._auto_takeoff, daemon=True).start()
        elif self.auto_takeoff:
            print("[auto_takeoff] quadrotor_msgs/SetTakeoffLand not importable — call the takeoff service manually")
        print("OmniRisk Node Ready!")
        rospy.spin()

    def _navigation_active_for_runtime(self):
        now_mono = time.monotonic()
        with self.rc_lock:
            if self.rc_state_received:
                stale = (now_mono - self.rc_state_last_monotonic
                         > self.rc_state_timeout)
                return (not stale and self.rc_active and
                        not self.rc_resume_pending)
        return self.nav_mode or self.external_goal_received

    def _invalidate_active_trajectory(self):
        with self.lock:
            self.ctrl_time = None
            self.repeat_first_trajectory_sample = False
            self.active_trajectory_time = None
            self.active_trajectory_version = None
            self.active_trajectory_commit_mono = None

    def _select_runtime_snapshot_locked(self, threat_detected):
        if threat_detected:
            self.last_threat_time = rospy.Time.now().to_sec()
        threat_active = (threat_detected or
                         (rospy.Time.now().to_sec() - self.last_threat_time)
                         < self.hold_after)
        previous_version = self.runtime_modes.version
        snapshot = self.runtime_modes.select(
            threat_active, self._navigation_active_for_runtime())
        if snapshot.version != previous_version:
            self.perception_slot.clear()
            self.inference_slot.clear()
            self.last_action_id = None
            self._invalidate_active_trajectory()
            with self.visualization_condition:
                self.visualization_pending = None
            rospy.loginfo(
                "OmniRisk runtime mode=%s version=%d", snapshot.mode.value,
                snapshot.version)
        return snapshot

    def _auto_takeoff(self):
        """Call the controller takeoff service after startup: wait for service + odom → arm + climb to hover_height.
        The controller state machine handles takeoff; control_pub sends nothing before that, then anchors home and takes over."""
        try:
            rospy.wait_for_service(self.takeoff_service, timeout=30.0)
        except rospy.ROSException:
            rospy.logwarn(f"[auto_takeoff] service {self.takeoff_service} unavailable, skipping (take off manually)")
            return
        while not self.odom_init and not rospy.is_shutdown():
            rospy.sleep(0.2)
        rospy.sleep(2.0)   # let the controller receive odom / become armable
        try:
            call = rospy.ServiceProxy(self.takeoff_service, SetTakeoffLand)
            res = call(takeoff=True, takeoff_altitude=self.hover_height)
            rospy.loginfo(f"[auto_takeoff] taking off to {self.hover_height}m → res={getattr(res, 'res', res)}")
        except rospy.ServiceException as e:
            rospy.logwarn(f"[auto_takeoff] service call failed: {e}")

    def callback_set_goal(self, data):
        with self.rc_lock:
            if self.rc_state_received and not self.rc_active:
                return
            if self.rc_state_received:
                self.rc_resume_pending = False
        with self.sync_condition:
            self.goal = np.asarray([data.pose.position.x, data.pose.position.y, self.hover_height])
            self.external_goal_received = True
            self.arrive = False
            self.last_action_id = None
        print(f"New Goal: ({data.pose.position.x:.1f}, {data.pose.position.y:.1f}, {self.goal[2]:.2f})")

    def callback_rc_active(self, data):
        active = bool(data.data)
        now_mono = time.monotonic()
        with self.rc_lock:
            previous = self.rc_active if self.rc_state_received else None
            self.rc_state_received = True
            self.rc_active = active
            self.rc_state_last_monotonic = now_mono
            if active:
                if previous is not True:
                    self.rc_resume_pending = True
            else:
                self.rc_resume_pending = False
            need_capture = (not active and
                            (previous is not False or self.rc_home is None))

        if active:
            if previous is not True:
                self._invalidate_active_trajectory()
            return
        if need_capture:
            self._capture_rc_home()

    def _capture_rc_home(self):
        if not self.home_set:
            return False
        with self.odom_lock:
            if not self.odom_init:
                return False
            position = self.odom.pose.pose.position
            current = np.array(
                [position.x, position.y, position.z], dtype=np.float64)
        with self.rc_lock:
            if self.rc_active:
                return False
            self.rc_home = np.array(
                [current[0], current[1], self.hover_height], dtype=np.float64)
            self.rc_home_z_ref = float(current[2])
            rc_home = self.rc_home.copy()
        with self.sync_condition:
            self.goal = rc_home
            self.external_goal_received = False
            self.arrive = False
            self.last_action_id = None
        self._invalidate_active_trajectory()
        self.desire_init = False
        rospy.loginfo(
            "RC neutral home captured: (%.2f, %.2f, %.2f)",
            rc_home[0], rc_home[1], rc_home[2])
        return True

    def _rc_home_required(self):
        now_mono = time.monotonic()
        with self.rc_lock:
            if not self.rc_state_received:
                return False
            stale = (now_mono - self.rc_state_last_monotonic
                     > self.rc_state_timeout)
            capture = stale and self.rc_active
            if capture:
                self.rc_active = False
                self.rc_resume_pending = False
            required = not self.rc_active or self.rc_resume_pending
            missing_home = required and self.rc_home is None
        if stale:
            rospy.logwarn_throttle(
                1.0, "RC goal state timed out; holding current position")
        if capture or missing_home:
            self._capture_rc_home()
        return required

    def _notify_workers_shutdown(self):
        with self.sync_condition:
            self.sync_condition.notify_all()
        self.perception_slot.clear()
        self.inference_slot.clear()
        with self.visualization_condition:
            self.visualization_condition.notify_all()

    def _record_pipeline_event(self, name, now_mono=None):
        now_mono = time.monotonic() if now_mono is None else float(now_mono)
        with self.pipeline_metric_lock:
            self.pipeline_event_times[name].append(now_mono)

    def _record_pipeline_value(self, name, value):
        with self.pipeline_metric_lock:
            self.pipeline_metric_values[name].append(float(value))

    def _record_pipeline_values(self, values):
        with self.pipeline_metric_lock:
            for name, value in values.items():
                self.pipeline_metric_values[name].append(float(value))

    @staticmethod
    def _event_rate(samples, now_mono, started_mono, window_sec):
        elapsed = min(float(window_sec), float(now_mono) - float(started_mono))
        if elapsed <= 0.0:
            return 0.0
        cutoff = float(now_mono) - float(window_sec)
        return sum(stamp >= cutoff for stamp in samples) / elapsed

    @staticmethod
    def _metric_percentiles(samples):
        if not samples:
            return 0.0, 0.0, 0.0
        return tuple(float(value) for value in np.percentile(samples, [50, 95, 100]))

    def _log_pipeline_metrics(self, _timer=None):
        now_mono = time.monotonic()
        with self.pipeline_metric_lock:
            rates = {
                name: self._event_rate(
                    samples, now_mono, self.pipeline_metrics_started_mono,
                    self.pipeline_metrics_period)
                for name, samples in self.pipeline_event_times.items()
            }
            percentiles = {
                name: self._metric_percentiles(samples)
                for name, samples in self.pipeline_metric_values.items()
            }
        rospy.loginfo(
            "OmniRisk metrics lidar_input_rate=%.1f tracks_input_rate=%.1f "
            "static_input_rate=%.1f complete_frame_rate=%.1f "
            "preprocess_started_rate=%.1f preprocess_completed_rate=%.1f "
            "inference_started_rate=%.1f inference_completed_rate=%.1f "
            "trajectory_commit_rate=%.1f visualization_publish_rate=%.1f "
            "preprocess_overwrite_count=%d "
            "inference_overwrite_count=%d stale_work_skip_count=%d static_fallback_count=%d "
            "odom_fallback_count=%d sync_timeout_count=%d stale_output_count=%d "
            "visualization_overwrite_count=%d" %
            (rates['lidar_input'], rates['tracks_input'], rates['static_input'],
             rates['complete_frame'], rates['preprocess_started'],
             rates['preprocess_completed'], rates['inference_started'],
             rates['inference_completed'], rates['trajectory_commit'],
             rates['visualization_publish'],
             self.preprocess_overwrite_count, self.inference_overwrite_count,
             self.stale_work_skip_count,
             self.static_fallback_count, self.odom_fallback_count,
             self.sync_timeout_count, self.output_stale_drop_count,
             self.visualization_overwrite_count))
        rospy.loginfo(
            "OmniRisk metrics P50/P95/max(ms) source_age_at_preprocess_ms=%.1f/%.1f/%.1f "
            "source_age_at_inference_start_ms=%.1f/%.1f/%.1f "
            "source_age_at_commit_ms=%.1f/%.1f/%.1f "
            "preprocess_time_ms=%.1f/%.1f/%.1f "
            "inference_time_ms=%.1f/%.1f/%.1f postprocess_time_ms=%.1f/%.1f/%.1f" %
            (percentiles['source_age_at_preprocess_ms']
             + percentiles['source_age_at_inference_start_ms']
             + percentiles['source_age_at_commit_ms']
             + percentiles['preprocess_time_ms']
             + percentiles['inference_time_ms']
             + percentiles['postprocess_time_ms']))
        rospy.loginfo(
            "OmniRisk preprocess stages P50/P95/max(ms) "
            "pointcloud_decode=%.2f/%.2f/%.2f "
            "static_range_decode=%.2f/%.2f/%.2f "
            "current_transform=%.2f/%.2f/%.2f static_transform=%.2f/%.2f/%.2f "
            "current_projection=%.2f/%.2f/%.2f static_projection=%.2f/%.2f/%.2f "
            "dynamic_rasterization=%.2f/%.2f/%.2f "
            "depth_completion=%.2f/%.2f/%.2f "
            "tensor_prep_h2d=%.2f/%.2f/%.2f" %
            (percentiles['pointcloud_decode_ms']
             + percentiles['static_range_decode_ms']
             + percentiles['current_transform_ms']
             + percentiles['static_transform_ms']
             + percentiles['current_projection_ms']
             + percentiles['static_projection_ms']
             + percentiles['dynamic_rasterization_ms']
             + percentiles['depth_completion_ms']
             + percentiles['tensor_prep_h2d_ms']))

    def _reset_pipeline_locked(self):
        for item_cache in (self.lidar_cache, self.odom_cache,
                           self.tracks_cache, self.static_cache,
                           self.static_pose_cache):
            item_cache.clear()
        self.sync_generation += 1
        self.last_assembled_stamp_ns = None
        self.last_sync_miss_stamp_ns = None
        self.frame_sequence = 0
        self.last_committed_sequence = -1
        self.last_action_id = None
        self._invalidate_active_trajectory()
        self.prev_current_valid = None
        self.prev_final_valid = None
        self.perception_slot.clear()
        self.inference_slot.clear()

    def _cache_synced_message(self, cache, data, stream_name):
        if data.header.stamp.to_nsec() == 0:
            rospy.logwarn_throttle(1.0, f"OmniRisk dropping zero-stamp {stream_name} message")
            return
        stamp_ns = data.header.stamp.to_nsec()
        with self.sync_condition:
            moved_backward = cache.append(stamp_ns, data)
            if moved_backward:
                self._reset_pipeline_locked()
                cache.append(stamp_ns, data)
                rospy.logwarn(
                    f"{stream_name} timestamp moved backward; reset OmniRisk sync caches")
            self.sync_condition.notify()

    def callback_tracks(self, data):
        """Store the tracker measurement message; parsing/prediction happens in the worker."""
        self._record_pipeline_event('tracks_input')
        self._cache_synced_message(self.tracks_cache, data, 'confirmed_tracks')

    # the first frame
    def callback_odometry(self, data):
        if data.header.stamp.to_nsec() == 0:
            rospy.logwarn_throttle(1.0, "OmniRisk dropping zero-stamp odom message")
            return
        # Skip messages with invalid (zero-norm) quaternion — happens during VIO initialization
        q = data.pose.pose.orientation
        if (q.x**2 + q.y**2 + q.z**2 + q.w**2) < 0.5:
            return
        with self.odom_lock:
            self.odom = data
            self.odom_init = True
        self._cache_synced_message(self.odom_cache, data, 'odom')
        if not self.desire_init:
            self.desire_pos = np.array((data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z))
            self.desire_vel = np.array((data.twist.twist.linear.x, data.twist.twist.linear.y, data.twist.twist.linear.z))
            self.desire_acc = np.array((0.0, 0.0, 0.0))
            ypr = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('ZYX', degrees=False)
            self.last_yaw = ypr[0]
        pos = np.array((data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z))
        # Wait until the controller has taken off and hovers before anchoring home: z above threshold and vertical speed settled. Before takeoff the planner sends
        # no pos_cmd (see control_pub), so it does not interfere with the controller takeoff state machine.
        # home z is always hover_height, not sampled from odom, so anchoring early or late never picks the wrong height.
        if not self.home_set:
            vz = data.twist.twist.linear.z
            if pos[2] > self.takeoff_alt_min and abs(vz) < self.vz_settle:
                self.home = np.array([pos[0], pos[1], self.hover_height])
                if not self.nav_mode and not self.external_goal_received:
                    self.goal = self.home.copy()   # in DODGE the endpoint term pulls the drone back to home
                self.home_set = True
                print(f"Home anchored (airborne): ({self.home[0]:.2f}, {self.home[1]:.2f}, {self.home[2]:.2f})")
        arrive_radius = (self.external_goal_arrive_radius
                         if self.external_goal_received else self.arrive_radius)
        if np.linalg.norm(pos - self.goal) < arrive_radius and not self.arrive:
            print("Arrive!")
            self.arrive = True

    def process_odom(self, yaw_only=False, runtime_config=None):
        if runtime_config is None:
            runtime_config = self.runtime_profiles[RuntimeMode.DODGE]
        # Rwb -> Rwc -> Rcw
        Rotation_wb = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                                   self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_matrix()
        if yaw_only and self.omnidirectional:
            # World-azimuth frame: the network plans in a gravity-aligned world-azimuth frame, body↔world does not apply heading yaw.
            # range/obs/endstate all use world azimuth (Rotation_wc=I), yaw is free → independent of heading.
            self.Rotation_wc = np.eye(3, dtype=np.float64)
        elif yaw_only:
            # (legacy heading-relative path) range image aligned to the yaw-only stabilized frame, Rz(yaw) removes roll/pitch
            yaw = np.arctan2(Rotation_wb[1, 0], Rotation_wb[0, 0])
            cy, sy = np.cos(yaw), np.sin(yaw)
            self.Rotation_wc = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        else:
            self.Rotation_wc = np.dot(Rotation_wb, self.Rotation_bc)
        Rotation_cw = self.Rotation_wc.T

        # vel and acc
        vel_w = self.desire_vel if self.plan_from_reference else np.array([self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z])
        vel_c = np.dot(Rotation_cw, vel_w)
        acc_w = self.desire_acc
        acc_c = np.dot(Rotation_cw, acc_w)

        # goal_dir: use actual odom position when plan_from_reference=False,
        # because desire_pos (reference trajectory) can advance far ahead of the
        # actual drone, causing goal_w to point backward once it overshoots the goal.
        if self.plan_from_reference:
            ref_pos = self.desire_pos
        else:
            ref_pos = np.array([self.odom.pose.pose.position.x,
                                self.odom.pose.pose.position.y,
                                self.odom.pose.pose.position.z])
        goal_w = self.goal - ref_pos
        goal_c = np.dot(Rotation_cw, goal_w)

        obs = np.concatenate((vel_c, acc_c, goal_c), axis=0).astype(np.float32)
        obs_norm = self.state_transform.normalize_obs(
            torch.from_numpy(obs[None, :]), runtime_config.vel_max,
            runtime_config.acc_max, runtime_config.goal_length)
        return obs_norm

    # ------------------------------------------------------------------ #
    #  LiDAR range image pipeline                                        #
    # ------------------------------------------------------------------ #

    def _project_to_range_image(self, pts):
        """Project planning-frame points onto the canonical H-1 spherical grid."""
        return project_points_to_range_image(
            pts, self.lidar_H, self.lidar_W, self.lidar_elev_min,
            self.lidar_elev_max, 0.1, self.lidar_max_dist)

    def _inpaint_range_image(self, range_img):
        """Fill holes with anisotropic push-pull; valid raw ranges are protected by a mask and never overwritten."""
        invalid = (~np.isfinite(range_img) |
                   (range_img >= self.lidar_max_dist))
        return complete_depth(range_img, invalid, self.lidar_max_dist)

    def _virtual_ceiling_range(self, pos_z):
        """Return the unfilled ceiling range for visibility and final composition."""
        ceiling = np.full((self.lidar_H, self.lidar_W), self.lidar_max_dist, dtype=np.float32)
        dz = self.virtual_ceiling_z - float(pos_z)
        if dz <= 0:
            return ceiling
        up = self._bp_sin_el > 1e-3
        ceil_r = np.full(self.lidar_H, self.lidar_max_dist, dtype=np.float32)
        ceil_r[up] = np.clip(dz / self._bp_sin_el[up],
                             self._ceil_t_floor, self.lidar_max_dist)
        return np.minimum(ceiling, ceil_r[:, None])

    def _make_range_image_msg(self, range_img, stamp, frame_id='body'):
        """Encode the range image as a color ROS Image (JET colormap, invalid pixels = black)."""
        norm     = np.clip(range_img / self.lidar_max_dist, 0.0, 1.0)
        gray     = np.uint8((1.0 - norm) * 255)            # near = bright, far = dark
        color    = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        color[range_img >= self.lidar_max_dist * 0.999] = 0  # holes = black
        color_rgb = color[:, :, ::-1].copy()               # BGR→RGB
        msg = Image()
        msg.header.stamp    = stamp
        msg.header.frame_id = frame_id
        msg.height, msg.width = color_rgb.shape[:2]
        msg.encoding = 'rgb8'
        msg.step     = msg.width * 3
        msg.data     = color_rgb.tobytes()
        return msg

    def _make_coverage_msg(self, range_img, stamp, frame_id='body'):
        """Encode the valid-pixel mask as a mono8 ROS Image (255 = valid, 0 = no return)."""
        coverage = np.uint8((range_img < self.lidar_max_dist * 0.999) * 255)
        msg = Image()
        msg.header.stamp    = stamp
        msg.header.frame_id = frame_id
        msg.height, msg.width = coverage.shape
        msg.encoding = 'mono8'
        msg.step     = msg.width
        msg.data     = coverage.tobytes()
        return msg

    def _publish_runtime_state(self, _timer=None):
        """Publish the actual control result independently from inference visuals."""
        if self.dodge_state_pub.get_num_connections() == 0:
            return
        with self.lock:
            mode = self.control_mode
        with self.odom_lock:
            if not self.odom_init:
                return
            position = self.odom.pose.pose.position
            t_wb = np.array([position.x, position.y, position.z], dtype=np.float64)
        stamp = rospy.Time.now()
        label, color, circle_points = runtime_state_geometry(
            mode, t_wb, self.threat_radius)

        circle = Marker()
        circle.header.stamp, circle.header.frame_id = stamp, 'world'
        circle.ns, circle.id = 'dodge_state', 0
        circle.type, circle.action = Marker.LINE_STRIP, Marker.ADD
        circle.pose.orientation.w = 1.0
        circle.scale.x = 0.05
        circle.color.r, circle.color.g, circle.color.b = color
        circle.color.a = 1.0
        circle.lifetime = rospy.Duration(0)
        circle.points = [Point(*point) for point in circle_points]

        text = Marker()
        text.header.stamp, text.header.frame_id = stamp, 'world'
        text.ns, text.id = 'dodge_state', 1
        text.type, text.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        text.pose.position.x, text.pose.position.y = t_wb[0], t_wb[1]
        text.pose.position.z = t_wb[2] + 1.0
        text.pose.orientation.w = 1.0
        text.scale.z = 0.5
        text.color.r, text.color.g, text.color.b = color
        text.color.a = 1.0
        text.lifetime = rospy.Duration(0)
        text.text = label

        legacy_fov_deletes = []
        for marker_id in range(3):
            marker = Marker()
            marker.header.stamp, marker.header.frame_id = stamp, 'world'
            marker.ns, marker.id = 'mid360_fov', marker_id
            marker.action = Marker.DELETE
            legacy_fov_deletes.append(marker)

        self.dodge_state_pub.publish(MarkerArray(
            markers=[circle, text] + legacy_fov_deletes))

    def _publish_mid360_fov(self, t_wb, rotation_wc, stamp):
        """Publish the heavy MID-360 field-of-view geometry best effort."""
        if self.mid360_fov_pub.get_num_connections() == 0:
            return

        fov_surface = Marker()
        fov_surface.header.stamp, fov_surface.header.frame_id = stamp, 'world'
        fov_surface.ns, fov_surface.id = 'mid360_fov', 0
        fov_surface.type, fov_surface.action = Marker.TRIANGLE_LIST, Marker.ADD
        fov_surface.pose.orientation.w = 1.0
        fov_surface.scale.x = fov_surface.scale.y = fov_surface.scale.z = 1.0
        fov_surface.color.r, fov_surface.color.g = 0.0, 0.65
        fov_surface.color.b, fov_surface.color.a = 1.0, 0.06

        fov_wire = Marker()
        fov_wire.header.stamp, fov_wire.header.frame_id = stamp, 'world'
        fov_wire.ns, fov_wire.id = 'mid360_fov', 1
        fov_wire.type, fov_wire.action = Marker.LINE_LIST, Marker.ADD
        fov_wire.pose.orientation.w = 1.0
        fov_wire.scale.x = 0.015
        fov_wire.color.r, fov_wire.color.g = 0.0, 0.75
        fov_wire.color.b, fov_wire.color.a = 1.0, 0.8

        def world_point(azimuth, elevation, distance):
            local = distance * np.array([
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ])
            world = rotation_wc @ local + t_wb
            return Point(*world)

        origin = Point(*t_wb)
        fov_display_radius = 5.0
        azimuths = np.linspace(0.0, 2.0 * np.pi, 37)
        elevations = np.linspace(self.lidar_elev_min, self.lidar_elev_max, 8)
        for az0, az1 in zip(azimuths[:-1], azimuths[1:]):
            lower0 = world_point(az0, self.lidar_elev_min, fov_display_radius)
            lower1 = world_point(az1, self.lidar_elev_min, fov_display_radius)
            upper0 = world_point(az0, self.lidar_elev_max, fov_display_radius)
            upper1 = world_point(az1, self.lidar_elev_max, fov_display_radius)
            fov_surface.points.extend([origin, lower1, lower0, origin, upper0, upper1])
            fov_wire.points.extend([lower0, lower1, upper0, upper1])
            for el0, el1 in zip(elevations[:-1], elevations[1:]):
                p00 = world_point(az0, el0, fov_display_radius)
                p01 = world_point(az1, el0, fov_display_radius)
                p10 = world_point(az0, el1, fov_display_radius)
                p11 = world_point(az1, el1, fov_display_radius)
                fov_surface.points.extend([p00, p01, p11, p00, p11, p10])
        for azimuth in azimuths[:-1:3]:
            lower = world_point(azimuth, self.lidar_elev_min, fov_display_radius)
            upper = world_point(azimuth, self.lidar_elev_max, fov_display_radius)
            fov_wire.points.extend([origin, lower, origin, upper])
            for el0, el1 in zip(elevations[:-1], elevations[1:]):
                fov_wire.points.extend([
                    world_point(azimuth, el0, fov_display_radius),
                    world_point(azimuth, el1, fov_display_radius),
                ])

        self.mid360_fov_pub.publish(MarkerArray(
            markers=[fov_surface, fov_wire]))

    def callback_static(self, data):
        """Cache the configured static background representation."""
        self._record_pipeline_event('static_input')
        self._cache_synced_message(self.static_cache, data, self.static_input_mode)

    def callback_static_pose(self, data):
        """Cache the exact planning pose used to generate a static range image."""
        self._record_pipeline_event('static_pose_input')
        self._cache_synced_message(
            self.static_pose_cache, data, 'static_range_pose')

    @staticmethod
    def _pointcloud_xyz(data):
        if not data.data or data.point_step < 12 or data.point_step % 4 != 0:
            return np.empty((0, 3), dtype=np.float32)
        pts = np.frombuffer(data.data, dtype=np.float32).reshape(
            -1, data.point_step // 4)[:, :3].copy()
        return pts[np.isfinite(pts).all(axis=1)]

    def _project_to_mask(self, pts):
        """Project diagnostic planning-frame points with the canonical grid."""
        return (self._project_to_range_image(pts) < self.lidar_max_dist).astype(np.float32)

    def _build_dynamic_inputs(self, mask_full, velocity_full):
        """Map dynamic mask and velocity with the same panorama column LUT as depth."""
        mask = np.ascontiguousarray(map_panorama_to_network(
            mask_full, self.lidar_W, self.width, self.omnidirectional),
            dtype=np.float32)
        velocity = map_panorama_to_network(
            velocity_full, self.lidar_W, self.width, self.omnidirectional).copy()
        velocity[mask == 0.0] = 0.0
        velocity = np.ascontiguousarray(
            velocity.transpose(2, 0, 1), dtype=np.float32)
        return (torch.from_numpy(mask.reshape(1, 1, self.height, self.width)).to(
                    self.device, non_blocking=True),
                torch.from_numpy(velocity.reshape(
                    1, 3, self.height, self.width)).to(self.device, non_blocking=True))

    def _backproject_range_image(self, range_img):
        """Backproject with the same canonical ray directions used for composition."""
        valid = range_img < self.lidar_max_dist * 0.999
        rows, cols = np.where(valid)
        r = range_img[rows, cols]
        return (self.ray_directions[rows, cols] * r[:, None]).astype(np.float32)

    @staticmethod
    def _make_xyz32_cloud(points, stamp, frame_id):
        """Build XYZ32 directly from contiguous bytes, avoiding per-point Python work."""
        count, payload, is_dense = pack_xyz32(points)
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = 1
        msg.width = count
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * count
        msg.data = payload
        msg.is_dense = is_dense
        return msg

    @staticmethod
    def _odom_state_from_msg(data):
        q = data.pose.pose.orientation
        p = data.pose.pose.position
        v = data.twist.twist.linear
        return {
            'position': np.array([p.x, p.y, p.z], dtype=np.float64),
            'quaternion': np.array([q.x, q.y, q.z, q.w], dtype=np.float64),
            'velocity': np.array([v.x, v.y, v.z], dtype=np.float64),
            'stamp': data.header.stamp,
        }

    def _interpolate_odom_locked(self, stamp_ns, arrival_deadline=None):
        before, after = self.odom_cache.bracket(stamp_ns)
        if arrival_deadline is not None:
            if before is not None and before.arrival_mono > arrival_deadline:
                before = None
            if after is not None and after.arrival_mono > arrival_deadline:
                after = None
        tolerance_ns = int(self.odom_sync_tolerance * 1e9)
        if before is not None and before.stamp_ns == stamp_ns:
            state = self._odom_state_from_msg(before.value)
            state['acceleration'] = self._odom_acceleration_locked(
                stamp_ns, arrival_deadline)
            return state, 0.0, False
        if after is not None and after.stamp_ns == stamp_ns:
            state = self._odom_state_from_msg(after.value)
            state['acceleration'] = self._odom_acceleration_locked(
                stamp_ns, arrival_deadline)
            return state, 0.0, False
        if before is not None and after is not None:
            left = stamp_ns - before.stamp_ns
            right = after.stamp_ns - stamp_ns
            if left <= tolerance_ns and right <= tolerance_ns and after.stamp_ns > before.stamp_ns:
                ratio = left / float(after.stamp_ns - before.stamp_ns)
                state0 = self._odom_state_from_msg(before.value)
                state1 = self._odom_state_from_msg(after.value)
                state = interpolate_kinematic_state(state0, state1, ratio)
                state['acceleration'] = self._odom_acceleration_locked(
                    stamp_ns, arrival_deadline)
                return state, max(left, right) / 1e6, False
        nearest = min(
            (item for item in (before, after) if item is not None),
            key=lambda item: abs(item.stamp_ns - stamp_ns),
            default=None)
        if nearest is not None and abs(nearest.stamp_ns - stamp_ns) <= tolerance_ns:
            state = self._odom_state_from_msg(nearest.value)
            state['acceleration'] = self._odom_acceleration_locked(
                stamp_ns, arrival_deadline)
            return state, abs(nearest.stamp_ns - stamp_ns) / 1e6, True
        return None, None, False

    def _odom_acceleration_locked(self, stamp_ns, arrival_deadline=None):
        previous, exact, following = self.odom_cache.neighbors(stamp_ns)
        if arrival_deadline is not None:
            if previous is not None and previous.arrival_mono > arrival_deadline:
                previous = None
            if exact is not None and exact.arrival_mono > arrival_deadline:
                exact = None
            if following is not None and following.arrival_mono > arrival_deadline:
                following = None
        tolerance_ns = int(self.odom_sync_tolerance * 1e9)
        left = previous
        right = following
        if exact is not None:
            if left is None or exact.stamp_ns - left.stamp_ns > tolerance_ns:
                left = exact
            if right is None or right.stamp_ns - exact.stamp_ns > tolerance_ns:
                right = exact
        if left is None or right is None or right.stamp_ns <= left.stamp_ns:
            return np.zeros(3, dtype=np.float64)
        if stamp_ns - left.stamp_ns > tolerance_ns or right.stamp_ns - stamp_ns > tolerance_ns:
            return np.zeros(3, dtype=np.float64)
        left_velocity = self._odom_state_from_msg(left.value)['velocity']
        right_velocity = self._odom_state_from_msg(right.value)['velocity']
        return (right_velocity - left_velocity) / ((right.stamp_ns - left.stamp_ns) / 1e9)

    def _current_odom_state(self):
        with self.odom_lock:
            if not self.odom_init:
                return None
            return self._odom_state_from_msg(self.odom)

    def _planning_rotation(self, odom_state):
        R_wb = R.from_quat(odom_state['quaternion']).as_matrix()
        if self.omnidirectional:
            return np.eye(3, dtype=np.float64), R_wb
        yaw = np.arctan2(R_wb[1, 0], R_wb[0, 0])
        cy, sy = np.cos(yaw), np.sin(yaw)
        R_yaw = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        return R_yaw, R_wb

    def _static_range_pose_state(self, pose_message, stamp_ns, arrival_deadline):
        state = decode_static_range_pose(
            pose_message, stamp_ns, self.static_range_frame_id)
        state['acceleration'] = self._odom_acceleration_locked(
            stamp_ns, arrival_deadline)
        return state

    def _process_odom_at(self, odom_state, Rotation_wc, runtime_config):
        Rotation_cw = Rotation_wc.T
        vel_c = Rotation_cw @ odom_state['velocity']
        acc_c = Rotation_cw @ odom_state['acceleration']
        goal_c = Rotation_cw @ (self.goal - odom_state['position'])
        obs = np.concatenate((vel_c, acc_c, goal_c), axis=0).astype(np.float32)
        return self.state_transform.normalize_obs(
            torch.from_numpy(obs[None, :]), runtime_config.vel_max,
            runtime_config.acc_max, runtime_config.goal_length)

    def _tracks_at(self, data, stamp_ns):
        measurement_ns = data.header.stamp.to_nsec()
        dt = max(0.0, (stamp_ns - measurement_ns) / 1e9)
        acceleration = self.track_gravity if self.track_ballistic_model else np.zeros(3, dtype=np.float32)
        tracks = []
        for obs in data.tracks:
            if not obs.confirmed:
                continue
            position = np.array(
                [obs.position.x, obs.position.y, obs.position.z], dtype=np.float32)
            velocity = np.array(
                [obs.velocity.x, obs.velocity.y, obs.velocity.z], dtype=np.float32)
            prediction = predict_kinematics(
                position, velocity, dt, self.track_prediction_horizon, acceleration)
            if prediction is None:
                return None
            predicted_position, predicted_velocity = prediction
            dimensions = np.array(
                [obs.dimensions.x, obs.dimensions.y, obs.dimensions.z], dtype=np.float32)
            if not np.isfinite(dimensions).all() or (dimensions <= 0.0).any():
                continue
            tracks.append({
                'track_uid': int(obs.track_uid),
                'centroid': predicted_position,
                'v': predicted_velocity,
                'size': dimensions,
            })
        return tracks

    def _report_incomplete_sync_locked(self, now_mono):
        for lidar in self.lidar_cache.newest_first():
            if (self.last_assembled_stamp_ns is not None and
                    lidar.stamp_ns <= self.last_assembled_stamp_ns):
                return
            if now_mono - lidar.arrival_mono < self.sync_timeout:
                return
            if self.last_sync_miss_stamp_ns == lidar.stamp_ns:
                return
            missing = []
            tracks = self.tracks_cache.exact(lidar.stamp_ns)
            if (tracks is None or
                    tracks.arrival_mono - lidar.arrival_mono > self.sync_timeout):
                missing.append('tracks')
            static = self.static_cache.exact(lidar.stamp_ns)
            if (static is not None and
                    static.arrival_mono - lidar.arrival_mono > self.sync_timeout):
                static = None
            if static is None and self.static_input_mode == 'pointcloud':
                static = self.static_cache.latest_before(
                    lidar.stamp_ns, int(self.static_fallback_tolerance * 1e9),
                    inclusive=False)
                if (static is not None and
                        static.arrival_mono - lidar.arrival_mono > self.sync_timeout):
                    static = None
            if static is None:
                missing.append(self.static_input_mode)
            if self.static_input_mode == 'range_image':
                pose = self.static_pose_cache.exact(lidar.stamp_ns)
                if (pose is None or
                        pose.arrival_mono - lidar.arrival_mono > self.sync_timeout):
                    missing.append('static_range_pose')
            else:
                state, _, _ = self._interpolate_odom_locked(
                    lidar.stamp_ns, lidar.arrival_mono + self.sync_timeout)
                if state is None:
                    missing.append('odom')
            if missing:
                self.last_sync_miss_stamp_ns = lidar.stamp_ns
                self.sync_timeout_count += 1
                rospy.logwarn_throttle(
                    1.0, "OmniRisk sync timeout; skipped perception stamp %.6f missing=%s" %
                    (lidar.stamp_ns / 1e9, ','.join(missing)))
                self.last_assembled_stamp_ns = lidar.stamp_ns
            return

    def _frame_assembly_worker(self):
        while not rospy.is_shutdown():
            with self.sync_condition:
                self.sync_condition.wait(timeout=0.02)
                if rospy.is_shutdown():
                    return
                now_mono = time.monotonic()
                required_caches = {'tracks': self.tracks_cache}
                if self.static_input_mode == 'range_image':
                    required_caches['static_pose'] = self.static_pose_cache
                selection = select_latest_complete(
                    self.lidar_cache,
                    required_caches,
                    self.static_cache,
                    self.last_assembled_stamp_ns,
                    int(self.static_fallback_tolerance * 1e9),
                    self.sync_timeout,
                    allow_static_fallback=(
                        self.static_input_mode == 'pointcloud'))
                if selection is None:
                    self._report_incomplete_sync_locked(now_mono)
                    continue
                arrival_deadline = (
                    selection.lidar.arrival_mono + self.sync_timeout)
                if self.static_input_mode == 'range_image':
                    try:
                        odom_state = self._static_range_pose_state(
                            selection.required['static_pose'].value,
                            selection.stamp_ns, arrival_deadline)
                    except ValueError as error:
                        rospy.logerr_throttle(
                            1.0, 'OmniRisk rejected static range pose: %s' % error)
                        self.last_assembled_stamp_ns = selection.stamp_ns
                        continue
                    odom_delta_ms = 0.0
                    odom_degraded = False
                else:
                    odom_state, odom_delta_ms, odom_degraded = (
                        self._interpolate_odom_locked(
                            selection.stamp_ns, arrival_deadline))
                    if odom_state is None:
                        self._report_incomplete_sync_locked(now_mono)
                        continue
                self.last_assembled_stamp_ns = selection.stamp_ns
                self.frame_sequence += 1
                if selection.static_degraded:
                    self.static_fallback_count += 1
                if odom_degraded:
                    self.odom_fallback_count += 1
                tracks = self._tracks_at(
                    selection.required['tracks'].value, selection.stamp_ns)
                if tracks is None:
                    rospy.logwarn_throttle(
                        1.0, "OmniRisk skipped frame because track prediction exceeds horizon")
                    continue
                threat_detected = any(
                    np.linalg.norm(track['centroid'] - odom_state['position'])
                    < self.threat_radius for track in tracks)
                runtime_snapshot = self._select_runtime_snapshot_locked(
                    threat_detected)
                frame = PerceptionFrame(
                    sequence=self.frame_sequence,
                    stamp_ns=selection.stamp_ns,
                    lidar=selection.lidar,
                    tracks_at_ref=tracks,
                    static=selection.static,
                    odom_state=odom_state,
                    odom_delta_ms=odom_delta_ms,
                    static_delta_ms=(selection.stamp_ns - selection.static.stamp_ns) / 1e6,
                    static_degraded=selection.static_degraded,
                    odom_degraded=odom_degraded,
                    generation=self.sync_generation,
                    runtime_snapshot=runtime_snapshot,
                    threat_detected=threat_detected,
                    lidar_arrival_mono=selection.lidar.arrival_mono)
                if self.perception_slot.publish(frame):
                    self.preprocess_overwrite_count += 1
                self._record_pipeline_event('complete_frame')

    def callback_lidar(self, data):
        """Lightweight ROS callback: cache only; decoding/inference run in the worker."""
        self._record_pipeline_event('lidar_input')
        self._cache_synced_message(self.lidar_cache, data, 'lidar')

    def _record_sync_metrics(self, lidar_arrival_mono, current_valid, final_valid,
                             static_delta_ms, odom_delta_ms):
        latency_ms = elapsed_since_arrival_ms(lidar_arrival_mono)
        self.output_latency_ms.append(latency_ms)
        self.sync_metric_count += 1
        current_coverage = float(current_valid.mean())
        final_coverage = float(final_valid.mean())
        current_change = (float(np.not_equal(current_valid, self.prev_current_valid).mean())
                          if self.prev_current_valid is not None else 0.0)
        final_change = (float(np.not_equal(final_valid, self.prev_final_valid).mean())
                        if self.prev_final_valid is not None else 0.0)
        self.prev_current_valid = current_valid
        self.prev_final_valid = final_valid
        if self.sync_metric_count % 20 == 0:
            latency_p95 = float(np.percentile(self.output_latency_ms, 95))
            log = rospy.logwarn if latency_p95 > self.output_age_warn_ms else rospy.loginfo
            log(
                "OmniRisk sync latency_p95=%.1fms latency_now=%.1fms odom_delta=%.1fms "
                "static_delta=%.1fms coverage(single/final)=%.3f/%.3f "
                "pixel_change(single/final)=%.3f/%.3f fallback(static/odom)=%d/%d timeouts=%d" %
                (latency_p95, latency_ms, odom_delta_ms,
                 static_delta_ms, current_coverage, final_coverage,
                 current_change, final_change, self.static_fallback_count,
                 self.odom_fallback_count, self.sync_timeout_count))

    def _preprocessing_worker(self):
        last_started_version = 0
        while not rospy.is_shutdown():
            item = self.perception_slot.take_after(last_started_version, timeout=0.2)
            if item is None:
                continue
            last_started_version = item.version
            frame = item.payload
            with self.sync_condition:
                if not self.runtime_modes.is_current(frame.runtime_snapshot):
                    continue
            started_mono = time.monotonic()
            source_age_ms = elapsed_since_arrival_ms(
                frame.lidar_arrival_mono, started_mono)
            self._record_pipeline_value(
                'source_age_at_preprocess_ms', source_age_ms)
            if not source_age_within_limit(
                    frame.lidar_arrival_mono, self.output_max_age_ms, started_mono):
                self.stale_work_skip_count += 1
                continue
            self._record_pipeline_event('preprocess_started', started_mono)
            try:
                prepared = self._prepare_perception_frame(frame, started_mono)
            except Exception as error:
                rospy.logerr("OmniRisk dropped preprocessing frame %.6f: %s" %
                             (frame.stamp_ns / 1e9, error))
                continue
            if prepared is None:
                continue
            completed_mono = time.monotonic()
            self._record_pipeline_event('preprocess_completed', completed_mono)
            self._record_pipeline_value(
                'preprocess_time_ms', (completed_mono - started_mono) * 1000.0)
            if not source_age_within_limit(
                    prepared.lidar_arrival_mono, self.output_max_age_ms,
                    completed_mono):
                self.stale_work_skip_count += 1
                continue
            with self.sync_condition:
                if (rospy.is_shutdown() or
                        prepared.generation != self.sync_generation or
                        not self.runtime_modes.is_current(
                            prepared.runtime_snapshot)):
                    continue
                if self.inference_slot.publish(prepared):
                    self.inference_overwrite_count += 1

    @torch.inference_mode()
    def _prepare_perception_frame(self, frame, started_mono):
        stage_times_ms = {}
        stamp = frame.lidar.value.header.stamp
        odom_state = frame.odom_state
        t_wb = odom_state['position'].astype(np.float32)
        Rotation_wc, R_wb = self._planning_rotation(odom_state)
        R_yaw = Rotation_wc.astype(np.float32)

        stage_started = time.perf_counter()
        current = self._pointcloud_xyz(frame.lidar.value)
        stage_times_ms['pointcloud_decode_ms'] = (
            time.perf_counter() - stage_started) * 1000.0
        static_world = None
        static_range = None
        if self.static_input_mode == 'range_image':
            stage_started = time.perf_counter()
            static_range = decode_static_range_image(
                frame.static.value, self.lidar_H, self.lidar_W,
                self.lidar_max_dist, self.static_range_frame_id,
                expected_stamp_ns=frame.stamp_ns)
            stage_times_ms['static_range_decode_ms'] = (
                time.perf_counter() - stage_started) * 1000.0
        else:
            stage_started = time.perf_counter()
            static_world = self._pointcloud_xyz(frame.static.value)
            stage_times_ms['pointcloud_decode_ms'] += (
                time.perf_counter() - stage_started) * 1000.0
            stage_times_ms['static_range_decode_ms'] = 0.0
        if current.shape[0] == 0:
            return None
        stage_started = time.perf_counter()
        if self.config.get('lidar_points_in_world_frame', False):
            current_world = current
        else:
            current_world = current @ R_wb.T + t_wb
        current_planning = (current_world - t_wb) @ R_yaw
        stage_times_ms['current_transform_ms'] = (
            time.perf_counter() - stage_started) * 1000.0
        if self.static_input_mode == 'pointcloud':
            stage_started = time.perf_counter()
            static_planning = (static_world - t_wb) @ R_yaw
            stage_times_ms['static_transform_ms'] = (
                time.perf_counter() - stage_started) * 1000.0
        else:
            stage_times_ms['static_transform_ms'] = 0.0

        stage_started = time.perf_counter()
        current_range = self._project_to_range_image(current_planning)
        stage_times_ms['current_projection_ms'] = (
            time.perf_counter() - stage_started) * 1000.0
        if self.static_input_mode == 'pointcloud':
            stage_started = time.perf_counter()
            static_range = self._project_to_range_image(static_planning)
            stage_times_ms['static_projection_ms'] = (
                time.perf_counter() - stage_started) * 1000.0
        else:
            stage_times_ms['static_projection_ms'] = 0.0
        base_range = np.minimum(current_range, static_range)
        tracks = frame.tracks_at_ref
        ceiling_range = (self._virtual_ceiling_range(t_wb[2])
                         if self.virtual_ceiling_enable else None)
        stage_started = time.perf_counter()
        range_img_full, mask_full, velocity_full = rasterize_dynamic_spheres(
            tracks, t_wb, R_yaw, base_range, self.ray_directions, 0.1,
            self.lidar_max_dist, self.dynamic_sphere_radius_min,
            self.dynamic_sphere_radius_max, self.dynamic_occlusion_tolerance,
            visibility_limit=ceiling_range)
        stage_times_ms['dynamic_rasterization_ms'] = (
            time.perf_counter() - stage_started) * 1000.0
        stage_started = time.perf_counter()
        range_img = self._inpaint_range_image(range_img_full)
        stage_times_ms['depth_completion_ms'] = (
            time.perf_counter() - stage_started) * 1000.0
        range_img[range_img > self.lidar_depth_clip] = self.lidar_max_dist
        if ceiling_range is not None:
            range_img = np.minimum(range_img, ceiling_range)

        tensor_stage_ms = 0.0
        stage_started = time.perf_counter()
        crop = map_panorama_to_network(
            range_img, self.lidar_W, self.width, self.omnidirectional)
        depth_norm = np.minimum(crop, self.max_dis) / self.max_dis
        invalid_net = ~np.isfinite(depth_norm) | (depth_norm >= 1.0)
        tensor_stage_ms += (time.perf_counter() - stage_started) * 1000.0
        if invalid_net.any():
            completion_started = time.perf_counter()
            depth_norm = complete_depth(depth_norm, invalid_net, 1.0)
            stage_times_ms['depth_completion_ms'] += (
                time.perf_counter() - completion_started) * 1000.0
        depth_norm = np.ascontiguousarray(depth_norm, dtype=np.float32)
        stage_started = time.perf_counter()
        depth_input = torch.from_numpy(
            depth_norm.reshape(1, 1, self.height, self.width)).to(
                self.device, non_blocking=True)
        tensor_stage_ms += (time.perf_counter() - stage_started) * 1000.0

        stage_started = time.perf_counter()
        mask_input, vel_input = self._build_dynamic_inputs(
            mask_full, velocity_full)
        tensor_stage_ms += (time.perf_counter() - stage_started) * 1000.0
        stage_started = time.perf_counter()
        obs_norm = self._process_odom_at(
            odom_state, Rotation_wc, frame.runtime_snapshot.config).to(
            self.device, non_blocking=True)
        obs_input = self.state_transform.prepare_input(obs_norm).contiguous()
        tensor_stage_ms += (time.perf_counter() - stage_started) * 1000.0
        stage_times_ms['tensor_prep_h2d_ms'] = tensor_stage_ms
        self._record_pipeline_values(stage_times_ms)
        completed_mono = time.monotonic()
        return PreparedInference(
            sequence=frame.sequence,
            stamp_ns=frame.stamp_ns,
            stamp=stamp,
            generation=frame.generation,
            runtime_snapshot=frame.runtime_snapshot,
            depth_input=depth_input,
            mask_input=mask_input,
            velocity_input=vel_input,
            obs_input=obs_input,
            odom_state_at_ref=odom_state,
            rotation_wc=Rotation_wc,
            t_wb=t_wb,
            threat_detected=frame.threat_detected,
            range_img=range_img,
            current_range=current_range,
            range_img_full=range_img_full,
            depth_norm=depth_norm,
            current_valid=current_range < self.lidar_max_dist * 0.999,
            final_valid=range_img_full < self.lidar_max_dist * 0.999,
            static_delta_ms=frame.static_delta_ms,
            odom_delta_ms=frame.odom_delta_ms,
            lidar_arrival_mono=frame.lidar_arrival_mono,
            preprocess_started_mono=started_mono,
            preprocess_completed_mono=completed_mono)

    def _inference_worker(self):
        last_started_version = 0
        while not rospy.is_shutdown():
            item = self.inference_slot.take_after(last_started_version, timeout=0.2)
            if item is None:
                continue
            last_started_version = item.version
            prepared = item.payload
            with self.sync_condition:
                if not self.runtime_modes.is_current(
                        prepared.runtime_snapshot):
                    continue
            started_mono = time.monotonic()
            source_age_ms = elapsed_since_arrival_ms(
                prepared.lidar_arrival_mono, started_mono)
            self._record_pipeline_value(
                'source_age_at_inference_start_ms', source_age_ms)
            if not source_age_within_limit(
                    prepared.lidar_arrival_mono, self.output_max_age_ms,
                    started_mono):
                self.stale_work_skip_count += 1
                continue
            self._record_pipeline_event('inference_started', started_mono)
            try:
                self._run_prepared_inference(prepared, started_mono)
            except Exception as error:
                rospy.logerr("OmniRisk dropped inference frame %.6f: %s" %
                             (prepared.stamp_ns / 1e9, error))

    @torch.inference_mode()
    def _run_prepared_inference(self, prepared, inference_started_mono):
        runtime_config = prepared.runtime_snapshot.config
        endstate_pred, static_pred, dynamic_pred = forward_with_contiguous_inputs(
            self.policy,
            prepared.depth_input, prepared.mask_input,
            prepared.velocity_input, prepared.obs_input)
        endstate_pred = endstate_pred.cpu().numpy()
        static_pred = static_pred.cpu().numpy()
        dynamic_pred = dynamic_pred.cpu().numpy()
        inference_completed_mono = time.monotonic()
        self._record_pipeline_event('inference_completed', inference_completed_mono)
        self._record_pipeline_value(
            'inference_time_ms',
            (inference_completed_mono - inference_started_mono) * 1000.0)

        postprocess_started_mono = inference_completed_mono
        endstate, static_score, dynamic_score = self.process_output(
            endstate_pred, static_pred, dynamic_pred, return_all_preds=True,
            runtime_config=runtime_config)
        endstate_c = endstate.reshape(-1, 3, 3).transpose(0, 2, 1)
        endstate_w = np.matmul(prepared.rotation_wc, endstate_c)
        current_odom = self._current_odom_state()
        if current_odom is None:
            self._record_pipeline_value(
                'postprocess_time_ms',
                (time.monotonic() - postprocess_started_mono) * 1000.0)
            return
        start_pos = current_odom['position']
        start_vel = current_odom['velocity']
        start_acc = np.asarray(self.desire_acc).copy()
        target_pos = (endstate_w[:, :, 0]
                      + prepared.odom_state_at_ref['position'][None, :])
        score = (np.asarray(static_score).reshape(-1)
                 + runtime_config.fusion_lambda
                 * np.asarray(dynamic_score).reshape(-1))
        trajectory_installed = False
        with self.sync_condition:
            commit_mono = time.monotonic()
            if (not self.runtime_modes.is_current(prepared.runtime_snapshot) or
                    not result_is_committable(
                    prepared.generation, self.sync_generation, prepared.sequence,
                    self.last_committed_sequence, prepared.lidar_arrival_mono,
                    self.output_max_age_ms, now_mono=commit_mono,
                    shutting_down=rospy.is_shutdown())):
                if (prepared.generation == self.sync_generation and
                        prepared.sequence > self.last_committed_sequence and
                        elapsed_since_arrival_ms(
                            prepared.lidar_arrival_mono, commit_mono) > self.output_max_age_ms):
                    self.output_stale_drop_count += 1
                    rospy.logwarn_throttle(
                        1.0, "OmniRisk discarded stale inference output age=%.1fms limit=%.1fms" %
                        (elapsed_since_arrival_ms(
                            prepared.lidar_arrival_mono, commit_mono),
                         self.output_max_age_ms))
                self._record_pipeline_value(
                    'postprocess_time_ms',
                    (commit_mono - postprocess_started_mono) * 1000.0)
                return
            continuing_selection = (
                prepared.runtime_snapshot.mode == RuntimeMode.DODGE
                and (rospy.Time.now().to_sec() - self.last_threat_time) < self.hold_after)
            previous_action_id = self.last_action_id if continuing_selection else None
            selection_score = add_endpoint_continuity_cost(
                score, target_pos, previous_action_id,
                self.trajectory_continuity_weight)
            action_id = select_action_with_hysteresis(
                selection_score, previous_action_id,
                self.trajectory_switch_margin)
            optimal_poly = (
                Poly5Solver(
                    start_pos[0], start_vel[0], start_acc[0], target_pos[action_id, 0],
                    endstate_w[action_id, 0, 1], endstate_w[action_id, 0, 2],
                    runtime_config.trajectory_time),
                Poly5Solver(
                    start_pos[1], start_vel[1], start_acc[1], target_pos[action_id, 1],
                    endstate_w[action_id, 1, 1], endstate_w[action_id, 1, 2],
                    runtime_config.trajectory_time),
                Poly5Solver(
                    start_pos[2], start_vel[2], start_acc[2], target_pos[action_id, 2],
                    endstate_w[action_id, 2, 1], endstate_w[action_id, 2, 2],
                    runtime_config.trajectory_time))
            if mode_allows_trajectory_control(prepared.runtime_snapshot.mode):
                with self.lock:
                    self.optimal_poly_x, self.optimal_poly_y, self.optimal_poly_z = optimal_poly
                    self.active_trajectory_time = runtime_config.trajectory_time
                    self.active_trajectory_version = prepared.runtime_snapshot.version
                    self.active_trajectory_commit_mono = time.monotonic()
                    self.ctrl_time = 0.0
                    self.repeat_first_trajectory_sample = True
                    self.fallback_hover_target = None
                    trajectory_installed = True
            self.last_action_id = action_id if (
                prepared.runtime_snapshot.mode == RuntimeMode.DODGE
                and (continuing_selection or prepared.threat_detected)) else None
            self.last_committed_sequence = prepared.sequence
        postprocess_completed_mono = time.monotonic()
        self._record_pipeline_value(
            'postprocess_time_ms',
            (postprocess_completed_mono - postprocess_started_mono) * 1000.0)
        self._record_pipeline_event('trajectory_commit', postprocess_completed_mono)
        self._record_pipeline_value(
            'source_age_at_commit_ms',
            elapsed_since_arrival_ms(
                prepared.lidar_arrival_mono, postprocess_completed_mono))
        if trajectory_installed:
            self.control_pub(None)

        if any(pub.get_num_connections() > 0 for pub in (
                self.range_cloud_pub, self.mid360_fov_pub, self.best_traj_pub,
                self.lattice_traj_pub, self.all_trajs_pub,
                self.dynamic_score_trajs_pub, self.range_image_single_pub,
                self.range_image_pub, self.range_image_inpaint_pub,
                self.range_coverage_pub, self.depth_input_pub)):
            self._queue_visualization({
                'generation': prepared.generation,
                'runtime_snapshot': prepared.runtime_snapshot,
                'runtime_config': runtime_config,
                'range_img': prepared.range_img,
                'perception_stamp': prepared.stamp,
                'trajectory_stamp': current_odom['stamp'],
                'current_range': prepared.current_range,
                'range_img_full': prepared.range_img_full,
                'depth_norm': prepared.depth_norm,
                't_wb': prepared.t_wb.copy(),
                'static_score': np.asarray(static_score).copy(),
                'dynamic_score': np.asarray(dynamic_score).copy(),
                'endstate_w': endstate_w.copy(),
                'anchor_pos': prepared.odom_state_at_ref['position'].copy(),
                'start_pos': start_pos.copy(), 'start_vel': start_vel.copy(),
                'start_acc': start_acc, 'Rotation_wc': prepared.rotation_wc.copy(),
                'optimal_poly': optimal_poly,
            })
        self._record_sync_metrics(
            prepared.lidar_arrival_mono, prepared.current_valid, prepared.final_valid,
            prepared.static_delta_ms, prepared.odom_delta_ms)

    def _queue_visualization(self, payload):
        with self.visualization_condition:
            if self.visualization_pending is not None:
                self.visualization_overwrite_count += 1
            self.visualization_pending = payload
            self.visualization_condition.notify()

    def _publish_range_visualizations(self, payload):
        stamp = payload['perception_stamp']
        for publisher, key in (
                (self.range_image_single_pub, 'current_range'),
                (self.range_image_pub, 'range_img_full'),
                (self.range_image_inpaint_pub, 'range_img')):
            if publisher.get_num_connections() > 0:
                publisher.publish(self._make_range_image_msg(
                    payload[key], stamp, frame_id='odom'))
        if self.range_coverage_pub.get_num_connections() > 0:
            self.range_coverage_pub.publish(self._make_coverage_msg(
                payload['range_img_full'], stamp, frame_id='odom'))
        if self.depth_input_pub.get_num_connections() > 0:
            depth = np.ascontiguousarray(payload['depth_norm'], dtype=np.float32)
            msg = Image()
            msg.header.stamp, msg.header.frame_id = stamp, 'odom'
            msg.height, msg.width = depth.shape
            msg.encoding, msg.is_bigendian = '32FC1', False
            msg.step, msg.data = msg.width * depth.dtype.itemsize, depth.tobytes()
            self.depth_input_pub.publish(msg)

    def _visualization_worker(self):
        min_idle_sec = 1.0 / self.visualization_max_rate_hz
        next_allowed_mono = 0.0
        while not rospy.is_shutdown():
            with self.visualization_condition:
                while self.visualization_pending is None and not rospy.is_shutdown():
                    self.visualization_condition.wait(timeout=0.2)
                if rospy.is_shutdown():
                    return
                remaining = next_allowed_mono - time.monotonic()
                if remaining > 0.0:
                    self.visualization_condition.wait(timeout=remaining)
                    continue
                payload = self.visualization_pending
                self.visualization_pending = None
            try:
                if (payload['generation'] != self.sync_generation or
                        not self.runtime_modes.is_current(
                            payload['runtime_snapshot'])):
                    continue
                self._publish_range_visualizations(payload)
                if self.range_cloud_pub.get_num_connections() > 0:
                    pts_planning = self._backproject_range_image(payload['range_img'])
                    pts_world = (pts_planning @ payload['Rotation_wc'].T
                                 + payload['t_wb'][None, :])
                    self.range_cloud_pub.publish(self._make_xyz32_cloud(
                        pts_world, payload['perception_stamp'], 'world'))
                self._publish_mid360_fov(
                    payload['t_wb'], payload['Rotation_wc'],
                    payload['perception_stamp'])
                self.visualize_trajectory(
                    payload['static_score'], payload['dynamic_score'], payload['endstate_w'],
                    anchor_pos=payload['anchor_pos'], start_pos=payload['start_pos'],
                    start_vel=payload['start_vel'], start_acc=payload['start_acc'],
                    rotation_wc=payload['Rotation_wc'], optimal_poly=payload['optimal_poly'],
                    stamp=payload['trajectory_stamp'],
                    runtime_config=payload['runtime_config'])
                self._record_pipeline_event('visualization_publish')
            except Exception as error:
                rospy.logwarn_throttle(1.0, f"OmniRisk visualization dropped: {error}")
            finally:
                next_allowed_mono = time.monotonic() + min_idle_sec

    @torch.inference_mode()
    def callback_depth(self, data):
        if self.config.get('use_lidar', False) or not self.odom_init:
            return
        trajectory_installed = False
        with self.sync_condition:
            runtime_snapshot = self._select_runtime_snapshot_locked(False)
        runtime_config = runtime_snapshot.config

        # 1. Depth Image Process (Be careful with the depth units in your application)
        time0 = time.time()
        if data.encoding == "32FC1":    # Simulator, meter
            depth = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width)
        elif data.encoding == "16UC1":  # RealSense, millimeter
            depth = np.frombuffer(data.data, dtype=np.uint16).reshape(data.height, data.width).astype(np.float32) / 1000.0
        else:
            raise ValueError(f"Unsupported depth encoding: {data.encoding}. Expected '32FC1' or '16UC1'.")

        if depth.shape[0] != self.height or depth.shape[1] != self.width:
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth = np.minimum(depth, self.max_dis) / self.max_dis

        # interpolated the nan value (experiment shows that treating nan directly as 0 produces similar results)
        nan_mask = np.isnan(depth) | (depth < self.min_dis / self.max_dis)
        interpolated_image = cv2.inpaint(np.uint8(depth * 255), np.uint8(nan_mask), 1, cv2.INPAINT_TELEA)
        interpolated_image = interpolated_image.astype(np.float32) / 255.0
        depth = interpolated_image.reshape([1, 1, self.height, self.width])
        # cv2.imshow("1", depth[0][0])
        # cv2.waitKey(1)

        # 2. Network Inference
        # input prepare
        time1 = time.time()
        depth_input = torch.from_numpy(depth).to(self.device, non_blocking=True)  # (non_blocking: copying speed 3x)
        # the depth path has no M-detector mask/velocity image, feed zeros (static head unaffected; dynamic score unused here)
        mask_input = torch.zeros((1, 1, self.height, self.width), dtype=torch.float32, device=self.device)
        vel_input  = torch.zeros((1, 3, self.height, self.width), dtype=torch.float32, device=self.device)
        obs_norm = self.process_odom(
            runtime_config=runtime_config).to(self.device, non_blocking=True)
        obs_input = self.state_transform.prepare_input(obs_norm)
        # torch.cuda.synchronize()

        time2 = time.time()
        # Forward (TensorRT: inference speed increased by 5x) — two-head
        endstate_pred, static_pred, dynamic_pred = forward_with_contiguous_inputs(
            self.policy, depth_input, mask_input, vel_input, obs_input)
        endstate_pred = endstate_pred.cpu().numpy()
        static_pred   = static_pred.cpu().numpy()
        dynamic_pred  = dynamic_pred.cpu().numpy()
        time3 = time.time()

        # 3. Post-Processing (rank by static score; λ fusion deferred to stage 4)
        # Replacing PyTorch operation on CUDA with NumPy operation on CPU (speed increased by 10x)
        endstate, static_score, dynamic_score = self.process_output(
            endstate_pred, static_pred, dynamic_pred,
            return_all_preds=self.visualize, runtime_config=runtime_config)
        # Vectorization: transform the prediction(P V A in body frame) to the world frame with the attitude (without the position)
        endstate_c = endstate.reshape(-1, 3, 3).transpose(0, 2, 1)  # [N, 9] -> [N, 3, 3] -> [px vx ax, py vy ay, pz vz az]
        endstate_w = np.matmul(self.Rotation_wc, endstate_c)

        action_id = np.argmin(static_score) if self.visualize else 0
        with self.sync_condition:
            if not self.runtime_modes.is_current(runtime_snapshot):
                return
            with self.lock:  # keep mode/version validation and trajectory install atomic
                start_pos = self.desire_pos if self.plan_from_reference else np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
                start_vel = self.desire_vel if self.plan_from_reference else np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
                start_acc = self.desire_acc.copy()
                optimal_poly = (
                    Poly5Solver(start_pos[0], start_vel[0], start_acc[0], endstate_w[action_id, 0, 0] + start_pos[0],
                                endstate_w[action_id, 0, 1], endstate_w[action_id, 0, 2], runtime_config.trajectory_time),
                    Poly5Solver(start_pos[1], start_vel[1], start_acc[1], endstate_w[action_id, 1, 0] + start_pos[1],
                                endstate_w[action_id, 1, 1], endstate_w[action_id, 1, 2], runtime_config.trajectory_time),
                    Poly5Solver(start_pos[2], start_vel[2], start_acc[2], endstate_w[action_id, 2, 0] + start_pos[2],
                                endstate_w[action_id, 2, 1], endstate_w[action_id, 2, 2], runtime_config.trajectory_time))
                if mode_allows_trajectory_control(runtime_snapshot.mode):
                    self.optimal_poly_x, self.optimal_poly_y, self.optimal_poly_z = optimal_poly
                    self.active_trajectory_time = runtime_config.trajectory_time
                    self.active_trajectory_version = runtime_snapshot.version
                    self.active_trajectory_commit_mono = time.monotonic()
                    self.ctrl_time = 0.0
                    self.repeat_first_trajectory_sample = True
                    self.fallback_hover_target = None
                    trajectory_installed = True
        time4 = time.time()
        if trajectory_installed:
            self.control_pub(None)
        self.visualize_trajectory(
            static_score, dynamic_score, endstate_w,
            start_pos=start_pos, start_vel=start_vel, start_acc=start_acc,
            optimal_poly=optimal_poly, runtime_config=runtime_config)
        self.publish_depth_cloud_world(depth[0, 0], data.header.stamp)
        time5 = time.time()

        self.print_time(time0, time1, time2, time3, time4, time5)

    def _publish_hold(self):
        """Hover by sending a static home command. Must use EMPTY (not READY): NetworkControl treats READY as pure
        acceleration feedforward (no position feedback, so drift with des_acc=0 diverges upward); only non-READY uses
        the kx/kv position feedback in publishHoverSO3Command, which actually holds home."""
        msg = PositionCommand()
        msg.header.stamp = rospy.Time.now()
        msg.trajectory_flag = msg.TRAJECTORY_STATUS_EMPTY
        msg.position.x, msg.position.y, msg.position.z = self.home
        msg.yaw = self.last_yaw          # vel/acc/jerk default to 0
        msg.yaw_dot = 0.0
        self.desire_pos = self.home.copy()
        self.desire_vel = np.zeros(3)
        self.desire_acc = np.zeros(3)
        self.last_control_msg = msg
        with self.lock:
            self.fallback_hover_target = None
            self.control_mode = RuntimeMode.HOVER
        self.ctrl_pub.publish(msg)

    def _publish_rc_home(self):
        with self.odom_lock:
            current_z = float(self.odom.pose.pose.position.z)
        with self.rc_lock:
            if self.rc_home is None:
                return False
            if self.rc_home_z_ref is None:
                self.rc_home_z_ref = current_z
            self.rc_home_z_ref, velocity_z = advance_height_reference(
                self.rc_home_z_ref, self.rc_home[2],
                self.rc_return_speed, self.ctrl_dt)
            target = self.rc_home.copy()
            target[2] = self.rc_home_z_ref

        msg = PositionCommand()
        msg.header.stamp = rospy.Time.now()
        msg.trajectory_flag = msg.TRAJECTORY_STATUS_EMPTY
        msg.position.x, msg.position.y, msg.position.z = target
        msg.velocity.x = 0.0
        msg.velocity.y = 0.0
        msg.velocity.z = velocity_z
        msg.acceleration.x = msg.acceleration.y = msg.acceleration.z = 0.0
        msg.jerk.x = msg.jerk.y = msg.jerk.z = 0.0
        msg.yaw = self.last_yaw
        msg.yaw_dot = 0.0
        self.desire_pos = target
        self.desire_vel = np.array([0.0, 0.0, velocity_z])
        self.desire_acc = np.zeros(3)
        self.last_control_msg = msg
        with self.lock:
            self.fallback_hover_target = None
            self.control_mode = RuntimeMode.HOVER
        self.ctrl_pub.publish(msg)
        return True

    def _current_odom_hover_pose(self):
        with self.odom_lock:
            if not self.odom_init:
                return None
            position = self.odom.pose.pose.position
            quaternion = self.odom.pose.pose.orientation
            current_position = (position.x, position.y, position.z)
            current_yaw = np.arctan2(
                2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
                1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z))
        return current_position, current_yaw

    def _publish_fallback_hover_locked(self, current_position, current_yaw):
        self.fallback_hover_target = latch_hover_target(
            self.fallback_hover_target, current_position, current_yaw)
        target = self.fallback_hover_target
        msg = PositionCommand()
        msg.header.stamp = rospy.Time.now()
        msg.trajectory_flag = msg.TRAJECTORY_STATUS_EMPTY
        msg.position.x, msg.position.y, msg.position.z = target.position
        msg.velocity.x = msg.velocity.y = msg.velocity.z = 0.0
        msg.acceleration.x = msg.acceleration.y = msg.acceleration.z = 0.0
        msg.jerk.x = msg.jerk.y = msg.jerk.z = 0.0
        msg.yaw = target.yaw
        msg.yaw_dot = 0.0
        self.desire_pos = np.asarray(target.position)
        self.desire_vel = np.zeros(3)
        self.desire_acc = np.zeros(3)
        self.last_control_msg = msg
        self.control_mode = RuntimeMode.HOVER
        self.ctrl_pub.publish(msg)

    def control_pub(self, _timer):
        # Takeoff not complete: send no pos_cmd, let the controller takeoff_land state machine lift the drone to hover height
        if not self.home_set:
            return
        threat_active = ((rospy.Time.now().to_sec() - self.last_threat_time)
                         < self.hold_after)
        # RC centering ends the old trajectory first; real dynamic threats still take priority for planner avoidance.
        rc_home_required = self._rc_home_required()
        if rc_home_required:
            if threat_active:
                # Avoidance may change altitude; after the threat, regenerate a speed-limited reference from the actual altitude.
                with self.rc_lock:
                    self.rc_home_z_ref = None
            else:
                self._publish_rc_home()
                return
        # dodge mode locks home only when there is no external goal and no threat; waypoint/stick goals also go through the planner.
        # nav mode has no such gate and always runs the regular trajectory tracking below.
        if not self.nav_mode and not self.external_goal_received and not threat_active:
            self._publish_hold()
            return
        hover_pose = self._current_odom_hover_pose()
        if hover_pose is None:
            return
        now_mono = time.monotonic()
        with self.sync_condition:
            with self.lock:  # mode/version selection and control output are one atomic decision
                if self.ctrl_time is None:
                    next_ctrl_time = next_clock_time = None
                else:
                    next_ctrl_time, next_clock_time = next_trajectory_sample(
                        self.ctrl_time, self.ctrl_dt,
                        self.repeat_first_trajectory_sample)
                if not trajectory_is_executable(
                        self.runtime_modes.mode, self.runtime_modes.version,
                        self.active_trajectory_version, next_ctrl_time,
                        self.active_trajectory_time,
                        self.active_trajectory_commit_mono, now_mono,
                        self.trajectory_refresh_timeout):
                    self._publish_fallback_hover_locked(*hover_pose)
                    return
                # dodge threats take priority over centered hover; arrive/EMPTY takes over after the threat window.
                if (self.arrive and self.last_control_msg is not None and
                        not threat_active):
                    self.desire_init = False   # ready for next rollout
                    self.last_control_msg.header.stamp = rospy.Time.now()
                    self.last_control_msg.trajectory_flag = self.last_control_msg.TRAJECTORY_STATUS_EMPTY
                    self.fallback_hover_target = None
                    self.control_mode = RuntimeMode.HOVER
                    self.ctrl_pub.publish(self.last_control_msg)
                    return
                self.ctrl_time = next_clock_time
                self.repeat_first_trajectory_sample = False
                control_msg = PositionCommand()
                control_msg.header.stamp = rospy.Time.now()
                control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_READY
                control_msg.position.x = self.optimal_poly_x.get_position(next_ctrl_time)
                control_msg.position.y = self.optimal_poly_y.get_position(next_ctrl_time)
                control_msg.position.z = self.optimal_poly_z.get_position(next_ctrl_time)
                control_msg.velocity.x = self.optimal_poly_x.get_velocity(next_ctrl_time)
                control_msg.velocity.y = self.optimal_poly_y.get_velocity(next_ctrl_time)
                control_msg.velocity.z = self.optimal_poly_z.get_velocity(next_ctrl_time)
                control_msg.acceleration.x = self.optimal_poly_x.get_acceleration(next_ctrl_time)
                control_msg.acceleration.y = self.optimal_poly_y.get_acceleration(next_ctrl_time)
                control_msg.acceleration.z = self.optimal_poly_z.get_acceleration(next_ctrl_time)
                control_msg.jerk.x = self.optimal_poly_x.get_jerk(next_ctrl_time)
                control_msg.jerk.y = self.optimal_poly_y.get_jerk(next_ctrl_time)
                control_msg.jerk.z = self.optimal_poly_z.get_jerk(next_ctrl_time)
                self.desire_pos = np.array([control_msg.position.x, control_msg.position.y, control_msg.position.z])
                self.desire_vel = np.array([control_msg.velocity.x, control_msg.velocity.y, control_msg.velocity.z])
                self.desire_acc = np.array([control_msg.acceleration.x, control_msg.acceleration.y, control_msg.acceleration.z])
                if self.omnidirectional:
                    # free yaw: heading does not follow velocity/goal, keeps a fixed orientation (omnidirectional LiDAR does not depend on heading).
                    control_msg.yaw = self.last_yaw
                    control_msg.yaw_dot = 0.0
                else:
                    goal_dir = self.goal - self.desire_pos
                    yaw, yaw_dot = calculate_yaw(self.desire_vel, goal_dir, self.last_yaw, self.ctrl_dt)
                    self.last_yaw = yaw
                    control_msg.yaw = yaw
                    control_msg.yaw_dot = yaw_dot
                self.desire_init = True
                self.last_control_msg = control_msg
                self.control_mode = self.runtime_modes.mode
                self.ctrl_pub.publish(control_msg)

    def process_output(self, endstate_pred, static_pred, dynamic_pred,
                       return_all_preds=False, runtime_config=None):
        if runtime_config is None:
            runtime_config = self.runtime_profiles[RuntimeMode.DODGE]
        endstate_pred = endstate_pred.reshape(9, self.lattice_primitive.traj_num).T
        static_pred   = static_pred.reshape(self.lattice_primitive.traj_num)
        dynamic_pred  = dynamic_pred.reshape(self.lattice_primitive.traj_num)

        # grid → lattice_id mapping: must match the training-side state_transform.lattice_reorder
        # (omni flips vertical only; legacy path reverses fully), otherwise inference picks the primitive at the mirrored azimuth.
        reorder = self.state_transform.lattice_reorder
        if not return_all_preds:
            action_id = int(np.argmin(static_pred))
            lattice_id = int(reorder[action_id])
            endstate = self.state_transform.pred_to_endstate_cpu(
                endstate_pred[action_id, :][np.newaxis, :], lattice_id,
                runtime_config.radio_range, runtime_config.vel_max,
                runtime_config.acc_max)
            static_score  = static_pred[action_id]
            dynamic_score = dynamic_pred[action_id]
        else:
            static_score  = static_pred
            dynamic_score = dynamic_pred
            endstate = self.state_transform.pred_to_endstate_cpu(
                endstate_pred, reorder, runtime_config.radio_range,
                runtime_config.vel_max, runtime_config.acc_max)

        return endstate, static_score, dynamic_score

    def publish_depth_cloud_world(self, depth_norm, stamp):
        """Convert depth image to point cloud in world frame and publish."""
        if self.depth_cloud_pub.get_num_connections() == 0:
            return
        s = self.depth_vis_stride
        h, w = depth_norm.shape
        # subsampled pixel grid in camera frame
        us = np.arange(0, w, s)
        vs = np.arange(0, h, s)
        uu, vv = np.meshgrid(us, vs)  # (H', W')
        d = depth_norm[vv, uu] * self.max_dis  # restore metric depth (m)
        valid = (d > self.min_dis) & (d < self.max_dis)
        d = d[valid]
        uu = uu[valid].astype(np.float32)
        vv = vv[valid].astype(np.float32)
        # back-project to OpenCV camera frame (Z forward, X right, Y down)
        x_cv = (uu - self.cam_cx) / self.cam_fx * d
        y_cv = (vv - self.cam_cy) / self.cam_fy * d
        z_cv = d
        # convert to FLU body/camera frame (X forward, Y left, Z up) used by the codebase
        pts_c = np.stack([z_cv, -x_cv, -y_cv], axis=1)  # (N, 3)
        # transform to world frame: p_w = R_wc @ p_c + pos_w
        pos_w = np.array([self.odom.pose.pose.position.x,
                          self.odom.pose.pose.position.y,
                          self.odom.pose.pose.position.z])
        pts_w = pts_c @ self.Rotation_wc.T + pos_w  # (N, 3)
        header = std_msgs.msg.Header()
        header.stamp = stamp
        header.frame_id = 'world'
        self.depth_cloud_pub.publish(point_cloud2.create_cloud_xyz32(header, pts_w.astype(np.float32)))

    def visualize_trajectory(self, static_score, dynamic_score, pred_endstate,
                             anchor_pos=None, start_pos=None, start_vel=None,
                             start_acc=None, rotation_wc=None, optimal_poly=None,
                             stamp=None, runtime_config=None):
        if runtime_config is None:
            runtime_config = self.runtime_profiles[RuntimeMode.DODGE]
        trajectory_time = runtime_config.trajectory_time
        dt = trajectory_time / 20.0
        if start_pos is None:
            start_pos = (self.desire_pos if self.plan_from_reference else
                         np.array((self.odom.pose.pose.position.x,
                                   self.odom.pose.pose.position.y,
                                   self.odom.pose.pose.position.z)))
        if start_vel is None:
            start_vel = (self.desire_vel if self.plan_from_reference else
                         np.array((self.odom.twist.twist.linear.x,
                                   self.odom.twist.twist.linear.y,
                                   self.odom.twist.twist.linear.z)))
        if start_acc is None:
            start_acc = self.desire_acc
        if anchor_pos is None:
            anchor_pos = start_pos
        if rotation_wc is None:
            rotation_wc = self.Rotation_wc
        if optimal_poly is None:
            optimal_poly = (self.optimal_poly_x, self.optimal_poly_y, self.optimal_poly_z)
        if stamp is None:
            stamp = rospy.Time.now()
        # best predicted trajectory
        if self.best_traj_pub.get_num_connections() > 0:
            t_values = np.arange(0, trajectory_time, dt)
            points_array = np.stack((
                optimal_poly[0].get_position(t_values),
                optimal_poly[1].get_position(t_values),
                optimal_poly[2].get_position(t_values)
            ), axis=-1)
            header = std_msgs.msg.Header()
            header.stamp = stamp
            header.frame_id = 'world'
            point_cloud_msg = point_cloud2.create_cloud_xyz32(header, points_array)
            self.best_traj_pub.publish(point_cloud_msg)
        # lattice primitive
        if self.visualize and self.lattice_traj_pub.get_num_connections() > 0:
            lattice_endstate = self.lattice_primitive.lattice_pos_node.cpu().numpy()
            lattice_endstate = lattice_endstate * (
                runtime_config.radio_range / self.lattice_primitive.radio_range)
            lattice_endstate = np.dot(lattice_endstate, rotation_wc.T)
            zero_state = np.zeros_like(lattice_endstate)
            lattice_poly_x = Polys5Solver(start_pos[0], start_vel[0], start_acc[0],
                                          lattice_endstate[:, 0] + anchor_pos[0], zero_state[:, 0], zero_state[:, 0], trajectory_time)
            lattice_poly_y = Polys5Solver(start_pos[1], start_vel[1], start_acc[1],
                                          lattice_endstate[:, 1] + anchor_pos[1], zero_state[:, 1], zero_state[:, 1], trajectory_time)
            lattice_poly_z = Polys5Solver(start_pos[2], start_vel[2], start_acc[2],
                                          lattice_endstate[:, 2] + anchor_pos[2], zero_state[:, 2], zero_state[:, 2], trajectory_time)
            t_values = np.arange(0, trajectory_time, dt)
            points_array = np.stack((
                lattice_poly_x.get_position(t_values),
                lattice_poly_y.get_position(t_values),
                lattice_poly_z.get_position(t_values)
            ), axis=-1)
            header = std_msgs.msg.Header()
            header.stamp = stamp
            header.frame_id = 'world'
            point_cloud_msg = point_cloud2.create_cloud_xyz32(header, points_array)
            self.lattice_traj_pub.publish(point_cloud_msg)
        # all predicted trajectories
        publish_score_cloud = self.all_trajs_pub.get_num_connections() > 0
        publish_dynamic_markers = self.dynamic_score_trajs_pub.get_num_connections() > 0
        if self.visualize and (publish_score_cloud or publish_dynamic_markers):
            all_poly_x = Polys5Solver(start_pos[0], start_vel[0], start_acc[0],
                                      pred_endstate[:, 0, 0] + anchor_pos[0], pred_endstate[:, 0, 1], pred_endstate[:, 0, 2], trajectory_time)
            all_poly_y = Polys5Solver(start_pos[1], start_vel[1], start_acc[1],
                                      pred_endstate[:, 1, 0] + anchor_pos[1], pred_endstate[:, 1, 1], pred_endstate[:, 1, 2], trajectory_time)
            all_poly_z = Polys5Solver(start_pos[2], start_vel[2], start_acc[2],
                                      pred_endstate[:, 2, 0] + anchor_pos[2], pred_endstate[:, 2, 1], pred_endstate[:, 2, 2], trajectory_time)
            t_values = np.arange(0, trajectory_time, dt)
            points_array = np.stack((
                all_poly_x.get_position(t_values),
                all_poly_y.get_position(t_values),
                all_poly_z.get_position(t_values)
            ), axis=-1)
            if publish_dynamic_markers:
                trajectories = points_array.reshape(-1, t_values.size, 3)
                self._publish_dynamic_score_trajectories(
                    trajectories, dynamic_score, start_pos, stamp)
            if publish_score_cloud:
                # One batch of trajectories carries 3 scalar channels; switch Channel Name in rviz to view each:
                # intensity (= static head) / dynamic (dynamic head) / total (fused criterion static+λ·dynamic)
                static_col  = np.repeat(np.asarray(static_score).reshape(-1),  t_values.size)
                dynamic_col = np.repeat(np.asarray(dynamic_score).reshape(-1), t_values.size)
                total_col = (static_col
                             + runtime_config.fusion_lambda * dynamic_col)
                score_points = np.column_stack(
                    (points_array, static_col, dynamic_col, total_col))
                header = std_msgs.msg.Header()
                header.stamp = stamp
                header.frame_id = 'world'
                fields = [PointField('x', 0, PointField.FLOAT32, 1), PointField('y', 4, PointField.FLOAT32, 1),
                          PointField('z', 8, PointField.FLOAT32, 1), PointField('intensity', 12, PointField.FLOAT32, 1),
                          PointField('dynamic', 16, PointField.FLOAT32, 1), PointField('total', 20, PointField.FLOAT32, 1)]
                point_cloud_msg = point_cloud2.create_cloud(header, fields, score_points)
                self.all_trajs_pub.publish(point_cloud_msg)

    def _publish_dynamic_score_trajectories(
            self, trajectories, dynamic_score, start_pos, stamp):
        """Publish continuous candidate curves colored by the dynamic-head score."""
        scores = np.asarray(dynamic_score, dtype=np.float32).reshape(-1)
        normalized, score_min, score_max = normalize_scores(scores)
        colors = score_colors(normalized)
        best_id = int(np.argmin(scores))
        markers = []

        def color_msg(rgb, alpha=1.0):
            color = std_msgs.msg.ColorRGBA()
            color.r, color.g, color.b = (float(value) for value in rgb)
            color.a = alpha
            return color

        for trajectory_id, (trajectory, rgb) in enumerate(zip(trajectories, colors)):
            line = Marker()
            line.header.stamp, line.header.frame_id = stamp, 'world'
            line.ns, line.id = 'dynamic_score_trajectories', trajectory_id
            line.type, line.action = Marker.LINE_STRIP, Marker.ADD
            line.pose.orientation.w = 1.0
            line.scale.x = 0.09 if trajectory_id == best_id else 0.045
            line.color = color_msg(rgb, 1.0 if trajectory_id == best_id else 0.82)
            line.points = [Point(*point) for point in trajectory]
            markers.append(line)

        best_endpoint = trajectories[best_id, -1]
        best_marker = Marker()
        best_marker.header.stamp, best_marker.header.frame_id = stamp, 'world'
        best_marker.ns, best_marker.id = 'dynamic_score_best', 0
        best_marker.type, best_marker.action = Marker.SPHERE, Marker.ADD
        best_marker.pose.position = Point(*best_endpoint)
        best_marker.pose.orientation.w = 1.0
        best_marker.scale.x = best_marker.scale.y = best_marker.scale.z = 0.18
        best_marker.color = color_msg(colors[best_id])
        markers.append(best_marker)

        best_text = Marker()
        best_text.header.stamp, best_text.header.frame_id = stamp, 'world'
        best_text.ns, best_text.id = 'dynamic_score_best', 1
        best_text.type, best_text.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        best_text.pose.position = Point(
            best_endpoint[0], best_endpoint[1], best_endpoint[2] + 0.28)
        best_text.pose.orientation.w = 1.0
        best_text.scale.z = 0.24
        best_text.color = color_msg(colors[best_id])
        best_text.text = f'LOWEST DYNAMIC  {scores[best_id]:.3f}'
        markers.append(best_text)

        legend_center = np.asarray(start_pos, dtype=np.float32).copy()
        legend_center[2] += 1.35
        legend_values = np.linspace(0.0, 1.0, 17, dtype=np.float32)
        legend_colors = score_colors(legend_values)
        legend = Marker()
        legend.header.stamp, legend.header.frame_id = stamp, 'world'
        legend.ns, legend.id = 'dynamic_score_legend', 0
        legend.type, legend.action = Marker.LINE_STRIP, Marker.ADD
        legend.pose.orientation.w = 1.0
        legend.scale.x = 0.10
        legend.points = [Point(legend_center[0] + offset, legend_center[1],
                               legend_center[2])
                         for offset in np.linspace(-0.8, 0.8, legend_values.size)]
        legend.colors = [color_msg(rgb) for rgb in legend_colors]
        markers.append(legend)

        legend_title = Marker()
        legend_title.header.stamp, legend_title.header.frame_id = stamp, 'world'
        legend_title.ns, legend_title.id = 'dynamic_score_legend', 1
        legend_title.type, legend_title.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        legend_title.pose.position = Point(
            legend_center[0], legend_center[1], legend_center[2] + 0.25)
        legend_title.pose.orientation.w = 1.0
        legend_title.scale.z = 0.25
        legend_title.color = color_msg((1.0, 1.0, 1.0))
        legend_title.text = 'DYNAMIC SCORE  (LOWER = SAFER)'
        markers.append(legend_title)

        score_mid = 0.5 * (score_min + score_max)
        for marker_id, offset, value, label, rgb in (
                (2, -0.8, score_min, 'LOW', legend_colors[0]),
                (3, 0.0, score_mid, 'MID', legend_colors[len(legend_colors) // 2]),
                (4, 0.8, score_max, 'HIGH', legend_colors[-1])):
            label_marker = Marker()
            label_marker.header.stamp, label_marker.header.frame_id = stamp, 'world'
            label_marker.ns, label_marker.id = 'dynamic_score_legend', marker_id
            label_marker.type, label_marker.action = Marker.TEXT_VIEW_FACING, Marker.ADD
            label_marker.pose.position = Point(
                legend_center[0] + offset, legend_center[1], legend_center[2] - 0.22)
            label_marker.pose.orientation.w = 1.0
            label_marker.scale.z = 0.20
            label_marker.color = color_msg(rgb)
            label_marker.text = f'{label} {value:.3f}'
            markers.append(label_marker)

        self.dynamic_score_trajs_pub.publish(MarkerArray(markers=markers))

    def print_time(self, time0, time1, time2, time3, time4, time5):
        """
        Performance reference: PyTorch model should take < 5 ms; TensorRT model should take < 1 ms

        Notes:
        - Running program and enabling RViz under WSL greatly increase processing time, and Ubuntu does not have these issues
        - LiDAR processing is latest-only; an overrun lowers output frequency instead of queuing old frames
        """
        self.time_interpolation = self.time_interpolation + (time1 - time0)
        self.time_prepare = self.time_prepare + (time2 - time1)
        self.time_forward = self.time_forward + (time3 - time2)
        self.time_process = self.time_process + (time4 - time3)
        self.time_visualize = self.time_visualize + (time5 - time4)
        self.count = self.count + 1

        total_time = (time5 - time0) * 1000
        tolerance = 1000.0 / self.depth_fps
        if total_time > tolerance:
            rospy.logwarn(f"Warn: Processing time {(time5 - time0) * 1000:.2f} ms exceeds {tolerance:.2f} ms; latest-only output rate will decrease")
            print(f"\033[34mCurrent Time Consuming:\033[0m "
                  f"depth-interpolation: \033[32m{1000 * (time1 - time0):.2f} ms\033[0m; "
                  f"data-prepare: \033[32m{1000 * (time2 - time1):.2f} ms\033[0m; "
                  f"network-inference: \033[32m{1000 * (time3 - time2):.2f} ms\033[0m; "
                  f"post-process: \033[32m{1000 * (time4 - time3):.2f} ms\033[0m; "
                  f"visualize-trajectory: \033[32m{1000 * (time5 - time4):.2f} ms\033[0m")
        if self.verbose or (total_time > tolerance):
            print(f"\033[34mAverage Time Consuming:\033[0m "
                  f"depth-interpolation: \033[32m{1000 * self.time_interpolation / self.count:.2f} ms\033[0m; "
                  f"data-prepare: \033[32m{1000 * self.time_prepare / self.count:.2f} ms\033[0m; "
                  f"network-inference: \033[32m{1000 * self.time_forward / self.count:.2f} ms\033[0m; "
                  f"post-process: \033[32m{1000 * self.time_process / self.count:.2f} ms\033[0m; "
                  f"visualize-trajectory: \033[32m{1000 * self.time_visualize / self.count:.2f} ms\033[0m")

    def warm_up(self):
        # two-head: front depth (static) + mask + velocity image (dynamic)
        depth   = torch.zeros((1, 1, self.height, self.width), dtype=torch.float32, device=self.device)
        mask    = torch.zeros((1, 1, self.height, self.width), dtype=torch.float32, device=self.device)
        vel_img = torch.zeros((1, 3, self.height, self.width), dtype=torch.float32, device=self.device)
        obs     = torch.zeros((1, 9), dtype=torch.float32, device=self.device)
        obs     = self.state_transform.prepare_input(obs)
        endstate_pred, _, _ = forward_with_contiguous_inputs(
            self.policy, depth, mask, vel_img, obs)
        _ = self.state_transform.pred_to_endstate(endstate_pred)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_tensorrt", type=int, default=0, help="use tensorrt or not")
    parser.add_argument("--trial", type=int, default=1, help="trial number")
    parser.add_argument("--epoch", type=int, default=50, help="epoch number")
    parser.add_argument("--weight_path", type=str, default=None,
                         help="explicit checkpoint path, overrides --trial/--epoch")
    parser.add_argument("--flight_mode", type=str, default=None, choices=["dodge", "nav"],
                         help="override the default flight_mode in settings")
    parser.add_argument("--tracks_topic", type=str, default=None,
                         help="override the default /m_detector/confirmed_tracks")
    parser.add_argument("--odom_topic", type=str, default=None,
                         help="override the default /ekf_quat/ekf_odom; use /sim/odom in simulation")
    parser.add_argument("--lidar_topic", type=str, default=None,
                         help="override the default /cloud_registered_no_point_filter; use /lidar_points in simulation"
                              " (frame sync is driven by this topic; change it for simulation in range_image mode too, otherwise inference never runs)")
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if args.weight_path:
        weight = args.weight_path
    else:
        weight = "omnirisk_trt.pth" if args.use_tensorrt else base_dir + "/saved/run_{}/epoch{}.pth".format(args.trial, args.epoch)
    print("load weight from:", weight)

    settings = {'use_tensorrt': args.use_tensorrt,
                'flight_mode': 'dodge',  # 'dodge' = in-place dodge demo | 'nav' = navigation with avoidance (cruise to goal)
                'goal': [0, 0, 0.5],      # goal position (dodge: overwritten by home at startup; nav: navigation goal, or overridden by /move_base_simple/goal)
                'external_goal_arrive_radius': 2.0, # arrive radius (m) in both modes after receiving a waypoint/stick goal
                # ---- Dodge demo (hover by default + threat-gated planner takeover) ----
                # if arrive_radius is unset, the default follows flight_mode (dodge=0.0 never arrives, nav=2.0)
                'threat_radius': 6.0,     # a confirmed track within this distance (m) is a threat → DODGE
                                          # constraint: mid360.yaml tracker/target_max_xy_distance (currently 7.0) ≥ threat_radius + 1m margin; recheck when changing this
                'dodge_hold_after': 0.2,  # keep dodging for this long (s) after the threat is gone, then hover at home
                'takeoff_alt_min': 0.5,   # take over once odom z exceeds this (m) and vz settles (home z is fixed to hover_height; this only sets timing)
                'vz_settle': 0.1,         # vertical speed below this (m/s) counts as stable hover
                'auto_takeoff': True,     # auto arm + takeoff after startup (False: call the takeoff service manually)
                'hover_height': 0.6,      # common flight height (m), must be > takeoff_alt_min; px4ctrl takeoff_height must match on the real vehicle
                'takeoff_service': '/network_controller_node/takeoff_land',
                'rc_active_topic': '/rc_goal_active', # stick active/centered state published by the RC node
                'rc_state_timeout': 0.6,  # treat as centered after this timeout (s)
                'rc_return_speed': 0.2,   # max vertical speed (m/s) when returning to hover_height after centering
                'pitch_angle_deg': -0,   # camera pitch (up is negative)
                # 'odom_topic': '/sim/odom',                         # odometry topic (simulation)
                'odom_topic': '/ekf_quat/ekf_odom',                  # high-rate control-grade EKF odometry
                # 'depth_topic': '/depth_image',                    # depth image topic (simulation, used when use_lidar=False)
                'depth_topic': '/camera/depth/image_rect_raw',      # depth image topic (real vehicle, used when use_lidar=False)
                'ctrl_topic': '/so3_control/pos_cmd',               # controller topic
                'plan_from_reference': False,
                'verbose': False,               # print timing?
                'visualize': False,         # off by default in flight; enable explicitly for debugging so it does not starve the critical path
                # camera intrinsics for depth point cloud (world frame) visualization
                'cam_fx': 98.92,
                'cam_fy': 79.14,
                'cam_cx': 81.0,
                'cam_cy': 47.53,
                'depth_vis_stride': 4,
                # ---- LiDAR input ----
                'use_lidar': True,                      # True: LiDAR inference; False: depth camera
                # 'lidar_topic': '/lidar_points',                    # simulated point cloud topic
                'lidar_topic': '/cloud_registered_no_point_filter',  # FAST-LIO online world-frame point cloud on the real vehicle
                'lidar_points_in_world_frame': True,                 # already in world frame, do not transform again
                'lidar_H': 96,                          # range image height (matches image_height)
                'lidar_W': 360,                         # range image width (360°, 1°/column)
                'lidar_elev_min_deg': -7.0,             # vertical FOV lower limit (MID-360 actual range)
                'lidar_elev_max_deg': 52.0,             # vertical FOV upper limit (MID-360 actual range)
                'lidar_max_dist': 20.0,                 # max range (m), matches training data
                'lidar_depth_clip': 20.0,                # inference depth cutoff (m): returns beyond this are treated as infinite (normalization still uses lidar_max_dist to match training)
                'sync_cache_duration': 0.25,             # short per-stamp cache for each input, not a processing queue
                'sync_timeout': 0.15,                    # bounded wait for a complete perception frame; skipped explicitly on timeout
                'static_fallback_tolerance': 0.06,       # pointcloud mode only; range image mode never falls back across frames
                'odom_sync_tolerance': 0.03,             # max one-sided time difference for odom interpolation/nearest value
                'output_age_warn_ms': 150.0,             # P95 local end-to-end latency warning threshold
                'output_max_age_ms': 120.0,              # inference results older than this local end-to-end age are not committed
                'tracks_topic': '/m_detector/confirmed_tracks',
                'static_input_mode': 'pointcloud',       # subscribe to the M-detector static cloud and project it into a range image on the planner side
                'static_range_topic': '/m_detector/static_range',
                'static_range_pose_topic': '/m_detector/static_range_pose',
                'static_range_frame_id': 'world',
                'static_topic': '/m_detector/std_points',# pointcloud mode input (world frame)
                # prediction horizon is read directly from /dyn_obj/tracker/max_prediction_horizon
                'track_ballistic_model': True,
                'track_gravity_world': [0.0, 0.0, -9.81],
                'dynamic_sphere_radius_min': 0.10,
                'dynamic_sphere_radius_max': 0.20,
                'dynamic_occlusion_tolerance': 0.15,
                'pipeline_metrics_period': 5.0,          # log period for pipeline rates, latency and coverage counts
                'visualization_max_rate_hz': 15.0,       # cap for the independent latest-only visualization worker, does not slow the control path
                # Virtual ceiling: after inpainting, overlay a world z=ceiling plane to fill the space above and discourage climbing
                'virtual_ceiling_enable': False,
                'virtual_ceiling_z': 5,               # absolute world height (m), adjust to the takeoff site ground height
                }
    if args.flight_mode is not None:
        settings['flight_mode'] = args.flight_mode
    if args.tracks_topic is not None:
        settings['tracks_topic'] = args.tracks_topic
    if args.odom_topic is not None:
        settings['odom_topic'] = args.odom_topic
    if args.lidar_topic is not None:
        settings['lidar_topic'] = args.lidar_topic
    OmniRiskNode(settings, weight)
