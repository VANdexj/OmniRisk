#ifndef M_DETECTOR_FRAME_TRACKER_H
#define M_DETECTOR_FRAME_TRACKER_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <ros/node_handle.h>
#include <std_msgs/Header.h>

#include <m-detector/Detection3D.h>
#include <types.h>

namespace m_detector
{

struct TrackerConfig
{
    bool enabled = false;
    std::string tracks_topic = "/m_detector/confirmed_tracks";
    std::string tentative_tracks_topic = "/m_detector/tentative_tracks";
    std::string predicted_tracks_topic = "/m_detector/predicted_tracks";
    std::string markers_topic = "/m_detector/confirmed_track_markers";
    double max_prediction_horizon = 0.15;
    double lost_track_prediction_time = 0.30;

    bool point_out_dynamic_support_en = true;
    double point_out_duplicate_distance = 0.25;

    double cluster_tolerance = 0.8;
    int cluster_min_points_process = 2;
    int cluster_min_points = 6;
    int cluster_max_points = 100000;

    int min_init_hits = 3;
    int init_window_size = 5;
    bool near_range_fast_confirm_en = false;
    double near_range_fast_confirm_distance = 1.0;
    int near_range_fast_confirm_hits = 2;
    int track_lifetime_frames = 3;
    int track_history_size = 20;

    double min_iou = 0.05;
    double max_center_distance = 2.5;
    double association_padding = 0.25;
    double association_mahalanobis_gate_sq = 11.34;
    bool duplicate_guard_en = false;
    double duplicate_guard_distance = 0.25;
    double duplicate_guard_bbox_gap = 0.05;
    double default_dt = 0.1;

    // When enabled, gravity_world is a known acceleration input to the 6D CV state.
    bool ballistic_model_en = false;
    double gravity_world_x = 0.0;
    double gravity_world_y = 0.0;
    double gravity_world_z = -9.81;

    double process_acceleration_stddev = 10.0;
    double dynamic_position_variance = 0.1;
    double max_target_speed = 8.0;
    double max_target_acceleration = 10.0;

    bool bbox_size_prior_en = false;
    double bbox_size_prior_min = 0.15;
    double bbox_size_prior_max = 0.45;

    bool target_distance_filter_en = false;
    double target_max_xy_distance = 5.0;
    double target_max_z_distance = 1.5;
    bool target_origin_bbox_filter_en = false;
    double target_origin_bbox_min_x = -6.0;
    double target_origin_bbox_max_x = 6.0;
    double target_origin_bbox_min_y = -6.0;
    double target_origin_bbox_max_y = 6.0;
    double target_origin_bbox_min_z = -0.5;
    double target_origin_bbox_max_z = 2.5;
    bool target_ground_filter_en = false;
    double target_min_world_z = 0.1;
    bool target_volume_filter_en = false;
    double target_min_bbox_volume = 0.0;
    double target_max_bbox_volume = 0.4 * 0.4 * 0.4;
    bool target_speed_filter_en = false;
    double target_min_speed = 0.1;
    double target_speed_filter_grace_time = 1.0;

    bool static_support_en = false;
    double static_support_hold_time = 1.0;
    double static_support_bbox_expand_ratio = 0.5;
    double static_support_bbox_padding = 0.05;
    int static_support_min_points = 3;
    double static_support_cluster_tolerance = 0.35;
    int static_support_cluster_max_points = 300;
    double static_support_max_residual = 0.25;
    double static_support_position_variance = 0.5;
    double static_support_min_activation_speed = 0.3;

    bool debug_track_lifecycle = false;
    bool debug_association_rejects = false;
    bool debug_print_detections = false;
    bool debug_print_association_matrix = false;
    bool debug_bbox_prior_rejects = false;
    bool debug_distance_rejects = false;
    bool debug_static_support = false;

    double marker_bbox_scale = 1.0;
    double marker_bbox_min_size = 0.05;
    double marker_velocity_arrow_scale = 1.0;
    double marker_velocity_min_speed = 0.05;
    double marker_velocity_max_arrow_length = 3.0;
    double marker_velocity_text_size = 0.35;
};

TrackerConfig LoadTrackerConfig(const ros::NodeHandle &node,
                                const std::string &parameter_prefix = "dyn_obj/tracker");

enum class TrackerInputMode : uint8_t
{
    FRAME_OUT = 1,
    POINT_OUT_ONLY = 2,
    PRECLUSTERED_DETECTIONS = 3
};

struct TrackerFrame
{
    std_msgs::Header header;
    PointCloudXYZI::ConstPtr frame_out;
    PointCloudXYZI::ConstPtr point_out;
    PointCloudXYZI::ConstPtr support_cloud;
    std::optional<std::vector<Detection3D>> preclustered_detections;
    Eigen::Vector3f odom_position = Eigen::Vector3f::Zero();
    bool has_odom = false;
    bool frame_out_available = false;
    double cloud_odom_stamp_delta_ms = 0.0;
};

struct TrackSnapshot
{
    uint32_t id = 0;
    uint32_t track_uid = 0;
    bool confirmed = false;
    bool matched = false;
    uint32_t age = 0;
    uint32_t total_hits = 0;
    uint32_t missed_frames = 0;
    uint32_t point_count = 0;
    float last_iou = 0.0f;
    Eigen::Vector3f position = Eigen::Vector3f::Zero();
    Eigen::Vector3f velocity = Eigen::Vector3f::Zero();
    Eigen::Vector3f dimensions = Eigen::Vector3f::Zero();
    std::vector<Eigen::Vector3f> history;
};

struct TrackerResult
{
    std_msgs::Header header;
    TrackerInputMode input_mode = TrackerInputMode::FRAME_OUT;
    std::size_t primary_point_count = 0;
    std::size_t point_out_point_count = 0;
    std::size_t support_point_count = 0;
    std::size_t detection_count = 0;
    std::size_t active_track_count = 0;
    double update_ms = 0.0;
    double cloud_odom_stamp_delta_ms = 0.0;
    std::vector<TrackSnapshot> tracks;
    std::vector<TrackSnapshot> tentative_tracks;
};

struct PredictedTrackerResult
{
    std_msgs::Header header;
    ros::Time measurement_stamp;
    double prediction_horizon = 0.0;
    bool stale = false;
    std::vector<TrackSnapshot> tracks;
};

PredictedTrackerResult PredictTrackerResult(const TrackerResult &measurement,
                                            const ros::Time &requested_stamp,
                                            double max_prediction_horizon,
                                            const Eigen::Vector3f &world_acceleration =
                                                Eigen::Vector3f::Zero());

class FrameTracker
{
public:
    explicit FrameTracker(const TrackerConfig &config);
    ~FrameTracker();

    FrameTracker(const FrameTracker &) = delete;
    FrameTracker &operator=(const FrameTracker &) = delete;
    FrameTracker(FrameTracker &&) noexcept;
    FrameTracker &operator=(FrameTracker &&) noexcept;

    TrackerResult update(const TrackerFrame &frame);
    const TrackerConfig &config() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace m_detector

#endif
