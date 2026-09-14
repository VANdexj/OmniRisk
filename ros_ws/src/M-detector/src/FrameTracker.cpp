#include <algorithm>
#include <m-detector/FrameTracker.h>
#include <cmath>
#include <deque>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ros/ros.h>
#include <opencv2/video/tracking.hpp>

#include <pcl/common/common.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <types.h>

namespace
{

enum class DetectionSource
{
    DYNAMIC = 0,
    STATIC_SUPPORT = 1
};

struct DetectionPolicy
{
    const char *name = "unknown";
    bool can_initialize_track = false;
    bool requires_preferred_track = false;
    bool uses_dynamic_reachability = false;
    bool uses_weak_measurement_variance = false;
    bool enforces_mahalanobis_gate = false;
    bool can_bootstrap_velocity = false;
    bool preserves_velocity = false;
    bool updates_dynamic_history = false;
};

const DetectionPolicy &PolicyFor(DetectionSource source)
{
    static const DetectionPolicy dynamic_policy{
        "dynamic", true, false, true, false, true, true, false, true};
    static const DetectionPolicy static_support_policy{
        "static_support", false, true, false, true, false, false, true, false};
    switch (source)
    {
    case DetectionSource::DYNAMIC:
        return dynamic_policy;
    case DetectionSource::STATIC_SUPPORT:
        return static_support_policy;
    }
    throw std::logic_error("unknown tracker detection source");
}

struct Detection
{
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    Eigen::Vector3f min_point = Eigen::Vector3f::Zero();
    Eigen::Vector3f max_point = Eigen::Vector3f::Zero();
    Eigen::Vector3f dimensions = Eigen::Vector3f::Zero();
    std::size_t point_count = 0;
    DetectionSource source = DetectionSource::DYNAMIC;
    int preferred_track_uid = -1;
};

struct Track
{
    int track_uid = -1;
    int display_id = -1;
    cv::KalmanFilter kf;
    Eigen::Vector3f dimensions = Eigen::Vector3f::Constant(0.5f);
    std::size_t point_count = 0;
    int age = 0;
    int total_hits = 0;
    int missed_frames = 0;
    bool confirmed = false;
    bool matched_this_frame = false;
    float last_iou = 0.0f;
    ros::Time last_stamp;
    ros::Time last_measurement_stamp;
    ros::Time last_dynamic_measurement_stamp;
    Eigen::Vector3f last_dynamic_measurement_position = Eigen::Vector3f::Zero();
    int dynamic_measurement_hits = 0;
    DetectionSource last_measurement_source = DetectionSource::DYNAMIC;
    int static_support_hits = 0;
    bool static_support_speed_gate_passed = false;
    float max_dynamic_speed = 0.0f;
    bool near_range_fast_candidate = false;
    ros::Time last_speed_output_pass_stamp;
    std::deque<bool> hit_window;
    std::deque<Eigen::Vector3f> history;

    Eigen::Vector3f position() const
    {
        return Eigen::Vector3f(kf.statePost.at<float>(0),
                               kf.statePost.at<float>(1),
                               kf.statePost.at<float>(2));
    }

    Eigen::Vector3f velocity() const
    {
        return Eigen::Vector3f(kf.statePost.at<float>(3),
                               kf.statePost.at<float>(4),
                               kf.statePost.at<float>(5));
    }
};

struct AssociationEvaluation
{
    bool eligible = false;
    float iou = 0.0f;
    float prediction_residual = std::numeric_limits<float>::infinity();
    double prediction_residual_limit = 0.0;
    double physical_displacement = 0.0;
    double physical_displacement_limit = std::numeric_limits<double>::infinity();
    double mahalanobis_squared = 1e6;
    double cost = 1e6;
};

struct SupportCloudFrame
{
    ros::Time stamp;
    PointCloudXYZI::ConstPtr cloud;
};

struct OdomDistanceGate
{
    float xy_distance = 0.0f;
    float z_distance = 0.0f;
};

float ComputeAabbIoU(const Eigen::Vector3f &center_a,
                     const Eigen::Vector3f &dims_a,
                     const Eigen::Vector3f &center_b,
                     const Eigen::Vector3f &dims_b)
{
    const Eigen::Vector3f half_a = 0.5f * dims_a;
    const Eigen::Vector3f half_b = 0.5f * dims_b;

    const Eigen::Vector3f min_a = center_a - half_a;
    const Eigen::Vector3f max_a = center_a + half_a;
    const Eigen::Vector3f min_b = center_b - half_b;
    const Eigen::Vector3f max_b = center_b + half_b;

    const Eigen::Vector3f overlap_min = min_a.cwiseMax(min_b);
    const Eigen::Vector3f overlap_max = max_a.cwiseMin(max_b);
    const Eigen::Vector3f overlap = (overlap_max - overlap_min).cwiseMax(Eigen::Vector3f::Zero());

    const float intersection = overlap.x() * overlap.y() * overlap.z();
    const float volume_a = dims_a.x() * dims_a.y() * dims_a.z();
    const float volume_b = dims_b.x() * dims_b.y() * dims_b.z();
    const float denom = volume_a + volume_b - intersection;
    if (denom <= 1e-6f)
    {
        return 0.0f;
    }
    return intersection / denom;
}

float ComputeAabbGap(const Detection &left, const Detection &right)
{
    const Eigen::Vector3f gap =
        (left.min_point - right.max_point)
            .cwiseMax(right.min_point - left.max_point)
            .cwiseMax(Eigen::Vector3f::Zero());
    return gap.norm();
}

std::vector<int> SolveHungarian(const std::vector<std::vector<double>> &cost_matrix)
{
    const int rows = static_cast<int>(cost_matrix.size());
    const int cols = rows == 0 ? 0 : static_cast<int>(cost_matrix.front().size());
    if (rows == 0 || cols == 0)
    {
        return std::vector<int>(rows, -1);
    }

    const int dim = std::max(rows, cols);
    std::vector<std::vector<double>> padded(dim + 1, std::vector<double>(dim + 1, 1.0));
    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            padded[row + 1][col + 1] = cost_matrix[row][col];
        }
    }

    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> u(dim + 1, 0.0);
    std::vector<double> v(dim + 1, 0.0);
    std::vector<int> p(dim + 1, 0);
    std::vector<int> way(dim + 1, 0);

    for (int row = 1; row <= dim; ++row)
    {
        p[0] = row;
        int col0 = 0;
        std::vector<double> minv(dim + 1, inf);
        std::vector<char> used(dim + 1, false);

        do
        {
            used[col0] = true;
            const int row0 = p[col0];
            int col1 = 0;
            double delta = inf;
            for (int col = 1; col <= dim; ++col)
            {
                if (used[col])
                {
                    continue;
                }
                const double cur = padded[row0][col] - u[row0] - v[col];
                if (cur < minv[col])
                {
                    minv[col] = cur;
                    way[col] = col0;
                }
                if (minv[col] < delta)
                {
                    delta = minv[col];
                    col1 = col;
                }
            }
            for (int col = 0; col <= dim; ++col)
            {
                if (used[col])
                {
                    u[p[col]] += delta;
                    v[col] -= delta;
                }
                else
                {
                    minv[col] -= delta;
                }
            }
            col0 = col1;
        } while (p[col0] != 0);

        do
        {
            const int col1 = way[col0];
            p[col0] = p[col1];
            col0 = col1;
        } while (col0 != 0);
    }

    std::vector<int> assignment(rows, -1);
    for (int col = 1; col <= dim; ++col)
    {
        const int row = p[col];
        if (row >= 1 && row <= rows && col <= cols)
        {
            assignment[row - 1] = col - 1;
        }
    }
    return assignment;
}

int CountHits(const std::deque<bool> &hit_window)
{
    return static_cast<int>(std::count(hit_window.begin(), hit_window.end(), true));
}

} // namespace

namespace m_detector
{

namespace
{

void RequireConfig(bool condition, const char *message)
{
    if (!condition)
    {
        throw std::invalid_argument(std::string("invalid dyn_obj/tracker config: ") + message);
    }
}

void ValidateTrackerConfig(const TrackerConfig &config)
{
    RequireConfig(!config.enabled || !config.tracks_topic.empty(), "tracks_topic must not be empty");
    RequireConfig(!config.enabled || !config.tentative_tracks_topic.empty(),
                  "tentative_tracks_topic must not be empty");
    RequireConfig(!config.enabled || !config.predicted_tracks_topic.empty(),
                  "predicted_tracks_topic must not be empty");
    RequireConfig(!config.enabled || !config.markers_topic.empty(), "markers_topic must not be empty");
    RequireConfig(config.max_prediction_horizon >= 0.0,
                  "max_prediction_horizon must be >= 0");
    RequireConfig(config.lost_track_prediction_time >= 0.0,
                  "lost_track_prediction_time must be >= 0");
    RequireConfig(config.point_out_duplicate_distance >= 0.0,
                  "point_out_duplicate_distance must be >= 0");
    RequireConfig(config.cluster_tolerance > 0.0, "cluster_tolerance must be > 0");
    RequireConfig(config.cluster_min_points_process > 0,
                  "cluster_min_points_process must be > 0");
    RequireConfig(config.cluster_min_points > 0, "cluster_min_points must be > 0");
    RequireConfig(config.cluster_min_points_process <= config.cluster_min_points,
                  "cluster_min_points_process must be <= cluster_min_points");
    RequireConfig(config.cluster_max_points >= config.cluster_min_points,
                  "cluster_max_points must be >= cluster_min_points");
    RequireConfig(config.min_init_hits > 0, "min_init_hits must be > 0");
    RequireConfig(config.init_window_size >= config.min_init_hits,
                  "init_window_size must be >= min_init_hits");
    RequireConfig(config.near_range_fast_confirm_distance >= 0.0,
                  "near_range_fast_confirm_distance must be >= 0");
    RequireConfig(config.near_range_fast_confirm_hits > 0 &&
                      config.near_range_fast_confirm_hits <= config.init_window_size,
                  "near_range_fast_confirm_hits must be in [1, init_window_size]");
    RequireConfig(config.track_lifetime_frames >= 0, "track_lifetime_frames must be >= 0");
    RequireConfig(config.track_history_size >= 2, "track_history_size must be >= 2");
    RequireConfig(config.min_iou >= -1.0 && config.min_iou <= 1.0,
                  "min_iou must be in [-1, 1]");
    RequireConfig(config.max_center_distance > 0.0, "max_center_distance must be > 0");
    RequireConfig(config.association_padding >= 0.0, "association_padding must be >= 0");
    RequireConfig(config.association_mahalanobis_gate_sq > 0.0,
                  "association_mahalanobis_gate_sq must be > 0");
    RequireConfig(config.duplicate_guard_distance >= 0.0,
                  "duplicate_guard_distance must be >= 0");
    RequireConfig(config.duplicate_guard_bbox_gap >= 0.0,
                  "duplicate_guard_bbox_gap must be >= 0");
    RequireConfig(config.default_dt > 0.0, "default_dt must be > 0");
    RequireConfig(std::isfinite(config.gravity_world_x) &&
                      std::isfinite(config.gravity_world_y) &&
                      std::isfinite(config.gravity_world_z),
                  "gravity_world components must be finite");
    const double gravity_norm = std::sqrt(config.gravity_world_x * config.gravity_world_x +
                                          config.gravity_world_y * config.gravity_world_y +
                                          config.gravity_world_z * config.gravity_world_z);
    RequireConfig(!config.ballistic_model_en || gravity_norm > 1e-6,
                  "ballistic_model_en requires a non-zero gravity_world vector");
    RequireConfig(config.process_acceleration_stddev >= 0.0,
                  "process_acceleration_stddev must be >= 0");
    RequireConfig(config.dynamic_position_variance > 0.0,
                  "dynamic_position_variance must be > 0");
    RequireConfig(config.max_target_speed > 0.0, "max_target_speed must be > 0");
    RequireConfig(config.max_target_acceleration >= 0.0,
                  "max_target_acceleration must be >= 0");
    RequireConfig(config.bbox_size_prior_min > 0.0,
                  "bbox_size_prior_min must be > 0");
    RequireConfig(config.bbox_size_prior_max >= config.bbox_size_prior_min,
                  "bbox_size_prior_max must be >= bbox_size_prior_min");
    RequireConfig(config.target_max_xy_distance >= 0.0, "target_max_xy_distance must be >= 0");
    RequireConfig(config.target_max_z_distance >= 0.0, "target_max_z_distance must be >= 0");
    RequireConfig(config.target_origin_bbox_min_x <= config.target_origin_bbox_max_x &&
                      config.target_origin_bbox_min_y <= config.target_origin_bbox_max_y &&
                      config.target_origin_bbox_min_z <= config.target_origin_bbox_max_z,
                  "target origin bbox minima must not exceed maxima");
    RequireConfig(config.target_min_bbox_volume >= 0.0, "target_min_bbox_volume must be >= 0");
    RequireConfig(config.target_max_bbox_volume >= config.target_min_bbox_volume,
                  "target_max_bbox_volume must be >= target_min_bbox_volume");
    RequireConfig(config.target_min_speed >= 0.0, "target_min_speed must be >= 0");
    RequireConfig(config.target_speed_filter_grace_time >= 0.0,
                  "target_speed_filter_grace_time must be >= 0");
    RequireConfig(config.static_support_hold_time >= 0.0,
                  "static_support_hold_time must be >= 0");
    RequireConfig(config.static_support_bbox_expand_ratio >= 0.0,
                  "static_support_bbox_expand_ratio must be >= 0");
    RequireConfig(config.static_support_bbox_padding >= 0.0,
                  "static_support_bbox_padding must be >= 0");
    RequireConfig(config.static_support_min_points > 0, "static_support_min_points must be > 0");
    RequireConfig(config.static_support_cluster_tolerance > 0.0,
                  "static_support_cluster_tolerance must be > 0");
    RequireConfig(config.static_support_cluster_max_points >= config.static_support_min_points,
                  "static_support_cluster_max_points must be >= static_support_min_points");
    RequireConfig(config.static_support_max_residual > 0.0,
                  "static_support_max_residual must be > 0");
    RequireConfig(config.static_support_position_variance > 0.0,
                  "static_support_position_variance must be > 0");
    RequireConfig(config.static_support_min_activation_speed >= 0.0,
                  "static_support_min_activation_speed must be >= 0");
    RequireConfig(config.marker_bbox_scale > 0.0, "marker_bbox_scale must be > 0");
    RequireConfig(config.marker_bbox_min_size > 0.0, "marker_bbox_min_size must be > 0");
    RequireConfig(config.marker_velocity_arrow_scale >= 0.0,
                  "marker_velocity_arrow_scale must be >= 0");
    RequireConfig(config.marker_velocity_min_speed >= 0.0,
                  "marker_velocity_min_speed must be >= 0");
    RequireConfig(config.marker_velocity_max_arrow_length > 0.0,
                  "marker_velocity_max_arrow_length must be > 0");
    RequireConfig(config.marker_velocity_text_size > 0.0,
                  "marker_velocity_text_size must be > 0");
}

TrackerConfig NormalizeTrackerConfig(TrackerConfig config)
{
    ValidateTrackerConfig(config);
    config.min_init_hits = std::max(1, config.min_init_hits);
    config.init_window_size = std::max(config.min_init_hits, config.init_window_size);
    config.near_range_fast_confirm_distance = std::max(0.0, config.near_range_fast_confirm_distance);
    config.near_range_fast_confirm_hits = std::max(1, config.near_range_fast_confirm_hits);
    config.track_lifetime_frames = std::max(0, config.track_lifetime_frames);
    config.track_history_size = std::max(2, config.track_history_size);
    config.marker_bbox_scale = std::max(0.01, config.marker_bbox_scale);
    config.marker_bbox_min_size = std::max(0.01, config.marker_bbox_min_size);
    config.marker_velocity_arrow_scale = std::max(0.0, config.marker_velocity_arrow_scale);
    config.marker_velocity_min_speed = std::max(0.0, config.marker_velocity_min_speed);
    config.marker_velocity_max_arrow_length = std::max(0.01, config.marker_velocity_max_arrow_length);
    config.marker_velocity_text_size = std::max(0.01, config.marker_velocity_text_size);
    config.target_max_xy_distance = std::max(0.0, config.target_max_xy_distance);
    config.target_max_z_distance = std::max(0.0, config.target_max_z_distance);
    if (config.target_origin_bbox_min_x > config.target_origin_bbox_max_x)
    {
        std::swap(config.target_origin_bbox_min_x, config.target_origin_bbox_max_x);
    }
    if (config.target_origin_bbox_min_y > config.target_origin_bbox_max_y)
    {
        std::swap(config.target_origin_bbox_min_y, config.target_origin_bbox_max_y);
    }
    if (config.target_origin_bbox_min_z > config.target_origin_bbox_max_z)
    {
        std::swap(config.target_origin_bbox_min_z, config.target_origin_bbox_max_z);
    }
    config.target_min_bbox_volume = std::max(0.0, config.target_min_bbox_volume);
    config.target_max_bbox_volume = std::max(config.target_min_bbox_volume, config.target_max_bbox_volume);
    config.target_min_speed = std::max(0.0, config.target_min_speed);
    config.target_speed_filter_grace_time = std::max(0.0, config.target_speed_filter_grace_time);
    config.static_support_hold_time = std::max(0.0, config.static_support_hold_time);
    config.static_support_bbox_expand_ratio = std::max(0.0, config.static_support_bbox_expand_ratio);
    config.static_support_bbox_padding = std::max(0.0, config.static_support_bbox_padding);
    config.static_support_min_points = std::max(1, config.static_support_min_points);
    config.static_support_cluster_tolerance = std::max(0.05, config.static_support_cluster_tolerance);
    config.static_support_cluster_max_points =
        std::max(config.static_support_min_points, config.static_support_cluster_max_points);
    config.static_support_max_residual = std::max(0.01, config.static_support_max_residual);
    config.static_support_min_activation_speed = std::max(0.0, config.static_support_min_activation_speed);
    config.point_out_duplicate_distance = std::max(0.0, config.point_out_duplicate_distance);
    return config;
}

} // namespace

TrackerConfig LoadTrackerConfig(const ros::NodeHandle &node, const std::string &parameter_prefix)
{
    TrackerConfig config;
#define LOAD_TRACKER_PARAM(name) \
    node.param(parameter_prefix + "/" #name, config.name, config.name)
    LOAD_TRACKER_PARAM(enabled);
    LOAD_TRACKER_PARAM(tracks_topic);
    LOAD_TRACKER_PARAM(tentative_tracks_topic);
    LOAD_TRACKER_PARAM(predicted_tracks_topic);
    LOAD_TRACKER_PARAM(markers_topic);
    LOAD_TRACKER_PARAM(max_prediction_horizon);
    LOAD_TRACKER_PARAM(lost_track_prediction_time);
    LOAD_TRACKER_PARAM(point_out_dynamic_support_en);
    LOAD_TRACKER_PARAM(point_out_duplicate_distance);
    LOAD_TRACKER_PARAM(cluster_tolerance);
    LOAD_TRACKER_PARAM(cluster_min_points_process);
    LOAD_TRACKER_PARAM(cluster_min_points);
    LOAD_TRACKER_PARAM(cluster_max_points);
    LOAD_TRACKER_PARAM(min_init_hits);
    LOAD_TRACKER_PARAM(init_window_size);
    LOAD_TRACKER_PARAM(near_range_fast_confirm_en);
    LOAD_TRACKER_PARAM(near_range_fast_confirm_distance);
    LOAD_TRACKER_PARAM(near_range_fast_confirm_hits);
    LOAD_TRACKER_PARAM(track_lifetime_frames);
    LOAD_TRACKER_PARAM(track_history_size);
    LOAD_TRACKER_PARAM(min_iou);
    LOAD_TRACKER_PARAM(max_center_distance);
    LOAD_TRACKER_PARAM(association_padding);
    LOAD_TRACKER_PARAM(association_mahalanobis_gate_sq);
    LOAD_TRACKER_PARAM(duplicate_guard_en);
    LOAD_TRACKER_PARAM(duplicate_guard_distance);
    LOAD_TRACKER_PARAM(duplicate_guard_bbox_gap);
    LOAD_TRACKER_PARAM(default_dt);
    LOAD_TRACKER_PARAM(ballistic_model_en);
    LOAD_TRACKER_PARAM(gravity_world_x);
    LOAD_TRACKER_PARAM(gravity_world_y);
    LOAD_TRACKER_PARAM(gravity_world_z);
    LOAD_TRACKER_PARAM(process_acceleration_stddev);
    LOAD_TRACKER_PARAM(dynamic_position_variance);
    LOAD_TRACKER_PARAM(max_target_speed);
    LOAD_TRACKER_PARAM(max_target_acceleration);
    LOAD_TRACKER_PARAM(bbox_size_prior_en);
    LOAD_TRACKER_PARAM(bbox_size_prior_min);
    LOAD_TRACKER_PARAM(bbox_size_prior_max);
    LOAD_TRACKER_PARAM(target_distance_filter_en);
    LOAD_TRACKER_PARAM(target_max_xy_distance);
    LOAD_TRACKER_PARAM(target_max_z_distance);
    LOAD_TRACKER_PARAM(target_origin_bbox_filter_en);
    LOAD_TRACKER_PARAM(target_origin_bbox_min_x);
    LOAD_TRACKER_PARAM(target_origin_bbox_max_x);
    LOAD_TRACKER_PARAM(target_origin_bbox_min_y);
    LOAD_TRACKER_PARAM(target_origin_bbox_max_y);
    LOAD_TRACKER_PARAM(target_origin_bbox_min_z);
    LOAD_TRACKER_PARAM(target_origin_bbox_max_z);
    LOAD_TRACKER_PARAM(target_ground_filter_en);
    LOAD_TRACKER_PARAM(target_min_world_z);
    LOAD_TRACKER_PARAM(target_volume_filter_en);
    LOAD_TRACKER_PARAM(target_min_bbox_volume);
    LOAD_TRACKER_PARAM(target_max_bbox_volume);
    LOAD_TRACKER_PARAM(target_speed_filter_en);
    LOAD_TRACKER_PARAM(target_min_speed);
    LOAD_TRACKER_PARAM(target_speed_filter_grace_time);
    LOAD_TRACKER_PARAM(static_support_en);
    LOAD_TRACKER_PARAM(static_support_hold_time);
    LOAD_TRACKER_PARAM(static_support_bbox_expand_ratio);
    LOAD_TRACKER_PARAM(static_support_bbox_padding);
    LOAD_TRACKER_PARAM(static_support_min_points);
    LOAD_TRACKER_PARAM(static_support_cluster_tolerance);
    LOAD_TRACKER_PARAM(static_support_cluster_max_points);
    LOAD_TRACKER_PARAM(static_support_max_residual);
    LOAD_TRACKER_PARAM(static_support_position_variance);
    LOAD_TRACKER_PARAM(static_support_min_activation_speed);
    LOAD_TRACKER_PARAM(debug_track_lifecycle);
    LOAD_TRACKER_PARAM(debug_association_rejects);
    LOAD_TRACKER_PARAM(debug_print_detections);
    LOAD_TRACKER_PARAM(debug_print_association_matrix);
    LOAD_TRACKER_PARAM(debug_bbox_prior_rejects);
    LOAD_TRACKER_PARAM(debug_distance_rejects);
    LOAD_TRACKER_PARAM(debug_static_support);
    LOAD_TRACKER_PARAM(marker_bbox_scale);
    LOAD_TRACKER_PARAM(marker_bbox_min_size);
    LOAD_TRACKER_PARAM(marker_velocity_arrow_scale);
    LOAD_TRACKER_PARAM(marker_velocity_min_speed);
    LOAD_TRACKER_PARAM(marker_velocity_max_arrow_length);
    LOAD_TRACKER_PARAM(marker_velocity_text_size);
#undef LOAD_TRACKER_PARAM
    return NormalizeTrackerConfig(std::move(config));
}

PredictedTrackerResult PredictTrackerResult(const TrackerResult &measurement,
                                            const ros::Time &requested_stamp,
                                            double max_prediction_horizon,
                                            const Eigen::Vector3f &world_acceleration)
{
    PredictedTrackerResult prediction;
    prediction.header = measurement.header;
    prediction.measurement_stamp = measurement.header.stamp;
    prediction.tracks = measurement.tracks;

    if (measurement.header.stamp.isZero() || requested_stamp.isZero())
    {
        return prediction;
    }

    const double requested_horizon =
        std::max(0.0, (requested_stamp - measurement.header.stamp).toSec());
    const double safe_max_horizon = std::max(0.0, max_prediction_horizon);
    prediction.prediction_horizon = std::min(requested_horizon, safe_max_horizon);
    prediction.stale = requested_horizon > safe_max_horizon + 1e-6;
    prediction.header.stamp = measurement.header.stamp +
        ros::Duration(prediction.prediction_horizon);

    const float horizon = static_cast<float>(prediction.prediction_horizon);
    const Eigen::Vector3f acceleration =
        world_acceleration.allFinite() ? world_acceleration : Eigen::Vector3f::Zero();
    for (auto &track : prediction.tracks)
    {
        track.position += track.velocity * horizon + 0.5f * acceleration * horizon * horizon;
        track.velocity += acceleration * horizon;
    }
    return prediction;
}

class FrameTracker::Impl
{
public:
    explicit Impl(const TrackerConfig &raw_config)
        : config_(NormalizeTrackerConfig(raw_config))
    {
        if (config_.ballistic_model_en)
        {
            ROS_INFO_STREAM("frame_tracker ballistic model enabled: gravity_world="
                            << FormatVec3(ModelAcceleration()));
        }
    }

    TrackerResult Update(const TrackerFrame &frame);
    const TrackerConfig &config() const { return config_; }

private:
    std::string FormatVec3(const Eigen::Vector3f &vec) const
    {
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss.precision(3);
        oss << "(" << vec.x() << ", " << vec.y() << ", " << vec.z() << ")";
        return oss.str();
    }

    const char *DetectionSourceName(DetectionSource source) const
    {
        return PolicyFor(source).name;
    }

    Eigen::Vector3f ModelAcceleration() const
    {
        if (!config_.ballistic_model_en)
        {
            return Eigen::Vector3f::Zero();
        }
        return Eigen::Vector3f(static_cast<float>(config_.gravity_world_x),
                               static_cast<float>(config_.gravity_world_y),
                               static_cast<float>(config_.gravity_world_z));
    }

    double ModelAccelerationMagnitude() const
    {
        return static_cast<double>(ModelAcceleration().norm());
    }

    void DebugPrintFrameState(const std_msgs::Header &header, const std::vector<Detection> &detections) const
    {
        if (!config_.debug_print_detections)
        {
            return;
        }

        ROS_INFO_STREAM("frame_tracker frame=" << frame_index_
                        << " stamp=" << header.stamp.toSec()
                        << " tracks=" << tracks_.size()
                        << " detections=" << detections.size());

        if (tracks_.empty())
        {
            ROS_INFO("frame_tracker tracks: none");
        }
        else
        {
            for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index)
            {
                const auto &track = tracks_[track_index];
                ROS_INFO_STREAM("frame_tracker track[" << track_index << "]"
                                << " track_uid=" << track.track_uid
                                << " display_id=" << track.display_id
                                << " confirmed=" << track.confirmed
                                << " age=" << track.age
                                << " hits=" << track.total_hits
                                << " missed=" << track.missed_frames
                                << " pos=" << FormatVec3(track.position())
                                << " vel=" << FormatVec3(track.velocity())
                                << " dims=" << FormatVec3(track.dimensions));
            }
        }

        if (detections.empty())
        {
            ROS_INFO("frame_tracker detections: none");
        }
        else
        {
            for (std::size_t det_index = 0; det_index < detections.size(); ++det_index)
            {
                const auto &detection = detections[det_index];
                ROS_INFO_STREAM("frame_tracker detection[" << det_index << "]"
                                << " point_count=" << detection.point_count
                                << " source=" << DetectionSourceName(detection.source)
                                << " centroid=" << FormatVec3(detection.centroid)
                                << " dims=" << FormatVec3(detection.dimensions)
                                << " min=" << FormatVec3(detection.min_point)
                                << " max=" << FormatVec3(detection.max_point));
            }
        }
    }

    void DebugPrintAssociationMatrix(const std::vector<Detection> &detections,
                                     const std::vector<std::vector<float>> &center_distances,
                                     const std::vector<std::vector<float>> &ious) const
    {
        if (!config_.debug_print_association_matrix)
        {
            return;
        }

        if (tracks_.empty() || detections.empty())
        {
            ROS_INFO_STREAM("frame_tracker association_matrix skipped: tracks=" << tracks_.size()
                            << " detections=" << detections.size());
            return;
        }

        for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index)
        {
            const auto &track = tracks_[track_index];
            for (std::size_t det_index = 0; det_index < detections.size(); ++det_index)
            {
                ROS_INFO_STREAM("frame_tracker compare"
                                << " track_uid=" << track.track_uid
                                << " display_id=" << track.display_id
                                << " track_index=" << track_index
                                << " det_index=" << det_index
                                << " center_distance=" << center_distances[track_index][det_index]
                                << " max_center_distance=" << config_.max_center_distance
                                << " iou=" << ious[track_index][det_index]
                                << " min_iou=" << config_.min_iou);
            }
        }
    }

    bool IsDuplicateDynamicDetection(const Detection &candidate,
                                     const std::vector<Detection> &detections) const
    {
        for (const auto &detection : detections)
        {
            if (!PolicyFor(detection.source).uses_dynamic_reachability)
            {
                continue;
            }
            if ((candidate.centroid - detection.centroid).norm() <=
                static_cast<float>(config_.point_out_duplicate_distance))
            {
                return true;
            }
        }
        return false;
    }

    void AppendPointOutDetections(std::vector<Detection> &detections,
                                  const PointCloudXYZI::ConstPtr &point_out,
                                  bool force_point_out) const
    {
        if ((!config_.point_out_dynamic_support_en && !force_point_out) || !point_out)
        {
            return;
        }

        std::vector<Detection> point_out_detections =
            ApplyDetectionFilters(ExtractDetections(point_out));
        for (const auto &detection : point_out_detections)
        {
            if (IsDuplicateDynamicDetection(detection, detections))
            {
                continue;
            }
            detections.push_back(detection);
        }
    }

    std::vector<Detection> ExtractDetections(const PointCloudXYZI::ConstPtr &cloud) const
    {
        std::vector<Detection> detections;
        if (!cloud || cloud->empty())
        {
            return detections;
        }

        pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>());
        tree->setInputCloud(cloud);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<PointType> extractor;
        extractor.setClusterTolerance(config_.cluster_tolerance);
        extractor.setMinClusterSize(config_.cluster_min_points_process);
        extractor.setMaxClusterSize(config_.cluster_max_points);
        extractor.setSearchMethod(tree);
        extractor.setInputCloud(cloud);
        extractor.extract(cluster_indices);

        detections.reserve(cluster_indices.size());
        for (const auto &indices : cluster_indices)
        {
            if (indices.indices.empty())
            {
                continue;
            }

            Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
            Eigen::Vector4f min_pt;
            Eigen::Vector4f max_pt;
            min_pt << std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), 1.0f;
            max_pt << -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), 1.0f;

            for (const int index : indices.indices)
            {
                const auto &point = cloud->points[index];
                centroid += Eigen::Vector3f(point.x, point.y, point.z);
                min_pt.x() = std::min(min_pt.x(), point.x);
                min_pt.y() = std::min(min_pt.y(), point.y);
                min_pt.z() = std::min(min_pt.z(), point.z);
                max_pt.x() = std::max(max_pt.x(), point.x);
                max_pt.y() = std::max(max_pt.y(), point.y);
                max_pt.z() = std::max(max_pt.z(), point.z);
            }

            centroid /= static_cast<float>(indices.indices.size());

            Detection detection;
            detection.centroid = centroid;
            detection.min_point = min_pt.head<3>();
            detection.max_point = max_pt.head<3>();
            detection.dimensions = (detection.max_point - detection.min_point).cwiseMax(Eigen::Vector3f::Constant(0.05f));
            detection.point_count = indices.indices.size();
            if (!PassesBboxSizePrior(detection))
            {
                continue;
            }
            detections.push_back(detection);
        }

        return detections;
    }

    std::vector<Detection> BuildPreclusteredDetections(
        const std::vector<m_detector::Detection3D> &preclustered) const
    {
        std::vector<Detection> detections;
        detections.reserve(preclustered.size());
        for (const auto &measurement : preclustered)
        {
            if (measurement.point_count < static_cast<std::size_t>(config_.cluster_min_points_process) ||
                measurement.point_count > static_cast<std::size_t>(config_.cluster_max_points) ||
                !measurement.centroid.allFinite() ||
                !measurement.min_point.allFinite() ||
                !measurement.max_point.allFinite() ||
                (measurement.max_point.array() < measurement.min_point.array()).any())
            {
                continue;
            }

            Detection detection;
            detection.centroid = measurement.centroid;
            detection.min_point = measurement.min_point;
            detection.max_point = measurement.max_point;
            detection.dimensions =
                (detection.max_point - detection.min_point)
                    .cwiseMax(Eigen::Vector3f::Constant(0.05f));
            detection.point_count = measurement.point_count;
            if (PassesBboxSizePrior(detection))
            {
                detections.push_back(detection);
            }
        }
        return detections;
    }

    bool FitsRecentTrackEnvelope(const Eigen::Vector3f &union_min,
                                 const Eigen::Vector3f &union_max) const
    {
        const float envelope_padding =
            static_cast<float>(config_.duplicate_guard_bbox_gap);
        return std::any_of(tracks_.begin(), tracks_.end(),
            [&](const Track &track)
            {
                if (track.missed_frames != 0)
                {
                    return false;
                }
                const Eigen::Vector3f half_dimensions =
                    0.5f * track.dimensions.cwiseMax(
                        Eigen::Vector3f::Constant(0.05f));
                const Eigen::Vector3f envelope_min =
                    track.position() - half_dimensions -
                    Eigen::Vector3f::Constant(envelope_padding);
                const Eigen::Vector3f envelope_max =
                    track.position() + half_dimensions +
                    Eigen::Vector3f::Constant(envelope_padding);
                return (union_min.array() >= envelope_min.array()).all() &&
                    (union_max.array() <= envelope_max.array()).all();
            });
    }

    bool ShouldJoinDynamicDetectionGroup(
        const std::vector<Detection> &detections,
        const std::vector<std::size_t> &group,
        std::size_t candidate_index) const
    {
        const Detection &candidate = detections[candidate_index];
        const bool within_centroid_complete_link =
            std::all_of(group.begin(), group.end(), [&](std::size_t member_index)
            {
                return (candidate.centroid - detections[member_index].centroid).norm() <=
                    static_cast<float>(config_.duplicate_guard_distance);
            });
        if (within_centroid_complete_link)
        {
            return true;
        }

        const bool within_bbox_complete_link =
            std::all_of(group.begin(), group.end(), [&](std::size_t member_index)
            {
                return ComputeAabbGap(candidate, detections[member_index]) <=
                    static_cast<float>(config_.duplicate_guard_bbox_gap);
            });
        if (!within_bbox_complete_link)
        {
            return false;
        }

        Eigen::Vector3f union_min = candidate.min_point;
        Eigen::Vector3f union_max = candidate.max_point;
        for (const std::size_t member_index : group)
        {
            union_min = union_min.cwiseMin(detections[member_index].min_point);
            union_max = union_max.cwiseMax(detections[member_index].max_point);
        }
        return FitsRecentTrackEnvelope(union_min, union_max);
    }

    std::vector<Detection> MergeNearbyDynamicDetections(
        const std::vector<Detection> &detections) const
    {
        if (!config_.duplicate_guard_en || detections.size() < 2)
        {
            return detections;
        }

        std::vector<std::size_t> order(detections.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right)
        {
            const Eigen::Vector3f &left_centroid = detections[left].centroid;
            const Eigen::Vector3f &right_centroid = detections[right].centroid;
            for (int axis = 0; axis < 3; ++axis)
            {
                if (left_centroid[axis] != right_centroid[axis])
                {
                    return left_centroid[axis] < right_centroid[axis];
                }
            }
            if (detections[left].point_count != detections[right].point_count)
            {
                return detections[left].point_count > detections[right].point_count;
            }
            return left < right;
        });

        std::vector<std::vector<std::size_t>> groups;
        for (const std::size_t detection_index : order)
        {
            bool grouped = false;
            if (PolicyFor(detections[detection_index].source).uses_dynamic_reachability)
            {
                for (auto &group : groups)
                {
                    const bool group_is_dynamic =
                        PolicyFor(detections[group.front()].source).uses_dynamic_reachability;
                    const bool within_complete_link = group_is_dynamic &&
                        ShouldJoinDynamicDetectionGroup(
                            detections, group, detection_index);
                    if (within_complete_link)
                    {
                        group.push_back(detection_index);
                        grouped = true;
                        break;
                    }
                }
            }
            if (!grouped)
            {
                groups.push_back({detection_index});
            }
        }

        std::vector<Detection> merged;
        merged.reserve(groups.size());
        for (const auto &group : groups)
        {
            Detection merged_detection = detections[group.front()];
            for (std::size_t member_offset = 1; member_offset < group.size(); ++member_offset)
            {
                const Detection &member_detection = detections[group[member_offset]];
                const std::size_t old_count = merged_detection.point_count;
                const std::size_t combined_count = old_count + member_detection.point_count;
                if (combined_count > 0)
                {
                    merged_detection.centroid =
                        (merged_detection.centroid * static_cast<float>(old_count) +
                         member_detection.centroid *
                             static_cast<float>(member_detection.point_count)) /
                        static_cast<float>(combined_count);
                }
                merged_detection.min_point =
                    merged_detection.min_point.cwiseMin(member_detection.min_point);
                merged_detection.max_point =
                    merged_detection.max_point.cwiseMax(member_detection.max_point);
                merged_detection.point_count = combined_count;
            }
            merged_detection.dimensions =
                (merged_detection.max_point - merged_detection.min_point)
                    .cwiseMax(Eigen::Vector3f::Constant(0.05f));
            merged.push_back(std::move(merged_detection));
        }
        return merged;
    }

    bool PassesBboxSizePrior(const Detection &detection) const
    {
        if (!config_.bbox_size_prior_en)
        {
            return true;
        }
        const bool pass =
            detection.dimensions.x() >= config_.bbox_size_prior_min &&
            detection.dimensions.x() <= config_.bbox_size_prior_max &&
            detection.dimensions.y() >= config_.bbox_size_prior_min &&
            detection.dimensions.y() <= config_.bbox_size_prior_max &&
            detection.dimensions.z() >= config_.bbox_size_prior_min &&
            detection.dimensions.z() <= config_.bbox_size_prior_max;
        if (!pass && config_.debug_bbox_prior_rejects)
        {
            ROS_INFO_STREAM("frame_tracker reject detection by bbox size prior: dims="
                            << FormatVec3(detection.dimensions)
                            << " min=" << config_.bbox_size_prior_min
                            << " max=" << config_.bbox_size_prior_max
                            << " points=" << detection.point_count);
        }
        return pass;
    }

    bool CanApplyDistanceFilter(const char *target_type) const
    {
        if (!config_.target_distance_filter_en)
        {
            return false;
        }
        if (!has_odom_)
        {
            ROS_WARN_STREAM_THROTTLE(2.0, "frame_tracker distance filter enabled for "
                                               << target_type
                                               << ", but this frame has no odometry; skipping distance filter.");
            return false;
        }
        return true;
    }

    OdomDistanceGate ComputeOdomDistanceGate(const Eigen::Vector3f &target_position) const
    {
        const Eigen::Vector3f delta = target_position - current_odom_position_;
        OdomDistanceGate distance;
        distance.xy_distance = std::hypot(delta.x(), delta.y());
        distance.z_distance = std::abs(delta.z());
        return distance;
    }

    bool PassesOdomDistanceGate(const OdomDistanceGate &distance) const
    {
        return distance.xy_distance <= static_cast<float>(config_.target_max_xy_distance) &&
               distance.z_distance <= static_cast<float>(config_.target_max_z_distance);
    }

    std::vector<Detection> FilterDetectionsByOdomDistance(const std::vector<Detection> &detections) const
    {
        if (!CanApplyDistanceFilter("detections"))
        {
            return detections;
        }

        std::vector<Detection> filtered;
        filtered.reserve(detections.size());
        for (const auto &detection : detections)
        {
            const OdomDistanceGate distance = ComputeOdomDistanceGate(detection.centroid);
            if (PassesOdomDistanceGate(distance))
            {
                filtered.push_back(detection);
                continue;
            }
            if (config_.debug_distance_rejects)
            {
                ROS_INFO_STREAM("frame_tracker reject detection by odom distance: xy_distance="
                                << distance.xy_distance
                                << " max_xy=" << config_.target_max_xy_distance
                                << " z_distance=" << distance.z_distance
                                << " max_z=" << config_.target_max_z_distance
                                << " centroid=" << FormatVec3(detection.centroid)
                                << " odom=" << FormatVec3(current_odom_position_)
                                << " points=" << detection.point_count);
            }
        }
        return filtered;
    }

    bool PassesOriginBboxGate(const Eigen::Vector3f &target_position) const
    {
        if (!config_.target_origin_bbox_filter_en)
        {
            return true;
        }
        return target_position.x() >= static_cast<float>(config_.target_origin_bbox_min_x) &&
               target_position.x() <= static_cast<float>(config_.target_origin_bbox_max_x) &&
               target_position.y() >= static_cast<float>(config_.target_origin_bbox_min_y) &&
               target_position.y() <= static_cast<float>(config_.target_origin_bbox_max_y) &&
               target_position.z() >= static_cast<float>(config_.target_origin_bbox_min_z) &&
               target_position.z() <= static_cast<float>(config_.target_origin_bbox_max_z);
    }

    std::vector<Detection> FilterDetectionsByOriginBbox(const std::vector<Detection> &detections) const
    {
        if (!config_.target_origin_bbox_filter_en)
        {
            return detections;
        }

        std::vector<Detection> filtered;
        filtered.reserve(detections.size());
        for (const auto &detection : detections)
        {
            if (PassesOriginBboxGate(detection.centroid))
            {
                filtered.push_back(detection);
                continue;
            }
            if (config_.debug_distance_rejects)
            {
                ROS_INFO_STREAM("frame_tracker reject detection by origin bbox: centroid="
                                << FormatVec3(detection.centroid)
                                << " min=(" << config_.target_origin_bbox_min_x << ", "
                                << config_.target_origin_bbox_min_y << ", "
                                << config_.target_origin_bbox_min_z << ")"
                                << " max=(" << config_.target_origin_bbox_max_x << ", "
                                << config_.target_origin_bbox_max_y << ", "
                                << config_.target_origin_bbox_max_z << ")"
                                << " points=" << detection.point_count);
            }
        }
        return filtered;
    }

    bool PassesGroundGate(const Eigen::Vector3f &target_position) const
    {
        if (!config_.target_ground_filter_en)
        {
            return true;
        }
        return target_position.z() >= static_cast<float>(config_.target_min_world_z);
    }

    std::vector<Detection> FilterDetectionsByGround(const std::vector<Detection> &detections) const
    {
        if (!config_.target_ground_filter_en)
        {
            return detections;
        }

        std::vector<Detection> filtered;
        filtered.reserve(detections.size());
        for (const auto &detection : detections)
        {
            if (PassesGroundGate(detection.centroid))
            {
                filtered.push_back(detection);
                continue;
            }
            if (config_.debug_distance_rejects)
            {
                ROS_INFO_STREAM("frame_tracker reject detection by ground filter: z="
                                << detection.centroid.z()
                                << " min_world_z=" << config_.target_min_world_z
                                << " centroid=" << FormatVec3(detection.centroid)
                                << " points=" << detection.point_count);
            }
        }
        return filtered;
    }

    float ComputeBboxVolume(const Eigen::Vector3f &dimensions) const
    {
        const Eigen::Vector3f safe_dimensions = dimensions.cwiseMax(Eigen::Vector3f::Zero());
        return safe_dimensions.x() * safe_dimensions.y() * safe_dimensions.z();
    }

    bool PassesVolumeGate(const Eigen::Vector3f &dimensions) const
    {
        if (!config_.target_volume_filter_en)
        {
            return true;
        }
        const float volume = ComputeBboxVolume(dimensions);
        return volume >= static_cast<float>(config_.target_min_bbox_volume) &&
               volume <= static_cast<float>(config_.target_max_bbox_volume);
    }

    std::vector<Detection> FilterDetectionsByVolume(const std::vector<Detection> &detections) const
    {
        if (!config_.target_volume_filter_en)
        {
            return detections;
        }

        std::vector<Detection> filtered;
        filtered.reserve(detections.size());
        for (const auto &detection : detections)
        {
            const float volume = ComputeBboxVolume(detection.dimensions);
            if (PassesVolumeGate(detection.dimensions))
            {
                filtered.push_back(detection);
                continue;
            }
            if (config_.debug_bbox_prior_rejects)
            {
                ROS_INFO_STREAM("frame_tracker reject detection by bbox volume: volume="
                                << volume
                                << " min_volume=" << config_.target_min_bbox_volume
                                << " max_volume=" << config_.target_max_bbox_volume
                                << " dims=" << FormatVec3(detection.dimensions)
                                << " centroid=" << FormatVec3(detection.centroid)
                                << " points=" << detection.point_count);
            }
        }
        return filtered;
    }

    std::vector<Detection> ApplyDetectionFilters(std::vector<Detection> detections) const
    {
        detections = FilterDetectionsByOdomDistance(detections);
        detections = FilterDetectionsByOriginBbox(detections);
        detections = FilterDetectionsByGround(detections);
        detections = FilterDetectionsByVolume(detections);
        return detections;
    }

    bool HasNearbyDynamicDetection(const Track &track,
                                   const std::vector<Detection> &detections,
                                   const ros::Time &stamp) const
    {
        const Eigen::Vector3f track_pos = track.position();
        const bool has_timed_dynamic_observation =
            !track.last_dynamic_measurement_stamp.isZero() &&
            !stamp.isZero() &&
            stamp > track.last_dynamic_measurement_stamp;
        const double dynamic_displacement_limit = has_timed_dynamic_observation
            ? DynamicDisplacementLimit(track, stamp)
            : 0.0;
        for (const auto &detection : detections)
        {
            if (!PolicyFor(detection.source).uses_dynamic_reachability)
            {
                continue;
            }
            if ((detection.centroid - track_pos).norm() <= static_cast<float>(config_.max_center_distance))
            {
                return true;
            }
            if (has_timed_dynamic_observation &&
                (detection.centroid - track.last_dynamic_measurement_position).norm() <=
                    static_cast<float>(dynamic_displacement_limit))
            {
                return true;
            }
        }
        return false;
    }

    bool IsStaticSupportTrackEligible(const Track &track, const std::vector<Detection> &detections, const ros::Time &stamp) const
    {
        if (!track.confirmed)
        {
            return false;
        }
        if (config_.static_support_min_activation_speed > 0.0 && !track.static_support_speed_gate_passed)
        {
            if (config_.debug_static_support)
            {
                ROS_INFO_STREAM("frame_tracker visible static support skipped by speed gate: track_uid="
                                << track.track_uid
                                << " display_id=" << track.display_id
                                << " max_dynamic_speed=" << track.max_dynamic_speed
                                << " required=" << config_.static_support_min_activation_speed);
            }
            return false;
        }
        if (HasNearbyDynamicDetection(track, detections, stamp))
        {
            return false;
        }
        if (track.last_dynamic_measurement_stamp.isZero() || stamp.isZero())
        {
            return true;
        }
        const double stale_time = (stamp - track.last_dynamic_measurement_stamp).toSec();
        return stale_time >= 0.0 && stale_time <= config_.static_support_hold_time;
    }

    double PredictionResidualLimit(const Track &track,
                                   const Detection &detection,
                                   const ros::Time &stamp) const
    {
        if (!PolicyFor(detection.source).uses_dynamic_reachability)
        {
            return std::max(config_.max_center_distance, config_.static_support_max_residual);
        }
        if (track.last_dynamic_measurement_stamp.isZero() || stamp.isZero())
        {
            return config_.max_center_distance;
        }

        const double dt = (stamp - track.last_dynamic_measurement_stamp).toSec();
        if (dt <= 0.0)
        {
            return config_.max_center_distance;
        }

        const double acceleration_residual =
            0.5 * config_.max_target_acceleration * dt * dt +
            config_.association_padding;
        if (track.dynamic_measurement_hits < 2)
        {
            return std::max(config_.max_center_distance,
                            config_.max_target_speed * dt +
                                0.5 * ModelAccelerationMagnitude() * dt * dt +
                                acceleration_residual);
        }
        return std::max(config_.max_center_distance, acceleration_residual);
    }

    double DynamicDisplacementLimit(const Track &track, const ros::Time &stamp) const
    {
        if (track.last_dynamic_measurement_stamp.isZero() || stamp.isZero())
        {
            return std::numeric_limits<double>::infinity();
        }
        const double dt = (stamp - track.last_dynamic_measurement_stamp).toSec();
        if (dt <= 0.0)
        {
            return 0.0;
        }
        return config_.max_target_speed * dt +
            0.5 * ModelAccelerationMagnitude() * dt * dt +
            0.5 * config_.max_target_acceleration * dt * dt +
            config_.association_padding;
    }

    AssociationEvaluation EvaluateAssociation(const Track &track,
                                              const Detection &detection,
                                              const ros::Time &stamp) const
    {
        AssociationEvaluation evaluation;
        const DetectionPolicy &policy = PolicyFor(detection.source);
        if (policy.requires_preferred_track &&
            detection.preferred_track_uid != track.track_uid)
        {
            return evaluation;
        }

        evaluation.prediction_residual = (track.position() - detection.centroid).norm();
        evaluation.prediction_residual_limit =
            PredictionResidualLimit(track, detection, stamp);
        if (evaluation.prediction_residual > evaluation.prediction_residual_limit)
        {
            return evaluation;
        }

        if (policy.uses_dynamic_reachability)
        {
            evaluation.physical_displacement =
                (detection.centroid - track.last_dynamic_measurement_position).norm();
            evaluation.physical_displacement_limit =
                DynamicDisplacementLimit(track, stamp);
            if (evaluation.physical_displacement > evaluation.physical_displacement_limit)
            {
                return evaluation;
            }
        }

        const Eigen::Vector3f padded_track_dims =
            (track.dimensions + Eigen::Vector3f::Constant(2.0f * config_.association_padding))
                .cwiseMax(Eigen::Vector3f::Constant(0.05f));
        const Eigen::Vector3f padded_detection_dims =
            (detection.dimensions + Eigen::Vector3f::Constant(2.0f * config_.association_padding))
                .cwiseMax(Eigen::Vector3f::Constant(0.05f));
        evaluation.iou = ComputeAabbIoU(track.position(),
                                        padded_track_dims,
                                        detection.centroid,
                                        padded_detection_dims);
        if (evaluation.iou < config_.min_iou)
        {
            return evaluation;
        }

        const double measurement_variance =
            policy.uses_weak_measurement_variance
                ? config_.static_support_position_variance
                : config_.dynamic_position_variance;
        evaluation.mahalanobis_squared = InnovationMahalanobisSquared(
            track, detection, measurement_variance);
        if (evaluation.mahalanobis_squared >= 1e6 ||
            (policy.enforces_mahalanobis_gate &&
             evaluation.mahalanobis_squared > config_.association_mahalanobis_gate_sq))
        {
            return evaluation;
        }

        evaluation.cost = evaluation.mahalanobis_squared +
            0.1 * (1.0 - static_cast<double>(evaluation.iou));
        evaluation.eligible = true;
        return evaluation;
    }

    bool BuildStaticSupportDetectionForTrack(const Track &track,
                                             const SupportCloudFrame &support_frame,
                                             Detection *detection_out) const
    {
        if (!support_frame.cloud || !detection_out)
        {
            return false;
        }

        const Eigen::Vector3f predicted = track.position();
        const Eigen::Vector3f half_box =
            track.dimensions.cwiseMax(Eigen::Vector3f::Constant(0.05f)) *
                static_cast<float>(0.5 * (1.0 + config_.static_support_bbox_expand_ratio)) +
            Eigen::Vector3f::Constant(static_cast<float>(config_.static_support_bbox_padding));
        const Eigen::Vector3f core_half_box =
            track.dimensions.cwiseMax(Eigen::Vector3f::Constant(0.05f)) * 0.5f +
            Eigen::Vector3f::Constant(static_cast<float>(config_.static_support_bbox_padding));

        pcl::PointCloud<PointType>::Ptr roi_cloud(new pcl::PointCloud<PointType>());
        roi_cloud->reserve(support_frame.cloud->size());
        for (const auto &point : support_frame.cloud->points)
        {
            const Eigen::Vector3f point_vec(point.x, point.y, point.z);
            const Eigen::Vector3f delta = point_vec - predicted;
            if (std::abs(delta.x()) <= half_box.x() &&
                std::abs(delta.y()) <= half_box.y() &&
                std::abs(delta.z()) <= half_box.z())
            {
                roi_cloud->push_back(point);
            }
        }
        if (static_cast<int>(roi_cloud->size()) < config_.static_support_min_points)
        {
            return false;
        }

        pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>());
        tree->setInputCloud(roi_cloud);
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<PointType> extractor;
        extractor.setClusterTolerance(static_cast<float>(config_.static_support_cluster_tolerance));
        extractor.setMinClusterSize(config_.static_support_min_points);
        extractor.setMaxClusterSize(config_.static_support_cluster_max_points);
        extractor.setSearchMethod(tree);
        extractor.setInputCloud(roi_cloud);
        extractor.extract(cluster_indices);
        if (cluster_indices.empty())
        {
            return false;
        }

        bool found = false;
        float best_score = std::numeric_limits<float>::max();
        Detection best_detection;
        const Eigen::Vector3f dim_limit =
            track.dimensions.cwiseMax(Eigen::Vector3f::Constant(0.05f)) *
                static_cast<float>(1.0 + config_.static_support_bbox_expand_ratio) +
            Eigen::Vector3f::Constant(static_cast<float>(2.0 * config_.static_support_bbox_padding));
        const int min_core_points = std::max(1, (config_.static_support_min_points + 1) / 2);

        for (const auto &indices : cluster_indices)
        {
            Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
            Eigen::Vector3f min_point = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
            Eigen::Vector3f max_point = Eigen::Vector3f::Constant(-std::numeric_limits<float>::max());
            int core_points = 0;
            for (const int index : indices.indices)
            {
                const auto &point = roi_cloud->points[index];
                const Eigen::Vector3f point_vec(point.x, point.y, point.z);
                const Eigen::Vector3f point_delta = point_vec - predicted;
                centroid += point_vec;
                min_point = min_point.cwiseMin(point_vec);
                max_point = max_point.cwiseMax(point_vec);
                if (std::abs(point_delta.x()) <= core_half_box.x() &&
                    std::abs(point_delta.y()) <= core_half_box.y() &&
                    std::abs(point_delta.z()) <= core_half_box.z())
                {
                    ++core_points;
                }
            }
            if (core_points < min_core_points)
            {
                continue;
            }
            centroid /= static_cast<float>(indices.indices.size());
            const Eigen::Vector3f centroid_delta = centroid - predicted;
            if (std::abs(centroid_delta.x()) > core_half_box.x() ||
                std::abs(centroid_delta.y()) > core_half_box.y() ||
                std::abs(centroid_delta.z()) > core_half_box.z())
            {
                continue;
            }
            const Eigen::Vector3f dims =
                (max_point - min_point).cwiseMax(Eigen::Vector3f::Constant(0.05f));
            if (dims.x() > dim_limit.x() ||
                dims.y() > dim_limit.y() ||
                dims.z() > dim_limit.z())
            {
                continue;
            }
            const float residual = (centroid - predicted).norm();
            if (residual > static_cast<float>(config_.static_support_max_residual))
            {
                continue;
            }

            Detection candidate;
            candidate.centroid = centroid;
            candidate.min_point = min_point;
            candidate.max_point = max_point;
            candidate.dimensions = dims;
            candidate.point_count = indices.indices.size();
            candidate.source = DetectionSource::STATIC_SUPPORT;
            candidate.preferred_track_uid = track.track_uid;
            if (!PassesBboxSizePrior(candidate))
            {
                continue;
            }

            const float score =
                residual + 0.2f * dims.norm() - 0.001f * static_cast<float>(indices.indices.size()) -
                0.002f * static_cast<float>(core_points);
            if (score < best_score)
            {
                best_score = score;
                best_detection = candidate;
                found = true;
            }
        }

        if (!found)
        {
            return false;
        }
        *detection_out = best_detection;
        return true;
    }

    std::vector<Detection> BuildStaticSupportDetections(
        const std::vector<Detection> &dynamic_detections,
        const ros::Time &stamp,
        const PointCloudXYZI::ConstPtr &support_cloud) const
    {
        std::vector<Detection> support_detections;
        if (!config_.static_support_en || tracks_.empty() || !support_cloud)
        {
            return support_detections;
        }
        const SupportCloudFrame support_frame{stamp, support_cloud};
        for (const auto &track : tracks_)
        {
            if (!IsStaticSupportTrackEligible(track, dynamic_detections, stamp))
            {
                continue;
            }
            Detection support_detection;
            if (!BuildStaticSupportDetectionForTrack(track, support_frame, &support_detection))
            {
                if (config_.debug_static_support)
                {
                    ROS_INFO_STREAM("frame_tracker visible static support skipped: track_uid="
                                    << track.track_uid
                                    << " display_id=" << track.display_id
                                    << " predicted=" << FormatVec3(track.position()));
                }
                continue;
            }
            support_detections.push_back(support_detection);
            if (config_.debug_static_support)
            {
                ROS_INFO_STREAM("frame_tracker visible static support detection: track_uid="
                                << track.track_uid
                                << " display_id=" << track.display_id
                                << " centroid=" << FormatVec3(support_detection.centroid)
                                << " predicted=" << FormatVec3(track.position())
                                << " dims=" << FormatVec3(support_detection.dimensions)
                                << " points=" << support_detection.point_count);
            }
        }
        return support_detections;
    }

    void PredictTracks(const ros::Time &stamp)
    {
        for (auto &track : tracks_)
        {
            const double dt = ComputeDt(track.last_stamp, stamp);
            UpdateTransitionMatrix(track.kf, dt);
            UpdateControlMatrix(track.kf, dt);
            UpdateProcessNoiseCov(track.kf, dt);
            track.kf.predict(ControlInput());
            track.last_stamp = stamp;
            track.age += 1;
            track.matched_this_frame = false;
            track.last_iou = 0.0f;
        }
    }

    void ConsolidateDuplicateConfirmedTracks()
    {
        if (!config_.duplicate_guard_en || tracks_.size() < 2)
        {
            return;
        }

        std::vector<std::size_t> priority(tracks_.size());
        std::iota(priority.begin(), priority.end(), 0);
        std::sort(priority.begin(), priority.end(), [&](std::size_t left, std::size_t right)
        {
            const Track &left_track = tracks_[left];
            const Track &right_track = tracks_[right];
            if (left_track.age != right_track.age)
            {
                return left_track.age > right_track.age;
            }
            if (left_track.total_hits != right_track.total_hits)
            {
                return left_track.total_hits > right_track.total_hits;
            }
            return left_track.track_uid < right_track.track_uid;
        });

        std::vector<bool> remove_track(tracks_.size(), false);
        for (const std::size_t survivor_index : priority)
        {
            if (remove_track[survivor_index] || !tracks_[survivor_index].confirmed)
            {
                continue;
            }
            for (const std::size_t candidate_index : priority)
            {
                if (candidate_index == survivor_index || remove_track[candidate_index] ||
                    !tracks_[candidate_index].confirmed)
                {
                    continue;
                }
                if ((tracks_[survivor_index].position() - tracks_[candidate_index].position()).norm() >
                    static_cast<float>(config_.duplicate_guard_distance))
                {
                    continue;
                }
                Track &survivor = tracks_[survivor_index];
                const Track &candidate = tracks_[candidate_index];
                if (!survivor.matched_this_frame && candidate.matched_this_frame)
                {
                    const int survivor_uid = survivor.track_uid;
                    const int survivor_display_id = survivor.display_id;
                    const int survivor_age = survivor.age;
                    const int survivor_total_hits = survivor.total_hits;
                    const int survivor_dynamic_hits = survivor.dynamic_measurement_hits;
                    const int survivor_static_hits = survivor.static_support_hits;
                    const float survivor_max_dynamic_speed = survivor.max_dynamic_speed;
                    const bool survivor_speed_gate = survivor.static_support_speed_gate_passed;
                    const bool survivor_near_range = survivor.near_range_fast_candidate;
                    const ros::Time survivor_speed_pass_stamp =
                        survivor.last_speed_output_pass_stamp;
                    std::deque<Eigen::Vector3f> survivor_history = survivor.history;

                    survivor = candidate;
                    survivor.track_uid = survivor_uid;
                    survivor.display_id = survivor_display_id;
                    survivor.age = std::max(survivor_age, candidate.age);
                    survivor.total_hits = std::max(
                        survivor_total_hits + 1, candidate.total_hits);
                    const bool candidate_has_dynamic_measurement =
                        PolicyFor(candidate.last_measurement_source).updates_dynamic_history;
                    survivor.dynamic_measurement_hits = std::max(
                        survivor_dynamic_hits + (candidate_has_dynamic_measurement ? 1 : 0),
                        candidate.dynamic_measurement_hits);
                    survivor.static_support_hits = std::max(
                        survivor_static_hits + (candidate_has_dynamic_measurement ? 0 : 1),
                        candidate.static_support_hits);
                    survivor.max_dynamic_speed = std::max(
                        survivor_max_dynamic_speed, candidate.max_dynamic_speed);
                    survivor.static_support_speed_gate_passed =
                        survivor_speed_gate || candidate.static_support_speed_gate_passed;
                    survivor.near_range_fast_candidate =
                        survivor_near_range || candidate.near_range_fast_candidate;
                    if (survivor.last_speed_output_pass_stamp < survivor_speed_pass_stamp)
                    {
                        survivor.last_speed_output_pass_stamp = survivor_speed_pass_stamp;
                    }
                    survivor.history = std::move(survivor_history);
                }
                remove_track[candidate_index] = true;
                if (config_.debug_track_lifecycle)
                {
                    ROS_INFO_STREAM("frame_tracker delete duplicate track: track_uid="
                                    << tracks_[candidate_index].track_uid
                                    << " survivor_uid=" << tracks_[survivor_index].track_uid
                                    << " distance="
                                    << (tracks_[survivor_index].position() -
                                        tracks_[candidate_index].position()).norm());
                }
            }
        }

        std::vector<Track> consolidated;
        consolidated.reserve(tracks_.size());
        for (std::size_t index = 0; index < tracks_.size(); ++index)
        {
            if (!remove_track[index])
            {
                consolidated.push_back(std::move(tracks_[index]));
            }
        }
        tracks_ = std::move(consolidated);
    }

    void PruneTracksByOdomDistance()
    {
        if (!CanApplyDistanceFilter("tracks"))
        {
            return;
        }
        tracks_.erase(std::remove_if(tracks_.begin(),
                                     tracks_.end(),
                                     [&](const Track &track)
                                     {
                                         const OdomDistanceGate distance = ComputeOdomDistanceGate(track.position());
                                         if (PassesOdomDistanceGate(distance))
                                         {
                                             return false;
                                         }
                                         if (config_.debug_distance_rejects)
                                         {
                                             ROS_INFO_STREAM("frame_tracker delete track by odom distance: track_uid="
                                                             << track.track_uid
                                                             << " display_id=" << track.display_id
                                                             << " xy_distance=" << distance.xy_distance
                                                             << " max_xy=" << config_.target_max_xy_distance
                                                             << " z_distance=" << distance.z_distance
                                                             << " max_z=" << config_.target_max_z_distance
                                                             << " pos=" << FormatVec3(track.position())
                                                             << " odom=" << FormatVec3(current_odom_position_));
                                         }
                                         return true;
                                     }),
                      tracks_.end());
    }

    void PruneTracksByOriginBbox()
    {
        if (!config_.target_origin_bbox_filter_en)
        {
            return;
        }
        tracks_.erase(std::remove_if(tracks_.begin(),
                                     tracks_.end(),
                                     [&](const Track &track)
                                     {
                                         const Eigen::Vector3f position = track.position();
                                         if (PassesOriginBboxGate(position))
                                         {
                                             return false;
                                         }
                                         if (config_.debug_distance_rejects)
                                         {
                                             ROS_INFO_STREAM("frame_tracker delete track by origin bbox: track_uid="
                                                             << track.track_uid
                                                             << " display_id=" << track.display_id
                                                             << " pos=" << FormatVec3(position)
                                                             << " min=(" << config_.target_origin_bbox_min_x << ", "
                                                             << config_.target_origin_bbox_min_y << ", "
                                                             << config_.target_origin_bbox_min_z << ")"
                                                             << " max=(" << config_.target_origin_bbox_max_x << ", "
                                                             << config_.target_origin_bbox_max_y << ", "
                                                             << config_.target_origin_bbox_max_z << ")");
                                         }
                                         return true;
                                     }),
                      tracks_.end());
    }

    void PruneTracksByGround()
    {
        if (!config_.target_ground_filter_en)
        {
            return;
        }
        tracks_.erase(std::remove_if(tracks_.begin(),
                                     tracks_.end(),
                                     [&](const Track &track)
                                     {
                                         const Eigen::Vector3f position = track.position();
                                         if (PassesGroundGate(position))
                                         {
                                             return false;
                                         }
                                         if (config_.debug_distance_rejects)
                                         {
                                             ROS_INFO_STREAM("frame_tracker delete track by ground filter: track_uid="
                                                             << track.track_uid
                                                             << " display_id=" << track.display_id
                                                             << " z=" << position.z()
                                                             << " min_world_z=" << config_.target_min_world_z
                                                             << " pos=" << FormatVec3(position));
                                         }
                                         return true;
                                     }),
                      tracks_.end());
    }

    void PruneTracksByVolume()
    {
        if (!config_.target_volume_filter_en)
        {
            return;
        }
        tracks_.erase(std::remove_if(tracks_.begin(),
                                     tracks_.end(),
                                     [&](const Track &track)
                                     {
                                         const float volume = ComputeBboxVolume(track.dimensions);
                                         if (PassesVolumeGate(track.dimensions))
                                         {
                                             return false;
                                         }
                                         if (config_.debug_bbox_prior_rejects)
                                         {
                                             ROS_INFO_STREAM("frame_tracker delete track by bbox volume: track_uid="
                                                             << track.track_uid
                                                             << " display_id=" << track.display_id
                                                             << " volume=" << volume
                                                             << " min_volume=" << config_.target_min_bbox_volume
                                                             << " max_volume=" << config_.target_max_bbox_volume
                                                             << " dims=" << FormatVec3(track.dimensions)
                                                             << " pos=" << FormatVec3(track.position()));
                                         }
                                         return true;
                                     }),
                      tracks_.end());
    }

    void AssociateAndUpdate(const std::vector<Detection> &detections, const ros::Time &stamp)
    {
        if (tracks_.empty())
        {
            for (const auto &detection : detections)
            {
                if (!CanInitializeTrack(detection))
                {
                    continue;
                }
                tracks_.push_back(CreateTrack(detection, stamp));
            }
            return;
        }

        if (detections.empty())
        {
            for (auto &track : tracks_)
            {
                RegisterMiss(track);
            }
            return;
        }

        std::vector<std::vector<double>> costs(tracks_.size(), std::vector<double>(detections.size(), 1.0));
        std::vector<std::vector<float>> ious(tracks_.size(), std::vector<float>(detections.size(), 0.0f));
        std::vector<std::vector<float>> center_distances(tracks_.size(), std::vector<float>(detections.size(), 0.0f));
        std::vector<std::vector<AssociationEvaluation>> evaluations(
            tracks_.size(), std::vector<AssociationEvaluation>(detections.size()));

        for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index)
        {
            for (std::size_t det_index = 0; det_index < detections.size(); ++det_index)
            {
                evaluations[track_index][det_index] = EvaluateAssociation(
                    tracks_[track_index], detections[det_index], stamp);
                const AssociationEvaluation &evaluation = evaluations[track_index][det_index];
                center_distances[track_index][det_index] = evaluation.prediction_residual;
                ious[track_index][det_index] = evaluation.iou;
                costs[track_index][det_index] = evaluation.cost;
            }
        }

        DebugPrintAssociationMatrix(detections, center_distances, ious);

        std::vector<int> assignment = SolveHungarian(costs);
        std::vector<bool> relaxed_match(tracks_.size(), false);
        std::vector<bool> detection_claimed(detections.size(), false);
        for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index)
        {
            const int det_index = assignment[track_index];
            if (det_index >= 0 && det_index < static_cast<int>(detections.size()) &&
                evaluations[track_index][det_index].eligible)
            {
                detection_claimed[det_index] = true;
            }
        }

        if (config_.duplicate_guard_en)
        {
            struct RelaxedCandidate
            {
                float distance = 0.0f;
                std::size_t track_index = 0;
                std::size_t detection_index = 0;
            };
            std::vector<RelaxedCandidate> candidates;
            for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index)
            {
                const int assigned_detection = assignment[track_index];
                if ((assigned_detection >= 0 &&
                     assigned_detection < static_cast<int>(detections.size()) &&
                     evaluations[track_index][assigned_detection].eligible) ||
                    !tracks_[track_index].confirmed)
                {
                    continue;
                }
                for (std::size_t det_index = 0; det_index < detections.size(); ++det_index)
                {
                    if (detection_claimed[det_index] ||
                        !PolicyFor(detections[det_index].source).uses_dynamic_reachability)
                    {
                        continue;
                    }
                    const float prediction_distance =
                        (tracks_[track_index].position() - detections[det_index].centroid).norm();
                    const Track &track = tracks_[track_index];
                    const bool has_timed_dynamic_observation =
                        !track.last_dynamic_measurement_stamp.isZero() &&
                        !stamp.isZero() &&
                        stamp > track.last_dynamic_measurement_stamp;
                    const float observation_distance =
                        (track.last_dynamic_measurement_position -
                         detections[det_index].centroid).norm();
                    const bool observation_reachable =
                        has_timed_dynamic_observation &&
                        observation_distance <= static_cast<float>(
                            DynamicDisplacementLimit(track, stamp));
                    if (prediction_distance <=
                            static_cast<float>(config_.duplicate_guard_distance) ||
                        observation_reachable)
                    {
                        const float recovery_distance = observation_reachable
                            ? observation_distance
                            : prediction_distance;
                        candidates.push_back(
                            {recovery_distance, track_index, det_index});
                    }
                }
            }
            std::sort(candidates.begin(), candidates.end(), [&](const RelaxedCandidate &left,
                                                                 const RelaxedCandidate &right)
            {
                if (left.distance != right.distance)
                {
                    return left.distance < right.distance;
                }
                const int left_uid = tracks_[left.track_index].track_uid;
                const int right_uid = tracks_[right.track_index].track_uid;
                if (left_uid != right_uid)
                {
                    return left_uid < right_uid;
                }
                return left.detection_index < right.detection_index;
            });
            for (const RelaxedCandidate &candidate : candidates)
            {
                if (relaxed_match[candidate.track_index] ||
                    detection_claimed[candidate.detection_index])
                {
                    continue;
                }
                assignment[candidate.track_index] =
                    static_cast<int>(candidate.detection_index);
                relaxed_match[candidate.track_index] = true;
                detection_claimed[candidate.detection_index] = true;
            }
        }

        std::vector<bool> detection_used(detections.size(), false);

        for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index)
        {
            const int det_index = assignment[track_index];
            if (det_index < 0 || det_index >= static_cast<int>(detections.size()))
            {
                RegisterMiss(tracks_[track_index]);
                continue;
            }

            const AssociationEvaluation &evaluation = evaluations[track_index][det_index];
            if (detection_used[det_index] ||
                (!evaluation.eligible && !relaxed_match[track_index]))
            {
                if (config_.debug_association_rejects)
                {
                    ROS_INFO_STREAM("frame_tracker reject match: track_uid=" << tracks_[track_index].track_uid
                                    << " display_id=" << tracks_[track_index].display_id
                                    << " det_index=" << det_index
                                    << " detection_used=" << detection_used[det_index]
                                    << " prediction_residual=" << evaluation.prediction_residual
                                    << " prediction_residual_limit=" << evaluation.prediction_residual_limit
                                    << " physical_displacement=" << evaluation.physical_displacement
                                    << " physical_displacement_limit=" << evaluation.physical_displacement_limit
                                    << " mahalanobis_sq=" << evaluation.mahalanobis_squared
                                    << " mahalanobis_gate_sq=" << config_.association_mahalanobis_gate_sq
                                    << " iou=" << evaluation.iou
                                    << " min_iou=" << config_.min_iou);
                }
                RegisterMiss(tracks_[track_index]);
                continue;
            }

            detection_used[det_index] = true;
            UpdateTrack(tracks_[track_index],
                        detections[det_index],
                        evaluation.iou,
                        stamp,
                        relaxed_match[track_index]);
        }

        for (std::size_t det_index = 0; det_index < detections.size(); ++det_index)
        {
            if (!detection_used[det_index])
            {
                if (!CanInitializeTrack(detections[det_index]))
                {
                    continue;
                }
                tracks_.push_back(CreateTrack(detections[det_index], stamp));
            }
        }
    }

    void PruneTracks(const ros::Time &stamp)
    {
        tracks_.erase(std::remove_if(tracks_.begin(),
                                     tracks_.end(),
                                     [&](const Track &track)
                                     {
                                         if (track.missed_frames > config_.track_lifetime_frames)
                                         {
                                             if (track.confirmed &&
                                                 IsWithinLostPredictionWindow(track, stamp))
                                             {
                                                 return false;
                                             }
                                             if (config_.debug_track_lifecycle)
                                             {
                                                 ROS_INFO_STREAM("frame_tracker delete track: track_uid=" << track.track_uid
                                                                 << " display_id=" << track.display_id
                                                                 << " reason=missed_frames"
                                                                 << " missed_frames=" << track.missed_frames
                                                                 << " limit=" << config_.track_lifetime_frames
                                                                 << " confirmed=" << track.confirmed
                                                                 << " total_hits=" << track.total_hits
                                                                 << " age=" << track.age);
                                             }
                                             return true;
                                         }
                                         const int required_hits = ConfirmationHitThreshold(track);
                                         if (!track.confirmed &&
                                             track.age >= config_.init_window_size &&
                                             CountHits(track.hit_window) < required_hits)
                                         {
                                             if (config_.debug_track_lifecycle)
                                             {
                                                 ROS_INFO_STREAM("frame_tracker delete track: track_uid=" << track.track_uid
                                                                 << " display_id=" << track.display_id
                                                                 << " reason=init_failed"
                                                                 << " hits=" << CountHits(track.hit_window)
                                                                 << " required_hits=" << required_hits
                                                                 << " near_range_fast=" << track.near_range_fast_candidate
                                                                 << " age=" << track.age
                                                                 << " init_window_size=" << config_.init_window_size);
                                             }
                                             return true;
                                         }
                                         return false;
                                     }),
                      tracks_.end());
    }

    void AppendTrackHistory()
    {
        for (auto &track : tracks_)
        {
            const Eigen::Vector3f pos = track.position();
            if (!track.history.empty())
            {
                const auto &last = track.history.back();
                if ((last - pos).cwiseAbs().maxCoeff() < 1e-4f)
                {
                    continue;
                }
            }
            track.history.push_back(pos);
            while (static_cast<int>(track.history.size()) > config_.track_history_size)
            {
                track.history.pop_front();
            }
        }
    }

    bool IsNearRangeDetection(const Detection &detection) const
    {
        if (!config_.near_range_fast_confirm_en ||
            config_.near_range_fast_confirm_distance <= 0.0 ||
            !has_odom_)
        {
            return false;
        }
        return (detection.centroid - current_odom_position_).norm() <=
            static_cast<float>(config_.near_range_fast_confirm_distance);
    }

    int ConfirmationHitThreshold(const Track &track) const
    {
        if (!config_.near_range_fast_confirm_en ||
            !track.near_range_fast_candidate ||
            !has_odom_)
        {
            return config_.min_init_hits;
        }
        return config_.near_range_fast_confirm_hits;
    }

    bool PassesSpeedOutputGate(Track &track, const ros::Time &stamp) const
    {
        if (!config_.target_speed_filter_en)
        {
            return true;
        }
        if (PolicyFor(track.last_measurement_source).preserves_velocity ||
            track.static_support_hits > 0)
        {
            return true;
        }

        const float speed = track.velocity().norm();
        const float safe_speed = std::isfinite(speed) ? speed : 0.0f;
        const ros::Time gate_stamp = stamp.isZero() ? track.last_stamp : stamp;
        if (safe_speed >= static_cast<float>(config_.target_min_speed))
        {
            if (!gate_stamp.isZero())
            {
                track.last_speed_output_pass_stamp = gate_stamp;
            }
            return true;
        }

        if (track.last_speed_output_pass_stamp.isZero())
        {
            return false;
        }
        if (gate_stamp.isZero())
        {
            return true;
        }

        const double low_speed_time = (gate_stamp - track.last_speed_output_pass_stamp).toSec();
        return low_speed_time < 0.0 || low_speed_time <= config_.target_speed_filter_grace_time;
    }

    bool IsWithinLostPredictionWindow(const Track &track, const ros::Time &stamp) const
    {
        if (track.last_dynamic_measurement_stamp.isZero() || stamp.isZero())
        {
            return false;
        }
        const double elapsed = (stamp - track.last_dynamic_measurement_stamp).toSec();
        return elapsed >= 0.0 &&
            elapsed <= config_.lost_track_prediction_time + 1e-6;
    }

    bool CanInitializeTrack(const Detection &detection) const
    {
        return PolicyFor(detection.source).can_initialize_track &&
            detection.point_count >= static_cast<std::size_t>(config_.cluster_min_points);
    }

    Track CreateTrack(const Detection &detection, const ros::Time &stamp)
    {
        if (!CanInitializeTrack(detection))
        {
            throw std::logic_error("detection does not satisfy track initialization policy");
        }
        Track track;
        track.track_uid = next_track_uid_++;
        track.kf = CreateKalmanFilter(detection.centroid);
        track.dimensions = detection.dimensions;
        track.point_count = detection.point_count;
        track.age = 1;
        track.total_hits = 1;
        track.missed_frames = 0;
        track.near_range_fast_candidate = IsNearRangeDetection(detection);
        track.confirmed = (ConfirmationHitThreshold(track) <= 1);
        track.matched_this_frame = true;
        track.last_iou = 1.0f;
        track.last_stamp = stamp;
        track.last_measurement_stamp = stamp;
        track.last_dynamic_measurement_stamp = stamp;
        track.last_dynamic_measurement_position = detection.centroid;
        track.dynamic_measurement_hits = 1;
        track.last_measurement_source = detection.source;
        track.static_support_hits = 0;
        track.static_support_speed_gate_passed = false;
        track.max_dynamic_speed = 0.0f;
        track.hit_window.push_back(true);
        if (track.confirmed)
        {
            track.display_id = next_display_id_++;
        }
        if (config_.debug_track_lifecycle)
        {
            ROS_INFO_STREAM("frame_tracker create track: track_uid=" << track.track_uid
                            << " display_id=" << track.display_id
                            << " confirmed=" << track.confirmed
                            << " source=" << DetectionSourceName(detection.source)
                            << " point_count=" << detection.point_count
                            << " centroid=(" << detection.centroid.x() << ", "
                            << detection.centroid.y() << ", "
                            << detection.centroid.z() << ")"
                            << " dims=(" << detection.dimensions.x() << ", "
                            << detection.dimensions.y() << ", "
                            << detection.dimensions.z() << ")");
        }
        return track;
    }

    void UpdateTrack(Track &track,
                     const Detection &detection,
                     float iou,
                     const ros::Time &stamp,
                     bool observation_recovery = false)
    {
        const DetectionPolicy &policy = PolicyFor(detection.source);
        cv::Mat measurement(3, 1, CV_32F);
        const Eigen::Vector3f previous_velocity = track.velocity();
        Eigen::Vector3f bootstrap_velocity = Eigen::Vector3f::Zero();
        bool bootstrap_dynamic_velocity = false;
        if (policy.can_bootstrap_velocity &&
            (track.dynamic_measurement_hits == 1 || observation_recovery) &&
            !track.last_dynamic_measurement_stamp.isZero() &&
            !stamp.isZero())
        {
            const double measurement_dt = (stamp - track.last_dynamic_measurement_stamp).toSec();
            if (measurement_dt > 1e-4)
            {
                bootstrap_velocity =
                    (detection.centroid - track.last_dynamic_measurement_position) /
                    static_cast<float>(measurement_dt) +
                    0.5f * ModelAcceleration() * static_cast<float>(measurement_dt);
                const float speed = bootstrap_velocity.norm();
                if (std::isfinite(speed) && speed > static_cast<float>(config_.max_target_speed))
                {
                    bootstrap_velocity *=
                        static_cast<float>(config_.max_target_speed) / speed;
                }
                bootstrap_dynamic_velocity = bootstrap_velocity.allFinite();
            }
        }
        measurement.at<float>(0) = detection.centroid.x();
        measurement.at<float>(1) = detection.centroid.y();
        measurement.at<float>(2) = detection.centroid.z();
        const cv::Mat dynamic_measurement_noise = track.kf.measurementNoiseCov.clone();
        if (policy.uses_weak_measurement_variance)
        {
            track.kf.measurementNoiseCov = cv::Mat::eye(3, 3, CV_32F) *
                static_cast<float>(config_.static_support_position_variance);
        }
        track.kf.correct(measurement);
        track.kf.measurementNoiseCov = dynamic_measurement_noise;
        if (bootstrap_dynamic_velocity)
        {
            track.kf.statePost.at<float>(3) = bootstrap_velocity.x();
            track.kf.statePost.at<float>(4) = bootstrap_velocity.y();
            track.kf.statePost.at<float>(5) = bootstrap_velocity.z();
        }
        else if (policy.preserves_velocity)
        {
            track.kf.statePost.at<float>(3) = previous_velocity.x();
            track.kf.statePost.at<float>(4) = previous_velocity.y();
            track.kf.statePost.at<float>(5) = previous_velocity.z();
        }
        ClampVelocityToDesignEnvelope(track.kf);

        track.point_count = detection.point_count;
        track.total_hits += 1;
        track.missed_frames = 0;
        track.matched_this_frame = true;
        track.last_iou = iou;
        track.last_stamp = stamp;
        track.last_measurement_stamp = stamp;
        track.last_measurement_source = detection.source;
        if (policy.updates_dynamic_history)
        {
            track.dimensions = 0.7f * track.dimensions + 0.3f * detection.dimensions;
            track.last_dynamic_measurement_stamp = stamp;
            track.last_dynamic_measurement_position = detection.centroid;
            track.dynamic_measurement_hits += 1;
            track.static_support_hits = 0;
            const float dynamic_speed = track.velocity().norm();
            if (std::isfinite(dynamic_speed))
            {
                track.max_dynamic_speed = std::max(track.max_dynamic_speed, dynamic_speed);
                if (config_.static_support_min_activation_speed <= 0.0 ||
                    dynamic_speed >= static_cast<float>(config_.static_support_min_activation_speed))
                {
                    track.static_support_speed_gate_passed = true;
                }
            }
        }
        else
        {
            track.static_support_hits += 1;
        }

        track.hit_window.push_back(true);
        while (static_cast<int>(track.hit_window.size()) > config_.init_window_size)
        {
            track.hit_window.pop_front();
        }

        if (policy.updates_dynamic_history && IsNearRangeDetection(detection))
        {
            track.near_range_fast_candidate = true;
        }

        const int required_hits = ConfirmationHitThreshold(track);
        if (!track.confirmed && CountHits(track.hit_window) >= required_hits)
        {
            track.confirmed = true;
            if (track.display_id < 0)
            {
                track.display_id = next_display_id_++;
            }
            if (config_.debug_track_lifecycle)
            {
                ROS_INFO_STREAM("frame_tracker confirm track: track_uid=" << track.track_uid
                                << " display_id=" << track.display_id
                                << " total_hits=" << track.total_hits
                                << " age=" << track.age
                                << " hit_window_hits=" << CountHits(track.hit_window)
                                << "/" << config_.init_window_size
                                << " required_hits=" << required_hits
                                << " near_range_fast=" << track.near_range_fast_candidate);
            }
        }
    }

    void ClampVelocityToDesignEnvelope(cv::KalmanFilter &kf) const
    {
        Eigen::Vector3f velocity(kf.statePost.at<float>(3),
                                 kf.statePost.at<float>(4),
                                 kf.statePost.at<float>(5));
        const float speed = velocity.norm();
        if (!std::isfinite(speed))
        {
            velocity.setZero();
        }
        else if (speed > static_cast<float>(config_.max_target_speed))
        {
            velocity *= static_cast<float>(config_.max_target_speed) / speed;
        }
        kf.statePost.at<float>(3) = velocity.x();
        kf.statePost.at<float>(4) = velocity.y();
        kf.statePost.at<float>(5) = velocity.z();
    }

    void RegisterMiss(Track &track)
    {
        track.missed_frames += 1;
        track.matched_this_frame = false;
        track.last_iou = 0.0f;
        track.hit_window.push_back(false);
        while (static_cast<int>(track.hit_window.size()) > config_.init_window_size)
        {
            track.hit_window.pop_front();
        }
    }

    cv::KalmanFilter CreateKalmanFilter(const Eigen::Vector3f &initial_position) const
    {
        cv::KalmanFilter kf(6, 3, 3, CV_32F);
        kf.transitionMatrix = cv::Mat::eye(6, 6, CV_32F);
        kf.measurementMatrix = cv::Mat::zeros(3, 6, CV_32F);
        kf.measurementMatrix.at<float>(0, 0) = 1.0f;
        kf.measurementMatrix.at<float>(1, 1) = 1.0f;
        kf.measurementMatrix.at<float>(2, 2) = 1.0f;

        kf.processNoiseCov = cv::Mat::zeros(6, 6, CV_32F);
        kf.measurementNoiseCov = cv::Mat::zeros(3, 3, CV_32F);
        for (int axis = 0; axis < 3; ++axis)
        {
            kf.measurementNoiseCov.at<float>(axis, axis) = static_cast<float>(config_.dynamic_position_variance);
        }

        kf.errorCovPost = cv::Mat::eye(6, 6, CV_32F);
        kf.statePost = cv::Mat::zeros(6, 1, CV_32F);
        kf.statePost.at<float>(0) = initial_position.x();
        kf.statePost.at<float>(1) = initial_position.y();
        kf.statePost.at<float>(2) = initial_position.z();
        UpdateProcessNoiseCov(kf, config_.default_dt);
        return kf;
    }

    double InnovationMahalanobisSquared(const Track &track,
                                        const Detection &detection,
                                        double measurement_variance) const
    {
        cv::Mat innovation(3, 1, CV_32F);
        const Eigen::Vector3f predicted = track.position();
        innovation.at<float>(0) = detection.centroid.x() - predicted.x();
        innovation.at<float>(1) = detection.centroid.y() - predicted.y();
        innovation.at<float>(2) = detection.centroid.z() - predicted.z();
        const cv::Mat innovation_covariance =
            track.kf.measurementMatrix * track.kf.errorCovPre *
                track.kf.measurementMatrix.t() +
            cv::Mat::eye(3, 3, CV_32F) * static_cast<float>(measurement_variance);
        cv::Mat normalized;
        if (!cv::solve(innovation_covariance, innovation, normalized, cv::DECOMP_CHOLESKY))
        {
            return 1e6;
        }
        const double distance = innovation.dot(normalized);
        return std::isfinite(distance) ? std::max(0.0, distance) : 1e6;
    }

    void UpdateTransitionMatrix(cv::KalmanFilter &kf, double dt) const
    {
        const float dt_f = static_cast<float>(std::max(dt, 1e-3));
        kf.transitionMatrix = cv::Mat::eye(6, 6, CV_32F);
        kf.transitionMatrix.at<float>(0, 3) = dt_f;
        kf.transitionMatrix.at<float>(1, 4) = dt_f;
        kf.transitionMatrix.at<float>(2, 5) = dt_f;
    }

    void UpdateControlMatrix(cv::KalmanFilter &kf, double dt) const
    {
        const float dt_f = static_cast<float>(std::max(dt, 1e-3));
        const float half_dt2 = 0.5f * dt_f * dt_f;
        kf.controlMatrix = cv::Mat::zeros(6, 3, CV_32F);
        for (int axis = 0; axis < 3; ++axis)
        {
            kf.controlMatrix.at<float>(axis, axis) = half_dt2;
            kf.controlMatrix.at<float>(axis + 3, axis) = dt_f;
        }
    }

    cv::Mat ControlInput() const
    {
        const Eigen::Vector3f acceleration = ModelAcceleration();
        cv::Mat input(3, 1, CV_32F);
        input.at<float>(0) = acceleration.x();
        input.at<float>(1) = acceleration.y();
        input.at<float>(2) = acceleration.z();
        return input;
    }

    void UpdateProcessNoiseCov(cv::KalmanFilter &kf, double dt) const
    {
        const float dt_f = static_cast<float>(std::max(dt, 1e-3));
        const float dt2 = dt_f * dt_f;
        const float dt3 = dt2 * dt_f;
        const float dt4 = dt2 * dt2;
        const float acceleration_variance = static_cast<float>(
            config_.process_acceleration_stddev * config_.process_acceleration_stddev);
        kf.processNoiseCov = cv::Mat::zeros(6, 6, CV_32F);
        for (int axis = 0; axis < 3; ++axis)
        {
            kf.processNoiseCov.at<float>(axis, axis) =
                0.25f * dt4 * acceleration_variance;
            kf.processNoiseCov.at<float>(axis, axis + 3) =
                0.5f * dt3 * acceleration_variance;
            kf.processNoiseCov.at<float>(axis + 3, axis) =
                0.5f * dt3 * acceleration_variance;
            kf.processNoiseCov.at<float>(axis + 3, axis + 3) =
                dt2 * acceleration_variance;
        }
    }

    double ComputeDt(const ros::Time &last_stamp, const ros::Time &current_stamp) const
    {
        if (last_stamp.isZero() || current_stamp.isZero())
        {
            return config_.default_dt;
        }

        const double dt = (current_stamp - last_stamp).toSec();
        if (dt <= 1e-4)
        {
            return config_.default_dt;
        }
        return dt;
    }

    // Backward jumps at or beyond this magnitude are treated as a sensor/odometry
    // timestamp reset (e.g. FAST-LIO's "lidar loop back" recovery) and resync instead
    // of permanently rejecting every subsequent frame. Same threshold value as
    // BoundedFrameQueue::kBackwardJumpResetSeconds, kept independent so this component
    // self-defends regardless of how it's called.
    static constexpr double kBackwardJumpResetSeconds = 5.0;

    TrackerConfig config_;
    bool has_odom_ = false;
    Eigen::Vector3f current_odom_position_ = Eigen::Vector3f::Zero();

    int next_track_uid_ = 0;
    int next_display_id_ = 0;
    std::size_t frame_index_ = 0;
    ros::Time last_frame_stamp_;
    std::vector<Track> tracks_;
};

TrackerResult FrameTracker::Impl::Update(const TrackerFrame &frame)
{
    const bool reset_for_backward_jump =
        !frame.header.stamp.isZero() &&
        !last_frame_stamp_.isZero() &&
        frame.header.stamp <= last_frame_stamp_ &&
        (last_frame_stamp_ - frame.header.stamp).toSec() >= kBackwardJumpResetSeconds;
    if (!frame.header.stamp.isZero() &&
        !last_frame_stamp_.isZero() &&
        frame.header.stamp <= last_frame_stamp_ &&
        (last_frame_stamp_ - frame.header.stamp).toSec() < kBackwardJumpResetSeconds)
    {
        throw std::invalid_argument("frame tracker requires strictly increasing timestamps");
    }
    if (reset_for_backward_jump)
    {
        tracks_.clear();
        frame_index_ = 0;
    }
    if (!frame.header.stamp.isZero())
    {
        last_frame_stamp_ = frame.header.stamp;
    }
    ++frame_index_;
    const ros::WallTime update_start = ros::WallTime::now();

    has_odom_ = frame.has_odom;
    current_odom_position_ = frame.odom_position;

    TrackerResult result;
    result.header = frame.header;
    if (frame.preclustered_detections)
    {
        result.input_mode = TrackerInputMode::PRECLUSTERED_DETECTIONS;
        for (const auto &detection : *frame.preclustered_detections)
        {
            result.primary_point_count += detection.point_count;
        }
    }
    else
    {
        result.input_mode = frame.frame_out_available
                                ? TrackerInputMode::FRAME_OUT
                                : TrackerInputMode::POINT_OUT_ONLY;
        result.primary_point_count =
            frame.frame_out_available && frame.frame_out ? frame.frame_out->size() : 0;
    }
    result.point_out_point_count = frame.point_out ? frame.point_out->size() : 0;
    result.support_point_count = frame.support_cloud ? frame.support_cloud->size() : 0;
    result.cloud_odom_stamp_delta_ms = frame.cloud_odom_stamp_delta_ms;

    PredictTracks(frame.header.stamp);
    ConsolidateDuplicateConfirmedTracks();
    std::vector<Detection> detections;
    if (frame.preclustered_detections)
    {
        detections = ApplyDetectionFilters(
            MergeNearbyDynamicDetections(
                BuildPreclusteredDetections(*frame.preclustered_detections)));
    }
    else if (frame.frame_out_available)
    {
        detections = ApplyDetectionFilters(ExtractDetections(frame.frame_out));
    }
    if (!frame.preclustered_detections)
    {
        AppendPointOutDetections(
            detections, frame.point_out, !frame.frame_out_available);
    }
    std::vector<Detection> static_support_detections =
        BuildStaticSupportDetections(detections, frame.header.stamp, frame.support_cloud);
    detections.insert(detections.end(),
                      static_support_detections.begin(),
                      static_support_detections.end());
    detections = ApplyDetectionFilters(std::move(detections));
    result.detection_count = detections.size();

    DebugPrintFrameState(frame.header, detections);
    AssociateAndUpdate(detections, frame.header.stamp);
    ConsolidateDuplicateConfirmedTracks();
    PruneTracksByOdomDistance();
    PruneTracksByOriginBbox();
    PruneTracksByGround();
    PruneTracksByVolume();
    PruneTracks(frame.header.stamp);
    AppendTrackHistory();
    result.active_track_count = tracks_.size();

    const auto make_snapshot = [](const Track &track) {
        TrackSnapshot snapshot;
        snapshot.id = track.display_id < 0 ? static_cast<uint32_t>(track.track_uid)
                                          : static_cast<uint32_t>(track.display_id);
        snapshot.track_uid = static_cast<uint32_t>(track.track_uid);
        snapshot.confirmed = track.confirmed;
        snapshot.matched = track.matched_this_frame;
        snapshot.age = static_cast<uint32_t>(track.age);
        snapshot.total_hits = static_cast<uint32_t>(track.total_hits);
        snapshot.missed_frames = static_cast<uint32_t>(track.missed_frames);
        snapshot.point_count = static_cast<uint32_t>(track.point_count);
        snapshot.last_iou = track.last_iou;
        snapshot.position = track.position();
        snapshot.velocity = track.velocity();
        snapshot.dimensions = track.dimensions;
        snapshot.history.assign(track.history.begin(), track.history.end());
        return snapshot;
    };

    for (auto &track : tracks_)
    {
        if (!track.confirmed)
        {
            result.tentative_tracks.push_back(make_snapshot(track));
            continue;
        }
        if (!PassesSpeedOutputGate(track, frame.header.stamp))
        {
            continue;
        }
        if (!IsWithinLostPredictionWindow(track, frame.header.stamp))
        {
            continue;
        }
        result.tracks.push_back(make_snapshot(track));
    }

    result.update_ms = (ros::WallTime::now() - update_start).toSec() * 1000.0;
    return result;
}

FrameTracker::FrameTracker(const TrackerConfig &config)
    : impl_(new Impl(config))
{
}

FrameTracker::~FrameTracker() = default;
FrameTracker::FrameTracker(FrameTracker &&) noexcept = default;
FrameTracker &FrameTracker::operator=(FrameTracker &&) noexcept = default;

TrackerResult FrameTracker::update(const TrackerFrame &frame)
{
    return impl_->Update(frame);
}

const TrackerConfig &FrameTracker::config() const
{
    return impl_->config();
}

} // namespace m_detector
