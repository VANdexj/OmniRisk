#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <vector>
#include <string>
#include "sensor_simulator.cuh"
#include "maps.hpp"

using namespace raycast;
namespace fs = std::filesystem;

// ─── Dynamic obstacle structs ────────────────────────────────────────────────

struct SphereObstacleSpec
{
    Eigen::Vector3f center{Eigen::Vector3f::Zero()};
    float radius{0.25f};
    Eigen::Vector3f velocity{Eigen::Vector3f::Zero()};
    Eigen::Vector3f acceleration{Eigen::Vector3f::Zero()};   // v5.2: usually zero
    int32_t obj_id{0};                                       // assigned per-sample, 0 = invalid
};

struct CylinderObstacleSpec
{
    Eigen::Vector3f center{Eigen::Vector3f::Zero()};
    float radius{0.35f};
    float height{1.5f};
    Eigen::Vector3f velocity{Eigen::Vector3f::Zero()};
    Eigen::Vector3f acceleration{Eigen::Vector3f::Zero()};   // v5.2: usually zero
    int32_t obj_id{0};
};

// v5: tracker-noise injection at dataset generation time.
struct TrackerNoiseConfig
{
    float sigma_v_default{0.10f};
    float sigma_a_default{0.30f};
    float id_swap_prob_default{0.02f};
    float new_track_bias_std{0.15f};
    int   age_default{3};         // dataset frames are i.i.d.; treat as "stable" track
    // degradation scenario mix (per-object, independent draws):
    float new_track_ratio{0.10f};   // age=0 + high uncertainty
    float reappear_ratio{0.05f};    // boost σ_v by 3x
    float accel_spike_ratio{0.05f}; // boost σ_v by 2x, σ_a by 3x
    float id_swap_ratio{0.05f};     // v_tracker zeroed
};

// CPA inverse placement: five buckets (ids in sync with BUCKET_* constants in policy/dataset.py)
//   1 no_dynamic  : empty mask, static baseline
//   2 intercept   : small miss + t* inside the planning window (forward flight collides, sideways flight escapes)
//   3 near_miss   : miss within the ball-radius + body narrow band (ranking boundary)
//   4 no_threat   : large miss or cannot catch up within the window (low score)
//   5 all_blocked : 4~6 balls with near-zero miss covering 360° (score saturates)
enum BucketId : int {
    BUCKET_NO_DYNAMIC  = 1,
    BUCKET_INTERCEPT   = 2,
    BUCKET_NEAR_MISS   = 3,
    BUCKET_NO_THREAT   = 4,
    BUCKET_ALL_BLOCKED = 5,
};

struct BucketDispatchOptions
{
    bool enabled{false};
    float w_no_dynamic{0.0f};
    float w_intercept{0.0f};
    float w_near_miss{0.0f};
    float w_no_threat{0.0f};
    float w_all_blocked{0.0f};
    // cumulative cdf, set during normalize (last bucket = remainder)
    float cdf_no_dynamic{0.0f};
    float cdf_intercept{0.0f};
    float cdf_near_miss{0.0f};
    float cdf_no_threat{0.0f};
};

// CPA inverse solve + drone intent sampling parameters (config block swarm_dataset.cpa).
// The first 12 values must match OmniRisk/config/traj_opt.yaml exactly (replicates _get_random_state/_get_random_goal).
struct CpaOptions
{
    // intent velocity / goal sampling
    float vel_max{6.0f};
    float vx_mean_unit{0.4f};
    float vx_std_unit{2.0f};
    float vy_mean_unit{0.0f};
    float vy_std_unit{1.2f};
    float vz_mean_unit{0.0f};
    float vz_std_unit{0.3f};
    float vel_norm_cap_factor{1.2f};
    float heading_vel_std_deg{25.0f};
    float goal_length{10.0f};
    float goal_pitch_std{10.0f};
    float goal_yaw_std{45.0f};
    // CPA geometry
    float cpa_time_min{0.3f};
    float cpa_time_max{1.667f};
    float miss_intercept_min{0.0f};
    float miss_intercept_max{0.6f};
    float miss_near_miss_min{0.6f};
    float miss_near_miss_max{1.2f};
    float miss_no_threat_min{3.0f};
    float miss_no_threat_max{8.0f};
    float miss_all_blocked_min{0.0f};
    float miss_all_blocked_max{0.15f};
    float sphere_dir_angle_min_deg{0.0f};
    float sphere_dir_angle_max_deg{180.0f};
    float speed_min{0.2f};
    float speed_max{4.0f};
    float rear_angle_threshold_deg{150.0f};
    float rear_speed_max{7.0f};
    int   max_resample{50};
    float max_spawn_distance{3.0f};   // hard cap on horizontal distance between cylinder spawn at t=0 and the body (resample if exceeded)
    float distractor_min_miss{2.0f};
    int   all_blocked_count_min{4};
    int   all_blocked_count_max{6};
    // threat = trunk-height vertical cylinder: radius range + vertical band (fixed, below tree crowns)
    float cyl_radius_min{0.3f};
    float cyl_radius_max{0.6f};
    float cyl_z_min{0.3f};
    float cyl_z_max{2.5f};
};

struct SwarmDatasetOptions
{
    bool enabled{false};
    int history_frames{1};
    float frame_dt{0.1f};
    int sphere_count_min{1};
    int sphere_count_max{3};
    float sphere_radius{0.25f};
    float dynamic_speed_min{0.2f};
    float dynamic_speed_max{3.0f};
    float sample_step{0.05f};
    float center_margin_xy{4.0f};
    float center_z_min{1.0f};
    float center_z_max{4.0f};
    float visible_depth_min{0.5f};
    float visible_depth_max{16.0f};
    float depth_margin{0.8f};
    int image_border{8};
    float camera_clearance{0.5f};
    float static_clearance{0.5f};
    float min_center_distance{0.6f};
    float drone_clearance_margin{0.1f};
    int max_attempts_per_obstacle{80};
    // CPA inverse placement: bucket-aware sample generation
    BucketDispatchOptions bucket;
    CpaOptions            cpa;
};

struct CylinderDatasetOptions
{
    bool enabled{false};
    int cylinder_count_min{1};
    int cylinder_count_max{2};
    float cylinder_radius_min{0.25f};
    float cylinder_radius_max{0.6f};
    float cylinder_height_min{1.0f};
    float cylinder_height_max{3.0f};
    float cylinder_speed_min{0.1f};
    float cylinder_speed_max{1.5f};
    float cylinder_static_clearance{0.4f};
    float cylinder_drone_clearance_margin{0.2f};
    float min_cylinder_distance{0.6f};
    float validate_future_seconds{0.5f};
    float trajectory_check_dt{0.05f};
};

// ─── Config loaders ──────────────────────────────────────────────────────────

SwarmDatasetOptions loadSwarmDatasetOptions(const YAML::Node &config)
{
    SwarmDatasetOptions o;
    if (!config["swarm_dataset"]) return o;
    const auto &n = config["swarm_dataset"];
    o.enabled              = n["enabled"]               ? n["enabled"].as<bool>()               : o.enabled;
    o.history_frames       = n["history_frames"]        ? n["history_frames"].as<int>()          : o.history_frames;
    o.frame_dt             = n["frame_dt"]              ? n["frame_dt"].as<float>()              : o.frame_dt;
    o.sphere_count_min     = n["sphere_count_min"]      ? n["sphere_count_min"].as<int>()        : o.sphere_count_min;
    o.sphere_count_max     = n["sphere_count_max"]      ? n["sphere_count_max"].as<int>()        : o.sphere_count_max;
    o.sphere_radius        = n["sphere_radius"]         ? n["sphere_radius"].as<float>()         : o.sphere_radius;
    o.dynamic_speed_min    = n["dynamic_speed_min"]     ? n["dynamic_speed_min"].as<float>()     : o.dynamic_speed_min;
    o.dynamic_speed_max    = n["dynamic_speed_max"]     ? n["dynamic_speed_max"].as<float>()     : o.dynamic_speed_max;
    o.sample_step          = n["sample_step"]           ? n["sample_step"].as<float>()           : o.sample_step;
    o.center_margin_xy     = n["center_margin_xy"]      ? n["center_margin_xy"].as<float>()      : o.center_margin_xy;
    if (n["center_z_range"]) { o.center_z_min = n["center_z_range"][0].as<float>(); o.center_z_max = n["center_z_range"][1].as<float>(); }
    o.visible_depth_min    = n["visible_depth_min"]     ? n["visible_depth_min"].as<float>()     : o.visible_depth_min;
    o.visible_depth_max    = n["visible_depth_max"]     ? n["visible_depth_max"].as<float>()     : o.visible_depth_max;
    o.depth_margin         = n["depth_margin"]          ? n["depth_margin"].as<float>()          : o.depth_margin;
    o.image_border         = n["image_border"]          ? n["image_border"].as<int>()            : o.image_border;
    o.camera_clearance     = n["camera_clearance"]      ? n["camera_clearance"].as<float>()      : o.camera_clearance;
    o.static_clearance     = n["static_clearance"]      ? n["static_clearance"].as<float>()      : o.static_clearance;
    o.min_center_distance  = n["min_center_distance"]   ? n["min_center_distance"].as<float>()   : o.min_center_distance;
    o.drone_clearance_margin = n["drone_clearance_margin"] ? n["drone_clearance_margin"].as<float>() : o.drone_clearance_margin;
    o.max_attempts_per_obstacle = n["max_attempts_per_obstacle"] ? n["max_attempts_per_obstacle"].as<int>() : o.max_attempts_per_obstacle;
    o.sphere_count_max  = std::max(o.sphere_count_min, o.sphere_count_max);
    o.dynamic_speed_max = std::max(o.dynamic_speed_min, o.dynamic_speed_max);

    // CPA inverse placement: bucket_dispatch ratios + cpa parameter block
    if (n["bucket_dispatch"])
    {
        const auto &b = n["bucket_dispatch"];
        o.bucket.enabled = b["enabled"] ? b["enabled"].as<bool>() : o.bucket.enabled;
        if (b["ratios"])
        {
            const auto &r = b["ratios"];
            o.bucket.w_no_dynamic  = r["no_dynamic"]  ? r["no_dynamic"].as<float>()  : 0.0f;
            o.bucket.w_intercept   = r["intercept"]   ? r["intercept"].as<float>()   : 0.0f;
            o.bucket.w_near_miss   = r["near_miss"]   ? r["near_miss"].as<float>()   : 0.0f;
            o.bucket.w_no_threat   = r["no_threat"]   ? r["no_threat"].as<float>()   : 0.0f;
            o.bucket.w_all_blocked = r["all_blocked"] ? r["all_blocked"].as<float>() : 0.0f;
        }
        if (o.bucket.enabled)
        {
            const float sum = o.bucket.w_no_dynamic + o.bucket.w_intercept +
                              o.bucket.w_near_miss + o.bucket.w_no_threat +
                              o.bucket.w_all_blocked;
            if (std::abs(sum - 1.0f) > 1e-3f)
                std::cerr << "[bucket_dispatch] ratios sum = " << sum
                          << " (expected 1.0); renormalizing.\n";
            if (sum <= 0.0f)
            {
                std::cerr << "[bucket_dispatch] all ratios are zero; disabling.\n";
                o.bucket.enabled = false;
            }
            else
            {
                o.bucket.w_no_dynamic  /= sum;
                o.bucket.w_intercept   /= sum;
                o.bucket.w_near_miss   /= sum;
                o.bucket.w_no_threat   /= sum;
                o.bucket.w_all_blocked /= sum;
                o.bucket.cdf_no_dynamic = o.bucket.w_no_dynamic;
                o.bucket.cdf_intercept  = o.bucket.cdf_no_dynamic + o.bucket.w_intercept;
                o.bucket.cdf_near_miss  = o.bucket.cdf_intercept  + o.bucket.w_near_miss;
                o.bucket.cdf_no_threat  = o.bucket.cdf_near_miss  + o.bucket.w_no_threat;
            }
        }
    }
    if (n["cpa"])
    {
        const auto &c = n["cpa"];
        auto F = [&](const char *k, float d) { return c[k] ? c[k].as<float>() : d; };
        auto I = [&](const char *k, int   d) { return c[k] ? c[k].as<int>()   : d; };
        o.cpa.vel_max             = F("vel_max",             o.cpa.vel_max);
        o.cpa.vx_mean_unit        = F("vx_mean_unit",        o.cpa.vx_mean_unit);
        o.cpa.vx_std_unit         = F("vx_std_unit",         o.cpa.vx_std_unit);
        o.cpa.vy_mean_unit        = F("vy_mean_unit",        o.cpa.vy_mean_unit);
        o.cpa.vy_std_unit         = F("vy_std_unit",         o.cpa.vy_std_unit);
        o.cpa.vz_mean_unit        = F("vz_mean_unit",        o.cpa.vz_mean_unit);
        o.cpa.vz_std_unit         = F("vz_std_unit",         o.cpa.vz_std_unit);
        o.cpa.vel_norm_cap_factor = F("vel_norm_cap_factor", o.cpa.vel_norm_cap_factor);
        o.cpa.heading_vel_std_deg = F("heading_vel_std_deg", o.cpa.heading_vel_std_deg);
        o.cpa.goal_length         = F("goal_length",         o.cpa.goal_length);
        o.cpa.goal_pitch_std      = F("goal_pitch_std",      o.cpa.goal_pitch_std);
        o.cpa.goal_yaw_std        = F("goal_yaw_std",        o.cpa.goal_yaw_std);
        o.cpa.cpa_time_min        = F("cpa_time_min",        o.cpa.cpa_time_min);
        o.cpa.cpa_time_max        = F("cpa_time_max",        o.cpa.cpa_time_max);
        o.cpa.miss_intercept_min  = F("miss_intercept_min",  o.cpa.miss_intercept_min);
        o.cpa.miss_intercept_max  = F("miss_intercept_max",  o.cpa.miss_intercept_max);
        o.cpa.miss_near_miss_min  = F("miss_near_miss_min",  o.cpa.miss_near_miss_min);
        o.cpa.miss_near_miss_max  = F("miss_near_miss_max",  o.cpa.miss_near_miss_max);
        o.cpa.miss_no_threat_min  = F("miss_no_threat_min",  o.cpa.miss_no_threat_min);
        o.cpa.miss_no_threat_max  = F("miss_no_threat_max",  o.cpa.miss_no_threat_max);
        o.cpa.miss_all_blocked_min = F("miss_all_blocked_min", o.cpa.miss_all_blocked_min);
        o.cpa.miss_all_blocked_max = F("miss_all_blocked_max", o.cpa.miss_all_blocked_max);
        o.cpa.sphere_dir_angle_min_deg = F("sphere_dir_angle_min_deg", o.cpa.sphere_dir_angle_min_deg);
        o.cpa.sphere_dir_angle_max_deg = F("sphere_dir_angle_max_deg", o.cpa.sphere_dir_angle_max_deg);
        o.cpa.speed_min           = F("speed_min",           o.cpa.speed_min);
        o.cpa.speed_max           = F("speed_max",           o.cpa.speed_max);
        o.cpa.rear_angle_threshold_deg = F("rear_angle_threshold_deg", o.cpa.rear_angle_threshold_deg);
        o.cpa.rear_speed_max      = F("rear_speed_max",      o.cpa.rear_speed_max);
        o.cpa.max_resample        = I("max_resample",        o.cpa.max_resample);
        o.cpa.max_spawn_distance  = F("max_spawn_distance",  o.cpa.max_spawn_distance);
        o.cpa.distractor_min_miss = F("distractor_min_miss", o.cpa.distractor_min_miss);
        o.cpa.all_blocked_count_min = I("all_blocked_count_min", o.cpa.all_blocked_count_min);
        o.cpa.all_blocked_count_max = I("all_blocked_count_max", o.cpa.all_blocked_count_max);
        o.cpa.all_blocked_count_max = std::max(o.cpa.all_blocked_count_min, o.cpa.all_blocked_count_max);
        o.cpa.cyl_radius_min      = F("cyl_radius_min",      o.cpa.cyl_radius_min);
        o.cpa.cyl_radius_max      = F("cyl_radius_max",      o.cpa.cyl_radius_max);
        o.cpa.cyl_z_min           = F("cyl_z_min",           o.cpa.cyl_z_min);
        o.cpa.cyl_z_max           = F("cyl_z_max",           o.cpa.cyl_z_max);
    }
    return o;
}

// CPA: pick a bucket id from per-frame categorical draw. Returns 0 if disabled.
int sampleBucketId(const BucketDispatchOptions &b, std::mt19937 &rng)
{
    if (!b.enabled) return 0;
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    const float r = u01(rng);
    if (r < b.cdf_no_dynamic) return BUCKET_NO_DYNAMIC;
    if (r < b.cdf_intercept)  return BUCKET_INTERCEPT;
    if (r < b.cdf_near_miss)  return BUCKET_NEAR_MISS;
    if (r < b.cdf_no_threat)  return BUCKET_NO_THREAT;
    return BUCKET_ALL_BLOCKED;
}

CylinderDatasetOptions loadCylinderDatasetOptions(const YAML::Node &config)
{
    CylinderDatasetOptions o;
    if (!config["cylinder_dataset"]) return o;
    const auto &n = config["cylinder_dataset"];
    o.enabled                    = n["enabled"]                    ? n["enabled"].as<bool>()               : o.enabled;
    o.cylinder_count_min         = n["cylinder_count_min"]         ? n["cylinder_count_min"].as<int>()     : o.cylinder_count_min;
    o.cylinder_count_max         = n["cylinder_count_max"]         ? n["cylinder_count_max"].as<int>()     : o.cylinder_count_max;
    o.cylinder_radius_min        = n["cylinder_radius_min"]        ? n["cylinder_radius_min"].as<float>()  : o.cylinder_radius_min;
    o.cylinder_radius_max        = n["cylinder_radius_max"]        ? n["cylinder_radius_max"].as<float>()  : o.cylinder_radius_max;
    o.cylinder_height_min        = n["cylinder_height_min"]        ? n["cylinder_height_min"].as<float>()  : o.cylinder_height_min;
    o.cylinder_height_max        = n["cylinder_height_max"]        ? n["cylinder_height_max"].as<float>()  : o.cylinder_height_max;
    o.cylinder_speed_min         = n["cylinder_speed_min"]         ? n["cylinder_speed_min"].as<float>()   : o.cylinder_speed_min;
    o.cylinder_speed_max         = n["cylinder_speed_max"]         ? n["cylinder_speed_max"].as<float>()   : o.cylinder_speed_max;
    o.cylinder_static_clearance  = n["cylinder_static_clearance"]  ? n["cylinder_static_clearance"].as<float>()  : o.cylinder_static_clearance;
    o.cylinder_drone_clearance_margin = n["cylinder_drone_clearance_margin"] ? n["cylinder_drone_clearance_margin"].as<float>() : o.cylinder_drone_clearance_margin;
    o.min_cylinder_distance      = n["min_cylinder_distance"]      ? n["min_cylinder_distance"].as<float>() : o.min_cylinder_distance;
    o.validate_future_seconds    = n["validate_future_seconds"]    ? n["validate_future_seconds"].as<float>() : o.validate_future_seconds;
    o.trajectory_check_dt       = n["trajectory_check_dt"]        ? n["trajectory_check_dt"].as<float>()  : o.trajectory_check_dt;
    o.cylinder_count_max  = std::max(o.cylinder_count_min,  o.cylinder_count_max);
    o.cylinder_radius_max = std::max(o.cylinder_radius_min, o.cylinder_radius_max);
    o.cylinder_height_max = std::max(o.cylinder_height_min, o.cylinder_height_max);
    o.cylinder_speed_max  = std::max(o.cylinder_speed_min,  o.cylinder_speed_max);
    return o;
}

// ─── Helper geometry functions ────────────────────────────────────────────────

Eigen::Vector3f pixelDepthToCameraPoint(int u, int v, float depth, const CameraParams &camera)
{
    const float y = -(static_cast<float>(u) - camera.cx) / camera.fx * depth;
    const float z = -(static_cast<float>(v) - camera.cy) / camera.fy * depth;
    return Eigen::Vector3f(depth, y, z);
}

Eigen::Vector3f cameraPointToWorld(const cudaMat::SE3<float> &T_wc, const Eigen::Vector3f &point_c)
{
    const float3 point_w = T_wc * make_float3(point_c.x(), point_c.y(), point_c.z());
    return Eigen::Vector3f(point_w.x, point_w.y, point_w.z);
}

Eigen::Vector3f getSphereCenterAtOffset(const SphereObstacleSpec &spec, float t)
{
    return spec.center + t * spec.velocity;
}

Eigen::Vector3f getCylinderCenterAtOffset(const CylinderObstacleSpec &spec, float t)
{
    return spec.center + t * spec.velocity;
}

std::vector<SphereObstacle> buildDynamicSphereFrame(const std::vector<SphereObstacleSpec> &specs, float t)
{
    std::vector<SphereObstacle> out;
    out.reserve(specs.size());
    for (const auto &s : specs)
    {
        const Eigen::Vector3f c = getSphereCenterAtOffset(s, t);
        SphereObstacle o;
        o.center = Vector3f(c.x(), c.y(), c.z());
        o.radius = s.radius;
        o.obj_id = s.obj_id;
        out.push_back(o);
    }
    return out;
}

std::vector<CylinderObstacle> buildDynamicCylinderFrame(const std::vector<CylinderObstacleSpec> &specs, float t)
{
    std::vector<CylinderObstacle> out;
    out.reserve(specs.size());
    for (const auto &s : specs)
    {
        const Eigen::Vector3f c = getCylinderCenterAtOffset(s, t);
        CylinderObstacle o;
        o.center = Vector3f(c.x(), c.y(), c.z());
        o.radius = s.radius;
        o.height = s.height;
        o.obj_id = s.obj_id;
        out.push_back(o);
    }
    return out;
}

// ─── CPA cylinder placement ──────────────────────────────────────────────────

// Number of pixels where a candidate cylinder is not occluded in the static panorama: project the near cylinder surface onto the range image grid (same
// az/elev mapping as the render kernel; render frame is world-axis aligned with camera pitch=0 → world direction = dir_b), and per pixel
// compare cylinder surface distance vs static distance; if the cylinder is closer it is visible. Used to reject directions blocked by trees/out of view.
int cylinderVisiblePixels(const CylinderObstacleSpec &spec,
                          const Eigen::Vector3f &O,
                          const cv::Mat &static_range,
                          const raycast::LidarParams &lidar)
{
    const int   H = lidar.vertical_lines;
    const int   W = lidar.horizontal_num;
    const float deg2rad = static_cast<float>(M_PI) / 180.0f;
    const float e_max  = lidar.vertical_angle_end   * deg2rad;
    const float e_min  = lidar.vertical_angle_start * deg2rad;
    const float two_pi = 2.0f * static_cast<float>(M_PI);
    const float eps    = 0.05f;

    const Eigen::Vector2f C = spec.center.head<2>();
    const Eigen::Vector2f f = O.head<2>() - C;
    const float dist_c = f.norm();
    if (dist_c <= spec.radius) return H * W;            // body inside the cylinder (already caught by the clearance check, fallback)
    const float z_bottom = spec.center.z() - 0.5f * spec.height;
    const float z_top    = spec.center.z() + 0.5f * spec.height;

    const float center_az = std::atan2(C.y() - O.y(), C.x() - O.x());
    const float half_az   = std::asin(std::min(1.0f, spec.radius / dist_c));
    const int   col_c     = (static_cast<int>(std::lround((center_az + static_cast<float>(M_PI)) / two_pi * W)) % W + W) % W;
    const int   hw        = static_cast<int>(std::ceil(half_az / (two_pi / W))) + 1;

    int count = 0;
    for (int dc = -hw; dc <= hw; ++dc)
    {
        const int col = ((col_c + dc) % W + W) % W;
        const float az = (static_cast<float>(col) / W) * two_pi - static_cast<float>(M_PI);
        const Eigen::Vector2f dir(std::cos(az), std::sin(az));
        // ray-circle: |O_xy + t·dir - C| = r, take the near hit t>0 = horizontal hit distance
        const float b    = 2.0f * f.dot(dir);
        const float cc   = f.dot(f) - spec.radius * spec.radius;
        const float disc = b * b - 4.0f * cc;
        if (disc < 0.0f) continue;
        const float dxy = (-b - std::sqrt(disc)) * 0.5f;
        if (dxy <= 0.0f) continue;                      // cylinder behind
        float e_lo = std::atan2(z_bottom - O.z(), dxy);
        float e_hi = std::atan2(z_top    - O.z(), dxy);
        e_lo = std::max(e_lo, e_min);
        e_hi = std::min(e_hi, e_max);
        if (e_hi <= e_lo) continue;                     // band outside vertical FOV
        int r_hi = static_cast<int>(std::floor((e_max - e_hi) / (e_max - e_min) * (H - 1)));  // high elev → small row
        int r_lo = static_cast<int>(std::ceil ((e_max - e_lo) / (e_max - e_min) * (H - 1)));
        r_hi = std::max(0, r_hi);
        r_lo = std::min(H - 1, r_lo);
        for (int row = r_hi; row <= r_lo; ++row)
        {
            const float elev = e_max - (static_cast<float>(row) / (H - 1)) * (e_max - e_min);
            const float ce   = std::cos(elev);
            if (ce < 1e-4f) continue;
            const float rng = dxy / ce;                 // euclidean distance to the cylinder surface
            if (rng >= lidar.max_lidar_dist) continue;
            if (rng < static_range.at<float>(row, col) - eps) ++count;
        }
    }
    return count;
}

// Validity of a vertical cylinder over the history window: bbox inside the world, horizontal camera clearance (only within the band), static clearance at 3 z samples,
// and finally the number of unoccluded pixels in the static panorama ≥ threshold (invisible directions make the placement loop resample).
bool validateMovingCpaCylinder(
    const CylinderObstacleSpec &spec,
    const SwarmDatasetOptions &options,
    const Eigen::Vector3f &camera_pos,
    pcl::KdTreeFLANN<pcl::PointXYZ> &static_kdtree,
    const Eigen::Vector3f &world_min,
    const Eigen::Vector3f &world_max,
    const cv::Mat &static_range,
    const raycast::LidarParams &lidar)
{
    const float required_drone_clearance  = spec.radius + options.camera_clearance + options.drone_clearance_margin;
    const float required_static_clearance = spec.radius + options.static_clearance;
    const int frame_count = std::max(1, options.history_frames);

    for (int fi = 0; fi < frame_count; ++fi)
    {
        const float t = (static_cast<float>(fi - (frame_count - 1))) * options.frame_dt;
        const Eigen::Vector3f center = getCylinderCenterAtOffset(spec, t);
        const float z_bottom = center.z() - 0.5f * spec.height;
        const float z_top    = center.z() + 0.5f * spec.height;

        if (center.x() < world_min.x() + spec.radius || center.x() > world_max.x() - spec.radius ||
            center.y() < world_min.y() + spec.radius || center.y() > world_max.y() - spec.radius ||
            z_bottom < world_min.z() || z_top > world_max.z())
            return false;

        if ((center.head<2>() - camera_pos.head<2>()).norm() < required_drone_clearance &&
            camera_pos.z() > z_bottom && camera_pos.z() < z_top)
            return false;

        // Static clearance: the cylinder only occupies [z_bottom,z_top]; only static points within that height range and horizontally close
        // count as collisions → ground (z≈0, below the band) and high tree crowns are ignored, so a pole can stand on the ground without false rejection.
        {
            const float half_h   = 0.5f * spec.height;
            const float search_r = std::sqrt(required_static_clearance * required_static_clearance + half_h * half_h);
            pcl::PointXYZ sp(center.x(), center.y(), center.z());
            std::vector<int> idx; std::vector<float> dist2;
            static_kdtree.radiusSearch(sp, search_r, idx, dist2);
            for (int id : idx)
            {
                const auto &pt = static_kdtree.getInputCloud()->points[id];
                if (pt.z <= z_bottom || pt.z >= z_top) continue;          // outside the band (including ground) does not count
                const float dxy = std::hypot(pt.x - center.x(), pt.y - center.y());
                if (dxy < required_static_clearance) return false;
            }
        }
    }
    // Visibility: unoccluded pixels in the static panorama ≥ threshold, otherwise resample direction (an invisible threat is noisy supervision)
    constexpr int kVisibleMinPixels = 10;
    if (cylinderVisiblePixels(spec, camera_pos, static_range, lidar) < kVisibleMinPixels)
        return false;
    return true;
}

// ─── CPA inverse cylinder placement: drone intent sampling + closed-form solve ───────────────────

// Per-frame drone intent: world-frame velocity + goal offset vector (replicates the training-side _get_random_state /
// _get_random_goal). heading_az is the intent azimuth (abstract, does not change the render pose).
struct DroneIntent
{
    float heading_az{0.0f};
    Eigen::Vector3f vel{Eigen::Vector3f::Zero()};
    Eigen::Vector3f goal{Eigen::Vector3f::Zero()};
};

DroneIntent sampleDroneIntent(const CpaOptions &c, std::mt19937 &rng)
{
    std::normal_distribution<float>       gauss(0.0f, 1.0f);
    std::uniform_real_distribution<float> u_pi(-static_cast<float>(M_PI), static_cast<float>(M_PI));
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    const float vmax = c.vel_max;
    const float cap  = c.vel_norm_cap_factor * vmax;
    const float vx_lognorm_mean = std::log(1.0f - c.vx_mean_unit);   // = log(0.6)
    const float vx_lognorm_sig  = std::log(c.vx_std_unit);           // = log(2.0)

    // 1) intent azimuth (uniform, abstract)
    DroneIntent out;
    out.heading_az = u_pi(rng);

    // 2) velocity: right-skewed vx (lognormal) + Gaussian vy/vz, rotated around heading with jitter
    Eigen::Vector3f vel;
    while (true)
    {
        vel.x() = vmax * (c.vx_mean_unit + c.vx_std_unit * gauss(rng));  // placeholder, overwritten immediately
        vel.y() = vmax * (c.vy_mean_unit + c.vy_std_unit * gauss(rng));
        vel.z() = vmax * (c.vz_mean_unit + c.vz_std_unit * gauss(rng));
        float rvx = -1.0f;
        while (rvx < 0.0f)
        {
            const float ln = std::exp(vx_lognorm_mean + vx_lognorm_sig * gauss(rng));
            rvx = -(vmax * ln) + cap;
        }
        vel.x() = rvx;
        if (vel.norm() < cap) break;
    }
    const float a  = out.heading_az + gauss(rng) * (c.heading_vel_std_deg * static_cast<float>(M_PI) / 180.0f);
    const float ca = std::cos(a), sa = std::sin(a);
    out.vel.x() = ca * vel.x() - sa * vel.y();
    out.vel.y() = sa * vel.x() + ca * vel.y();
    out.vel.z() = vel.z();

    // 3) goal offset vector (world frame, around heading)
    std::normal_distribution<float> gp_d(0.0f, c.goal_pitch_std);
    std::normal_distribution<float> gy_d(0.0f, c.goal_yaw_std);
    const float gp = gp_d(rng) * static_cast<float>(M_PI) / 180.0f;
    const float gy = out.heading_az + gy_d(rng) * static_cast<float>(M_PI) / 180.0f;
    Eigen::Vector3f gdir(std::cos(gy) * std::cos(gp), std::sin(gy) * std::cos(gp), std::sin(gp));
    if (u01(rng) < 0.1f) gdir *= u01(rng) * 10.0f;   // 10% near goals
    out.goal = c.goal_length * gdir;
    return out;
}

// Closed-form CPA solve (horizontal plane): given the drone line (d0,u), closest-approach time t*, horizontal miss d_miss,
// cylinder horizontal speed, approach angle theta (relative to -û_xy), side sign side (±1) and miss sign ms (±1),
// solve the cylinder start (xy) and horizontal velocity. The vertical band is fixed at [z_center±h/2].
// By construction the horizontal t* and d_miss are the exact CPA values (miss vector ⊥ horizontal relative velocity).
CylinderObstacleSpec buildCpaCylinder(const Eigen::Vector3f &d0, const Eigen::Vector3f &u,
                                      float t_star, float d_miss, float speed,
                                      float theta, float side, float ms,
                                      float z_center, float height)
{
    const Eigen::Vector2f u_xy(u.x(), u.y());
    const Eigen::Vector2f uhat = (u_xy.norm() > 1e-6f) ? u_xy.normalized() : Eigen::Vector2f(1, 0);
    const Eigen::Vector2f a = -uhat;                       // facing the drone velocity = horizontal approach axis
    const Eigen::Vector2f a_perp(-a.y(), a.x());
    const Eigen::Vector2f what = (std::cos(theta) * a + std::sin(theta) * side * a_perp).normalized();
    const Eigen::Vector2f w = speed * what;
    const Eigen::Vector2f q = w - u_xy;
    const Eigen::Vector2f qhat = (q.norm() > 1e-6f) ? q.normalized() : a;
    const Eigen::Vector2f qperp(-qhat.y(), qhat.x());
    const Eigen::Vector2f m = d_miss * ms * qperp;         // ⊥ q (horizontal)
    const Eigen::Vector2f c_xy = d0.head<2>() + u_xy * t_star + m - w * t_star;
    CylinderObstacleSpec spec;
    spec.center   = Eigen::Vector3f(c_xy.x(), c_xy.y(), z_center);
    spec.velocity = Eigen::Vector3f(w.x(), w.y(), 0.0f);
    spec.height   = height;
    return spec;
}

// Given approach angle theta, lower bound for a closed-form speed: for theta>90° (from behind) the cylinder must outrun the drone along its heading,
// w·û = -speed·cosθ ≥ |u_xy| → speed ≥ |u_xy| / (-cosθ).
float cpaSpeedLowerBound(float theta, float u_norm, float speed_floor)
{
    const float c = std::cos(theta);
    if (c < -1e-3f) return std::max(speed_floor, u_norm / (-c));
    return speed_floor;
}

// Solve one main threat cylinder: sample t*/d_miss/theta/speed from the bucket miss range, solve horizontally + validate,
// resample up to max_resample. Returns true and fills out on success.
bool solveCpaMainCylinder(
    const CpaOptions &c, float miss_min, float miss_max,
    const SwarmDatasetOptions &options,
    const Eigen::Vector3f &d0, const Eigen::Vector3f &u,
    pcl::KdTreeFLANN<pcl::PointXYZ> &static_kdtree,
    const Eigen::Vector3f &world_min, const Eigen::Vector3f &world_max,
    const cv::Mat &static_range, const raycast::LidarParams &lidar,
    std::mt19937 &rng, CylinderObstacleSpec &out)
{
    const float deg2rad  = static_cast<float>(M_PI) / 180.0f;
    const float ang_lo   = c.sphere_dir_angle_min_deg * deg2rad;
    const float ang_hi   = c.sphere_dir_angle_max_deg * deg2rad;
    const float rear_thr = c.rear_angle_threshold_deg * deg2rad;
    const float u_xy_norm = u.head<2>().norm();
    const float margin   = 0.3f;
    const float height   = c.cyl_z_max - c.cyl_z_min;
    // band follows drone height: z_center = body z (clamped into world z bounds to avoid ground false rejection),
    // the cylinder covers the body vertically → always within LiDAR FOV and a real 3D threat (a fixed low band drops out of view for high-flying frames).
    const float half_h   = 0.5f * height;
    const float z_center = std::max(world_min.z() + half_h, std::min(d0.z(), world_max.z() - half_h));
    std::uniform_real_distribution<float> u_tstar(c.cpa_time_min, c.cpa_time_max);
    std::uniform_real_distribution<float> u_miss(miss_min, miss_max);
    std::uniform_real_distribution<float> u_theta(ang_lo, ang_hi);
    std::uniform_real_distribution<float> u_radius(c.cyl_radius_min, c.cyl_radius_max);
    std::uniform_real_distribution<float> u_sign(0.0f, 1.0f);

    for (int attempt = 0; attempt < c.max_resample; ++attempt)
    {
        const float theta = u_theta(rng);
        const float lo    = cpaSpeedLowerBound(theta, u_xy_norm, c.speed_min) + margin;
        const float cap   = (theta > rear_thr) ? c.rear_speed_max : c.speed_max;
        if (lo > cap) continue;                            // no solution for this approach direction (chasing cylinder cannot catch up)
        std::uniform_real_distribution<float> u_speed(lo, cap);
        const float side = (u_sign(rng) < 0.5f) ? -1.0f : 1.0f;
        const float ms   = (u_sign(rng) < 0.5f) ? -1.0f : 1.0f;
        CylinderObstacleSpec spec = buildCpaCylinder(
            d0, u, u_tstar(rng), u_miss(rng), u_speed(rng), theta, side, ms, z_center, height);
        spec.radius = u_radius(rng);
        // hard spawn distance cap: horizontal distance from cylinder center at t=0 to body ≤ max_spawn_distance, so every dynamic cylinder appears close
        if ((spec.center.head<2>() - d0.head<2>()).norm() > c.max_spawn_distance) continue;
        if (validateMovingCpaCylinder(spec, options, d0, static_kdtree, world_min, world_max, static_range, lidar))
        {
            out = spec;
            return true;
        }
    }
    return false;
}

// CPA inverse solve (by spawn azimuth): given drone (d0,u), horizontal azimuth phi of the cylinder relative to the body at t=0, spawn distance r_spawn,
// closest-approach time t*, horizontal miss d_miss (sign ms) and vertical band. Relative velocity points along -phi at the body → exact CPA,
// CPA time = t*, miss = d_miss. Taking phi as input lets all_blocked spread threats over all azimuths.
CylinderObstacleSpec buildCpaCylinderFromBearing(
    const Eigen::Vector3f &d0, const Eigen::Vector3f &u,
    float phi, float r_spawn, float t_star, float d_miss, float ms,
    float z_center, float height)
{
    const Eigen::Vector2f ehat(std::cos(phi), std::sin(phi));
    const Eigen::Vector2f eperp(-ehat.y(), ehat.x());
    const Eigen::Vector2f u_xy(u.x(), u.y());
    const float rho = r_spawn / t_star;                // |relative velocity| = closing speed
    const Eigen::Vector2f w = u_xy - rho * ehat;       // absolute horizontal cylinder velocity (relative velocity = -rho·ehat pointing at the body)
    const Eigen::Vector2f m = d_miss * ms * eperp;     // miss perpendicular to relative velocity
    const Eigen::Vector2f c_xy = d0.head<2>() + r_spawn * ehat + m;
    CylinderObstacleSpec spec;
    spec.center   = Eigen::Vector3f(c_xy.x(), c_xy.y(), z_center);
    spec.velocity = Eigen::Vector3f(w.x(), w.y(), 0.0f);
    spec.height   = height;
    return spec;
}

// Solve one collision cylinder inside the azimuth sector [phi_center ± phi_half]: controls spawn azimuth directly so all_blocked
// threats surround the drone instead of collapsing to one azimuth with independent theta sampling. The speed cap still depends on the cylinder's approach (head-on/from behind);
// azimuths that need excessive speed (e.g. straight behind) are skipped honestly (returns false) and the caller moves on.
bool solveCpaCylinderInSector(
    const CpaOptions &c, float miss_min, float miss_max,
    float phi_center, float phi_half,
    const SwarmDatasetOptions &options,
    const Eigen::Vector3f &d0, const Eigen::Vector3f &u,
    pcl::KdTreeFLANN<pcl::PointXYZ> &static_kdtree,
    const Eigen::Vector3f &world_min, const Eigen::Vector3f &world_max,
    const cv::Mat &static_range, const raycast::LidarParams &lidar,
    std::mt19937 &rng, CylinderObstacleSpec &out)
{
    const float rear_thr = c.rear_angle_threshold_deg * static_cast<float>(M_PI) / 180.0f;
    const Eigen::Vector2f u_xy = u.head<2>();
    const Eigen::Vector2f uhat = (u_xy.norm() > 1e-6f) ? u_xy.normalized() : Eigen::Vector2f(1, 0);
    const float height   = c.cyl_z_max - c.cyl_z_min;
    const float half_h   = 0.5f * height;
    const float z_center = std::max(world_min.z() + half_h, std::min(d0.z(), world_max.z() - half_h));
    const float r_min    = std::min(options.camera_clearance + c.cyl_radius_max + 0.2f, c.max_spawn_distance);
    std::uniform_real_distribution<float> u_phi(phi_center - phi_half, phi_center + phi_half);
    std::uniform_real_distribution<float> u_r(r_min, c.max_spawn_distance);
    std::uniform_real_distribution<float> u_tstar(c.cpa_time_min, c.cpa_time_max);
    std::uniform_real_distribution<float> u_miss(miss_min, miss_max);
    std::uniform_real_distribution<float> u_radius(c.cyl_radius_min, c.cyl_radius_max);
    std::uniform_real_distribution<float> u_sign(0.0f, 1.0f);

    for (int attempt = 0; attempt < c.max_resample; ++attempt)
    {
        const float phi   = u_phi(rng);
        const float ms    = (u_sign(rng) < 0.5f) ? -1.0f : 1.0f;
        CylinderObstacleSpec spec = buildCpaCylinderFromBearing(
            d0, u, phi, u_r(rng), u_tstar(rng), u_miss(rng), ms, z_center, height);
        const Eigen::Vector2f w = spec.velocity.head<2>();
        const float speed = w.norm();
        if (speed < c.speed_min) continue;                 // cylinder barely moves = degenerate static
        const float cos_theta = w.normalized().dot(-uhat); // angle of cylinder approach relative to facing the drone velocity
        const float theta = std::acos(std::max(-1.0f, std::min(1.0f, cos_theta)));
        const float cap   = (theta > rear_thr) ? c.rear_speed_max : c.speed_max;
        if (speed > cap) continue;                         // this azimuth needs excessive speed (common: straight behind)
        spec.radius = u_radius(rng);
        if (validateMovingCpaCylinder(spec, options, d0, static_kdtree, world_min, world_max, static_range, lidar))
        {
            out = spec;
            return true;
        }
    }
    return false;
}

// ─── Cylinder obstacle placement ─────────────────────────────────────────────

bool validateMovingCylinder(
    const CylinderObstacleSpec &spec,
    const CylinderDatasetOptions &options,
    const Eigen::Vector3f &camera_pos,
    pcl::KdTreeFLANN<pcl::PointXYZ> &static_kdtree,
    const Eigen::Vector3f &world_min,
    const Eigen::Vector3f &world_max)
{
    const float required_static_clearance = spec.radius + options.cylinder_static_clearance;
    const int steps = std::max(1, static_cast<int>(std::ceil(options.validate_future_seconds / options.trajectory_check_dt)));

    for (int step = 0; step <= steps; ++step)
    {
        const float t = std::min(options.validate_future_seconds, step * options.trajectory_check_dt);
        const Eigen::Vector3f center = getCylinderCenterAtOffset(spec, t);
        const float z_bottom = center.z() - 0.5f * spec.height;
        const float z_top    = center.z() + 0.5f * spec.height;

        if (center.x() < world_min.x() + spec.radius || center.x() > world_max.x() - spec.radius ||
            center.y() < world_min.y() + spec.radius || center.y() > world_max.y() - spec.radius ||
            z_bottom < world_min.z() || z_top > world_max.z())
            return false;

        if ((center - camera_pos).norm() < spec.radius + options.cylinder_drone_clearance_margin)
            return false;

        for (float z : {z_bottom, center.z(), z_top})
        {
            pcl::PointXYZ sp(center.x(), center.y(), z);
            std::vector<int> idx(1); std::vector<float> dist2(1);
            if (static_kdtree.nearestKSearch(sp, 1, idx, dist2) > 0 && std::sqrt(dist2[0]) < required_static_clearance)
                return false;
        }
    }
    return true;
}

std::vector<CylinderObstacleSpec> samplePerImageCylinders(
    const CylinderDatasetOptions &cyl_options,
    const SwarmDatasetOptions &swarm_options,
    const cv::Mat &static_depth_image,
    const CameraParams &camera,
    const cudaMat::SE3<float> &T_wc,
    const Eigen::Vector3f &camera_pos,
    pcl::KdTreeFLANN<pcl::PointXYZ> &static_kdtree,
    const Eigen::Vector3f &world_min,
    const Eigen::Vector3f &world_max,
    std::mt19937 &generator)
{
    std::vector<CylinderObstacleSpec> cylinders;
    if (!cyl_options.enabled || cyl_options.cylinder_count_max <= 0) return cylinders;

    const int x_low  = std::max(0, swarm_options.image_border);
    const int x_high = std::min(camera.image_width  - 1, camera.image_width  - 1 - swarm_options.image_border);
    const int y_low  = std::max(0, swarm_options.image_border);
    const int y_high = std::min(camera.image_height - 1, camera.image_height - 1 - swarm_options.image_border);
    if (x_low > x_high || y_low > y_high) return cylinders;

    std::vector<cv::Point> candidate_pixels;
    candidate_pixels.reserve((x_high - x_low + 1) * (y_high - y_low + 1));
    for (int v = y_low; v <= y_high; ++v)
        for (int u = x_low; u <= x_high; ++u)
        {
            const float d = static_depth_image.at<float>(v, u);
            if (d <= swarm_options.visible_depth_min + swarm_options.depth_margin) continue;
            if (d >= camera.max_depth_dist - swarm_options.depth_margin) continue;
            candidate_pixels.emplace_back(u, v);
        }
    if (candidate_pixels.empty()) return cylinders;

    std::uniform_int_distribution<int> count_dist(cyl_options.cylinder_count_min, cyl_options.cylinder_count_max);
    std::uniform_int_distribution<int> pixel_dist(0, candidate_pixels.size() - 1);
    std::uniform_real_distribution<float> radius_dist(cyl_options.cylinder_radius_min, cyl_options.cylinder_radius_max);
    std::uniform_real_distribution<float> height_dist(cyl_options.cylinder_height_min, cyl_options.cylinder_height_max);
    std::uniform_real_distribution<float> speed_dist(cyl_options.cylinder_speed_min, cyl_options.cylinder_speed_max);
    std::uniform_real_distribution<float> unit_dist(-1.0f, 1.0f);

    const int target_count = count_dist(generator);
    cylinders.reserve(target_count);

    for (int ti = 0; ti < target_count; ++ti)
    {
        bool placed = false;
        for (int attempt = 0; attempt < swarm_options.max_attempts_per_obstacle; ++attempt)
        {
            const cv::Point pixel = candidate_pixels[pixel_dist(generator)];
            const float static_depth = static_depth_image.at<float>(pixel.y, pixel.x);
            const float max_center_depth = std::min(swarm_options.visible_depth_max, static_depth - swarm_options.depth_margin);
            if (max_center_depth <= swarm_options.visible_depth_min) continue;

            std::uniform_real_distribution<float> depth_dist(swarm_options.visible_depth_min, max_center_depth);
            CylinderObstacleSpec spec;
            spec.radius = radius_dist(generator);
            spec.height = height_dist(generator);
            spec.center = cameraPointToWorld(T_wc, pixelDepthToCameraPoint(pixel.x, pixel.y, depth_dist(generator), camera));

            const float z_bottom = spec.center.z() - 0.5f * spec.height;
            const float z_top    = spec.center.z() + 0.5f * spec.height;
            if (spec.center.x() < world_min.x() + spec.radius || spec.center.x() > world_max.x() - spec.radius ||
                spec.center.y() < world_min.y() + spec.radius || spec.center.y() > world_max.y() - spec.radius ||
                z_bottom < world_min.z() || z_top > world_max.z())
                continue;

            Eigen::Vector3f vdir(unit_dist(generator), unit_dist(generator), 0.35f * unit_dist(generator));
            if (vdir.norm() < 1e-4f) continue;
            vdir.normalize();
            spec.velocity = vdir * speed_dist(generator);

            if (!validateMovingCylinder(spec, cyl_options, camera_pos, static_kdtree, world_min, world_max))
                continue;

            bool overlaps = false;
            for (const auto &existing : cylinders)
            {
                if ((existing.center.head<2>() - spec.center.head<2>()).norm() <
                    existing.radius + spec.radius + cyl_options.min_cylinder_distance)
                { overlaps = true; break; }
            }
            if (overlaps) continue;

            cylinders.push_back(spec);
            placed = true;
            break;
        }
        if (!placed)
            std::cerr << "[cylinder_dataset] failed to place cylinder " << ti
                      << " after " << swarm_options.max_attempts_per_obstacle << " attempts.\n";
    }
    return cylinders;
}

// ─── Tracker-noise sampling (v5 + v5.2 schema) ─────────────────────────────────
//
// Each per-object draw is independent — degradation scenarios are mutually exclusive (we pick
// the first one whose Bernoulli fires).  Returns the post-degradation
// (v_tracker, v_unc, a_tracker, a_unc, age) for one object.

struct ObjNoiseSample
{
    Eigen::Vector3f v_tracker;
    Eigen::Vector3f v_unc;
    Eigen::Vector3f a_tracker;
    Eigen::Vector3f a_unc;
    int age;
};

ObjNoiseSample sampleTrackerNoise(const TrackerNoiseConfig &cfg,
                                  const Eigen::Vector3f &v_sim,
                                  const Eigen::Vector3f &a_sim,
                                  std::mt19937 &rng)
{
    std::normal_distribution<float>       gauss(0.0f, 1.0f);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    ObjNoiseSample s;
    s.age = cfg.age_default;
    float sigma_v = cfg.sigma_v_default;
    float sigma_a = cfg.sigma_a_default;

    const float r = u01(rng);
    float c0 = cfg.new_track_ratio;
    float c1 = c0 + cfg.reappear_ratio;
    float c2 = c1 + cfg.accel_spike_ratio;
    float c3 = c2 + cfg.id_swap_ratio;

    bool zero_v_tracker = false;
    bool use_new_track_bias = false;

    if (r < c0)
    {
        s.age = 0;
        sigma_v = std::max(sigma_v, cfg.new_track_bias_std);
        sigma_a *= 2.0f;
        use_new_track_bias = true;
    }
    else if (r < c1)
    {
        sigma_v *= 3.0f;
        sigma_a *= 2.0f;
        s.age = std::max(1, cfg.age_default - 1);
    }
    else if (r < c2)
    {
        sigma_v *= 2.0f;
        sigma_a *= 3.0f;
    }
    else if (r < c3)
    {
        // ID swap: tracker reports a completely wrong velocity (effectively zero).
        zero_v_tracker = true;
    }

    // v_tracker
    if (zero_v_tracker)
        s.v_tracker = Eigen::Vector3f::Zero();
    else if (use_new_track_bias)
        s.v_tracker = Eigen::Vector3f(gauss(rng), gauss(rng), gauss(rng)) * cfg.new_track_bias_std;
    else
        s.v_tracker = v_sim + Eigen::Vector3f(gauss(rng), gauss(rng), gauss(rng)) * sigma_v;

    s.v_unc = Eigen::Vector3f::Constant(sigma_v);

    // a_tracker (v5.2): even when a_sim==0 we report a tracker estimate noised
    // around zero so downstream losses learn graceful fallback.
    s.a_tracker = a_sim + Eigen::Vector3f(gauss(rng), gauss(rng), gauss(rng)) * sigma_a;
    s.a_unc     = Eigen::Vector3f::Constant(sigma_a);

    return s;
}

TrackerNoiseConfig loadTrackerNoiseConfig(const YAML::Node &config)
{
    TrackerNoiseConfig c;
    if (!config["tracker_noise"]) return c;
    const auto &n = config["tracker_noise"];
    if (n["sigma_v"])             c.sigma_v_default      = n["sigma_v"].as<float>();
    if (n["sigma_a"])             c.sigma_a_default      = n["sigma_a"].as<float>();
    if (n["id_swap_prob"])        c.id_swap_prob_default = n["id_swap_prob"].as<float>();
    if (n["new_track_bias_std"])  c.new_track_bias_std   = n["new_track_bias_std"].as<float>();
    if (n["age_default"])         c.age_default          = n["age_default"].as<int>();
    if (n["new_track_ratio"])     c.new_track_ratio      = n["new_track_ratio"].as<float>();
    if (n["reappear_ratio"])      c.reappear_ratio       = n["reappear_ratio"].as<float>();
    if (n["accel_spike_ratio"])   c.accel_spike_ratio    = n["accel_spike_ratio"].as<float>();
    if (n["id_swap_ratio"])       c.id_swap_ratio        = n["id_swap_ratio"].as<float>();
    return c;
}

// ─── objects.csv writer (v5 unified schema) ─────────────────────────────────
//
// Per-row layout: SPHERE_MAX × slot_fields + CYL_MAX × slot_fields
// Each slot is 57 fields:
//   obj_id, type, cx, cy, cz, r, h,
//   vx_sim,vy_sim,vz_sim, ax_sim,ay_sim,az_sim,
//   vx_trk,vy_trk,vz_trk, ax_trk,ay_trk,az_trk,
//   vx_unc,vy_unc,vz_unc, ax_unc,ay_unc,az_unc,
//   age,
//   future_px_{1..T}, future_py_{1..T}, future_pz_{1..T},  (3*T = 15 with T=5)
//   bbox_cx,bbox_cy,bbox_cz, bbox_sx,bbox_sy,bbox_sz
//
// type=0 sphere (h=0), type=1 cylinder. Invalid slot: obj_id=0, all zeros.

static constexpr int OBJECTS_T_FUTURE     = 5;
static constexpr float OBJECTS_FUTURE_DT  = 0.1f;
// Per-slot field count breakdown (one obstacle = one slot):
//   2  obj_id, type
//   5  cx, cy, cz, r, h
//   6  vx_sim,vy_sim,vz_sim, ax_sim,ay_sim,az_sim
//   6  vx_trk,vy_trk,vz_trk, ax_trk,ay_trk,az_trk
//   6  vx_unc,vy_unc,vz_unc, ax_unc,ay_unc,az_unc
//   1  age
//   3*T_FUTURE  future centroid (x,y,z) per step
//   6  bbox_cx,cy,cz, bbox_sx,sy,sz
static constexpr int OBJECTS_SLOT_FIELDS  = 2 + 5 + 6 + 6 + 6 + 1 + 3 * OBJECTS_T_FUTURE + 6;
// = 47 with T_FUTURE = 5

void writeObjectsHeader(std::ofstream &file, int sphere_count_max, int cylinder_count_max)
{
    auto slot_header = [&file](const std::string &prefix)
    {
        file << prefix << "obj_id," << prefix << "type,"
             << prefix << "cx,"     << prefix << "cy,"  << prefix << "cz,"
             << prefix << "r,"      << prefix << "h,"
             << prefix << "vx_sim," << prefix << "vy_sim," << prefix << "vz_sim,"
             << prefix << "ax_sim," << prefix << "ay_sim," << prefix << "az_sim,"
             << prefix << "vx_trk," << prefix << "vy_trk," << prefix << "vz_trk,"
             << prefix << "ax_trk," << prefix << "ay_trk," << prefix << "az_trk,"
             << prefix << "vx_unc," << prefix << "vy_unc," << prefix << "vz_unc,"
             << prefix << "ax_unc," << prefix << "ay_unc," << prefix << "az_unc,"
             << prefix << "age";
        for (int k = 1; k <= OBJECTS_T_FUTURE; ++k)
            file << "," << prefix << "fpx_" << k << "," << prefix << "fpy_" << k
                 << "," << prefix << "fpz_" << k;
        file << "," << prefix << "bbox_cx," << prefix << "bbox_cy," << prefix << "bbox_cz"
             << "," << prefix << "bbox_sx," << prefix << "bbox_sy," << prefix << "bbox_sz";
    };
    bool first = true;
    for (int i = 0; i < sphere_count_max; ++i)
    {
        if (!first) file << ",";
        first = false;
        slot_header("s" + std::to_string(i) + "_");
    }
    for (int j = 0; j < cylinder_count_max; ++j)
    {
        if (!first) file << ",";
        first = false;
        slot_header("c" + std::to_string(j) + "_");
    }
    file << "\n";
}

static void writeZeroSlot(std::ofstream &file)
{
    for (int k = 0; k < OBJECTS_SLOT_FIELDS; ++k)
    {
        if (k > 0) file << ",";
        file << 0;
    }
}

template <typename SpecT>
static void writeOneSlot(std::ofstream &file,
                         const SpecT &spec,
                         int type_code,
                         float h,
                         const ObjNoiseSample &n,
                         const Eigen::Vector3f &bbox_center,
                         const Eigen::Vector3f &bbox_size)
{
    file << spec.obj_id << "," << type_code << ","
         << spec.center.x() << "," << spec.center.y() << "," << spec.center.z() << ","
         << spec.radius << "," << h << ","
         << spec.velocity.x()     << "," << spec.velocity.y()     << "," << spec.velocity.z()     << ","
         << spec.acceleration.x() << "," << spec.acceleration.y() << "," << spec.acceleration.z() << ","
         << n.v_tracker.x()       << "," << n.v_tracker.y()       << "," << n.v_tracker.z()       << ","
         << n.a_tracker.x()       << "," << n.a_tracker.y()       << "," << n.a_tracker.z()       << ","
         << n.v_unc.x()           << "," << n.v_unc.y()           << "," << n.v_unc.z()           << ","
         << n.a_unc.x()           << "," << n.a_unc.y()           << "," << n.a_unc.z()           << ","
         << n.age;
    // future_traj: second-order extrapolation (a_sim=0 ⇒ linear)
    for (int k = 1; k <= OBJECTS_T_FUTURE; ++k)
    {
        const float tk = k * OBJECTS_FUTURE_DT;
        Eigen::Vector3f p = spec.center + spec.velocity * tk + 0.5f * spec.acceleration * tk * tk;
        file << "," << p.x() << "," << p.y() << "," << p.z();
    }
    file << "," << bbox_center.x() << "," << bbox_center.y() << "," << bbox_center.z()
         << "," << bbox_size.x()   << "," << bbox_size.y()   << "," << bbox_size.z();
}

void writeObjectsCSVRow(std::ofstream &file,
                        const std::vector<SphereObstacleSpec> &spheres,
                        const std::vector<CylinderObstacleSpec> &cylinders,
                        int sphere_count_max, int cylinder_count_max,
                        const TrackerNoiseConfig &noise,
                        std::mt19937 &rng)
{
    std::normal_distribution<float> bbox_jitter(0.0f, 0.05f);

    bool first = true;

    for (int i = 0; i < sphere_count_max; ++i)
    {
        if (!first) file << ",";
        first = false;
        if (i < (int)spheres.size())
        {
            const auto &s = spheres[i];
            ObjNoiseSample n = sampleTrackerNoise(noise, s.velocity, s.acceleration, rng);
            Eigen::Vector3f bbox_center(s.center.x() + bbox_jitter(rng),
                                        s.center.y() + bbox_jitter(rng),
                                        s.center.z() + bbox_jitter(rng));
            Eigen::Vector3f bbox_size(2.0f * s.radius, 2.0f * s.radius, 2.0f * s.radius);
            writeOneSlot(file, s, 0, 0.0f, n, bbox_center, bbox_size);
        }
        else
        {
            writeZeroSlot(file);
        }
    }
    for (int j = 0; j < cylinder_count_max; ++j)
    {
        if (!first) file << ",";
        first = false;
        if (j < (int)cylinders.size())
        {
            const auto &c = cylinders[j];
            ObjNoiseSample n = sampleTrackerNoise(noise, c.velocity, c.acceleration, rng);
            Eigen::Vector3f bbox_center(c.center.x() + bbox_jitter(rng),
                                        c.center.y() + bbox_jitter(rng),
                                        c.center.z() + bbox_jitter(rng));
            Eigen::Vector3f bbox_size(2.0f * c.radius, 2.0f * c.radius, c.height);
            writeOneSlot(file, c, 1, c.height, n, bbox_center, bbox_size);
        }
        else
        {
            writeZeroSlot(file);
        }
    }
    file << "\n";
}

// Save the obj_id mask as a 16-bit PNG. cv::Mat input is CV_16SC1; we cast to
// CV_16UC1 (obj_id is non-negative so the cast preserves values).
void saveMaskAs16BitPNG(const cv::Mat &mask_int16, const std::string &filepath)
{
    cv::Mat m_u16;
    mask_int16.convertTo(m_u16, CV_16UC1);
    cv::imwrite(filepath, m_u16);
}

void prepareSavePath(const std::string &path, bool print=false)
{
    if (fs::exists(path))
    {
        if (print)
            std::cout << "Directory exists. Removing: " << path << std::endl;
        fs::remove_all(path);
    }
    fs::create_directories(path);
    if (print)
        std::cout << "Created new dataset directory: " << path << std::endl;
}

void savePointCloudAsPLY(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const std::string &path)
{
    if (pcl::io::savePLYFileBinary(path, *cloud) == -1)
        std::cerr << "Failed to save ply file to " << path << std::endl;
}

void saveDepthAs16BitPNG(const cv::Mat &depth_float, float max_depth_dist, const std::string &filepath)
{
    cv::Mat depth_scaled;
    depth_scaled = depth_float / max_depth_dist; // normalize to 0~1

    // clip [0,1]
    cv::threshold(depth_scaled, depth_scaled, 1.0, 1.0, cv::THRESH_TRUNC);
    cv::threshold(depth_scaled, depth_scaled, 0.0, 0.0, cv::THRESH_TOZERO);

    // convert to uint16
    depth_scaled.convertTo(depth_scaled, CV_16UC1, 65535.0);

    cv::imwrite(filepath, depth_scaled);
}


Eigen::Quaternionf RPY2Quat(float roll_deg, float pitch_deg, float yaw_deg)
{
    float roll = roll_deg * M_PI / 180.0f;
    float pitch = pitch_deg * M_PI / 180.0f;
    float yaw = yaw_deg * M_PI / 180.0f;
    Eigen::AngleAxisf rollAngle(roll, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitchAngle(pitch, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf yawAngle(yaw, Eigen::Vector3f::UnitZ());
    return yawAngle * pitchAngle * rollAngle;
}

void printProgressBar(int current, int total, int bar_width = 50)
{
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(bar_width * progress);

    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i)
    {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0f) << "%";
    std::cout.flush();
}

int main(int argc, char **argv)
{
    YAML::Node config = YAML::LoadFile(CONFIG_FILE_PATH);

    // 0. Dynamic obstacle config
    SwarmDatasetOptions swarm_options    = loadSwarmDatasetOptions(config);
    CylinderDatasetOptions cyl_options   = loadCylinderDatasetOptions(config);
    TrackerNoiseConfig    tracker_noise  = loadTrackerNoiseConfig(config);

    // 1. Camera parameters
    CameraParams camera;
    camera.fx = config["camera"]["fx"].as<float>();
    camera.fy = config["camera"]["fy"].as<float>();
    camera.cx = config["camera"]["cx"].as<float>();
    camera.cy = config["camera"]["cy"].as<float>();
    camera.image_width = config["camera"]["image_width"].as<int>();
    camera.image_height = config["camera"]["image_height"].as<int>();
    camera.max_depth_dist = config["camera"]["max_depth_dist"].as<float>();
    camera.normalize_depth = config["camera"]["normalize_depth"].as<bool>();
    float pitch = config["camera"]["pitch"].as<float>() * M_PI / 180.0;
    Eigen::AngleAxisf angle_axis(pitch, Eigen::Vector3f::UnitY());
    Eigen::Quaternionf quat_bc(angle_axis);

    // 1b. LiDAR parameters (for generating the 360° range image)
    LidarParams lidar;
    lidar.vertical_lines        = config["lidar"]["vertical_lines"].as<int>();
    lidar.vertical_angle_start  = config["lidar"]["vertical_angle_start"].as<float>();
    lidar.vertical_angle_end    = config["lidar"]["vertical_angle_end"].as<float>();
    lidar.horizontal_num        = config["lidar"]["horizontal_num"].as<int>();
    lidar.horizontal_resolution = config["lidar"]["horizontal_resolution"].as<float>();
    lidar.max_lidar_dist        = config["lidar"]["max_lidar_dist"].as<float>();

    // 2. Map parameters
    float resolution = config["resolution"].as<float>();
    int occupy_threshold = config["occupy_threshold"].as<int>();
    int seed = config["seed"].as<int>();
    int sizeX = config["x_length"].as<int>();
    int sizeY = config["y_length"].as<int>();
    int sizeZ = config["z_length"].as<int>();
    double scale = 1 / resolution;
    sizeX *= scale;
    sizeY *= scale;
    sizeZ *= scale;

    // 3. Dataset parameters
    std::string save_path = config["save_path"].as<std::string>();
    int env_num = config["env_num"].as<int>();
    int image_num = config["image_num"].as<int>();
    float roll_range = config["roll_range"].as<float>();
    float pitch_range = config["pitch_range"].as<float>();
    float x_range = config["x_range"].as<float>();
    float y_range = config["y_range"].as<float>();
    float z_min = config["z_range"][0].as<float>();
    float z_max = config["z_range"][1].as<float>();
    float safe_dist = config["safe_dist"].as<float>();
    float ply_res = config["ply_res"].as<float>();

    // center alignment, compute offset
    int dataset_num = env_num * image_num;
    float x_min = -x_range / 2.0f;
    float y_min = -y_range / 2.0f;

    std::cout << "Map extent (m): "
              << "X: [" << -sizeX * resolution / 2.0 << ", " << sizeX * resolution / 2.0 << "], "
              << "Y: [" << -sizeY * resolution / 2.0 << ", " << sizeY * resolution / 2.0 << "], "
              << "Z: [" << 0 << ", " << sizeZ * resolution << "]" << std::endl;

    std::cout << "Sampling extent (m): "
              << "X: [" << x_min << ", " << x_min + x_range << "], "
              << "Y: [" << y_min << ", " << y_min + y_range << "], "
              << "Z: [" << z_min << ", " << z_max << "]" << std::endl;

    std::cout << "Angle range (deg): "
              << "Roll: [" << -roll_range << ", " << roll_range << "], "
              << "Pitch: [" << -pitch_range << ", " << pitch_range << "], "
              << "Yaw: [0, 360]" << std::endl;

    // collect all data
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> normal_distribution(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform_uniform(0.0f, 1.0f);
    prepareSavePath(save_path, true);
    for (int map_i = 0; map_i < env_num; ++map_i)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        mocka::Maps::BasicInfo info;
        info.sizeX = sizeX;
        info.sizeY = sizeY;
        info.sizeZ = sizeZ;
        info.seed = seed + map_i;
        info.scale = scale;
        info.cloud = cloud;

        mocka::Maps map;
        map.setParam(config);
        map.setInfo(info);
        map.generate(config["maze_type"].as<int>());

        // build GridMap
        GridMap grid_map(cloud, resolution, occupy_threshold);

        // save map (filtered first)
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setLeafSize(ply_res, ply_res, ply_res);
        sor.filter(*filtered_cloud);
        pcl::PointXYZ min_pt, max_pt;
        pcl::getMinMax3D(*filtered_cloud, min_pt, max_pt);
        Eigen::Vector3f world_min(min_pt.x, min_pt.y, min_pt.z);
        Eigen::Vector3f world_max(max_pt.x, max_pt.y, max_pt.z);

        std::string image_path = save_path + std::to_string(map_i) + "/";
        prepareSavePath(image_path);

        savePointCloudAsPLY(filtered_cloud, save_path + "pointcloud-" + std::to_string(map_i) + ".ply");

        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(filtered_cloud);

        // open pose and dynamic obstacle CSV files
        std::ofstream pose_file(save_path + "pose-" + std::to_string(map_i) + ".csv");
        pose_file << "px,py,pz,qw,qx,qy,qz\n";

        // CPA: intercept/near_miss = 1 main cylinder (no distractors); no_threat = 1 non-threat cylinder; all_blocked = N cylinders
        const int total_cyl_max = std::max(swarm_options.sphere_count_max,
                                           swarm_options.cpa.all_blocked_count_max);

        // v5: unified objects.csv. CPA threats are vertical cylinders: sphere slots are zero, everything uses cylinder slots.
        const int obj_sphere_max = 0;
        const int obj_cyl_max    = swarm_options.enabled ? total_cyl_max : 0;
        std::ofstream objects_file;
        if (obj_sphere_max > 0 || obj_cyl_max > 0)
        {
            objects_file.open(save_path + "objects-" + std::to_string(map_i) + ".csv");
            writeObjectsHeader(objects_file, obj_sphere_max, obj_cyl_max);
        }

        // v3 sideways flight: per-frame bucket label CSV (single int per row).
        // Missing file means all S1+S2; the Python dataloader falls back to the old distribution for backward compatibility.
        std::ofstream bucket_file;
        if (swarm_options.bucket.enabled)
        {
            bucket_file.open(save_path + "bucket-" + std::to_string(map_i) + ".csv");
            bucket_file << "bucket\n";
        }

        // CPA: per-frame drone intent (world-frame velocity + goal offset), synced frame by frame with pose/objects/bucket.
        std::ofstream drone_state_file;
        if (swarm_options.bucket.enabled)
        {
            drone_state_file.open(save_path + "drone_state-" + std::to_string(map_i) + ".csv");
            drone_state_file << "vx,vy,vz,gx,gy,gz\n";
        }

        // collect data for the current environment
        for (int image_i = 0; image_i < image_num; ++image_i)
        {
            Eigen::Vector3f pos;
            float dist;
            do {
                pos.x() = x_min + uniform_uniform(rng) * x_range;
                pos.y() = y_min + uniform_uniform(rng) * y_range;
                pos.z() = z_min + uniform_uniform(rng) * (z_max - z_min);
                pcl::PointXYZ searchPoint(pos.x(), pos.y(), pos.z());
                std::vector<int> pointIdxNKNSearch(1);
                std::vector<float> pointNKNSquaredDistance(1);
                kdtree.nearestKSearch(searchPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance);
                dist = std::sqrt(pointNKNSquaredDistance[0]);
            } while (dist < safe_dist);

            float roll  = normal_distribution(rng) * roll_range  / 3.0f;
            float pitch = normal_distribution(rng) * pitch_range / 3.0f;
            float yaw   = uniform_uniform(rng) * 360.0f;

            Eigen::Quaternionf quat    = RPY2Quat(roll, pitch, yaw);
            Eigen::Quaternionf quat_wc = quat * quat_bc;

            cudaMat::SE3<float> T_wc(quat_wc.w(), quat_wc.x(), quat_wc.y(), quat_wc.z(),
                                     pos.x(), pos.y(), pos.z());

            // phase5/6 world-azimuth frame: LiDAR render frame = world-axis aligned (yaw zeroed too, not just roll/pitch),
            //   range/mask azimuth columns = world azimuth (independent of heading yaw) → free yaw.
            Eigen::Quaternionf quat_world_aligned   = RPY2Quat(0.0f, 0.0f, 0.0f);
            Eigen::Quaternionf quat_wc_world        = quat_world_aligned * quat_bc;
            cudaMat::SE3<float> T_wc_yaw_only(
                quat_wc_world.w(), quat_wc_world.x(),
                quat_wc_world.y(), quat_wc_world.z(),
                pos.x(), pos.y(), pos.z());

            // Visibility reference for cylinder placement: render a static-only panorama first (no obstacles) and compare cylinder vs static distance for occlusion.
            cv::Mat static_range;
            renderRangeImage(&grid_map, &lidar, T_wc_yaw_only, static_range,
                             std::vector<SphereObstacle>{}, std::vector<CylinderObstacle>{});

            // 1. Sample drone intent + place cylinders by CPA bucket (horizontal inverse solve, gated per bucket)
            std::vector<SphereObstacleSpec>   dynamic_obstacles;   // retired: always empty
            std::vector<CylinderObstacleSpec> dynamic_cylinders;
            int bucket_id = 0;
            DroneIntent intent;   // per-frame intent (written to drone_state CSV); zero when buckets are disabled
            if (swarm_options.enabled && swarm_options.bucket.enabled)
            {
                const CpaOptions &cpa = swarm_options.cpa;
                intent = sampleDroneIntent(cpa, rng);
                const Eigen::Vector3f &u = intent.vel;
                bucket_id = sampleBucketId(swarm_options.bucket, rng);

                // Non-threat cylinders: same CPA inverse solve, but miss from the no_threat range (3~8m) → "present but not threatening",
                // relative velocity controlled, miss always > main threat, never becomes the nearest threat and pollutes the bucket label. No random placement.
                auto placeNonThreat = [&](int n) {
                    for (int i = 0; i < n; ++i)
                    {
                        CylinderObstacleSpec s;
                        if (solveCpaMainCylinder(cpa, cpa.miss_no_threat_min, cpa.miss_no_threat_max,
                                                 swarm_options, pos, u, kdtree, world_min, world_max, static_range, lidar, rng, s))
                            dynamic_cylinders.push_back(s);
                    }
                };

                switch (bucket_id)
                {
                case BUCKET_NO_DYNAMIC:
                    break;                                  // empty mask
                case BUCKET_INTERCEPT:
                case BUCKET_NEAR_MISS:
                {
                    const bool is_int = (bucket_id == BUCKET_INTERCEPT);
                    const float mlo = is_int ? cpa.miss_intercept_min : cpa.miss_near_miss_min;
                    const float mhi = is_int ? cpa.miss_intercept_max : cpa.miss_near_miss_max;
                    CylinderObstacleSpec main_cyl;
                    if (solveCpaMainCylinder(cpa, mlo, mhi, swarm_options, pos, u,
                                             kdtree, world_min, world_max, static_range, lidar, rng, main_cyl))
                    {
                        dynamic_cylinders.push_back(main_cyl);   // main threat cylinder only, no background distractors
                    }
                    else
                    {
                        bucket_id = BUCKET_NO_THREAT;          // fall back when the inverse solve fails
                        placeNonThreat(1);
                    }
                    break;
                }
                case BUCKET_NO_THREAT:
                    placeNonThreat(1);                          // single non-threat cylinder
                    break;
                case BUCKET_ALL_BLOCKED:
                {
                    std::uniform_int_distribution<int> nb(cpa.all_blocked_count_min, cpa.all_blocked_count_max);
                    const int n = nb(rng);
                    // Explicit azimuth coverage: split the circle into n sectors (random phase), each cylinder solves its spawn azimuth in its own sector →
                    // threats surround the drone instead of theta sampling collapsing to one azimuth. Sectors needing excessive speed (e.g. straight behind)
                    // are skipped honestly (forward/sideways still spread as a fan).
                    std::uniform_real_distribution<float> u_phase(0.0f, 2.0f * static_cast<float>(M_PI));
                    const float phase  = u_phase(rng);
                    const float sector = 2.0f * static_cast<float>(M_PI) / static_cast<float>(n);
                    for (int i = 0; i < n; ++i)
                    {
                        const float phi_c = phase + (static_cast<float>(i) + 0.5f) * sector;
                        CylinderObstacleSpec s;
                        if (solveCpaCylinderInSector(cpa, cpa.miss_all_blocked_min, cpa.miss_all_blocked_max,
                                                     phi_c, 0.5f * sector, swarm_options, pos, u,
                                                     kdtree, world_min, world_max, static_range, lidar, rng, s))
                            dynamic_cylinders.push_back(s);
                    }
                    if (dynamic_cylinders.empty())
                        bucket_id = BUCKET_NO_THREAT;       // extreme case: everything failed, fall back
                    break;
                }
                default:
                    break;
                }

                // Honesty guard: no_threat (including fallbacks) with no non-threat cylinder placed → revert to no_dynamic,
                // so an empty scene never masquerades as the no_threat bucket.
                if (bucket_id == BUCKET_NO_THREAT && dynamic_cylinders.empty())
                    bucket_id = BUCKET_NO_DYNAMIC;
            }
            // Threats are the CPA-solved vertical cylinders above (dynamic_cylinders); random clutter cylinders are retired.

            // 1b. v5: assign per-sample obj_id (1-based, sphere then cylinder).
            //     obj_id == 0 is reserved for "static/no-hit" in the mask channel.
            {
                int next_obj_id = 1;
                for (auto &s : dynamic_obstacles) s.obj_id = next_obj_id++;
                for (auto &c : dynamic_cylinders) c.obj_id = next_obj_id++;
            }

            // 2. Build the render frame (t=0, current position)
            const std::vector<SphereObstacle>   frame_spheres   = buildDynamicSphereFrame(dynamic_obstacles, 0.0f);
            const std::vector<CylinderObstacle> frame_cylinders = buildDynamicCylinderFrame(dynamic_cylinders, 0.0f);

            // 3. LiDAR range image with obstacles + per-pixel obj_id mask
            //    Rendered with world-axis aligned T_wc (T_wc_yaw_only is now de-yawed to the world-azimuth frame):
            //    roll/pitch/yaw never enter the LiDAR frame, azimuth columns are world azimuth, no post-processing needed.
            cv::Mat range_image, mask_image;
            renderRangeImage(&grid_map, &lidar, T_wc_yaw_only, range_image, mask_image,
                              frame_spheres, frame_cylinders);
            std::string range_filename = image_path + "/range_" + std::to_string(image_i) + ".png";
            saveDepthAs16BitPNG(range_image, lidar.max_lidar_dist, range_filename);
            std::string mask_filename  = image_path + "/mask_"  + std::to_string(image_i) + ".png";
            saveMaskAs16BitPNG(mask_image, mask_filename);

            // pose.csv stores the world-axis aligned attitude (identity), consistent with the de-yawed LiDAR render frame;
            // the dataloader then gets R_WB=I, R_Bw=I → obs/goal land in the world-azimuth frame (= range image frame).
            pose_file << std::fixed << std::setprecision(6)
                      << pos.x() << "," << pos.y() << "," << pos.z() << ","
                      << "1.000000,0.000000,0.000000,0.000000\n";

            // 4. v5: write the unified objects.csv (with v_tracker / v_unc / future_traj / bbox / age)
            if (objects_file.is_open())
                writeObjectsCSVRow(objects_file, dynamic_obstacles, dynamic_cylinders,
                                   obj_sphere_max, obj_cyl_max, tracker_noise, rng);

            // v3: bucket label (appended in sync with pose / objects, one int column)
            if (bucket_file.is_open())
                bucket_file << bucket_id << "\n";

            // CPA: drone intent velocity + goal offset (synced frame by frame with pose/bucket)
            if (drone_state_file.is_open())
                drone_state_file << std::fixed << std::setprecision(6)
                                 << intent.vel.x()  << "," << intent.vel.y()  << "," << intent.vel.z()  << ","
                                 << intent.goal.x() << "," << intent.goal.y() << "," << intent.goal.z() << "\n";

            printProgressBar(map_i * image_num + image_i + 1, dataset_num);
        }

        pose_file.close();
        if (objects_file.is_open()) objects_file.close();
        if (bucket_file.is_open()) bucket_file.close();
        if (drone_state_file.is_open()) drone_state_file.close();
        grid_map.freeGridMap();
    }

    std::cout << "\nDataset generation completed!" << std::endl;

    return 0;
}
