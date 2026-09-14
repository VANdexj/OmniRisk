#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/common/eigen.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include <pcl_ros/point_cloud.h>
#include <cv_bridge/cv_bridge.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PointStamped.h>
#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <yaml-cpp/yaml.h>
#include "sensor_simulator.cuh"
#include <chrono>
#include "maps.hpp"

using namespace raycast;

class SensorSimulator {
public:
    SensorSimulator(ros::NodeHandle &nh) : nh_(nh) {
        YAML::Node config = YAML::LoadFile(CONFIG_FILE_PATH);
        // load camera params
        camera = new CameraParams();
        camera->fx = config["camera"]["fx"].as<float>();
        camera->fy = config["camera"]["fy"].as<float>();
        camera->cx = config["camera"]["cx"].as<float>();
        camera->cy = config["camera"]["cy"].as<float>();
        camera->image_width = config["camera"]["image_width"].as<int>();
        camera->image_height = config["camera"]["image_height"].as<int>();
        camera->max_depth_dist = config["camera"]["max_depth_dist"].as<float>();
        camera->normalize_depth = config["camera"]["normalize_depth"].as<bool>();
        float pitch = config["camera"]["pitch"].as<float>() * M_PI / 180.0;
        quat_bc = Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY());

        // load lidar params
        lidar = new LidarParams();
        lidar->vertical_lines = config["lidar"]["vertical_lines"].as<int>();
        lidar->vertical_angle_start = config["lidar"]["vertical_angle_start"].as<float>();
        lidar->vertical_angle_end = config["lidar"]["vertical_angle_end"].as<float>();
        lidar->horizontal_num = config["lidar"]["horizontal_num"].as<int>();
        lidar->horizontal_resolution = config["lidar"]["horizontal_resolution"].as<float>();
        lidar->max_lidar_dist = config["lidar"]["max_lidar_dist"].as<float>();

        render_lidar = config["render_lidar"].as<bool>();
        render_depth = config["render_depth"].as<bool>();
        float depth_fps = config["depth_fps"].as<float>();
        float lidar_fps = config["lidar_fps"].as<float>();
        depth_pub_duration = ros::Duration(1 / depth_fps);
        lidar_pub_duration = ros::Duration(1 / lidar_fps);
        
        std::string ply_file = config["ply_file"].as<std::string>();
        std::string odom_topic = config["odom_topic"].as<std::string>();
        std::string depth_topic = config["depth_topic"].as<std::string>();
        std::string lidar_topic = config["lidar_topic"].as<std::string>();

        // load map params
        bool use_random_map = config["random_map"].as<bool>();
        float resolution = config["resolution"].as<float>();
        int occupy_threshold = config["occupy_threshold"].as<int>();
        pcl_pub = nh.advertise<sensor_msgs::PointCloud2>("/sim/static_map", 1, true);
        int seed = config["seed"].as<int>();
        int sizeX = config["x_length"].as<int>();
        int sizeY = config["y_length"].as<int>();
        int sizeZ = config["z_length"].as<int>();
        int type = config["maze_type"].as<int>();
        double scale = 1 / resolution;
        sizeX = sizeX * scale;
        sizeY = sizeY * scale;
        sizeZ = sizeZ * scale;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        if (use_random_map) {
            printf("1.Generate Random Map... \n");
            mocka::Maps::BasicInfo info;
            info.sizeX      = sizeX;
            info.sizeY      = sizeY;
            info.sizeZ      = sizeZ;
            info.seed       = seed;
            info.scale      = scale;
            info.cloud      = cloud;

            mocka::Maps map;
            map.setParam(config);
            map.setInfo(info);
            map.generate(type);
            static_poles_  = map.getStaticPoles();
            static_walls_  = map.getStaticWalls();
            arena_x_half_  = map.getArenaXHalf();
            arena_y_half_  = map.getArenaYHalf();
            maze_type_     = type;
        }
        else {
            printf("1.Reading Point Cloud %s... \n", ply_file.c_str());
            if (pcl::io::loadPLYFile(ply_file, *cloud) == -1) {
                PCL_ERROR("Couldn't read PLY file \n");
            }
        }
        pcl::toROSMsg(*cloud, output);
        output.header.frame_id = "world";

        std::cout<<"Pointloud size:"<<cloud->points.size()<<std::endl;
        printf("2.Mapping... \n");
        grid_map = new GridMap(cloud, resolution, occupy_threshold);

        initDynamicObstacles(config, cloud);

        ros::Time next_depth_pub_time = ros::Time::now();
        ros::Time next_lidar_pub_time = ros::Time::now();

        // ROS
        image_pub_ = nh_.advertise<sensor_msgs::Image>(depth_topic, 1);
        point_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(lidar_topic, 1);
        odom_sub_ = nh_.subscribe(odom_topic, 1, &SensorSimulator::odomCallback, this, ros::TransportHints().tcpNoDelay());
        timer_map_   = nh_.createTimer(ros::Duration(1), &SensorSimulator::timerMapCallback, this);

        if (sim_dynamic_enabled_) {
            float obs_hz = 100.f;
            if (config["sim_dynamic"] && config["sim_dynamic"]["update_hz"])
                obs_hz = config["sim_dynamic"]["update_hz"].as<float>();
            timer_obstacles_ = nh_.createTimer(ros::Duration(1.0 / obs_hz),
                                               &SensorSimulator::timerObstacleCallback, this);
            marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/sim/dynamic_obstacle_markers", 1, true);
        }

        if (scenario_enabled_) {
            // RViz "Publish Point" tool → /clicked_point → manually launch a wave of balls
            scenario_trigger_sub_ = nh_.subscribe("/clicked_point", 1,
                                                  &SensorSimulator::scenarioTriggerCallback, this);
        }

        if (arena_x_half_ > 0.f) {
            static_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
                "/sim/static_obstacle_markers", 1, true);
            publishStaticMarkers();
        }

        collision_check_enabled_ = config["collision_check"] &&
                                   config["collision_check"]["enabled"].as<bool>();
        if (collision_check_enabled_) {
            if (config["collision_check"]["drone_radius"])
                drone_radius_ = config["collision_check"]["drone_radius"].as<float>();
            pole_contact_.assign(static_poles_.size(), 0);
            sphere_contact_.assign(sphere_obstacles_.size(), 0);
            cyl_contact_.assign(cylinder_obstacles_.size(), 0);
            collision_marker_pub_ = nh_.advertise<visualization_msgs::Marker>(
                "/sim/collision_counter", 1, true);
            ROS_INFO("[collision_check] enabled: drone_radius=%.2f, %zu static poles",
                     drone_radius_, static_poles_.size());
        }

        printf("3.Simulation Ready! \n");
        ros::spin();
    }

    void odomCallback(const nav_msgs::Odometry::ConstPtr &msg);

    void renderDepthCallback(const ros::Time stamp);

    void renderLidarCallback(const ros::Time stamp);

    void timerMapCallback(const ros::TimerEvent &);

    void timerObstacleCallback(const ros::TimerEvent &);

private:
    // Omnidirectional avoidance test ball spec (angles in radians): offset = lateral offset (m); aim_up = aim point raise (m);
    // spawn_delay = delay after trigger (s), staggers a wave so each ball re-solves its collision course from the current drone position.
    struct BallSpec { float bearing, elev, speed, lead_time, offset, aim_up, spawn_delay; };

    void initDynamicObstacles(const YAML::Node& config, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
    void parseScenario(const YAML::Node& n);
    void armScenarioWave();                       // trigger: reset the whole wave, queue by spawn_delay
    void serviceScenarioSpawns();                 // each tick: spawn due balls on a course computed from the current drone position
    void spawnScenarioBall(const BallSpec& spec); // single ball: solve the collision course from the current pos/vel
    void scenarioTriggerCallback(const geometry_msgs::PointStamped::ConstPtr& msg);
    void publishObstacleMarkers();
    void publishStaticMarkers();
    void checkCollisions();
    void publishCollisionMarker();

    bool render_depth{false};
    bool render_lidar{false};
    Eigen::Quaternionf quat;
    Eigen::Quaternionf quat_bc, quat_wc;
    Eigen::Vector3f pos;

    CameraParams* camera;
    LidarParams* lidar;
    GridMap* grid_map;
    sensor_msgs::PointCloud2 output;

    // dynamic obstacles
    bool sim_dynamic_enabled_{false};
    std::vector<SphereObstacle>   sphere_obstacles_;
    std::vector<CylinderObstacle> cylinder_obstacles_;
    std::vector<std::array<float,3>> sphere_velocities_;
    std::vector<std::array<float,3>> cyl_velocities_;
    float world_min_[3]{}, world_max_[3]{};

    // Omnidirectional avoidance test scenario: balls are triggered manually by RViz "Publish Point" (/clicked_point), then fly straight at constant speed,
    // speed locked, no bounces. A wave spawns staggered by spawn_delay, each ball re-solves its collision course from pos/vel at spawn time
    // (CPA≈0), so later balls still aim precisely after the drone has dodged earlier ones — no longer using the stale position at trigger time.
    bool scenario_enabled_{false};
    bool odom_init_{false};
    std::string scenario_name_;
    float scenario_sphere_radius_{0.25f};
    Eigen::Vector3f vel_ = Eigen::Vector3f::Zero();
    std::vector<BallSpec> scenario_specs_;
    std::vector<char> scenario_spawned_;          // whether each ball in this wave has spawned (same order as scenario_specs_)
    bool scenario_armed_{false};                  // whether this wave still has balls waiting to spawn
    ros::Time scenario_trigger_time_;             // trigger time of this wave, for spawn_delay
    ros::Subscriber scenario_trigger_sub_;

    // static obstacle markers (arena poles + ground)
    std::vector<mocka::StaticPole> static_poles_;
    std::vector<mocka::StaticWall> static_walls_;
    float arena_x_half_{0.f}, arena_y_half_{0.f};
    int   maze_type_{0};        // restricts dynamic cylinder grounding to the maze9 corridor scene
    ros::Publisher static_marker_pub_;

    // collision check: count collisions with obstacles (static poles/dynamic balls/dynamic cylinders) using ground-truth pose
    bool  collision_check_enabled_{false};
    float drone_radius_{0.2f};
    int   collision_count_static_{0}, collision_count_dynamic_{0};
    std::vector<char> pole_contact_, sphere_contact_, cyl_contact_;  // rising-edge debounce
    ros::Publisher collision_marker_pub_;

    ros::NodeHandle nh_;
    ros::Publisher image_pub_, point_cloud_pub_;
    ros::Publisher pcl_pub;
    ros::Publisher marker_pub_;
    ros::Subscriber odom_sub_;
    ros::Timer timer_depth_, timer_lidar_, timer_map_, timer_obstacles_;

    ros::Time next_depth_pub_time, next_lidar_pub_time;
    ros::Duration depth_pub_duration, lidar_pub_duration;
    double depth_time{0.0}, lidar_time{0.0};
    int depth_count{0}, lidar_count{0};
    // mocka::Maps map;
};



void SensorSimulator::renderDepthCallback(const ros::Time stamp) {
    if (!render_depth)
        return;

    auto start = std::chrono::high_resolution_clock::now();

    cudaMat::SE3<float> T_wc(quat_wc.w(), quat_wc.x(), quat_wc.y(), quat_wc.z(), pos.x(), pos.y(), pos.z());
    cv::Mat depth_image;
    renderDepthImage(grid_map, camera, T_wc, depth_image, sphere_obstacles_, cylinder_obstacles_);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    depth_time += elapsed.count();
    depth_count++;
    // std::cout << "Image render time: " << elapsed.count() << " s" << std::endl;

    sensor_msgs::Image ros_image;
    cv_bridge::CvImage cv_image;
    cv_image.header.stamp = stamp;
    cv_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    cv_image.image = depth_image;
    cv_image.toImageMsg(ros_image);
    image_pub_.publish(ros_image);
}

void SensorSimulator::timerMapCallback(const ros::TimerEvent&) {
    pcl_pub.publish(output);
}

void SensorSimulator::renderLidarCallback(const ros::Time stamp) {
    if (!render_lidar)
        return;

    auto start = std::chrono::high_resolution_clock::now();

    cudaMat::SE3<float> T_wc(quat.w(), quat.x(), quat.y(), quat.z(), pos.x(), pos.y(), pos.z());
    pcl::PointCloud<pcl::PointXYZ> lidar_points;
    renderLidarPointcloud(grid_map, lidar, T_wc, lidar_points, sphere_obstacles_, cylinder_obstacles_);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    lidar_time += elapsed.count();
    lidar_count++;
    // std::cout << "LiDAR render time: " << elapsed.count() << " s" << std::endl;

    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(lidar_points, output);
    output.header.stamp = stamp;
    output.header.frame_id = "odom";
    point_cloud_pub_.publish(output);
}

void SensorSimulator::odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    quat.x() = msg->pose.pose.orientation.x;
    quat.y() = msg->pose.pose.orientation.y;
    quat.z() = msg->pose.pose.orientation.z;
    quat.w() = msg->pose.pose.orientation.w;
    quat_wc = quat * quat_bc;

    pos.x() = msg->pose.pose.position.x;
    pos.y() = msg->pose.pose.position.y;
    pos.z() = msg->pose.pose.position.z;

    vel_.x() = msg->twist.twist.linear.x;
    vel_.y() = msg->twist.twist.linear.y;
    vel_.z() = msg->twist.twist.linear.z;
    odom_init_ = true;

    checkCollisions();

    ros::Time tnow = ros::Time::now();

    // avoid a large time gap when simulated odom messages are interrupted
    if (fabs((tnow - next_depth_pub_time).toSec()) > 10 * depth_pub_duration.toSec())
        next_depth_pub_time = tnow;
    if (fabs((tnow - next_lidar_pub_time).toSec()) > 10 * lidar_pub_duration.toSec())
        next_lidar_pub_time = tnow;

    if (tnow >= next_depth_pub_time){
        next_depth_pub_time += depth_pub_duration;
        renderDepthCallback(msg->header.stamp);
    }
    if (tnow >= next_lidar_pub_time){
        next_lidar_pub_time += lidar_pub_duration;
        renderLidarCallback(msg->header.stamp);
    }
    ros::Duration render_duration = ros::Time::now() - tnow;
    if (render_duration > depth_pub_duration || render_duration > lidar_pub_duration){
        // Performance reference: should take < 1 ms on 3060 GPU & Ubuntu 20.04
        ROS_WARN("Current Rendering time: %.2f ms, delay too much!", 1000 * render_duration.toSec());
        std::cout << "Average Depth Rendering time: " << (depth_time / (depth_count + 1e-8)) * 1000 << " ms" << std::endl;
        std::cout << "Average Lidar Rendering time: " << (lidar_time / (lidar_count + 1e-8)) * 1000 << " ms" << std::endl;
    }
}

void SensorSimulator::parseScenario(const YAML::Node& n)
{
    scenario_name_          = n["scenario"].as<std::string>();
    scenario_sphere_radius_ = n["sphere_radius"] ? n["sphere_radius"].as<float>() : 0.25f;
    const YAML::Node specs = n["scenarios"] ? n["scenarios"][scenario_name_] : YAML::Node();
    if (!specs || !specs.IsSequence()) {
        ROS_ERROR("[sim_scenario] scenario '%s' not found under scenarios:", scenario_name_.c_str());
        return;
    }
    const float deg2rad = static_cast<float>(M_PI) / 180.0f;
    for (const auto& s : specs) {
        BallSpec b;
        b.bearing   = s["bearing_deg"].as<float>() * deg2rad;
        b.elev      = (s["elev_deg"] ? s["elev_deg"].as<float>() : 0.0f) * deg2rad;
        b.speed     = s["speed"].as<float>();
        b.lead_time = s["lead_time"].as<float>();
        b.offset      = s["offset"]      ? s["offset"].as<float>()      : 0.0f;
        b.aim_up      = s["aim_up"]      ? s["aim_up"].as<float>()      : 0.0f;
        b.spawn_delay = s["spawn_delay"] ? s["spawn_delay"].as<float>() : 0.0f;
        scenario_specs_.push_back(b);
    }
    ROS_INFO("[sim_scenario] '%s': %zu balls — trigger via RViz 'Publish Point' (/clicked_point)",
             scenario_name_.c_str(), scenario_specs_.size());
}


// RViz "Publish Point" trigger: clear the previous wave and queue this one (the clicked coordinate is ignored).
void SensorSimulator::scenarioTriggerCallback(const geometry_msgs::PointStamped::ConstPtr&)
{
    if (!scenario_enabled_) return;
    if (!odom_init_) { ROS_WARN("[sim_scenario] trigger ignored: no odom yet"); return; }
    armScenarioWave();
}

// Queue on trigger: clear the previous wave (including leftover markers) and record trigger time; each ball spawns at its spawn_delay.
void SensorSimulator::armScenarioWave()
{
    sphere_obstacles_.clear();
    sphere_velocities_.clear();
    sphere_contact_.clear();
    scenario_spawned_.assign(scenario_specs_.size(), 0);
    scenario_trigger_time_ = ros::Time::now();
    scenario_armed_ = true;

    visualization_msgs::MarkerArray del;          // clear leftover markers from the previous wave so no ghosts remain while staggered
    visualization_msgs::Marker m;
    m.ns = "dyn_obs"; m.action = visualization_msgs::Marker::DELETEALL;
    del.markers.push_back(m);
    marker_pub_.publish(del);

    serviceScenarioSpawns();                       // spawn balls with spawn_delay=0 immediately
    ROS_INFO("[sim_scenario] '%s' armed: %zu balls queued by spawn_delay",
             scenario_name_.c_str(), scenario_specs_.size());
}

// Called every tick: due balls spawn on a course computed from the current pos/vel (later balls re-aim if the drone has drifted).
void SensorSimulator::serviceScenarioSpawns()
{
    if (!scenario_armed_) return;
    const float elapsed = (ros::Time::now() - scenario_trigger_time_).toSec();
    bool all_done = true;
    for (size_t i = 0; i < scenario_specs_.size(); ++i) {
        if (scenario_spawned_[i]) continue;
        if (elapsed >= scenario_specs_[i].spawn_delay) {
            spawnScenarioBall(scenario_specs_[i]);
            scenario_spawned_[i] = 1;
        } else {
            all_done = false;
        }
    }
    if (all_done) scenario_armed_ = false;
    publishObstacleMarkers();
}

// Single ball spawn: closed-form collision course from the current pos/vel (CPA≈0), then straight flight with locked speed.
void SensorSimulator::spawnScenarioBall(const BallSpec& spec)
{
    if (spec.speed < 0.2f || spec.speed > 4.0f)
        ROS_WARN("[sim_scenario] ball speed %.2f out of training range [0.2,4.0]", spec.speed);

    // Drone course: prefer velocity azimuth; fall back to heading yaw when cruise speed is too low (hover/not airborne).
    float psi;
    if (vel_.head<2>().norm() > 0.2f) {
        psi = std::atan2(vel_.y(), vel_.x());
    } else {
        psi = std::atan2(2.0f * (quat.w() * quat.z() + quat.x() * quat.y()),
                         1.0f - 2.0f * (quat.y() * quat.y() + quat.z() * quat.z()));
    }

    // approach unit vector (spawn azimuth relative to drone course)
    const float az = psi + spec.bearing, el = spec.elev;
    const Eigen::Vector3f come_from(std::cos(el) * std::cos(az),
                                    std::cos(el) * std::sin(az),
                                    std::sin(el));
    // Closed-form lead construction (always solvable, CPA=0):
    //   intercept point P* = predicted drone position τ s ahead; ball velocity toward the drone path (= −approach·speed);
    //   spawn point = τ s of flight back from P* → the ball reaches P* exactly after τ s.
    const Eigen::Vector3f v_o   = -come_from * spec.speed;
    // lateral offset: shift the aim point along the horizontal left of the approach (+90°) by offset, passing by at CPA≈|offset|
    const Eigen::Vector3f left_hat(-std::sin(az), std::cos(az), 0.0f);
    // aim_up: raise the aim point by aim_up meters; with elev>0 the ball dives from above and stays within the upper lidar FOV.
    const Eigen::Vector3f Pstar = pos + vel_ * spec.lead_time + left_hat * spec.offset
                                  + Eigen::Vector3f(0.0f, 0.0f, spec.aim_up);
    const Eigen::Vector3f p_o   = Pstar - v_o * spec.lead_time;
    SphereObstacle obs;
    obs.center = {p_o.x(), p_o.y(), p_o.z()};
    obs.radius = scenario_sphere_radius_;
    sphere_obstacles_.push_back(obs);
    sphere_velocities_.push_back({v_o.x(), v_o.y(), v_o.z()});
    sphere_contact_.push_back(0);                  // collision debounce buffer grows with each ball
}

void SensorSimulator::initDynamicObstacles(const YAML::Node& config, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    // Test scenario first: balls spawn only when maybeSpawnScenario() triggers (needs drone odom).
    // reuses sim_dynamic_enabled_ to create the obstacle update timer + marker publisher.
    if (config["sim_scenario"] && config["sim_scenario"]["enabled"].as<bool>()) {
        parseScenario(config["sim_scenario"]);
        scenario_enabled_ = true;
        sim_dynamic_enabled_ = true;
        return;
    }

    if (!config["sim_dynamic"] || !config["sim_dynamic"]["enabled"].as<bool>())
        return;
    sim_dynamic_enabled_ = true;
    const auto& n = config["sim_dynamic"];

    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(*cloud, min_pt, max_pt);
    world_min_[0] = min_pt.x; world_min_[1] = min_pt.y; world_min_[2] = min_pt.z;
    world_max_[0] = max_pt.x; world_max_[1] = max_pt.y; world_max_[2] = max_pt.z;

    // maze types that expose their traversable half-extent (arena, corridor)
    // confine spawn/bounce to it instead of the whole map's point-cloud bbox,
    // e.g. a 60m-long x 4m-wide corridor would otherwise let obstacles roam
    // across the full (much wider) map bbox and clip through the corridor walls.
    if (arena_x_half_ > 0.f) {
        world_min_[0] = -arena_x_half_; world_max_[0] = arena_x_half_;
        world_min_[1] = -arena_y_half_; world_max_[1] = arena_y_half_;
    }

    const float z_lo = n["z_min"] ? n["z_min"].as<float>() : 0.5f;
    const float z_hi = n["z_max"] ? n["z_max"].as<float>() : 4.0f;
    const float margin = 1.5f;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> rx(world_min_[0] + margin, world_max_[0] - margin);
    std::uniform_real_distribution<float> ry(world_min_[1] + margin, world_max_[1] - margin);
    std::uniform_real_distribution<float> rangle(0.f, 2.f * static_cast<float>(M_PI));

    const int   n_sph    = n["sphere_count"]    ? n["sphere_count"].as<int>()    : 0;
    const float sph_r    = n["sphere_radius"]   ? n["sphere_radius"].as<float>() : 0.25f;
    const float sph_vmin = n["sphere_speed_min"]? n["sphere_speed_min"].as<float>(): 0.5f;
    const float sph_vmax = n["sphere_speed_max"]? n["sphere_speed_max"].as<float>(): 2.0f;
    std::uniform_real_distribution<float> rsph_v(sph_vmin, sph_vmax);
    std::uniform_real_distribution<float> rsph_z(z_lo, z_hi);
    std::uniform_real_distribution<float> relev(-0.3f, 0.3f);

    for (int i = 0; i < n_sph; ++i) {
        SphereObstacle obs;
        obs.center = {rx(rng), ry(rng), rsph_z(rng)};
        obs.radius = sph_r;
        sphere_obstacles_.push_back(obs);
        const float spd = rsph_v(rng), az = rangle(rng), el = relev(rng);
        sphere_velocities_.push_back({spd * std::cos(el) * std::cos(az),
                                      spd * std::cos(el) * std::sin(az),
                                      spd * std::sin(el)});
    }

    const int   n_cyl    = n["cylinder_count"]       ? n["cylinder_count"].as<int>()        : 0;
    const float cyl_rmin = n["cylinder_radius_min"]  ? n["cylinder_radius_min"].as<float>() : 0.25f;
    const float cyl_rmax = n["cylinder_radius_max"]  ? n["cylinder_radius_max"].as<float>() : 0.50f;
    const float cyl_hmin = n["cylinder_height_min"]  ? n["cylinder_height_min"].as<float>() : 1.0f;
    const float cyl_hmax = n["cylinder_height_max"]  ? n["cylinder_height_max"].as<float>() : 2.5f;
    const float cyl_vmin = n["cylinder_speed_min"]   ? n["cylinder_speed_min"].as<float>()  : 0.3f;
    const float cyl_vmax = n["cylinder_speed_max"]   ? n["cylinder_speed_max"].as<float>()  : 1.5f;
    std::uniform_real_distribution<float> rcyl_r(cyl_rmin, cyl_rmax);
    std::uniform_real_distribution<float> rcyl_h(cyl_hmin, cyl_hmax);
    std::uniform_real_distribution<float> rcyl_v(cyl_vmin, cyl_vmax);

    for (int i = 0; i < n_cyl; ++i) {
        CylinderObstacle obs;
        obs.radius = rcyl_r(rng);
        obs.height = rcyl_h(rng);
        float cz;
        if (maze_type_ == 9) {
            cz = 0.5f * obs.height;   // corridor scene requires grounding: cylinder bottom sits on the ground z=0, not floating
        } else {
            const float cz_lo = std::max(z_lo, world_min_[2] + 0.5f * obs.height);
            const float cz_hi = std::min(z_hi, world_max_[2] - 0.5f * obs.height);
            cz = (cz_lo < cz_hi)
                ? std::uniform_real_distribution<float>(cz_lo, cz_hi)(rng)
                : (cz_lo + cz_hi) * 0.5f;
        }
        obs.center = {rx(rng), ry(rng), cz};
        cylinder_obstacles_.push_back(obs);
        const float spd = rcyl_v(rng), az = rangle(rng);
        cyl_velocities_.push_back({spd * std::cos(az), spd * std::sin(az), 0.f});
    }

    ROS_INFO("[sim_dynamic] enabled: %d spheres, %d cylinders", n_sph, n_cyl);
}

void SensorSimulator::timerObstacleCallback(const ros::TimerEvent& event)
{
    if (!sim_dynamic_enabled_) return;
    const float dt = static_cast<float>(
        std::min(event.current_real.toSec() - event.last_real.toSec(), 0.1));

    // Test scenario: after spawning, balls fly straight with locked speed and no bounces (sphere_obstacles_ is empty before spawning).
    if (scenario_enabled_) {
        serviceScenarioSpawns();        // queued balls that are due spawn first (solved from the current drone position)
        for (size_t i = 0; i < sphere_obstacles_.size(); ++i) {
            auto& c = sphere_obstacles_[i].center;
            const auto& v = sphere_velocities_[i];
            c.x += v[0] * dt; c.y += v[1] * dt; c.z += v[2] * dt;
        }
        publishObstacleMarkers();
        return;
    }

    auto bounce = [](float& pos, float& vel, float lo, float hi) {
        if (lo >= hi) return;
        if (pos < lo) { pos = lo; vel =  std::abs(vel); }
        if (pos > hi) { pos = hi; vel = -std::abs(vel); }
    };

    for (int i = 0; i < (int)sphere_obstacles_.size(); ++i) {
        auto& c = sphere_obstacles_[i].center;
        auto& v = sphere_velocities_[i];
        c.x += v[0] * dt; c.y += v[1] * dt; c.z += v[2] * dt;
        const float r = sphere_obstacles_[i].radius;
        bounce(c.x, v[0], world_min_[0]+r, world_max_[0]-r);
        bounce(c.y, v[1], world_min_[1]+r, world_max_[1]-r);
        bounce(c.z, v[2], world_min_[2]+r, world_max_[2]-r);
    }

    for (int i = 0; i < (int)cylinder_obstacles_.size(); ++i) {
        auto& c = cylinder_obstacles_[i].center;
        auto& v = cyl_velocities_[i];
        c.x += v[0] * dt; c.y += v[1] * dt;
        const float r  = cylinder_obstacles_[i].radius;
        const float h2 = 0.5f * cylinder_obstacles_[i].height;
        bounce(c.x, v[0], world_min_[0]+r,  world_max_[0]-r);
        bounce(c.y, v[1], world_min_[1]+r,  world_max_[1]-r);
        bounce(c.z, v[2], world_min_[2]+h2, world_max_[2]-h2);
    }

    publishObstacleMarkers();
}

void SensorSimulator::publishObstacleMarkers()
{
    visualization_msgs::MarkerArray arr;
    const ros::Time now = ros::Time::now();
    int idx = 0;

    for (const auto& obs : sphere_obstacles_) {
        visualization_msgs::Marker m;
        m.header.frame_id = "world";
        m.header.stamp    = now;
        m.ns = "dyn_obs"; m.id = idx++;
        m.action = visualization_msgs::Marker::ADD;
        m.type   = visualization_msgs::Marker::SPHERE;
        m.pose.position.x = obs.center.x;
        m.pose.position.y = obs.center.y;
        m.pose.position.z = obs.center.z;
        m.pose.orientation.w = 1.0;
        m.scale.x = m.scale.y = m.scale.z = 2.f * obs.radius;
        m.color.r = 1.0f; m.color.g = 0.3f; m.color.b = 0.1f; m.color.a = 0.75f;
        arr.markers.push_back(m);
    }

    for (const auto& obs : cylinder_obstacles_) {
        visualization_msgs::Marker m;
        m.header.frame_id = "world";
        m.header.stamp    = now;
        m.ns = "dyn_obs"; m.id = idx++;
        m.action = visualization_msgs::Marker::ADD;
        m.type   = visualization_msgs::Marker::CYLINDER;
        m.pose.position.x = obs.center.x;
        m.pose.position.y = obs.center.y;
        m.pose.position.z = obs.center.z;
        m.pose.orientation.w = 1.0;
        m.scale.x = m.scale.y = 2.f * obs.radius;
        m.scale.z = obs.height;
        m.color.r = 0.1f; m.color.g = 0.4f; m.color.b = 1.0f; m.color.a = 0.75f;
        arr.markers.push_back(m);
    }

    marker_pub_.publish(arr);
}

void SensorSimulator::publishStaticMarkers()
{
    visualization_msgs::MarkerArray arr;
    const ros::Time now = ros::Time::now();
    int id = 0;

    // static poles
    for (const auto& p : static_poles_) {
        visualization_msgs::Marker m;
        m.header.frame_id = "world";
        m.header.stamp    = now;
        m.ns    = "static_poles";
        m.id    = id++;
        m.action = visualization_msgs::Marker::ADD;
        m.type   = visualization_msgs::Marker::CYLINDER;
        m.pose.position.x = p.cx;
        m.pose.position.y = p.cy;
        m.pose.position.z = p.height * 0.5f;
        m.pose.orientation.w = 1.0;
        m.scale.x = m.scale.y = 2.f * p.r;
        m.scale.z = p.height;
        m.color.r = 0.5f; m.color.g = 0.5f; m.color.b = 0.5f; m.color.a = 1.0f;
        arr.markers.push_back(m);
    }

    // corridor side walls
    for (const auto& w : static_walls_) {
        visualization_msgs::Marker m;
        m.header.frame_id = "world";
        m.header.stamp    = now;
        m.ns    = "static_walls";
        m.id    = id++;
        m.action = visualization_msgs::Marker::ADD;
        m.type   = visualization_msgs::Marker::CUBE;
        m.pose.position.x = w.cx;
        m.pose.position.y = w.cy;
        m.pose.position.z = w.height * 0.5f;
        m.pose.orientation.w = 1.0;
        m.scale.x = 2.f * w.half_len_x;
        m.scale.y = 2.f * w.half_thick_y;
        m.scale.z = w.height;
        m.color.r = 0.8f; m.color.g = 0.6f; m.color.b = 0.2f; m.color.a = 0.5f;
        arr.markers.push_back(m);
    }

    static_marker_pub_.publish(arr);
}

void SensorSimulator::checkCollisions()
{
    if (!collision_check_enabled_) return;
    const float r = drone_radius_;

    bool new_event = false;

    // static poles: full-height vertical cylinders, horizontal distance + z within pole height
    for (size_t i = 0; i < static_poles_.size(); ++i) {
        const auto& p = static_poles_[i];
        const float dx = pos.x() - p.cx, dy = pos.y() - p.cy;
        const bool hit = std::sqrt(dx * dx + dy * dy) < r + p.r &&
                         pos.z() > -r && pos.z() < p.height + r;
        if (hit && !pole_contact_[i]) { ++collision_count_static_; new_event = true; }
        pole_contact_[i] = hit;
    }

    // dynamic balls
    for (size_t i = 0; i < sphere_obstacles_.size(); ++i) {
        const auto& c = sphere_obstacles_[i].center;
        const float dx = pos.x() - c.x, dy = pos.y() - c.y, dz = pos.z() - c.z;
        const bool hit = std::sqrt(dx * dx + dy * dy + dz * dz) < r + sphere_obstacles_[i].radius;
        if (hit && !sphere_contact_[i]) { ++collision_count_dynamic_; new_event = true; }
        sphere_contact_[i] = hit;
    }

    // dynamic cylinders (finite height)
    for (size_t i = 0; i < cylinder_obstacles_.size(); ++i) {
        const auto& c = cylinder_obstacles_[i].center;
        const float dx = pos.x() - c.x, dy = pos.y() - c.y;
        const float h2 = 0.5f * cylinder_obstacles_[i].height;
        const bool hit = std::sqrt(dx * dx + dy * dy) < r + cylinder_obstacles_[i].radius &&
                         std::fabs(pos.z() - c.z) < h2 + r;
        if (hit && !cyl_contact_[i]) { ++collision_count_dynamic_; new_event = true; }
        cyl_contact_[i] = hit;
    }

    if (new_event)
        ROS_WARN("[collision] total=%d (static %d | dynamic %d)",
                 collision_count_static_ + collision_count_dynamic_,
                 collision_count_static_, collision_count_dynamic_);

    publishCollisionMarker();
}

void SensorSimulator::publishCollisionMarker()
{
    const int total = collision_count_static_ + collision_count_dynamic_;
    visualization_msgs::Marker m;
    m.header.frame_id = "world";
    m.header.stamp    = ros::Time::now();
    m.ns   = "collision_counter";
    m.id   = 0;
    m.action = visualization_msgs::Marker::ADD;
    m.type   = visualization_msgs::Marker::TEXT_VIEW_FACING;
    m.pose.position.x = pos.x();
    m.pose.position.y = pos.y();
    m.pose.position.z = pos.z() + 10.0f;   // floats above the drone
    m.pose.orientation.w = 1.0;
    m.scale.z = 1;                       // text height (m)
    m.color.r = total > 0 ? 1.0f : 0.1f;   // white background: red with collisions, dark gray without
    m.color.g = 0.1f;
    m.color.b = 0.1f;
    m.color.a = 1.0f;
    m.text = "Collisions: " + std::to_string(total) +
             " (S" + std::to_string(collision_count_static_) +
             "|D" + std::to_string(collision_count_dynamic_) + ")";
    collision_marker_pub_.publish(m);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "sensor_simulator_node");
    ros::NodeHandle nh;

    SensorSimulator sensor_simulator(nh);
    return 0;
}
