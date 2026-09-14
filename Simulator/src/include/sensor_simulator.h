#ifndef SENSOR_SIMULATOR_H
#define SENSOR_SIMULATOR_H
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/common/eigen.h>
#include <pcl/octree/octree_search.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include <pcl_ros/point_cloud.h>
#include <cv_bridge/cv_bridge.h>
#include <visualization_msgs/MarkerArray.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>
#include <yaml-cpp/yaml.h>
#include "maps.hpp"

// Simulated dynamic obstacle: height == 0 → sphere, height > 0 → vertical cylinder
struct SimDynObstacle {
    Eigen::Vector3f center;
    Eigen::Vector3f velocity;
    float radius{0.25f};
    float height{0.0f};
};

class SensorSimulator {
public:
    SensorSimulator(ros::NodeHandle &nh) : nh_(nh) {
        YAML::Node config = YAML::LoadFile(CONFIG_FILE_PATH);
        // load camera params
        fx = config["camera"]["fx"].as<float>();
        fy = config["camera"]["fy"].as<float>();
        cx = config["camera"]["cx"].as<float>();
        cy = config["camera"]["cy"].as<float>();
        image_width = config["camera"]["image_width"].as<int>();
        image_height = config["camera"]["image_height"].as<int>();
        max_depth_dist = config["camera"]["max_depth_dist"].as<float>();
        normalize_depth = config["camera"]["normalize_depth"].as<bool>();

        // load lidar params
        vertical_lines = config["lidar"]["vertical_lines"].as<int>();
        vertical_angle_start = config["lidar"]["vertical_angle_start"].as<float>();
        vertical_angle_end = config["lidar"]["vertical_angle_end"].as<float>();
        horizontal_num = config["lidar"]["horizontal_num"].as<int>();
        horizontal_resolution = config["lidar"]["horizontal_resolution"].as<float>();
        max_lidar_dist = config["lidar"]["max_lidar_dist"].as<float>();

        render_lidar = config["render_lidar"].as<bool>();
        render_depth = config["render_depth"].as<bool>();
        float depth_fps = config["depth_fps"].as<float>();
        float lidar_fps = config["lidar_fps"].as<float>();

        std::string ply_file = config["ply_file"].as<std::string>();
        std::string odom_topic = config["odom_topic"].as<std::string>();
        std::string depth_topic = config["depth_topic"].as<std::string>();
        std::string lidar_topic = config["lidar_topic"].as<std::string>();

        // load map params
        float resolution = config["resolution"].as<float>();
        int expand_y_times = config["expand_y_times"].as<int>();
        int expand_x_times = config["expand_x_times"].as<int>();

        bool use_random_map = config["random_map"].as<bool>();
        int seed = config["seed"].as<int>();
        int sizeX = config["x_length"].as<int>();
        int sizeY = config["y_length"].as<int>();
        int sizeZ = config["z_length"].as<int>();
        int type = config["maze_type"].as<int>();
        double scale = 1 / resolution;
        sizeX = sizeX * scale;
        sizeY = sizeY * scale;
        sizeZ = sizeZ * scale;

        cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
        // generate random map
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
            printf("2.Mapping... \n");
        }
        else {
            pcl::PointCloud<pcl::PointXYZ>::Ptr orig_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            printf("1.Reading Point Cloud... \n");
            if (pcl::io::loadPLYFile(ply_file, *orig_cloud) == -1) {
                PCL_ERROR("Couldn't read PLY file \n");
                return;
            }

            printf("2.Processing... \n");
            *cloud = *orig_cloud;
            for (int i = 0; i < expand_x_times; ++i) {
                expand_cloud(cloud, 0);
            }
            for (int i = 0; i < expand_y_times; ++i) {
                expand_cloud(cloud, 1);
            }
        }

        octree = pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::Ptr(
            new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(resolution));
        octree->setInputCloud(cloud);
        octree->addPointsFromInputCloud();

        // init dynamic obstacles (after the octree is built)
        initDynamicObstacles(config);

        image_pub_ = nh_.advertise<sensor_msgs::Image>(depth_topic, 1);
        point_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(lidar_topic, 1);
        odom_sub_ = nh_.subscribe(odom_topic, 1, &SensorSimulator::odomCallback, this, ros::TransportHints().tcpNoDelay());
        timer_depth_ = nh_.createTimer(ros::Duration(1 / depth_fps), &SensorSimulator::timerDepthCallback, this);
        timer_lidar_ = nh_.createTimer(ros::Duration(1 / lidar_fps), &SensorSimulator::timerLidarCallback, this);

        if (sim_dynamic_enabled_) {
            float obs_hz = 100.f;
            if (config["sim_dynamic"] && config["sim_dynamic"]["update_hz"])
                obs_hz = config["sim_dynamic"]["update_hz"].as<float>();
            timer_obstacles_ = nh_.createTimer(ros::Duration(1.0 / obs_hz),
                                               &SensorSimulator::timerObstacleCallback, this);
            marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/sim/dynamic_obstacle_markers", 1, true);
        }

        if (!static_poles_.empty()) {
            static_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
                "/sim/static_obstacle_markers", 1, true);
            publishStaticMarkers();
        }

        printf("3.Simulation Ready! \n");
        ros::spin();
    }

    void odomCallback(const nav_msgs::Odometry::ConstPtr &msg);
    cv::Mat renderDepthImage();
    pcl::PointCloud<pcl::PointXYZINormal> renderLidarPointcloud();
    void timerDepthCallback(const ros::TimerEvent &);
    void timerLidarCallback(const ros::TimerEvent &);
    void timerObstacleCallback(const ros::TimerEvent &);
    void expand_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr expanded_cloud, int direction = 0);
    void publishStaticMarkers();

private:
    void initDynamicObstacles(const YAML::Node& config);
    float findNearestObstacleHit(const Eigen::Vector3f& ro, const Eigen::Vector3f& rd, float max_dist) const;
    void publishObstacleMarkers();

    bool render_depth{false};
    bool render_lidar{false};
    bool odom_init{false};
    Eigen::Quaternionf quat;
    Eigen::Vector3f pos;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::Ptr octree;

    // camera param
    float fx = 80.0f;
    float fy = 80.0f;
    float cx = 80.0f;
    float cy = 45.0f;
    int image_width = 160;
    int image_height = 90;
    float max_depth_dist{20};
    bool normalize_depth{false};

    // lidar param
    int vertical_lines = 16;
    float vertical_angle_start = -15.0;
    float vertical_angle_end = 15.0;
    int horizontal_num = 360;
    float horizontal_resolution = 1.0;
    float max_lidar_dist{50};

    // dynamic obstacles
    bool sim_dynamic_enabled_{false};
    std::vector<SimDynObstacle> dynamic_obstacles_;
    Eigen::Vector3f world_min_, world_max_;
    float dyn_z_lo_{0.f}, dyn_z_hi_{15.f};  // sphere z bounce bounds from config
    ros::Timer timer_obstacles_;
    ros::Publisher marker_pub_;

    // static obstacle markers
    std::vector<mocka::StaticPole> static_poles_;
    ros::Publisher static_marker_pub_;

    ros::NodeHandle nh_;
    ros::Publisher image_pub_, point_cloud_pub_;
    ros::Subscriber odom_sub_;
    ros::Timer timer_depth_, timer_lidar_;
};


#endif //SENSOR_SIMULATOR_H
