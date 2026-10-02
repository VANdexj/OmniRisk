#include <iostream>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <string>

using std::string;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "initial_map_publisher");
    ros::NodeHandle nh;

    bool initial_map_from_pcd_;
    string initial_map_pcd_name_;
    nh.param<bool>("wxx/initial_map_from_pcd", initial_map_from_pcd_, true);
    nh.param<string>("wxx/initial_map_pcd_name", initial_map_pcd_name_, "initial_map.pcd");
    ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>("initial_map", 1);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    sensor_msgs::PointCloud2::Ptr msg(new sensor_msgs::PointCloud2);

    if( initial_map_from_pcd_ )
    {
        pcl::PCDReader reader;
        string file_name_map_kdtree = initial_map_pcd_name_;
        string all_points_dir_map_kdtree(string(string(ROOT_DIR) + "PCD/") + file_name_map_kdtree);
        if( reader.read(all_points_dir_map_kdtree, *cloud) == -1 )
        {
            std::cout << "\033[1;31m[wxx] The initial pcd file [" << initial_map_pcd_name_ << "] does not exist!!!" << "\033[0m" << std::endl;
            exit(0);
        }

        pcl::toROSMsg(*cloud, *msg);
        msg->header.frame_id = "world";  
    }

    std::cout << 111 << std::endl;

    ros::Rate rate(1); 
    while (ros::ok())
    {
        msg->header.stamp = ros::Time::now();

        if( initial_map_from_pcd_ )
            pub.publish(msg);

        rate.sleep();
    }

    return 0;
}