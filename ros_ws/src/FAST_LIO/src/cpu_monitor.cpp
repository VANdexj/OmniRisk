#include <iostream>
#include <ros/ros.h>
#include <string>
#include <fast_lio/CPUUsage.h>
#include "Utils/cpu_monitor.hpp"

using std::string;

bool cpu_monitor_en_ = true;
int cpu_count_;
int cpu_monitor_filter_type_;
double cpu_monitor_filter_filter_param1_, cpu_monitor_filter_filter_param2_;
ros::Publisher pubCpuUsage_;
std::unique_ptr<CPUMonitor> cpu_monitor_ptr_;
double cpu_monitor_freq_;

void cpu_usage_cbk(const ros::TimerEvent& event) 
{
    fast_lio::CPUUsage msg;
    std::vector<double> cpu_usage = cpu_monitor_ptr_->get_cpu_usage();

    for (double usage : cpu_usage) {
        msg.cpu_usage.push_back(static_cast<float>(usage));
    }

    pubCpuUsage_.publish(msg);

    // for (size_t i = 0; i < cpu_usage.size(); ++i) {
    //     std::ostringstream oss;
    //     std::cout << "Core " << std::setw(3) << std::right << i << ": " 
    //         << std::fixed << std::setprecision(2) << std::setw(6) << cpu_usage[i] << "%" << std::endl;
    // }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "cpu_monitor");
    ros::NodeHandle nh;


    // [wxx] cpu_monitor
    nh.param<bool>("wxx/cpu_monitor_en", cpu_monitor_en_, false);
    nh.param<int>("wxx/cpu_monitor_filter_type", cpu_monitor_filter_type_, 0);
    nh.param<double>("wxx/cpu_monitor_filter_filter_param1", cpu_monitor_filter_filter_param1_, 0.0);
    nh.param<double>("wxx/cpu_monitor_filter_filter_param2", cpu_monitor_filter_filter_param2_, 0.0);
    nh.param<double>("wxx/cpu_monitor_freq", cpu_monitor_freq_, 10.0);

    ros::Timer cpu_usage_timer;
    if( cpu_monitor_en_ )
    {
        pubCpuUsage_ = nh.advertise<fast_lio::CPUUsage>("/cpu_usage", 100000);

        switch (cpu_monitor_filter_type_) 
        {
            case 0:
                cpu_monitor_ptr_ = std::make_unique<CPUMonitor>(CPUMonitor::FilterType::SMA, cpu_monitor_filter_filter_param1_);
                break;
            case 1:
                cpu_monitor_ptr_ = std::make_unique<CPUMonitor>(CPUMonitor::FilterType::EMA, cpu_monitor_filter_filter_param1_);
                break;
            case 2:
                cpu_monitor_ptr_ = std::make_unique<CPUMonitor>(CPUMonitor::FilterType::LowPass, cpu_monitor_filter_filter_param1_, cpu_monitor_filter_filter_param2_);
                break;
            default:
                std::cout << "\033[1;33m[wxx] cpu_monitor_filter_type can only be [0, 1, 2]." << "\033[0m" << std::endl;
                exit(0);
        }

        cpu_usage_timer = nh.createTimer(ros::Duration(1.0/cpu_monitor_freq_), cpu_usage_cbk);
    }

    std::cout << "\033[1;33m[cpu_monitor] work start!" << "\033[0m" << std::endl;
    
    ros::spin();

    return 0;
}