#ifndef M_DETECTOR_FRAME_TRACKER_ROS_PUBLISHER_H
#define M_DETECTOR_FRAME_TRACKER_ROS_PUBLISHER_H

#include <ros/node_handle.h>
#include <ros/publisher.h>
#include <std_msgs/Header.h>
#include <visualization_msgs/MarkerArray.h>

#include <m-detector/FrameTracker.h>

namespace m_detector
{

class FrameTrackerRosPublisher
{
public:
    FrameTrackerRosPublisher(ros::NodeHandle &node, const TrackerConfig &config);

    double publish(const TrackerResult &result);
    double publishEmpty(const std_msgs::Header &header);

private:
    void addMarkers(const TrackSnapshot &track,
                    const std_msgs::Header &header,
                    visualization_msgs::MarkerArray &markers) const;

    TrackerConfig config_;
    ros::Publisher tracks_publisher_;
    ros::Publisher tentative_tracks_publisher_;
    ros::Publisher predicted_tracks_publisher_;
    ros::Publisher markers_publisher_;
};

} // namespace m_detector

#endif
