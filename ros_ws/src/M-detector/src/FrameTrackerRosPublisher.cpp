#include <m-detector/FrameTrackerRosPublisher.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

#include <geometry_msgs/Point.h>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>

#include <m_detector/TrackedObject.h>
#include <m_detector/TrackedObjectArray.h>
#include <m_detector/PredictedTrackedObjectArray.h>

namespace
{

m_detector::TrackedObject ToMessage(const m_detector::TrackSnapshot &track)
{
    m_detector::TrackedObject message;
    message.id = track.id;
    message.track_uid = track.track_uid;
    message.confirmed = track.confirmed;
    message.matched = track.matched;
    message.age = track.age;
    message.total_hits = track.total_hits;
    message.missed_frames = track.missed_frames;
    message.point_count = track.point_count;
    message.last_iou = track.last_iou;
    message.position.x = track.position.x();
    message.position.y = track.position.y();
    message.position.z = track.position.z();
    message.velocity.x = track.velocity.x();
    message.velocity.y = track.velocity.y();
    message.velocity.z = track.velocity.z();
    message.dimensions.x = track.dimensions.x();
    message.dimensions.y = track.dimensions.y();
    message.dimensions.z = track.dimensions.z();
    return message;
}

} // namespace

namespace m_detector
{

FrameTrackerRosPublisher::FrameTrackerRosPublisher(ros::NodeHandle &node,
                                                   const TrackerConfig &config)
    : config_(config),
      tracks_publisher_(node.advertise<TrackedObjectArray>(config.tracks_topic, 1)),
      tentative_tracks_publisher_(
          node.advertise<TrackedObjectArray>(config.tentative_tracks_topic, 1)),
      predicted_tracks_publisher_(
          node.advertise<PredictedTrackedObjectArray>(config.predicted_tracks_topic, 1)),
      markers_publisher_(node.advertise<visualization_msgs::MarkerArray>(config.markers_topic, 1))
{
}

double FrameTrackerRosPublisher::publish(const TrackerResult &result)
{
    const ros::WallTime publish_start = ros::WallTime::now();
    TrackedObjectArray track_array;
    track_array.header = result.header;
    for (const auto &track : result.tracks)
    {
        track_array.tracks.push_back(ToMessage(track));
    }

    TrackedObjectArray tentative_array;
    tentative_array.header = result.header;
    for (const auto &track : result.tentative_tracks)
    {
        tentative_array.tracks.push_back(ToMessage(track));
    }

    ros::Time requested_prediction_stamp = ros::Time::now();
    if (requested_prediction_stamp.isZero())
    {
        requested_prediction_stamp = result.header.stamp;
    }
    const Eigen::Vector3f world_acceleration = config_.ballistic_model_en
        ? Eigen::Vector3f(static_cast<float>(config_.gravity_world_x),
                          static_cast<float>(config_.gravity_world_y),
                          static_cast<float>(config_.gravity_world_z))
        : Eigen::Vector3f::Zero();
    const PredictedTrackerResult prediction = PredictTrackerResult(
        result, requested_prediction_stamp, config_.max_prediction_horizon, world_acceleration);
    PredictedTrackedObjectArray predicted_array;
    predicted_array.header = prediction.header;
    predicted_array.measurement_stamp = prediction.measurement_stamp;
    predicted_array.prediction_horizon = ros::Duration(prediction.prediction_horizon);
    predicted_array.stale = prediction.stale;
    for (const auto &track : prediction.tracks)
    {
        predicted_array.tracks.push_back(ToMessage(track));
    }

    visualization_msgs::MarkerArray marker_array;
    visualization_msgs::Marker clear_marker;
    clear_marker.header = prediction.header;
    clear_marker.action = visualization_msgs::Marker::DELETEALL;
    marker_array.markers.push_back(clear_marker);

    for (const auto &track : prediction.tracks)
    {
        addMarkers(track, prediction.header, marker_array);
    }

    tracks_publisher_.publish(track_array);
    tentative_tracks_publisher_.publish(tentative_array);
    predicted_tracks_publisher_.publish(predicted_array);
    markers_publisher_.publish(marker_array);
    return (ros::WallTime::now() - publish_start).toSec() * 1000.0;
}

double FrameTrackerRosPublisher::publishEmpty(const std_msgs::Header &header)
{
    TrackerResult empty;
    empty.header = header;
    return publish(empty);
}

void FrameTrackerRosPublisher::addMarkers(
    const TrackSnapshot &track,
    const std_msgs::Header &header,
    visualization_msgs::MarkerArray &marker_array) const
{
    const Eigen::Vector3f visual_dims =
        (track.dimensions * static_cast<float>(config_.marker_bbox_scale))
            .cwiseMax(Eigen::Vector3f::Constant(static_cast<float>(config_.marker_bbox_min_size)));
    const int marker_base_id = static_cast<int>(track.id) * 10;

    visualization_msgs::Marker bbox_marker;
    bbox_marker.header = header;
    bbox_marker.ns = "confirmed_bbox";
    bbox_marker.id = marker_base_id;
    bbox_marker.type = visualization_msgs::Marker::CUBE;
    bbox_marker.action = visualization_msgs::Marker::ADD;
    bbox_marker.pose.orientation.w = 1.0;
    bbox_marker.pose.position.x = track.position.x();
    bbox_marker.pose.position.y = track.position.y();
    bbox_marker.pose.position.z = track.position.z();
    bbox_marker.scale.x = visual_dims.x();
    bbox_marker.scale.y = visual_dims.y();
    bbox_marker.scale.z = visual_dims.z();
    bbox_marker.color.r = track.matched ? 0.1f : 1.0f;
    bbox_marker.color.g = 1.0f;
    bbox_marker.color.b = 0.1f;
    bbox_marker.color.a = 0.35f;
    marker_array.markers.push_back(bbox_marker);

    visualization_msgs::Marker text_marker;
    text_marker.header = header;
    text_marker.ns = "confirmed_id";
    text_marker.id = marker_base_id + 1;
    text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::Marker::ADD;
    text_marker.pose.orientation.w = 1.0;
    text_marker.pose.position.x = track.position.x();
    text_marker.pose.position.y = track.position.y();
    text_marker.pose.position.z = track.position.z() + visual_dims.z() * 0.6f + 0.3f;
    text_marker.scale.z = 0.4;
    text_marker.color.r = 1.0f;
    text_marker.color.g = 1.0f;
    text_marker.color.b = 1.0f;
    text_marker.color.a = 0.95f;
    text_marker.text = "ID:" + std::to_string(track.id);
    marker_array.markers.push_back(text_marker);

    visualization_msgs::Marker path_marker;
    path_marker.header = header;
    path_marker.ns = "confirmed_path";
    path_marker.id = marker_base_id + 2;
    path_marker.type = visualization_msgs::Marker::LINE_STRIP;
    path_marker.action = visualization_msgs::Marker::ADD;
    path_marker.pose.orientation.w = 1.0;
    path_marker.scale.x = 0.08;
    path_marker.color.r = 0.1f;
    path_marker.color.g = 0.7f;
    path_marker.color.b = 1.0f;
    path_marker.color.a = 0.95f;
    for (const auto &history_point : track.history)
    {
        geometry_msgs::Point point;
        point.x = history_point.x();
        point.y = history_point.y();
        point.z = history_point.z();
        path_marker.points.push_back(point);
    }
    marker_array.markers.push_back(path_marker);

    const float speed = track.velocity.norm();
    const float display_speed = std::isfinite(speed) ? speed : 0.0f;
    if (display_speed >= static_cast<float>(config_.marker_velocity_min_speed))
    {
        const float arrow_length =
            std::min(display_speed * static_cast<float>(config_.marker_velocity_arrow_scale),
                     static_cast<float>(config_.marker_velocity_max_arrow_length));
        if (arrow_length > 0.0f)
        {
            const Eigen::Vector3f direction = track.velocity / display_speed;
            geometry_msgs::Point arrow_start;
            arrow_start.x = track.position.x();
            arrow_start.y = track.position.y();
            arrow_start.z = track.position.z();
            geometry_msgs::Point arrow_end;
            arrow_end.x = track.position.x() + direction.x() * arrow_length;
            arrow_end.y = track.position.y() + direction.y() * arrow_length;
            arrow_end.z = track.position.z() + direction.z() * arrow_length;

            visualization_msgs::Marker velocity_arrow;
            velocity_arrow.header = header;
            velocity_arrow.ns = "confirmed_velocity_arrow";
            velocity_arrow.id = marker_base_id + 3;
            velocity_arrow.type = visualization_msgs::Marker::ARROW;
            velocity_arrow.action = visualization_msgs::Marker::ADD;
            velocity_arrow.pose.orientation.w = 1.0;
            velocity_arrow.scale.x = 0.05;
            velocity_arrow.scale.y = 0.14;
            velocity_arrow.scale.z = 0.20;
            velocity_arrow.color.r = 1.0f;
            velocity_arrow.color.g = 0.55f;
            velocity_arrow.color.b = 0.0f;
            velocity_arrow.color.a = 0.95f;
            velocity_arrow.points.push_back(arrow_start);
            velocity_arrow.points.push_back(arrow_end);
            marker_array.markers.push_back(velocity_arrow);
        }
    }

    std::ostringstream speed_text;
    speed_text << "v: " << std::fixed << std::setprecision(2) << display_speed << " m/s";

    visualization_msgs::Marker velocity_text;
    velocity_text.header = header;
    velocity_text.ns = "confirmed_velocity_text";
    velocity_text.id = marker_base_id + 4;
    velocity_text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    velocity_text.action = visualization_msgs::Marker::ADD;
    velocity_text.pose.orientation.w = 1.0;
    velocity_text.pose.position.x = track.position.x();
    velocity_text.pose.position.y = track.position.y();
    velocity_text.pose.position.z = track.position.z() + visual_dims.z() * 0.6f + 0.75f;
    velocity_text.scale.z = config_.marker_velocity_text_size;
    velocity_text.color.r = 1.0f;
    velocity_text.color.g = 0.75f;
    velocity_text.color.b = 0.1f;
    velocity_text.color.a = 0.95f;
    velocity_text.text = speed_text.str();
    marker_array.markers.push_back(velocity_text);
}

} // namespace m_detector
