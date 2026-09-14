#include <ros/ros.h>
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <iostream>
#include <exception>
#include <csignal>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <Python.h>
#include <boost/bind/bind.hpp>
#include <ros/ros.h>
#include <Eigen/Core>
#include <types.h>
#include <m-detector/DynObjFilter.h>
#include <m-detector/FrameTracker.h>
#include <m-detector/FrameTrackerRosPublisher.h>
#include <m-detector/BoundedFrameQueue.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <pcl/filters/random_sample.h>
#include <Eigen/Eigen>
#include <eigen_conversions/eigen_msg.h>
#include <timer.h>

// #include "preprocess.h"

using namespace std;

shared_ptr<DynObjFilter> DynObjFilt(new DynObjFilter());
std::unique_ptr<m_detector::FrameTracker> frame_tracker;
std::unique_ptr<m_detector::FrameTrackerRosPublisher> frame_tracker_publisher;

int occlude_windows = 3;
int point_index = 0;
float VER_RESOLUTION_MAX = 0.01;
float HOR_RESOLUTION_MAX = 0.01;
bool timing_enabled = false;
string points_topic, odom_topic;
string frame_id = "camera_init";
string odom_visual_frame_id = "";
string odom_visual_child_frame_id = "";
bool odom_visual_publish_tf = false;
float odom_visual_lidar_offset_x = 0.0f;
float odom_visual_lidar_offset_y = 0.0f;
float odom_visual_lidar_offset_z = 0.0f;
string out_folder, out_folder_origin;
int dataset = 0;
int cur_frame = 0;
double total_compute_time_ms = 0.0;
size_t processed_frame_count = 0;
int input_queue_size = 5;
int sync_queue_size = 5;
int pending_queue_size = 5;
double sync_max_interval = 0.01;

ros::Publisher pub_pcl_dyn, pub_pcl_dyn_extend, pub_pcl_std, pub_odom;
std::unique_ptr<tf::TransformBroadcaster> odom_tf_broadcaster;

namespace {
struct SynchronizedInput
{
    sensor_msgs::PointCloud2ConstPtr cloud;
    nav_msgs::OdometryConstPtr odom;
    ros::WallTime queued_at;
};

std::unique_ptr<m_detector::BoundedFrameQueue<SynchronizedInput>> pending_input_queue;

double MeasurementAgeMs(const ros::Time &stamp)
{
    const ros::Time now = ros::Time::now();
    if (stamp.isZero() || now.isZero())
    {
        return -1.0;
    }
    return std::max(0.0, (now - stamp).toSec() * 1000.0);
}

const sensor_msgs::PointField *FindField(const sensor_msgs::PointCloud2 &msg, const std::string &name) {
    for (const auto &field : msg.fields) {
        if (field.name == name) {
            return &field;
        }
    }
    return nullptr;
}

float ReadNumericFieldAsFloat(const sensor_msgs::PointCloud2 &msg,
                              const sensor_msgs::PointField &field,
                              const size_t point_index) {
    const uint8_t *data_ptr = &msg.data[point_index * msg.point_step + field.offset];
    switch (field.datatype) {
    case sensor_msgs::PointField::INT8:
        return static_cast<float>(*reinterpret_cast<const int8_t *>(data_ptr));
    case sensor_msgs::PointField::UINT8:
        return static_cast<float>(*reinterpret_cast<const uint8_t *>(data_ptr));
    case sensor_msgs::PointField::INT16:
        return static_cast<float>(*reinterpret_cast<const int16_t *>(data_ptr));
    case sensor_msgs::PointField::UINT16:
        return static_cast<float>(*reinterpret_cast<const uint16_t *>(data_ptr));
    case sensor_msgs::PointField::INT32:
        return static_cast<float>(*reinterpret_cast<const int32_t *>(data_ptr));
    case sensor_msgs::PointField::UINT32:
        return static_cast<float>(*reinterpret_cast<const uint32_t *>(data_ptr));
    case sensor_msgs::PointField::FLOAT32:
        return *reinterpret_cast<const float *>(data_ptr);
    case sensor_msgs::PointField::FLOAT64:
        return static_cast<float>(*reinterpret_cast<const double *>(data_ptr));
    default:
        return 0.0f;
    }
}

void ConvertPointCloud2ToXYZINormal(const sensor_msgs::PointCloud2 &msg, PointCloudXYZI &cloud_out) {
    const auto point_count = static_cast<size_t>(msg.width) * static_cast<size_t>(msg.height);
    cloud_out.clear();
    cloud_out.reserve(point_count);

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(msg, "z");

    const sensor_msgs::PointField *intensity_field = FindField(msg, "intensity");
    const sensor_msgs::PointField *curvature_field = FindField(msg, "curvature");
    const sensor_msgs::PointField *normal_x_field = FindField(msg, "normal_x");
    const sensor_msgs::PointField *normal_y_field = FindField(msg, "normal_y");
    const sensor_msgs::PointField *normal_z_field = FindField(msg, "normal_z");
    const sensor_msgs::PointField *timestamp_field = FindField(msg, "timestamp");
    const sensor_msgs::PointField *row_pos_field = FindField(msg, "row_pos");
    const sensor_msgs::PointField *col_pos_field = FindField(msg, "col_pos");

    for (size_t i = 0; i < point_count; ++i, ++iter_x, ++iter_y, ++iter_z) {
        PointType point;
        point.x = *iter_x;
        point.y = *iter_y;
        point.z = *iter_z;
        point.intensity = intensity_field ? ReadNumericFieldAsFloat(msg, *intensity_field, i) : 0.0f;
        point.curvature = curvature_field ? ReadNumericFieldAsFloat(msg, *curvature_field, i) : 0.0f;
        point.normal_x = normal_x_field ? ReadNumericFieldAsFloat(msg, *normal_x_field, i) : 0.0f;
        point.normal_y = normal_y_field ? ReadNumericFieldAsFloat(msg, *normal_y_field, i) : 0.0f;
        point.normal_z = normal_z_field ? ReadNumericFieldAsFloat(msg, *normal_z_field, i) : 0.0f;

        // Reuse any available per-point metadata when the legacy fields are absent.
        if (!curvature_field && timestamp_field) {
            point.curvature = ReadNumericFieldAsFloat(msg, *timestamp_field, i);
        }
        if (!normal_x_field && row_pos_field) {
            point.normal_x = ReadNumericFieldAsFloat(msg, *row_pos_field, i);
        }
        if (!normal_y_field && col_pos_field) {
            point.normal_y = ReadNumericFieldAsFloat(msg, *col_pos_field, i);
        }

        cloud_out.push_back(point);
    }

    cloud_out.width = static_cast<uint32_t>(cloud_out.size());
    cloud_out.height = 1;
    cloud_out.is_dense = msg.is_dense;
}
} // namespace

M3D RotationFromOdom(const nav_msgs::Odometry &odom) {
    Eigen::Quaterniond cur_q;
    tf::quaternionMsgToEigen(odom.pose.pose.orientation, cur_q);
    return cur_q.matrix();
}

V3D PositionFromOdom(const nav_msgs::Odometry &odom) {
    return V3D(odom.pose.pose.position.x,
               odom.pose.pose.position.y,
               odom.pose.pose.position.z);
}

void PublishVisualizationOdom(const nav_msgs::OdometryConstPtr &odom_msg) {
    const nav_msgs::Odometry &cur_odom = *odom_msg;
    const M3D odom_rotation = RotationFromOdom(cur_odom);

    nav_msgs::Odometry vis_odom = cur_odom;
    const std::string &vis_frame_id = odom_visual_frame_id.empty() ? cur_odom.header.frame_id : odom_visual_frame_id;
    const std::string &vis_child_frame_id = odom_visual_child_frame_id.empty()
                                                ? (cur_odom.child_frame_id.empty() ? std::string("base_link") : cur_odom.child_frame_id)
                                                : odom_visual_child_frame_id;
    if (!vis_frame_id.empty()) {
        vis_odom.header.frame_id = vis_frame_id;
    }
    vis_odom.child_frame_id = vis_child_frame_id;

    // Apply optional LiDAR extrinsic offset so the visualized pose represents the LiDAR center
    // instead of the IMU/body center.
    if (odom_visual_lidar_offset_x != 0.0f || odom_visual_lidar_offset_y != 0.0f || odom_visual_lidar_offset_z != 0.0f) {
        Eigen::Vector3d t_body(odom_visual_lidar_offset_x, odom_visual_lidar_offset_y, odom_visual_lidar_offset_z);
        Eigen::Vector3d t_world = odom_rotation * t_body;
        vis_odom.pose.pose.position.x += t_world(0);
        vis_odom.pose.pose.position.y += t_world(1);
        vis_odom.pose.pose.position.z += t_world(2);
    }

    pub_odom.publish(vis_odom);

    if (odom_visual_publish_tf && odom_tf_broadcaster && !vis_frame_id.empty() && !vis_child_frame_id.empty()) {
        tf::Transform transform;
        tf::Quaternion q;
        tf::quaternionMsgToTF(vis_odom.pose.pose.orientation, q);
        transform.setRotation(q);
        transform.setOrigin(tf::Vector3(vis_odom.pose.pose.position.x,
                                        vis_odom.pose.pose.position.y,
                                        vis_odom.pose.pose.position.z));
        odom_tf_broadcaster->sendTransform(tf::StampedTransform(transform, cur_odom.header.stamp,
                                                                 vis_frame_id, vis_child_frame_id));
    }
}

void SynchronizedInputCallback(const sensor_msgs::PointCloud2ConstPtr &cloud,
                               const nav_msgs::OdometryConstPtr &odom) {
    if (cloud->header.stamp.isZero() || odom->header.stamp.isZero()) {
        ROS_WARN_THROTTLE(1.0, "Dropping synchronized M-detector input with a zero timestamp.");
        return;
    }
    const double stamp_delta = std::fabs((cloud->header.stamp - odom->header.stamp).toSec());
    if (stamp_delta > sync_max_interval + 1e-6) {
        ROS_WARN_STREAM_THROTTLE(1.0,
                                 "Dropping M-detector input pair with stamp delta "
                                     << std::fixed << std::setprecision(3)
                                     << stamp_delta * 1000.0 << " ms.");
        return;
    }

    SynchronizedInput input{cloud, odom, ros::WallTime::now()};
    const auto result = pending_input_queue->push(std::move(input), cloud->header.stamp);
    if (result == m_detector::BoundedFrameQueuePushResult::DROPPED_OLDEST) {
        ROS_WARN_STREAM_THROTTLE(1.0,
                                 "M-detector pending queue is full; dropped the oldest frame. "
                                     "total_dropped=" << pending_input_queue->droppedOldestCount()
                                     << " capacity=" << pending_queue_size);
    } else if (result == m_detector::BoundedFrameQueuePushResult::REJECTED_OUT_OF_ORDER) {
        ROS_WARN_STREAM_THROTTLE(1.0,
                                 "M-detector rejected out-of-order synchronized input at "
                                     << cloud->header.stamp.toSec());
    } else if (result == m_detector::BoundedFrameQueuePushResult::ACCEPTED_AFTER_RESET) {
        ROS_WARN_STREAM("M-detector input timestamp jumped backward to "
                             << cloud->header.stamp.toSec()
                             << "; resynchronized pending queue ordering.");
    }
}

void ProcessIntegratedTracker(const std::optional<DetectorFrame> &detector_frame,
                              const PointCloudXYZI::ConstPtr &support_cloud,
                              const V3D &odom_position,
                              double cur_time,
                              double odom_time,
                              const std_msgs::Header &cloud_header) {
    CsvTimingRecorder &timing = CsvTimingRecorder::instance();
    timing.setMetric("tracker_enabled", frame_tracker ? 1.0 : 0.0);
    timing.setMetric(
        "tracker_preclustered",
        detector_frame && detector_frame->preclustered_detections
            ? 1.0
            : 0.0);
    if (!frame_tracker || !frame_tracker_publisher) {
        return;
    }

    const double stamp_delta_ms =
        cloud_header.stamp.isZero()
            ? 0.0
            : std::fabs(cloud_header.stamp.toSec() - odom_time) * 1000.0;
    timing.setMetric("cloud_odom_stamp_delta_ms", stamp_delta_ms);
    if (stamp_delta_ms > 50.0) {
        ROS_WARN_STREAM_THROTTLE(2.0,
                                 "Point cloud and synchronized odometry stamps differ by "
                                     << std::fixed << std::setprecision(3)
                                     << stamp_delta_ms << " ms.");
    }
    // use the raw cloud stamp: a fromSec(double) round trip loses tens of ns and breaks downstream exact ns matching
    const ros::Time measurement_stamp =
        cloud_header.stamp.isZero() ? ros::Time().fromSec(cur_time) : cloud_header.stamp;

    if (!detector_frame) {
        std_msgs::Header header;
        header.stamp = measurement_stamp;
        header.frame_id = frame_id;
        const double publish_ms = frame_tracker_publisher->publishEmpty(header);
        timing.setMetric("tracker_frame_failed", 1.0);
        timing.setMetric("tracker_publish", publish_ms);
        ROS_ERROR_STREAM_THROTTLE(1.0,
                                  "Integrated tracker skipped a failed detector frame at "
                                      << std::fixed << std::setprecision(6) << cur_time);
        return;
    }

    m_detector::TrackerFrame tracker_frame;
    tracker_frame.header = detector_frame->header;
    tracker_frame.header.stamp = measurement_stamp;
    tracker_frame.frame_out = detector_frame->frame_out;
    tracker_frame.point_out = detector_frame->point_out;
    tracker_frame.support_cloud = support_cloud;
    tracker_frame.preclustered_detections = detector_frame->preclustered_detections;
    tracker_frame.odom_position = odom_position.cast<float>();
    tracker_frame.has_odom = true;
    tracker_frame.frame_out_available = detector_frame->frame_out_available;
    tracker_frame.cloud_odom_stamp_delta_ms = stamp_delta_ms;
    timing.setMetric(
        "measurement_age_at_tracker_ms",
        MeasurementAgeMs(tracker_frame.header.stamp));

    if (!tracker_frame.preclustered_detections &&
        !tracker_frame.frame_out_available) {
        ROS_WARN_STREAM_THROTTLE(2.0,
                                 "Integrated tracker is running in point_out-only degraded mode.");
    }

    try {
        m_detector::TrackerResult tracker_result = frame_tracker->update(tracker_frame);
        const double publish_ms = frame_tracker_publisher->publish(tracker_result);
        timing.setMetric("tracker_mode", static_cast<double>(tracker_result.input_mode));
        timing.setMetric("tracker_input_points", static_cast<double>(tracker_result.primary_point_count));
        timing.setMetric("tracker_point_out_points", static_cast<double>(tracker_result.point_out_point_count));
        timing.setMetric("tracker_support_points", static_cast<double>(tracker_result.support_point_count));
        timing.setMetric("tracker_detection_count", static_cast<double>(tracker_result.detection_count));
        timing.setMetric("tracker_track_count", static_cast<double>(tracker_result.active_track_count));
        timing.setMetric("tracker_confirmed_count", static_cast<double>(tracker_result.tracks.size()));
        timing.setMetric("tracker_update", tracker_result.update_ms);
        timing.setMetric("tracker_publish", publish_ms);
        timing.setMetric("tracker_frame_failed", 0.0);
    } catch (const std::exception &error) {
        const double publish_ms = frame_tracker_publisher->publishEmpty(tracker_frame.header);
        timing.setMetric("tracker_publish", publish_ms);
        timing.setMetric("tracker_frame_failed", 1.0);
        ROS_ERROR_STREAM_THROTTLE(1.0, "Integrated tracker frame failed: " << error.what());
    } catch (...) {
        const double publish_ms = frame_tracker_publisher->publishEmpty(tracker_frame.header);
        timing.setMetric("tracker_publish", publish_ms);
        timing.setMetric("tracker_frame_failed", 1.0);
        ROS_ERROR_THROTTLE(1.0, "Integrated tracker frame failed with an unknown exception.");
    }
}

void ProcessSynchronizedInput(const SynchronizedInput &input) {
    const double queue_wait_ms =
        (ros::WallTime::now() - input.queued_at).toSec() * 1000.0;
    const double measurement_age_at_start_ms =
        MeasurementAgeMs(input.cloud->header.stamp);
    PointCloudXYZI::Ptr cur_pc(new PointCloudXYZI());
    ConvertPointCloud2ToXYZINormal(*input.cloud, *cur_pc);
    const std_msgs::Header cloud_header = input.cloud->header;
    const M3D cur_rot = RotationFromOdom(*input.odom);
    const V3D cur_pos = PositionFromOdom(*input.odom);
    const double cur_time = cloud_header.stamp.toSec();
    const double odom_time = input.odom->header.stamp.toSec();
    const double compute_start_time = omp_get_wtime();
    CsvTimingRecorder::instance().beginFrame(cur_frame, cur_time, cur_pc->size());
    CsvTimingRecorder::instance().setMetric("queue_wait_ms", queue_wait_ms);
    CsvTimingRecorder::instance().setMetric(
        "measurement_age_at_start_ms",
        measurement_age_at_start_ms);
    ScopedTiming frame_total_timer("frame_total");
    string file_name = out_folder;
    stringstream ss;
    ss << setw(6) << setfill('0') << cur_frame;
    file_name += ss.str();
    file_name.append(".label");
    string file_name_origin = out_folder_origin;
    stringstream sss;
    sss << setw(6) << setfill('0') << cur_frame;
    file_name_origin += sss.str();
    file_name_origin.append(".label");

    if (!out_folder.empty() || !out_folder_origin.empty()) {
        DynObjFilt->set_path(
            out_folder.empty() ? string() : file_name,
            out_folder_origin.empty() ? string() : file_name_origin);
    }
    std::optional<DetectorFrame> detector_frame;
    bool tracker_processed_from_fast_frame = false;
    FastDetectorFrameCallback fast_frame_callback;
    if (frame_tracker && frame_tracker_publisher) {
        fast_frame_callback = [&](const DetectorFrame &fast_frame) {
            ProcessIntegratedTracker(
                std::optional<DetectorFrame>(fast_frame),
                cur_pc,
                cur_pos,
                cur_time,
                odom_time,
                cloud_header);
            // The callback delivery is independent of any later failure
            // while rebuilding the full detector output clouds.
            tracker_processed_from_fast_frame = true;
        };
    }
    {
        ScopedTiming filter_timer("filter_total");
        detector_frame = DynObjFilt->filter(
            cur_pc,
            cur_rot,
            cur_pos,
            cur_time,
            fast_frame_callback);
    }
    if (!tracker_processed_from_fast_frame) {
        ProcessIntegratedTracker(detector_frame, cur_pc, cur_pos, cur_time, odom_time, cloud_header);
    }
    if (detector_frame) {
        ScopedTiming publish_timer("publish_total");
        DynObjFilt->publish_dyn(
            pub_pcl_dyn, pub_pcl_dyn_extend, pub_pcl_std,
            cur_time, cloud_header.stamp);
    }
    frame_total_timer.stop();
    CsvTimingRecorder::instance().setMetric(
        "measurement_age_at_frame_end_ms",
        MeasurementAgeMs(cloud_header.stamp));
    CsvTimingRecorder::instance().endFrame();
    const double current_compute_time_ms = (omp_get_wtime() - compute_start_time) * 1000.0;
    total_compute_time_ms += current_compute_time_ms;
    processed_frame_count += 1;
    const double average_compute_time_ms =
        total_compute_time_ms / static_cast<double>(processed_frame_count);
    ROS_INFO_STREAM("Processed dynfilter frame " << cur_frame
                                                 << ", input points: " << cur_pc->size()
                                                 << ", current compute: " << std::fixed << std::setprecision(3)
                                                 << current_compute_time_ms << " ms"
                                                 << ", average compute: " << average_compute_time_ms << " ms");
    cur_frame++;
}

void ProcessingWorker() {
    while (const auto entry = pending_input_queue->waitAndTake()) {
        try {
            ProcessSynchronizedInput(entry->value);
        } catch (const std::exception &error) {
            ROS_ERROR_STREAM("M-detector worker dropped frame " << std::fixed
                             << std::setprecision(6) << entry->stamp.toSec()
                             << ": " << error.what());
        } catch (...) {
            ROS_ERROR_STREAM("M-detector worker dropped frame " << std::fixed
                             << std::setprecision(6) << entry->stamp.toSec()
                             << " after an unknown exception.");
        }
    }
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "dynfilter_odom");
    ros::NodeHandle nh;
    nh.param<string>("dyn_obj/points_topic", points_topic, "");
    nh.param<string>("dyn_obj/odom_topic", odom_topic, "");
    nh.param<string>("dyn_obj/frame_id", frame_id, "camera_init");
    nh.param<string>("dyn_obj/odom_visual_frame_id", odom_visual_frame_id, "");
    nh.param<string>("dyn_obj/odom_visual_child_frame_id", odom_visual_child_frame_id, "");
    nh.param<bool>("dyn_obj/odom_visual_publish_tf", odom_visual_publish_tf, false);
    nh.param<float>("dyn_obj/odom_visual_lidar_offset_x", odom_visual_lidar_offset_x, 0.0f);
    nh.param<float>("dyn_obj/odom_visual_lidar_offset_y", odom_visual_lidar_offset_y, 0.0f);
    nh.param<float>("dyn_obj/odom_visual_lidar_offset_z", odom_visual_lidar_offset_z, 0.0f);
    nh.param<int>("dyn_obj/input_queue_size", input_queue_size, 5);
    nh.param<int>("dyn_obj/sync_queue_size", sync_queue_size, 5);
    nh.param<int>("dyn_obj/pending_queue_size", pending_queue_size, 5);
    nh.param<double>("dyn_obj/sync_max_interval", sync_max_interval, 0.01);
    input_queue_size = std::max(1, input_queue_size);
    sync_queue_size = std::max(1, sync_queue_size);
    pending_queue_size = std::max(1, pending_queue_size);
    sync_max_interval = std::max(0.0, sync_max_interval);
    pending_input_queue.reset(
        new m_detector::BoundedFrameQueue<SynchronizedInput>(
            static_cast<std::size_t>(pending_queue_size)));
    nh.param<string>("dyn_obj/out_file", out_folder, "");
    nh.param<string>("dyn_obj/out_file_origin", out_folder_origin, "");
    string timing_csv_file;
    nh.param<bool>("dyn_obj/timing_enabled", timing_enabled, false);
    nh.param<string>("dyn_obj/timing_csv_file", timing_csv_file, "");
    CsvTimingRecorder::instance().configure(timing_csv_file,
                                            timing_enabled);
    ROS_INFO_STREAM("dynfilter subscribing points_topic=" << points_topic << " odom_topic=" << odom_topic);
    ROS_INFO_STREAM("odometry visualization: frame_id=" << (odom_visual_frame_id.empty() ? std::string("<use original>") : odom_visual_frame_id)
                    << " child_frame_id=" << (odom_visual_child_frame_id.empty() ? std::string("<use original/base_link>") : odom_visual_child_frame_id)
                    << " publish_tf=" << odom_visual_publish_tf
                    << " lidar_offset=[" << odom_visual_lidar_offset_x << "," << odom_visual_lidar_offset_y << "," << odom_visual_lidar_offset_z << "]");

    DynObjFilt->init(nh);

    m_detector::TrackerConfig tracker_config;
    try {
        tracker_config = m_detector::LoadTrackerConfig(nh);
    } catch (const std::invalid_argument &error) {
        ROS_FATAL_STREAM(error.what());
        return 1;
    }
    if (tracker_config.enabled) {
        const DynObjFilter &selected_filter = *DynObjFilt;
        if (!selected_filter.points_in_world_frame || frame_id != "world") {
            // sim feeds body-frame clouds; the detector transforms them with odom, frame_out/point_out are always world frame, tracker still works
            ROS_WARN_STREAM("Integrated tracker running with non-world-frame input: points_in_world_frame="
                            << selected_filter.points_in_world_frame
                            << " frame_id=" << frame_id);
        }
        frame_tracker.reset(new m_detector::FrameTracker(tracker_config));
        frame_tracker_publisher.reset(new m_detector::FrameTrackerRosPublisher(nh, tracker_config));
        ROS_INFO_STREAM("Integrated tracker enabled: tracks_topic=" << tracker_config.tracks_topic
                        << " markers_topic=" << tracker_config.markers_topic);
    } else {
        ROS_INFO("Integrated tracker disabled by dyn_obj/tracker/enabled.");
    }
    /*** ROS subscribe and publisher initialization ***/
    pub_pcl_dyn_extend = nh.advertise<sensor_msgs::PointCloud2>("/m_detector/frame_out", 1);
    pub_pcl_dyn = nh.advertise<sensor_msgs::PointCloud2>("/m_detector/point_out", 1);
    pub_pcl_std = nh.advertise<sensor_msgs::PointCloud2>("/m_detector/std_points", 1);
    pub_odom = nh.advertise<nav_msgs::Odometry>("/m_detector/odometry", 1);
    if (odom_visual_publish_tf) {
        odom_tf_broadcaster.reset(new tf::TransformBroadcaster());
    }
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_pcl(
        nh, points_topic, static_cast<uint32_t>(input_queue_size));
    message_filters::Subscriber<nav_msgs::Odometry> sub_odom(
        nh, odom_topic, static_cast<uint32_t>(input_queue_size));
    using InputSyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::PointCloud2, nav_msgs::Odometry>;
    message_filters::Synchronizer<InputSyncPolicy> input_sync(
        InputSyncPolicy(static_cast<uint32_t>(sync_queue_size)), sub_pcl, sub_odom);
    input_sync.getPolicy()->setMaxIntervalDuration(ros::Duration(sync_max_interval));
    sub_odom.registerCallback(boost::bind(
        &PublishVisualizationOdom, boost::placeholders::_1));
    input_sync.registerCallback(boost::bind(
        &SynchronizedInputCallback,
        boost::placeholders::_1,
        boost::placeholders::_2));
    ROS_INFO_STREAM("dynfilter subscribed with bounded pending synchronization: input_queue="
                    << input_queue_size << " sync_queue=" << sync_queue_size
                    << " pending_queue=" << pending_queue_size
                    << " max_interval_ms=" << sync_max_interval * 1000.0);
    std::thread processing_thread(ProcessingWorker);

    ros::spin();
    pending_input_queue->close();
    processing_thread.join();
    sub_pcl.unsubscribe();
    sub_odom.unsubscribe();
    DynObjFilt.reset();
    return 0;
}
