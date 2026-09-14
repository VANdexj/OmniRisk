#include <gtest/gtest.h>

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>

#include <m-detector/FrameTracker.h>

namespace
{

m_detector::TrackerConfig TestConfig()
{
    m_detector::TrackerConfig config;
    config.cluster_tolerance = 0.3;
    config.cluster_min_points = 2;
    config.min_init_hits = 1;
    config.init_window_size = 1;
    config.near_range_fast_confirm_hits = 1;
    config.track_lifetime_frames = 2;
    config.max_prediction_horizon = 0.2;
    config.lost_track_prediction_time = 0.2;
    config.bbox_size_prior_en = false;
    config.target_distance_filter_en = false;
    config.target_origin_bbox_filter_en = false;
    config.target_ground_filter_en = false;
    config.target_volume_filter_en = false;
    config.static_support_en = false;
    return config;
}

m_detector::TrackerConfig NearRangeFastConfirmConfig()
{
    m_detector::TrackerConfig config = TestConfig();
    config.min_init_hits = 2;
    config.init_window_size = 3;
    config.near_range_fast_confirm_en = true;
    config.near_range_fast_confirm_distance = 1.5;
    config.near_range_fast_confirm_hits = 1;
    return config;
}

void ExpectSingleHitTentativeTrack(const m_detector::TrackerResult &result)
{
    EXPECT_TRUE(result.tracks.empty());
    ASSERT_EQ(result.tentative_tracks.size(), 1u);
    EXPECT_EQ(result.tentative_tracks.front().total_hits, 1u);
}

PointCloudXYZI::Ptr CloudAt(float x, float y = 0.0f, float z = 1.0f)
{
    PointCloudXYZI::Ptr cloud(new PointCloudXYZI());
    PointType point;
    point.x = x - 0.04f;
    point.y = y;
    point.z = z;
    cloud->push_back(point);
    point.x = x + 0.04f;
    cloud->push_back(point);
    point.x = x;
    point.y = y + 0.04f;
    point.z = z + 0.04f;
    cloud->push_back(point);
    return cloud;
}

PointCloudXYZI::Ptr CloudWithPointCountAt(float x, std::size_t point_count)
{
    PointCloudXYZI::Ptr cloud(new PointCloudXYZI());
    cloud->reserve(point_count);
    for (std::size_t index = 0; index < point_count; ++index)
    {
        PointType point;
        const float centered_index = static_cast<float>(index) -
            0.5f * static_cast<float>(point_count - 1);
        point.x = x + 0.02f * centered_index;
        point.y = index % 2 == 0 ? -0.01f : 0.01f;
        point.z = 1.0f + 0.01f * static_cast<float>(index % 3);
        cloud->push_back(point);
    }
    return cloud;
}

PointCloudXYZI::Ptr CloudsInOrder(std::initializer_list<float> centers)
{
    PointCloudXYZI::Ptr combined(new PointCloudXYZI());
    for (const float center : centers)
    {
        *combined += *CloudAt(center);
    }
    return combined;
}

m_detector::TrackerFrame Frame(double stamp,
                               const PointCloudXYZI::ConstPtr &frame_out,
                               const PointCloudXYZI::ConstPtr &point_out,
                               const PointCloudXYZI::ConstPtr &support,
                               bool frame_out_available = true)
{
    m_detector::TrackerFrame frame;
    frame.header.stamp = ros::Time().fromSec(stamp);
    frame.header.frame_id = "world";
    frame.frame_out = frame_out;
    frame.point_out = point_out;
    frame.support_cloud = support;
    frame.has_odom = true;
    frame.frame_out_available = frame_out_available;
    return frame;
}

m_detector::Detection3D DetectionAt(float x, float y = 0.0f, float z = 1.0f)
{
    m_detector::Detection3D detection;
    detection.centroid = Eigen::Vector3f(x, y, z);
    detection.min_point = detection.centroid - Eigen::Vector3f::Constant(0.08f);
    detection.max_point = detection.centroid + Eigen::Vector3f::Constant(0.08f);
    detection.point_count = 10;
    return detection;
}

m_detector::Detection3D DetectionWithPointCountAt(float x, std::size_t point_count)
{
    m_detector::Detection3D detection = DetectionAt(x);
    detection.point_count = point_count;
    return detection;
}

m_detector::Detection3D DetectionBox(float min_x, float max_x,
                                     float y = 0.0f, float z = 1.0f)
{
    m_detector::Detection3D detection;
    detection.min_point = Eigen::Vector3f(min_x, y - 0.08f, z - 0.08f);
    detection.max_point = Eigen::Vector3f(max_x, y + 0.08f, z + 0.08f);
    detection.centroid = (detection.min_point + detection.max_point) * 0.5f;
    detection.point_count = 10;
    return detection;
}

m_detector::Detection3D DetectionWithDimensions(float x, float dimension)
{
    m_detector::Detection3D detection;
    detection.min_point = Eigen::Vector3f(x - dimension * 0.5f,
                                          -dimension * 0.5f,
                                          1.0f - dimension * 0.5f);
    detection.max_point = Eigen::Vector3f(x + dimension * 0.5f,
                                          dimension * 0.5f,
                                          1.0f + dimension * 0.5f);
    detection.centroid = (detection.min_point + detection.max_point) * 0.5f;
    detection.point_count = 10;
    return detection;
}

m_detector::TrackerFrame DetectionFrame(
    double stamp, std::initializer_list<m_detector::Detection3D> detections)
{
    m_detector::TrackerFrame frame;
    frame.header.stamp = ros::Time().fromSec(stamp);
    frame.header.frame_id = "world";
    frame.preclustered_detections =
        std::vector<m_detector::Detection3D>(detections);
    frame.has_odom = true;
    return frame;
}

TEST(FrameTrackerTest, VolumeGateRejectsTargetsOutsideConfiguredRange)
{
    m_detector::TrackerConfig config = TestConfig();
    config.target_volume_filter_en = true;
    config.target_min_bbox_volume = 0.002;
    config.target_max_bbox_volume = 0.02;
    m_detector::FrameTracker tracker(config);

    const auto result = tracker.update(DetectionFrame(
        11.0,
        {DetectionWithDimensions(1.0f, 0.1f),
         DetectionWithDimensions(3.0f, 0.2f),
         DetectionWithDimensions(5.0f, 0.3f)}));

    ASSERT_EQ(result.tracks.size(), 1u);
    EXPECT_NEAR(result.tracks.front().position.x(), 3.0f, 1e-3f);
}

TEST(FrameTrackerTest, ConfirmedTrackUsesTheInputFrameHeader)
{
    m_detector::FrameTracker tracker(TestConfig());
    const auto cloud = CloudAt(1.0f);

    const m_detector::TrackerResult result = tracker.update(Frame(12.5, cloud, nullptr, cloud));

    ASSERT_EQ(result.tracks.size(), 1u);
    EXPECT_EQ(result.header.frame_id, "world");
    EXPECT_DOUBLE_EQ(result.header.stamp.toSec(), 12.5);
    EXPECT_EQ(result.input_mode, m_detector::TrackerInputMode::FRAME_OUT);
    EXPECT_NEAR(result.tracks.front().position.x(), 1.0f, 1e-3f);
}

TEST(FrameTrackerTest, TentativeTrackIsAvailableWithoutPollutingConfirmedTracks)
{
    m_detector::TrackerConfig config = TestConfig();
    config.min_init_hits = 3;
    config.init_window_size = 3;
    config.near_range_fast_confirm_en = false;
    m_detector::FrameTracker tracker(config);

    const auto result = tracker.update(
        Frame(13.0, CloudAt(1.0f), nullptr, nullptr));

    EXPECT_TRUE(result.tracks.empty());
    ASSERT_EQ(result.tentative_tracks.size(), 1u);
    EXPECT_FALSE(result.tentative_tracks.front().confirmed);
    EXPECT_EQ(result.tentative_tracks.front().total_hits, 1u);
}

TEST(FrameTrackerTest, NearRangeFastConfirmationUsesOdomRelativeDistance)
{
    m_detector::FrameTracker tracker(NearRangeFastConfirmConfig());

    auto frame = DetectionFrame(14.0, {DetectionAt(101.0f, 0.0f, 1.0f)});
    frame.odom_position = Eigen::Vector3f(100.0f, 0.0f, 1.0f);
    const auto result = tracker.update(frame);

    ASSERT_EQ(result.tracks.size(), 1u);
    EXPECT_TRUE(result.tentative_tracks.empty());
}

TEST(FrameTrackerTest, WorldOriginProximityDoesNotTriggerFastConfirmation)
{
    m_detector::FrameTracker tracker(NearRangeFastConfirmConfig());

    auto frame = DetectionFrame(14.1, {DetectionAt(0.0f, 0.0f, 1.0f)});
    frame.odom_position = Eigen::Vector3f(100.0f, 0.0f, 1.0f);
    const auto result = tracker.update(frame);

    ExpectSingleHitTentativeTrack(result);
}

TEST(FrameTrackerTest, MissingOdomDisablesNearRangeFastConfirmation)
{
    m_detector::FrameTracker tracker(NearRangeFastConfirmConfig());

    auto frame = DetectionFrame(14.2, {DetectionAt(0.0f, 0.0f, 1.0f)});
    frame.has_odom = false;
    const auto result = tracker.update(frame);

    ExpectSingleHitTentativeTrack(result);
}

TEST(FrameTrackerTest, MissingOdomSuspendsFastThresholdForExistingCandidate)
{
    m_detector::TrackerConfig config = NearRangeFastConfirmConfig();
    config.min_init_hits = 3;
    config.near_range_fast_confirm_hits = 2;
    m_detector::FrameTracker tracker(config);

    auto first_frame = DetectionFrame(14.25, {DetectionAt(101.0f, 0.0f, 1.0f)});
    first_frame.odom_position = Eigen::Vector3f(100.0f, 0.0f, 1.0f);
    const auto first_result = tracker.update(first_frame);
    ExpectSingleHitTentativeTrack(first_result);

    auto missing_odom_frame =
        DetectionFrame(14.35, {DetectionAt(101.0f, 0.0f, 1.0f)});
    missing_odom_frame.has_odom = false;
    const auto missing_odom_result = tracker.update(missing_odom_frame);

    EXPECT_TRUE(missing_odom_result.tracks.empty());
    ASSERT_EQ(missing_odom_result.tentative_tracks.size(), 1u);
    EXPECT_EQ(missing_odom_result.tentative_tracks.front().total_hits, 2u);
}

TEST(FrameTrackerTest, NearRangeBoundaryTriggersFastConfirmation)
{
    m_detector::FrameTracker tracker(NearRangeFastConfirmConfig());

    auto frame = DetectionFrame(14.3, {DetectionAt(101.5f, 0.0f, 1.0f)});
    frame.odom_position = Eigen::Vector3f(100.0f, 0.0f, 1.0f);
    const auto result = tracker.update(frame);

    ASSERT_EQ(result.tracks.size(), 1u);
    EXPECT_TRUE(result.tentative_tracks.empty());
}

TEST(FrameTrackerTest, MissingFrameOutForcesPointOutOnlyTracking)
{
    m_detector::TrackerConfig config = TestConfig();
    config.point_out_dynamic_support_en = false;
    m_detector::FrameTracker tracker(config);
    const auto point_out = CloudAt(2.0f);

    const m_detector::TrackerResult result =
        tracker.update(Frame(20.0, nullptr, point_out, point_out, false));

    ASSERT_EQ(result.tracks.size(), 1u);
    EXPECT_EQ(result.input_mode, m_detector::TrackerInputMode::POINT_OUT_ONLY);
    EXPECT_NEAR(result.tracks.front().position.x(), 2.0f, 1e-3f);
}

TEST(FrameTrackerTest, PointOutSupplementsAvailableFrameOut)
{
    m_detector::FrameTracker tracker(TestConfig());
    const auto frame_out = CloudAt(1.0f, 0.0f);
    const auto point_out = CloudAt(1.0f, 2.0f);

    const m_detector::TrackerResult result =
        tracker.update(Frame(25.0, frame_out, point_out, frame_out));

    EXPECT_EQ(result.input_mode, m_detector::TrackerInputMode::FRAME_OUT);
    EXPECT_EQ(result.tracks.size(), 2u);
}

TEST(FrameTrackerTest, PreclusteredDetectionBypassesPointCloudReclustering)
{
    m_detector::FrameTracker tracker(TestConfig());
    m_detector::TrackerFrame frame =
        Frame(27.0, nullptr, CloudAt(4.0f), nullptr, false);
    m_detector::Detection3D detection;
    detection.centroid = Eigen::Vector3f(1.0f, 0.0f, 1.0f);
    detection.min_point = Eigen::Vector3f(0.9f, -0.1f, 0.9f);
    detection.max_point = Eigen::Vector3f(1.1f, 0.1f, 1.1f);
    detection.point_count = 8;
    frame.preclustered_detections =
        std::vector<m_detector::Detection3D>{detection};

    const m_detector::TrackerResult result = tracker.update(frame);

    ASSERT_EQ(result.tracks.size(), 1u);
    EXPECT_EQ(result.input_mode,
              m_detector::TrackerInputMode::PRECLUSTERED_DETECTIONS);
    EXPECT_EQ(result.primary_point_count, 8u);
    EXPECT_EQ(result.point_out_point_count, 3u);
    EXPECT_NEAR(result.tracks.front().position.x(), 1.0f, 1e-3f);
}

TEST(FrameTrackerTest, EmptyPreclusteredBatchDoesNotFallBackToPointOut)
{
    m_detector::FrameTracker tracker(TestConfig());
    m_detector::TrackerFrame first =
        Frame(28.0, nullptr, nullptr, nullptr, false);
    m_detector::Detection3D detection;
    detection.centroid = Eigen::Vector3f(1.0f, 0.0f, 1.0f);
    detection.min_point = Eigen::Vector3f(0.9f, -0.1f, 0.9f);
    detection.max_point = Eigen::Vector3f(1.1f, 0.1f, 1.1f);
    detection.point_count = 8;
    first.preclustered_detections =
        std::vector<m_detector::Detection3D>{detection};
    ASSERT_EQ(tracker.update(first).tracks.size(), 1u);

    m_detector::TrackerFrame empty =
        Frame(28.1, nullptr, CloudAt(4.0f), nullptr, false);
    empty.preclustered_detections =
        std::vector<m_detector::Detection3D>{};
    const m_detector::TrackerResult result = tracker.update(empty);

    ASSERT_EQ(result.tracks.size(), 1u);
    EXPECT_EQ(result.input_mode,
              m_detector::TrackerInputMode::PRECLUSTERED_DETECTIONS);
    EXPECT_EQ(result.primary_point_count, 0u);
    EXPECT_FALSE(result.tracks.front().matched);
}

TEST(FrameTrackerTest, WeakPreclusteredDetectionsConfirmAndUpdateAnExistingTrack)
{
    m_detector::TrackerConfig config = TestConfig();
    config.cluster_min_points_process = 2;
    config.cluster_min_points = 3;
    config.min_init_hits = 2;
    config.init_window_size = 6;
    config.near_range_fast_confirm_en = false;
    m_detector::FrameTracker tracker(config);

    const auto first = tracker.update(DetectionFrame(
        29.0, {DetectionWithPointCountAt(0.0f, 5)}));
    ASSERT_TRUE(first.tracks.empty());
    ASSERT_EQ(first.tentative_tracks.size(), 1u);
    const uint32_t track_uid = first.tentative_tracks.front().track_uid;

    const auto second = tracker.update(DetectionFrame(
        29.1, {DetectionWithPointCountAt(0.1f, 2)}));
    ASSERT_EQ(second.tracks.size(), 1u);
    EXPECT_TRUE(second.tentative_tracks.empty());
    EXPECT_EQ(second.tracks.front().track_uid, track_uid);
    EXPECT_EQ(second.tracks.front().point_count, 2u);

    const auto third = tracker.update(DetectionFrame(
        29.2, {DetectionWithPointCountAt(0.2f, 2)}));
    const auto fourth = tracker.update(DetectionFrame(
        29.3, {DetectionWithPointCountAt(0.3f, 3)}));
    const auto fifth = tracker.update(DetectionFrame(
        29.4, {DetectionWithPointCountAt(0.4f, 4)}));

    ASSERT_EQ(third.tracks.size(), 1u);
    ASSERT_EQ(fourth.tracks.size(), 1u);
    ASSERT_EQ(fifth.tracks.size(), 1u);
    EXPECT_EQ(third.tracks.front().track_uid, track_uid);
    EXPECT_EQ(fourth.tracks.front().track_uid, track_uid);
    EXPECT_EQ(fifth.tracks.front().track_uid, track_uid);
    EXPECT_EQ(fifth.tracks.front().total_hits, 5u);
}

TEST(FrameTrackerTest, WeakFrameOutClustersConfirmAndUpdateAnExistingTrack)
{
    m_detector::TrackerConfig config = TestConfig();
    config.cluster_min_points_process = 2;
    config.cluster_min_points = 3;
    config.min_init_hits = 2;
    config.init_window_size = 6;
    config.near_range_fast_confirm_en = false;
    m_detector::FrameTracker tracker(config);

    const auto first = tracker.update(Frame(
        29.5, CloudWithPointCountAt(0.0f, 5), nullptr, nullptr));
    ASSERT_TRUE(first.tracks.empty());
    ASSERT_EQ(first.tentative_tracks.size(), 1u);
    const uint32_t track_uid = first.tentative_tracks.front().track_uid;

    const auto second = tracker.update(Frame(
        29.6, CloudWithPointCountAt(0.1f, 2), nullptr, nullptr));
    ASSERT_EQ(second.tracks.size(), 1u);
    EXPECT_EQ(second.tracks.front().track_uid, track_uid);

    tracker.update(Frame(29.7, CloudWithPointCountAt(0.2f, 2), nullptr, nullptr));
    tracker.update(Frame(29.8, CloudWithPointCountAt(0.3f, 3), nullptr, nullptr));
    const auto fifth = tracker.update(Frame(
        29.9, CloudWithPointCountAt(0.4f, 4), nullptr, nullptr));

    ASSERT_EQ(fifth.tracks.size(), 1u);
    EXPECT_EQ(fifth.tracks.front().track_uid, track_uid);
    EXPECT_EQ(fifth.tracks.front().total_hits, 5u);
}

TEST(FrameTrackerTest, WeakDetectionCannotInitializeANewTrack)
{
    m_detector::TrackerConfig config = TestConfig();
    config.cluster_min_points_process = 2;
    config.cluster_min_points = 3;
    m_detector::FrameTracker tracker(config);

    const auto result = tracker.update(DetectionFrame(
        30.0, {DetectionWithPointCountAt(0.0f, 2)}));

    EXPECT_EQ(result.detection_count, 1u);
    EXPECT_EQ(result.active_track_count, 0u);
    EXPECT_TRUE(result.tracks.empty());
    EXPECT_TRUE(result.tentative_tracks.empty());
}

TEST(FrameTrackerTest, DetectionBelowProcessingThresholdRegistersAMiss)
{
    m_detector::TrackerConfig config = TestConfig();
    config.cluster_min_points_process = 2;
    config.cluster_min_points = 3;
    m_detector::FrameTracker tracker(config);

    ASSERT_EQ(tracker.update(DetectionFrame(
        30.1, {DetectionWithPointCountAt(0.0f, 5)})).tracks.size(), 1u);
    const auto result = tracker.update(DetectionFrame(
        30.2, {DetectionWithPointCountAt(0.1f, 1)}));

    ASSERT_EQ(result.tracks.size(), 1u);
    EXPECT_FALSE(result.tracks.front().matched);
    EXPECT_EQ(result.tracks.front().missed_frames, 1u);
    EXPECT_EQ(result.detection_count, 0u);
}

TEST(FrameTrackerTest, UnmatchedWeakDetectionCannotCreateASecondTrack)
{
    m_detector::TrackerConfig config = TestConfig();
    config.cluster_min_points_process = 2;
    config.cluster_min_points = 3;
    m_detector::FrameTracker tracker(config);

    const auto first = tracker.update(DetectionFrame(
        30.3, {DetectionWithPointCountAt(0.0f, 5)}));
    ASSERT_EQ(first.tracks.size(), 1u);
    const uint32_t track_uid = first.tracks.front().track_uid;

    const auto result = tracker.update(DetectionFrame(30.4, {
        DetectionWithPointCountAt(0.1f, 3),
        DetectionWithPointCountAt(3.0f, 2)}));

    ASSERT_EQ(result.tracks.size(), 1u);
    EXPECT_EQ(result.tracks.front().track_uid, track_uid);
}

TEST(FrameTrackerTest, WeakSecondMeasurementCanReleaseNearRangeTrackFromSpeedGate)
{
    m_detector::TrackerConfig config = TestConfig();
    config.cluster_min_points_process = 2;
    config.cluster_min_points = 3;
    config.min_init_hits = 2;
    config.init_window_size = 3;
    config.near_range_fast_confirm_en = true;
    config.near_range_fast_confirm_distance = 1.5;
    config.near_range_fast_confirm_hits = 1;
    config.target_speed_filter_en = true;
    config.target_min_speed = 0.1;
    m_detector::FrameTracker tracker(config);

    const auto first = tracker.update(DetectionFrame(
        30.5, {DetectionWithPointCountAt(0.5f, 5)}));
    EXPECT_EQ(first.active_track_count, 1u);
    EXPECT_TRUE(first.tracks.empty());
    EXPECT_TRUE(first.tentative_tracks.empty());

    const auto second = tracker.update(DetectionFrame(
        30.6, {DetectionWithPointCountAt(0.7f, 2)}));

    ASSERT_EQ(second.tracks.size(), 1u);
    EXPECT_TRUE(second.tracks.front().matched);
    EXPECT_EQ(second.tracks.front().point_count, 2u);
    EXPECT_GT(second.tracks.front().velocity.x(), 0.1f);
}

TEST(FrameTrackerTest, CurrentRawCloudCanSupportAMissingDynamicDetection)
{
    m_detector::TrackerConfig config = TestConfig();
    config.point_out_dynamic_support_en = false;
    config.static_support_en = true;
    config.static_support_min_activation_speed = 0.0;
    config.static_support_bbox_expand_ratio = 2.0;
    config.static_support_bbox_padding = 0.2;
    config.static_support_max_residual = 0.3;
    m_detector::FrameTracker tracker(config);
    const auto target = CloudAt(1.5f);

    ASSERT_EQ(tracker.update(Frame(30.0, target, nullptr, target)).tracks.size(), 1u);
    const m_detector::TrackerResult supported =
        tracker.update(Frame(30.1, PointCloudXYZI::Ptr(new PointCloudXYZI()), nullptr, target));

    ASSERT_EQ(supported.tracks.size(), 1u);
    EXPECT_TRUE(supported.tracks.front().matched);
    EXPECT_NEAR(supported.tracks.front().position.x(), 1.5f, 0.05f);
}

TEST(FrameTrackerTest, StaticSupportPreservesFastTargetVelocity)
{
    m_detector::TrackerConfig config = TestConfig();
    config.min_iou = -1.0;
    config.max_center_distance = 1.0;
    config.static_support_en = true;
    config.static_support_min_activation_speed = 0.0;
    config.static_support_bbox_expand_ratio = 4.0;
    config.static_support_bbox_padding = 0.2;
    config.static_support_max_residual = 0.4;
    m_detector::FrameTracker tracker(config);

    tracker.update(Frame(35.0, CloudAt(0.0f), nullptr, CloudAt(0.0f)));
    const auto moving = tracker.update(
        Frame(35.1, CloudAt(0.5f), nullptr, CloudAt(0.5f)));
    ASSERT_EQ(moving.tracks.size(), 1u);
    ASSERT_GT(moving.tracks.front().velocity.x(), 4.5f);

    const PointCloudXYZI::Ptr empty(new PointCloudXYZI());
    const auto supported = tracker.update(
        Frame(35.2, empty, nullptr, CloudAt(1.0f)));

    ASSERT_EQ(supported.tracks.size(), 1u);
    EXPECT_TRUE(supported.tracks.front().matched);
    EXPECT_GT(supported.tracks.front().velocity.x(), 4.0f);
}

TEST(FrameTrackerTest, EmptyFramesEventuallyRemoveATrack)
{
    m_detector::TrackerConfig config = TestConfig();
    config.point_out_dynamic_support_en = false;
    config.track_lifetime_frames = 2;
    m_detector::FrameTracker tracker(config);
    const auto target = CloudAt(0.5f);
    const PointCloudXYZI::Ptr empty(new PointCloudXYZI());

    ASSERT_EQ(tracker.update(Frame(40.0, target, nullptr, target)).tracks.size(), 1u);
    const auto first_miss = tracker.update(Frame(40.1, empty, nullptr, empty));
    ASSERT_EQ(first_miss.tracks.size(), 1u);
    EXPECT_FALSE(first_miss.tracks.front().matched);
    EXPECT_EQ(first_miss.tracks.front().missed_frames, 1u);

    const auto boundary_miss = tracker.update(Frame(40.2, empty, nullptr, empty));
    ASSERT_EQ(boundary_miss.tracks.size(), 1u);
    EXPECT_EQ(boundary_miss.tracks.front().missed_frames, 2u);

    EXPECT_TRUE(tracker.update(Frame(40.3, empty, nullptr, empty)).tracks.empty());
}

TEST(FrameTrackerTest, ShortMissKeepsCachedShapeAtPredictedPosition)
{
    m_detector::TrackerConfig config = TestConfig();
    config.ballistic_model_en = false;
    config.lost_track_prediction_time = 0.11;
    config.min_iou = -1.0;
    m_detector::FrameTracker tracker(config);

    ASSERT_EQ(tracker.update(Frame(41.0, CloudAt(1.0f), nullptr, nullptr)).tracks.size(), 1u);
    const auto observed = tracker.update(Frame(41.05, CloudAt(1.2f), nullptr, nullptr));
    ASSERT_EQ(observed.tracks.size(), 1u);

    const PointCloudXYZI::Ptr empty(new PointCloudXYZI());
    const auto missed = tracker.update(Frame(41.1, empty, nullptr, empty));

    ASSERT_EQ(missed.tracks.size(), 1u);
    EXPECT_FALSE(missed.tracks.front().matched);
    EXPECT_GT(missed.tracks.front().position.x(), observed.tracks.front().position.x());
    EXPECT_TRUE(missed.tracks.front().dimensions.isApprox(
        observed.tracks.front().dimensions, 1e-6f));
}

TEST(FrameTrackerTest, ContinuityTimeoutStopsOutputButKeepsIdentityForReacquisition)
{
    m_detector::TrackerConfig config = TestConfig();
    config.lost_track_prediction_time = 0.1;
    config.track_lifetime_frames = 10;
    m_detector::FrameTracker tracker(config);
    const auto target = CloudAt(1.0f);
    const PointCloudXYZI::Ptr empty(new PointCloudXYZI());

    const auto observed = tracker.update(Frame(42.0, target, nullptr, nullptr));
    ASSERT_EQ(observed.tracks.size(), 1u);
    const uint32_t track_uid = observed.tracks.front().track_uid;
    ASSERT_EQ(tracker.update(Frame(42.1, empty, nullptr, empty)).tracks.size(), 1u);

    const auto expired = tracker.update(Frame(42.11, empty, nullptr, empty));
    EXPECT_TRUE(expired.tracks.empty());
    EXPECT_EQ(expired.active_track_count, 1u);

    const auto reacquired = tracker.update(Frame(42.2, target, nullptr, nullptr));
    ASSERT_EQ(reacquired.tracks.size(), 1u);
    EXPECT_EQ(reacquired.tracks.front().track_uid, track_uid);
    EXPECT_TRUE(reacquired.tracks.front().matched);
}

TEST(FrameTrackerTest, StaticSupportDoesNotExtendDynamicContinuityWindow)
{
    m_detector::TrackerConfig config = TestConfig();
    config.lost_track_prediction_time = 0.1;
    config.track_lifetime_frames = 10;
    config.static_support_en = true;
    config.static_support_min_activation_speed = 0.0;
    config.static_support_bbox_expand_ratio = 2.0;
    config.static_support_bbox_padding = 0.2;
    config.static_support_max_residual = 0.3;
    m_detector::FrameTracker tracker(config);
    const auto target = CloudAt(1.0f);
    auto wider_support = CloudAt(1.0f);
    wider_support->points[0].x -= 0.15f;
    wider_support->points[1].x += 0.15f;
    const PointCloudXYZI::Ptr empty(new PointCloudXYZI());

    const auto observed = tracker.update(Frame(43.0, target, nullptr, target));
    ASSERT_EQ(observed.tracks.size(), 1u);
    const auto supported = tracker.update(Frame(43.05, empty, nullptr, wider_support));
    ASSERT_EQ(supported.tracks.size(), 1u);
    EXPECT_TRUE(supported.tracks.front().dimensions.isApprox(
        observed.tracks.front().dimensions, 1e-6f));

    const auto expired = tracker.update(Frame(43.11, empty, nullptr, wider_support));

    EXPECT_TRUE(expired.tracks.empty());
    EXPECT_EQ(expired.active_track_count, 1u);
}

TEST(FrameTrackerTest, LostTrackPredictionTimeIsIndependentOfPublishPredictionHorizon)
{
    m_detector::TrackerConfig config = TestConfig();
    config.max_prediction_horizon = 0.01;
    config.lost_track_prediction_time = 0.3;
    config.track_lifetime_frames = 10;
    m_detector::FrameTracker tracker(config);
    const PointCloudXYZI::Ptr empty(new PointCloudXYZI());

    ASSERT_EQ(tracker.update(Frame(44.0, CloudAt(1.0f), nullptr, nullptr)).tracks.size(), 1u);
    const auto predicted = tracker.update(Frame(44.2, empty, nullptr, empty));

    ASSERT_EQ(predicted.tracks.size(), 1u);
    EXPECT_FALSE(predicted.tracks.front().matched);
}

TEST(FrameTrackerTest, LostTrackPredictionWindowOutlivesFrameBasedLifetime)
{
    m_detector::TrackerConfig config = TestConfig();
    config.lost_track_prediction_time = 0.3;
    config.track_lifetime_frames = 1;
    m_detector::FrameTracker tracker(config);
    const PointCloudXYZI::Ptr empty(new PointCloudXYZI());

    const auto observed = tracker.update(Frame(44.3, CloudAt(1.0f), nullptr, nullptr));
    ASSERT_EQ(observed.tracks.size(), 1u);
    const auto second_miss = tracker.update(Frame(44.4, empty, nullptr, empty));
    ASSERT_EQ(second_miss.tracks.size(), 1u);
    const auto third_miss = tracker.update(Frame(44.5, empty, nullptr, empty));

    ASSERT_EQ(third_miss.tracks.size(), 1u);
    EXPECT_EQ(third_miss.tracks.front().track_uid, observed.tracks.front().track_uid);
    EXPECT_EQ(third_miss.tracks.front().missed_frames, 2u);
}

TEST(FrameTrackerTest, LostBallisticTrackUsesGravityAndKeepsIdentity)
{
    m_detector::TrackerConfig config = TestConfig();
    config.ballistic_model_en = true;
    config.gravity_world_z = -10.0;
    config.lost_track_prediction_time = 0.3;
    config.track_lifetime_frames = 10;
    config.max_target_speed = 20.0;
    config.min_iou = -1.0;
    m_detector::FrameTracker tracker(config);

    const auto first = tracker.update(DetectionFrame(
        44.5, {DetectionAt(0.0f, 0.0f, 1.0f)}));
    ASSERT_EQ(first.tracks.size(), 1u);
    const auto observed = tracker.update(DetectionFrame(
        44.6, {DetectionAt(0.4f, 0.0f, 1.45f)}));
    ASSERT_EQ(observed.tracks.size(), 1u);

    const PointCloudXYZI::Ptr empty(new PointCloudXYZI());
    const auto predicted = tracker.update(Frame(44.7, empty, nullptr, empty));

    ASSERT_EQ(predicted.tracks.size(), 1u);
    EXPECT_EQ(predicted.tracks.front().track_uid, observed.tracks.front().track_uid);
    EXPECT_FALSE(predicted.tracks.front().matched);
    EXPECT_TRUE(predicted.tracks.front().position.isApprox(
        observed.tracks.front().position +
            0.1f * observed.tracks.front().velocity +
            Eigen::Vector3f(0.0f, 0.0f, -0.05f),
        1e-4f));
    EXPECT_TRUE(predicted.tracks.front().velocity.isApprox(
        observed.tracks.front().velocity + Eigen::Vector3f(0.0f, 0.0f, -1.0f),
        1e-4f));
}

TEST(FrameTrackerTest, InvalidStructuralConfigFailsAtConstruction)
{
    m_detector::TrackerConfig invalid_cluster = TestConfig();
    invalid_cluster.cluster_tolerance = 0.0;
    EXPECT_THROW(m_detector::FrameTracker tracker(invalid_cluster), std::invalid_argument);

    m_detector::TrackerConfig invalid_order = TestConfig();
    invalid_order.cluster_max_points = invalid_order.cluster_min_points - 1;
    EXPECT_THROW(m_detector::FrameTracker tracker(invalid_order), std::invalid_argument);

    m_detector::TrackerConfig invalid_processing_minimum = TestConfig();
    invalid_processing_minimum.cluster_min_points_process = 0;
    EXPECT_THROW(
        m_detector::FrameTracker tracker(invalid_processing_minimum),
        std::invalid_argument);

    m_detector::TrackerConfig invalid_processing_order = TestConfig();
    invalid_processing_order.cluster_min_points_process =
        invalid_processing_order.cluster_min_points + 1;
    EXPECT_THROW(
        m_detector::FrameTracker tracker(invalid_processing_order),
        std::invalid_argument);

    m_detector::TrackerConfig invalid_timing = TestConfig();
    invalid_timing.default_dt = 0.0;
    EXPECT_THROW(m_detector::FrameTracker tracker(invalid_timing), std::invalid_argument);

    m_detector::TrackerConfig invalid_lost_prediction_time = TestConfig();
    invalid_lost_prediction_time.lost_track_prediction_time = -0.01;
    EXPECT_THROW(
        m_detector::FrameTracker tracker(invalid_lost_prediction_time),
        std::invalid_argument);

    m_detector::TrackerConfig invalid_duplicate_guard = TestConfig();
    invalid_duplicate_guard.duplicate_guard_distance = -0.01;
    EXPECT_THROW(
        m_detector::FrameTracker tracker(invalid_duplicate_guard),
        std::invalid_argument);

    m_detector::TrackerConfig invalid_duplicate_bbox_gap = TestConfig();
    invalid_duplicate_bbox_gap.duplicate_guard_bbox_gap = -0.01;
    EXPECT_THROW(
        m_detector::FrameTracker tracker(invalid_duplicate_bbox_gap),
        std::invalid_argument);
}

TEST(FrameTrackerTest, NonMonotonicFrameTimestampIsRejected)
{
    m_detector::FrameTracker tracker(TestConfig());
    tracker.update(Frame(45.0, CloudAt(0.0f), nullptr, nullptr));

    EXPECT_THROW(
        tracker.update(Frame(45.0, CloudAt(0.1f), nullptr, nullptr)),
        std::invalid_argument);
    EXPECT_THROW(
        tracker.update(Frame(44.9, CloudAt(0.1f), nullptr, nullptr)),
        std::invalid_argument);
}

TEST(FrameTrackerTest, LargeBackwardTimestampJumpResyncsInsteadOfRejectingForever)
{
    m_detector::FrameTracker tracker(TestConfig());
    tracker.update(Frame(45.0, CloudAt(0.0f), nullptr, nullptr));

    // Small backward step: still ordinary out-of-order noise, rejected as before.
    EXPECT_THROW(
        tracker.update(Frame(44.9, CloudAt(0.1f), nullptr, nullptr)),
        std::invalid_argument);

    // Large backward jump (e.g. a sensor/odometry timestamp reset): resync instead of
    // throwing on every frame for the rest of the process lifetime.
    EXPECT_NO_THROW(tracker.update(Frame(1.0, CloudAt(0.1f), nullptr, nullptr)));

    // Ordering now tracks the new, lower stamp.
    EXPECT_NO_THROW(tracker.update(Frame(1.1, CloudAt(0.1f), nullptr, nullptr)));
}

TEST(FrameTrackerTest, LargeBackwardTimestampJumpClearsCachedTargets)
{
    m_detector::FrameTracker tracker(TestConfig());
    ASSERT_EQ(tracker.update(Frame(45.0, CloudAt(1.0f), nullptr, nullptr)).tracks.size(), 1u);

    const PointCloudXYZI::Ptr empty(new PointCloudXYZI());
    const auto reset = tracker.update(Frame(1.0, empty, nullptr, empty));

    EXPECT_TRUE(reset.tracks.empty());
    EXPECT_EQ(reset.active_track_count, 0u);
}

TEST(FrameTrackerTest, IdenticalInputSequencesProduceIdenticalResults)
{
    m_detector::FrameTracker first(TestConfig());
    m_detector::FrameTracker second(TestConfig());

    for (int index = 0; index < 4; ++index)
    {
        const auto cloud = CloudAt(0.2f * static_cast<float>(index));
        const auto first_result = first.update(Frame(50.0 + 0.1 * index, cloud, nullptr, cloud));
        const auto second_result = second.update(Frame(50.0 + 0.1 * index, cloud, nullptr, cloud));
        ASSERT_EQ(first_result.tracks.size(), second_result.tracks.size());
        ASSERT_EQ(first_result.tracks.size(), 1u);
        EXPECT_EQ(first_result.tracks.front().track_uid, second_result.tracks.front().track_uid);
        EXPECT_TRUE(first_result.tracks.front().position.isApprox(
            second_result.tracks.front().position, 1e-6f));
        EXPECT_TRUE(first_result.tracks.front().velocity.isApprox(
            second_result.tracks.front().velocity, 1e-6f));
    }
}

TEST(FrameTrackerTest, SecondDynamicMeasurementBootstrapsFastTargetVelocity)
{
    m_detector::TrackerConfig config = TestConfig();
    config.max_center_distance = 1.2;
    config.min_iou = -1.0;
    m_detector::FrameTracker tracker(config);

    ASSERT_EQ(tracker.update(Frame(60.0, CloudAt(0.0f), nullptr, nullptr)).tracks.size(), 1u);
    const auto result = tracker.update(Frame(60.1, CloudAt(0.8f), nullptr, nullptr));

    ASSERT_EQ(result.tracks.size(), 1u);
    EXPECT_NEAR(result.tracks.front().velocity.x(), 8.0f, 0.25f);
    EXPECT_NEAR(result.tracks.front().velocity.y(), 0.0f, 0.05f);
    EXPECT_NEAR(result.tracks.front().velocity.z(), 0.0f, 0.05f);
}

TEST(FrameTrackerTest, TentativeFastTargetSurvivesADroppedFrame)
{
    m_detector::TrackerConfig config = TestConfig();
    config.min_init_hits = 3;
    config.init_window_size = 4;
    config.near_range_fast_confirm_en = false;
    config.max_center_distance = 1.0;
    config.min_iou = -1.0;
    config.max_target_speed = 8.0;
    config.max_target_acceleration = 10.0;
    m_detector::FrameTracker tracker(config);

    const auto first = tracker.update(Frame(70.0, CloudAt(0.0f), nullptr, nullptr));
    ASSERT_TRUE(first.tracks.empty());
    ASSERT_EQ(first.tentative_tracks.size(), 1u);
    const auto track_uid = first.tentative_tracks.front().track_uid;

    const auto after_gap = tracker.update(Frame(70.2, CloudAt(1.6f), nullptr, nullptr));

    ASSERT_TRUE(after_gap.tracks.empty());
    ASSERT_EQ(after_gap.tentative_tracks.size(), 1u);
    EXPECT_EQ(after_gap.tentative_tracks.front().track_uid, track_uid);
    EXPECT_NEAR(after_gap.tentative_tracks.front().velocity.x(), 8.0f, 0.25f);
}

TEST(FrameTrackerTest, MatureTrackRejectsPhysicallyUnreachableDetection)
{
    m_detector::TrackerConfig config = TestConfig();
    config.max_center_distance = 100.0;
    config.min_iou = -1.0;
    config.association_padding = 0.5;
    config.association_mahalanobis_gate_sq = 1e6;
    config.max_target_speed = 8.0;
    config.max_target_acceleration = 10.0;
    m_detector::FrameTracker tracker(config);

    const auto first = tracker.update(Frame(75.0, CloudAt(0.0f), nullptr, nullptr));
    ASSERT_EQ(first.tracks.size(), 1u);
    const auto track_uid = first.tracks.front().track_uid;
    ASSERT_EQ(tracker.update(Frame(75.1, CloudAt(0.8f), nullptr, nullptr)).tracks.size(), 1u);

    const auto result = tracker.update(Frame(75.2, CloudAt(3.0f), nullptr, nullptr));

    ASSERT_EQ(result.tracks.size(), 2u);
    const auto original = std::find_if(
        result.tracks.begin(), result.tracks.end(),
        [track_uid](const m_detector::TrackSnapshot &track) {
            return track.track_uid == track_uid;
        });
    ASSERT_NE(original, result.tracks.end());
    EXPECT_FALSE(original->matched);
}

TEST(FrameTrackerTest, VelocityRemainsWithinDesignEnvelopeAfterEveryCorrection)
{
    m_detector::TrackerConfig config = TestConfig();
    config.max_center_distance = 100.0;
    config.min_iou = -1.0;
    config.association_padding = 0.5;
    config.association_mahalanobis_gate_sq = 1e6;
    config.max_target_speed = 8.0;
    config.max_target_acceleration = 10.0;
    m_detector::FrameTracker tracker(config);

    const auto first = tracker.update(Frame(77.0, CloudAt(0.0f), nullptr, nullptr));
    ASSERT_EQ(first.tracks.size(), 1u);
    const auto track_uid = first.tracks.front().track_uid;
    tracker.update(Frame(77.1, CloudAt(0.8f), nullptr, nullptr));

    const auto result = tracker.update(Frame(77.2, CloudAt(2.05f), nullptr, nullptr));

    ASSERT_EQ(result.tracks.size(), 1u);
    EXPECT_EQ(result.tracks.front().track_uid, track_uid);
    EXPECT_LE(result.tracks.front().velocity.norm(), 8.0f + 1e-4f);
}

TEST(FrameTrackerTest, AssociationFollowsInnovationInsteadOfDetectionOrder)
{
    m_detector::TrackerConfig config = TestConfig();
    config.max_center_distance = 3.0;
    config.min_iou = -1.0;
    config.association_padding = 0.0;
    m_detector::FrameTracker tracker(config);

    const auto first = tracker.update(
        Frame(80.0, CloudsInOrder({0.0f, 2.0f}), nullptr, nullptr));
    ASSERT_EQ(first.tracks.size(), 2u);
    const auto left_uid = first.tracks[0].position.x() < first.tracks[1].position.x()
                              ? first.tracks[0].track_uid
                              : first.tracks[1].track_uid;

    const auto second = tracker.update(
        Frame(80.1, CloudsInOrder({1.8f, 0.2f}), nullptr, nullptr));

    ASSERT_EQ(second.tracks.size(), 2u);
    const auto left_track = std::find_if(
        second.tracks.begin(), second.tracks.end(),
        [left_uid](const m_detector::TrackSnapshot &track) {
            return track.track_uid == left_uid;
        });
    ASSERT_NE(left_track, second.tracks.end());
    EXPECT_LT(left_track->position.x(), 0.6f);
    EXPECT_GT(left_track->velocity.x(), 0.0f);
}

TEST(FrameTrackerTest, NearbyRejectedDetectionReacquiresTheConfirmedTrack)
{
    m_detector::TrackerConfig config = TestConfig();
    config.min_iou = -1.0;
    config.association_mahalanobis_gate_sq = 1e-6;
    config.duplicate_guard_en = true;
    config.duplicate_guard_distance = 0.25;
    m_detector::FrameTracker tracker(config);

    const auto first = tracker.update(
        DetectionFrame(90.0, {DetectionAt(0.0f)}));
    ASSERT_EQ(first.tracks.size(), 1u);
    const uint32_t original_uid = first.tracks.front().track_uid;

    const auto reacquired = tracker.update(
        DetectionFrame(90.1, {DetectionAt(0.2f)}));

    ASSERT_EQ(reacquired.tracks.size(), 1u);
    EXPECT_EQ(reacquired.tracks.front().track_uid, original_uid);
    EXPECT_TRUE(reacquired.tracks.front().matched);
    EXPECT_EQ(reacquired.tracks.front().missed_frames, 0u);
}

TEST(FrameTrackerTest, ReachableDetectionRecoversAConfirmedTrackAfterPredictionDrift)
{
    m_detector::TrackerConfig config = TestConfig();
    config.ballistic_model_en = false;
    config.max_center_distance = 0.25;
    config.min_iou = -1.0;
    config.association_padding = 0.0;
    config.association_mahalanobis_gate_sq = 1e6;
    config.max_target_speed = 8.0;
    config.max_target_acceleration = 0.0;
    config.duplicate_guard_en = true;
    config.duplicate_guard_distance = 0.25;
    m_detector::FrameTracker tracker(config);

    const auto first = tracker.update(DetectionFrame(90.2, {DetectionAt(0.0f)}));
    ASSERT_EQ(first.tracks.size(), 1u);
    const uint32_t track_uid = first.tracks.front().track_uid;
    ASSERT_EQ(tracker.update(DetectionFrame(90.3, {DetectionAt(0.8f)})).tracks.size(), 1u);

    const auto recovered = tracker.update(DetectionFrame(90.4, {DetectionAt(0.9f)}));

    ASSERT_EQ(recovered.tracks.size(), 1u);
    EXPECT_EQ(recovered.tracks.front().track_uid, track_uid);
    EXPECT_TRUE(recovered.tracks.front().matched);
    EXPECT_NEAR(recovered.tracks.front().velocity.x(), 1.0f, 0.1f);
}

TEST(FrameTrackerTest, PhysicallyUnreachableDetectionDoesNotStealDriftedTrack)
{
    m_detector::TrackerConfig config = TestConfig();
    config.ballistic_model_en = false;
    config.max_center_distance = 0.25;
    config.min_iou = -1.0;
    config.association_padding = 0.0;
    config.association_mahalanobis_gate_sq = 1e6;
    config.max_target_speed = 8.0;
    config.max_target_acceleration = 0.0;
    config.duplicate_guard_en = true;
    config.duplicate_guard_distance = 0.25;
    m_detector::FrameTracker tracker(config);

    const auto first = tracker.update(DetectionFrame(90.5, {DetectionAt(0.0f)}));
    ASSERT_EQ(first.tracks.size(), 1u);
    const uint32_t track_uid = first.tracks.front().track_uid;
    ASSERT_EQ(tracker.update(DetectionFrame(90.6, {DetectionAt(0.8f)})).tracks.size(), 1u);

    const auto result = tracker.update(DetectionFrame(90.7, {DetectionAt(2.0f)}));

    ASSERT_EQ(result.tracks.size(), 2u);
    const auto original = std::find_if(
        result.tracks.begin(), result.tracks.end(),
        [track_uid](const m_detector::TrackSnapshot &track) {
            return track.track_uid == track_uid;
        });
    ASSERT_NE(original, result.tracks.end());
    EXPECT_FALSE(original->matched);
}

TEST(FrameTrackerTest, DriftRecoveryRunsBeforeOriginBoundaryTrackPruning)
{
    m_detector::TrackerConfig config = TestConfig();
    config.ballistic_model_en = false;
    config.max_center_distance = 0.25;
    config.min_iou = -1.0;
    config.association_padding = 0.0;
    config.association_mahalanobis_gate_sq = 1e6;
    config.max_target_speed = 8.0;
    config.max_target_acceleration = 0.0;
    config.duplicate_guard_en = true;
    config.duplicate_guard_distance = 0.25;
    config.target_origin_bbox_filter_en = true;
    config.target_origin_bbox_min_x = -1.0;
    config.target_origin_bbox_max_x = 1.5;
    m_detector::FrameTracker tracker(config);

    const auto first = tracker.update(DetectionFrame(90.8, {DetectionAt(0.0f)}));
    ASSERT_EQ(first.tracks.size(), 1u);
    const uint32_t track_uid = first.tracks.front().track_uid;
    ASSERT_EQ(tracker.update(DetectionFrame(90.9, {DetectionAt(0.8f)})).tracks.size(), 1u);

    const auto recovered = tracker.update(DetectionFrame(91.0, {DetectionAt(0.9f)}));

    ASSERT_EQ(recovered.tracks.size(), 1u);
    EXPECT_EQ(recovered.tracks.front().track_uid, track_uid);
    EXPECT_TRUE(recovered.tracks.front().matched);
}

TEST(FrameTrackerTest, ReachableDynamicDetectionSuppressesDriftedStaticSupport)
{
    m_detector::TrackerConfig config = TestConfig();
    config.ballistic_model_en = false;
    config.max_center_distance = 0.25;
    config.min_iou = -1.0;
    config.association_padding = 0.0;
    config.association_mahalanobis_gate_sq = 1e6;
    config.max_target_speed = 8.0;
    config.max_target_acceleration = 0.0;
    config.duplicate_guard_en = true;
    config.duplicate_guard_distance = 0.25;
    config.static_support_en = true;
    config.static_support_min_activation_speed = 0.0;
    config.static_support_bbox_expand_ratio = 2.0;
    config.static_support_bbox_padding = 0.2;
    config.static_support_max_residual = 0.3;
    m_detector::FrameTracker tracker(config);

    const auto first = tracker.update(DetectionFrame(91.1, {DetectionAt(0.0f)}));
    ASSERT_EQ(first.tracks.size(), 1u);
    const uint32_t track_uid = first.tracks.front().track_uid;
    ASSERT_EQ(tracker.update(DetectionFrame(91.2, {DetectionAt(0.8f)})).tracks.size(), 1u);

    m_detector::TrackerFrame frame =
        Frame(91.3, nullptr, nullptr, CloudAt(1.6f), false);
    frame.preclustered_detections =
        std::vector<m_detector::Detection3D>{DetectionAt(0.9f)};
    const auto recovered = tracker.update(frame);

    ASSERT_EQ(recovered.tracks.size(), 1u);
    EXPECT_EQ(recovered.tracks.front().track_uid, track_uid);
    EXPECT_TRUE(recovered.tracks.front().matched);
    EXPECT_NEAR(recovered.tracks.front().velocity.x(), 1.0f, 0.1f);
}

TEST(FrameTrackerTest, TentativeTrackCannotUsePredictionDriftRecovery)
{
    m_detector::TrackerConfig config = TestConfig();
    config.min_init_hits = 2;
    config.init_window_size = 4;
    config.near_range_fast_confirm_en = false;
    config.ballistic_model_en = false;
    config.max_center_distance = 0.25;
    config.min_iou = -1.0;
    config.association_padding = 0.0;
    config.association_mahalanobis_gate_sq = 1e-6;
    config.max_target_speed = 8.0;
    config.max_target_acceleration = 0.0;
    config.duplicate_guard_en = true;
    config.duplicate_guard_distance = 0.25;
    m_detector::FrameTracker tracker(config);

    const auto first = tracker.update(DetectionFrame(91.4, {DetectionAt(0.0f)}));
    ASSERT_EQ(first.tentative_tracks.size(), 1u);
    const uint32_t first_uid = first.tentative_tracks.front().track_uid;

    const auto result = tracker.update(DetectionFrame(91.5, {DetectionAt(0.5f)}));

    EXPECT_TRUE(result.tracks.empty());
    ASSERT_EQ(result.tentative_tracks.size(), 2u);
    const auto original = std::find_if(
        result.tentative_tracks.begin(), result.tentative_tracks.end(),
        [first_uid](const m_detector::TrackSnapshot &track) {
            return track.track_uid == first_uid;
        });
    ASSERT_NE(original, result.tentative_tracks.end());
    EXPECT_FALSE(original->matched);
}

TEST(FrameTrackerTest, TrackDeletedByLifecycleCannotRecoverItsIdentity)
{
    m_detector::TrackerConfig config = TestConfig();
    config.lost_track_prediction_time = 0.05;
    config.track_lifetime_frames = 1;
    config.duplicate_guard_en = true;
    m_detector::FrameTracker tracker(config);
    const PointCloudXYZI::Ptr empty(new PointCloudXYZI());

    const auto observed = tracker.update(DetectionFrame(91.6, {DetectionAt(0.0f)}));
    ASSERT_EQ(observed.tracks.size(), 1u);
    const uint32_t old_uid = observed.tracks.front().track_uid;
    tracker.update(Frame(91.7, empty, nullptr, empty));
    const auto deleted = tracker.update(Frame(91.8, empty, nullptr, empty));
    EXPECT_EQ(deleted.active_track_count, 0u);

    const auto replacement = tracker.update(DetectionFrame(91.9, {DetectionAt(0.0f)}));

    ASSERT_EQ(replacement.tracks.size(), 1u);
    EXPECT_NE(replacement.tracks.front().track_uid, old_uid);
}

TEST(FrameTrackerTest, NearbyPreclusteredFragmentsCreateOneTrack)
{
    m_detector::TrackerConfig config = TestConfig();
    config.duplicate_guard_en = true;
    config.duplicate_guard_distance = 0.25;
    m_detector::FrameTracker tracker(config);

    const auto result = tracker.update(
        DetectionFrame(91.0, {DetectionAt(0.0f), DetectionAt(0.2f)}));

    ASSERT_EQ(result.tracks.size(), 1u);
    EXPECT_EQ(result.tracks.front().point_count, 20u);
    EXPECT_NEAR(result.tracks.front().position.x(), 0.1f, 1e-4f);
    EXPECT_NEAR(result.tracks.front().dimensions.x(), 0.36f, 1e-4f);
}

TEST(FrameTrackerTest, NearRangeFragmentationOfExistingObjectDoesNotCreateSecondTrack)
{
    m_detector::TrackerConfig config = TestConfig();
    config.min_init_hits = 2;
    config.init_window_size = 6;
    config.near_range_fast_confirm_en = true;
    config.near_range_fast_confirm_distance = 1.5;
    config.near_range_fast_confirm_hits = 1;
    config.min_iou = -1.0;
    config.duplicate_guard_en = true;
    config.duplicate_guard_distance = 0.25;
    m_detector::FrameTracker tracker(config);

    const auto whole = tracker.update(
        DetectionFrame(91.1, {DetectionBox(-0.25f, 0.25f)}));
    ASSERT_EQ(whole.tracks.size(), 1u);
    const uint32_t original_uid = whole.tracks.front().track_uid;

    const auto split = tracker.update(DetectionFrame(
        91.2, {DetectionBox(-0.25f, -0.02f), DetectionBox(0.02f, 0.25f)}));

    ASSERT_EQ(split.tracks.size(), 1u);
    EXPECT_EQ(split.tracks.front().track_uid, original_uid);
    EXPECT_EQ(split.tracks.front().point_count, 20u);
}

TEST(FrameTrackerTest, PersistentFragmentationOfExistingObjectDoesNotCreateSecondTrack)
{
    m_detector::TrackerConfig config = TestConfig();
    config.min_init_hits = 2;
    config.init_window_size = 6;
    config.near_range_fast_confirm_en = false;
    config.min_iou = -1.0;
    config.duplicate_guard_en = true;
    config.duplicate_guard_distance = 0.25;
    m_detector::FrameTracker tracker(config);

    const auto whole = tracker.update(
        DetectionFrame(91.25, {DetectionBox(-0.25f, 0.25f)}));
    ASSERT_EQ(whole.tentative_tracks.size(), 1u);
    const uint32_t original_uid = whole.tentative_tracks.front().track_uid;

    const auto first_split = tracker.update(DetectionFrame(
        91.35, {DetectionBox(-0.25f, -0.02f), DetectionBox(0.02f, 0.25f)}));
    ASSERT_EQ(first_split.tracks.size(), 1u);
    EXPECT_TRUE(first_split.tentative_tracks.empty());

    const auto persistent_split = tracker.update(DetectionFrame(
        91.45, {DetectionBox(-0.25f, -0.02f), DetectionBox(0.02f, 0.25f)}));

    ASSERT_EQ(persistent_split.tracks.size(), 1u);
    EXPECT_TRUE(persistent_split.tentative_tracks.empty());
    EXPECT_EQ(persistent_split.tracks.front().track_uid, original_uid);
    EXPECT_EQ(persistent_split.tracks.front().point_count, 20u);
}

TEST(FrameTrackerTest, CloseAabbsWithoutSharedTrackHistoryRemainDistinct)
{
    m_detector::TrackerConfig config = TestConfig();
    config.duplicate_guard_en = true;
    config.duplicate_guard_distance = 0.25;
    m_detector::FrameTracker tracker(config);

    const auto result = tracker.update(DetectionFrame(
        91.4, {DetectionBox(-0.12f, 0.12f), DetectionBox(0.15f, 0.39f)}));

    ASSERT_EQ(result.tracks.size(), 2u);
    EXPECT_NE(result.tracks[0].track_uid, result.tracks[1].track_uid);
}

TEST(FrameTrackerTest, DistinctTargetsOutsideDuplicateGuardRemainSeparate)
{
    m_detector::TrackerConfig config = TestConfig();
    config.duplicate_guard_en = true;
    config.duplicate_guard_distance = 0.25;
    m_detector::FrameTracker tracker(config);

    const auto result = tracker.update(
        DetectionFrame(91.5, {DetectionAt(0.0f), DetectionAt(0.3f)}));

    ASSERT_EQ(result.tracks.size(), 2u);
    EXPECT_NE(result.tracks[0].track_uid, result.tracks[1].track_uid);
}

TEST(FrameTrackerTest, FragmentChainDoesNotBridgeDistinctTargets)
{
    m_detector::TrackerConfig config = TestConfig();
    config.duplicate_guard_en = true;
    config.duplicate_guard_distance = 0.25;
    m_detector::FrameTracker tracker(config);

    const auto result = tracker.update(
        DetectionFrame(91.6, {
            DetectionAt(0.0f), DetectionAt(0.2f), DetectionAt(0.4f)}));

    ASSERT_EQ(result.tracks.size(), 2u);
}

TEST(FrameTrackerTest, DuplicateConfirmedTrackIsRetiredAfterMeasurementsConverge)
{
    m_detector::TrackerConfig config = TestConfig();
    config.min_iou = -1.0;
    config.duplicate_guard_en = true;
    config.duplicate_guard_distance = 0.25;
    m_detector::FrameTracker tracker(config);

    const auto separate = tracker.update(
        DetectionFrame(92.0, {DetectionAt(0.0f), DetectionAt(0.3f)}));
    ASSERT_EQ(separate.tracks.size(), 2u);
    const uint32_t canonical_uid = std::min(
        separate.tracks[0].track_uid, separate.tracks[1].track_uid);

    const auto converging = tracker.update(
        DetectionFrame(92.1, {DetectionAt(0.1f), DetectionAt(0.2f)}));
    ASSERT_EQ(converging.tracks.size(), 1u);
    EXPECT_EQ(converging.tracks.front().track_uid, canonical_uid);
    EXPECT_TRUE(converging.tracks.front().matched);
}

TEST(FrameTrackerTest, OlderConfirmedTrackWinsDuplicateArbitration)
{
    m_detector::TrackerConfig config = TestConfig();
    config.min_iou = -1.0;
    config.duplicate_guard_en = true;
    config.duplicate_guard_distance = 0.25;
    m_detector::FrameTracker tracker(config);

    const auto original = tracker.update(
        DetectionFrame(93.0, {DetectionAt(0.0f)}));
    ASSERT_EQ(original.tracks.size(), 1u);
    const uint32_t original_uid = original.tracks.front().track_uid;

    const auto with_new_target = tracker.update(
        DetectionFrame(93.1, {DetectionAt(0.0f), DetectionAt(0.3f)}));
    ASSERT_EQ(with_new_target.tracks.size(), 2u);

    const auto converged = tracker.update(
        DetectionFrame(93.2, {DetectionAt(0.1f), DetectionAt(0.2f)}));

    ASSERT_EQ(converged.tracks.size(), 1u);
    EXPECT_EQ(converged.tracks.front().track_uid, original_uid);
    EXPECT_TRUE(converged.tracks.front().matched);
}

TEST(FrameTrackerTest, PredictionAdvancesConfirmedStateToRequestedTime)
{
    m_detector::TrackerResult measurement;
    measurement.header.stamp = ros::Time().fromSec(100.0);
    measurement.header.frame_id = "world";
    m_detector::TrackSnapshot track;
    track.confirmed = true;
    track.position = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
    track.velocity = Eigen::Vector3f(8.0f, 0.0f, 0.0f);
    measurement.tracks.push_back(track);

    const auto prediction = m_detector::PredictTrackerResult(
        measurement, ros::Time().fromSec(100.1), 0.15);

    ASSERT_EQ(prediction.tracks.size(), 1u);
    EXPECT_FALSE(prediction.stale);
    EXPECT_NEAR(prediction.prediction_horizon, 0.1, 1e-6);
    EXPECT_NEAR(prediction.header.stamp.toSec(), 100.1, 1e-6);
    EXPECT_NEAR(prediction.tracks.front().position.x(), 1.8f, 1e-5f);
}

TEST(FrameTrackerTest, BallisticPredictionAppliesGravityToPositionAndVelocity)
{
    m_detector::TrackerResult measurement;
    measurement.header.stamp = ros::Time().fromSec(100.0);
    measurement.header.frame_id = "world";
    m_detector::TrackSnapshot track;
    track.confirmed = true;
    track.position = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
    track.velocity = Eigen::Vector3f(4.0f, 0.0f, 5.0f);
    measurement.tracks.push_back(track);

    const auto prediction = m_detector::PredictTrackerResult(
        measurement,
        ros::Time().fromSec(100.2),
        0.3,
        Eigen::Vector3f(0.0f, 0.0f, -9.81f));

    ASSERT_EQ(prediction.tracks.size(), 1u);
    EXPECT_NEAR(prediction.tracks.front().position.x(), 1.8f, 1e-5f);
    EXPECT_NEAR(prediction.tracks.front().position.z(), 3.8038f, 1e-4f);
    EXPECT_NEAR(prediction.tracks.front().velocity.z(), 3.038f, 1e-4f);
}

TEST(FrameTrackerTest, BallisticBootstrapAndPredictionPreserveTrackIdentity)
{
    m_detector::TrackerConfig config = TestConfig();
    config.ballistic_model_en = true;
    config.gravity_world_z = -9.81;
    config.max_target_speed = 12.0;
    config.max_target_acceleration = 6.0;
    config.process_acceleration_stddev = 3.0;
    config.association_padding = 0.05;
    config.max_center_distance = 0.2;
    config.min_iou = -1.0;
    config.association_mahalanobis_gate_sq = 1e6;
    m_detector::FrameTracker tracker(config);

    const auto first = tracker.update(DetectionFrame(110.0, {DetectionAt(0.0f, 0.0f, 1.0f)}));
    ASSERT_EQ(first.tracks.size(), 1u);
    const int track_uid = first.tracks.front().track_uid;

    const auto second = tracker.update(DetectionFrame(
        110.1, {DetectionAt(0.4f, 0.0f, 1.35095f)}));
    ASSERT_EQ(second.tracks.size(), 1u);
    EXPECT_EQ(second.tracks.front().track_uid, track_uid);
    EXPECT_NEAR(second.tracks.front().velocity.z(), 3.019f, 1e-3f);

    const auto third = tracker.update(DetectionFrame(
        110.2, {DetectionAt(0.8f, 0.0f, 1.6038f)}));
    ASSERT_EQ(third.tracks.size(), 1u);
    EXPECT_EQ(third.tracks.front().track_uid, track_uid);
    EXPECT_NEAR(third.tracks.front().position.z(), 1.6038f, 0.03f);
}

TEST(FrameTrackerTest, BallisticConfigRejectsInvalidGravity)
{
    m_detector::TrackerConfig nonfinite = TestConfig();
    nonfinite.ballistic_model_en = true;
    nonfinite.gravity_world_z = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(m_detector::FrameTracker tracker(nonfinite), std::invalid_argument);

    m_detector::TrackerConfig zero = TestConfig();
    zero.ballistic_model_en = true;
    zero.gravity_world_x = 0.0;
    zero.gravity_world_y = 0.0;
    zero.gravity_world_z = 0.0;
    EXPECT_THROW(m_detector::FrameTracker tracker(zero), std::invalid_argument);
}

TEST(FrameTrackerTest, PredictionClampsStaleMeasurementToSafeHorizon)
{
    m_detector::TrackerResult measurement;
    measurement.header.stamp = ros::Time().fromSec(100.0);
    measurement.header.frame_id = "world";
    m_detector::TrackSnapshot track;
    track.confirmed = true;
    track.position = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
    track.velocity = Eigen::Vector3f(8.0f, 0.0f, 0.0f);
    measurement.tracks.push_back(track);

    const auto prediction = m_detector::PredictTrackerResult(
        measurement, ros::Time().fromSec(100.4), 0.15);

    ASSERT_EQ(prediction.tracks.size(), 1u);
    EXPECT_TRUE(prediction.stale);
    EXPECT_NEAR(prediction.prediction_horizon, 0.15, 1e-6);
    EXPECT_NEAR(prediction.header.stamp.toSec(), 100.15, 1e-6);
    EXPECT_NEAR(prediction.tracks.front().position.x(), 2.2f, 1e-5f);
}

} // namespace
