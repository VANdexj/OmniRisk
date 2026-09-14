#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

class Timer {
public:
    explicit Timer(const std::string &name = "Timer")
        : name_(name), is_running_(false) {}

    void tik() {
        start_time_ = Clock::now();
        is_running_ = true;
    }

    void tok() {
        if (!is_running_) {
            std::cerr << "[" << name_ << "] Warning: Timer was not started. Call tik() first.\n";
            return;
        }
        const double milliseconds = elapsedMilliseconds(start_time_, Clock::now());
        std::cout << "[" << name_ << "] Elapsed time: " << std::fixed
                  << std::setprecision(3) << milliseconds << " ms\n";
        is_running_ = false;
    }

    void reset() { is_running_ = false; }

private:
    using Clock = std::chrono::steady_clock;

    static double elapsedMilliseconds(const Clock::time_point &start,
                                      const Clock::time_point &end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    std::string name_;
    Clock::time_point start_time_;
    bool is_running_;
};

// One CSV row is emitted for each processed frame. Extra metrics can be
// recorded with setMetric(), while ScopedTiming records wall-clock costs.
class CsvTimingRecorder {
public:
    using Clock = std::chrono::steady_clock;

    static CsvTimingRecorder &instance() {
        static CsvTimingRecorder recorder;
        return recorder;
    }

    void configure(const std::string &csv_path, bool enabled) {
        std::lock_guard<std::mutex> lock(mutex_);
        enabled_ = enabled && !csv_path.empty();
        active_ = false;
        output_.close();
        if (!enabled_) return;

        output_.open(csv_path, std::ios::out | std::ios::trunc);
        if (!output_.is_open()) {
            enabled_ = false;
            std::cerr << "[CsvTimingRecorder] Failed to open " << csv_path << '\n';
            return;
        }
        output_ << "frame,scan_time,points";
        for (const std::string &column : metricColumns()) output_ << ',' << column;
        output_ << '\n';
        output_.flush();
        std::cout << "[CsvTimingRecorder] Writing frame timings to " << csv_path << '\n';
    }

    bool enabled() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return enabled_;
    }

    void beginFrame(int frame, double scan_time, size_t points) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!enabled_) return;
        frame_ = frame;
        scan_time_ = scan_time;
        points_ = points;
        metrics_.clear();
        active_ = true;
    }

    void setMetric(const std::string &name, double milliseconds) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_) metrics_[name] = milliseconds;
    }

    void setMetrics(
        std::initializer_list<std::pair<const char *, double>> values) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!active_) return;
        for (const auto &value : values) metrics_[value.first] = value.second;
    }

    void addMetric(const std::string &name, double milliseconds) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_) metrics_[name] += milliseconds;
    }

    void endFrame() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!enabled_ || !active_) return;
        output_ << frame_ << ',' << std::setprecision(15) << scan_time_ << ',' << points_;
        output_ << std::fixed << std::setprecision(6);
        for (const std::string &column : metricColumns()) {
            const auto it = metrics_.find(column);
            output_ << ',' << (it == metrics_.end() ? 0.0 : it->second);
        }
        output_ << '\n';
        output_.flush();
        active_ = false;
    }

private:
    CsvTimingRecorder() = default;

    static const std::vector<std::string> &metricColumns() {
        static const std::vector<std::string> columns = {
            "input_preparation",
            "cpu_preprocessing",
            "cpu_case1",
            "cpu_case23",
            "cpu_voxel_cluster",
            "cpu_cluster_track",
            "cpu_cluster_input_rosmsg",
            "cpu_cluster_bbox_build",
            "cpu_cluster_point_assign",
            "cpu_cluster_ground_refine",
            "cpu_cluster_cleanup",
            "cpu_cluster_label_update",
            "cpu_depth_map_maintain",
            "cpu_output_build",
            "cpu_postprocessing",
            "steady_points_publish",
            "publish_total",
            "frame_total",
            "steady_accumulate_total",
            "steady_voxel_filter_total",
            "steady_to_rosmsg_total",
            "steady_ros_publish_total",
            "cpu_classification_total",
            "cpu_case1_total",
            "cpu_case2_total",
            "cpu_case3_total",
            "cpu_case23_total",
            "cpu_cluster_total",
            "cluster_voxel_total",
            "filter_total",
            "tracker_enabled",
            "tracker_mode",
            "tracker_input_points",
            "tracker_point_out_points",
            "tracker_support_points",
            "tracker_detection_count",
            "tracker_track_count",
            "tracker_confirmed_count",
            "tracker_update",
            "tracker_publish",
            "tracker_frame_failed",
            "tracker_preclustered",
            "queue_wait_ms",
            "measurement_age_at_start_ms",
            "measurement_age_at_tracker_ms",
            "measurement_age_at_frame_end_ms",
            "cloud_odom_stamp_delta_ms",
        };
        return columns;
    }

    mutable std::mutex mutex_;
    std::ofstream output_;
    std::map<std::string, double> metrics_;
    bool enabled_ = false;
    bool active_ = false;
    int frame_ = 0;
    double scan_time_ = 0.0;
    size_t points_ = 0;
};

class ScopedTiming {
public:
    explicit ScopedTiming(const std::string &metric_name)
        : metric_name_(metric_name),
          start_(CsvTimingRecorder::Clock::now()),
          running_(CsvTimingRecorder::instance().enabled()) {}

    ~ScopedTiming() { stop(); }

    double stop() {
        if (!running_) return 0.0;
        const double milliseconds =
            std::chrono::duration<double, std::milli>(
                CsvTimingRecorder::Clock::now() - start_).count();
        CsvTimingRecorder::instance().addMetric(metric_name_, milliseconds);
        running_ = false;
        return milliseconds;
    }

private:
    std::string metric_name_;
    CsvTimingRecorder::Clock::time_point start_;
    bool running_;
};

#endif // TIMER_H
