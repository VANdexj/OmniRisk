#ifndef CPU_MONITOR_HPP
#define CPU_MONITOR_HPP

#include <vector>
#include <deque>
#include <numeric>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <cmath>

struct CPUData {
    unsigned long long user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice;
    CPUData() : user(0), nice(0), system(0), idle(0), iowait(0), irq(0), softirq(0), steal(0), guest(0), guest_nice(0) {}
};

class IFilter {
public:
    virtual double filter(double value) = 0;
    virtual ~IFilter() = default;
};

class SimpleMovingAverageFilter : public IFilter {
public:
    SimpleMovingAverageFilter(size_t window_size) : window_size_(window_size) {}

    double filter(double value) override {
        buffer_.push_back(value);
        if (buffer_.size() > window_size_) {
            buffer_.pop_front();
        }
        return std::accumulate(buffer_.begin(), buffer_.end(), 0.0) / buffer_.size();
    }

private:
    std::deque<double> buffer_;
    size_t window_size_;
};

class ExponentialMovingAverageFilter : public IFilter {
public:
    ExponentialMovingAverageFilter(double alpha) : alpha_(alpha), initialized_(false), value_(0.0) {}

    double filter(double input) override {
        if (!initialized_) {
            value_ = input;
            initialized_ = true;
        } else {
            value_ = alpha_ * input + (1.0 - alpha_) * value_;
        }
        return value_;
    }

private:
    double alpha_;
    bool initialized_;
    double value_;
};

class LowPassFilter : public IFilter {
public:
    LowPassFilter(double cutoff_freq, double sample_rate)
        : alpha_(calculate_alpha(cutoff_freq, sample_rate)), last_value_(0.0) {}

    double filter(double value) override {
        last_value_ = alpha_ * value + (1.0 - alpha_) * last_value_;
        return last_value_;
    }

private:
    double alpha_;
    double last_value_;

    static double calculate_alpha(double cutoff_freq, double sample_rate) {
        double dt = 1.0 / sample_rate;
        double rc = 1.0 / (2.0 * M_PI * cutoff_freq);
        return dt / (rc + dt);
    }
};

class CPUMonitor {
public:
    enum class FilterType { SMA, EMA, LowPass };

    CPUMonitor(FilterType filter_type = FilterType::LowPass, double param1 = 0.5, double param2 = 2.0) 
        : cpu_count_(get_cpu_count_from_sys()) {
        set_filter(filter_type, param1, param2);
        prev_cpu_data_ = read_all_cpu_stats();
    }

    void set_filter(FilterType filter_type, double param1, double param2 = 0.0) {
        filters_.clear();
        for (int i = 0; i < cpu_count_; ++i) {
            switch (filter_type) {
                case FilterType::SMA:
                    filters_.push_back(std::make_unique<SimpleMovingAverageFilter>(static_cast<size_t>(param1)));
                    break;
                case FilterType::EMA:
                    filters_.push_back(std::make_unique<ExponentialMovingAverageFilter>(param1));
                    break;
                case FilterType::LowPass:
                    filters_.push_back(std::make_unique<LowPassFilter>(param1, param2));
                    break;
            }
        }
    }
    std::vector<double> get_cpu_usage() {
        std::vector<CPUData> current_cpu_data = read_all_cpu_stats();
        std::vector<double> usage(cpu_count_);

        for (int i = 0; i < cpu_count_; ++i) {
            double raw_usage = calculate_cpu_usage(prev_cpu_data_[i], current_cpu_data[i]);
            usage[i] = filters_[i]->filter(raw_usage);
        }

        prev_cpu_data_ = current_cpu_data;
        return usage;
    }

    int get_cpu_count() const {
        return cpu_count_;
    }

    // static method, callable without an instance
    static int get_cpu_count_from_sys() {
        std::ifstream proc_cpuinfo("/proc/cpuinfo");
        if (!proc_cpuinfo.is_open()) {
            throw std::runtime_error("Unable to open /proc/cpuinfo");
        }

        std::string line;
        int cpu_count = 0;
        while (std::getline(proc_cpuinfo, line)) {
            if (line.substr(0, 9) == "processor") {
                cpu_count++;
            }
        }
        return cpu_count > 0 ? cpu_count : throw std::runtime_error("Unable to determine CPU count");
    }

private:
    int cpu_count_;
    std::vector<CPUData> prev_cpu_data_;
    std::vector<std::unique_ptr<IFilter>> filters_;

    static CPUData read_cpu_stats(int core_id) {
        std::ifstream stat_file("/proc/stat");
        if (!stat_file.is_open()) {
            throw std::runtime_error("Unable to open /proc/stat");
        }

        std::string line;
        CPUData cpu_data;
        std::string cpu_prefix = "cpu" + std::to_string(core_id) + " ";

        while (std::getline(stat_file, line)) {
            if (line.compare(0, cpu_prefix.length(), cpu_prefix) == 0) {
                std::istringstream ss(line.substr(cpu_prefix.length()));
                ss >> cpu_data.user >> cpu_data.nice >> cpu_data.system >> cpu_data.idle
                   >> cpu_data.iowait >> cpu_data.irq >> cpu_data.softirq >> cpu_data.steal
                   >> cpu_data.guest >> cpu_data.guest_nice;
                return cpu_data;
            }
        }

        throw std::runtime_error("CPU core not found: " + std::to_string(core_id));
    }

    std::vector<CPUData> read_all_cpu_stats() {
        std::vector<CPUData> cpu_data(cpu_count_);
        for (int i = 0; i < cpu_count_; ++i) {
            cpu_data[i] = read_cpu_stats(i);
        }
        return cpu_data;
    }

    static double calculate_cpu_usage(const CPUData& first, const CPUData& second) {
        unsigned long long total_time1 = first.user + first.nice + first.system + first.idle + 
                                         first.iowait + first.irq + first.softirq + first.steal;
        unsigned long long total_time2 = second.user + second.nice + second.system + second.idle + 
                                         second.iowait + second.irq + second.softirq + second.steal;

        unsigned long long idle_time1 = first.idle + first.iowait;
        unsigned long long idle_time2 = second.idle + second.iowait;

        unsigned long long total_time_diff = total_time2 - total_time1;
        unsigned long long idle_time_diff = idle_time2 - idle_time1;

        if (total_time_diff == 0) {
            return 0.0;
        }

        return (1.0 - static_cast<double>(idle_time_diff) / total_time_diff) * 100.0;
    }
};

#endif // CPU_MONITOR_HPP

