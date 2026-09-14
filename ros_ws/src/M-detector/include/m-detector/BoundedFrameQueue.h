#ifndef M_DETECTOR_BOUNDED_FRAME_QUEUE_H
#define M_DETECTOR_BOUNDED_FRAME_QUEUE_H

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

#include <ros/time.h>

namespace m_detector
{

enum class BoundedFrameQueuePushResult
{
    ACCEPTED,
    ACCEPTED_AFTER_RESET,
    DROPPED_OLDEST,
    REJECTED_OUT_OF_ORDER,
    REJECTED_CLOSED
};

template <typename T>
class BoundedFrameQueue
{
public:
    explicit BoundedFrameQueue(std::size_t capacity)
        : capacity_(capacity)
    {
        if (capacity_ == 0)
        {
            throw std::invalid_argument("BoundedFrameQueue capacity must be positive");
        }
    }

    struct Entry
    {
        T value;
        ros::Time stamp;
    };

    BoundedFrameQueuePushResult push(T value, const ros::Time &stamp)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_)
        {
            return BoundedFrameQueuePushResult::REJECTED_CLOSED;
        }
        bool reset_ordering = false;
        if (has_last_stamp_ && stamp <= last_stamp_)
        {
            if ((last_stamp_ - stamp).toSec() < kBackwardJumpResetSeconds)
            {
                return BoundedFrameQueuePushResult::REJECTED_OUT_OF_ORDER;
            }
            // Large backward jump (sensor/odometry timestamp reset, e.g. FAST-LIO's
            // "lidar loop back" recovery) rather than normal jitter: resync instead of
            // rejecting every subsequent frame for the rest of the node's lifetime.
            reset_ordering = true;
        }

        const bool dropped_oldest = pending_.size() >= capacity_;
        if (dropped_oldest)
        {
            pending_.pop_front();
            ++dropped_oldest_count_;
        }
        pending_.push_back(Entry{std::move(value), stamp});
        has_last_stamp_ = true;
        last_stamp_ = stamp;
        condition_.notify_one();
        if (reset_ordering)
        {
            return BoundedFrameQueuePushResult::ACCEPTED_AFTER_RESET;
        }
        return dropped_oldest ? BoundedFrameQueuePushResult::DROPPED_OLDEST
                              : BoundedFrameQueuePushResult::ACCEPTED;
    }

    std::optional<Entry> tryTake()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return takeLocked();
    }

    std::optional<Entry> waitAndTake()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this]() { return closed_ || !pending_.empty(); });
        return takeLocked();
    }

    void close()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        condition_.notify_all();
    }

    std::size_t droppedOldestCount() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_oldest_count_;
    }

private:
    std::optional<Entry> takeLocked()
    {
        if (pending_.empty())
        {
            return std::nullopt;
        }
        std::optional<Entry> result(std::move(pending_.front()));
        pending_.pop_front();
        return result;
    }

    static constexpr double kBackwardJumpResetSeconds = 5.0;

    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<Entry> pending_;
    bool closed_ = false;
    bool has_last_stamp_ = false;
    ros::Time last_stamp_;
    std::size_t dropped_oldest_count_ = 0;
};

} // namespace m_detector

#endif
