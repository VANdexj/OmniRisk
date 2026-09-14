#include <gtest/gtest.h>

#include <chrono>
#include <future>

#include <m-detector/BoundedFrameQueue.h>

namespace
{

TEST(BoundedFrameQueueTest, FivePendingFramesAreRetainedInFifoOrder)
{
    m_detector::BoundedFrameQueue<int> queue(5);

    for (int value = 1; value <= 5; ++value)
    {
        EXPECT_EQ(queue.push(value, ros::Time().fromSec(value)),
                  m_detector::BoundedFrameQueuePushResult::ACCEPTED);
    }

    for (int value = 1; value <= 5; ++value)
    {
        const auto entry = queue.tryTake();
        ASSERT_TRUE(entry);
        EXPECT_EQ(entry->value, value);
        EXPECT_DOUBLE_EQ(entry->stamp.toSec(), value);
    }
    EXPECT_FALSE(queue.tryTake());
}

TEST(BoundedFrameQueueTest, SixthPendingFrameDropsOnlyTheOldestFrame)
{
    m_detector::BoundedFrameQueue<int> queue(5);

    for (int value = 1; value <= 5; ++value)
    {
        EXPECT_EQ(queue.push(value, ros::Time().fromSec(value)),
                  m_detector::BoundedFrameQueuePushResult::ACCEPTED);
    }
    EXPECT_EQ(queue.push(6, ros::Time().fromSec(6.0)),
              m_detector::BoundedFrameQueuePushResult::DROPPED_OLDEST);
    EXPECT_EQ(queue.droppedOldestCount(), 1u);

    for (int value = 2; value <= 6; ++value)
    {
        const auto entry = queue.tryTake();
        ASSERT_TRUE(entry);
        EXPECT_EQ(entry->value, value);
        EXPECT_DOUBLE_EQ(entry->stamp.toSec(), value);
    }
    EXPECT_FALSE(queue.tryTake());
}

TEST(BoundedFrameQueueTest, OutOfOrderFrameCannotEvictQueuedWork)
{
    m_detector::BoundedFrameQueue<int> queue(5);

    EXPECT_EQ(queue.push(20, ros::Time().fromSec(2.0)),
              m_detector::BoundedFrameQueuePushResult::ACCEPTED);
    EXPECT_EQ(queue.push(10, ros::Time().fromSec(1.0)),
              m_detector::BoundedFrameQueuePushResult::REJECTED_OUT_OF_ORDER);

    const auto entry = queue.tryTake();
    ASSERT_TRUE(entry);
    EXPECT_EQ(entry->value, 20);
    EXPECT_DOUBLE_EQ(entry->stamp.toSec(), 2.0);
}

TEST(BoundedFrameQueueTest, LargeBackwardJumpResetsOrderingInsteadOfRejectingForever)
{
    m_detector::BoundedFrameQueue<int> queue(5);

    EXPECT_EQ(queue.push(20, ros::Time().fromSec(20.0)),
              m_detector::BoundedFrameQueuePushResult::ACCEPTED);
    // Small backward step: still treated as ordinary out-of-order noise.
    EXPECT_EQ(queue.push(19, ros::Time().fromSec(19.0)),
              m_detector::BoundedFrameQueuePushResult::REJECTED_OUT_OF_ORDER);
    // Large backward jump (e.g. a sensor/odometry timestamp reset): resync instead of
    // rejecting every frame for the rest of the node's lifetime.
    EXPECT_EQ(queue.push(1, ros::Time().fromSec(1.0)),
              m_detector::BoundedFrameQueuePushResult::ACCEPTED_AFTER_RESET);
    // Ordering now tracks the new, lower stamp.
    EXPECT_EQ(queue.push(2, ros::Time().fromSec(2.0)),
              m_detector::BoundedFrameQueuePushResult::ACCEPTED);

    for (int value : {20, 1, 2})
    {
        const auto entry = queue.tryTake();
        ASSERT_TRUE(entry);
        EXPECT_EQ(entry->value, value);
    }
    EXPECT_FALSE(queue.tryTake());
}

TEST(BoundedFrameQueueTest, ClosingQueueWakesBlockedWaiterAndRejectsNewFrames)
{
    using namespace std::chrono_literals;
    m_detector::BoundedFrameQueue<int> queue(5);
    std::promise<void> waiter_started;
    std::future<void> waiter_started_future = waiter_started.get_future();
    auto waiter = std::async(std::launch::async, [&queue, &waiter_started]() {
        waiter_started.set_value();
        return queue.waitAndTake();
    });

    waiter_started_future.wait();
    EXPECT_EQ(waiter.wait_for(50ms), std::future_status::timeout);
    queue.close();

    EXPECT_EQ(waiter.wait_for(1s), std::future_status::ready);
    EXPECT_FALSE(waiter.get());
    EXPECT_EQ(queue.push(10, ros::Time().fromSec(1.0)),
              m_detector::BoundedFrameQueuePushResult::REJECTED_CLOSED);
}

} // namespace
