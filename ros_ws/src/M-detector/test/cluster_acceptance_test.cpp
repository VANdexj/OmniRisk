#include <gtest/gtest.h>

#include <m-detector/ClusterAcceptance.h>

TEST(ClusterAcceptanceTest, RejectsMixedStaticBackgroundAtNormalThreshold)
{
    EXPECT_FALSE(m_detector::ClusterEvidenceAccepted(4, 10, false, 0.85f, 0.5f));
    EXPECT_TRUE(m_detector::ClusterEvidenceAccepted(9, 10, false, 0.85f, 0.5f));
}

TEST(ClusterAcceptanceTest, UsesInclusiveThresholdBoundary)
{
    EXPECT_TRUE(m_detector::ClusterEvidenceAccepted(17, 20, false, 0.85f, 0.5f));
    EXPECT_TRUE(m_detector::ClusterEvidenceAccepted(1, 2, true, 0.85f, 0.5f));
}

TEST(ClusterAcceptanceTest, KeepsNearRangeEvidenceThresholdIndependent)
{
    EXPECT_FALSE(m_detector::ClusterEvidenceAccepted(6, 10, false, 0.85f, 0.5f));
    EXPECT_TRUE(m_detector::ClusterEvidenceAccepted(6, 10, true, 0.85f, 0.5f));
}

TEST(ClusterAcceptanceTest, RejectsEmptyEvidence)
{
    EXPECT_FALSE(m_detector::ClusterEvidenceAccepted(0, 0, false, 0.85f, 0.5f));
    EXPECT_FALSE(m_detector::ClusterEvidenceAccepted(0, 10, true, 0.85f, 0.5f));
}
