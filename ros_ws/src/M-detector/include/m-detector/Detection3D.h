#ifndef M_DETECTOR_DETECTION_3D_H
#define M_DETECTOR_DETECTION_3D_H

#include <cstddef>

#include <Eigen/Core>

namespace m_detector
{

// A compact, pre-clustered dynamic-object measurement shared by detector and tracker.
struct Detection3D
{
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    Eigen::Vector3f min_point = Eigen::Vector3f::Zero();
    Eigen::Vector3f max_point = Eigen::Vector3f::Zero();
    std::size_t point_count = 0;
};

} // namespace m_detector

#endif
