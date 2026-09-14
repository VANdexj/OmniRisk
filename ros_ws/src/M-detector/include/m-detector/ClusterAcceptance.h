#ifndef M_DETECTOR_CLUSTER_ACCEPTANCE_H
#define M_DETECTOR_CLUSTER_ACCEPTANCE_H

namespace m_detector
{

inline bool ClusterShapeAccepted(
    int cluster_min_pixel_number,
    float voxel_resolution,
    float size_x,
    float size_y,
    float size_z)
{
    return cluster_min_pixel_number == 1 ||
           ((size_x > voxel_resolution + 0.001f) &&
            (size_y > voxel_resolution + 0.001f)) ||
           ((size_x > voxel_resolution + 0.001f) &&
            (size_z > voxel_resolution + 0.001f)) ||
           ((size_y > voxel_resolution + 0.001f) &&
            (size_z > voxel_resolution + 0.001f));
}

inline bool ClusterEvidenceAccepted(
    int dynamic_point_count,
    int total_point_count,
    bool has_near_range_fast_point,
    float trustable_threshold,
    float near_range_trustable_threshold)
{
    if (dynamic_point_count <= 0 || total_point_count <= 0) {
        return false;
    }
    const float threshold = has_near_range_fast_point
                                ? near_range_trustable_threshold
                                : trustable_threshold;
    return static_cast<float>(dynamic_point_count) >=
           threshold * static_cast<float>(total_point_count);
}

} // namespace m_detector

#endif
