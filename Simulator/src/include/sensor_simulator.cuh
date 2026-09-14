#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include "cuda_toolkit/se3.cuh"
#include <cmath>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/common/common.h> // For pcl::getMinMax3D
#include <pcl/point_cloud.h>   // For pcl::PointCloud
#include <pcl/point_types.h>   // For pcl::PointXYZ
#include <chrono>

namespace raycast
{
    struct Vector3f
    {
        float x, y, z;
        __device__ __host__ Vector3f() : x(0.0f), y(0.0f), z(0.0f) {}
        __device__ __host__ Vector3f(float x_val, float y_val, float z_val)
            : x(x_val), y(y_val), z(z_val) {}
    };

    struct Vector3i
    {
        int x, y, z;
        __device__ __host__ Vector3i() : x(0), y(0), z(0) {}
        __device__ __host__ Vector3i(int x_val, int y_val, int z_val)
            : x(x_val), y(y_val), z(z_val) {}
    };

    struct CameraParams
    {
        float fx = 80.0f; // focal length x
        float fy = 80.0f; // focal length y
        float cx = 80.0f; // principal point x (image center)
        float cy = 45.0f; // principal point y (image center)
        int image_width = 160;
        int image_height = 90;
        float max_depth_dist{20};
        bool normalize_depth{false};
    };

    struct LidarParams
    {
        int vertical_lines = 16;            // 16 vertical lines
        float vertical_angle_start = -15.0; // start vertical angle
        float vertical_angle_end = 15.0;    // end vertical angle
        int horizontal_num = 360;           // 360 horizontal points
        float horizontal_resolution = 1.0;  // 1 deg horizontal resolution
        float max_lidar_dist{50};
    };

    struct SphereObstacle
    {
        Vector3f center;
        float radius = 0.0f;
        int32_t obj_id = 0;   // 0 = no id assigned; >0 = dynamic-object id
    };

    struct CylinderObstacle
    {
        Vector3f center;
        float radius = 0.0f;
        float height = 0.0f;
        int32_t obj_id = 0;
    };

    // Per-ray hit description; obj_id==0 means "no dynamic hit / static" so that
    // a zero-initialized mask buffer encodes "static or empty" by default.
    struct HitResult
    {
        float   t = -1.0f;     // hit distance (m); <=0 means miss
        int32_t obj_id = 0;    // 0 = static/miss, >0 = dynamic-object id
    };

    class GridMap
    {
        public:
            GridMap(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float resolution, int occupy_threshold);
            ~GridMap() {};
            void freeGridMap() {cudaFree(map_cuda_);}
            __host__ __device__ Vector3i Pos2Vox(const Vector3f &pos);
            __host__ __device__ Vector3f Vox2Pos(const Vector3i &vox);
            __host__ __device__ int Vox2Idx(const Vector3i &vox);
            __host__ __device__ Vector3i Idx2Vox(int idx);
            __device__ int symmetricIndex(int index, int length);
            __device__ int mapQuery(const Vector3f &pos);

            float raycast_step_; // raycast step

            __host__ __device__ float getResolution() const { return resolution_; }
            __host__ __device__ float getOriginX()    const { return origin_x_;   }
            __host__ __device__ float getOriginY()    const { return origin_y_;   }
            __host__ __device__ float getOriginZ()    const { return origin_z_;   }
        private:
            // map param
            int *map_cuda_;
            float resolution_;                                           // grid resolution
            float origin_x_, origin_y_, origin_z_;                       // origin coordinates
            int grid_size_x_, grid_size_y_, grid_size_z_, grid_size_yz_; // grid sizes
            int occupy_threshold_;                                        // occupancy threshold
    };

    __global__ void cameraRaycastKernel(float *depth_values, GridMap grid_map, CameraParams camera_param, cudaMat::SE3<float> T_wc,
                                       const SphereObstacle *dynamic_obstacles, int dynamic_obstacle_count,
                                       const CylinderObstacle *dynamic_cylinders, int dynamic_cylinder_count);
    __global__ void lidarRaycastKernel(Vector3f *point_values, GridMap grid_map, LidarParams lidar_param, cudaMat::SE3<float> T_wc,
                                       const SphereObstacle *dynamic_obstacles, int dynamic_obstacle_count,
                                       const CylinderObstacle *dynamic_cylinders, int dynamic_cylinder_count);
    __global__ void rangeImageRaycastKernel(float *range_values, int16_t *mask_values,
                                            GridMap grid_map, LidarParams lidar_param, cudaMat::SE3<float> T_wb,
                                            const SphereObstacle *dynamic_obstacles, int dynamic_obstacle_count,
                                            const CylinderObstacle *dynamic_cylinders, int dynamic_cylinder_count);

    void renderDepthImage(GridMap *grid_map, CameraParams *camera_param, cudaMat::SE3<float>& T_wc, cv::Mat &depth_image,
                          const std::vector<SphereObstacle> &dynamic_obstacles = {},
                          const std::vector<CylinderObstacle> &dynamic_cylinders = {});
    void renderLidarPointcloud(GridMap *grid_map, LidarParams *lidar_param, cudaMat::SE3<float>& T_wc, pcl::PointCloud<pcl::PointXYZ>& lidar_points,
                               const std::vector<SphereObstacle> &dynamic_obstacles = {},
                               const std::vector<CylinderObstacle> &dynamic_cylinders = {});
    void renderRangeImage(GridMap *grid_map, LidarParams *lidar_param, cudaMat::SE3<float>& T_wb, cv::Mat &range_image,
                          const std::vector<SphereObstacle> &dynamic_obstacles = {},
                          const std::vector<CylinderObstacle> &dynamic_cylinders = {});

    // v5: extended overload that also writes a per-pixel dynamic-object id mask.
    // mask_image is allocated as CV_16SC1 with the same shape as range_image.
    // Pixels hitting a dynamic obstacle store the obstacle's obj_id; otherwise 0.
    void renderRangeImage(GridMap *grid_map, LidarParams *lidar_param, cudaMat::SE3<float>& T_wb,
                          cv::Mat &range_image, cv::Mat &mask_image,
                          const std::vector<SphereObstacle> &dynamic_obstacles = {},
                          const std::vector<CylinderObstacle> &dynamic_cylinders = {});
}
#endif // CUDA_UTILS_CUH
