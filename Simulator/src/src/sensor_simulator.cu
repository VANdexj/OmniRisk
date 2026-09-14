#include "sensor_simulator.cuh"

namespace raycast
{
    __device__ __forceinline__ float dotFloat3(const float3 &a, const float3 &b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    __device__ __forceinline__ float3 subFloat3(const float3 &a, const float3 &b)
    {
        return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    __device__ HitResult findNearestSphereHit(const float3 &ray_origin_w,
                                              const float3 &ray_dir_w,
                                              const SphereObstacle *dynamic_obstacles,
                                              int dynamic_obstacle_count,
                                              float max_ray_length)
    {
        HitResult result;
        float best_hit = max_ray_length + 1.0f;
        int   best_id  = 0;
        for (int i = 0; i < dynamic_obstacle_count; ++i)
        {
            const SphereObstacle obstacle = dynamic_obstacles[i];
            if (obstacle.radius <= 0.0f) continue;
            const float3 center_w = make_float3(obstacle.center.x, obstacle.center.y, obstacle.center.z);
            const float3 oc = subFloat3(ray_origin_w, center_w);
            const float b = dotFloat3(oc, ray_dir_w);
            const float c = dotFloat3(oc, oc) - obstacle.radius * obstacle.radius;
            const float discriminant = b * b - c;
            if (discriminant < 0.0f) continue;
            const float sqrt_d = sqrtf(discriminant);
            float hit = -b - sqrt_d;
            if (hit <= 0.0f) hit = -b + sqrt_d;
            if (hit > 0.0f && hit <= max_ray_length && hit < best_hit)
            {
                best_hit = hit;
                best_id  = obstacle.obj_id;
            }
        }
        if (best_hit <= max_ray_length) { result.t = best_hit; result.obj_id = best_id; }
        return result;
    }

    __device__ void updateBestCylinderHit(float candidate,
                                          const float3 &ray_origin_w,
                                          const float3 &ray_dir_w,
                                          const CylinderObstacle &obstacle,
                                          float max_ray_length,
                                          float &best_hit, int &best_id)
    {
        if (candidate <= 0.0f || candidate > max_ray_length || candidate >= best_hit) return;
        const float z = ray_origin_w.z + candidate * ray_dir_w.z;
        const float z_min = obstacle.center.z - 0.5f * obstacle.height;
        const float z_max = obstacle.center.z + 0.5f * obstacle.height;
        if (z >= z_min && z <= z_max) { best_hit = candidate; best_id = obstacle.obj_id; }
    }

    __device__ void updateBestCylinderCapHit(float cap_z,
                                             const float3 &ray_origin_w,
                                             const float3 &ray_dir_w,
                                             const CylinderObstacle &obstacle,
                                             float max_ray_length,
                                             float &best_hit, int &best_id)
    {
        if (fabsf(ray_dir_w.z) < 1e-5f) return;
        const float hit = (cap_z - ray_origin_w.z) / ray_dir_w.z;
        if (hit <= 0.0f || hit > max_ray_length || hit >= best_hit) return;
        const float x = ray_origin_w.x + hit * ray_dir_w.x - obstacle.center.x;
        const float y = ray_origin_w.y + hit * ray_dir_w.y - obstacle.center.y;
        if (x * x + y * y <= obstacle.radius * obstacle.radius) { best_hit = hit; best_id = obstacle.obj_id; }
    }

    __device__ HitResult findNearestCylinderHit(const float3 &ray_origin_w,
                                                const float3 &ray_dir_w,
                                                const CylinderObstacle *dynamic_cylinders,
                                                int dynamic_cylinder_count,
                                                float max_ray_length)
    {
        HitResult result;
        float best_hit = max_ray_length + 1.0f;
        int   best_id  = 0;
        for (int i = 0; i < dynamic_cylinder_count; ++i)
        {
            const CylinderObstacle obstacle = dynamic_cylinders[i];
            if (obstacle.radius <= 0.0f || obstacle.height <= 0.0f) continue;
            const float ox = ray_origin_w.x - obstacle.center.x;
            const float oy = ray_origin_w.y - obstacle.center.y;
            const float a = ray_dir_w.x * ray_dir_w.x + ray_dir_w.y * ray_dir_w.y;
            const float b = 2.0f * (ox * ray_dir_w.x + oy * ray_dir_w.y);
            const float c = ox * ox + oy * oy - obstacle.radius * obstacle.radius;
            if (a > 1e-6f)
            {
                const float discriminant = b * b - 4.0f * a * c;
                if (discriminant >= 0.0f)
                {
                    const float sqrt_d = sqrtf(discriminant);
                    updateBestCylinderHit((-b - sqrt_d) / (2.0f * a), ray_origin_w, ray_dir_w, obstacle, max_ray_length, best_hit, best_id);
                    updateBestCylinderHit((-b + sqrt_d) / (2.0f * a), ray_origin_w, ray_dir_w, obstacle, max_ray_length, best_hit, best_id);
                }
            }
            const float z_min = obstacle.center.z - 0.5f * obstacle.height;
            const float z_max = obstacle.center.z + 0.5f * obstacle.height;
            updateBestCylinderCapHit(z_min, ray_origin_w, ray_dir_w, obstacle, max_ray_length, best_hit, best_id);
            updateBestCylinderCapHit(z_max, ray_origin_w, ray_dir_w, obstacle, max_ray_length, best_hit, best_id);
        }
        if (best_hit <= max_ray_length) { result.t = best_hit; result.obj_id = best_id; }
        return result;
    }

    // Pick the nearest of two HitResults (negative t = miss).
    __device__ HitResult nearestPositiveHit(const HitResult &a, const HitResult &b)
    {
        if (a.t > 0.0f && b.t > 0.0f) return (a.t < b.t) ? a : b;
        if (a.t > 0.0f) return a;
        if (b.t > 0.0f) return b;
        return HitResult{};
    }

    GridMap::GridMap(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float resolution, int occupy_threshold = 1){
        const float epsilon = 0.001f;   // avoid numerical error causing (1) empty rows in the map (2) dropped edge points
        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*cloud, min_pt, max_pt);
        float length = max_pt(0) - min_pt(0) + 2 * epsilon;  // make sure the max value on each boundary is reachable
        float width  = max_pt(1) - min_pt(1) + 2 * epsilon;
        float height = max_pt(2) - min_pt(2) + 2 * epsilon;
        Vector3f origin(min_pt(0), min_pt(1), min_pt(2));
        Vector3f map_size(length, width, height);
        origin_x_ = origin.x;
        origin_y_ = origin.y;
        origin_z_ = origin.z;

        Vector3i grid_size;
        grid_size.x = ceil(map_size.x / resolution);
        grid_size.y = ceil(map_size.y / resolution);
        grid_size.z = ceil(map_size.z / resolution);
        int grid_total_size = grid_size.x * grid_size.y * grid_size.z;

        resolution_   = resolution;
        grid_size_x_  = grid_size.x, 
        grid_size_y_  = grid_size.y, 
        grid_size_z_  = grid_size.z, 
        grid_size_yz_ = grid_size.y * grid_size.z;
        occupy_threshold_ = occupy_threshold;
        raycast_step_ = resolution;

        std::vector<int> h_map(grid_total_size, 0);
        // points lying exactly on voxel boundaries can leave empty rows; add a small offset
        for (size_t i = 0; i < cloud->points.size(); i++) {
            Vector3f point(cloud->points[i].x + epsilon, cloud->points[i].y + epsilon, cloud->points[i].z + epsilon);
            int idx = Vox2Idx(Pos2Vox(point));
            if (idx < grid_total_size) {
                h_map[idx]++;
            }
        }
        cudaMalloc((void **)&map_cuda_, grid_total_size * sizeof(int));
        cudaMemcpy(map_cuda_, h_map.data(), grid_total_size * sizeof(int), cudaMemcpyHostToDevice);
    }

    __host__ __device__ Vector3i GridMap::Pos2Vox(const Vector3f &pos)
    {
        Vector3i vox;
        vox.x = floor((pos.x - origin_x_) / resolution_);
        vox.y = floor((pos.y - origin_y_) / resolution_);
        vox.z = floor((pos.z - origin_z_) / resolution_);
        return vox;
    }

    __host__ __device__ Vector3f GridMap::Vox2Pos(const Vector3i &vox)
    {
        Vector3f pos;
        pos.x = (vox.x + 0.5f) * resolution_ + origin_x_;
        pos.y = (vox.y + 0.5f) * resolution_ + origin_y_;
        pos.z = (vox.z + 0.5f) * resolution_ + origin_z_;
        return pos;
    }

    __host__ __device__ int GridMap::Vox2Idx(const Vector3i &vox)
    {
        return vox.x * grid_size_yz_ + vox.y * grid_size_z_ + vox.z;
    }

    __host__ __device__ Vector3i GridMap::Idx2Vox(int idx)
    {
        return Vector3i(idx / grid_size_yz_, (idx % grid_size_yz_) / grid_size_z_, idx % grid_size_z_);
    }

    __device__ int GridMap::symmetricIndex(int index, int length)
    {
        index = index % (2 * length - 2);
        if (index < 0)
        {
            index += (2 * length - 2);
        }

        if (index >= length)
        {
            index = 2 * length - 2 - index;
        }
        return index;
    }

    // 0: free; 1: occupied
    __device__  int GridMap::mapQuery(const Vector3f &pos){
        Vector3i vox = Pos2Vox(pos);
        vox.x = symmetricIndex(vox.x, grid_size_x_);
        vox.y = symmetricIndex(vox.y, grid_size_y_);

        if (vox.z >= grid_size_z_ || vox.z < 0)
            return 0;

        int idx = Vox2Idx(vox);
        if (map_cuda_[idx] > occupy_threshold_)
            return 1;
        return 0;
    }

    __global__ void cameraRaycastKernel(float *depth_values, GridMap grid_map, CameraParams camera_param, cudaMat::SE3<float> T_wc,
                                        const SphereObstacle *dynamic_obstacles, int dynamic_obstacle_count,
                                        const CylinderObstacle *dynamic_cylinders, int dynamic_cylinder_count)
    {
        int u = threadIdx.x;
        int v = blockIdx.x;

        if (u >= camera_param.image_width || v >= camera_param.image_height) return;

        float y = -(u - camera_param.cx) / camera_param.fx;
        float z = -(v - camera_param.cy) / camera_param.fy;
        float x = 1.0f;

        float length = sqrtf(x * x + y * y + z * z);
        x /= length; y /= length; z /= length;

        const float3 ray_dir_c = make_float3(x, y, z);
        const float3 ray_dir_w = T_wc.rotate(ray_dir_c);
        const float3 ray_origin_w = T_wc.getTranslation();

        const float max_ray_length = camera_param.max_depth_dist / fmaxf(x, 1e-4f);
        const HitResult sphere_hit   = findNearestSphereHit(ray_origin_w, ray_dir_w, dynamic_obstacles, dynamic_obstacle_count, max_ray_length);
        const HitResult cylinder_hit = findNearestCylinderHit(ray_origin_w, ray_dir_w, dynamic_cylinders, dynamic_cylinder_count, max_ray_length);
        const HitResult dyn_hit_r    = nearestPositiveHit(sphere_hit, cylinder_hit);
        const float dynamic_hit      = dyn_hit_r.t;
        const float dynamic_depth = dynamic_hit > 0.0f ? dynamic_hit * x : camera_param.max_depth_dist;

        float dx = 0.5f * grid_map.raycast_step_;
        float dy = (y / x) * dx;
        float dz = (z / x) * dx;

        int scale = 0;
        float depth = dynamic_depth;

        while (1)
        {
            scale += 1;
            float point_x = scale * dx;
            float point_y = scale * dy;
            float point_z = scale * dz;

            if (point_x >= camera_param.max_depth_dist) { depth = dynamic_depth; break; }
            if (dynamic_hit > 0.0f && point_x >= dynamic_depth) { depth = dynamic_depth; break; }

            float3 point_c = make_float3(point_x, point_y, point_z);
            float3 point_w = T_wc * point_c;
            Vector3f point(point_w.x, point_w.y, point_w.z);

            if (grid_map.mapQuery(point) == 1)
            {
                Vector3i occ_vox_w = grid_map.Pos2Vox(point);
                Vector3f occ_point_w = grid_map.Vox2Pos(occ_vox_w);
                float3 occ_point_c = T_wc.inv() * make_float3(occ_point_w.x, occ_point_w.y, occ_point_w.z);
                depth = occ_point_c.x;
                break;
            }
        }

        if (camera_param.normalize_depth)
            depth = depth / camera_param.max_depth_dist;
        depth_values[v * camera_param.image_width + u] = depth;
    }

    void renderDepthImage(GridMap *grid_map, CameraParams *camera_param, cudaMat::SE3<float> &T_wc, cv::Mat &depth_image,
                          const std::vector<SphereObstacle> &dynamic_obstacles,
                          const std::vector<CylinderObstacle> &dynamic_cylinders)
    {
        float *depth_values;
        size_t num_elements = camera_param->image_width * camera_param->image_height;
        cudaMallocManaged(&depth_values, num_elements * sizeof(float));

        SphereObstacle *obs_cuda = nullptr;
        if (!dynamic_obstacles.empty())
        {
            cudaMallocManaged(&obs_cuda, dynamic_obstacles.size() * sizeof(SphereObstacle));
            cudaMemcpy(obs_cuda, dynamic_obstacles.data(), dynamic_obstacles.size() * sizeof(SphereObstacle), cudaMemcpyHostToDevice);
        }
        CylinderObstacle *cyl_cuda = nullptr;
        if (!dynamic_cylinders.empty())
        {
            cudaMallocManaged(&cyl_cuda, dynamic_cylinders.size() * sizeof(CylinderObstacle));
            cudaMemcpy(cyl_cuda, dynamic_cylinders.data(), dynamic_cylinders.size() * sizeof(CylinderObstacle), cudaMemcpyHostToDevice);
        }

        cameraRaycastKernel<<<camera_param->image_height, camera_param->image_width>>>(
            depth_values, *grid_map, *camera_param, T_wc,
            obs_cuda, dynamic_obstacles.size(), cyl_cuda, dynamic_cylinders.size());
        cudaDeviceSynchronize();

        depth_image.create(camera_param->image_height, camera_param->image_width, CV_32FC1);
        cudaMemcpy(depth_image.data, depth_values, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

        if (obs_cuda) cudaFree(obs_cuda);
        if (cyl_cuda) cudaFree(cyl_cuda);
        cudaFree(depth_values);
    }

    __global__ void lidarRaycastKernel(Vector3f *point_values, GridMap grid_map, LidarParams lidar_param, cudaMat::SE3<float> T_wc,
                                       const SphereObstacle *dynamic_obstacles, int dynamic_obstacle_count,
                                       const CylinderObstacle *dynamic_cylinders, int dynamic_cylinder_count)
    {
        int h = threadIdx.x;
        int v = blockIdx.x;

        if (h >= lidar_param.horizontal_num || v >= lidar_param.vertical_lines) return;

        const float vertical_resolution = (lidar_param.vertical_angle_end - lidar_param.vertical_angle_start) / (lidar_param.vertical_lines - 1);
        const float vertical_angle   = lidar_param.vertical_angle_start + v * vertical_resolution;
        const float sin_vert = std::sin(vertical_angle * M_PI / 180.0f);
        const float cos_vert = std::cos(vertical_angle * M_PI / 180.0f);
        const float horizontal_angle = h * lidar_param.horizontal_resolution;
        const float sin_horz = std::sin(horizontal_angle * M_PI / 180.0f);
        const float cos_horz = std::cos(horizontal_angle * M_PI / 180.0f);

        const Vector3f ray_dir_local(cos_vert * cos_horz, cos_vert * sin_horz, sin_vert);
        const float3 ray_dir_c    = make_float3(ray_dir_local.x, ray_dir_local.y, ray_dir_local.z);
        const float3 ray_dir_w    = T_wc.rotate(ray_dir_c);
        const float3 ray_origin_w = T_wc.getTranslation();

        const HitResult sphere_hit   = findNearestSphereHit(ray_origin_w, ray_dir_w, dynamic_obstacles, dynamic_obstacle_count, lidar_param.max_lidar_dist);
        const HitResult cylinder_hit = findNearestCylinderHit(ray_origin_w, ray_dir_w, dynamic_cylinders, dynamic_cylinder_count, lidar_param.max_lidar_dist);
        const HitResult dyn_hit_r    = nearestPositiveHit(sphere_hit, cylinder_hit);
        const float dynamic_hit      = dyn_hit_r.t;

        const float dx = ray_dir_local.x * grid_map.raycast_step_;
        const float dy = ray_dir_local.y * grid_map.raycast_step_;
        const float dz = ray_dir_local.z * grid_map.raycast_step_;

        int scale = 0;
        Vector3f point_value(0, 0, 0);

        while (1)
        {
            scale += 1;
            const float point_x = scale * dx;
            const float point_y = scale * dy;
            const float point_z = scale * dz;
            const float ray_length = sqrtf(point_x * point_x + point_y * point_y + point_z * point_z);

            if (ray_length >= lidar_param.max_lidar_dist) break;

            if (dynamic_hit > 0.0f && ray_length >= dynamic_hit)
            {
                point_value = Vector3f(ray_dir_local.x * dynamic_hit,
                                       ray_dir_local.y * dynamic_hit,
                                       ray_dir_local.z * dynamic_hit);
                break;
            }

            float3 point_c = make_float3(point_x, point_y, point_z);
            float3 point_w = T_wc * point_c;
            Vector3f point(point_w.x, point_w.y, point_w.z);

            if (grid_map.mapQuery(point) == 1)
            {
                point_value = Vector3f(point_x, point_y, point_z);
                Vector3i vox_body = grid_map.Pos2Vox(point_value);
                point_value = grid_map.Vox2Pos(vox_body);
                break;
            }
        }

        point_values[v * lidar_param.horizontal_num + h] = point_value;
    }

    void renderLidarPointcloud(GridMap *grid_map, LidarParams *lidar_param, cudaMat::SE3<float> &T_wc, pcl::PointCloud<pcl::PointXYZ> &lidar_points,
                               const std::vector<SphereObstacle> &dynamic_obstacles,
                               const std::vector<CylinderObstacle> &dynamic_cylinders)
    {
        Vector3f *point_values;
        size_t num_elements = lidar_param->vertical_lines * lidar_param->horizontal_num;
        cudaMallocManaged(&point_values, num_elements * sizeof(Vector3f));

        SphereObstacle *obs_cuda = nullptr;
        if (!dynamic_obstacles.empty())
        {
            cudaMallocManaged(&obs_cuda, dynamic_obstacles.size() * sizeof(SphereObstacle));
            cudaMemcpy(obs_cuda, dynamic_obstacles.data(), dynamic_obstacles.size() * sizeof(SphereObstacle), cudaMemcpyHostToDevice);
        }
        CylinderObstacle *cyl_cuda = nullptr;
        if (!dynamic_cylinders.empty())
        {
            cudaMallocManaged(&cyl_cuda, dynamic_cylinders.size() * sizeof(CylinderObstacle));
            cudaMemcpy(cyl_cuda, dynamic_cylinders.data(), dynamic_cylinders.size() * sizeof(CylinderObstacle), cudaMemcpyHostToDevice);
        }

        lidarRaycastKernel<<<lidar_param->vertical_lines, lidar_param->horizontal_num>>>(
            point_values, *grid_map, *lidar_param, T_wc,
            obs_cuda, dynamic_obstacles.size(), cyl_cuda, dynamic_cylinders.size());
        cudaDeviceSynchronize();

        std::vector<Vector3f> cpu_points(num_elements);
        cudaMemcpy(cpu_points.data(), point_values, num_elements * sizeof(Vector3f), cudaMemcpyDeviceToHost);

        lidar_points.points.clear();
        lidar_points.points.reserve(num_elements);
        for (const auto &point : cpu_points)
        {
            if (point.x != 0 || point.y != 0 || point.z != 0)
                lidar_points.points.emplace_back(point.x, point.y, point.z);
        }

        if (obs_cuda) cudaFree(obs_cuda);
        if (cyl_cuda) cudaFree(cyl_cuda);
        cudaFree(point_values);
    }

    // cast a spherical ray per range-image pixel and traverse voxels exactly with 3D DDA
    // layout: row=0 → max elevation (elev_end), row=H-1 → min elevation (elev_start)
    //         col=0 → azimuth -π (rear), col=W/2 → azimuth 0 (front)
    // on hit, range = t where the ray enters the voxel (exact face hit), no voxel-center quantization
    __global__ void rangeImageRaycastKernel(float *range_values, int16_t *mask_values,
                                            GridMap grid_map, LidarParams lidar_param, cudaMat::SE3<float> T_wb,
                                            const SphereObstacle *dynamic_obstacles, int dynamic_obstacle_count,
                                            const CylinderObstacle *dynamic_cylinders, int dynamic_cylinder_count)
    {
        int col = threadIdx.x;
        int row = blockIdx.x;
        if (col >= lidar_param.horizontal_num || row >= lidar_param.vertical_lines) return;

        const float INF = 1e30f;
        const float PI  = 3.14159265358979323846f;

        float H = (float)lidar_param.vertical_lines;
        float W = (float)lidar_param.horizontal_num;
        float elev_max = lidar_param.vertical_angle_end   * PI / 180.0f;
        float elev_min = lidar_param.vertical_angle_start * PI / 180.0f;

        float elev = elev_max - (row / (H - 1.0f)) * (elev_max - elev_min);
        float azim = ((float)col / W) * 2.0f * PI - PI;

        float cos_e = cosf(elev), sin_e = sinf(elev);
        float cos_a = cosf(azim), sin_a = sinf(azim);
        float3 dir_b = make_float3(cos_e * cos_a, cos_e * sin_a, sin_e);  // unit vector in FLU body frame

        // ray origin and direction in world frame
        float3 O = T_wb.getTranslation();
        float3 D = T_wb.rotate(dir_b);

        // nearest dynamic obstacle hit distance + obj_id (for the mask channel)
        const HitResult sphere_hit   = findNearestSphereHit(O, D, dynamic_obstacles, dynamic_obstacle_count, lidar_param.max_lidar_dist);
        const HitResult cylinder_hit = findNearestCylinderHit(O, D, dynamic_cylinders, dynamic_cylinder_count, lidar_param.max_lidar_dist);
        const HitResult dyn_hit_r    = nearestPositiveHit(sphere_hit, cylinder_hit);
        const float dynamic_hit      = dyn_hit_r.t;
        const int   dynamic_obj_id   = dyn_hit_r.obj_id;

        float res = grid_map.getResolution();
        float ox  = grid_map.getOriginX();
        float oy  = grid_map.getOriginY();
        float oz  = grid_map.getOriginZ();

        // voxel containing the ray origin
        int vox_x = (int)floorf((O.x - ox) / res);
        int vox_y = (int)floorf((O.y - oy) / res);
        int vox_z = (int)floorf((O.z - oz) / res);

        // world coordinates of the first voxel wall along each axis
        float bx = ox + (float)(vox_x + (D.x >= 0.0f ? 1 : 0)) * res;
        float by = oy + (float)(vox_y + (D.y >= 0.0f ? 1 : 0)) * res;
        float bz = oz + (float)(vox_z + (D.z >= 0.0f ? 1 : 0)) * res;

        float tMaxX   = (fabsf(D.x) > 1e-9f) ? (bx - O.x) / D.x : INF;
        float tMaxY   = (fabsf(D.y) > 1e-9f) ? (by - O.y) / D.y : INF;
        float tMaxZ   = (fabsf(D.z) > 1e-9f) ? (bz - O.z) / D.z : INF;
        float tDeltaX = (fabsf(D.x) > 1e-9f) ? res / fabsf(D.x) : INF;
        float tDeltaY = (fabsf(D.y) > 1e-9f) ? res / fabsf(D.y) : INF;
        float tDeltaZ = (fabsf(D.z) > 1e-9f) ? res / fabsf(D.z) : INF;

        float range_val = lidar_param.max_lidar_dist;
        int   mask_val  = 0;        // 0 = static / no-hit
        float t = 0.0f;

        while (t < lidar_param.max_lidar_dist)
        {
            // dynamic obstacles take priority
            if (dynamic_hit > 0.0f && t >= dynamic_hit)
            {
                range_val = dynamic_hit;
                mask_val  = dynamic_obj_id;
                break;
            }

            float t_exit = fminf(tMaxX, fminf(tMaxY, tMaxZ));
            float t_mid  = (t + t_exit) * 0.5f;
            float3 p = make_float3(O.x + t_mid * D.x, O.y + t_mid * D.y, O.z + t_mid * D.z);

            if (grid_map.mapQuery(Vector3f(p.x, p.y, p.z)))
            {
                range_val = t;  // exact face-hit distance
                mask_val  = 0;  // static
                break;
            }

            if (tMaxX < tMaxY && tMaxX < tMaxZ) {
                t = tMaxX; tMaxX += tDeltaX;
            } else if (tMaxY < tMaxZ) {
                t = tMaxY; tMaxY += tDeltaY;
            } else {
                t = tMaxZ; tMaxZ += tDeltaZ;
            }
        }

        // If the loop ended via t-bound without hitting either, the dynamic hit
        // (if present) is still beyond the map; honor it.
        if (range_val >= lidar_param.max_lidar_dist && dynamic_hit > 0.0f)
        {
            range_val = dynamic_hit;
            mask_val  = dynamic_obj_id;
        }

        const int idx = row * lidar_param.horizontal_num + col;
        range_values[idx] = range_val;
        if (mask_values) mask_values[idx] = (int16_t)mask_val;
    }

    static void renderRangeImageImpl(GridMap *grid_map, LidarParams *lidar_param,
                                     cudaMat::SE3<float> &T_wb,
                                     cv::Mat &range_image, cv::Mat *mask_image,
                                     const std::vector<SphereObstacle> &dynamic_obstacles,
                                     const std::vector<CylinderObstacle> &dynamic_cylinders)
    {
        float   *range_values = nullptr;
        int16_t *mask_values  = nullptr;
        size_t num_elements = lidar_param->vertical_lines * lidar_param->horizontal_num;
        cudaMallocManaged(&range_values, num_elements * sizeof(float));
        if (mask_image)
            cudaMallocManaged(&mask_values, num_elements * sizeof(int16_t));

        SphereObstacle *obs_cuda = nullptr;
        if (!dynamic_obstacles.empty())
        {
            cudaMallocManaged(&obs_cuda, dynamic_obstacles.size() * sizeof(SphereObstacle));
            cudaMemcpy(obs_cuda, dynamic_obstacles.data(), dynamic_obstacles.size() * sizeof(SphereObstacle), cudaMemcpyHostToDevice);
        }
        CylinderObstacle *cyl_cuda = nullptr;
        if (!dynamic_cylinders.empty())
        {
            cudaMallocManaged(&cyl_cuda, dynamic_cylinders.size() * sizeof(CylinderObstacle));
            cudaMemcpy(cyl_cuda, dynamic_cylinders.data(), dynamic_cylinders.size() * sizeof(CylinderObstacle), cudaMemcpyHostToDevice);
        }

        rangeImageRaycastKernel<<<lidar_param->vertical_lines, lidar_param->horizontal_num>>>(
            range_values, mask_values, *grid_map, *lidar_param, T_wb,
            obs_cuda, dynamic_obstacles.size(), cyl_cuda, dynamic_cylinders.size());
        cudaDeviceSynchronize();

        range_image.create(lidar_param->vertical_lines, lidar_param->horizontal_num, CV_32FC1);
        cudaMemcpy(range_image.data, range_values, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

        if (mask_image)
        {
            mask_image->create(lidar_param->vertical_lines, lidar_param->horizontal_num, CV_16SC1);
            cudaMemcpy(mask_image->data, mask_values, num_elements * sizeof(int16_t), cudaMemcpyDeviceToHost);
        }

        if (obs_cuda) cudaFree(obs_cuda);
        if (cyl_cuda) cudaFree(cyl_cuda);
        cudaFree(range_values);
        if (mask_values) cudaFree(mask_values);
    }

    void renderRangeImage(GridMap *grid_map, LidarParams *lidar_param, cudaMat::SE3<float> &T_wb, cv::Mat &range_image,
                          const std::vector<SphereObstacle> &dynamic_obstacles,
                          const std::vector<CylinderObstacle> &dynamic_cylinders)
    {
        renderRangeImageImpl(grid_map, lidar_param, T_wb, range_image, nullptr,
                             dynamic_obstacles, dynamic_cylinders);
    }

    void renderRangeImage(GridMap *grid_map, LidarParams *lidar_param, cudaMat::SE3<float> &T_wb,
                          cv::Mat &range_image, cv::Mat &mask_image,
                          const std::vector<SphereObstacle> &dynamic_obstacles,
                          const std::vector<CylinderObstacle> &dynamic_cylinders)
    {
        renderRangeImageImpl(grid_map, lidar_param, T_wb, range_image, &mask_image,
                             dynamic_obstacles, dynamic_cylinders);
    }
}