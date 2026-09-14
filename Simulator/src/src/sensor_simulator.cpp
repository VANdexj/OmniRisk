#include "sensor_simulator.h"

// ─── CPU analytic ray-obstacle intersection (ported from sensor_simulator.cu) ───

static float hitSphere(const Eigen::Vector3f& ro, const Eigen::Vector3f& rd,
                       const SimDynObstacle& obs, float max_dist)
{
    const Eigen::Vector3f oc = ro - obs.center;
    const float b = oc.dot(rd);
    const float c = oc.dot(oc) - obs.radius * obs.radius;
    const float disc = b * b - c;
    if (disc < 0.f) return -1.f;
    const float sq = std::sqrt(disc);
    float t = -b - sq;
    if (t <= 0.f) t = -b + sq;
    return (t > 0.f && t <= max_dist) ? t : -1.f;
}

static float hitCylinder(const Eigen::Vector3f& ro, const Eigen::Vector3f& rd,
                          const SimDynObstacle& obs, float max_dist)
{
    float best = max_dist + 1.f;
    const float ox = ro.x() - obs.center.x();
    const float oy = ro.y() - obs.center.y();
    const float a  = rd.x() * rd.x() + rd.y() * rd.y();
    const float b  = 2.f * (ox * rd.x() + oy * rd.y());
    const float c  = ox * ox + oy * oy - obs.radius * obs.radius;
    const float z_lo = obs.center.z() - 0.5f * obs.height;
    const float z_hi = obs.center.z() + 0.5f * obs.height;

    // side intersection (infinite cylinder clipped to its valid height range)
    if (a > 1e-6f) {
        const float disc = b * b - 4.f * a * c;
        if (disc >= 0.f) {
            const float sq = std::sqrt(disc);
            for (float t : {(-b - sq) / (2.f * a), (-b + sq) / (2.f * a)}) {
                if (t <= 0.f || t > max_dist || t >= best) continue;
                const float z = ro.z() + t * rd.z();
                if (z >= z_lo && z <= z_hi) best = t;
            }
        }
    }

    // top/bottom cap intersection
    auto checkCap = [&](float cap_z) {
        if (std::abs(rd.z()) < 1e-5f) return;
        const float t = (cap_z - ro.z()) / rd.z();
        if (t <= 0.f || t > max_dist || t >= best) return;
        const float dx = ro.x() + t * rd.x() - obs.center.x();
        const float dy = ro.y() + t * rd.y() - obs.center.y();
        if (dx * dx + dy * dy <= obs.radius * obs.radius) best = t;
    };
    checkCap(z_lo);
    checkCap(z_hi);

    return (best <= max_dist) ? best : -1.f;
}

// Iterate over all dynamic obstacles and return the nearest hit t (Euclidean distance), or -1 if none
float SensorSimulator::findNearestObstacleHit(const Eigen::Vector3f& ro,
                                               const Eigen::Vector3f& rd,
                                               float max_dist) const
{
    float best = -1.f;
    for (const auto& obs : dynamic_obstacles_) {
        const float t = (obs.height <= 0.f)
            ? hitSphere(ro, rd, obs, max_dist)
            : hitCylinder(ro, rd, obs, max_dist);
        if (t > 0.f && (best < 0.f || t < best))
            best = t;
    }
    return best;
}

// ─── Depth image rendering ───────────────────────────────────────────────────

cv::Mat SensorSimulator::renderDepthImage(){

    cv::Mat depth_image(image_height, image_width, CV_32FC1, cv::Scalar(max_depth_dist));
    const Eigen::Matrix3f R_wc = quat.toRotationMatrix();
    const Eigen::Matrix3f R_cw = R_wc.inverse();

#pragma omp parallel for collapse(2)
    for (int v = 0; v < image_height; ++v) {
        for (int u = 0; u < image_width; ++u) {
            // ray direction in camera frame (not normalized)
            const float y_r = -(static_cast<float>(u) - cx) / fx;
            const float z_r = -(static_cast<float>(v) - cy) / fy;
            const float len  = std::sqrt(1.f + y_r * y_r + z_r * z_r);
            const float x_norm = 1.f / len;  // x component after normalization (for t → depth conversion)

            const Eigen::Vector3f d(x_norm, y_r / len, z_r / len);
            const Eigen::Vector3f ray_dir = R_wc * d;
            const Eigen::Vector3f ray_origin = pos;

            float distance = max_depth_dist;

            // static map: Octree ray intersection
            std::vector<int> pointIdxVec;
            if (octree->getIntersectedVoxelIndices(ray_origin, ray_dir, pointIdxVec, 1)) {
                const Eigen::Vector3f p_world = cloud->points[pointIdxVec[0]].getVector3fMap();
                const Eigen::Vector3f p_cam   = R_cw * (p_world - pos);
                const float d_static = p_cam(0);
                if (d_static > 0.f && d_static < distance)
                    distance = d_static;
            }

            // dynamic obstacles: analytic intersection
            if (sim_dynamic_enabled_ && !dynamic_obstacles_.empty()) {
                // max_ray_t: upper bound on Euclidean ray length (distance = t * x_norm ⟹ t = distance / x_norm)
                const float max_ray_t = distance / x_norm;
                const float obs_t = findNearestObstacleHit(ray_origin, ray_dir, max_ray_t);
                if (obs_t > 0.f) {
                    const float obs_depth = obs_t * x_norm;
                    if (obs_depth < distance) distance = obs_depth;
                }
            }

            if (normalize_depth) distance /= max_depth_dist;
            depth_image.at<float>(v, u) = distance;
        }
    }

    return depth_image;
}

// ─── LiDAR point cloud rendering ─────────────────────────────────────────────

pcl::PointCloud<pcl::PointXYZINormal> SensorSimulator::renderLidarPointcloud() {
    const Eigen::Matrix3f R_wc = quat.toRotationMatrix();
    const Eigen::Matrix3f R_cw = R_wc.inverse();
    pcl::PointCloud<pcl::PointXYZINormal> lidar_points;
    const float vertical_resolution = (vertical_angle_end - vertical_angle_start) / (vertical_lines - 1);
    std::vector<pcl::PointCloud<pcl::PointXYZINormal>> line_clouds(vertical_lines);

#pragma omp parallel for
    for (int v = 0; v < vertical_lines; ++v) {
        const float vertical_angle = vertical_angle_start + v * vertical_resolution;
        const float sin_vert = std::sin(vertical_angle * M_PI / 180.0);
        const float cos_vert = std::cos(vertical_angle * M_PI / 180.0);

        for (int h = 0; h < horizontal_num; ++h) {
            const float horizontal_angle = h * horizontal_resolution;
            const float sin_horz = std::sin(horizontal_angle * M_PI / 180.0);
            const float cos_horz = std::cos(horizontal_angle * M_PI / 180.0);

            // ray direction in body frame (FLU, normalized)
            const Eigen::Vector3f ray_dir_body(cos_vert * cos_horz, cos_vert * sin_horz, sin_vert);
            const Eigen::Vector3f ray_dir_world = R_wc * ray_dir_body;

            float hit_dist = max_lidar_dist;
            Eigen::Vector3f hit_body = Eigen::Vector3f::Zero();
            bool has_hit = false;

            // static map: Octree intersection
            std::vector<int> pointIdxVec;
            if (octree->getIntersectedVoxelIndices(pos, ray_dir_world, pointIdxVec, 1)) {
                const Eigen::Vector3f p_world = cloud->points[pointIdxVec[0]].getVector3fMap();
                const Eigen::Vector3f p_body  = R_cw * (p_world - pos);
                const float d = p_body.norm();
                if (d < max_lidar_dist) {
                    hit_dist  = d;
                    hit_body  = p_body;
                    has_hit   = true;
                }
            }

            // dynamic obstacles: analytic intersection, bounded by the current static hit distance
            if (sim_dynamic_enabled_ && !dynamic_obstacles_.empty()) {
                const float obs_t = findNearestObstacleHit(pos, ray_dir_world, hit_dist);
                if (obs_t > 0.f && obs_t < hit_dist) {
                    hit_body = obs_t * ray_dir_body;
                    has_hit  = true;
                }
            }

            if (has_hit) {
                pcl::PointXYZINormal p;
                p.x = hit_body.x();
                p.y = hit_body.y();
                p.z = hit_body.z();
                p.intensity = 0.f;
                p.normal_x = p.normal_y = p.normal_z = 0.f;
                p.curvature = 0.f;
                line_clouds[v].points.push_back(p);
            }
        }
    }

    for (int i = 0; i < vertical_lines; ++i)
        lidar_points += line_clouds[i];
    lidar_points.width    = lidar_points.points.size();
    lidar_points.height   = 1;
    lidar_points.is_dense = true;
    return lidar_points;
}

// ─── Dynamic obstacle initialization ─────────────────────────────────────────

void SensorSimulator::initDynamicObstacles(const YAML::Node& config)
{
    if (!config["sim_dynamic"] || !config["sim_dynamic"]["enabled"].as<bool>())
        return;
    sim_dynamic_enabled_ = true;
    const auto& n = config["sim_dynamic"];

    // compute world bounds from the point cloud
    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(*cloud, min_pt, max_pt);
    world_min_ = Eigen::Vector3f(min_pt.x, min_pt.y, min_pt.z);
    world_max_ = Eigen::Vector3f(max_pt.x, max_pt.y, max_pt.z);

    const float z_lo = n["z_min"] ? n["z_min"].as<float>() : 0.5f;
    const float z_hi = n["z_max"] ? n["z_max"].as<float>() : 4.0f;
    dyn_z_lo_ = z_lo;
    dyn_z_hi_ = z_hi;
    const float margin = 1.5f;  // min distance to the boundary

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> rx(world_min_.x() + margin, world_max_.x() - margin);
    std::uniform_real_distribution<float> ry(world_min_.y() + margin, world_max_.y() - margin);
    std::uniform_real_distribution<float> rz(z_lo, z_hi);
    std::uniform_real_distribution<float> rangle(0.f, 2.f * static_cast<float>(M_PI));

    // spherical obstacles
    const int n_spheres     = n["sphere_count"]    ? n["sphere_count"].as<int>()     : 0;
    const float sph_radius  = n["sphere_radius"]   ? n["sphere_radius"].as<float>()  : 0.25f;
    const float sph_spd_min = n["sphere_speed_min"]? n["sphere_speed_min"].as<float>(): 0.5f;
    const float sph_spd_max = n["sphere_speed_max"]? n["sphere_speed_max"].as<float>(): 2.0f;
    std::uniform_real_distribution<float> rsph_spd(sph_spd_min, sph_spd_max);
    std::uniform_real_distribution<float> relev(-0.3f, 0.3f);  // small elevation so balls move roughly horizontally

    for (int i = 0; i < n_spheres; ++i) {
        SimDynObstacle obs;
        obs.radius = sph_radius;
        obs.height = 0.f;
        obs.center = Eigen::Vector3f(rx(rng), ry(rng), rz(rng));
        const float spd = rsph_spd(rng);
        const float az  = rangle(rng);
        const float el  = relev(rng);
        obs.velocity = spd * Eigen::Vector3f(
            std::cos(el) * std::cos(az),
            std::cos(el) * std::sin(az),
            std::sin(el));
        dynamic_obstacles_.push_back(obs);
    }

    // cylinder obstacles
    const int   n_cyls    = n["cylinder_count"]       ? n["cylinder_count"].as<int>()        : 0;
    const float cyl_r_min = n["cylinder_radius_min"]  ? n["cylinder_radius_min"].as<float>() : 0.25f;
    const float cyl_r_max = n["cylinder_radius_max"]  ? n["cylinder_radius_max"].as<float>() : 0.50f;
    const float cyl_h_min = n["cylinder_height_min"]  ? n["cylinder_height_min"].as<float>() : 1.0f;
    const float cyl_h_max = n["cylinder_height_max"]  ? n["cylinder_height_max"].as<float>() : 2.5f;
    const float cyl_s_min = n["cylinder_speed_min"]   ? n["cylinder_speed_min"].as<float>()  : 0.3f;
    const float cyl_s_max = n["cylinder_speed_max"]   ? n["cylinder_speed_max"].as<float>()  : 1.5f;
    std::uniform_real_distribution<float> rcyl_r(cyl_r_min, cyl_r_max);
    std::uniform_real_distribution<float> rcyl_h(cyl_h_min, cyl_h_max);
    std::uniform_real_distribution<float> rcyl_s(cyl_s_min, cyl_s_max);

    for (int i = 0; i < n_cyls; ++i) {
        SimDynObstacle obs;
        obs.radius = rcyl_r(rng);
        obs.height = rcyl_h(rng);
        // cylinder center Z must keep both caps within the valid range
        const float cyl_z_lo = std::max(z_lo, world_min_.z() + 0.5f * obs.height);
        const float cyl_z_hi = std::min(z_hi, world_max_.z() - 0.5f * obs.height);
        const float cz = (cyl_z_lo < cyl_z_hi)
            ? std::uniform_real_distribution<float>(cyl_z_lo, cyl_z_hi)(rng)
            : (cyl_z_lo + cyl_z_hi) * 0.5f;
        obs.center = Eigen::Vector3f(rx(rng), ry(rng), cz);
        const float spd = rcyl_s(rng);
        const float az  = rangle(rng);
        obs.velocity = spd * Eigen::Vector3f(std::cos(az), std::sin(az), 0.f);  // cylinders move horizontally only
        dynamic_obstacles_.push_back(obs);
    }

    ROS_INFO("[sim_dynamic] enabled: %d spheres, %d cylinders | world [%.1f~%.1f, %.1f~%.1f, %.1f~%.1f]",
             n_spheres, n_cyls,
             world_min_.x(), world_max_.x(),
             world_min_.y(), world_max_.y(),
             world_min_.z(), world_max_.z());
}

// ─── Dynamic obstacle position update ────────────────────────────────────────

void SensorSimulator::timerObstacleCallback(const ros::TimerEvent& event)
{
    if (!sim_dynamic_enabled_ || dynamic_obstacles_.empty()) return;

    const double now  = event.current_real.toSec();
    const double last = event.last_real.toSec();
    const float dt = static_cast<float>(std::min(now - last, 0.1));  // at most 100ms, avoids a large jump on the first frame

    for (auto& obs : dynamic_obstacles_) {
        obs.center += obs.velocity * dt;

        // bounce off the bounds (axis-aligned): sphere z is limited to the configured dyn_z_lo_/dyn_z_hi_, cylinders use world bounds
        const bool is_sphere = (obs.height == 0.f);
        const float z_margin = is_sphere ? obs.radius : 0.5f * obs.height;
        const float z_lo = is_sphere ? (dyn_z_lo_ + obs.radius) : (world_min_.z() + z_margin);
        const float z_hi = is_sphere ? (dyn_z_hi_ - obs.radius) : (world_max_.z() - z_margin);
        const float lo[3] = { world_min_.x() + obs.radius,
                               world_min_.y() + obs.radius,
                               z_lo };
        const float hi[3] = { world_max_.x() - obs.radius,
                               world_max_.y() - obs.radius,
                               z_hi };

        for (int i = 0; i < 3; ++i) {
            if (lo[i] >= hi[i]) continue;  // map too small, skip this axis
            if (obs.center[i] < lo[i]) { obs.center[i] = lo[i]; obs.velocity[i] =  std::abs(obs.velocity[i]); }
            if (obs.center[i] > hi[i]) { obs.center[i] = hi[i]; obs.velocity[i] = -std::abs(obs.velocity[i]); }
        }
    }

    publishObstacleMarkers();
}

// ─── RViz visualization (Marker) ─────────────────────────────────────────────

void SensorSimulator::publishStaticMarkers()
{
    visualization_msgs::MarkerArray arr;
    for (int i = 0; i < static_cast<int>(static_poles_.size()); ++i) {
        const auto& p = static_poles_[i];
        visualization_msgs::Marker m;
        m.header.frame_id = "world";
        m.header.stamp    = ros::Time::now();
        m.ns              = "static_poles";
        m.id              = i;
        m.action          = visualization_msgs::Marker::ADD;
        m.type            = visualization_msgs::Marker::CYLINDER;
        m.pose.position.x = p.cx;
        m.pose.position.y = p.cy;
        m.pose.position.z = p.height * 0.5f;
        m.pose.orientation.w = 1.0;
        m.scale.x = m.scale.y = 2.f * p.r;
        m.scale.z = p.height;
        m.color.r = 0.55f; m.color.g = 0.55f; m.color.b = 0.55f; m.color.a = 1.0f;
        arr.markers.push_back(m);
    }
    static_marker_pub_.publish(arr);
}

void SensorSimulator::publishObstacleMarkers()
{
    visualization_msgs::MarkerArray arr;
    const ros::Time now = ros::Time::now();

    for (int idx = 0; idx < static_cast<int>(dynamic_obstacles_.size()); ++idx) {
        const auto& obs = dynamic_obstacles_[idx];
        visualization_msgs::Marker m;
        m.header.frame_id = "world";
        m.header.stamp    = now;
        m.ns              = "dyn_obs";
        m.id              = idx;
        m.action          = visualization_msgs::Marker::ADD;
        m.pose.position.x = obs.center.x();
        m.pose.position.y = obs.center.y();
        m.pose.position.z = obs.center.z();
        m.pose.orientation.w = 1.0;
        m.color.a = 0.75f;

        if (obs.height <= 0.f) {
            m.type = visualization_msgs::Marker::SPHERE;
            m.scale.x = m.scale.y = m.scale.z = 2.f * obs.radius;
            m.color.r = 1.0f; m.color.g = 0.3f; m.color.b = 0.1f;
        } else {
            m.type = visualization_msgs::Marker::CYLINDER;
            m.scale.x = m.scale.y = 2.f * obs.radius;
            m.scale.z = obs.height;
            m.color.r = 0.1f; m.color.g = 0.4f; m.color.b = 1.0f;
        }
        arr.markers.push_back(m);
    }

    marker_pub_.publish(arr);
}

// ─── ROS timer callbacks ─────────────────────────────────────────────────────

void SensorSimulator::timerDepthCallback(const ros::TimerEvent&) {
    if (!odom_init || !render_depth)
        return;
    cv::Mat depth_image = renderDepthImage();
    sensor_msgs::Image ros_image;
    cv_bridge::CvImage cv_image;
    cv_image.header.stamp = ros::Time::now();
    cv_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    cv_image.image = depth_image;
    cv_image.toImageMsg(ros_image);
    image_pub_.publish(ros_image);
}

void SensorSimulator::timerLidarCallback(const ros::TimerEvent&) {
    if (!odom_init || !render_lidar)
        return;
    pcl::PointCloud<pcl::PointXYZINormal> lidar_points = renderLidarPointcloud();
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(lidar_points, output);
    output.header.stamp = ros::Time::now();
    output.header.frame_id = "odom";
    point_cloud_pub_.publish(output);
}

void SensorSimulator::odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    quat.x() = msg->pose.pose.orientation.x;
    quat.y() = msg->pose.pose.orientation.y;
    quat.z() = msg->pose.pose.orientation.z;
    quat.w() = msg->pose.pose.orientation.w;

    pos.x() = msg->pose.pose.position.x;
    pos.y() = msg->pose.pose.position.y;
    pos.z() = msg->pose.pose.position.z;

    odom_init = true;
}

// ─── Point cloud expansion (original logic, unchanged) ───────────────────────

void SensorSimulator::expand_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr expanded_cloud, int direction) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZ>());
    *cloud_temp = *expanded_cloud;
    pcl::PointXYZ min_point, max_point;
    pcl::getMinMax3D(*expanded_cloud, min_point, max_point);

    float min_value = (direction == 0) ? min_point.x : min_point.y;
    float max_value = (direction == 0) ? max_point.x : max_point.y;

    for (const auto& point : cloud_temp->points) {
        pcl::PointXYZ mirrored_point = point;
        if (direction == 0) {
            mirrored_point.x = 2 * min_value - point.x;
        } else {
            mirrored_point.y = 2 * min_value - point.y;
        }
        expanded_cloud->push_back(mirrored_point);
    }

    float offset = max_value - min_value;
    for (auto& point : expanded_cloud->points) {
        if (direction == 0) {
            point.x += offset;
        } else {
            point.y += offset;
        }
    }
}
