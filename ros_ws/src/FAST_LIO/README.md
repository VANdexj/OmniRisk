
# User Guide
## 1. Unknown environment

```bash
# LiDAR localization and mapping in an unknown environment; saves a map new.pcd that can be used for localization
roslaunch fast_lio mapping_mid360.launch
# Publish the live point cloud to the FY ground station
roslaunch fast_lio fy_series_publisher.launch
```



## 2. Known environment
### 2.1. Rename the ```new.pcd``` built in the previous step to ```origin.pcd```

### 2.2. Run

```bash
# GICP-match the known map origin.pcd against the cloud captured while the LiDAR is static, to get the initial LiDAR pose in the known map
roslaunch fast_lio initial_align.launch
# Localize in the known map origin.pcd (with incremental mapping)
roslaunch fast_lio odom_mid360_with_map.launch
# Add this if the point cloud needs to be streamed to the ground station
roslaunch fast_lio fy_series_publisher.launch
```
### 2.3. Set the initial guess
To take the initial position from environment variables as before, write those values into the matching fields in ```initial_align.launch```:
```bash
	<!-- environment variable! -->
	<param name="wxx/initial_align/Init_Body_pos_x" type="double" value="0.0" />
	<param name="wxx/initial_align/Init_Body_pos_y" type="double" value="0.0" />
	<param name="wxx/initial_align/Init_Body_pos_z" type="double" value="0.0" />
```

# Individual launch files
> Below are the function, inputs and outputs of each package, to help check for miswired topics on first use

> Also listed are the main parameters you may need to tune. For example, if compute is tight (often shown as red ekf errors; check whether the odom output rate matches the LiDAR rate), adjust the compute-related parameters below, e.g. increase ```point_filter_num```. For finer requirements, contact the author


## 1. **mapping_mid360.launch**
> Function:
LiDAR localization and mapping in an **unknown environment**; saves a map new.pcd that can be used for localization

* Inputs:
    * raw livox point cloud: /livox/lidar
    * raw livox IMU: /livox/imu
* Outputs:
    * body (not LiDAR) odometry: /Odometry
    * body (not LiDAR) pose: /PoseStamped 
    * LiDAR point cloud in world frame: /cloud_registered

### Main parameters

* Compute-related (in mapping_mid360.launch)
```xml
<!-- downsample the input cloud by (count % point_filter_num == 0) -->
<param name="point_filter_num" type="int" value="3"/>
<!-- ICP iterations per fastlio update -->
<param name="max_icp_times" type="int" value="2" />
<!-- total iterations per fastlio update (sum over all ICP inner iterations) -->
<param name="max_iteration" type="int" value="4" />
<!-- voxel downsample size of the input cloud -->
<param name="filter_size_surf" type="double" value="0.25" />
<!-- voxel downsample size of the map cloud -->
<param name="filter_size_map" type="double" value="0.5" />
```
* Common options (mid360.yaml)
```xml
<!-- points closer than this (m) are dropped -->
preprocess/blind: 0.5
<!-- odom and clouds from this package are in body frame; this sets the LiDAR pose relative to the body -->
<!-- Lidar_Odom = Lidar_wrt_Body_R*(Body_Odom + Lidar_wrt_Body_T) -->
Lidar_wrt_Body_R:
- 0.0000000
- 0.9661348
- 0.2580377
- -1.0000000
- 0.0000000
- 0.0000000
- 0.0000000
- -0.2580377
- 0.9661348
Lidar_wrt_Body_T:
- 0
- 0
- 0
```
## 2. **initial_align.launch**

> Function: GICP-match the known map origin.pcd against the cloud captured while the LiDAR is static, to get the initial LiDAR pose in the known map


* Inputs:
    * raw livox point cloud: /livox/lidar
    * raw livox IMU (for gravity alignment, which makes matching easier): /livox/imu
    * point cloud map of the known environment: origin.pcd
* Outputs:
    * initial body (not LiDAR) odometry: /initial_odom_for_lio
    * cloud aligned to the known map, i.e. the LiDAR cloud in the known map's world frame: /initial_cloud_from_odom

### Main parameters
```yaml
wxx:
  initial_align:
    voxelgrid_filter_size:  0.2 # downsampling of the map and input cloud
    max_iteration:          10  # GICP iterations
    icp_mode:               4   # ICP variant, 4 tested best
    initial_map_size:       50  # only use map points within this radius

    Init_Body_pos_x: 0.0
    Init_Body_pos_y: 0.0       # initial guess of the position
    Init_Body_pos_z: 0.0

zty:
  advanced_by_scan_context: true    # use Scan Context for a more robust initial guess
  init_zone_width:          10.0    # range the initial guess can match within
  init_zone_height:         10.0
  init_resolution:          1.0     # resolution of sampled initial guesses
  scancontext:
    test_PC_NUM_RING:       40      # SC: number of rings
    test_PC_NUM_SECTOR:     120     # SC: number of sectors
    test_PC_MAX_RADIUS:     20      # SC: max distance
    print_detail_score:     false   # print match scores of all samples against the current frame
```




## 3. **odom_mid360_with_map.launch**
> Function:
Localization (with incremental mapping) in the point cloud map origin.pcd of a **known environment**

> **NOTE**: requires the initial body pose in the known map from ```initial_align.launch``` on topic ```/initial_odom_for_lio```


* Inputs:
    * raw livox point cloud: /livox/lidar
    * raw livox IMU: /livox/imu
* Outputs:
    * body (not LiDAR) odometry: /Odometry
    * body (not LiDAR) pose: /PoseStamped 
    * LiDAR point cloud in world frame: /cloud_registered


### Main parameters

* All parameters of **mapping_mid360.launch** above also apply here

* Other common options (mid360.yaml)
```xml
<!-- set this to false if incremental mapping is not needed at all -->
wxx/map_incremental: true
```
