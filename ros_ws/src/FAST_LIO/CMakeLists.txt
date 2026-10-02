cmake_minimum_required(VERSION 2.8.3)
project(fast_lio)

# Add third-party directory
add_subdirectory(thirdparty/fast_gicp)

# Set build type and C++ standard
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -pthread -fexceptions")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")

# Define ROOT_DIR macro
add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

# Processor core count and multi-processing setup
include(ProcessorCount)
ProcessorCount(N)
message("Processor number: ${N}")
if(N GREATER 4)
  add_definitions(-DMP_EN -DMP_PROC_NUM=3)
  message("Core for MP: 3")
elseif(N GREATER 3)
  add_definitions(-DMP_EN -DMP_PROC_NUM=2)
  message("Core for MP: 2")
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

# Find required packages
find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

find_package(PythonLibs REQUIRED)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  livox_ros_driver2
  message_generation
  eigen_conversions
  cv_bridge
)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(PCL 1.8 REQUIRED)

# Print Eigen directory
message("Eigen: ${EIGEN3_INCLUDE_DIR}")

# Include directories
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  include
  ${PROJECT_BINARY_DIR}/devel/include # Make sure to include generated message headers
)

# Add message files
add_message_files(
  FILES
  Pose6D.msg
  CPUUsage.msg
  LioDebug.msg
)

# Generate message headers
generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

# Catkin package definition
catkin_package(
  CATKIN_DEPENDS livox_ros_driver2 geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime
  DEPENDS EIGEN3 PCL BOOST
  INCLUDE_DIRS include
)

# Add executables and their dependencies
add_executable(fastlio_mapping src/laserMapping.cpp include/ikd-Tree/ikd_Tree.cpp src/preprocess.cpp)
add_dependencies(fastlio_mapping ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(fastlio_mapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES})
target_include_directories(fastlio_mapping PRIVATE ${PYTHON_INCLUDE_DIRS})

add_executable(initial_map_publisher src/initial_map_publisher.cpp)
add_dependencies(initial_map_publisher ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(initial_map_publisher ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(initial_align src/initial_align.cpp src/preprocess.cpp include/scancontext/Scancontext.cpp)
add_dependencies(initial_align fast_gicp ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(initial_align ${catkin_LIBRARIES} ${PCL_LIBRARIES} fast_gicp)

add_executable(cpu_monitor src/cpu_monitor.cpp)
add_dependencies(cpu_monitor ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(cpu_monitor ${catkin_LIBRARIES})

# Optional executable (commented out)
# add_executable(fy_series_publisher src/fy_series_publisher.cpp)
# target_link_libraries(fy_series_publisher ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${Boost_LIBRARIES})
