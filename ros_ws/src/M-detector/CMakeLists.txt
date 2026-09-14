cmake_minimum_required(VERSION 3.5)
project(m_detector)

SET(CMAKE_BUILD_TYPE "Release")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


ADD_COMPILE_OPTIONS(-std=c++17 )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3" )

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions")

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")
  # if(N GREATER 15)
  #   add_definitions(-DMP_EN)
  #   add_definitions(-DMP_PROC_NUM=14)
  #   message("core for MP:  4")
  # elseif(N GREATER 5)
  if(N GREATER 5)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=4)
    message("core for MP:  3")
  elseif(N GREATER 3)
    math(EXPR PROC_NUM "${N} - 2")
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM="${PROC_NUM}")
    message("core for MP:  ${PROC_NUM}")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)

function(target_enable_openmp target_name)
  if(OpenMP_CXX_FOUND)
    if(TARGET OpenMP::OpenMP_CXX)
      target_link_libraries(${target_name} OpenMP::OpenMP_CXX)
    else()
      target_compile_options(${target_name} PRIVATE ${OpenMP_CXX_FLAGS})
      target_link_libraries(${target_name} ${OpenMP_CXX_FLAGS})
    endif()
  endif()
endfunction()

find_package(PythonLibs REQUIRED)
# find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

find_package(TBB REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_filters
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  visualization_msgs
  # livox_ros_driver
  message_generation
  eigen_conversions
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(OpenCV REQUIRED)

message(Eigen: ${EIGEN3_INCLUDE_DIR})

include_directories(
	${catkin_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  include)

add_message_files(
  FILES
  PredictedTrackedObjectArray.msg
  TrackedObject.msg
  TrackedObjectArray.msg
)

generate_messages(
 DEPENDENCIES
 geometry_msgs
 sensor_msgs
 std_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs message_filters nav_msgs roscpp rospy sensor_msgs std_msgs visualization_msgs message_runtime
  DEPENDS EIGEN3 PCL
  INCLUDE_DIRS
)


add_library(frame_tracker_core src/FrameTracker.cpp)
add_dependencies(frame_tracker_core ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(frame_tracker_core ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(frame_tracker_test test/frame_tracker_test.cpp)
  if(TARGET frame_tracker_test)
    add_dependencies(frame_tracker_test ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
    target_link_libraries(frame_tracker_test frame_tracker_core gtest_main ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})
  endif()

  catkin_add_gtest(bounded_frame_queue_test test/bounded_frame_queue_test.cpp)
  if(TARGET bounded_frame_queue_test)
    target_link_libraries(bounded_frame_queue_test gtest_main ${catkin_LIBRARIES})
  endif()

  catkin_add_gtest(cluster_acceptance_test test/cluster_acceptance_test.cpp)
  if(TARGET cluster_acceptance_test)
    target_link_libraries(cluster_acceptance_test gtest_main ${catkin_LIBRARIES})
  endif()
endif()

add_executable(dynfilter src/dynfilter_with_odom.cpp src/DynObjFilter.cpp src/DynObjCluster.cpp src/FrameTrackerRosPublisher.cpp)
add_dependencies(dynfilter ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(dynfilter frame_tracker_core ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} ${OpenCV_LIBS} TBB::tbb)
target_include_directories(dynfilter PRIVATE ${PYTHON_INCLUDE_DIRS})
target_enable_openmp(dynfilter)

add_executable(display_prediction src/display_prediction.cpp )
target_link_libraries(display_prediction ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} ${OpenCV_LIBS} TBB::tbb)
target_include_directories(display_prediction PRIVATE ${PYTHON_INCLUDE_DIRS})


add_executable(cal_recall src/cal_recall_multi.cpp)
target_link_libraries(cal_recall ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} ${OpenCV_LIBS} TBB::tbb)
target_include_directories(cal_recall PRIVATE ${PYTHON_INCLUDE_DIRS})
target_enable_openmp(cal_recall)
