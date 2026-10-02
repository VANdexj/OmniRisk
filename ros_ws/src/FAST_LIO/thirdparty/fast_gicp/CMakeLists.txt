cmake_minimum_required(VERSION 3.10.0)
project(fast_gicp)

option(BUILD_VGICP_CUDA "Build GPU-powered VGICP" OFF)
option(BUILD_apps "Build application programs" OFF)
option(BUILD_test "Build test programs" OFF)
option(BUILD_PYTHON_BINDINGS "Build python bindings" OFF)

set(CMAKE_BUILD_TYPE "Release")

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
add_definitions(${PCL_DEFINITIONS})

if(DEFINED ENV{ROS_VERSION})
  set(ROS_VERSION $ENV{ROS_VERSION})
endif()

if(NOT BUILD_PYTHON_BINDINGS)
  if(${ROS_VERSION})
    if(${ROS_VERSION} EQUAL 1)
      find_package(catkin)
    elseif (${ROS_VERSION} EQUAL 2)
      find_package(ament_cmake)
    endif()
  endif()
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(BUILD_VGICP_CUDA)
  find_package(CUDA REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})
  link_directories(${CUDA_LIBRARY_DIRS})
endif()

###################################
## catkin specific configuration ##
###################################
if(catkin_FOUND)
  catkin_package(
    INCLUDE_DIRS include
    LIBRARIES fast_gicp
  )
endif()

###########
## Build ##
###########

add_library(fast_gicp SHARED
  src/fast_gicp/gicp/lsq_registration.cpp
  src/fast_gicp/gicp/fast_gicp.cpp
  src/fast_gicp/gicp/fast_gicp_st.cpp
  src/fast_gicp/gicp/fast_vgicp.cpp
)
target_link_libraries(fast_gicp
  ${PCL_LIBRARIES}
)
if (OPENMP_FOUND)
    if (TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(fast_gicp OpenMP::OpenMP_CXX)
    endif ()
endif ()
target_include_directories(fast_gicp PUBLIC
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

### APPS ###
if(BUILD_apps)
  # add_executable(gicp_align src/align.cpp)
  # add_dependencies(gicp_align fast_gicp)
  # target_link_libraries(gicp_align
  #   ${PCL_LIBRARIES}
  #   fast_gicp
  # )

  # add_executable(gicp_kitti src/kitti.cpp)
  # add_dependencies(gicp_kitti fast_gicp)
  # target_link_libraries(gicp_kitti
  #   ${PCL_LIBRARIES}
  #   fast_gicp
  # )
endif()

### TEST ###
if(BUILD_test)
  find_package(GTest REQUIRED)

  # add_executable(gicp_test src/test/gicp_test.cpp)
  # add_dependencies(gicp_test fast_gicp)
  # target_link_libraries(gicp_test ${GTEST_LIBRARIES} ${PCL_LIBRARIES} fast_gicp)
  # gtest_add_tests(TARGET gicp_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} EXTRA_ARGS "${CMAKE_SOURCE_DIR}/data")
endif()

if(catkin_FOUND)
  ###################################
  ## catkin specific configuration ##
  ###################################
  install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})

  install(DIRECTORY include/
    DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
    FILES_MATCHING PATTERN "*.hpp")
elseif (ament_cmake_FOUND)
  ##################################
  ## ament specific configuration ##
  ##################################
  ament_export_include_directories(include)
  ament_export_libraries(fast_gicp)
  ament_package()

  install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib)

  install(
      DIRECTORY "include/"
      DESTINATION include
    )
endif()
