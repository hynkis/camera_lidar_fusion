cmake_minimum_required(VERSION 3.0.2)
project(camera_lidar_fusion)

find_package(catkin REQUIRED COMPONENTS
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
  message_filters
)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package(
 LIBRARIES coordinate Opencv
 CATKIN_DEPENDS pcl_conversions pcl_ros roscpp sensor_msgs std_msgs cv_bridge image_transport
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(camera_lidar_fusion_sync src/camera_lidar_fusion_sync.cpp)
add_dependencies(camera_lidar_fusion_sync ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(camera_lidar_fusion_sync
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${OpenCV_LIBS}
)