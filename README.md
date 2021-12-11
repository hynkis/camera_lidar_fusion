# Camera LiDAR Fusion
A ROS package for the Camera LiDAR fusion.
- Synchronizes Multiple Cameras (Front, Left, Right, Compressed images) and LiDAR (Pointcloud) messages using message_filter.
- Fuse Camera and LiDAR using LiDAR-to-Camera transformation matrix.
- Visualize Camera-LiDAR fusion images.

## Install
```bash
sudo apt install ros-melodic-pcl-ros ros-melodic-pcl-conversions 
sudo apt install ros-melodic-image-transport ros-melodic-image-transport-plugins
```

## How to use
### 1. Configure sensors & Update your Camera-LiDAR transformation matrix.
### 2. Build the ROS package & Run ROS node for Camera-LiDAR fusion.
```bash
cd ~/catkin_ws
catkin_make
rosrun camera_lidar_fusion camera_lidar_fusion_sync
```

