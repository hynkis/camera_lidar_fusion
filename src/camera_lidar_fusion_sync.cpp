#include <iostream>

#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/highgui/highgui.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>

using namespace std;
using namespace ros;

double ORIGINAL_ROW = 960; // 1920
double ORIGINAL_COL = 604; // 1208
double TARGET_ROW = 960; // 503
double TARGET_COL = 604; // 800
double ratio_row = TARGET_ROW / ORIGINAL_ROW;
double ratio_col = TARGET_COL / ORIGINAL_COL;

cv::Mat lidar_to_camera_center, lidar_to_camera_left, lidar_to_camera_right;
int r = 0;
int g = 0;
int b = 0;
Eigen::Matrix4f trans_center, trans_left, trans_right;
std::vector<float> vec_center, vec_left, vec_right;
        
void ImagePointSyncCallback(const sensor_msgs::CompressedImageConstPtr& msg_img_center, const sensor_msgs::CompressedImageConstPtr& msg_img_left, const sensor_msgs::CompressedImageConstPtr& msg_img_right, const sensor_msgs::PointCloud2ConstPtr& msg_points)
{
    // Get CV Image (from compressed image message)
    cv::Mat img_center = cv::imdecode(cv::Mat(msg_img_center->data), 1); //convert compressed image data to cv::Mat
    cv::Mat img_left   = cv::imdecode(cv::Mat(msg_img_left->data), 1);
    cv::Mat img_right  = cv::imdecode(cv::Mat(msg_img_right->data), 1);
    img_center.copyTo(lidar_to_camera_center);
    img_left.copyTo(lidar_to_camera_left);
    img_right.copyTo(lidar_to_camera_right);
    
    // Get Pointcloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*msg_points, *cloud);

    pcl::PointCloud<pcl::PointXYZI>::Ptr ptr_transformed_center(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr ptr_transformed_left(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr ptr_transformed_right(new pcl::PointCloud<pcl::PointXYZI>);

    pcl::transformPointCloud(*cloud, *ptr_transformed_center, trans_center);
    pcl::transformPointCloud(*cloud, *ptr_transformed_left, trans_left);
    pcl::transformPointCloud(*cloud, *ptr_transformed_right, trans_right);

    // Iteration w.r.t. points (for Visualization of the Camera-LiDAR fusion)
    for (auto &point : cloud->points)
    {
        pcl::PointXYZI point_buf_center, point_buf_left, point_buf_right;
        
        // Depth for coloring
        double depth = sqrt(pow(point.x, 2.0) + pow(point.y, 2));
        int depth_to_color = (1 - std::min(depth, 50.0) / 50.0) * 255.0;
        g = depth_to_color;

        // for center cam + lidar
        point_buf_center.x = ((vec_center.at(0) * point.x + vec_center.at(1) * point.y + vec_center.at(2) * point.z + vec_center.at(3)) / (vec_center.at(8) * point.x + vec_center.at(9) * point.y + vec_center.at(10) * point.z + vec_center.at(11)));
        point_buf_center.y = ((vec_center.at(4) * point.x + vec_center.at(5) * point.y + vec_center.at(6) * point.z + vec_center.at(7)) / (vec_center.at(8) * point.x + vec_center.at(9) * point.y + vec_center.at(10) * point.z + vec_center.at(11)));
        point_buf_center.z = (vec_center.at(8) * point.x + vec_center.at(9) * point.y + vec_center.at(10) * point.z + vec_center.at(11));
        point_buf_center.x *= ratio_col;
        point_buf_center.y *= ratio_row;
        circle(lidar_to_camera_center, cv::Point(point_buf_center.x, point_buf_center.y), 2, cv::Scalar(b, g, r), -1, 8, 0);

        // for left cam + lidar
        point_buf_left.x = ((vec_left.at(0) * point.x + vec_left.at(1) * point.y + vec_left.at(2) * point.z + vec_left.at(3)) / (vec_left.at(8) * point.x + vec_left.at(9) * point.y + vec_left.at(10) * point.z + vec_left.at(11)));
        point_buf_left.y = ((vec_left.at(4) * point.x + vec_left.at(5) * point.y + vec_left.at(6) * point.z + vec_left.at(7)) / (vec_left.at(8) * point.x + vec_left.at(9) * point.y + vec_left.at(10) * point.z + vec_left.at(11)));
        point_buf_left.z = (vec_left.at(8) * point.x + vec_left.at(9) * point.y + vec_left.at(10) * point.z + vec_left.at(11));
        point_buf_left.x *= ratio_col;
        point_buf_left.y *= ratio_row;
        circle(lidar_to_camera_left, cv::Point(point_buf_left.x, point_buf_left.y), 2, cv::Scalar(b, g, r), -1, 8, 0);

        // for right cam + lidar
        point_buf_right.x = ((vec_right.at(0) * point.x + vec_right.at(1) * point.y + vec_right.at(2) * point.z + vec_right.at(3)) / (vec_right.at(8) * point.x + vec_right.at(9) * point.y + vec_right.at(10) * point.z + vec_right.at(11)));
        point_buf_right.y = ((vec_right.at(4) * point.x + vec_right.at(5) * point.y + vec_right.at(6) * point.z + vec_right.at(7)) / (vec_right.at(8) * point.x + vec_right.at(9) * point.y + vec_right.at(10) * point.z + vec_right.at(11)));
        point_buf_right.z = (vec_right.at(8) * point.x + vec_right.at(9) * point.y + vec_right.at(10) * point.z + vec_right.at(11));
        point_buf_right.x *= ratio_col;
        point_buf_right.y *= ratio_row;
        circle(lidar_to_camera_right, cv::Point(point_buf_right.x, point_buf_right.y), 2, cv::Scalar(b, g, r), -1, 8, 0);
    }

    // View visualization image
    if (lidar_to_camera_center.empty() || lidar_to_camera_left.empty() || lidar_to_camera_right.empty())
    {
        std::cout << "image is empty" << std::endl;
    }
    else
    {
        cv::imshow("cam_center", lidar_to_camera_center);
        cv::imshow("cam_left", lidar_to_camera_left);
        cv::imshow("cam_right", lidar_to_camera_right);
        cv::waitKey(1);
    }
}

int main(int argc, char** argv){
    ros::init (argc, argv, "LidarCameraCalibration");
    ros::NodeHandle _nh;
    printf("Initiate: LidarCameraCalibration\n");

    // for center cam (4x4 LiDAR-to-Camera transformation matrix)
    trans_center << -0.197307, 0.393827, -0.00539561, -0.442059,
                    -0.159261, 0.00514399, 0.427143, -0.634573,
                    -0.000410322, -1.00294e-05, 1.87862e-05, -0.000903967,
                    0, 0, 0, 1;
    vec_center = {-0.197307, 0.393827, -0.00539561, -0.442059,
                  -0.159261, 0.00514399, 0.427143, -0.634573,
                  -0.000410322, -1.00294e-05, 1.87862e-05, -0.000903967,
                  0, 0, 0, 1};;

    // for left cam (4x4 LiDAR-to-Camera transformation matrix)
    trans_left << 0.349557, -0.162533, 0.00420289, 0.666721,
                  0.0997905, 0.0614498, -0.361939, 0.51198,
                  0.000273125, 0.000207988, -1.23612e-05, 0.000656519,   
                  0, 0, 0, 1;
    vec_left = {0.349557, -0.162533, 0.00420289, 0.666721,
                0.0997905, 0.0614498, -0.361939, 0.51198,
                0.000273125, 0.000207988, -1.23612e-05, 0.000656519, 
                0, 0, 0, 1};

    // for right cam (4x4 LiDAR-to-Camera transformation matrix)
    trans_right << -0.0885233, -0.517636, -0.0176861, -0.13794,
                    0.163749, -0.0916227, -0.511527, 0.638755,
                    0.000421721, -0.000293506, -9.07933e-06, 0.00073718,  
                    0, 0, 0, 1;
    vec_right = {-0.0885233, -0.517636, -0.0176861, -0.13794,
                  0.163749, -0.0916227, -0.511527, 0.638755,
                  0.000421721, -0.000293506, -9.07933e-06, 0.00073718,
                  0, 0, 0, 1};

    ros::Rate loop_rate(30);

    message_filters::Subscriber<sensor_msgs::CompressedImage> compressed_image_center_sub(_nh, "/gmsl_camera1/compressed", 1);
    message_filters::Subscriber<sensor_msgs::CompressedImage> compressed_image_left_sub(_nh, "/gmsl_camera3/compressed", 1);
    message_filters::Subscriber<sensor_msgs::CompressedImage> compressed_image_right_sub(_nh, "/gmsl_camera2/compressed", 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> point_cloud_sub(_nh, "/velodyne_points", 1);

    // for synchronized message subscribtion
    typedef message_filters::sync_policies::ApproximateTime <sensor_msgs::CompressedImage, sensor_msgs::CompressedImage, sensor_msgs::CompressedImage, sensor_msgs::PointCloud2> MySyncPolicy;
    message_filters::Synchronizer <MySyncPolicy> sync(MySyncPolicy(10), compressed_image_center_sub, compressed_image_left_sub, compressed_image_right_sub, point_cloud_sub);
    sync.registerCallback(boost::bind(&ImagePointSyncCallback, _1, _2, _3, _4));
    
    while(ros::ok())
    {
      ros::spinOnce();
      loop_rate.sleep();
    }
    printf("Terminate: MinCut_Segmentation\n");

    return 0;
}
