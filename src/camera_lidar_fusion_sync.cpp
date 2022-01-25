#include <iostream>
#include <time.h>
#include <ctime>

#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include "eurecar_msgs/px2ObjDetection.h"
#include "eurecar_msgs/px2LaneDetection.h"
#include "eurecar_msgs/px2SingleLaneDetection.h"
#include "eurecar_msgs/px2DetectionResult.h"
#include <detection_msgs/TrackingObjectArray.h>
#include <detection_msgs/BoundingBoxArray.h>

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

int NUM_CAMERAS = 3;
double DW_SCORE_THRESHOLD = 0.95; // threshold for driveworks detection score
double DW_SIZE_THRES = 20; // threshold for driveworks detection size
double ORIGINAL_ROW = 960; // 1920
double ORIGINAL_COL = 604; // 1208
double TARGET_ROW = 960; // 503
double TARGET_COL = 604; // 800
double ratio_row = TARGET_ROW / ORIGINAL_ROW;
double ratio_col = TARGET_COL / ORIGINAL_COL;

// Publisher & Subscriber
ros::Publisher pubDWResultsBBoxArray; // detection_msgs::BoundingBoxArray

// Data structure for driveworks results
struct PointState
{
    // uvw values in Pixel coordinates
    double u; // pixel_x
    double v; // pixel_y
    double w; // depth
    // xyz values in XYZ coordinates
    double x;
    double y;
    double z;
};

// Detection results msgs (Pixel coordinate)
std::vector<eurecar_msgs::px2ObjDetection> px2ObjDetections; // [center,left,right]
// Detection results (XY coordinate)
std::vector<PointState> dw_results_XYZ_center;

// Transformation Matrix (LiDAR to Camera)
Eigen::Matrix4f trans_center, trans_left, trans_right;
std::vector<float> vec_center, vec_left, vec_right;

// Variables
int r = 0;
int g = 0;
int b = 0;
cv::Mat lidar_to_camera_center, lidar_to_camera_left, lidar_to_camera_right;
bool bGetDWResultsCenter = false;
bool bGetDWResultsLeft = false;
bool bGetDWResultsRight = false;

// Messages
detection_msgs::BoundingBoxArray bboxes_result_states_center;
detection_msgs::BoundingBoxArray bboxes_result_states_left;
detection_msgs::BoundingBoxArray bboxes_result_states_right;

std::vector<double> median_function(vector<double> &v)
{
    // Reference: https://stackoverflow.com/questions/1719070/what-is-the-right-approach-when-using-stl-container-for-median-calculation/1719155#1719155
    
    // find median index
    size_t n = v.size() / 2;
    nth_element(v.begin(), v.begin()+n, v.end());
    
    // return median_index and median_value 
    std::vector<double> output;
    output.push_back(n);
    output.push_back(v[n]);

    return output;
}

detection_msgs::BoundingBox setBbox3D(double point_x, double point_y, double point_z, int label_id)
{
    detection_msgs::BoundingBox bbox3D;
    bbox3D.header.frame_id = "velodyne";
    bbox3D.header.stamp = ros::Time::now();
    bbox3D.label = label_id;
    bbox3D.dimensions.x = 3.0; // car dimensions
    bbox3D.dimensions.y = 2.0; //
    bbox3D.dimensions.z = 1.7; //
    bbox3D.pose.position.x = point_x;
    bbox3D.pose.position.y = point_y;
    bbox3D.pose.position.z = point_z;
    bbox3D.pose.orientation.x = 0;
    bbox3D.pose.orientation.y = 0;
    bbox3D.pose.orientation.z = 0;
    bbox3D.pose.orientation.w = 1;
    return bbox3D;
}

// Callback for Driveworks detection results (center/left/right)
void callbackDWDetection(const eurecar_msgs::px2DetectionResult::ConstPtr& msg, int cam_idx)
{
    px2ObjDetections[cam_idx - 1] = msg->obj_detection_result;
    if (cam_idx == 1)
        bGetDWResultsCenter = true;
    else if (cam_idx == 2)
        bGetDWResultsRight = true;
    else if (cam_idx == 3)
        bGetDWResultsLeft = true;
}

void callbackImagePointSync(const sensor_msgs::CompressedImageConstPtr& msg_img_center, const sensor_msgs::CompressedImageConstPtr& msg_img_left, const sensor_msgs::CompressedImageConstPtr& msg_img_right, const sensor_msgs::PointCloud2ConstPtr& msg_points)
{
    // Get CV Image (from compressed image message)
    cv::Mat img_center = cv::imdecode(cv::Mat(msg_img_center->data), 1); //convert compressed image data to cv::Mat
    cv::Mat img_left   = cv::imdecode(cv::Mat(msg_img_left->data), 1);
    cv::Mat img_right  = cv::imdecode(cv::Mat(msg_img_right->data), 1);
    img_center.copyTo(lidar_to_camera_center);
    img_left.copyTo(lidar_to_camera_left);
    img_right.copyTo(lidar_to_camera_right);

    // Initialize depth image (16 bit)
    cv::Mat depth_img_center(img_center.rows, img_center.cols, CV_16UC1, cv::Scalar(255*255));
    
    // Get Pointcloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*msg_points, *cloud);

    pcl::PointCloud<pcl::PointXYZI>::Ptr ptr_transformed_center(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr ptr_transformed_left(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr ptr_transformed_right(new pcl::PointCloud<pcl::PointXYZI>);

    pcl::transformPointCloud(*cloud, *ptr_transformed_center, trans_center);
    pcl::transformPointCloud(*cloud, *ptr_transformed_left, trans_left);
    pcl::transformPointCloud(*cloud, *ptr_transformed_right, trans_right);

    // Get Object Detection Results
    eurecar_msgs::px2ObjDetection obj_detections_center = px2ObjDetections[0]; // cam_idx == 1
    eurecar_msgs::px2ObjDetection obj_detections_left = px2ObjDetections[2]; // cam_idx == 3
    eurecar_msgs::px2ObjDetection obj_detections_right = px2ObjDetections[1]; // cam_idx == 2
    int num_obj_det_center = obj_detections_center.bboxs.size();
    int num_obj_det_left = obj_detections_left.bboxs.size();
    int num_obj_det_right = obj_detections_right.bboxs.size();

    // Initialize data containers
    // - initialize points container for each bbox (bbox points container)
    std::vector<std::vector<PointState>> bbox_points_containers_center; // Dimension: BBox > Points > PointState
    std::vector<std::vector<double>> bbox_depths_containers_center; // Dimension: BBox > Points > double

    // - initialize result state of bboxs (PixelX,PixelY,Depth,X,Y,Z) for center cam
    std::vector<PointState> bbox_result_states_center; // Dimension: BBox > PointState

    // - initialize detection bbox message (BoundingBoxArray)
    detection_msgs::BoundingBoxArray bboxes_result_states_current_center;

    // - resize data
    bbox_points_containers_center.resize(num_obj_det_center);
    bbox_depths_containers_center.resize(num_obj_det_center);
    bbox_result_states_center.resize(num_obj_det_center);

    // Iteration w.r.t. points (for Visualization of the Camera-LiDAR fusion)
    for (auto &point : cloud->points)
    {
        pcl::PointXYZI point_buf_center, point_buf_left, point_buf_right;
        
        // Point Buffer (U,V,W)
        // - for center cam + lidar
        point_buf_center.x = ((vec_center.at(0) * point.x + vec_center.at(1) * point.y + vec_center.at(2) * point.z + vec_center.at(3)) / (vec_center.at(8) * point.x + vec_center.at(9) * point.y + vec_center.at(10) * point.z + vec_center.at(11)));
        point_buf_center.y = ((vec_center.at(4) * point.x + vec_center.at(5) * point.y + vec_center.at(6) * point.z + vec_center.at(7)) / (vec_center.at(8) * point.x + vec_center.at(9) * point.y + vec_center.at(10) * point.z + vec_center.at(11)));
        point_buf_center.z = (vec_center.at(8) * point.x + vec_center.at(9) * point.y + vec_center.at(10) * point.z + vec_center.at(11));
        point_buf_center.x *= ratio_col;
        point_buf_center.y *= ratio_row;
        // // - for lidar-camera fusion visualization
        // circle(lidar_to_camera_center, cv::Point(point_buf_center.x, point_buf_center.y), 2, cv::Scalar(b, g, r), -1, 8, 0);

        // Gathering bbox_states_center and bbox_depths_container_center
        if (bGetDWResultsCenter)
        {
            // px2ObjDetections[0]: Object Detection results of Center
            for (int bbox_idx = 0; bbox_idx < px2ObjDetections[0].bboxs.size(); bbox_idx++)
            {
                // Check score && class (car)
                if (px2ObjDetections[0].objHypotheses[bbox_idx].score > DW_SCORE_THRESHOLD &&
                    px2ObjDetections[0].bboxs[bbox_idx].size_x > DW_SIZE_THRES)
                {
                    // - check inside ROI of center cam
                    double pixel_xmin = px2ObjDetections[0].bboxs[bbox_idx].center.x - px2ObjDetections[0].bboxs[bbox_idx].size_x/2;
                    double pixel_xmax = px2ObjDetections[0].bboxs[bbox_idx].center.x + px2ObjDetections[0].bboxs[bbox_idx].size_x/2;
                    double pixel_ymin = px2ObjDetections[0].bboxs[bbox_idx].center.y - px2ObjDetections[0].bboxs[bbox_idx].size_y/2;
                    double pixel_ymax = px2ObjDetections[0].bboxs[bbox_idx].center.y + px2ObjDetections[0].bboxs[bbox_idx].size_y/2;

                    if (point_buf_center.x >= pixel_xmin && point_buf_center.x <= pixel_xmax &&
                        point_buf_center.y >= pixel_ymin && point_buf_center.y <= pixel_ymax)
                    {
                        // - compute depth
                        double depth = sqrt(pow(point.x, 2.0) + pow(point.y, 2));
                        // - gather points in each bbox container
                        PointState bbox_state;
                        bbox_state.u = point_buf_center.x;
                        bbox_state.v = point_buf_center.y;
                        bbox_state.w = depth;
                        bbox_state.x = point.x;
                        bbox_state.y = point.y;
                        bbox_state.z = point.z;
                        bbox_points_containers_center[bbox_idx].push_back(bbox_state);
                        bbox_depths_containers_center[bbox_idx].push_back(depth);
                    }
                }
            }
        }

        // Point Buffer (U,V,W)
        // - for left cam + lidar
        point_buf_left.x = ((vec_left.at(0) * point.x + vec_left.at(1) * point.y + vec_left.at(2) * point.z + vec_left.at(3)) / (vec_left.at(8) * point.x + vec_left.at(9) * point.y + vec_left.at(10) * point.z + vec_left.at(11)));
        point_buf_left.y = ((vec_left.at(4) * point.x + vec_left.at(5) * point.y + vec_left.at(6) * point.z + vec_left.at(7)) / (vec_left.at(8) * point.x + vec_left.at(9) * point.y + vec_left.at(10) * point.z + vec_left.at(11)));
        point_buf_left.z = (vec_left.at(8) * point.x + vec_left.at(9) * point.y + vec_left.at(10) * point.z + vec_left.at(11));
        point_buf_left.x *= ratio_col;
        point_buf_left.y *= ratio_row;

        if (bGetDWResultsLeft)
        {
            // - check inside bbox & inside ROI of left cam
            // - put point in points container left
        }

        // Point Buffer (U,V,W)
        // - for right cam + lidar
        point_buf_right.x = ((vec_right.at(0) * point.x + vec_right.at(1) * point.y + vec_right.at(2) * point.z + vec_right.at(3)) / (vec_right.at(8) * point.x + vec_right.at(9) * point.y + vec_right.at(10) * point.z + vec_right.at(11)));
        point_buf_right.y = ((vec_right.at(4) * point.x + vec_right.at(5) * point.y + vec_right.at(6) * point.z + vec_right.at(7)) / (vec_right.at(8) * point.x + vec_right.at(9) * point.y + vec_right.at(10) * point.z + vec_right.at(11)));
        point_buf_right.z = (vec_right.at(8) * point.x + vec_right.at(9) * point.y + vec_right.at(10) * point.z + vec_right.at(11));
        point_buf_right.x *= ratio_col;
        point_buf_right.y *= ratio_row;

        if (bGetDWResultsRight)
        {
            // - check inside bbox & inside ROI of right cam
            // - put point in points container right
        }

    }

    // Estimate depth using median filter for each bbox points container && assign visualization message
    if (bbox_depths_containers_center.size() != 0)
    {
        for (int bbox_idx=0; bbox_idx < bbox_depths_containers_center.size(); bbox_idx++)
        {
            // Check num of depth(point state) in each bbox
            if (bbox_depths_containers_center[bbox_idx].size() == 0)
            {
                // TODO Do process when no lidar point in bbox
            }
            else
            {
                // - compute median index and value
                auto median_data = median_function(bbox_depths_containers_center[bbox_idx]);
                int median_index = median_data[0];
                double median_value = median_data[1];

                // - assign point state && gather bbox_result_state
                PointState bbox_result_state;
                bbox_result_state = bbox_points_containers_center[bbox_idx][median_index]; // copy median's [u,v,w,x,y,z] of each bbox 
                bbox_result_states_center[bbox_idx] = bbox_result_state; 

                // - assign boundingbox message
                detection_msgs::BoundingBox bbox_result_state_msg = setBbox3D(bbox_result_state.x, bbox_result_state.y, bbox_result_state.z, bbox_idx);
                bboxes_result_states_current_center.boxes.push_back(bbox_result_state_msg);
            }
        }
    }

    // Update current bbox results message
    bboxes_result_states_center = bboxes_result_states_current_center;
    bboxes_result_states_center.header.frame_id = "velodyne";
    bboxes_result_states_center.header.stamp = ros::Time::now();

    // Publish boundingbox message
    pubDWResultsBBoxArray.publish(bboxes_result_states_center);


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
    ros::NodeHandle nh_;
    ros::Rate loop_rate(30);

    printf("Initiate: LidarCameraCalibration\n");

    // for center cam (4x4 LiDAR-to-Camera transformation matrix)
    trans_center << -0.197307, 0.393827, -0.00539561, -0.442059,
                    -0.159261, 0.00514399, 0.427143, -0.634573,
                    -0.000410322, -1.00294e-05, 1.87862e-05, -0.000903967,
                    0, 0, 0, 1;
    vec_center = {-0.197307, 0.393827, -0.00539561, -0.442059,
                  -0.159261, 0.00514399, 0.427143, -0.634573,
                  -0.000410322, -1.00294e-05, 1.87862e-05, -0.000903967,
                  0, 0, 0, 1};

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


    // Subscriber (Asynchronous)
    // - define list of subscriber for drive px2 result (center/left/right)
    std::vector<ros::Subscriber> sub_dw_det_result_list;
    ROS_INFO("Initializing Subscribers for detection results");
    // - resize list of subscriber
    sub_dw_det_result_list.resize(NUM_CAMERAS);
    px2ObjDetections.resize(NUM_CAMERAS);
    // - initialize list of subscriber
    std::string det_topic_base = "/gmsl_camera";
    for (int cam_idx=1; cam_idx <= NUM_CAMERAS; cam_idx++)
    {
        std::string dw_det_result_topic = det_topic_base + std::to_string(cam_idx) + "/DwDetectionResult";
        sub_dw_det_result_list[cam_idx - 1] = nh_.subscribe<eurecar_msgs::px2DetectionResult>(
                                                dw_det_result_topic, 1,
                                                boost::bind(&callbackDWDetection, _1, cam_idx));
    }

    // Subscriber (Synchronous)
    message_filters::Subscriber<sensor_msgs::CompressedImage> sub_compressed_image_center(nh_, "/gmsl_camera1/compressed", 1);
    message_filters::Subscriber<sensor_msgs::CompressedImage> sub_compressed_image_left(nh_, "/gmsl_camera3/compressed", 1);
    message_filters::Subscriber<sensor_msgs::CompressedImage> sub_compressed_image_right(nh_, "/gmsl_camera2/compressed", 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_point_cloud(nh_, "/velodyne_points", 1);

    // for synchronized message subscribtion
    typedef message_filters::sync_policies::ApproximateTime <sensor_msgs::CompressedImage, sensor_msgs::CompressedImage, sensor_msgs::CompressedImage, sensor_msgs::PointCloud2> MySyncPolicy;
    message_filters::Synchronizer <MySyncPolicy> sync(MySyncPolicy(10), sub_compressed_image_center, sub_compressed_image_left, sub_compressed_image_right, sub_point_cloud);
    sync.registerCallback(boost::bind(&callbackImagePointSync, _1, _2, _3, _4));

    // Publisher
    pubDWResultsBBoxArray = nh_.advertise<detection_msgs::BoundingBoxArray>("/Fusion/BoundingBoxArray",10);
    
    while(ros::ok())
    {
      ros::spinOnce();
      loop_rate.sleep();
    }
    printf("Terminate: MinCut_Segmentation\n");

    return 0;
}
