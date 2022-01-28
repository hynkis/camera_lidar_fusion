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
std::string FRAME_ID = "velodyne";
double DW_SCORE_THRESHOLD = 0.95; // threshold for driveworks detection score
double DW_SIZE_THRES = 20; // threshold for driveworks detection size
double ORIGINAL_ROW = 960; // 1920
double ORIGINAL_COL = 604; // 1208
double TARGET_ROW = 960; // 503
double TARGET_COL = 604; // 800
double RATIO_ROW = TARGET_ROW / ORIGINAL_ROW;
double RATIO_COL = TARGET_COL / ORIGINAL_COL;
double VEHICLE_DIM_X = 2.0; // 3.0
double VEHICLE_DIM_Y = 2.0; // 2.0
double VEHICLE_DIM_Z = 1.7; // 1.7

int SOFT_NMS_METHOD = 3; // linear: 1 / gaussian: 2 / original NMS: else
double SOFT_NMS_SIGMA = 0.5; // 0.5
double SOFT_NMS_IOU_THRES = 0.05; // 0.3
double HARD_NMS_IOU_THRES = 0.1;
double SOFT_NMS_SCORE_THRES = DW_SCORE_THRESHOLD; // default: 0.01

// Publisher & Subscriber
ros::Publisher pubDWResultsBBoxArrayCenter; // detection_msgs::BoundingBoxArray
ros::Publisher pubDWResultsBBoxArrayLeft;
ros::Publisher pubDWResultsBBoxArrayRight;
ros::Publisher pubDWResultsBBoxArrayFusion;

// Data structure for driveworks results
struct DetectionState
{
    // uvw values in Pixel coordinates (Center point)
    double u; // pixel_x
    double v; // pixel_y
    double w; // depth
    // xyz values in XYZ coordinates
    double x;
    double y;
    double z;
    // bounding box size
    double bbox_w;
    double bbox_h;
    // detection info (score, class id)
    double score;
    double id;
    // camera info
    int cam_num;
};

// Detection results msgs (Pixel coordinate)
std::vector<eurecar_msgs::px2ObjDetection> px2ObjDetections; // [center,left,right]

// Transformation Matrix (LiDAR to Camera)
Eigen::Matrix4f trans_center, trans_left, trans_right;
std::vector<float> vec_center, vec_left, vec_right;

// Variables
cv::Mat lidar_to_camera_center, lidar_to_camera_left, lidar_to_camera_right;
bool bGetDWResultsCenter = false;
bool bGetDWResultsLeft = false;
bool bGetDWResultsRight = false;

std::vector<DetectionState> m_bbox_result_states_center;
std::vector<DetectionState> m_bbox_result_states_left;
std::vector<DetectionState> m_bbox_result_states_right;

// Messages
detection_msgs::BoundingBoxArray m_bbox_msg_result_states_center;
detection_msgs::BoundingBoxArray m_bbox_msg_result_states_left;
detection_msgs::BoundingBoxArray m_bbox_msg_result_states_right;
detection_msgs::BoundingBoxArray bboxes_result_states_fusion;

// Utils
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

float calIOUWithNoRotation(const DetectionState& bbox1, const DetectionState& bbox2)
{
    // - determine (x,y) of intersection rectangle
    float inter_tx = std::max(bbox1.u - bbox1.bbox_w/2., bbox2.u - bbox2.bbox_w/2.); // top (x,y) of inter. rect.
    float inter_ty = std::max(bbox1.v - bbox1.bbox_h/2., bbox2.v - bbox2.bbox_h/2.);
    float inter_bx = std::min(bbox1.u + bbox1.bbox_w/2., bbox2.u + bbox2.bbox_w/2.); // bottom (x,y) of inter. rect.
    float inter_by = std::min(bbox1.v + bbox1.bbox_h/2., bbox2.v + bbox2.bbox_h/2.);
    // - determine width, height of intersection rectangle
    float inter_w = inter_bx - inter_tx + 1;
    float inter_h = inter_by - inter_ty + 1;
    // - no overlap
    if (inter_w < 0 || inter_h < 0)
    {
        return 0.;
    }
    // - calculate iou
    float inter_area = inter_w * inter_h;
    float bbox1_area = bbox1.bbox_w * bbox1.bbox_h;
    float bbox2_area = bbox2.bbox_w * bbox2.bbox_h;
    float iou = inter_area / (bbox1_area + bbox2_area - inter_area + 1e-4);

    return iou;
}

float calXYIOUWithNoRotation(const DetectionState& bbox1, const DetectionState& bbox2)
{
    // - determine (x,y) of intersection rectangle
    float inter_tx = std::max(bbox1.x - VEHICLE_DIM_X/2., bbox2.x - VEHICLE_DIM_X/2.); // top (x,y) of inter. rect.
    float inter_ty = std::max(bbox1.y - VEHICLE_DIM_Y/2., bbox2.y - VEHICLE_DIM_Y/2.);
    float inter_bx = std::min(bbox1.x + VEHICLE_DIM_X/2., bbox2.x + VEHICLE_DIM_X/2.); // bottom (x,y) of inter. rect.
    float inter_by = std::min(bbox1.y + VEHICLE_DIM_Y/2., bbox2.y + VEHICLE_DIM_Y/2.);
    // - determine width, height of intersection rectangle
    float inter_w = inter_bx - inter_tx + 1;
    float inter_h = inter_by - inter_ty + 1;
    // - no overlap
    if (inter_w < 0 || inter_h < 0)
    {
        return 0.;
    }
    // - calculate iou
    float inter_area = inter_w * inter_h;
    float bbox1_area = VEHICLE_DIM_X * VEHICLE_DIM_Y;
    float bbox2_area = VEHICLE_DIM_X * VEHICLE_DIM_Y;
    float iou = inter_area / (bbox1_area + bbox2_area - inter_area + 1e-4);

    return iou;
}


void IOUFilteringWithNoRotation(std::vector<DetectionState>& bboxes,
                                const int& method, const float& sigma,
                                const float& iou_thres,
                                const float& iou_thres_hard,
                                const float& score_threshold)
{
    // IOU Filtering
    // : Soft Non-Maximum Suppression (Linear/Gaussian/Original) or Hard IOU filtering
    // : if some bboxes has (iou > iou_thres), reduce confidence
    // (reference) https://eehoeskrap.tistory.com/407 [Enough is not enough]

    if (bboxes.empty())
    {
        return;
    }
    // 
    int N = bboxes.size();
    float max_score, max_pos, cur_pos, weight;
    DetectionState tmp_bbox, index_bbox;

    // Sort by distance

    for (int i=0; i < N; ++i)
    {
        // Soft NMS
        // - init values
        max_score = bboxes[i].score;
        max_pos = i;
        tmp_bbox = bboxes[i];
        cur_pos = i;

        // - get max score bbox
        while (cur_pos < N)
        {
            if (max_score < bboxes[cur_pos].score)
            {
                max_score = bboxes[cur_pos].score;
                max_pos = cur_pos;
            }
            cur_pos++;
        }

        // - add max bbox as a detection
        bboxes[i] = bboxes[max_pos];
        // - swap i th bbox with position of max score bbox
        bboxes[max_pos] = tmp_bbox;
        tmp_bbox = bboxes[i];
        cur_pos = i + 1;

        // 
        while (cur_pos < N)
        {
            //
            index_bbox = bboxes[cur_pos];

            float area = index_bbox.bbox_h * index_bbox.bbox_w;
            // float iou = calIOUWithNoRotation(tmp_bbox, index_bbox);
            float iou = calXYIOUWithNoRotation(tmp_bbox, index_bbox);
            // - pass this bbox (no overlap)
            if (iou <= 0)
            {
                cur_pos++;
                continue; 
            }
            // - NMS method (linear/gaussian/original NMS)
            // - Lineara Soft-NMS
            if (method == 1)
            {
                if (iou > iou_thres)
                    weight = 1 - iou;
                else
                    weight = 1;
            }
            // - Gaussian Soft-NMS
            else if (method == 2)
            {
                weight = exp(-(iou * iou) / sigma);
            }
            // - Original NMS
            else
            {
                if (iou > iou_thres)
                    weight = 0;
                else
                    weight = 1;
            }
            // - apply hard iou threshold
            if (iou > iou_thres_hard)
            {
                weight = 0;
            }
            // - apply weight at score
            bboxes[cur_pos].score *= weight;

            if (bboxes[cur_pos].score <= score_threshold)
            {
                bboxes[cur_pos] = bboxes[N - 1];
                N--;
                cur_pos = cur_pos + 1;
            }
            cur_pos++;
        }
    }
    bboxes.resize(N);

    // Remove (0,0) data
    // - use 'remove_if'
    // (reference) https://stackoverflow.com/questions/17270837/stdvector-removing-elements-which-fulfill-some-conditions
    bboxes.erase(std::remove_if
    (
        bboxes.begin(), bboxes.end(),
        [](const DetectionState& bbox_state)
        {
            return (bbox_state.x == 0 && bbox_state.y == 0); // condition
        }
    ), bboxes.end());
}


detection_msgs::BoundingBox setBbox3D(double point_x, double point_y, double point_z, int label_id)
{
    detection_msgs::BoundingBox bbox3D;
    bbox3D.header.frame_id = FRAME_ID;
    bbox3D.header.stamp = ros::Time::now();
    bbox3D.label = label_id;
    bbox3D.dimensions.x = VEHICLE_DIM_X; // 3.0 car dimensions
    bbox3D.dimensions.y = VEHICLE_DIM_Y; // 2.0 
    bbox3D.dimensions.z = VEHICLE_DIM_Z; // 1.7
    bbox3D.pose.position.x = point_x;
    bbox3D.pose.position.y = point_y;
    bbox3D.pose.position.z = point_z;
    bbox3D.pose.orientation.x = 0;
    bbox3D.pose.orientation.y = 0;
    bbox3D.pose.orientation.z = 0;
    bbox3D.pose.orientation.w = 1;
    return bbox3D;
}

void filterPointInObjBBOX(pcl::PointXYZI fusion_point, pcl::PointXYZI point, eurecar_msgs::px2ObjDetection obj_detections,
                          std::vector<std::vector<DetectionState>> &bbox_points_containers,
                          std::vector<std::vector<double>> &bbox_depths_containers)
{
    // tips: '&' in &bbox_points_containers, &bbox_depths_containers for updating those variables 
    for (int bbox_idx = 0; bbox_idx < obj_detections.bboxs.size(); bbox_idx++)
    {
        // Check score && bbox size && class (car, pede) [id: car(0), traffic_sign(1), bicycle(2), traffic_signal(3), Pedestrian(4)]
        if (obj_detections.objHypotheses[bbox_idx].score > DW_SCORE_THRESHOLD &&
            obj_detections.bboxs[bbox_idx].size_x > DW_SIZE_THRES &&
            (obj_detections.objHypotheses[bbox_idx].id == 0 ||
             obj_detections.objHypotheses[bbox_idx].id == 2 ||
             obj_detections.objHypotheses[bbox_idx].id == 4))
        {
            // - check inside ROI of center cam
            double pixel_xmin = obj_detections.bboxs[bbox_idx].center.x - obj_detections.bboxs[bbox_idx].size_x/2;
            double pixel_xmax = obj_detections.bboxs[bbox_idx].center.x + obj_detections.bboxs[bbox_idx].size_x/2;
            double pixel_ymin = obj_detections.bboxs[bbox_idx].center.y - obj_detections.bboxs[bbox_idx].size_y/2;
            double pixel_ymax = obj_detections.bboxs[bbox_idx].center.y + obj_detections.bboxs[bbox_idx].size_y/2;

            if (fusion_point.x >= pixel_xmin && fusion_point.x <= pixel_xmax &&
                fusion_point.y >= pixel_ymin && fusion_point.y <= pixel_ymax)
            {
                // - compute depth
                double depth = sqrt(pow(point.x, 2.0) + pow(point.y, 2));
                // - gather states in each bbox container
                DetectionState bbox_state;
                bbox_state.u = fusion_point.x;
                bbox_state.v = fusion_point.y;
                bbox_state.w = depth;
                bbox_state.x = point.x;
                bbox_state.y = point.y;
                bbox_state.z = point.z;

                bbox_points_containers[bbox_idx].push_back(bbox_state);
                bbox_depths_containers[bbox_idx].push_back(depth);
            }
        }
    }
}


void estimateDepthAndAssignBBOX(std::vector<std::vector<DetectionState>> bbox_points_containers,
                                std::vector<std::vector<double>> bbox_depths_containers,
                                std::vector<DetectionState> &bbox_result_states,
                                eurecar_msgs::px2ObjDetection obj_detections,
                                detection_msgs::BoundingBoxArray &bbox_msg_result_states)
{
    // tips: '&' in &bbox_result_states, &bbox_msg_result_states for updating those variables 
    for (int bbox_idx=0; bbox_idx < bbox_depths_containers.size(); bbox_idx++)
    {
        // Check num of depth(point state) in each bbox
        if (bbox_depths_containers[bbox_idx].size() == 0)
        {
            // TODO Do process when no lidar point in bbox
        }
        else
        {
            // - compute median index and value
            auto median_data = median_function(bbox_depths_containers[bbox_idx]);
            int median_index = median_data[0];
            double median_value = median_data[1];

            // - assign point state && gather bbox_result_state
            DetectionState bbox_result_state;
            bbox_result_state = bbox_points_containers[bbox_idx][median_index]; // copy median's [u,v,w,x,y,z] of each bbox
            
            bbox_result_state.bbox_w = obj_detections.bboxs[bbox_idx].size_x; // copy info of current bbox
            bbox_result_state.bbox_h = obj_detections.bboxs[bbox_idx].size_y;
            bbox_result_state.score = obj_detections.objHypotheses[bbox_idx].score;
            bbox_result_state.id = obj_detections.objHypotheses[bbox_idx].id;
            
            bbox_result_states[bbox_idx] = bbox_result_state; 

            // - assign boundingbox message
            detection_msgs::BoundingBox bbox_result_state_msg = setBbox3D(bbox_result_state.x, bbox_result_state.y, bbox_result_state.z, bbox_idx);
            bbox_msg_result_states.boxes.push_back(bbox_result_state_msg);
        }
    }
}

void boxFusion(std::vector<DetectionState> &bbox_result_states_center,
               std::vector<DetectionState> &bbox_result_states_left,
               std::vector<DetectionState> &bbox_result_states_right,
               std::vector<DetectionState> &bbox_result_states_fusion)
{
    // Box Fusion using Soft Non-Maximum Suppression (Soft NMS)
    // - concatenate bbox results (center/left/right)
    bbox_result_states_fusion.insert(bbox_result_states_fusion.end(),
                                     std::make_move_iterator(bbox_result_states_center.begin()),
                                     std::make_move_iterator(bbox_result_states_center.end()));
    bbox_result_states_fusion.insert(bbox_result_states_fusion.end(),
                                     std::make_move_iterator(bbox_result_states_left.begin()),
                                     std::make_move_iterator(bbox_result_states_left.end()));
    bbox_result_states_fusion.insert(bbox_result_states_fusion.end(),
                                     std::make_move_iterator(bbox_result_states_right.begin()),
                                     std::make_move_iterator(bbox_result_states_right.end()));
    // - box fusion using Soft-NMS
    IOUFilteringWithNoRotation(bbox_result_states_fusion, SOFT_NMS_METHOD, SOFT_NMS_SIGMA, SOFT_NMS_IOU_THRES, HARD_NMS_IOU_THRES, SOFT_NMS_SCORE_THRES);
}

// Callback for Driveworks detection results (center/left/right)
void callbackDWDetection(const eurecar_msgs::px2DetectionResult::ConstPtr& msg, int cam_idx)
{
    px2ObjDetections[cam_idx - 1] = msg->obj_detection_result;
    if (cam_idx == 1)
    {
        bGetDWResultsCenter = true;
    }
    else if (cam_idx == 2)
    {
        bGetDWResultsRight = true;
    }
    else if (cam_idx == 3)
    {
        bGetDWResultsLeft = true;
    }
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
    std::vector<std::vector<DetectionState>> bbox_points_containers_center; // Dimension: BBox > Points > DetectionState
    std::vector<std::vector<DetectionState>> bbox_points_containers_left;
    std::vector<std::vector<DetectionState>> bbox_points_containers_right;
    std::vector<std::vector<double>> bbox_depths_containers_center; // Dimension: BBox > Points > double
    std::vector<std::vector<double>> bbox_depths_containers_left;
    std::vector<std::vector<double>> bbox_depths_containers_right;

    // - initialize result state of bboxs (PixelX,PixelY,Depth,X,Y,Z) for center cam
    std::vector<DetectionState> bbox_result_states_center; // Dimension: BBox > DetectionState
    std::vector<DetectionState> bbox_result_states_left;
    std::vector<DetectionState> bbox_result_states_right;

    // - initialize detection bbox message (BoundingBoxArray)
    detection_msgs::BoundingBoxArray bbox_msg_result_states_center;
    detection_msgs::BoundingBoxArray bbox_msg_result_states_left;
    detection_msgs::BoundingBoxArray bbox_msg_result_states_right;

    // - resize data
    bbox_points_containers_center.resize(num_obj_det_center);
    bbox_points_containers_left.resize(num_obj_det_left);
    bbox_points_containers_right.resize(num_obj_det_right);

    bbox_depths_containers_center.resize(num_obj_det_center);
    bbox_depths_containers_left.resize(num_obj_det_left);
    bbox_depths_containers_right.resize(num_obj_det_right);

    bbox_result_states_center.resize(num_obj_det_center);
    bbox_result_states_left.resize(num_obj_det_left);
    bbox_result_states_right.resize(num_obj_det_right);

    // Iteration w.r.t. points (for Visualization of the Camera-LiDAR fusion)
    for (auto &point : cloud->points)
    {
        pcl::PointXYZI point_buf_center, point_buf_left, point_buf_right;
        
        // Point Buffer (U,V,W)
        // - for center cam + lidar
        point_buf_center.x = ((vec_center.at(0) * point.x + vec_center.at(1) * point.y + vec_center.at(2) * point.z + vec_center.at(3)) / (vec_center.at(8) * point.x + vec_center.at(9) * point.y + vec_center.at(10) * point.z + vec_center.at(11)));
        point_buf_center.y = ((vec_center.at(4) * point.x + vec_center.at(5) * point.y + vec_center.at(6) * point.z + vec_center.at(7)) / (vec_center.at(8) * point.x + vec_center.at(9) * point.y + vec_center.at(10) * point.z + vec_center.at(11)));
        point_buf_center.z = (vec_center.at(8) * point.x + vec_center.at(9) * point.y + vec_center.at(10) * point.z + vec_center.at(11));
        point_buf_center.x *= RATIO_COL;
        point_buf_center.y *= RATIO_ROW;
        // // - for lidar-camera fusion visualization
        // circle(lidar_to_camera_center, cv::Point(point_buf_center.x, point_buf_center.y), 2, cv::Scalar(b, g, r), -1, 8, 0);

        // Gathering bbox_states_center and bbox_depths_container_center
        if (bGetDWResultsCenter)
        {
            // - check inside bbox & inside ROI of center cam
            // - put point in points container center
            filterPointInObjBBOX(point_buf_center, point, obj_detections_center, bbox_points_containers_center, bbox_depths_containers_center);
        }

        // Point Buffer (U,V,W)
        // - for left cam + lidar
        point_buf_left.x = ((vec_left.at(0) * point.x + vec_left.at(1) * point.y + vec_left.at(2) * point.z + vec_left.at(3)) / (vec_left.at(8) * point.x + vec_left.at(9) * point.y + vec_left.at(10) * point.z + vec_left.at(11)));
        point_buf_left.y = ((vec_left.at(4) * point.x + vec_left.at(5) * point.y + vec_left.at(6) * point.z + vec_left.at(7)) / (vec_left.at(8) * point.x + vec_left.at(9) * point.y + vec_left.at(10) * point.z + vec_left.at(11)));
        point_buf_left.z = (vec_left.at(8) * point.x + vec_left.at(9) * point.y + vec_left.at(10) * point.z + vec_left.at(11));
        point_buf_left.x *= RATIO_COL;
        point_buf_left.y *= RATIO_ROW;

        if (bGetDWResultsLeft)
        {
            // - check inside bbox & inside ROI of left cam
            // - put point in points container left
            filterPointInObjBBOX(point_buf_left, point, obj_detections_left, bbox_points_containers_left, bbox_depths_containers_left);
        }

        // Point Buffer (U,V,W)
        // - for right cam + lidar
        point_buf_right.x = ((vec_right.at(0) * point.x + vec_right.at(1) * point.y + vec_right.at(2) * point.z + vec_right.at(3)) / (vec_right.at(8) * point.x + vec_right.at(9) * point.y + vec_right.at(10) * point.z + vec_right.at(11)));
        point_buf_right.y = ((vec_right.at(4) * point.x + vec_right.at(5) * point.y + vec_right.at(6) * point.z + vec_right.at(7)) / (vec_right.at(8) * point.x + vec_right.at(9) * point.y + vec_right.at(10) * point.z + vec_right.at(11)));
        point_buf_right.z = (vec_right.at(8) * point.x + vec_right.at(9) * point.y + vec_right.at(10) * point.z + vec_right.at(11));
        point_buf_right.x *= RATIO_COL;
        point_buf_right.y *= RATIO_ROW;

        if (bGetDWResultsRight)
        {
            // - check inside bbox & inside ROI of right cam
            // - put point in points container right
            filterPointInObjBBOX(point_buf_right, point, obj_detections_right, bbox_points_containers_right, bbox_depths_containers_right);
        }

    }

    // Estimate depth using median filter for each bbox points container && assign visualization message
    if (bbox_depths_containers_center.size() != 0)
    {
        // - estimate depth using median function
        // - assign boundingbox array message
        estimateDepthAndAssignBBOX(bbox_points_containers_center,
                                   bbox_depths_containers_center,
                                   bbox_result_states_center,
                                   obj_detections_center,
                                   bbox_msg_result_states_center);
    }
    if (bbox_depths_containers_left.size() != 0)
    {
        // - estimate depth using median function
        // - assign boundingbox array message
        estimateDepthAndAssignBBOX(bbox_points_containers_left,
                                   bbox_depths_containers_left,
                                   bbox_result_states_left,
                                   obj_detections_left,
                                   bbox_msg_result_states_left);
    }
    if (bbox_depths_containers_right.size() != 0)
    {
        // - estimate depth using median function
        // - assign boundingbox array message
        estimateDepthAndAssignBBOX(bbox_points_containers_right,
                                   bbox_depths_containers_right,
                                   bbox_result_states_right,
                                   obj_detections_right,
                                   bbox_msg_result_states_right);
    }

    // Update current detection results
    m_bbox_result_states_center = bbox_result_states_center;
    m_bbox_result_states_left = bbox_result_states_left;
    m_bbox_result_states_right = bbox_result_states_right;

    // Update bbox results message
    m_bbox_msg_result_states_center = bbox_msg_result_states_center;
    m_bbox_msg_result_states_center.header.frame_id = FRAME_ID;
    m_bbox_msg_result_states_center.header.stamp = ros::Time::now();

    m_bbox_msg_result_states_left = bbox_msg_result_states_left;
    m_bbox_msg_result_states_left.header.frame_id = FRAME_ID;
    m_bbox_msg_result_states_left.header.stamp = ros::Time::now();

    m_bbox_msg_result_states_right = bbox_msg_result_states_right;
    m_bbox_msg_result_states_right.header.frame_id = FRAME_ID;
    m_bbox_msg_result_states_right.header.stamp = ros::Time::now();

    // Publish boundingbox message
    pubDWResultsBBoxArrayCenter.publish(m_bbox_msg_result_states_center);
    pubDWResultsBBoxArrayLeft.publish(m_bbox_msg_result_states_left);
    pubDWResultsBBoxArrayRight.publish(m_bbox_msg_result_states_right);


    // // View visualization image
    // if (lidar_to_camera_center.empty() || lidar_to_camera_left.empty() || lidar_to_camera_right.empty())
    // {
    //     std::cout << "image is empty" << std::endl;
    // }
    // else
    // {
    //     cv::imshow("cam_center", lidar_to_camera_center);
    //     cv::imshow("cam_left", lidar_to_camera_left);
    //     cv::imshow("cam_right", lidar_to_camera_right);
    //     cv::waitKey(1);
    // }
}

void run()
{
    // Fuse current detection results from multi cameras
    std::vector<DetectionState> bbox_result_states_fusion;
    boxFusion(m_bbox_result_states_center, m_bbox_result_states_left, m_bbox_result_states_right, bbox_result_states_fusion);

    // - assign bounding box array message
    detection_msgs::BoundingBoxArray bbox_msg_result_state_fusion;
    bbox_msg_result_state_fusion.header.frame_id = FRAME_ID;
    bbox_msg_result_state_fusion.header.stamp = ros::Time::now();
    std::cout << "bbox_result_states_fusion.size() : " << bbox_result_states_fusion.size() << std::endl;
    for (auto &bbox_result_state : bbox_result_states_fusion)
    {
        // - assign bounding box message
        detection_msgs::BoundingBox bbox_msg_result_state = setBbox3D(bbox_result_state.x, bbox_result_state.y, bbox_result_state.z, bbox_result_state.id);
        bbox_msg_result_state_fusion.boxes.push_back(bbox_msg_result_state);
    }
    // - publish message
    pubDWResultsBBoxArrayFusion.publish(bbox_msg_result_state_fusion);
}

int main(int argc, char** argv){
    ros::init (argc, argv, "LidarCameraCalibration");
    ros::NodeHandle nh_;
    ros::Rate loop_rate(30); // as fast as possible

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
    pubDWResultsBBoxArrayCenter = nh_.advertise<detection_msgs::BoundingBoxArray>("/Fusion/BoundingBoxArray/Center",10);
    pubDWResultsBBoxArrayLeft = nh_.advertise<detection_msgs::BoundingBoxArray>("/Fusion/BoundingBoxArray/Left",10);
    pubDWResultsBBoxArrayRight = nh_.advertise<detection_msgs::BoundingBoxArray>("/Fusion/BoundingBoxArray/Right",10);
    pubDWResultsBBoxArrayFusion = nh_.advertise<detection_msgs::BoundingBoxArray>("/Fusion/BoundingBoxArray",10);
    
    while(ros::ok())
    {
        // Subscribe messages
        ros::spinOnce();
        // Run main process
        run();
        // Rate control
        loop_rate.sleep();
    }
    printf("Terminate: MinCut_Segmentation\n");

    return 0;
}
