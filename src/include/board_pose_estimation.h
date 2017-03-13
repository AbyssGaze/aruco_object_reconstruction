#ifndef BOARD_POSE_ESTIMATION_H_
#define BOARD_POSE_ESTIMATION_H_
#include <vector>
#include <Eigen/Dense>

// the board include file
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

#include <boost/make_shared.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>

#include <pcl/io/pcd_io.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>

#include <pcl/features/normal_3d.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/surface/marching_cubes.h>
#include <pcl/surface/marching_cubes_rbf.h>
#include <pcl/surface/marching_cubes_hoppe.h>

#include "MyPointRepresentation.h"

class board{
public:
    //convenient typedefs
    typedef pcl::PointXYZ PointT;
    typedef pcl::PointCloud<PointT> PointCloud;
    typedef pcl::PointNormal PointNormalT;
    typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

public:
	board();
public:
    void createBoard();

    bool poseEstimation(cv::Mat &image, cv::Mat &depth_mat);

    void readCameraParam(std::string cam_param);

private:
    void ICPTransform(const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt, PointCloud::Ptr output, Eigen::Matrix4f &final_transform, bool downsample);

    double frameDistance();

    void convertXYZPointCloud(cv::Mat &image, cv::Mat &depth_mat, PointCloud::Ptr cur_cloud);

    void createMesh();

    void writeDate(cv::Mat image, cv::Mat depth_mat, PointCloud::Ptr cur_cloud);
private:
	cv::Mat camera_matrics_;
	cv::Mat camera_dist_coffs_;
    cv::Ptr<cv::aruco::Board> board_ptr_;
    cv::Mat board_image_;
    int total_capture_frame_;
    int capture_frame_num_;

    cv::Vec3d cur_rvec_;
    cv::Vec3d cur_tvec_;

    cv::Vec3d key_rvec_;
    cv::Vec3d key_tvec_;

    Eigen::Matrix4f key_trans_;


    double distance_tresh_;

    PointCloud::Ptr merge_cloud_;
    PointCloud::Ptr cur_cloud_;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

    int vp_1, vp_2;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb_;

    std::stringstream pcd_file, depth_file, rgb_file, pose_file;
    std::vector<int> params;

};


#endif
