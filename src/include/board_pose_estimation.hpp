#ifndef BOARD_POSE_ESTIMATION_HPP
#define BOARD_POSE_ESTIMATION_HPP

#include "board_pose_estimation.h"

using namespace std;
using namespace cv;
using namespace pcl;

board::board():capture_frame_num_(0)
{
    ;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void board::createBoard(){
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    //marker length
    double length = 0.065;
    auto t = length;
    cout << t << endl;
    vector<vector<Point3f> > marker_corner_vec;
    vector<int> ids_vec;
    //marker left up corner coordinate
    vector<double> corners_x = {0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0, 0, 0, 0, 0, 0.48, 0.48, 0.48, 0.48, 0.48};
    vector<double> corners_y = {0, 0, 0, 0, 0, 0, 0, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.08, 0.16, 0.24, 0.32, 0.40, 0.08, 0.16, 0.24, 0.32, 0.40};

    //CCW order is important
    //here we order the four corners for:left up, left bottom, right bottom and right up
    for(unsigned i = 0; i < corners_x.size(); ++i){
        vector<Point3f> marker_corners(4, Point3f(0, 0, 0));
        marker_corners[0].x = corners_x[i];
        marker_corners[0].y = corners_y[i];
        marker_corners[1].x = corners_x[i];
        marker_corners[1].y = corners_y[i] + length;
        marker_corners[2].x = corners_x[i] + length;
        marker_corners[2].y = corners_y[i] + length;
        marker_corners[3].x = corners_x[i] + length;
        marker_corners[3].y = corners_y[i];
        marker_corner_vec.push_back(marker_corners);
        ids_vec.push_back(i + 1);
    }

    // create the board
    board_ptr_ = aruco::Board::create(marker_corner_vec, dictionary, ids_vec);

    //translate the board to mat
    aruco::drawPlanarBoard(board_ptr_, Size(3000, 3000), board_image_, 70, 1);
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void board::readCameraParam(std::string param){
//    FileStorage fs("test.yml", FileStorage::WRITE);

//    Mat cameraMatrix = (Mat_<double>(3,3) << 558.341390, 0, 314.763671, 0, 558.387543, 240.992295, 0, 0, 1);
//    Mat distCoeffs = (Mat_<double>(5,1) << 0.062568, -0.096148, 0.000140, -0.006248, 0);
//    fs << "cameraMatrix" << cameraMatrix << "distCoeffs" << distCoeffs;
//    fs.release();
//    FileStorage fs("../param.yaml", FileStorage::READ);
    FileStorage fs(param, FileStorage::READ);

    fs["cameraMatrix"] >> camera_matrics_;
    fs["distCoeffs"] >> camera_dist_coffs_;
    fs["captureNum"] >> total_capture_frame_;
    fs.release();
}

/////////////////////////////////////////////////////////////////////////////////////////////////
bool board::poseEstimation(Mat &image){
    vector<int> ids;
    vector<vector<Point2f>> corners;

    //the marker dictionary
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);

    //detect all markers in image
    aruco::detectMarkers(image, dictionary, corners, ids);

    if(ids.size()){
        aruco::drawDetectedMarkers(image, corners, ids);

        int valid = aruco::estimatePoseBoard(corners, ids, board_ptr_, camera_matrics_, camera_dist_coffs_, cur_tvec_, cur_tvec_);

        if(valid > 0){
            aruco::drawAxis(image, camera_matrics_, camera_dist_coffs_, cur_tvec_, cur_tvec_, 0.1);
            //write the depth image, rgb image and the camera pose

            return true;
        }
    }
    return false;
}
/////////////////////////////////////////////////////////////////////////////////////////////////

void board::ICPTransform(PointCloud<PointXYZ>::Ptr &sourceCloud, PointCloud<PointXYZ> &targetCloud, Eigen::Affine3f &transformation)
{
    // Compute surface normals and curvature
    CloudPtr cloud1(new CloudT);
    CloudPtr cloud2(new CloudT);
    VoxelGrid<PointXYZ> sor;
    sor.setInputCloud(sourceCloud);
    float leafSize = 0.002f;
    sor.setLeafSize(leafSize, leafSize, leafSize);
    sor.filter(*cloud1);

    transformPointCloud(*targetCloud, *targetCloud, transformation);
    sor.setInputCloud(targetCloud);
    sor.setLeafSize(leafSize, leafSize, leafSize);
    sor.filter(*cloud2);

    PointCloud<PointNormal>::Ptr points_with_normals_src(new pcl::PointCloud<PointNormal>);
    PointCloud<PointNormal>::Ptr points_with_normals_tgt(new pcl::PointCloud<PointNormal>);

    NormalEstimation<PointXYZ, PointNormal> norm_est;
    search::KdTree<PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    norm_est.setSearchMethod(tree);
    norm_est.setKSearch(30);

    norm_est.setInputCloud(cloud1);
    norm_est.compute(*points_with_normals_src);
    copyPointCloud(*cloud1, *points_with_normals_src);

    norm_est.setInputCloud(cloud2);
    norm_est.compute(*points_with_normals_tgt);
    copyPointCloud(*cloud2, *points_with_normals_tgt);

    //
    // Instantiate our custom point representation (defined above) ...
    MyPointRepresentation point_representation;
    //// ... and weight the 'curvature' dimension so that it is balanced against x, y, and z
    float alpha[4] = { 1.0, 1.0, 1.0, 1.0 };
    point_representation.setRescaleValues(alpha);

    //
    // Align
    IterativeClosestPointNonLinear<pcl::PointNormal, pcl::PointNormal> reg;
    reg.setTransformationEpsilon(1e-6);
    // Set the maximum distance between two correspondences (src<->tgt) to 10cm
    // Note: adjust this based on the size of your datasets

    // Set the point representation
    reg.setPointRepresentation(boost::make_shared<const MyPointRepresentation>(point_representation));

    reg.setInputSource(points_with_normals_src);
    reg.setInputTarget(points_with_normals_tgt);

    //
    // Run the same optimization in a loop and visualize the results
    Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity(), prev, targetToSource;
    PointCloud<PointNormal>::Ptr reg_result = points_with_normals_src;
    reg.setMaximumIterations(10);
    for (int i = 0; i < 8; ++i)
    {
        double correspondenceDistance = 0.03;
        correspondenceDistance -= i * 0.01;
        reg.setMaxCorrespondenceDistance(correspondenceDistance);
        //PCL_INFO("Iteration Nr. %d.\n", i);

        // save cloud for visualization purpose
        points_with_normals_src = reg_result;

        // Estimate
        reg.setInputSource(points_with_normals_src);
        reg.align(*reg_result);

        //accumulate transformation between each Iteration
        Ti = reg.getFinalTransformation() * Ti;

        //if the difference between this transformation and the previous one
        //is smaller than the threshold, refine the process by reducing
        //the maximal correspondence distance
        if (fabs((reg.getLastIncrementalTransformation() - prev).sum()) < reg.getTransformationEpsilon())
            reg.setMaxCorrespondenceDistance(reg.getMaxCorrespondenceDistance() - 0.001);

        prev = reg.getLastIncrementalTransformation();

        leafSize = leafSize + i * 0.001;

        sor.setInputCloud(cloud1);
        sor.setLeafSize(leafSize, leafSize, leafSize);
        sor.filter(*cloud1);
        sor.setInputCloud(cloud2);
        sor.setLeafSize(leafSize, leafSize, leafSize);
        sor.filter(*cloud2);

    }
    //std::cout << "overlaping ratio:" << ;
    // Get the transformation from target to source
    targetToSource = Ti.inverse();
    transformPointCloud(*targetCloud, *targetCloud, targetToSource);

    //pcl::registration::CorrespondenceRejectorTrimmed overlap;

    //overlap.setSourcePoints(cloud1);
    //overlap.setTargetPoints(cloud2);
    //int overlapRatio = overlap.getOverlapRatio();
}


#endif // BOARD_POSE_ESTIMATION_HPP
