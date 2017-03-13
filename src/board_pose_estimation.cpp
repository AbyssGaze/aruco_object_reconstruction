
#include "include/board_pose_estimation.h"
#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <opencv2/core/eigen.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/surface/gp3.h>
#include <fstream>

using namespace std;
using namespace cv;
using namespace pcl;

using pcl::visualization::PointCloudColorHandlerGenericField;
using pcl::visualization::PointCloudColorHandlerCustom;

board::board():capture_frame_num_(0), vp_1(0), vp_2(0),merge_cloud_(new PointCloud), cur_cloud_(new PointCloud), cloud_rgb_(new pcl::PointCloud<PointXYZRGB>), viewer( new visualization::PCLVisualizer("3D viewer"))
{
//    merge_cloud_ = new PointCloud;
//    cur_cloud_ = new PointCloud;
    viewer->addCoordinateSystem(0.1);

    params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    params.push_back(0);
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void board::writeDate(Mat image, Mat depth_mat, PointCloud::Ptr cloud){
    rgb_file.str("");
    depth_file.str("");
    pcd_file.str("");
    pose_file.str("");

    rgb_file << "..//image//rocker_device_rgb_20170310_" << setw( 5 ) << setfill( '0' ) << capture_frame_num_ << ".jpg";
    depth_file << "..//image//rocker_device_depth_20170310_" << setw( 5 ) << setfill( '0' )<< capture_frame_num_ << ".png";
    pcd_file << "..//image//rocker_device_pcd_20170310_" << setw( 5 ) << setfill( '0' ) << capture_frame_num_ << ".pcd";
    pose_file << "..//image//rocker_device_pose_20170310_" << setw( 5 ) << setfill( '0' ) << capture_frame_num_ << ".tra";

    imwrite(rgb_file.str(), image);
    imwrite(depth_file.str(), depth_mat, params);

    transformPointCloud (*cloud_rgb_, *cloud_rgb_, key_trans_);

    io::savePCDFile (pcd_file.str(), *cloud_rgb_, true);
    fstream fs(pose_file.str(), ios::out | ios::binary);

    fs.write((char*)key_trans_.data(), 4 * 4 * sizeof(float));

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
void board::readCameraParam(string param){
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
    fs["distanceThresh"] >> distance_tresh_;
    fs.release();
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void board::convertXYZPointCloud(Mat &image, Mat &depth_mat, PointCloud::Ptr cur_cloud){
//    cur_cloud_->clear();
    cloud_rgb_->clear();
    for (unsigned int y = 0; y < depth_mat.rows; ++y)
    {
        for (unsigned int x = 0; x < depth_mat.cols; ++x)
        {
            PointT point;
            PointXYZRGB point_rgb;
            unsigned short d = depth_mat.at<unsigned short>(y, x);

            if (d != 0)
            {
                point.z = (double)d / (double)1000; // Convert from mm to meters
                point.x = ((double)x - camera_matrics_.at<double>(0,2)) * point.z / camera_matrics_.at<double>(0,0);
                point.y = ((double)y - camera_matrics_.at<double>(1,2)) * point.z / camera_matrics_.at<double>(1,1);

                point_rgb.z = point.z;
                point_rgb.x = point.x;
                point_rgb.y = point.y;
                point_rgb.b = image.at<Vec3b>(y,x)[0];
                point_rgb.g = image.at<Vec3b>(y,x)[1];
                point_rgb.r = image.at<Vec3b>(y,x)[2];
                cloud_rgb_->points.push_back(point_rgb);
                cur_cloud->points.push_back(point);
            }
        }
    }



    Mat cur_rvec_mat;
    Rodrigues(cur_rvec_, cur_rvec_mat);
    //current frame pose transform
    Eigen::Matrix3f cur_rvec_matrix;
    cv2eigen(cur_rvec_mat, cur_rvec_matrix);

    Eigen::Matrix4f targetToSource = Eigen::Matrix4f::Identity ();

    targetToSource.block<3,3>(0,0) = cur_rvec_matrix;
    targetToSource(0, 3) = cur_tvec_[0];
    targetToSource(1, 3) = cur_tvec_[1];
    targetToSource(2, 3) = cur_tvec_[2];


//    targetToSource.block<3,1>(0,3) = cur_tvec_;
    key_trans_ = targetToSource.inverse();
    transformPointCloud (*cur_cloud, *cur_cloud, key_trans_);


    cout << "1: the cloud point size is:" << cur_cloud->points.size() << endl;

    StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud (cur_cloud);
    sor.setMeanK (50);
    sor.setStddevMulThresh (1.0);
    sor.filter (*cur_cloud);
    cout << "2: the cloud point size is:" << cur_cloud->points.size() << endl;


    // Create the filtering object
    PassThrough<PointT> pass;
    pass.setInputCloud (cur_cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0.0, 0.6);
    pass.filter (*cur_cloud);

    pass.setInputCloud (cur_cloud);
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (0.02, 0.5);
    pass.filter (*cur_cloud);

    pass.setInputCloud (cur_cloud);
    pass.setFilterFieldName ("y");
    pass.setFilterLimits (0.02, 0.5);
    //pass.setFilterLimitsNegative (true);
    pass.filter (*cur_cloud);
    cout << "3: the cloud point size is:" << cur_cloud->points.size() << endl;


    transformPointCloud (*cloud_rgb_, *cloud_rgb_, key_trans_);
    StatisticalOutlierRemoval<PointXYZRGB> sor1;
    sor1.setInputCloud (cloud_rgb_);
    sor1.setMeanK (50);
    sor1.setStddevMulThresh (1.0);
    sor1.filter (*cloud_rgb_);

    // Create the filtering object
    PassThrough<PointXYZRGB> pass1;
    pass1.setInputCloud (cloud_rgb_);
    pass1.setFilterFieldName ("z");
    pass1.setFilterLimits (0.0, 0.6);
    pass1.filter (*cloud_rgb_);

    pass1.setInputCloud (cloud_rgb_);
    pass1.setFilterFieldName ("x");
    pass1.setFilterLimits (0.02, 0.5);
    pass1.filter (*cloud_rgb_);

    pass1.setInputCloud (cloud_rgb_);
    pass1.setFilterFieldName ("y");
    pass1.setFilterLimits (0.02, 0.5);
    //pass.setFilterLimitsNegative (true);
    pass1.filter (*cloud_rgb_);
}
/////////////////////////////////////////////////////////////////////////////////////////////////
double board::frameDistance(){
    return sqrt((key_tvec_[0] - cur_tvec_[0]) * (key_tvec_[0] - cur_tvec_[0]) + (key_tvec_[1] - cur_tvec_[1]) * (key_tvec_[1] - cur_tvec_[1]) + (key_tvec_[2] - cur_tvec_[2]) * (key_tvec_[2] - cur_tvec_[2]));
}
/////////////////////////////////////////////////////////////////////////////////////////////////
bool board::poseEstimation(Mat &image, Mat &depth_mat){
    Mat img_src = image.clone();
    vector<int> ids;
    vector<vector<Point2f>> corners;

    //the marker dictionary
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);

    //detect all markers in image
    aruco::detectMarkers(image, dictionary, corners, ids);


    if(ids.size() > 10){
        aruco::drawDetectedMarkers(image, corners, ids);

        int valid = aruco::estimatePoseBoard(corners, ids, board_ptr_, camera_matrics_, camera_dist_coffs_, cur_rvec_, cur_tvec_);

        if(valid > 0){
            PointCloud::Ptr cur_cloud (new PointCloud);
            aruco::drawAxis(image, camera_matrics_, camera_dist_coffs_, cur_rvec_, cur_tvec_, 0.1);
            imshow("src", img_src);

            if(abs(cur_rvec_[0]) >= 0){
                if(capture_frame_num_ == 0){
                    key_rvec_ = cur_rvec_;
                    key_tvec_ = cur_tvec_;
                    capture_frame_num_++;
                    convertXYZPointCloud(img_src, depth_mat, cur_cloud);
                    if(cur_cloud->points.size() == 0) return false;
                    merge_cloud_ = cur_cloud;


                    viewer->removePointCloud("source");
                    viewer->addPointCloud (cur_cloud, "source");
                    viewer->spin ();
                    writeDate(image, depth_mat, cur_cloud);

                }
                else{

                    if(frameDistance() > distance_tresh_){
                        cout << capture_frame_num_ << ": " << frameDistance() << endl;

                        key_rvec_ = cur_rvec_;
                        key_tvec_ = cur_tvec_;
                        capture_frame_num_++;
                        convertXYZPointCloud(img_src, depth_mat, cur_cloud);
                        if(cur_cloud->points.size() == 0) return false;

                        ICPTransform(merge_cloud_, cur_cloud, cur_cloud, key_trans_, true);
                        merge_cloud_ = cur_cloud;
                        cout << merge_cloud_->points.size() << endl;

                        viewer->removePointCloud("source");

                        viewer->addPointCloud (merge_cloud_, "source");
                        viewer->spin ();
                        writeDate(image, depth_mat, cur_cloud);


                        if(total_capture_frame_ == capture_frame_num_){
                            io::savePCDFile ("merge_cloud.pcd", *merge_cloud_, true);
//                            createMesh();
                            return false;
                        }

                    }
                }
            }
            return true;
        }
    }
    return false;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void board::ICPTransform(const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt, PointCloud::Ptr output, Eigen::Matrix4f &final_transform, bool downsample = false)
{
    //
    // Downsample for consistency and speed
    // \note enable this for large datasets
    PointCloud::Ptr src (new PointCloud);
    PointCloud::Ptr tgt (new PointCloud);
    pcl::VoxelGrid<PointT> grid;
    if (downsample)
    {
      grid.setLeafSize (0.05, 0.05, 0.05);
      grid.setInputCloud (cloud_src);
      grid.filter (*src);

      grid.setInputCloud (cloud_tgt);
      grid.filter (*tgt);
    }
    else
    {
      src = cloud_src;
      tgt = cloud_tgt;
    }


    // Compute surface normals and curvature
    PointCloudWithNormals::Ptr points_with_normals_src (new PointCloudWithNormals);
    PointCloudWithNormals::Ptr points_with_normals_tgt (new PointCloudWithNormals);

    NormalEstimation<PointT, PointNormalT> norm_est;
    search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
    norm_est.setSearchMethod (tree);
    norm_est.setKSearch (30);

    norm_est.setInputCloud (src);
    norm_est.compute (*points_with_normals_src);
    copyPointCloud (*src, *points_with_normals_src);

    norm_est.setInputCloud (tgt);
    norm_est.compute (*points_with_normals_tgt);
    copyPointCloud (*tgt, *points_with_normals_tgt);

    //
    // Instantiate our custom point representation (defined above) ...
    MyPointRepresentation point_representation;
    // ... and weight the 'curvature' dimension so that it is balanced against x, y, and z
    float alpha[4] = {1.0, 1.0, 1.0, 1.0};
    point_representation.setRescaleValues (alpha);

    //
    // Align
    IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;
    reg.setTransformationEpsilon (1e-6);
    // Set the maximum distance between two correspondences (src<->tgt) to 10cm
    // Note: adjust this based on the size of your datasets
    reg.setMaxCorrespondenceDistance (0.1);
    // Set the point representation
    reg.setPointRepresentation (boost::make_shared<const MyPointRepresentation> (point_representation));

    reg.setInputSource (points_with_normals_src);
    reg.setInputTarget (points_with_normals_tgt);



    //
    // Run the same optimization in a loop and visualize the results
    Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity (), prev, targetToSource;
    PointCloudWithNormals::Ptr reg_result = points_with_normals_src;
    reg.setMaximumIterations (2);
    for (int i = 0; i < 30; ++i)
    {
      PCL_INFO ("Iteration Nr. %d.\n", i);

      // save cloud for visualization purpose
      points_with_normals_src = reg_result;

      // Estimate
      reg.setInputSource (points_with_normals_src);
      reg.align (*reg_result);

          //accumulate transformation between each Iteration
      Ti = reg.getFinalTransformation () * Ti;

          //if the difference between this transformation and the previous one
          //is smaller than the threshold, refine the process by reducing
          //the maximal correspondence distance
      if (fabs ((reg.getLastIncrementalTransformation () - prev).sum ()) < reg.getTransformationEpsilon ())
        reg.setMaxCorrespondenceDistance (reg.getMaxCorrespondenceDistance () - 0.001);

      prev = reg.getLastIncrementalTransformation ();

      // visualize current state
//      showCloudsRight(points_with_normals_tgt, points_with_normals_src);
    }

      //
    // Get the transformation from target to source
    targetToSource = Ti.inverse();
    //
    // Transform target back in source frame
    transformPointCloud (*cloud_tgt, *output, targetToSource);

//    viewer->removePointCloud ("source");
//    viewer->removePointCloud ("target");

//    PointCloudColorHandlerCustom<PointT> cloud_tgt_h (output, 0, 255, 0);
//    PointCloudColorHandlerCustom<PointT> cloud_src_h (cloud_src, 255, 0, 0);
//    viewer->addPointCloud (output, cloud_tgt_h, "target", vp_2);
//    viewer->addPointCloud (cloud_src, cloud_src_h, "source", vp_2);

      PCL_INFO ("Press q to continue the registration.\n");

//    viewer->removePointCloud ("source");
//    viewer->removePointCloud ("target");

    //add the source to the transformed target
    *output += *cloud_src;

    final_transform = targetToSource;
    key_trans_ = targetToSource;


}
/////////////////////////////////////////////////////////////////////////////////////////////////
void board::createMesh(){
    // Normal estimation*
     NormalEstimation<PointT, Normal> n;
     pcl::PointCloud<Normal>::Ptr normals (new pcl::PointCloud<Normal>);
     search::KdTree<PointT>::Ptr tree (new search::KdTree<PointT>);
     tree->setInputCloud (merge_cloud_);
     n.setInputCloud (merge_cloud_);
     n.setSearchMethod (tree);
     n.setKSearch (20);
     n.compute (*normals);
     cout << "compute normals!" << endl;

     // Concatenate the XYZ and normal fields*
    PointCloudWithNormals::Ptr cloud_with_normals (new  PointCloudWithNormals);
    concatenateFields (*merge_cloud_, *normals, *cloud_with_normals);
    cout << "concatenateFields!" << endl;

    //* cloud_with_normals = cloud + normals

    // Create search tree*
    search::KdTree<PointNormalT>::Ptr tree2 (new search::KdTree<PointNormalT>);
    tree2->setInputCloud (cloud_with_normals);
    cout << "create search tree!" << endl;

    PolygonMesh triangles;
    MarchingCubes<PointNormalT> *m;

    m = new MarchingCubesHoppe<PointNormalT>() ;
    m->setGridResolution(10, 10, 10);
//        m->setPercentageExtendGrid(0.2);
    m->setIsoLevel(0.01);
    m->setInputCloud(cloud_with_normals);
    m->setSearchMethod(tree2);
    m->reconstruct(triangles);


//    // Initialize objects
//    GreedyProjectionTriangulation<PointNormalT> gp3;

//    // Set the maximum distance between connected points (maximum edge length)
//    gp3.setSearchRadius (0.025);

//    // Set typical values for the parameters
//    gp3.setMu (2.5);
//    gp3.setMaximumNearestNeighbors (100);
//    gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
//    gp3.setMinimumAngle(M_PI/18); // 10 degrees
//    gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
//    gp3.setNormalConsistency(false);

//    // Get result
//    gp3.setInputCloud (cloud_with_normals);
//    gp3.setSearchMethod (tree2);
//    gp3.reconstruct (triangles);
//    cout << "fineshed reconstruct!" << endl;
    //save triangles
    io::savePLYFile ("merge_mesh.ply", triangles);
}
