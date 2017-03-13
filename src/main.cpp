#include "include/board_pose_estimation.h"
#include <pcl/io/openni2_grabber.h>
#include <pcl/io/png_io.h>
#include <fstream>

using namespace std;
using namespace cv;
using namespace pcl;



static int num_s1 = 0, num_s2 = 0, num_s3 = 0;
Vec3d pre_rvec, pre_tvec;

board b;

void callback (const boost::shared_ptr<io::Image>& rgb, const boost::shared_ptr<io::DepthImage>& depth, float constant)
{


    int sizes[2] = {480, 640};
    Mat rgb_mat(2, sizes, CV_8UC3);
    Mat depth_mat(2, sizes,CV_16UC1);

    rgb->fillRGB(640, 480, rgb_mat.data, 640*3);
    cvtColor(rgb_mat,rgb_mat,CV_RGB2BGR);

    depth->fillDepthImageRaw(640, 480, (unsigned short*) depth_mat.data);

//    imshow("rgb", rgb_mat);
//    imshow("depth", depth_mat*16);

    Vec3d rvec, tvec;

    if(b.poseEstimation(rgb_mat, depth_mat))
    {
        vector<int> params;
        params.push_back(CV_IMWRITE_PNG_COMPRESSION);
        params.push_back(0);
        stringstream rgbFile, depthFile;
        rgbFile << "..//image//seq_color" << setw( 5 ) << setfill( '0' ) << num_s1 << ".jpg";
        depthFile << "..//image//cloud_" << setw( 5 ) << setfill( '0' ) << num_s1 << ".pcd";
//        imwrite(rgbFile.str(), rgb_mat);
//        imwrite(depthFile.str(), depth_mat, params);
//        io::savePCDFile (depthFile.str(), *cloud, true);
        imshow("mat", rgb_mat);
        ++ num_s1;
        pre_rvec = rvec;
        pre_tvec = tvec;
    }

}
int main (int argc, char** argv)
{
    if(argc < 2){
        PCL_INFO("The input arguments less than neccesary!");
        return -1;
    }
    b.createBoard();
    b.readCameraParam(argv[1]);


    Grabber* interface = new io::OpenNI2Grabber("#1");
    if (interface == 0)
        return -1;
    boost::function<void (const boost::shared_ptr<io::Image>&, const boost::shared_ptr<io::DepthImage>&, float)> f(&callback);
    interface->registerCallback (f);
    interface->start ();
    while (true)
    {
        boost::this_thread::sleep (boost::posix_time::seconds (1));
        if((char)waitKey(30) == 'q')
            break;
    }

    interface->stop ();
    return 0;
}
