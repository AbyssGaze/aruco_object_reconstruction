#ifndef LABEL_MULTI_VIEW_H
#define LABEL_MULTI_VIEW_H

#include <iostream>
#include <string>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>


class lable{
public:
    //convenient typedefs
    typedef pcl::PointXYZ PointT;
    typedef pcl::PointCloud<PointT> PointCloud;
    typedef pcl::PointNormal PointNormalT;
    typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

public:
    void readModel(std::string fileName);
    void ICPMatch();
private:
    PointCloud model_;


};


#endif
