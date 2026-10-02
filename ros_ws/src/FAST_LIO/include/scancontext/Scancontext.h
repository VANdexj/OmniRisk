#pragma once

#include <ctime>
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>
#include <algorithm> 
#include <cstdlib>
#include <memory>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

#include "scancontext/nanoflann.hpp"
#include "scancontext/KDTreeVectorOfVectorsAdaptor.h"
#include "tic_toc.h"

using namespace Eigen;
using namespace nanoflann;

using std::cout;
using std::endl;
using std::make_pair;

using std::atan2;
using std::cos;
using std::sin;

using SCPointType = pcl::PointXYZ; // using xyz only. but a user can exchange the original bin encoding function (i.e., max hegiht) to max intensity (for detail, refer 20 ICRA Intensity Scan Context)
using KeyMat = std::vector<std::vector<float> >;
using InvKeyTree = KDTreeVectorOfVectorsAdaptor< KeyMat, float >;


// namespace SC2
// {

void coreImportTest ( void );


// sc param-independent helper functions 
float xy2theta( const float & _x, const float & _y );
MatrixXd circshift( MatrixXd &_mat, int _num_shift );
std::vector<float> eig2stdvec( MatrixXd _eigmat );


class SCManager
{
public: 
    SCManager( ) = default; // reserving data space (of std::vector) could be considered. but the descriptor is lightweight so don't care.

    Eigen::MatrixXd makeScancontext( pcl::PointCloud<SCPointType> & _scan_down );
    Eigen::MatrixXd makeRingkeyFromScancontext( Eigen::MatrixXd &_desc );
    Eigen::MatrixXd makeSectorkeyFromScancontext( Eigen::MatrixXd &_desc );

    int fastAlignUsingVkey ( MatrixXd & _vkey1, MatrixXd & _vkey2 ); 
    double distDirectSC ( MatrixXd &_sc1, MatrixXd &_sc2 ); // "d" (eq 5) in the original paper (IROS 18)
    std::pair<double, int> distanceBtnScanContext ( MatrixXd &_sc1, MatrixXd &_sc2 ); // "D" (eq 6) in the original paper (IROS 18)

    // User-side API
    void makeAndSaveScancontextAndKeys( pcl::PointCloud<SCPointType> & _scan_down );
    std::pair<int, float> detectLoopClosureID( void ); // int: nearest node index, float: relative yaw  
    std::pair<int, float> detect_init_pose ( bool print_detail_score );
    // for ltslam 
    // User-side API for multi-session
    void saveScancontextAndKeys( Eigen::MatrixXd _scd );
    std::pair<int, float> detectLoopClosureIDBetweenSession ( std::vector<float>& curr_key,  Eigen::MatrixXd& curr_desc);

    const Eigen::MatrixXd& getConstRefRecentSCD(void);

public:
    // hyper parameters ()
    const double LIDAR_HEIGHT = 0.0; // lidar height : add this for simply directly using lidar scan in the lidar local coord (not robot base coord) / if you use robot-coord-transformed lidar scans, just set this as 0.

    int    PC_NUM_RING = 20; // 20 in the original paper (IROS 18)
    int    PC_NUM_SECTOR = 60; // 60 in the original paper (IROS 18)
    double PC_MAX_RADIUS = 80; // 80 meter max in the original paper (IROS 18)
    double PC_UNIT_SECTORANGLE = 360.0 / double(PC_NUM_SECTOR);
    double PC_UNIT_RINGGAP = PC_MAX_RADIUS / double(PC_NUM_RING);

    // tree
    const int    NUM_EXCLUDE_RECENT = 30; // simply just keyframe gap (related with loopClosureFrequency in yaml), but node position distance-based exclusion is ok. 
    const int    NUM_CANDIDATES_FROM_TREE = 3; // 10 is enough. (refer the IROS 18 paper)

    // loop thres
    const double SEARCH_RATIO = 0.1; // for fast comparison, no Brute-force, but search 10 % is okay. // not was in the original conf paper, but improved ver.
    // const double SC_DIST_THRES = 0.13; // empirically 0.1-0.2 is fine (rare false-alarms) for 20x60 polar context (but for 0.15 <, DCS or ICP fit score check (e.g., in LeGO-LOAM) should be required for robustness)

    double SC_DIST_THRES = 0.2; // 0.4-0.6 is good choice for using with robust kernel (e.g., Cauchy, DCS) + icp fitness threshold / if not, recommend 0.1-0.15
    // const double SC_DIST_THRES = 0.7; // 0.4-0.6 is good choice for using with robust kernel (e.g., Cauchy, DCS) + icp fitness threshold / if not, recommend 0.1-0.15

    // config 
    const int    TREE_MAKING_PERIOD_ = 30; // i.e., remaking tree frequency, to avoid non-mandatory every remaking, to save time cost / in the LeGO-LOAM integration, it is synchronized with the loop detection callback (which is 1Hz) so it means the tree is updated evrey 10 sec. But you can use the smaller value because it is enough fast ~ 5-50ms wrt N.
    int          tree_making_period_conter = 0;

    // setter
    void setSCdistThres(double _new_thres);
    void setMaximumRadius(double _max_r);

    // data 
    std::vector<double> polarcontexts_timestamp_; // optional.
    std::vector<Eigen::MatrixXd> polarcontexts_;
    std::vector<Eigen::MatrixXd> polarcontext_invkeys_;
    std::vector<Eigen::MatrixXd> polarcontext_vkeys_;

    KeyMat polarcontext_invkeys_mat_;
    KeyMat polarcontext_invkeys_to_search_;
    std::unique_ptr<InvKeyTree> polarcontext_tree_;

    bool is_tree_batch_made = false;
    std::unique_ptr<InvKeyTree> polarcontext_tree_batch_;

    Eigen::MatrixXd target_scancontext;
    Eigen::MatrixXd source_scancontext;
    std::vector<Eigen::Matrix3d> custom_R_candidate;
    std::vector<Eigen::Vector3d> custom_T_candidate;
    std::vector<pcl::PointCloud<pcl::PointXYZ>> target_cloud_candidate_;


}; // SCManager


class SCViewer
{
public:
    SCViewer( ) = default; 

    void displayScanContext(const Eigen::MatrixXd& target_desc, const Eigen::MatrixXd& source_desc)
    {
        // cout << "desc matrix size: " << desc.rows() << " x " << desc.cols() << std::endl;
        if(target_desc.rows() > 0 && target_desc.cols() > 0 && source_desc.rows() > 0 && source_desc.cols() > 0)
        {
            cv::Mat combined_desc_img;
            cv::Mat target_desc_img(target_desc.rows(), target_desc.cols(), CV_8UC3);
            double target_min_val = target_desc.minCoeff();
            double target_max_val = target_desc.maxCoeff();
            cv::Mat source_desc_img(source_desc.rows(), source_desc.cols(), CV_8UC3);
            double source_min_val = source_desc.minCoeff();
            double source_max_val = source_desc.maxCoeff();

            double min_val = std::min(target_min_val, source_min_val);
            double max_val = std::max(target_max_val, source_max_val);
            for (int row_idx = 0; row_idx < target_desc.rows(); row_idx++)
            {
                for (int col_idx = 0; col_idx < target_desc.cols(); col_idx++)
                {
                    double target_normalized_val = (target_desc(row_idx, col_idx) - min_val) / (max_val - min_val);
                    uchar b, g, r;
                    if (target_normalized_val <= 0.25) {
                        b = static_cast<uchar>(255);
                        g = static_cast<uchar>(255 * target_normalized_val * 4);
                        r = 0;
                    } else if (target_normalized_val <= 0.5) {
                        b = static_cast<uchar>(255 * (1 - (target_normalized_val - 0.25) * 4));
                        g = 255;
                        r = 0;
                    } else if (target_normalized_val <= 0.75) {
                        b = 0;
                        g = static_cast<uchar>(255);
                        r = static_cast<uchar>(255 * (target_normalized_val - 0.5) * 4);
                    } else {
                        b = 0;
                        g = static_cast<uchar>(255 * (1 - (target_normalized_val - 0.75) * 4));
                        r = 255;
                    }
                    target_desc_img.at<cv::Vec3b>(row_idx, col_idx) = cv::Vec3b(b, g, r);
                }
            }
            for (int row_idx = 0; row_idx < source_desc.rows(); row_idx++)
            {
                for (int col_idx = 0; col_idx < source_desc.cols(); col_idx++)
                {
                    double source_normalized_val = (source_desc(row_idx, col_idx) - min_val) / (max_val - min_val);
                    uchar b, g, r;
                    if (source_normalized_val <= 0.25) {
                        b = static_cast<uchar>(255);
                        g = static_cast<uchar>(255 * source_normalized_val * 4);
                        r = 0;
                    } else if (source_normalized_val <= 0.5) {
                        b = static_cast<uchar>(255 * (1 - (source_normalized_val - 0.25) * 4));
                        g = 255;
                        r = 0;
                    } else if (source_normalized_val <= 0.75) {
                        b = 0;
                        g = static_cast<uchar>(255);
                        r = static_cast<uchar>(255 * (source_normalized_val - 0.5) * 4);
                    } else {
                        b = 0;
                        g = static_cast<uchar>(255 * (1 - (source_normalized_val - 0.75) * 4));
                        r = 255;
                    }
                    source_desc_img.at<cv::Vec3b>(row_idx, col_idx) = cv::Vec3b(b, g, r);
                }
            }
            if (target_desc_img.rows == source_desc_img.rows) 
            { 
                cv::Mat temp;
                int spacing_width = 5;
                cv::hconcat(target_desc_img, cv::Mat::zeros(target_desc_img.rows, spacing_width, target_desc_img.type()), temp);
                cv::hconcat(temp, source_desc_img, combined_desc_img);
                // cv::hconcat(target_desc_img, source_desc_img, combined_desc_img);
                cv::namedWindow("ScanContext", cv::WINDOW_NORMAL);
                cv::resizeWindow("ScanContext", combined_desc_img.cols, combined_desc_img.rows);
                cv::imshow("ScanContext", combined_desc_img);
            } 
            else 
            {
                std::cout << "Error: The two Scancontexts must have the same dimensions." << std::endl;
            }            
            cv::waitKey(1);
        }
    }
};






















// } // namespace SC2
