#include "ekf.h"

#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Range.h>
#include <nav_msgs/Odometry.h>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <unsupported/Eigen/MatrixFunctions>
// #include <geometry_msgs/Accel.h>
#include "conversion.h"

using namespace std;
using namespace Eigen;
// 20200531: time synchronization
// 20200105: ekf_node_vio.cpp and ekf_node_mocap.cpp merge into one (ekf_node_vio.cpp) and the differences between them is the odom format
// X_state: p q v gb ab   with time stamp aligned between imu and img
/*
    EKF model
    prediction:
    xt~ = xt-1 + dt*f(xt-1, ut, 0)
    sigmat~ = Ft*sigmat-1*Ft' + Vt*Qt*Vt'
    Update:
    Kt = sigmat~*Ct'*(Ct*sigmat~*Ct' + Wt*Rt*Wt')^-1
    xt = xt~ + Kt*(zt - g(xt~,0))
    sigmat = sigmat~ - Kt*Ct*sigmat~
*/
/*
   -pi ~ pi crossing problem:
   1. the model prpagation: X_state should be limited to [-pi,pi] after predicting and updating
   2. inovation crossing: (measurement - g(X_state)) should also be limited to [-pi,pi] when getting the inovation.
   z_measurement is normally in [-pi~pi]
*/

// imu frame is imu body frame

// odom: pose px,py pz orientation qw qx qy qz
// imu: acc: x y z gyro: wx wy wz

#define TimeSync 1 // time synchronize or not
#define RePub 0    // re publish the odom when repropagation

#define POS_DIFF_THRESHOLD (0.3f)

constexpr double kOdomBufferToleranceMs = 1.0;

ros::Publisher odom_pub, ahead_odom_pub;
ros::Publisher cam_odom_pub;
ros::Publisher acc_filtered_pub;
ros::Time imu_back_time = ros::Time(0), imu_front_time = ros::Time(0);

// state
geometry_msgs::Pose pose;
Vector3d position, orientation, velocity;

// Now set up the relevant matrices
// states X [p q pdot]  [px,py,pz, wx,wy,wz, vx,vy,vz]
size_t stateSize;            // x = [p q pdot bg ba]
size_t errorstateSize;       // x = [p q pdot bg ba]
size_t stateSize_pqv;        // x = [p q pdot]
size_t measurementSize;      // z = [p q]
size_t inputSize;            // u = [w a]
VectorXd X_state(stateSize); // x (in most literature)
VectorXd u_input;
VectorXd Z_measurement;              // z
MatrixXd StateCovariance;            // sigma
MatrixXd Kt_kalmanGain;              // Kt
VectorXd X_state_correct(stateSize); // x (in most literature)
MatrixXd StateCovariance_correct;    // sigma
// MatrixXd Ct_stateToMeasurement;                  // Ct
//  VectorXd innovation;                         // z - Hx

MatrixXd Qt;
MatrixXd Rt;
Vector3d u_gyro;
Vector3d u_acc;
Vector3d gravity(0., 0., -9.8); // need to estimate the bias 9.8099
Vector3d bg_0(0., 0., 0);       // need to estimate the bias
Vector3d ba_0(0., 0., 0);       // need to estimate the bias  0.1
Vector3d ng(0., 0., 0.);
Vector3d na(0., 0., 0.);
Vector3d nbg(0., 0., 0.);
Vector3d nba(0., 0., 0.);
Matrix3d rotation_imu;

// Vector3d q_last;
//! changed by wz
Quaterniond q_last;
Vector3d bg_last;
Vector3d ba_last;

// Qt imu covariance matrix  smaller believe system(imu) more
double imu_trans_x = 0.0;
double imu_trans_y = 0.0;
double imu_trans_z = 0.0;
double gyro_cov = 0.01;
double acc_cov = 0.01;
// Rt visual odomtry covariance smaller believe measurement more
double position_cov = 0.1;
double q_rp_cov = 0.1;
double q_yaw_cov = 0.1;
double scale_g;
double dt = 0.005; // second
double t_last, t_now;
bool first_frame_imu = true;
bool first_frame_tag_odom = true;
bool test_odomtag_call = false;
bool odomtag_call = false;

double time_now, time_last;
double time_odom_tag_now;
// double diff_time;

double cutoff_freq = 20;
string world_frame_id = "world";

//zyh added
double offset_px, offset_py, offset_pz;
Eigen::Vector3d first_odom_get_;

// world frame points velocity
deque<pair<VectorXd, sensor_msgs::Imu>> sys_seq;
deque<MatrixXd> cov_seq;
double dt_0_rp; // the dt for the first frame in repropagation
void seq_keep(const sensor_msgs::Imu::ConstPtr &imu_msg)
{
#define seqsize 100
    if (sys_seq.size() < seqsize)
    {
        sys_seq.push_back(make_pair(X_state, *imu_msg)); // X_state before propagation and imu at that time
        cov_seq.push_back(StateCovariance);
    }
    else
    {
        sys_seq.pop_front();
        sys_seq.push_back(make_pair(X_state, *imu_msg));
        cov_seq.pop_front();
        cov_seq.push_back(StateCovariance);
    }
    imu_front_time = sys_seq.front().second.header.stamp;
    imu_back_time = sys_seq.back().second.header.stamp;
    // ensure that the later frame time > the former one
}
// choose the coordinate frame imu for the measurement
bool search_proper_frame(double odom_time)
{
    if (sys_seq.size() == 0)
    {
        ROS_ERROR("sys_seq.size() == 0. if appear this error, should check the code");
        return false;
    }
    if (sys_seq.size() == 1)
    {
        ROS_ERROR("sys_seq.size() == 0. if appear this error, should check the code");
        return false;
    }

    size_t rightframe = sys_seq.size() - 1;
    bool find_proper_frame = false;
    for (size_t i = 1; i < sys_seq.size(); i++) // TODO: it better to search from the middle instead in the front
    {
        double time_before = odom_time - sys_seq[i - 1].second.header.stamp.toSec();
        double time_after = odom_time - sys_seq[i].second.header.stamp.toSec();
        if ((time_before >= 0) && (time_after < 0))
        {
            if (abs(time_before) > abs(time_after))
            {
                rightframe = i;
            }
            else
            {
                rightframe = i - 1;
            }

            if (rightframe != 0)
            {
                dt_0_rp = sys_seq[rightframe].second.header.stamp.toSec() - sys_seq[rightframe - 1].second.header.stamp.toSec();
            }
            else
            { // if rightframe is the first frame in the seq, set dt_0_rp as the next dt
                dt_0_rp = sys_seq[rightframe + 1].second.header.stamp.toSec() - sys_seq[rightframe].second.header.stamp.toSec();
            }

            find_proper_frame = true;
            break;
        }
    }
    if (!find_proper_frame)
    {
        if ((odom_time - sys_seq[0].second.header.stamp.toSec()) <= 0) // if odom time before the first frame, set first frame
        {
            rightframe = 0;
            // if rightframe is the first frame in the seq, set dt_0_rp as the next dt
            dt_0_rp = sys_seq[rightframe + 1].second.header.stamp.toSec() - sys_seq[rightframe].second.header.stamp.toSec();
        }
        if ((odom_time - sys_seq[sys_seq.size() - 1].second.header.stamp.toSec()) >= 0) // if odom time after the last frame, set last frame
        {
            rightframe = sys_seq.size() - 1;
            dt_0_rp = sys_seq[rightframe].second.header.stamp.toSec() - sys_seq[rightframe - 1].second.header.stamp.toSec();
        }
        // no process, set the latest one
    }

    // set the right frame as the first frame in the queue
    for (size_t i = 0; i < rightframe; i++)
    {
        sys_seq.pop_front();
        cov_seq.pop_front();
    }

    if (find_proper_frame)
    {
        return true;
    }
    else
    {
        return false;
    }
}
void re_propagate()
{
    for (size_t i = 1; i < sys_seq.size(); i++)
    {
        // re-prediction for the rightframe
        dt = sys_seq[i].second.header.stamp.toSec() - sys_seq[i - 1].second.header.stamp.toSec();

        u_gyro(0) = sys_seq[i].second.angular_velocity.x;
        u_gyro(1) = sys_seq[i].second.angular_velocity.y;
        u_gyro(2) = sys_seq[i].second.angular_velocity.z;
        u_acc(0) = sys_seq[i].second.linear_acceleration.x;
        u_acc(1) = sys_seq[i].second.linear_acceleration.y;
        u_acc(2) = sys_seq[i].second.linear_acceleration.z;

        MatrixXd Ft;
        MatrixXd Vt;

        // q_last = sys_seq[i].first.segment<3>(3);   // last X2
        //! changed by wz
        // q_last = sys_seq[i].first.segment<4>(3); // last X2
        q_last.w() = sys_seq[i].first(3);
        q_last.x() = sys_seq[i].first(4);
        q_last.y() = sys_seq[i].first(5);
        q_last.z() = sys_seq[i].first(6);

        // bg_last = sys_seq[i].first.segment<3>(9);  // last X4
        // ba_last = sys_seq[i].first.segment<3>(12); // last X5
        //! changed by wz
        bg_last = sys_seq[i].first.segment<3>(10); // last X4
        ba_last = sys_seq[i].first.segment<3>(13); // last X5

        // Ft = MatrixXd::Identity(stateSize, stateSize) + dt * diff_f_diff_x(q_last, u_gyro, u_acc, bg_last, ba_last);
        //! changed by wz
        Ft = MatrixXd::Identity(errorstateSize, errorstateSize) + dt * diff_f_diff_x(q_last, u_gyro, u_acc, bg_last, ba_last);

        Vt = dt * diff_f_diff_n(q_last);

        // X_state += dt * F_model(u_gyro, u_acc);
        //! changed by wz
        X_state = upate_state_Quaterniond_F_model(X_state, u_gyro, u_acc, dt);

        // if (X_state(3) > PI)
        //     X_state(3) -= 2 * PI;
        // if (X_state(3) < -PI)
        //     X_state(3) += 2 * PI;
        // if (X_state(4) > PI)
        //     X_state(4) -= 2 * PI;
        // if (X_state(4) < -PI)
        //     X_state(4) += 2 * PI;
        // if (X_state(5) > PI)
        //     X_state(5) -= 2 * PI;
        // if (X_state(5) < -PI)
        //     X_state(5) += 2 * PI;  //! changed by wz
        StateCovariance = Ft * StateCovariance * Ft.transpose() + Vt * Qt * Vt.transpose();

#if RePub
        system_pub(X_state, sys_seq[i].second.header.stamp); // choose to publish the repropagation or not
#endif
    }
}

void imu_callback(const sensor_msgs::Imu::ConstPtr &msg)
{
    // wmywmy
    // imu_time = msg->header.stamp;
    sensor_msgs::Imu::Ptr new_msg(new sensor_msgs::Imu(*msg));
    Eigen::Vector3d temp1, temp2;

    temp1[0] = new_msg->linear_acceleration.x;
    temp1[1] = new_msg->linear_acceleration.y;
    temp1[2] = new_msg->linear_acceleration.z;
    temp2 = rotation_imu * temp1;
    new_msg->linear_acceleration.x = scale_g * temp2[0];
    new_msg->linear_acceleration.y = scale_g * temp2[1];
    new_msg->linear_acceleration.z = scale_g * temp2[2];

    temp1[0] = new_msg->angular_velocity.x;
    temp1[1] = new_msg->angular_velocity.y;
    temp1[2] = new_msg->angular_velocity.z;
    temp2 = rotation_imu * temp1;
    new_msg->angular_velocity.x = temp2[0];
    new_msg->angular_velocity.y = temp2[1];
    new_msg->angular_velocity.z = temp2[2];
    // wmywmy

    // seq_keep(msg);
    // nav_msgs::Odometry odom_fusion;
    // your code for propagation
    if (!first_frame_tag_odom)
    { // get the initial pose and orientation in the first frame of measurement
        if (first_frame_imu)
        {
            first_frame_imu = false;
            time_now = new_msg->header.stamp.toSec();
            time_last = time_now;
#if TimeSync
            seq_keep(new_msg); // keep before propagation
#endif

            system_pub(X_state, new_msg->header.stamp);
            // cout << "first frame imu" << endl;
        }
        else
        {
#if TimeSync
            seq_keep(new_msg); // keep before propagation
#endif
            // cout << "\033[1;32m[ INFO] [IMU] [TimeSync] [seq_keep] [OK] \033[0m" << endl;
            time_now = new_msg->header.stamp.toSec();
            dt = time_now - time_last;

            if (odomtag_call)
            {
                odomtag_call = false;
                // diff_time = time_now - time_odom_tag_now;
                // if(diff_time<0)
                // {
                //     cout << "diff time: " << diff_time << endl;  //???!!! exist !!!???
                //     cout << "timeimu: " << time_now - 1.60889e9 << " time_odom: " << time_odom_tag_now - 1.60889e9 << endl;
                //     // cout << "diff time: " << diff_time << endl;  //about 30ms
                // }
            }
            MatrixXd Ft;
            MatrixXd Vt;

            u_gyro(0) = new_msg->angular_velocity.x;
            u_gyro(1) = new_msg->angular_velocity.y;
            u_gyro(2) = new_msg->angular_velocity.z;
            u_acc(0) = new_msg->linear_acceleration.x;
            u_acc(1) = new_msg->linear_acceleration.y;
            u_acc(2) = new_msg->linear_acceleration.z;

            // q_last = X_state.segment<3>(3);   // last X2
            // bg_last = X_state.segment<3>(9);  // last X4
            // ba_last = X_state.segment<3>(12); // last X5
            //! changed by wz
            q_last.w() = X_state(3);
            q_last.x() = X_state(4);
            q_last.y() = X_state(5);
            q_last.z() = X_state(6);
            // cout << "q_last" << endl
            //      << q_last << endl;

            bg_last = X_state.segment<3>(10); // last X4
            // cout << "bg_last" << endl
            //  << bg_last << endl;

            ba_last = X_state.segment<3>(13); // last X5

            // cout << "ba_last" << endl
            //  << ba_last << endl;
            //! changed by wz
            // cout << "dt:" << dt << endl;
            Ft = MatrixXd::Identity(errorstateSize, errorstateSize) + dt * diff_f_diff_x(q_last, u_gyro, u_acc, bg_last, ba_last);

            // cout << "Ft" << endl
            //      << Ft << endl;

            Vt = dt * diff_f_diff_n(q_last);
            // cout << "Vt" << endl
            //      << Vt << endl;

            // acc_f_pub(u_acc, msg->header.stamp);
            // X_state += dt * F_model(u_gyro, u_acc);
            //! changed by wz
            // cout << "X_state_old" << endl
            //      << X_state << endl;
            X_state = upate_state_Quaterniond_F_model(X_state, u_gyro, u_acc, dt);
            // cout << "X_state" << endl
            //      << X_state << endl;

            // if (X_state(3) > PI)  //! changed by wz
            //     X_state(3) -= 2 * PI;
            // if (X_state(3) < -PI)
            //     X_state(3) += 2 * PI;
            // if (X_state(4) > PI)
            //     X_state(4) -= 2 * PI;
            // if (X_state(4) < -PI)
            //     X_state(4) += 2 * PI;
            // if (X_state(5) > PI)
            //     X_state(5) -= 2 * PI;
            // if (X_state(5) < -PI)
            //     X_state(5) += 2 * PI;
            // cout << "!!!!!!!!!!!!!!!!!!!!position:" << X_state(0) << " " << X_state(1) << " " << X_state(2) << endl;

            StateCovariance = Ft * StateCovariance * Ft.transpose() + Vt * Qt * Vt.transpose();

            time_last = time_now;

            // Eigen::VectorXd X_state_ahead = X_state + 0.01 * F_model(u_gyro, u_acc);
            //! changed by wz
            Eigen::VectorXd X_state_ahead = upate_state_Quaterniond_F_model(X_state, u_gyro, u_acc, 0.01);

            // if(test_odomtag_call) //no frequency boost
            // {
            //     test_odomtag_call = false;
            //     system_pub(msg->header.stamp);
            // }
            system_pub(X_state, new_msg->header.stamp);
            ahead_system_pub(X_state_ahead, new_msg->header.stamp);

            // cout << "[IMU] [TimeSync] [OK]" << endl;

            // system_pub(X_state, ros::Time::now());
            // ahead_system_pub(X_state_ahead, ros::Time::now());
        }
    }
}

// Rotation from the camera frame to the IMU frame
Matrix3d Rc_i;
Vector3d tc_i; //  cam in imu frame
int cnt = 0;
Vector3d INNOVATION_;
Matrix3d Rr_i;
Vector3d tr_i; //  rigid body in imu frame
// msg is imu in world
// VectorXd get_pose_from_VIOodom(const nav_msgs::Odometry::ConstPtr &msg)
// {
//     Matrix3d Rr_w; // rigid body in world
//     Vector3d tr_w;
//     Matrix3d Ri_w;
//     Vector3d ti_w;
//     Vector3d p_temp;
//     p_temp(0) = msg->pose.pose.position.x;
//     p_temp(1) = msg->pose.pose.position.y;
//     p_temp(2) = msg->pose.pose.position.z;
//     // quaternion2euler:  ZYX  roll pitch yaw
//     Quaterniond q;
//     q.w() = msg->pose.pose.orientation.w;
//     q.x() = msg->pose.pose.orientation.x;
//     q.y() = msg->pose.pose.orientation.y;
//     q.z() = msg->pose.pose.orientation.z;

//     // Euler transform
//     //  Ri_w = q.toRotationMatrix();
//     //  ti_w = p_temp;
//     Rr_w = q.toRotationMatrix();
//     tr_w = p_temp;
//     Ri_w = Rr_w * Rr_i.inverse();
//     ti_w = tr_w - Ri_w * tr_i;
//     Vector3d euler = mat2euler(Ri_w);

//     VectorXd pose = VectorXd::Random(6);
//     pose.segment<3>(0) = ti_w;
//     pose.segment<3>(3) = euler;

//     return pose;
// }

//! changed by wz
VectorXd get_pose_from_VIOodom(const nav_msgs::Odometry::ConstPtr &msg)
{
    // cout << "get_pose_from_VIOodom" << endl;
    Matrix3d Rr_w; // rigid body in world
    Vector3d tr_w;
    Matrix3d Ri_w;
    Vector3d ti_w;
    Vector3d p_temp;
    p_temp(0) = msg->pose.pose.position.x;
    p_temp(1) = msg->pose.pose.position.y;
    p_temp(2) = msg->pose.pose.position.z;
    // quaternion2euler:  ZYX  roll pitch yaw
    Quaterniond q;
    q.w() = msg->pose.pose.orientation.w;
    q.x() = msg->pose.pose.orientation.x;
    q.y() = msg->pose.pose.orientation.y;
    q.z() = msg->pose.pose.orientation.z;

    // Euler transform
    //  Ri_w = q.toRotationMatrix();
    //  ti_w = p_temp;
    Rr_w = q.toRotationMatrix();
    tr_w = p_temp;
    Ri_w = Rr_w * Rr_i.inverse();
    ti_w = tr_w - Ri_w * tr_i;
    // Vector3d euler = mat2euler(Ri_w);
    //! changed by wz
    Quaterniond q_wi = Quaterniond(Ri_w);

    // VectorXd pose = VectorXd::Random(6);
    //! changed by wz
    VectorXd pose = VectorXd::Random(7);
    pose.segment<3>(0) = ti_w;
    // pose.segment<3>(3) = euler;
    //! changed by wz
    pose.segment<4>(3) = Vector4d(q_wi.w(), q_wi.x(), q_wi.y(), q_wi.z());

    // cout << "pose: " << pose << endl;

    return pose;
}

void update_lastest_state()
{
    MatrixXd Ct;
    MatrixXd Wt;
    Ct = diff_g_diff_x();
    // cout << "Ct: " << Ct << endl;
    Wt = diff_g_diff_v();
    // cout << "Wt: " << Wt << endl;

    Kt_kalmanGain = StateCovariance * Ct.transpose() * (Ct * StateCovariance * Ct.transpose() + Wt * Rt * Wt.transpose()).inverse();
    // cout << "Kt_kalmanGain: " << Kt_kalmanGain << endl;
    VectorXd gg = g_model();
    // VectorXd innovation = Z_measurement - gg;
    // VectorXd innovation_t = gg;
    VectorXd innovation = VectorXd::Zero(6);
    innovation.segment<3>(0) = Z_measurement.segment<3>(0) - gg.segment<3>(0);
    Quaterniond q_gg(gg(3), gg(4), gg(5), gg(6));
    Quaterniond q_Z_measurement(Z_measurement(3), Z_measurement(4), Z_measurement(5), Z_measurement(6));
    innovation.segment<3>(3) = (q_Z_measurement * q_gg.inverse()).vec();
    VectorXd innovation_t = gg;
    // Prevent innovation changing suddenly when euler from -Pi to Pi
    float pos_diff = sqrt(innovation(0) * innovation(0) + innovation(1) * innovation(1) + innovation(2) * innovation(2));
    if (pos_diff > POS_DIFF_THRESHOLD)
    {
        ROS_ERROR("posintion diff too much between measurement and model prediction!!!   pos_diff setting: %f  but the diff measured is %f ", POS_DIFF_THRESHOLD, pos_diff);
        // return;
    }

    // if (innovation(3) > 6)
    //     innovation(3) -= 2 * PI;
    // if (innovation(3) < -6)
    //     innovation(3) += 2 * PI;
    // if (innovation(4) > 6)
    //     innovation(4) -= 2 * PI;
    // if (innovation(4) < -6)
    //     innovation(4) += 2 * PI;
    // if (innovation(5) > 6)
    //     innovation(5) -= 2 * PI;
    // if (innovation(5) < -6)
    //     innovation(5) += 2 * PI;
    INNOVATION_ = innovation_t.segment<3>(3);
    // X_state += Kt_kalmanGain * (innovation);
    X_state = boxplus(X_state, Kt_kalmanGain * (innovation));
    // if (X_state(3) > PI)
    //     X_state(3) -= 2 * PI;
    // if (X_state(3) < -PI)
    //     X_state(3) += 2 * PI;
    // if (X_state(4) > PI)
    //     X_state(4) -= 2 * PI;
    // if (X_state(4) < -PI)
    //     X_state(4) += 2 * PI;
    // if (X_state(5) > PI)
    //     X_state(5) -= 2 * PI;
    // if (X_state(5) < -PI)
    //     X_state(5) += 2 * PI;
    StateCovariance = StateCovariance - Kt_kalmanGain * Ct * StateCovariance;

    // ROS_INFO("time cost: %f\n", (clock() - t) / CLOCKS_PER_SEC);
    // cout << "z " << Z_measurement(2) << " k " << Kt_kalmanGain(2) << " inn " << innovation(2) << endl;

    test_odomtag_call = true;
    odomtag_call = true;

    if (INNOVATION_(0) > 6 || INNOVATION_(1) > 6 || INNOVATION_(2) > 6)
        cout << "\ninnovation: \n"
             << INNOVATION_ << endl;
    if (INNOVATION_(0) < -6 || INNOVATION_(1) < -6 || INNOVATION_(2) < -6)
        cout << "\ninnovation: \n"
             << INNOVATION_ << endl;
    // monitor the position changing
    if ((innovation(0) > 1.5) || (innovation(1) > 1.5) || (innovation(2) > 1.5) ||
        (innovation(0) < -1.5) || (innovation(1) < -1.5) || (innovation(2) < -1.5))
        ROS_ERROR("posintion diff too much between measurement and model prediction!!!");
    if (cnt == 10 || cnt == 50 || cnt == 90)
    {
        // cout << "Ct: \n" << Ct << "\nWt:\n" << Wt << endl;
        // cout << "Kt_kalmanGain: \n" << Kt_kalmanGain << endl;
        // cout << "\ninnovation: \n" << Kt_kalmanGain*innovation  << "\ndt:\n" << dt << endl;
        // cout << "\ninnovation: \n" << Kt_kalmanGain*innovation  << endl;
        // cout << "\ninnovation: \n" << INNOVATION_ << endl;
    }
    cnt++;
    if (cnt > 100)
        cnt = 101;
}
Vector3d rotation_2_lie_algebra(Matrix3d R)
{

    Eigen::Vector3d omega;
    double theta = std::acos((R.trace() - 1) / 2);

    if (theta < 1e-6)
    {
        omega << 0, 0, 0;
    }
    else
    {
        omega << R(2, 1) - R(1, 2),
            R(0, 2) - R(2, 0),
            R(1, 0) - R(0, 1);
        omega = omega * (theta / (2 * std::sin(theta)));
    }
    // Quaterniond q(R);
    // // axis-angle
    // Eigen::AngleAxisd angle_axis(q);
    // omega = angle_axis.angle() * angle_axis.axis();
    return omega;
}

void vioodom_callback(const nav_msgs::Odometry::ConstPtr &msg)
{ // assume that the odom_tag from camera is sychronized with the imus and without delay. !!!

    double delaytime_ms = (imu_back_time - msg->header.stamp).toSec() * 1000;
    double buffertime_ms = (imu_front_time - msg->header.stamp).toSec() * 1000;
    if (buffertime_ms > kOdomBufferToleranceMs)
    {
        ROS_ERROR_THROTTLE(1.0, "delaytime %.2f, buf_size %d", delaytime_ms, int(sys_seq.size()));
        ROS_ERROR_THROTTLE(1.0, "odom time out of buffer time range: %.2fms (> %.2fms), ekf may be wrong !!!!",
                           buffertime_ms, kOdomBufferToleranceMs);
    } 
    // else
    // {
        // ROS_INFO("delaytime %.2f, buf_size %d", delaytime_ms, int(sys_seq.size()));
        // ROS_INFO("odom time out of buffer time range: %.2fms", buffertime_ms);
    // }

    // your code for update
    static Eigen::Vector3d last_pos(0, 0, 0);
    if (first_frame_tag_odom)
    { // system begins in first odom frame
        first_frame_tag_odom = false;
        time_odom_tag_now = msg->header.stamp.toSec();

        VectorXd odom_pose = get_pose_from_VIOodom(msg);
        X_state.segment<3>(0) = odom_pose.segment<3>(0);
        // X_state.segment<3>(3) = odom_pose.segment<3>(3);
        //! changed by wz
        X_state.segment<4>(3) = odom_pose.segment<4>(3);

        world_frame_id = msg->header.frame_id;

        last_pos(0) = msg->pose.pose.position.x;
        last_pos(1) = msg->pose.pose.position.y;
        last_pos(2) = msg->pose.pose.position.z;

        first_odom_get_ = last_pos;

        // cout << "last_pos: "<<last_pos.transpose()<<endl;

        // cout << "\033[1;33m"
        //  << "odom_tag init"
        //  << "\033[0m" << endl;

        // cout << X_state.segment<3>(0).transpose()<<endl;
    }
    else
    {
        // cout << "\033[1;33m"
        //  << "odom_tag update"
        //  << "\033[0m" << endl;
        if (abs(last_pos(0) - msg->pose.pose.position.x) > 0.5 ||
            abs(last_pos(1) - msg->pose.pose.position.y) > 0.5 ||
            abs(last_pos(2) - msg->pose.pose.position.z) > 0.5)
        {
            // return;
        }

        last_pos(0) = msg->pose.pose.position.x;
        last_pos(1) = msg->pose.pose.position.y;
        last_pos(2) = msg->pose.pose.position.z;

        time_odom_tag_now = msg->header.stamp.toSec();
        //    double t = clock();

        VectorXd odom_pose = get_pose_from_VIOodom(msg);
        // Eigen::Vector3d euler_odom(odom_pose(3), odom_pose(4), odom_pose(5));
        //! changed by wz
        Eigen::Quaterniond q_odom(odom_pose(3), odom_pose(4), odom_pose(5), odom_pose(6));
        Matrix3d R_odom;
        // R_odom = euler2mat(euler_odom);
        //! changed by wz
        R_odom = q_odom.toRotationMatrix();
        // double cos_theta_theshold = cos(PI / 180 * 30);
        // double cos_theta = (R_odom * Eigen::Vector3d(0, 0, 1)).dot(Eigen::Vector3d(0, 0, -1));
        // if (cos_theta > cos_theta_theshold)
        // {
        //     ROS_ERROR("flip over!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        // }

#if TimeSync
        // call back to the proper time
        if (sys_seq.size() == 0)
        {
            // ROS_ERROR("sys_seq.size() == 0");
            update_lastest_state();
            cam_system_pub(msg->header.stamp);
            return;
        }
        // call back to the proper time
        if (sys_seq.size() == 1)
        {
            // ROS_ERROR("sys_seq.size() == 1");
            update_lastest_state();
            cam_system_pub(msg->header.stamp);
            return;
        }
        search_proper_frame(time_odom_tag_now);

#endif
        Z_measurement.segment<3>(0) = odom_pose.segment<3>(0);
        // Z_measurement.segment<3>(3) = odom_pose.segment<3>(3);
        //! changed by wz
        Z_measurement.segment<4>(3) = odom_pose.segment<4>(3);

        // cam_system_pub(msg->header.stamp);

#if !TimeSync // no aligned
        MatrixXd Ct;
        MatrixXd Wt;
        Ct = diff_g_diff_x();
        Wt = diff_g_diff_v();

        Kt_kalmanGain = StateCovariance * Ct.transpose() * (Ct * StateCovariance * Ct.transpose() + Wt * Rt * Wt.transpose()).inverse();
        VectorXd gg = g_model();
        VectorXd innovation = Z_measurement - gg;
        VectorXd innovation_t = gg;

        // Prevent innovation changing suddenly when euler from -Pi to Pi
        float pos_diff = sqrt(innovation(0) * innovation(0) + innovation(1) * innovation(1) + innovation(2) * innovation(2));
        if (pos_diff > POS_DIFF_THRESHOLD)
        {
            ROS_ERROR("posintion diff too much between measurement and model prediction!!!   pos_diff setting: %f  but the diff measured is %f ", POS_DIFF_THRESHOLD, pos_diff);
            return;
        }
        if (innovation(3) > 6)
            innovation(3) -= 2 * PI;
        if (innovation(3) < -6)
            innovation(3) += 2 * PI;
        if (innovation(4) > 6)
            innovation(4) -= 2 * PI;
        if (innovation(4) < -6)
            innovation(4) += 2 * PI;
        if (innovation(5) > 6)
            innovation(5) -= 2 * PI;
        if (innovation(5) < -6)
            innovation(5) += 2 * PI;
        INNOVATION_ = innovation_t.segment<3>(3);
        X_state += Kt_kalmanGain * (innovation);
        1.4571 X_state(3) -= 2 * PI;
        if (X_state(3) < -PI)
            X_state(3) += 2 * PI;
        if (X_state(4) > PI)
            X_state(4) -= 2 * PI;
        if (X_state(4) < -PI)
            X_state(4) += 2 * PI;
        if (X_state(5) > PI)
            X_state(5) -= 2 * PI;
        if (X_state(5) < -PI)
            X_state(5) += 2 * PI;
        StateCovariance = StateCovariance - Kt_kalmanGain * Ct * StateCovariance;

        // ROS_INFO("time cost: %f\n", (clock() - t) / CLOCKS_PER_SEC);
        // cout << "z " << Z_measurement(2) << " k " << Kt_kalmanGain(2) << " inn " << innovation(2) << endl;

        test_odomtag_call = true;
        odomtag_call = true;

        if (INNOVATION_(0) > 6 || INNOVATION_(1) > 6 || INNOVATION_(2) > 6)
            cout << "\ninnovation: \n"
                 << INNOVATION_ << endl;
        if (INNOVATION_(0) < -6 || INNOVATION_(1) < -6 || INNOVATION_(2) < -6)
            cout << "\ninnovation: \n"
                 << INNOVATION_ << endl;
        // monitor the position changing
        if ((innovation(0) > 1.5) || (innovation(1) > 1.5) || (innovation(2) > 1.5) ||
            (innovation(0) < -1.5) || (innovation(1) < -1.5) || (innovation(2) < -1.5))
            ROS_ERROR("posintion diff too much between measurement and model prediction!!!");
        if (cnt == 10 || cnt == 50 || cnt == 90)
        {
            // cout << "Ct: \n" << Ct << "\nWt:\n" << Wt << endl;
            // cout << "Kt_kalmanGain: \n" << Kt_kalmanGain << endl;
            // cout << "\ninnovation: \n" << Kt_kalmanGain*innovation  << "\ndt:\n" << dt << endl;
            // cout << "\ninnovation: \n" << Kt_kalmanGain*innovation  << endl;
            // cout << "\ninnovation: \n" << INNOVATION_ << endl;
        }
        cnt++;
        if (cnt > 100)
            cnt = 101;

#else // time sync
        MatrixXd Ct;
        MatrixXd Wt;

        // re-prediction for the rightframe
        dt = dt_0_rp;

        u_gyro(0) = sys_seq[0].second.angular_velocity.x;
        u_gyro(1) = sys_seq[0].second.angular_velocity.y;
        u_gyro(2) = sys_seq[0].second.angular_velocity.z;
        u_acc(0) = sys_seq[0].second.linear_acceleration.x;
        u_acc(1) = sys_seq[0].second.linear_acceleration.y;
        u_acc(2) = sys_seq[0].second.linear_acceleration.z;

        MatrixXd Ft;
        MatrixXd Vt;

        X_state = sys_seq[0].first;
        StateCovariance = cov_seq[0];

        // q_last = sys_seq[0].first.segment<3>(3);   // last X2
        // bg_last = sys_seq[0].first.segment<3>(9);  // last X4
        // ba_last = sys_seq[0].first.segment<3>(12); // last X5
        //! changed by wz
        // q_last = sys_seq[0].first.segment<4>(3);   // last X2
        q_last.w() = sys_seq[0].first(3);
        q_last.x() = sys_seq[0].first(4);
        q_last.y() = sys_seq[0].first(5);
        q_last.z() = sys_seq[0].first(6);

        bg_last = sys_seq[0].first.segment<3>(10); // last X4
        ba_last = sys_seq[0].first.segment<3>(13); // last X5

        // Ft = MatrixXd::Identity(stateSize, stateSize) + dt * diff_f_diff_x(q_last, u_gyro, u_acc, bg_last, ba_last);
        //! changed by wz
        Ft = MatrixXd::Identity(errorstateSize, errorstateSize) + dt * diff_f_diff_x(q_last, u_gyro, u_acc, bg_last, ba_last);

        // std::cout << "Ft" << std::endl
        //           << Ft << std::endl;

        Vt = dt * diff_f_diff_n(q_last);

        // std::cout << "Vt" << std::endl
        //   << Vt << std::endl;

        // X_state += dt * F_model(u_gyro, u_acc);
        //! changed by wz
        X_state = upate_state_Quaterniond_F_model(X_state, u_gyro, u_acc, dt);
        // std::cout << "X_state" << std::endl
        //   << X_state << std::endl;
        // if (X_state(3) > PI)  //! changed by wz
        //     X_state(3) -= 2 * PI;
        // if (X_state(3) < -PI)
        //     X_state(3) += 2 * PI;
        // if (X_state(4) > PI)
        //     X_state(4) -= 2 * PI;
        // if (X_state(4) < -PI)
        //     X_state(4) += 2 * PI;
        // if (X_state(5) > PI)
        //     X_state(5) -= 2 * PI;
        // if (X_state(5) < -PI)
        //     X_state(5) += 2 * PI;
        StateCovariance = Ft * StateCovariance * Ft.transpose() + Vt * Qt * Vt.transpose();
        // std::cout << "StateCovariance" << std::endl
        //           << StateCovariance << std::endl;

        // re-update for the rightframe
        Ct = diff_g_diff_x();
        // std::cout << "Ct" << std::endl
        //           << Ct << std::endl;
        Wt = diff_g_diff_v();

        // std::cout << "Wt" << std::endl
        //           << Wt << std::endl;

        Kt_kalmanGain = StateCovariance * Ct.transpose() * (Ct * StateCovariance * Ct.transpose() + Wt * Rt * Wt.transpose()).inverse();
        // std::cout << "Kt_kalmanGain" << std::endl
        //           << Kt_kalmanGain << std::endl;

        VectorXd gg = g_model();
        // std::cout << "gg" << std::endl
        //           << gg << std::endl;
        // std::cout << "Z_measurement" << std::endl
        //           << Z_measurement << std::endl;
        // VectorXd innovation = Z_measurement - gg;
        VectorXd innovation = VectorXd::Zero(6);
        innovation.segment<3>(0) = Z_measurement.segment<3>(0) - gg.segment<3>(0);
        // std::cout << "innovation_first" << std::endl
        //           << innovation << std::endl;
        Quaterniond q_gg(gg(3), gg(4), gg(5), gg(6));
        q_gg.normalized();
        Quaterniond q_Z_measurement(Z_measurement(3), Z_measurement(4), Z_measurement(5), Z_measurement(6));
        q_Z_measurement.normalized();
        // Quaterniond error_q = q_Z_measurement * q_gg.inverse();
        Quaterniond error_q = q_gg.inverse() * q_Z_measurement;
        error_q.normalized();
        // cout << "error_q.vec()" << endl
        //      << error_q.vec() << endl;
        Matrix3d error_R = error_q.toRotationMatrix();
        // rotation matrix Lie algebra
        innovation.segment<3>(3) = rotation_2_lie_algebra(error_R);
        // std::cout << "innovation" << std::endl
        //           << innovation << std::endl;
        // std::cout << "\033[1;32m innovation" << std::endl
        //           << innovation << "\033[0m" << std::endl;
        // std::cout << "\033[1;33m position diff: " << sqrt(innovation(0) * innovation(0) + innovation(1) * innovation(1) + innovation(2) * innovation(2)) << std::endl;
        // std::cout << "angle diff: " << sqrt(innovation(3) * innovation(3) + innovation(4) * innovation(4) + innovation(5) * innovation(5)) << "\033[0m" << std::endl;
        VectorXd innovation_t = gg;

        float pos_diff = sqrt(innovation(0) * innovation(0) + innovation(1) * innovation(1) + innovation(2) * innovation(2));
        if (pos_diff > POS_DIFF_THRESHOLD)
        {
            ROS_ERROR("posintion diff too much between measurement and model prediction!!!   pos_diff setting: %f  but the diff measured is %f ", POS_DIFF_THRESHOLD, pos_diff);
            // return;
        }

        // Prevent innovation changing suddenly when euler from -Pi to Pi
        // if (innovation(3) > 6)
        //     innovation(3) -= 2 * PI;
        // if (innovation(3) < -6)w = angle_axis.angle() * angle_axis.axis();
        //     innovation(3) += 2 * PI;
        // if (innovation(4) > 6)
        //     innovation(4) -= 2 * PI;
        // if (innovation(4) < -6)
        //     innovation(4) += 2 * PI;
        // if (innovation(5) > 6)
        //     innovation(5) -= 2 * PI;
        // if (innovation(5) < -6)
        //     innovation(5) += 2 * PI;

        // Quaterniond test_q = Quaterniond(X_state(3), X_state(4), X_state(5), X_state(6)) * error_q;
        // std::cout << "\033[1;31m test_q" << std::endl
        //           << test_q << "\033[0m" << std::endl;

        INNOVATION_ = innovation_t.segment<3>(3);

        VectorXd dx(errorstateSize);
        dx = VectorXd::Zero(errorstateSize);
        dx += Kt_kalmanGain * (innovation);
        // dx += (innovation);
        // dx.segment<6>(0) = innovation.segment<6>(0);
        // std::cout << "dx" << std::endl
        //           << dx << std::endl;

        // X_state += Kt_kalmanGain * (innovation);
        X_state = boxplus(X_state, dx);
        Vector3d X_state_p = X_state.segment<3>(0);
        VectorXd final_innovation = VectorXd::Zero(6);
        final_innovation.segment<3>(0) = Z_measurement.segment<3>(0) - X_state_p;
        // std::cout << "\033[1;32m Z_measurement.segment<3>(0)" << std::endl
        //           << Z_measurement.segment<3>(0) << "\033[0m" << std::endl;
        // std::cout << "X_state_p" << std::endl
        //           << X_state_p << std::endl;

        Quaterniond q_X_state_q(X_state(3), X_state(4), X_state(5), X_state(6));
        Quaterniond q_Z_measurement_(Z_measurement(3), Z_measurement(4), Z_measurement(5), Z_measurement(6));
        Quaterniond error_q_ = q_Z_measurement_ * q_X_state_q.inverse();
        Matrix3d error_R_ = error_q_.toRotationMatrix();
        final_innovation.segment<3>(3) = rotation_2_lie_algebra(error_R_);
        // std::cout << "\033[1;32m final_innovation" << std::endl
        //           << final_innovation << "\033[0m" << std::endl;
        // std::cout << "\033[1;33m position diff: " << sqrt(final_innovation(0) * final_innovation(0) + final_innovation(1) * final_innovation(1) + final_innovation(2) * final_innovation(2)) << std::endl;
        // std::cout << "angle diff: " << sqrt(final_innovation(3) * final_innovation(3) + final_innovation(4) * final_innovation(4) + final_innovation(5) * final_innovation(5)) << "\033[0m" << std::endl;

        // std::cout << "X_state_update" << std::endl
        //           << X_state << std::endl;
        // if (X_state(3) > PI)
        //     X_state(3) -= 2 * PI;
        // if (X_state(3) < -PI)
        //     X_state(3) += 2 * PI;
        // if (X_state(4) > PI)
        //     X_state(4) -= 2 * PI;
        // if (X_state(4) < -PI)
        //     X_state(4) += 2 * PI;
        // if (X_state(5) > PI)
        //     X_state(5) -= 2 * PI;
        // if (X_state(5) < -PI)
        //     X_state(5) += 2 * PI;
        StateCovariance = StateCovariance - Kt_kalmanGain * Ct * StateCovariance;

        // system_pub(X_state, sys_seq[0].second.header.stamp); // choose to publish the repropagation or not
        // std::cout << "re_propagate" << std::endl;

        re_propagate();

#endif
        cam_system_pub(msg->header.stamp);
    }
}
Matrix3d lie_algebra_2_rotation(Vector3d v)
{

    Eigen::Matrix3d R;
    double theta = v.norm();

    if (theta < 1e-6)
    {
        R = Eigen::Matrix3d::Identity();
    }
    else
    {
        Eigen::Matrix3d skew;
        skew << 0, -v(2), v(1),
            v(2), 0, -v(0),
            -v(1), v(0), 0;
        R = Eigen::Matrix3d::Identity() + skew / theta * std::sin(theta) + skew * skew / theta / theta * (1 - std::cos(theta));
        // R = skew.exp();
    }

    return R;
}
VectorXd boxplus(VectorXd x, VectorXd dx)
{
    VectorXd x_plus(x.rows());
    x_plus(0) = x(0) + dx(0);
    x_plus(1) = x(1) + dx(1);
    x_plus(2) = x(2) + dx(2);

    Vector3d dv(dx(3), dx(4), dx(5));
    Matrix3d dR = lie_algebra_2_rotation(dv);
    Quaterniond x_q(x(3), x(4), x(5), x(6));
    Matrix3d x_R = x_q.toRotationMatrix();
    Matrix3d x_R_plus = x_R * dR;
    Quaterniond x_q_plus(x_R_plus);
    x_q_plus.normalized();
    x_plus(3) = x_q_plus.w();
    x_plus(4) = x_q_plus.x();
    x_plus(5) = x_q_plus.y();
    x_plus(6) = x_q_plus.z();

    x_plus(7) = x(7) + dx(6);
    x_plus(8) = x(8) + dx(7);
    x_plus(9) = x(9) + dx(8);

    x_plus(10) = x(10) + dx(9);
    x_plus(11) = x(11) + dx(10);
    x_plus(12) = x(12) + dx(11);

    x_plus(13) = x(13) + dx(12);
    x_plus(14) = x(14) + dx(13);
    x_plus(15) = x(15) + dx(14);

    return x_plus;
}

Quaterniond q_gt, q_gt0;
bool first_gt = true;
void gt_callback(const nav_msgs::Odometry::ConstPtr &msg)
{
    q_gt.w() = msg->pose.pose.orientation.w;
    q_gt.x() = msg->pose.pose.orientation.x;
    q_gt.y() = msg->pose.pose.orientation.y;
    q_gt.z() = msg->pose.pose.orientation.z;

    if (first_gt && !first_frame_tag_odom)
    {
        first_gt = false;
        q_gt0 = q_gt;
        // q_gt0 = q_gt0.normalized();
    }
}

int main(int argc, char **argv)
{
  int core_id = 5;
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);
  if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) == -1) 
  {
      std::cerr << "Failed to set CPU affinity for thread: ekf "<< std::endl;
  } 
  else 
  {
      std::cout << "Successfully set CPU affinity to core " << core_id << std::endl;
  }
    ros::init(argc, argv, "ekf");
    ros::NodeHandle n("~");
    ros::Subscriber s1 = n.subscribe("imu", 1000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber s2 = n.subscribe("bodyodometry", 40, vioodom_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber s4 = n.subscribe("gt_", 40, gt_callback, ros::TransportHints().tcpNoDelay());
    odom_pub = n.advertise<nav_msgs::Odometry>("ekf_odom", 1000);             // freq = imu freq
    ahead_odom_pub = n.advertise<nav_msgs::Odometry>("ahead_ekf_odom", 1000); // freq = imu freq
    cam_odom_pub = n.advertise<nav_msgs::Odometry>("cam_ekf_odom", 1000);
    acc_filtered_pub = n.advertise<geometry_msgs::PoseStamped>("acc_filtered", 1000);

    n.getParam("gyro_cov", gyro_cov);
    n.getParam("acc_cov", acc_cov);
    n.getParam("position_cov", position_cov);
    n.getParam("q_rp_cov", q_rp_cov);
    n.getParam("q_yaw_cov", q_yaw_cov);
    n.getParam("imu_trans_x", imu_trans_x);
    n.getParam("imu_trans_y", imu_trans_y);
    n.getParam("imu_trans_z", imu_trans_z);
    n.getParam("cutoff_freq", cutoff_freq);
    n.getParam("offset_px", offset_px);
    n.getParam("offset_py", offset_py);
    n.getParam("offset_pz", offset_pz);
    

    cout << "Q:" << gyro_cov << " " << acc_cov << " R: " << position_cov << " " << q_rp_cov << " " << q_yaw_cov << endl;

    std::vector<double> Rri, tri, rotationimu;
    n.getParam("Rr_i", Rri);
    n.getParam("tr_i", tri);
    n.getParam("scale_g", scale_g);
    n.getParam("rotation_imu", rotationimu);
    Rr_i = Quaterniond(Rri.at(0), Rri.at(1), Rri.at(2), Rri.at(3)).toRotationMatrix();
    for (int i=0; i<3; i++){
        for (int j=0; j<3; j++){
            rotation_imu(i, j) = rotationimu[i*3+j];
        }
    }
    cout << "rotation_imu = " << endl 
         << rotation_imu << endl;
    cout << "scale_g = " << scale_g << endl;
    tr_i << tri.at(0), tri.at(1), tri.at(2);
    cout << "Rr_i: " << endl
         << Rr_i << endl;
    cout << "tr_i: " << endl
         << tr_i << endl;

    initsys();
    cout << "initsys" << endl;

    // cout << "======================" << endl;
    // double r = atan2(1,-100);
    // double p = asin(-0.707);
    // double y = atan2(-1, -100);
    // cout << "r: " << r << " p: " << p << " y: " << y << endl;
    // cout << "======================" << endl;

    ros::spin();
}

void acc_f_pub(Vector3d acc, ros::Time stamp)
{
    geometry_msgs::PoseStamped Accel_filtered;
    Accel_filtered.header.frame_id = "world";
    Accel_filtered.header.stamp = stamp;
    // Accel_filtered.header.stamp = ros::Time::now();
    Vector3d Acc_ = get_filtered_acc(acc);
    Accel_filtered.pose.position.x = Acc_[0];
    Accel_filtered.pose.position.y = Acc_[1];
    Accel_filtered.pose.position.z = Acc_[2];

    Quaterniond q;
    // q = euler2quaternion(X_state.segment<3>(3));
    //! changed by wz
    q = X_state.segment<4>(3);
    // q = q.normalized();
    Accel_filtered.pose.orientation.w = q.w();
    Accel_filtered.pose.orientation.x = q.x();
    Accel_filtered.pose.orientation.y = q.y();
    Accel_filtered.pose.orientation.z = q.z();

    // Accel_filtered.pose.orientation.w = q_gt.w();
    // Accel_filtered.pose.orientation.x = q_gt.x();
    // Accel_filtered.pose.orientation.y = q_gt.y();
    // Accel_filtered.pose.orientation.z = q_gt.z();

    cout << "q_gt0: " << quaternion2euler(q_gt0) << endl;
    cout << "q_gt: " << quaternion2euler(q_gt) << endl;
    cout << "q_vio: " << mat2euler(q_gt0.toRotationMatrix() * euler2quaternion(X_state.segment<3>(3)).toRotationMatrix()) << endl;
    cout << "q_gt0*vio != q_gt: " << quaternion2euler(q_gt0 * q) << endl;
    // cout << "q_gt0*vio*q_gt0^-1 = q_gt: " << quaternion2euler(q_gt0 * q * q_gt0.inverse()) << endl;   //q1*euler2quaternion(V)*q1.inverse()  = q1.toRotationMatrix() * V  TODO why?

    acc_filtered_pub.publish(Accel_filtered);
}
void ahead_system_pub(const Eigen::VectorXd &X_state_in, ros::Time stamp)
{
    nav_msgs::Odometry odom_fusion;
    odom_fusion.header.stamp = stamp;
    odom_fusion.header.frame_id = world_frame_id;
    // odom_fusion.header.frame_id = "imu";

    Quaterniond q;
    // q = euler2quaternion(X_state_in.segment<3>(3));
    //! changed by wz
    q = X_state_in.segment<4>(3);
    odom_fusion.pose.pose.orientation.w = q.w();
    odom_fusion.pose.pose.orientation.x = q.x();
    odom_fusion.pose.pose.orientation.y = q.y();
    odom_fusion.pose.pose.orientation.z = q.z();
    // odom_fusion.twist.twist.linear.x = X_state_in(6);
    // odom_fusion.twist.twist.linear.y = X_state_in(7);
    // odom_fusion.twist.twist.linear.z = X_state_in(8);
    //! changed by wz
    odom_fusion.twist.twist.linear.x = X_state_in(7);
    odom_fusion.twist.twist.linear.y = X_state_in(8);
    odom_fusion.twist.twist.linear.z = X_state_in(9);

    Vector3d pos_center(X_state_in(0), X_state_in(1), X_state_in(2)), pos_center2;
    pos_center2 = pos_center + q.toRotationMatrix() * Vector3d(imu_trans_x, imu_trans_y, imu_trans_z);
    odom_fusion.pose.pose.position.x = pos_center2(0);
    odom_fusion.pose.pose.position.y = pos_center2(1);
    odom_fusion.pose.pose.position.z = pos_center2(2);

    ahead_odom_pub.publish(odom_fusion);
}
void system_pub(const Eigen::VectorXd &X_state_in, ros::Time stamp)
{
    nav_msgs::Odometry odom_fusion;
    odom_fusion.header.stamp = stamp;
    odom_fusion.header.frame_id = "world";
    // odom_fusion.header.frame_id = world_frame_id;
    // odom_fusion.header.frame_id = "imu";

    Quaterniond q;
    // q = euler2quaternion(X_state_in.segment<3>(3));
    //! changed by wz
    q.w() = X_state_in(3);
    q.x() = X_state_in(4);
    q.y() = X_state_in(5);
    q.z() = X_state_in(6);
    odom_fusion.pose.pose.orientation.w = q.w();
    odom_fusion.pose.pose.orientation.x = q.x();
    odom_fusion.pose.pose.orientation.y = q.y();
    odom_fusion.pose.pose.orientation.z = q.z();
    // odom_fusion.twist.twist.linear.x = X_state_in(6);
    // odom_fusion.twist.twist.linear.y = X_state_in(7);
    // odom_fusion.twist.twist.linear.z = X_state_in(8);
    //! changed by wz
    odom_fusion.twist.twist.linear.x = X_state_in(7);
    odom_fusion.twist.twist.linear.y = X_state_in(8);
    odom_fusion.twist.twist.linear.z = X_state_in(9);

    Vector3d pos_center(X_state_in(0), X_state_in(1), X_state_in(2)),pos_center1, pos_center2;
    Vector3d pos_center_filter;

    // static bool first_flag = true;
    // static bool second_flag = true;
    static Vector3d last_pos_center(pos_center);
    static Vector3d last_last_pos_center(pos_center);
    static Vector3d current_pos_center(pos_center);

    double sample_freq = 200;
    double freq = sample_freq / cutoff_freq;
    double ohm = tan(PI / freq);
    double c = 1.0 + 2.0 * cos(PI / 4.0) * ohm + ohm * ohm;
    double b0 = ohm * ohm / c;
    double b1 = 2.0 * b0;
    double b2 = b0;
    double a1 = 2.0 * (ohm * ohm - 1.0) / c;
    double a2 = (1.0 - 2.0 * cos(PI / 4.0) * ohm + ohm * ohm) / c;
    // if (first_flag)
    // {
    //     last_last_pos_center = pos_center;
    //     first_flag = false;
    //     return;
    // }
    // if (!first_flag & second_flag)
    // {
    //     last_pos_center = pos_center;
    //     second_flag = false;
    //     return;
    // }
    current_pos_center = pos_center - a1 * last_pos_center - a2 * last_last_pos_center;

    pos_center_filter = b0 * current_pos_center + b1 * last_pos_center + b2 * last_last_pos_center;

    last_last_pos_center = last_pos_center;

    last_pos_center = current_pos_center;
    // cout << "pos_center!!!!!!!!!!!!!!" << endl
    //  << pos_center << endl;

    pos_center1 = pos_center + q.toRotationMatrix() * Vector3d(imu_trans_x, imu_trans_y, imu_trans_z);
    pos_center2 = pos_center_filter + q.toRotationMatrix() * Vector3d(imu_trans_x, imu_trans_y, imu_trans_z);
    // cout << "q.toRotationMatrix()" << endl
    //      << q.toRotationMatrix() << endl;
    // cout << "imu_trans:" << imu_trans_x << " " << imu_trans_y << " " << imu_trans_z << endl;
    // cout << "pos_center2???????????????????????????" << endl
    //      << pos_center2 << endl;
    if(cutoff_freq < 1.0e-5)
    {
        odom_fusion.pose.pose.position.x = pos_center1(0);
        odom_fusion.pose.pose.position.y = pos_center1(1);
        odom_fusion.pose.pose.position.z = pos_center1(2);
    } else {
        odom_fusion.pose.pose.position.x = pos_center2(0);
        odom_fusion.pose.pose.position.y = pos_center2(1);
        odom_fusion.pose.pose.position.z = pos_center2(2);
    }

    odom_fusion.pose.pose.position.x += offset_px;
    odom_fusion.pose.pose.position.y += offset_py;
    odom_fusion.pose.pose.position.z += offset_pz;

    static int cnt = 0;
    if(cnt < 1000){
        cnt++;
        return;
    } 
    
    odom_pub.publish(odom_fusion);
}
void cam_system_pub(ros::Time stamp)
{
    nav_msgs::Odometry odom_fusion;
    odom_fusion.header.stamp = stamp;
    odom_fusion.header.frame_id = world_frame_id;
    // odom_fusion.header.frame_id = "imu";
    // odom_fusion.pose.pose.position.x = Z_measurement(0);
    // odom_fusion.pose.pose.position.y = Z_measurement(1);
    // odom_fusion.pose.pose.position.z = Z_measurement(2);
    odom_fusion.pose.pose.position.x = Z_measurement(0);
    odom_fusion.pose.pose.position.y = Z_measurement(1);
    odom_fusion.pose.pose.position.z = Z_measurement(2);
    Quaterniond q;

    // q = euler2quaternion(Z_measurement.segment<3>(3));
    //! changed by wz
    q = Z_measurement.segment<4>(3);

    odom_fusion.pose.pose.orientation.w = q.w();
    odom_fusion.pose.pose.orientation.x = q.x();
    odom_fusion.pose.pose.orientation.y = q.y();
    odom_fusion.pose.pose.orientation.z = q.z();
    // odom_fusion.twist.twist.linear.x = Z_measurement(3);
    // odom_fusion.twist.twist.linear.y = Z_measurement(4);
    // odom_fusion.twist.twist.linear.z = Z_measurement(5);

    // odom_fusion.twist.twist.angular.x = INNOVATION_(0);
    // odom_fusion.twist.twist.angular.y = INNOVATION_(1);
    // odom_fusion.twist.twist.angular.z = INNOVATION_(2);
    // Vector3d pp, qq, v, bg, ba;
    // getState(pp, qq, v, bg, ba);
    // odom_fusion.twist.twist.angular.x = ba(0);
    // odom_fusion.twist.twist.angular.y = ba(1); ///??????why work??????????//////
    // odom_fusion.twist.twist.angular.z = ba(2);
    // odom_fusion.twist.twist.angular.x = diff_time;
    // odom_fusion.twist.twist.angular.y = dt;
    cam_odom_pub.publish(odom_fusion);
}

// process model
void initsys()
{
    //  camera position in the IMU frame = (0.05, 0.05, 0)
    // camera orientaion in the IMU frame = Quaternion(0, 1, 0, 0); w x y z, respectively
    //					   RotationMatrix << 1, 0, 0,
    //							             0, -1, 0,
    //                                       0, 0, -1;
    // set the cam2imu params
    Rc_i = Quaterniond(0, 1, 0, 0).toRotationMatrix();
    // cout << "R_cam" << endl << Rc_i << endl;
    tc_i << 0.05, 0.05, 0;

    //  rigid body position in the IMU frame = (0, 0, 0.04)
    // rigid body orientaion in the IMU frame = Quaternion(1, 0, 0, 0); w x y z, respectively
    //					   RotationMatrix << 1, 0, 0,
    //						 	             0, 1, 0,
    //                                       0, 0, 1;

    // states X [p q pdot bg ba]  [px,py,pz, wx,wy,wz, vx,vy,vz bgx,bgy,bgz bax,bay,baz]
    // stateSize = 15;                      // x = [p q pdot bg ba]

    stateSize = 16; //! changed by wz

    errorstateSize = 15; // ! changed by wz

    // stateSize_pqv = 9;                   // x = [p q pdot]

    stateSize_pqv = 10; // ! changed by wz

    // measurementSize = 6;                 // z = [p q]

    measurementSize = 7; //! changed by wz

    inputSize = 6;                       // u = [w a]
    X_state = VectorXd::Zero(stateSize); // x
    // velocity
    // X_state(6) = 0;
    // X_state(7) = 0;
    // X_state(8) = 0;
    X_state(3) = 1.0; //! changed by wz
    X_state(4) = 0;
    X_state(5) = 0;
    X_state(6) = 0;
    X_state(7) = 0;
    X_state(8) = 0;
    X_state(9) = 0;
    // bias
    X_state.segment<3>(10) = bg_0; //! changed by wz
    X_state.segment<3>(13) = ba_0;
    u_input = VectorXd::Zero(inputSize);
    Z_measurement = VectorXd::Zero(measurementSize); // z
    // StateCovariance = MatrixXd::Identity(stateSize, stateSize);     // sigma
    //! changed by wz
    StateCovariance = MatrixXd::Identity(errorstateSize, errorstateSize); // sigma

    Kt_kalmanGain = MatrixXd::Identity(stateSize, measurementSize); // Kt
    // Ct_stateToMeasurement = MatrixXd::Identity(stateSize, measurementSize);         // Ct
    X_state_correct = X_state;
    StateCovariance_correct = StateCovariance;

    Qt = MatrixXd::Identity(inputSize, inputSize); // 6x6 input [gyro acc]covariance
    // Rt = MatrixXd::Identity(measurementSize, measurementSize); // 6x6 measurement [p q]covariance
    //! changed by wz
    Rt = MatrixXd::Identity(measurementSize - 1, measurementSize - 1); // 6x6 measurement [p q]covariance
    // MatrixXd temp_Rt = MatrixXd::Identity(measurementSize, measurementSize);

    // You should also tune these parameters
    // Q imu covariance matrix; Rt visual odomtry covariance matrix
    // //Rt visual odomtry covariance smaller believe measurement more
    Qt.topLeftCorner(3, 3) = gyro_cov * Qt.topLeftCorner(3, 3);
    Qt.bottomRightCorner(3, 3) = acc_cov * Qt.bottomRightCorner(3, 3);
    Rt.topLeftCorner(3, 3) = position_cov * Rt.topLeftCorner(3, 3);
    Rt.bottomRightCorner(3, 3) = q_rp_cov * Rt.bottomRightCorner(3, 3);
    Rt.bottomRightCorner(1, 1) = q_yaw_cov * Rt.bottomRightCorner(1, 1);
}

// void getState(Vector3d &p, Vector3d &q, Vector3d &v, Vector3d &bg, Vector3d &ba)
// {
//     p = X_state.segment<3>(0);
//     q = X_state.segment<3>(3);
//     v = X_state.segment<3>(6);
//     bg = X_state.segment<3>(9);
//     ba = X_state.segment<3>(12);
// }
//! changed by wz
void getState(Vector3d &p, Quaterniond &q, Vector3d &v, Vector3d &bg, Vector3d &ba)
{
    p = X_state.segment<3>(0);
    // q = X_state.segment<4>(3);
    q = Quaterniond(X_state(3), X_state(4), X_state(5), X_state(6));
    v = X_state.segment<3>(7);
    bg = X_state.segment<3>(10);
    ba = X_state.segment<3>(13);
}

// VectorXd get_filtered_acc(Vector3d acc)
// {
//     Vector3d q, ba;
//     q = X_state.segment<3>(3);
//     ba = X_state.segment<3>(12);

//     // return (euler2mat(q)*(acc-ba-na));
//     // return (q_gt.toRotationMatrix()*(acc-ba-na));  //false
//     return ((acc - ba - na));
//     // return ((acc-na));
//     // return (euler2mat(q)*(acc));
// }
//! changed by wz
VectorXd get_filtered_acc(Vector3d acc)
{
    Vector3d ba;
    ba = X_state.segment<3>(13);
    return ((acc - ba - na));
}

// VectorXd F_model(Vector3d gyro, Vector3d acc)
// {
//     // IMU is in FLU frame
//     // Transform IMU frame into "world" frame whose original point is FLU's original point and the XOY plain is parallel with the ground and z axis is up
//     VectorXd f(VectorXd::Zero(stateSize));
//     Vector3d p, q, v, bg, ba;
//     getState(p, q, v, bg, ba);
//     f.segment<3>(0) = v;
//     f.segment<3>(3) = w_Body2Euler(q) * (gyro - bg - ng);
//     f.segment<3>(6) = gravity + euler2mat(q) * (acc - ba - na);
//     f.segment<3>(9) = nbg;
//     f.segment<3>(12) = nba;

//     return f;
// }

//! changed by wz
VectorXd F_model(Vector3d gyro, Vector3d acc)
{
    // IMU is in FLU frame
    // Transform IMU frame into "world" frame whose original point is FLU's original point and the XOY plain is parallel with the ground and z axis is up
    VectorXd f(VectorXd::Zero(stateSize));
    Vector3d p, v, bg, ba;
    Quaterniond q;
    getState(p, q, v, bg, ba);
    f.segment<3>(0) = v;                          // 0,1,2
    f.segment<3>(4) = q * (gyro - bg - ng) * 0.5; // 4,5,6
    f.segment<3>(7) = gravity + q * (acc - ba - na);
    f.segment<3>(10) = nbg;
    f.segment<3>(13) = nba;

    return f;
}

//? added by wz
VectorXd upate_state_Quaterniond_F_model(VectorXd X_state, Vector3d gyro, Vector3d acc, double dt)
{
    // IMU is in FLU frame
    // Transform IMU frame into "world" frame whose original point is FLU's original point and the XOY plain is parallel with the ground and z axis is up
    VectorXd f(VectorXd::Zero(stateSize));

    VectorXd upate_X_state(VectorXd::Zero(stateSize));

    Vector3d p, v, bg, ba;
    Quaterniond q;
    getState(p, q, v, bg, ba);

    f.segment<3>(0) = v;                          // 0,1,2
    f.segment<3>(4) = q * (gyro - bg - ng) * 0.5; // 4,5,6
    f.segment<3>(7) = gravity + q * (acc - ba - na);
    f.segment<3>(10) = nbg;
    f.segment<3>(13) = nba;

    upate_X_state.segment<3>(0) = X_state.segment<3>(0) + v * dt + 0.5 * (gravity + q * (acc - ba - na)) * dt * dt;
    // upate_X_state.segment<3>(0) = X_state.segment<3>(0) + v * dt;
    Quaterniond delta_q(1.0, 0.5 * (gyro(0) - bg(0) - ng(0)) * dt, 0.5 * (gyro(1) - bg(1) - ng(1)) * dt, 0.5 * (gyro(2) - bg(2) - ng(2)) * dt);
    Quaterniond upate_q = (q * delta_q).normalized();
    upate_X_state(3) = upate_q.w();
    upate_X_state(4) = upate_q.x();
    upate_X_state(5) = upate_q.y();
    upate_X_state(6) = upate_q.z();
    upate_X_state.segment<3>(7) = X_state.segment<3>(7) + (gravity + q * (acc - ba - na)) * dt;
    upate_X_state.segment<3>(10) = X_state.segment<3>(10) + nbg * dt;
    upate_X_state.segment<3>(13) = X_state.segment<3>(13) + nba * dt;
    return upate_X_state;
}

// VectorXd g_model()
// {
//     VectorXd g(VectorXd::Zero(measurementSize));

//     g.segment<6>(0) = X_state.segment<6>(0);

//     // if(g(3) > PI)  g(3) -= 2*PI;
//     // if(g(3) < -PI) g(3) += 2*PI;
//     // if(g(4) > PI)  g(4) -= 2*PI;
//     // if(g(4) < -PI) g(4) += 2*PI;
//     // if(g(5) > PI)  g(5) -= 2*PI;
//     // if(g(5) < -PI) g(5) += 2*PI;

//     return g;
// }
//! changed by wz
VectorXd g_model()
{
    VectorXd g(VectorXd::Zero(measurementSize));

    g.segment<7>(0) = X_state.segment<7>(0);

    // if(g(3) > PI)  g(3) -= 2*PI;
    // if(g(3) < -PI) g(3) += 2*PI;
    // if(g(4) > PI)  g(4) -= 2*PI;
    // if(g(4) < -PI) g(4) += 2*PI;
    // if(g(5) > PI)  g(5) -= 2*PI;
    // if(g(5) < -PI) g(5) += 2*PI;

    return g;
}

// F_model G_model Jocobian
// diff_f()/diff_x (x_t-1  ut  noise=0)   At     Ft = I+dt*At
// MatrixXd diff_f_diff_x(Vector3d q_last, Vector3d gyro, Vector3d acc, Vector3d bg_last, Vector3d ba_last)
// {
//     double cr = cos(q_last(0));
//     double sr = sin(q_last(0));
//     double cp = cos(q_last(1));
//     double sp = sin(q_last(1));
//     double cy = cos(q_last(2));
//     double sy = sin(q_last(2));

//     // ng na = 0 nbg nba = 0
//     double Ax = acc(0) - ba_last(0);
//     double Ay = acc(1) - ba_last(1);
//     double Az = acc(2) - ba_last(2);
//     // double Wx = gyro(0) - bg_last(0);
//     double Wy = gyro(1) - bg_last(1);
//     double Wz = gyro(2) - bg_last(2);

//     MatrixXd diff_f_diff_x_jacobian(MatrixXd::Zero(stateSize, stateSize));
//     MatrixXd diff_f_diff_x_jacobian_pqv(MatrixXd::Zero(stateSize_pqv, stateSize_pqv));

//     diff_f_diff_x_jacobian_pqv << 0, 0, 0, 0, 0, 0, 1, 0, 0,
//         0, 0, 0, 0, 0, 0, 0, 1, 0,
//         0, 0, 0, 0, 0, 0, 0, 0, 1,
//         0, 0, 0, (sp * (Wy * cr - Wz * sr)) / cp, (Wz * cr + Wy * sr) / (cp * cp), 0, 0, 0, 0,
//         0, 0, 0, (-Wz * cr - Wy * sr), 0, 0, 0, 0, 0,
//         0, 0, 0, (Wy * cr - Wz * sr) / cp, (sp * (Wz * cr + Wy * sr)) / (cp * cp), 0, 0, 0, 0,
//         0, 0, 0, (Ay * (sr * sy + cr * cy * sp) + Az * (cr * sy - cy * sp * sr)), (Az * cp * cr * cy - Ax * cy * sp + Ay * cp * cy * sr), (Az * (cy * sr - cr * sp * sy) - Ay * (cr * cy + sp * sr * sy) - Ax * cp * sy), 0, 0, 0,
//         0, 0, 0, (-Ay * (cy * sr - cr * sp * sy) - Az * (cr * cy + sp * sr * sy)), (Az * cp * cr * sy - Ax * sp * sy + Ay * cp * sr * sy), (Az * (sr * sy + cr * cy * sp) - Ay * (cr * sy - cy * sp * sr) + Ax * cp * cy), 0, 0, 0,
//         0, 0, 0, (Ay * cp * cr - Az * cp * sr), (-Ax * cp - Az * cr * sp - Ay * sp * sr), 0, 0, 0, 0;

//     diff_f_diff_x_jacobian.block<9, 9>(0, 0) = diff_f_diff_x_jacobian_pqv;
//     diff_f_diff_x_jacobian.block<3, 3>(3, 9) = -w_Body2Euler(q_last);
//     diff_f_diff_x_jacobian.block<3, 3>(6, 12) = -euler2mat(q_last);

//     return diff_f_diff_x_jacobian;

//     // cp != 0 pitch != 90° !!!!!!!!!
// }
//? added by wz
Matrix3d hat(Vector3d v)
{
    Matrix3d v_hat;
    v_hat << 0, -v(2), v(1),
        v(2), 0, -v(0),
        -v(1), v(0), 0;
    return v_hat;
}

//! changed by wz
MatrixXd diff_f_diff_x(Quaterniond q_last, Vector3d gyro, Vector3d acc, Vector3d bg_last, Vector3d ba_last)
{

    // MatrixXd diff_f_diff_x_jacobian(MatrixXd::Zero(stateSize, stateSize));
    // MatrixXd diff_f_diff_x_jacobian_pqv(MatrixXd::Zero(stateSize_pqv, stateSize_pqv));
    MatrixXd diff_f_diff_x_jacobian(MatrixXd::Zero(errorstateSize, errorstateSize));
    diff_f_diff_x_jacobian.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity(); // dp/dv
    diff_f_diff_x_jacobian.block<3, 3>(3, 9) = -Eigen::Matrix3d::Identity();
    diff_f_diff_x_jacobian.block<3, 3>(6, 3) = -q_last.toRotationMatrix() * hat(acc - ba_last); //!!!!
    diff_f_diff_x_jacobian.block<3, 3>(6, 12) = -q_last.toRotationMatrix();
    return diff_f_diff_x_jacobian;
}

// diff_f()/diff_n (x_t-1  ut  noise=0)  Ut    Vt = dt*Ut
// MatrixXd diff_f_diff_n(Vector3d q_last)
// {
//     MatrixXd diff_f_diff_n_jacobian(MatrixXd::Zero(stateSize, inputSize));
//     diff_f_diff_n_jacobian.block<3, 3>(3, 0) = -w_Body2Euler(q_last);
//     diff_f_diff_n_jacobian.block<3, 3>(6, 3) = -euler2mat(q_last);

//     return diff_f_diff_n_jacobian;
// }

//! changed by wz
MatrixXd diff_f_diff_n(Quaterniond q_last)
{
    MatrixXd diff_f_diff_n_jacobian(MatrixXd::Zero(errorstateSize, inputSize));
    diff_f_diff_n_jacobian.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
    diff_f_diff_n_jacobian.block<3, 3>(6, 3) = -q_last.toRotationMatrix();

    return diff_f_diff_n_jacobian;
}
// // diff_g()/diff_x  (xt~ noise=0)  Ct
// MatrixXd diff_g_diff_x()
// {
//     MatrixXd diff_g_diff_x_jacobian(MatrixXd::Zero(measurementSize, stateSize));
//     diff_g_diff_x_jacobian.block<3, 3>(0, 0) = MatrixXd::Identity(3, 3);
//     diff_g_diff_x_jacobian.block<3, 3>(3, 3) = MatrixXd::Identity(3, 3);

//     return diff_g_diff_x_jacobian;
// }

//! changed by wz
MatrixXd diff_g_diff_x()
{
    // MatrixXd diff_g_diff_x_jacobian(MatrixXd::Zero(measurementSize, stateSize));
    // diff_g_diff_x_jacobian.block<3, 3>(0, 0) = MatrixXd::Identity(3, 3);
    // diff_g_diff_x_jacobian.block<3, 3>(3, 3) = MatrixXd::Identity(3, 3);

    MatrixXd diff_g_diff_x_jacobian(MatrixXd::Zero(measurementSize - 1, errorstateSize));
    diff_g_diff_x_jacobian.block<3, 3>(0, 0) = MatrixXd::Identity(3, 3);
    diff_g_diff_x_jacobian.block<3, 3>(3, 3) = MatrixXd::Identity(3, 3);

    return diff_g_diff_x_jacobian;
}
// // diff_g()/diff_v  (xt~ noise=0) Wt
// MatrixXd diff_g_diff_v()
// {
//     MatrixXd diff_g_diff_v_jacobian(MatrixXd::Identity(measurementSize, measurementSize));

//     return diff_g_diff_v_jacobian;
// }
//! changed by wz
MatrixXd diff_g_diff_v()
{
    MatrixXd diff_g_diff_v_jacobian(MatrixXd::Identity(measurementSize - 1, measurementSize - 1));

    return diff_g_diff_v_jacobian;
}
