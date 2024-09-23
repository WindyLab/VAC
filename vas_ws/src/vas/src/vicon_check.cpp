#include <iostream>
#include <fstream>
#include <set>
#include <string>
#include <mutex>
#include <ros/ros.h>
#include <Eigen/Dense>
#include <geometry_msgs/TransformStamped.h>
#include <vas/TransformStampedList.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>


typedef message_filters::sync_policies::ApproximateTime<geometry_msgs::TransformStamped, 
                                                        geometry_msgs::TransformStamped,
                                                        geometry_msgs::TransformStamped> SyncPolicyVicon;

typedef message_filters::sync_policies::ApproximateTime<vas::TransformStampedList, 
                                                        vas::TransformStampedList> SyncPolicyEstimation;

// Function to convert geometry_msgs::Transform to Eigen::Matrix4d
Eigen::Matrix4d convertTransformToMat(const geometry_msgs::Transform& transform) {
    Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();

    Eigen::Quaterniond q(transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z);
    Eigen::Matrix3d rot = q.toRotationMatrix();

    mat.block<3, 3>(0, 0) = rot;
    mat(0, 3) = transform.translation.x;
    mat(1, 3) = transform.translation.y;
    mat(2, 3) = transform.translation.z;
    return mat;
}

class ViconSub {
public:
    ViconSub() : nh_("~"), loop_rate_(1000), current_id_(0) {
        host_name_set_ = {"VSWARM2", "VSWARM3","VSWARM4"};
        
        for (const auto& tar : host_name_set_) {
            std::string topic = "/vicon/" + tar + "/" + tar;
            std::cout << topic << std::endl;
            subs_.push_back(std::make_shared<message_filters::Subscriber<geometry_msgs::TransformStamped>>(nh_, topic, 1));
        }
        std::cout << subs_.size() << std::endl;

        sync_.reset(new message_filters::Synchronizer<SyncPolicyVicon>(SyncPolicyVicon(10), *subs_[0], *subs_[1],*subs_[2]));
        sync_->registerCallback(boost::bind(&ViconSub::callback, this, _1, _2,_3));

        sub_plkf_.push_back(std::make_shared<message_filters::Subscriber<vas::TransformStampedList>>(nh_, "/estimation", 1));
        sub_plkf_.push_back(std::make_shared<message_filters::Subscriber<vas::TransformStampedList>>(nh_, "/observation", 1));
        sync_plkf_.reset(new message_filters::Synchronizer<SyncPolicyEstimation>(SyncPolicyEstimation(10), *sub_plkf_[0], *sub_plkf_[1]));
        sync_plkf_->registerCallback(boost::bind(&ViconSub::callback_plkf, this, _1, _2));

        est_err_pub_ = nh_.advertise<geometry_msgs::TransformStamped>("/est_err", 10);
        obs_err_pub_ = nh_.advertise<geometry_msgs::TransformStamped>("/obs_err", 10);

        relative_xyz_0_ = new double[3]{0,0,0};
        relative_xyz_1_ = new double[3]{0,0,0};

        out_file_.open("res.txt", std::ios::out | std::ios::app);
        if (!out_file_) {
            ROS_ERROR("Failed to open file for writing");
        }
    }

    void process() {
        while (ros::ok()) {
            loop_rate_.sleep();
            ros::spinOnce();
        }
    }
    ~ViconSub() {
        delete[] relative_xyz_0_;
        delete[] relative_xyz_1_;
        if (out_file_.is_open()) {
            out_file_.close();
        }
    }

private:
    void compute_vicon_gt(const geometry_msgs::TransformStamped::ConstPtr& cur,
                          const geometry_msgs::TransformStamped::ConstPtr& tar,
                          double* relative_xyz) {
        geometry_msgs::Transform pose_current = cur->transform;
        geometry_msgs::Transform pose_target = tar->transform;
        Eigen::Matrix4d pose_cu_tf = convertTransformToMat(pose_current);
        Eigen::Matrix4d pose_tar_tf = convertTransformToMat(pose_target);
        Eigen::Matrix4d delta = pose_cu_tf.inverse() * pose_tar_tf;
        relative_xyz[0] = delta(0,3);
        relative_xyz[1] = delta(1,3);
        relative_xyz[2] = delta(2,3);
    }


    void callback( const geometry_msgs::TransformStamped::ConstPtr& tar0,
                   const geometry_msgs::TransformStamped::ConstPtr& tar2,
                   const geometry_msgs::TransformStamped::ConstPtr& tar4) {
       // std::lock_guard<std::mutex> lock(mutex_);
        std::vector<geometry_msgs::TransformStamped::ConstPtr> msg_list = {tar0, tar2,tar4};
        double rela_02[3] = {0,0,0};
        double rela_04[3] = {0,0,0};
        double rela_24[3] = {0,0,0};
        

        compute_vicon_gt(msg_list[0],msg_list[1],rela_02);
        compute_vicon_gt(msg_list[0],msg_list[2],rela_04);
        compute_vicon_gt(msg_list[1],msg_list[2],rela_24);

        Eigen::Vector2f vicon0(rela_02[0],rela_02[1]);
        Eigen::Vector2f vicon1(rela_04[0],rela_04[1]);
        Eigen::Vector2f vicon2(rela_24[0],rela_24[1]);

        ROS_INFO("relative distance %f %f %f",vicon0.norm(),vicon1.norm(),vicon2.norm());

        if (out_file_.is_open()) {
            out_file_ << vicon0.norm() << " " << vicon1.norm() << " " << vicon2.norm() << std::endl;
        } else {
            ROS_ERROR("File is not open for writing");
        }
    }

    void callback_plkf( const vas::TransformStampedList::ConstPtr& est,
                        const vas::TransformStampedList::ConstPtr& obs) {
        Eigen::Vector2f ob(obs->transforms[0].transform.translation.x,obs->transforms[0].transform.translation.y);
        Eigen::Vector2f es(est->transforms[0].transform.translation.x,est->transforms[0].transform.translation.y);
        Eigen::Vector2f vicon(relative_xyz_0_[0],relative_xyz_0_[1]);
        Eigen::Vector2f es_err = es - vicon;
        Eigen::Vector2f ob_err = ob - vicon;

       // std::cout << ob.normalized() << std::endl;
        std::cout << vicon.norm() << std::endl;
        //ROS_INFO("est x %f y %f", es_err(0),es_err(1));
       //  ROS_INFO("ob x %f y %f", ob_err(0),ob_err(1));
        geometry_msgs::TransformStamped estimation_err;
        geometry_msgs::TransformStamped observation_err;
        std_msgs::Header header;
        header.stamp = ros::Time::now();
        estimation_err.header = header;
        observation_err.header = header;
        estimation_err.transform.translation.x = abs(es_err(0));
        estimation_err.transform.translation.y = abs(es_err(1));
        estimation_err.transform.translation.z = es_err.norm();
        observation_err.transform.translation.x = abs(ob_err(0));
        observation_err.transform.translation.y = abs(ob_err(1));
        observation_err.transform.translation.z = ob_err.norm();

        est_err_pub_.publish(estimation_err);
        obs_err_pub_.publish(observation_err);

       // std::cout << std::endl;
    }

    ros::NodeHandle nh_;
    ros::Rate loop_rate_;
    ros::Publisher est_err_pub_;
    ros::Publisher obs_err_pub_;
    int current_id_;
    std::ofstream out_file_;
    std::mutex mutex_;
    double* relative_xyz_0_;
    double* relative_xyz_1_;
    std::set<std::string> host_name_set_;
    std::vector<std::shared_ptr<message_filters::Subscriber<geometry_msgs::TransformStamped>>> subs_;
    std::vector<std::shared_ptr<message_filters::Subscriber<vas::TransformStampedList>>> sub_plkf_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicyVicon>> sync_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicyEstimation>> sync_plkf_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "vicon_check_node");
    ViconSub vs;
    vs.process();
    return 0;
}