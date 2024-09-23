#include <memory>
#include <fstream>
#include <mutex>
#include <unistd.h>
#include <limits.h>
#include <string>
#include <regex>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/TransformStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>

#include "json.hpp"
#include "timer.hpp"

// typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, 
//                                                         sensor_msgs::Image,
//                                                         sensor_msgs::Image,
//                                                         sensor_msgs::Image,
//                                                         geometry_msgs::TransformStamped> SyncPolicy;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, 
                                                        sensor_msgs::Image,
                                                        sensor_msgs::Image,
                                                        sensor_msgs::Image> SyncPolicy;
class ImageSynchronizer {
public:
    ImageSynchronizer(const ros::NodeHandle& nh,const int host_id):nh_(nh),host_id_(host_id) {
        image0_sub_.subscribe(nh_, "/cam0/image_raw", 1);
        image1_sub_.subscribe(nh_, "/cam1/image_raw", 1);
        image2_sub_.subscribe(nh_, "/cam2/image_raw", 1);
        image3_sub_.subscribe(nh_, "/cam3/image_raw", 1);
        pose_sub_.subscribe(nh_, "/vicon/VSWARM" + std::to_string(host_id) + "/VSWARM" + std::to_string(host_id), 1);

        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>   (SyncPolicy(10), 
                                                                                image0_sub_, 
                                                                                image1_sub_,
                                                                                image2_sub_,
                                                                                image3_sub_
                                                                                );

        sync_->registerCallback(boost::bind(&ImageSynchronizer::imageCallback,this, _1, _2,_3,_4));
        it_ = std::make_shared<image_transport::ImageTransport>(nh);
        img_pub_ = it_->advertise("/robot/omni_img", 1);
        load_cam_parameters(host_id_);

        cv::Mat new_camera_mat = cv::Mat::eye(3,3,CV_32FC1);
        new_camera_mat.at<float>(0,0) = 205;
        new_camera_mat.at<float>(1,1) = 205;
        if (host_id_ == 999) {
            new_camera_mat.at<float>(0,2) = 320;
            sz_ = cv::Size(640,360);
            cut_ = cv::Rect(115, 90, 410, 180);    //cut image without remap
        } else {
            new_camera_mat.at<float>(0,2) = 205;
            sz_ = cv::Size(410,360);
            cut_ = cv::Rect(0, 90, 410, 180);  //cut image for remap

        }
        new_camera_mat.at<float>(1,2) = 180;

        images_ = std::vector<cv::Mat>(4,cv::Mat());

        rate_ = std::make_shared<ros::Rate>(20);
        map1_vec_ = std::vector<cv::Mat>(4,cv::Mat());
        map2_vec_ = std::vector<cv::Mat>(4,cv::Mat());

        for (int i = 0; i < 4; i++) {
            cv::Mat map1,map2;
            cv::initUndistortRectifyMap(cameraMatrix_vec_[i], distCoeffs_vec_[i], cv::Mat(), 
            new_camera_mat, sz_, CV_32FC1, map1, map2);
            map1_vec_[i] = map1.clone();
            map2_vec_[i] = map2.clone();
        }

    }

    void load_cam_parameters(const int& host_id) {
        std::string path = "../cali_file/";
        std::stringstream oss;
        oss << std::setw(3) << std::setfill('0') << host_id;
        std::string cali_f = path + oss.str() + ".json";
        std::cout << cali_f << std::endl;
        std::ifstream file(cali_f);
        nlohmann::json json_f;
        file >> json_f;
        for (int cam_id = 0; cam_id < 4; cam_id++) {
            cv::Mat cam_mat = cv::Mat::eye(3,3,CV_32FC1);
            cv::Mat dist_coff = cv::Mat(5,1,CV_32FC1);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    cam_mat.at<float>(i, j) = json_f["cam" + std::to_string(cam_id) + "_K"][i][j];
                }
            }
            for (int i = 0; i < 5; ++i) {
                dist_coff.at<float>(0, i) = json_f["cam" + std::to_string(cam_id) + "_d"][0][i];
            }
            cameraMatrix_vec_.push_back(cam_mat);
            distCoeffs_vec_.push_back(dist_coff);

            std::cout << cam_mat << std::endl;
            std::cout << dist_coff << std::endl;

        }

    }

    void imageConcatenate() {
        while(ros::ok()) {
            cv::Mat dst0,dst1,dst2,dst3;
            if (!img0_.empty() && !img1_.empty() && !img2_.empty() && !img3_.empty()) {
                mutex_.lock();

                if (host_id_ != 999) {
                    cv::remap(img0_, dst0, map1_vec_[0], map2_vec_[0], cv::INTER_NEAREST);
                    cv::remap(img1_, dst1, map1_vec_[1], map2_vec_[1], cv::INTER_NEAREST);
                    cv::remap(img2_, dst2, map1_vec_[2], map2_vec_[2], cv::INTER_NEAREST);
                    cv::remap(img3_, dst3, map1_vec_[3], map2_vec_[3], cv::INTER_NEAREST);
                } else {
                    dst0 = img0_;
                    dst1 = img1_;
                    dst2 = img2_;
                    dst3 = img3_;
                }

                std::cout << dst0.rows << " " << dst0.cols << " " << host_id_ << std::endl;
                images_[0] = dst0(cut_);
                images_[1] = dst1(cut_);
                images_[2] = dst2(cut_);
                images_[3] = dst3(cut_);
                cv::hconcat(images_, output_);
                std_msgs::Header header;
                header.stamp = current_time_;
              //  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "mono8", output_).toImageMsg();
                sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "rgb8", output_).toImageMsg();
                img_pub_.publish(msg);
                mutex_.unlock();
            }
            ros::spinOnce();
            rate_->sleep();
        }

       // cv::imshow("Dense Optical Flow", dense_flow_img);
        // std::vector<cv::Mat> flow_vec(4,cv::Mat());
        // flow_vec[0] = opticalFlow(img0_pre_, images_[0]);
        // flow_vec[1] = opticalFlow(img1_pre_, images_[1]);
        // flow_vec[2] = opticalFlow(img2_pre_, images_[2]);
        // flow_vec[3] = opticalFlow(img3_pre_, images_[3]);

        //cv::Mat flow_cat;
       // cv::hconcat(flow_vec, flow_cat);
        // if (!output_.empty()) {
        //     cv::imshow("output",output_);
        // }
        // // if (!flow_cat.empty()) {
        // //     cv::imshow("flow_cat",flow_cat);
        // // }
        // cv::waitKey(1);
    }


    void imageConcatenateCB() {
        cv::Mat dst0,dst1,dst2,dst3;
        cv::remap(img0_, dst0, map1_vec_[0], map2_vec_[0], cv::INTER_LINEAR);
        cv::remap(img1_, dst1, map1_vec_[1], map2_vec_[1], cv::INTER_LINEAR);
        cv::remap(img2_, dst2, map1_vec_[2], map2_vec_[2], cv::INTER_LINEAR);
        cv::remap(img3_, dst3, map1_vec_[3], map2_vec_[3], cv::INTER_LINEAR);
        images_[0] = dst0(cut_);
        images_[1] = dst1(cut_);
        images_[2] = dst2(cut_);
        images_[3] = dst3(cut_);
        cv::hconcat(images_, output_);
        std_msgs::Header header;
        header.stamp = ros::Time::now();
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "mono8", output_).toImageMsg();
        img_pub_.publish(msg);
    }


    Eigen::Affine3d convertPoseMsg(const geometry_msgs::TransformStampedConstPtr& transformStamped) {
        Eigen::Translation3d translation(transformStamped->transform.translation.x,
                                        transformStamped->transform.translation.y,
                                        transformStamped->transform.translation.z);
        Eigen::Quaterniond rotation(transformStamped->transform.rotation.w,
                                    transformStamped->transform.rotation.x,
                                    transformStamped->transform.rotation.y,
                                    transformStamped->transform.rotation.z);
        
        Eigen::Affine3d transform = translation * rotation;
        return transform;
    }   

    void imageCallback( const sensor_msgs::ImageConstPtr& image0,
                        const sensor_msgs::ImageConstPtr& image1,
                        const sensor_msgs::ImageConstPtr& image2,
                        const sensor_msgs::ImageConstPtr& image3) {
        // Process synchronized images
        //printf("%d\n",image0->header.stamp.sec);
        cv_bridge::CvImagePtr cv_ptr0 = cv_bridge::toCvCopy(image0, sensor_msgs::image_encodings::TYPE_8UC3);
        cv_bridge::CvImagePtr cv_ptr1 = cv_bridge::toCvCopy(image1, sensor_msgs::image_encodings::TYPE_8UC3);
        cv_bridge::CvImagePtr cv_ptr2 = cv_bridge::toCvCopy(image2, sensor_msgs::image_encodings::TYPE_8UC3);
        cv_bridge::CvImagePtr cv_ptr3 = cv_bridge::toCvCopy(image3, sensor_msgs::image_encodings::TYPE_8UC3);

        mutex_.lock();
        img0_ = cv_ptr0->image.clone();
        img1_ = cv_ptr1->image.clone();
        img2_ = cv_ptr2->image.clone();
        img3_ = cv_ptr3->image.clone();
        current_time_ = image0->header.stamp;
        mutex_.unlock();
    }

    cv::Mat opticalFlow(const cv::Mat& frame1,const cv::Mat& frame2) {
        if (frame1.empty() || frame2.empty()) {
            return cv::Mat();
        }
        double pyr_scale = 0.5;
        int levels = 3;
        int winsize = 15;
        int iterations = 3;
        int poly_n = 5;
        double poly_sigma = 1.1;
        int flags = 0;

        cv::Mat flow;
        cv::calcOpticalFlowFarneback(frame1, frame2, flow, \
                pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);

        cv::Mat flow_parts[2];
        split(flow, flow_parts);

        cv::Mat magnitude, angle;
        cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);

        cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX);
        magnitude.convertTo(magnitude, CV_8UC1);

        cv::Mat dense_flow_img = magnitude;
        return dense_flow_img;
    }

private:
    ros::NodeHandle nh_;
    const int host_id_;
    message_filters::Subscriber<sensor_msgs::Image> image0_sub_;
    message_filters::Subscriber<sensor_msgs::Image> image1_sub_;
    message_filters::Subscriber<sensor_msgs::Image> image2_sub_;
    message_filters::Subscriber<sensor_msgs::Image> image3_sub_;
    message_filters::Subscriber<geometry_msgs::TransformStamped> pose_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    std::shared_ptr<image_transport::ImageTransport> it_;
    image_transport::Publisher img_pub_;
    ros::Time current_time_;
    std::vector<cv::Mat> cameraMatrix_vec_;
    std::vector<cv::Mat> distCoeffs_vec_;
    std::vector<cv::Mat> map1_vec_;
    std::vector<cv::Mat> map2_vec_;

    cv::Size sz_;
    cv::Mat output_;
    cv::Rect cut_;
    std::mutex mutex_;
    cv::Mat img0_,img1_,img2_,img3_;
    cv::Mat img0_pre_,img1_pre_,img2_pre_,img3_pre_;
    std::shared_ptr<ros::Rate> rate_;

    std::vector<cv::Mat> images_;

    Eigen::Affine3d pose_last_;
    Eigen::Affine3d pose_current_;
};


int get_host_id() {
    char hostname[HOST_NAME_MAX];
    // get host name
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        std::cout << "Hostname: " << hostname << std::endl;
        
        std::string hostname_str(hostname);
        std::regex re("\\d+");
        std::smatch match;
        
        if (std::regex_search(hostname_str, match, re)) {
            // get first number
            int number = std::stoi(match.str(0));
            std::cout << "Extracted number: " << number << std::endl;
            return number;
        } else {
            std::cerr << "No number found in hostname" << std::endl;
            return -1;
        }
    } else {
        std::cerr << "Error getting hostname" << std::endl;
        return -1;
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "image_synchronizer");
    ros::NodeHandle nh("~"); // Private NodeHandle to access private parameters
    int host_id = -1;
    if (nh.getParam("host_id", host_id)) {
        ROS_INFO("host_id: %d", host_id);
    } else {
        host_id = get_host_id();
    }

     if (host_id == -1) {
        ROS_WARN("Parameter 'host_id' not set");
        return -1;
     }

    ImageSynchronizer synchronizer(nh,host_id);
    synchronizer.imageConcatenate();
    return 0;
}
