#include <thread>
#include <mutex>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <sensor_msgs/Image.h>
#include <vas/bbx.h>
#include <vas/TransformStampedList.h>
#include <std_msgs/Header.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TransformStamped.h>
#include <opencv2/opencv.hpp>
#include "visual_target.h"
#include "get_param.h"
#include "timer.hpp"

const int MAX_N = 4;
const int MAX_OB = 4;

const Eigen::MatrixXf Q = (Eigen::MatrixXf(6, 6) << 
    0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.1).finished().array().square().matrix();

void computeDistanceMatrix(const Eigen::Matrix<float,MAX_N,2>& points,
                                 Eigen::Matrix<float,MAX_N,MAX_N>& distance_mat) {
    for (int i = 0; i < MAX_N; ++i) {
        for (int j = 0; j < MAX_N; ++j) {
            if (i == j) {
                distance_mat(i, j) = 0; // Distance to itself is 0
            } else {
                distance_mat(i, j) = std::sqrt(  (points(i, 0) - points(j, 0)) * (points(i, 0) - points(j, 0)) +
                                                 (points(i, 1) - points(j, 1)) * (points(i, 1) - points(j, 1)));
            }
        }
    }
}

class PLKF {
public:
    PLKF( const float& k,
          const float& A,
          const float& v_max,
          const int& mode,
          const float& dist_metric,
          const int& topo_num,
          const int& check_att_value,
          const float& att_thres) : 
                        nh_("~"), 
                        loop_rate_(30),
                        k_(k),
                        A_(A),
                        v_max_(v_max),
                        mode_(mode),
                        dist_metric_(dist_metric),
                        topo_num_(topo_num),
                        check_att_value_(check_att_value),
                        att_thres_(att_thres) {
        est_pub_ = nh_.advertise<vas::TransformStampedList>("/estimation", 10);
        obs_pub_ = nh_.advertise<vas::TransformStampedList>("/observation", 10);
        att_debug_pub_ = nh_.advertise<sensor_msgs::Image>("/att_result", 10);

        omni_img_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::Image>>(nh_, "/robot/v_att", 1);
       // original_img_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::Image>>(nh_, "/robot/omni_img", 1);

        bbx_sub_ = std::make_shared<message_filters::Subscriber<vas::bbx>>(nh_, "/bbx", 1);
        vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/robot/velcmd", 10);

        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), *omni_img_sub_, *bbx_sub_);
        sync_->registerCallback(boost::bind(&PLKF::callback, this, _1, _2));

        // variables for estimation
        first_ob_ = true;
        target_states_ = std::vector<float>(6*MAX_N,0);
        PEst_ = std::vector<float>(6*6*MAX_N,0);
        for(int k = 0; k < MAX_N; k++) {
            all_tar_id_.insert(k);
        }
        yaw_last_ = 0;
    }

    Eigen::VectorXf motion_model(const Eigen::MatrixXf& F, const Eigen::VectorXf &x) {
        return F * x;
    }

    Eigen::VectorXf observation_model(const Eigen::MatrixXf& C,const Eigen::VectorXf &sPre) {
        return C * sPre;
    }

    std::pair<int, float> bearing_observation_select(const Eigen::MatrixXf &p_est, 
    const Eigen::MatrixXf &p_new_obs,const std::set<int>& used_ob_id) {
        float delta_min = std::numeric_limits<float>::infinity();
        int target_id = -1;
        //std::cout << "p_new_obs:" << p_new_obs << std::endl;
        //std::cout << "p_est:" << p_est << std::endl;

        for (int i = 0; i < p_new_obs.rows(); ++i) {
            if (used_ob_id.count(i) > 0) {
                continue;
            } 
            Eigen::Vector3f p_new_ob_vec(p_new_obs.row(i)(0),p_new_obs.row(i)(1),p_new_obs.row(i)(2));
            if (p_new_ob_vec.isZero()) {
                continue;
            }
            Eigen::Vector3f pos_est(p_est(0),p_est(1),p_est(2));
            float delta = (pos_est - p_new_ob_vec).norm();
            if (delta < delta_min) {
                delta_min = delta;
                target_id = i;
            }
        }
        return std::make_pair(target_id, delta_min);
    }

    void plkf_estimation(const int tar_id,const Eigen::VectorXf &y, const Eigen::MatrixXf &R_plkf) {
        Eigen::Map<Eigen::Matrix<float, MAX_N, 6, Eigen::RowMajor>> target_states_mat(target_states_.data());
        Eigen::Map<Eigen::Matrix<float, 6*MAX_N, 6, Eigen::RowMajor>> target_P_mat(PEst_.data());
        auto sEst = target_states_mat.row(tar_id);
        auto PEst = target_P_mat.block<6, 6>(tar_id * 6, 0);

        float dt = 0.05;
        Eigen::MatrixXf F = Eigen::MatrixXf::Identity(6, 6);
        F(0, 3) = dt;
        F(1, 4) = dt;
        F(2, 5) = dt;
        Eigen::MatrixXf C(3, 6);
        C << Eigen::MatrixXf::Identity(3, 3), Eigen::MatrixXf::Zero(3, 3);

        // Predict
        Eigen::VectorXf sPred = motion_model(F,sEst.transpose());
        //std::cout << "sEst: " << sEst << std::endl;
        //std::cout << "F: " << F << std::endl;

        //std::cout << "sPred: " << sPred << std::endl;
        //std::cout << "y: " << y << std::endl;

        Eigen::MatrixXf PPred = F * PEst * F.transpose() + Q;
        Eigen::MatrixXf fPred = observation_model(C,sPred);
        Eigen::VectorXf innov = y - fPred;
        Eigen::MatrixXf S = C * PPred * C.transpose() + R_plkf;
        Eigen::MatrixXf K = PPred * C.transpose() * S.inverse();
        sEst = sPred + K * innov;
        PEst = (Eigen::MatrixXf::Identity(6, 6) - K * C) * PPred;
        target_states_mat.row(tar_id) = sEst;
        target_P_mat.block<6, 6>(tar_id * 6, 0) = PEst;
    }

    void target_ass( const Eigen::MatrixXf& target_states_mat, 
                     const Eigen::MatrixXf& target_pos_observation,
                     std::vector<int>& ass,
                     std::set<int>& valid_tar_id ) {
            
            std::set<int> used_ob_id;
            for(int tar_id = 0; tar_id < MAX_N; tar_id++) {
                auto id_pair = bearing_observation_select(target_states_mat.row(tar_id),target_pos_observation,used_ob_id);
                auto &ob_id = id_pair.first;
                auto &delta_min = id_pair.second;

                //std::cout << "id:" << ob_id << " delta min:" << delta_min << std::endl;
                ass[tar_id] = ob_id;
                used_ob_id.insert(ob_id);

                if (delta_min < 0.5) {
                    valid_tar_id.insert(ob_id);
                }
            }

    }

    void dist_id(const std::vector<float>& dist_vec,std::vector<int>& indices) {
        for (int i = 0; i < MAX_N; i++) {
            indices[i] = i;
        }

        // Sort indices based on dist_vec values
        std::sort(indices.begin(), indices.end(), [&dist_vec](int i1, int i2) {
            return dist_vec[i1] < dist_vec[i2];
        });

    }

    void draw_result( const std::vector<float>& metric_vec, 
                      const float& threshold,
                      const std::vector<float>& valid_tar_vec, 
                      cv::Mat& temp_att_debug) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& bbx = bbx_;
        const int img_height = std::stoi(ParamManager::getInstance().getParam("img_height"));
        const int img_width = std::stoi(ParamManager::getInstance().getParam("img_width"));
        const float att_thres1 = std::stof(ParamManager::getInstance().getParam("att_thres1"));
        const float att_thres2 = std::stof(ParamManager::getInstance().getParam("att_thres2"));
        for (size_t i = 0; i < bbx.size(); i += 4) {
            float x_min = bbx[i];
            float y_min = bbx[i + 1];
            float x_max = bbx[i + 2];
            float y_max = bbx[i + 3];

            // flip display rectangle
            x_min = img_width - x_min - 1;
            x_max = img_width - x_max - 1;
            y_max = img_height - y_max - 1;
            y_min = img_height - y_min - 1;
            float x = (x_min + x_max) / 2.0f;
            float y = (y_min + y_max) / 2.0f;

            cv::Point pt1(x_min, y_min); // Top-left corner
            cv::Point pt2(x_max, y_max); // Bottom-right corner
            size_t idx = i / 4;
            cv::Scalar color(255,255,255);
           // if (metric_vec[idx] > threshold ) {
            if (valid_tar_vec[idx] > att_thres1 && valid_tar_vec[idx] <= att_thres2) {
                color = cv::Scalar(255,0,0);
            } else if (valid_tar_vec[idx] > att_thres2) {
                color = cv::Scalar(0,255,0);
            }
            cv::rectangle(temp_att_debug, pt1, pt2, color, 1,cv::LINE_8, 0);
            if (idx < metric_vec.size()) {
                std::string att_str = std::to_string(metric_vec[idx]);
                // Position for the text (centered)
                cv::Point text_org((x_min + x_max - att_str.size() * 5) / 2, (y_min + y_max + 15) / 2);
                // Write the text on the image
                cv::putText(temp_att_debug, att_str, text_org, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            }
        }
    }

    void check_att_value ( const std::vector<float>& att_vec,
                           const std::vector<float>& dist_vec,
                           const std::vector<float>& valid_tar_vec,
                         cv::Mat& temp_att_debug ) {
        int img_height = std::stoi(ParamManager::getInstance().getParam("img_height"));
        int img_width = std::stoi(ParamManager::getInstance().getParam("img_width"));
        temp_att_debug = cv::Mat::zeros(img_height,img_width,CV_8UC3);
        if (mode_ == 1) {  //visual attention
            draw_result(att_vec,att_thres_,valid_tar_vec,temp_att_debug);
        } else if (mode_ == 2) {  //metric
            draw_result(dist_vec,dist_metric_,valid_tar_vec,temp_att_debug);
        }
        // cv::imshow("debug",temp_att_debug);
        // cv::waitKey(100);
    }


    void publish_command( const std::vector<float>& att_vec,  //observation id
                            const std::vector<int>& ass,   // ass map: ob_id = ass[i];   state_id -> observation_id
                            std::vector<float>& valid_tar) {
        Eigen::Map<Eigen::Matrix<float, MAX_N, 6, Eigen::RowMajor>> target_states_mat(target_states_.data());
        int nonZeroCount = std::count_if(valid_tar.begin(), valid_tar.end(), [](int x) {
            return x != 0;
        });
        std::vector<float> dist_vec(MAX_N,-1); //all other targets
        compute_distance_metric(ass,dist_vec);  // value in dist_vec = 1 / relative_distance, observation id
        const float att_thres1 = std::stof(ParamManager::getInstance().getParam("att_thres1"));
        const float att_thres2 = std::stof(ParamManager::getInstance().getParam("att_thres2"));
        const float angular_z_max = std::stof(ParamManager::getInstance().getParam("angular_z_max"));
        const float k_ = std::stof(ParamManager::getInstance().getParam("k"));
        const float k_mig = std::stof(ParamManager::getInstance().getParam("k_mig"));
        const float kp_angular_z = std::stof(ParamManager::getInstance().getParam("kp_angular_z"));
        const float kd_angular_z = std::stof(ParamManager::getInstance().getParam("kd_angular_z"));

        int valid_target_num = 0;

        // mode selection
        switch (mode_) {
            case 0:
                printf("Using visual only selection!\n");
                for (int k = 0; k < MAX_N; k++) {
                    valid_tar[k] = 1;
                }
                break;
            case 1:
                 printf("Using visual attention selection!\n");
                 for (size_t k = 0; k < att_vec.size(); k++) {

                    if (att_vec[k] > att_thres1) {
                        valid_tar[k] = att_vec[k];
                        valid_target_num ++;
                    } 
                    else {
                        valid_tar[k] = 0;
                    }
                    printf("ob %ld att value ",k);
                    print_colored_float(valid_tar[k]);
                 }
                 if (valid_target_num == 0) {
                    for (int k = 0; k < MAX_N; k++) {
                        printf("dist %f %f\n",dist_vec[k],dist_metric_);
                        if (dist_vec[k] > dist_metric_ ) {
                            valid_tar[k] = 1;
                        } else {
                            valid_tar[k] = 0;
                        }
                    }
                    printf("\n");
                 }
                 printf("\n");
                 break;
            case 2:
                printf("Using range selection! distance threshold: %f \n",dist_metric_);
                for (int k = 0; k < MAX_N; k++) {
                    printf("dist %f %f\n",dist_vec[k],dist_metric_);
                    if (dist_vec[k] > dist_metric_  ) {
                        valid_tar[k] = 1;
                        valid_target_num ++;
                    } else {
                         valid_tar[k] = 0;
                    }
                }
                printf("\n");
                break;
            case 3:
                printf("Using topological selection! topo number:%d \n",topo_num_);
                std::vector<int> indices(MAX_N); // Create a vector of indices
                dist_id(dist_vec,indices);
                printf("id:");
                for (int k = 0; k < MAX_N; k++) {
                    printf("%d ",indices[k]);
                }
                printf("\n");
                int cnt = 0;
                for (int k = 0; k < MAX_N; k++) {
                    if (abs(dist_vec[indices[k]]) > 1e-01) {
                        valid_tar[indices[k]] = 1;
                        cnt++;
                    }
                    if (cnt == topo_num_) {
                        break;
                    }
                }
                break;
        }
        Eigen::Vector2f v(0,0);
        for(int i = 0; i < MAX_N; i++) {
            int ob_id = ass[i];
            if (!valid_tar[ob_id] || abs(dist_vec[ob_id]) > 1000) {
                continue;
            }
            float norm_r = dist_vec[ob_id];
            float left = 1 - A_ * (norm_r * norm_r * norm_r);
            if (valid_tar[i] > att_thres1) {
                ROS_INFO("norm_r att 1 %f %f\n",1 / norm_r,left);
                v = v + left * Eigen::Vector2f(target_states_mat.row(i)(0),target_states_mat.row(i)(1));
                v = v * k_;
            }
            if (valid_tar[ob_id] > att_thres2) {
                auto r_mig = Eigen::Vector2f(target_states_mat.row(i)(0),target_states_mat.row(i)(1));
                v = v + k_mig * r_mig.normalized();
                ROS_INFO("norm_r att 2 %f %f\n",1 / norm_r,left);
            }
        }
                

        if (v.norm() > v_max_) {
            v = v.normalized() * v_max_;
        }
        // if (v(0) < 0) {
        //     v(0) = 0;
        // }
        geometry_msgs::Twist control_cmd;
        control_cmd.linear.x = v(0);
        control_cmd.linear.y = v(1);

        float x_delta = atan2(v(1),v(0));
        float angular_z = kp_angular_z * (x_delta) + kd_angular_z * (x_delta - yaw_last_);
        if (angular_z > angular_z_max) {
            angular_z = angular_z_max;
        } else if (angular_z < -angular_z_max) {
            angular_z = -angular_z_max;
        }
        // if (angular_z < 0) {
        //     angular_z = 0;
        // }
        control_cmd.angular.x = 0;
        control_cmd.angular.y = 0;
        control_cmd.angular.z = angular_z;
        yaw_last_ = x_delta;

        //  control_cmd.linear.x = 0;
        //  control_cmd.linear.y = 0;
        //  control_cmd.angular.z = 0;
        ROS_INFO("v %f %f %f\n",v(0),v(1),control_cmd.angular.z);
        vel_pub_.publish(control_cmd);
        sleep(0.5);
    }

    void compute_distance_metric(const std::vector<int>& ass,std::vector<float>& dist_vec) {
        std::lock_guard<std::mutex> lock(mutex_);
        Eigen::Map<Eigen::Matrix<float, MAX_N, 6, Eigen::RowMajor>> target_states_mat(target_states_.data());
        dist_vec.resize(MAX_N,0);
        for (int i = 0; i < MAX_N; i++) {
            int ob_id = ass[i];
            if (ob_id == -1) {
                ob_id = i;
            }
            Eigen::Vector2f tar(target_states_mat(i,0),target_states_mat(i,1));
            dist_vec[ob_id] = 1 / tar.norm();
        }
    }

    void process() {
        Eigen::MatrixXf target_current_observation(MAX_OB, 4);
        Eigen::MatrixXf target_pos_observation(MAX_OB, 3);
        target_current_observation.setZero();
        target_pos_observation.setZero();
        //ObservationProcessor processor(320.0f, 180 - 30, 345.0f,410.0f, false);
        ObservationProcessor processor(205.0f, 90.0f, 205.0f, 410.0f, false);

        const float sigma_mu = std::stof(ParamManager::getInstance().getParam("sigma_mu"));
        const float sigma_omega = std::stof(ParamManager::getInstance().getParam("sigma_omega"));
        const float sigma_mu2 = sigma_mu * sigma_mu;
        const float sigma_omega2 = sigma_omega * sigma_omega;

        while (ros::ok()) {
            std::vector<float> avg_prob = {-1,-1,-1,-1};
            mutex_.lock();
            TicToc timer;
            timer.tic();
            target_pos_observation.setZero();
            processor.compute_target_observation(bbx_, attention_img_,avg_prob, target_current_observation,target_pos_observation);
            mutex_.unlock();
            Eigen::Map<Eigen::Matrix<float, MAX_N, 6, Eigen::RowMajor>> target_states_mat(target_states_.data());
            Eigen::Map<Eigen::Matrix<float, 6*MAX_N, 6, Eigen::RowMajor>> target_P_mat(PEst_.data());
            if (!target_pos_observation.isZero() && first_ob_) {
                target_states_mat.block(0, 0, MAX_N, target_pos_observation.cols()) = target_pos_observation;
                first_ob_ = false;
            }

            //select valid target and apply plkf update
            std::set<int> valid_tar_id;
            std::vector<int> ass(MAX_N,-1);
            target_ass(target_states_mat,target_pos_observation,ass,valid_tar_id);

            // std::cout << "ass data: ";
            // for (std::vector<int>::iterator it = ass.begin(); it != ass.end(); ++it) {
            //     std::cout << *it << " ";
            // }
            // std::cout << std::endl;

            // std::cout << "valid_tar_id: ";
            // for (std::set<int>::iterator it = valid_tar_id.begin(); it != valid_tar_id.end(); ++it) {
            //     std::cout << *it << " ";
            // }
            // std::cout << std::endl;

            for(int tar_id = 0; tar_id < MAX_N; tar_id++) {
                int ob_id = ass[tar_id];
                if (valid_tar_id.count(ob_id) > 0) {  //update
                    //=========compute R============
                    float p_norm = target_pos_observation.row(ob_id).norm();
                    Eigen::VectorXf g = target_pos_observation.row(ob_id).normalized();
                    float theta2 = target_current_observation.row(ob_id)(3) * target_current_observation.row(ob_id)(3);

                    Eigen::MatrixXf R_ob = p_norm * p_norm * ( sigma_mu2 * Eigen::MatrixXf::Identity(3,3) + 
                    sigma_omega2 / theta2 * g * g.transpose());
                    //std::cout << "update " << std::endl;
                    //==============================

                    plkf_estimation(tar_id,target_pos_observation.row(ob_id),R_ob);
                } else { // reset states
                    target_states_mat.row(tar_id)(0) = target_pos_observation.row(ob_id)(0);
                    target_states_mat.row(tar_id)(1) = target_pos_observation.row(ob_id)(1);
                    target_states_mat.row(tar_id)(2) = target_pos_observation.row(ob_id)(2);
                    target_states_mat.row(tar_id)(3) = 0;
                    target_states_mat.row(tar_id)(4) = 0;
                    target_states_mat.row(tar_id)(5) = 0;
                }
            }
            // std::cout << "target_states_:\n" << target_states_mat << std::endl;
            // std::cout << "target_pos_observation:\n" << target_pos_observation << std::endl;
            // std::cout << std::endl;
            float t = timer.toc();
            ROS_INFO("plkf:%f ms\n",t);

            //============== publish control command ===================================
            std::vector<float> valid_tar(MAX_N,0);
            publish_command(avg_prob,ass,valid_tar);
            //============== check and publish attention value ===========================
            if (check_att_value_) {
                std::vector<float> dist_vec;
                // std::cout << "ass data: ";
                // for (std::vector<int>::iterator it = ass.begin(); it != ass.end(); ++it) {
                //     std::cout << *it << " ";
                // }
                compute_distance_metric(ass,dist_vec);
                cv::Mat temp_att_debug;
                check_att_value(avg_prob,dist_vec,valid_tar,temp_att_debug);
                std_msgs::Header header;
                header.stamp = ros::Time::now();
                sensor_msgs::ImagePtr msg_att_value = cv_bridge::CvImage(header, "bgr8", temp_att_debug).toImageMsg();
                att_debug_pub_.publish(msg_att_value);
            }
            //============== publish estimation and observation result===================
            vas::TransformStampedList esti_transform_list_msg;
            esti_transform_list_msg.header.stamp = ros::Time::now();
            for (int k = 0; k < target_states_mat.rows(); k++) {
                geometry_msgs::TransformStamped transform;
                transform.header.stamp = ros::Time::now();
                transform.transform.translation.x = target_states_mat.row(k)(0);
                transform.transform.translation.y = target_states_mat.row(k)(1);
                transform.transform.translation.z = target_states_mat.row(k)(2);
                transform.transform.rotation.x = target_states_mat.row(k)(3);
                transform.transform.rotation.y = target_states_mat.row(k)(4);
                transform.transform.rotation.z = target_states_mat.row(k)(5);
                transform.transform.rotation.w = 1.0;
                esti_transform_list_msg.transforms.push_back(transform);
            }
            est_pub_.publish(esti_transform_list_msg);
            vas::TransformStampedList obs_transform_list_msg;
            obs_transform_list_msg.header.stamp = ros::Time::now();
            for (int k = 0; k < target_pos_observation.rows(); k++) {
                geometry_msgs::TransformStamped transform;
                transform.header.stamp = ros::Time::now();
                transform.transform.translation.x = target_pos_observation.row(k)(0);
                transform.transform.translation.y = target_pos_observation.row(k)(1);
                transform.transform.translation.z = target_pos_observation.row(k)(2);
                obs_transform_list_msg.transforms.push_back(transform);
            }
            obs_pub_.publish(obs_transform_list_msg);
            loop_rate_.sleep();
            ros::spinOnce();
        }
    }

private:

    void check_histgram(const cv::Mat& src) {
        cv::Mat hist;
        int histSize = 256;
        float range[] = {0, 256};
        const float* histRange = {range};
        cv::calcHist(&src, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
        int hist_w = 512; 
        int hist_h = 400;
        int bin_w = cvRound((double) hist_w / histSize);
        cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
        normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
        for (int i = 1; i < histSize; i++) {
            cv::line(histImage, cv::Point(bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1))),
            cv::Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
            cv::Scalar(255, 0, 0), 2, 8, 0);
        }
        // cv::imshow("Histogram", histImage);
        // cv::waitKey(1);
    }

    void callback(const sensor_msgs::ImageConstPtr& data0, 
                  const vas::bbxConstPtr& data1) {
        std::lock_guard<std::mutex> lock(mutex_);
        try {
            ROS_INFO("Callback triggered");
            current_time_ = data0->header.stamp;
            attention_img_ = cv_bridge::toCvCopy(data0, sensor_msgs::image_encodings::MONO8)->image;
           // omni_img_ = cv_bridge::toCvCopy(data2, sensor_msgs::image_encodings::RGB8)->image;
            bbx_ = data1->data;
            cv::equalizeHist(attention_img_, attention_img_);
            // check_histgram(attention_img_);
            // // Display the image
            // cv::imshow("attention_img", attention_img_);
            // cv::waitKey(1);
        }
        catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }

    ros::NodeHandle nh_;
    ros::Publisher v_att_debug_pub_;
    ros::Publisher est_pub_;
    ros::Publisher obs_pub_;
    ros::Publisher att_debug_pub_;
    ros::Rate loop_rate_;
    ros::Publisher vel_pub_;

    std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> omni_img_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> original_img_sub_;
    std::shared_ptr<message_filters::Subscriber<vas::bbx>> bbx_sub_;
    std::mutex mutex_;
    ros::Time current_time_;
    cv::Mat omni_img_;
    cv::Mat attention_img_;

    std::vector<float> bbx_;
    std::vector<float> target_states_;
    std::vector<float> PEst_;
    std::set<int> all_tar_id_;

    //params
    const float k_; 
    const float v_max_;
    const float A_;
    const int mode_;
    const float dist_metric_;
    const int topo_num_;
    const int check_att_value_;
    const float att_thres_;
    float yaw_last_;
    
    bool first_ob_;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,vas::bbx> SyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
};

void signalHandler(int signum) {
    ros::shutdown();
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "plkf_node");
    ParamManager::getInstance().loadParam();
    
    const float k = std::stof(ParamManager::getInstance().getParam("k"));
    const float k_sep = std::stof(ParamManager::getInstance().getParam("k_sep"));
    const float v_max = std::stof(ParamManager::getInstance().getParam("v_max"));
    const int mode =  std::stoi(ParamManager::getInstance().getParam("neighbor_selection"));
    const float A = k_sep * k_sep * k_sep;
    const float dist_metric = std::stof(ParamManager::getInstance().getParam("dist_metric"));
    const int topo_num = std::stoi(ParamManager::getInstance().getParam("topo_num"));
    const int check_att_value = std::stoi(ParamManager::getInstance().getParam("check_att_value"));
    const float att_thres = std::stof(ParamManager::getInstance().getParam("att_thres"));

    PLKF plkf_sub(k,A,v_max,mode,dist_metric,topo_num,check_att_value,att_thres);
    plkf_sub.process();
    return 0;
}