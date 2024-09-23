#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "visual_target.h"
#include "get_param.h"


// ANSI escape sequences for color settings
#define RESET   "\033[0m"
#define RED     "\033[31m"      // Red
#define GREEN   "\033[32m"      // Green
#define YELLOW  "\033[33m"      // Yellow
#define BLUE    "\033[34m"      // Blue

void print_colored_float(const float& value) {
    if (value < 0.4f) {
        printf(RED "%.2f" RESET "\n", value);
    } else if (value < 0.8f) {
        printf(BLUE "%.2f" RESET "\n", value);
    } else {
        printf(GREEN "%.2f" RESET "\n", value);
    }
}

Eigen::Matrix4f get_x_mat(const float& theta) {
    Eigen::Matrix4f transformation_matrix = Eigen::Matrix4f::Identity();
    transformation_matrix(1, 1) = std::cos(theta);
    transformation_matrix(1, 2) = -std::sin(theta);
    transformation_matrix(2, 1) = std::sin(theta);
    transformation_matrix(2, 2) = std::cos(theta);
    return transformation_matrix;
}

Eigen::Matrix4f get_z_mat(const float& theta) {
    Eigen::Matrix4f transformation_matrix = Eigen::Matrix4f::Identity();
    transformation_matrix(0, 0) = std::cos(theta);
    transformation_matrix(0, 1) = -std::sin(theta);
    transformation_matrix(1, 0) = std::sin(theta);
    transformation_matrix(1, 1) = std::cos(theta);
    return transformation_matrix;
}

Eigen::Matrix4f get_y_mat(const float& theta) {
    Eigen::Matrix4f transformation_matrix = Eigen::Matrix4f::Identity();
    transformation_matrix(0, 0) = std::cos(theta);
    transformation_matrix(0, 2) = std::sin(theta);
    transformation_matrix(2, 0) = -std::sin(theta);
    transformation_matrix(2, 2) = std::cos(theta);
    return transformation_matrix;
}

Eigen::Vector3f VisualTarget::compute_relative_bv(float cx, float cy, float focal) {
        float u = x_ - single_width_ * id_;
        float v = y_;
        float xp = (u - cx) / focal;
        float yp = (v - cy) / focal;
        Eigen::Vector3f bearing(xp, yp, 1.0);
        bearing.normalize();

        float delta_x = abs((u - cx));
        float delta_y = abs((v - cy));
        float focal2 = focal*focal;
        float l_left = focal2 + (delta_x - s_pixel_/2)*(delta_x - s_pixel_/2) + delta_y * delta_y;
        l_left = sqrt(l_left);
        float l_right = focal2 + (delta_x + s_pixel_/2)*(delta_x + s_pixel_/2) + delta_y * delta_y;
        l_right = sqrt(l_right);

        float theta = (l_left*l_left + l_right*l_right - s_pixel_*s_pixel_) / l_left / l_right * 0.5;
        theta = acos(theta);
        distance_ = robot_size_ / theta;

        bearing *= distance_;
        Eigen::Vector4f p_in_cam(bearing(0), bearing(1), bearing(2), 1.0);
        Eigen::Matrix4f camera_body;
        if (id_ == 3) {
            camera_body = get_z_mat(M_PI) * get_x_mat(M_PI / 2);
        } else if (id_ == 2) {
            camera_body = get_z_mat(M_PI) * get_x_mat(M_PI / 2) * get_z_mat(M_PI / 2);
        } else if (id_ == 1) {
            camera_body = get_z_mat(M_PI) * get_x_mat(M_PI / 2) * get_z_mat(M_PI);
        } else if (id_ == 0) {
            camera_body = get_z_mat(M_PI) * get_x_mat(M_PI / 2) * get_z_mat(-M_PI / 2);
        }
        Eigen::Vector4f p_in_body = camera_body.inverse() * p_in_cam;
        p_ = p_in_body.head<3>();
        return p_;
}

float ObservationProcessor::compute_avg_att(const cv::Mat& att_map, const int& u, const int& v) {
    int sum = 0;
    int count = 0;
    for (int du = -11; du <= 11; ++du) {
        for (int dv = -11; dv <= 11; ++dv) {
            int x = u + du;
            int y = v + dv;
            if (x >= 0 && x < att_map.cols && y >= 0 && y < att_map.rows) {
                sum += att_map.at<uint8_t>(y, x);
                ++count;
            }
        }
    }
    if (count == 0) {
        return -1;
    }
    return static_cast<float>(sum) / count / 255.0;
}

void ObservationProcessor::compute_target_observation(const std::vector<float>& bbx, const cv::Mat& att_map,
                                            std::vector<float>& avg_prob,
                                            Eigen::MatrixXf& target_observations,
                                            Eigen::MatrixXf& target_position
                                            ) {
    std::vector<int> valid_target_id;
    float target_sz = std::stof(ParamManager::getInstance().getParam("target_sz"));
    float resize_scale = std::stof(ParamManager::getInstance().getParam("vatt_rz"));
    
    for (size_t i = 0; i < bbx.size(); i += 4) {
        float x_min = bbx[i];
        float y_min = bbx[i + 1];
        float x_max = bbx[i + 2];
        float y_max = bbx[i + 3];
        float x = (x_min + x_max) / 2.0f;
        float y = (y_min + y_max) / 2.0f;

        int img_id = static_cast<int>(std::floor(x / single_img_width));

        VisualTarget v_tar(img_id, x, y, x_max - x_min, single_img_width);
        Eigen::Vector3f p_in_body = v_tar.compute_relative_bv(cx, cy, focal);
        Eigen::Vector3f unit_bearing = p_in_body.normalized();
        
        int ob_id = static_cast<int>(std::floor(i / 4));
        int vatt_u = x * resize_scale;
        int vatt_v = y * resize_scale;  //resize to attention map scale

        float avg_att_value = compute_avg_att(att_map,vatt_u,vatt_v);
        avg_prob[ob_id] = avg_att_value;
        //ROS_INFO("vatt_u %d vatt_v %d %d %d",vatt_u,vatt_v,att_map.rows,att_map.cols);
        //ROS_INFO("image id %d avg_att_value %f %f",img_id,avg_att_value,resize_scale);

        target_observations.row(ob_id)(0) = unit_bearing.x();
        target_observations.row(ob_id)(1) = unit_bearing.y();
        target_observations.row(ob_id)(2) = unit_bearing.z();
        target_observations.row(ob_id)(3) = target_sz / p_in_body.norm();
        target_position.row(ob_id)(0) = p_in_body(0);
        target_position.row(ob_id)(1) = p_in_body(1);
        target_position.row(ob_id)(2) = p_in_body(2);
    }
}