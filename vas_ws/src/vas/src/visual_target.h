#include <Eigen/Dense>
#include <iostream>
#include <cstdio>
#include <vector>
#include <opencv2/opencv.hpp>

void print_colored_float(const float& value);
Eigen::Matrix4f get_x_mat(const float& theta);
Eigen::Matrix4f get_z_mat(const float& theta);
Eigen::Matrix4f get_y_mat(const float& theta);

class VisualTarget {
public:
    int id_;
    float x_, y_, s_pixel_, single_width_;
    Eigen::Vector3f p_;
    float distance_;
    float robot_size_;

    VisualTarget(int img_id, float original_x, float original_y, float bbx_x_size, float single_width)
        : id_(img_id), 
        x_(original_x), 
        y_(original_y), 
        s_pixel_(bbx_x_size), 
        single_width_(single_width), 
        robot_size_(0.3) {
            p_.setZero();
        }

    VisualTarget(const Eigen::Vector3f& p_in_body)
        : p_(p_in_body),robot_size_(0.3) {
        }

    Eigen::Vector3f compute_relative_bv(float cx, float cy, float focal);
    //void compute_image_projection();
};

class ObservationProcessor {
public:

void compute_target_observation( const std::vector<float>& bbx, 
                                 const cv::Mat& att_map,
                                 std::vector<float>& avg_prob,
                                 Eigen::MatrixXf& target_observations,
                                 Eigen::MatrixXf& target_position);

    ObservationProcessor(const float cx, const float cy, const float focal, const float single_img_width, const bool dump_data)
        : cx(cx), cy(cy), focal(focal), single_img_width(single_img_width), dump_data(dump_data) {
        processed_frame = 0;
    };

private:
    float cx, cy, focal, single_img_width;
    bool dump_data;
    Eigen::MatrixXf bearing_angle_vec;
    int processed_frame;
    float compute_avg_att(const cv::Mat& att_map, const int& u, const int& v);
};
