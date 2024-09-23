#include <unistd.h>
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

bool configureCamera(cv::VideoCapture &cap, int device_id, int width = 640, int height = 360, int fps = 30)
{
    cap.open(device_id);
    if (!cap.isOpened())
    {
        ROS_ERROR("Failed to open camera with device ID %d.", device_id);
        return false;
    }

    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    cap.set(cv::CAP_PROP_FPS, fps);
    return true;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "usb_cam_pub");
    ros::NodeHandle nh;
    ros::Publisher image_pub = nh.advertise<sensor_msgs::CompressedImage>("camera/image/compressed", 1);

    cv::VideoCapture cap0, cap2, cap4, cap6;
    configureCamera(cap0, 0);
    sleep(1);

    configureCamera(cap2, 2);
    sleep(1);

    configureCamera(cap4, 4);
    sleep(1);

    configureCamera(cap4, 6);
    sleep(1);


    // if (!configureCamera(cap0, 0) || !configureCamera(cap2, 2) ||
    //     !configureCamera(cap4, 4) || !configureCamera(cap4, 6))
    // {
    //     return -1;
    // }

    cv::Mat frame0, frame2, frame4, frame6, concatenated_frame;
    ros::Rate loop_rate(30);  // 30Hz
    sleep(1);
    while (nh.ok())
    {
        cap0 >> frame0;
        cap2 >> frame2;
        cap4 >> frame4;
        cap6 >> frame6;

        if (frame0.empty())
        {
            ROS_WARN("Failed to capture image from one or more cameras.%d",0);
        }
        if (frame2.empty())
        {
            ROS_WARN("Failed to capture image from one or more cameras.%d",2);
        }
        if (frame4.empty())
        {
            ROS_WARN("Failed to capture image from one or more cameras.%d",4);
        }
        if (frame6.empty())
        {
            ROS_WARN("Failed to capture image from one or more cameras.%d",6);
        }
        if (frame0.empty() || frame2.empty() || frame4.empty() || frame6.empty())
        {
            continue;
        }
        cv::hconcat(frame0, frame2, concatenated_frame);
        cv::hconcat(concatenated_frame, frame4, concatenated_frame);
      //  cv::hconcat(concatenated_frame, frame6, concatenated_frame);

        std::vector<uchar> buf;
        cv::imencode(".jpg", concatenated_frame, buf);
        sensor_msgs::CompressedImage msg;
        msg.header.stamp = ros::Time::now();
        msg.format = "jpeg";
        msg.data = buf;

        image_pub.publish(msg);

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
