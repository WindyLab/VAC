"""
This file is used for publishing frames
"""
import rospy
import tf2_ros
import geometry_msgs.msg

def publish_temporary_frame():
    rospy.init_node('temporary_frame_publisher')

    static_broadcaster = tf2_ros.StaticTransformBroadcaster()

    transform_stamped = geometry_msgs.msg.TransformStamped()

    transform_stamped.header.stamp = rospy.Time.now()
    transform_stamped.header.frame_id = "/world"  # base frame
    transform_stamped.child_frame_id = "/odom_frame"
    transform_stamped.transform.translation.x = 0.0
    transform_stamped.transform.translation.y = 0.0
    transform_stamped.transform.translation.z = 0.0
    transform_stamped.transform.rotation.x = 0.0
    transform_stamped.transform.rotation.y = 0.0
    transform_stamped.transform.rotation.z = 0.0
    transform_stamped.transform.rotation.w = 1.0

    static_broadcaster.sendTransform(transform_stamped)

    rospy.spin()

if __name__ == '__main__':
    try:
        publish_temporary_frame()
    except rospy.ROSInterruptException:
        pass