import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import cv2
import io
from sixel import SixelWriter

class ImageChecker():
    def __init__(self):
        rospy.init_node('image_checker_node',anonymous=True)
        self.bridge = CvBridge()
        self.loop_rate = rospy.Rate(1000)
        self.img = None
        self.img_msg_header = None
        message_list = []
        omni_img_sub = message_filters.Subscriber('/cam0/image_raw', Image)
        message_list.append(omni_img_sub)
        ts = message_filters.ApproximateTimeSynchronizer(message_list, 10, 1, allow_headerless=True)
        ts.registerCallback(self.callback)


    def callback(self,data0):
        #self.omni_img = self.bridge.imgmsg_to_cv2(data0, "bgr8")
        self.img = self.bridge.imgmsg_to_cv2(data0, "mono8")
        im_bytes = cv2.imencode(".png",self.img)[1].tobytes()
        mem_file = io.BytesIO(im_bytes)
        w = SixelWriter()
        w.draw(mem_file)


if __name__ == '__main__':
    img_checker = ImageChecker()
    print("checker ok!")
    rospy.spin()