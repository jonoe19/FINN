import rospy
import os
import cv2
import math
from geometry_msgs.msg import Twist
from mavros_msgs.srv import CommandBool, SetMode
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()

def takeoff(altitude):
    rospy.wait_for_service('mavros/cmd/takeoff')
    takeoff_service = rospy.ServiceProxy('mavros/cmd/takeoff', CommandBool)
    response = takeoff_service(altitude)
    return response.success

def set_offboard_mode():
    rospy.wait_for_service('mavros/set_mode')
    set_mode_service = rospy.ServiceProxy('mavros/set_mode', SetMode)
    response = set_mode_service(custom_mode='OFFBOARD')
    return response.mode_sent

def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

def take_image(x, y):
    msg = rospy.wait_for_message('/iris_with_standoffs_demo/front_camera/image_raw', Image)
    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    image_name = f'x{x}_y{y}.jpg'
    cv2.imwrite(os.path.join('images', image_name), cv_image)

def move_drone(x, y, z):
    msg = Twist()
    msg.linear.x = x
    msg.linear.y = y
    msg.linear.z = z
    pub = rospy.Publisher('/iris_with_standoffs_demo/cmd_vel', Twist, queue_size=10)
    pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node('move_drone')
    sub = rospy.Subscriber('/iris_with_standoffs_demo/front_camera/image_raw', Image, take_image)

    A = (12, -1, 1)
    B = (-12, -1, 1)
    distance_between = distance(A, B)

    set_offboard_mode()
    takeoff(1)

    take_image(x=A[0], y=A[1])
    rospy.sleep(1)

    distance_moved = 0

    while distance_moved < distance_between:
        distance_moved += 1
        move_drone(-1, 0, 0)
        rospy.sleep(1)
        take_image(x=A[0]-distance_moved, y=A[1])

    rospy.sleep(1)
    move_drone(0, 0, -1)