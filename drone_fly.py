import rospy
import os
import cv2
import math
from geometry_msgs.msg import Twist
from mavros_msgs.srv import CommandBool, SetMode
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Global variables
bridge = CvBridge()
image_counter = 0

def takeoff(altitude):
    rospy.wait_for_service('mavros/cmd/takeoff')
    takeoff_service = rospy.ServiceProxy('mavros/cmd/takeoff', CommandBool)
    response = takeoff_service(altitude)
    return response.success

def distance(point1, point2):
    # Calculate the distance between two points
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

def set_offboard_mode():
    rospy.wait_for_service('mavros/set_mode')
    set_mode_service = rospy.ServiceProxy('mavros/set_mode', SetMode)
    response = set_mode_service(custom_mode='OFFBOARD')
    return response.mode_sent

def take_image():
    global image_counter

    # Wait for the /iris_with_standoffs_demo/front_camera/image_raw topic to be published
    msg = rospy.wait_for_message('/iris_with_standoffs_demo/front_camera/image_raw', Image)

    # Convert the image message to a OpenCV image
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr(e)
        return

    # Save the image to a file
    image_name = "image_{}.png".format(image_counter)
    cv2.imwrite(os.path.join('images', image_name), cv_image)
    image_counter += 1

def move_drone(x, y, z):
    # Create a Twist message and fill in the necessary fields
    msg = Twist()
    msg.linear.x = x
    msg.linear.y = y
    msg.linear.z = z

    # Publish the message to the /iris_with_standoffs_demo/cmd_vel topic
    pub = rospy.Publisher('/iris_with_standoffs_demo/cmd_vel', Twist, queue_size=10)
    pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node('move_drone')

    # Create the images folder if it does not exist
    if not os.path.exists('images'):
        os.makedirs('images')

    # Subscribe to the /iris_with_standoffs_demo/front_camera/image_raw topic
    sub = rospy.Subscriber('/iris_with_standoffs_demo/front_camera/image_raw', Image, take_image)

    # Defines points
    A = (12, -1, 1)
    B = (-12, -1, 1)
    distance_between = distance(A, B)

    # Take off
    set_offboard_mode()
    takeoff(1)

    # Take an image at point A
    take_image()
    rospy.sleep(1)

    # Move the drone towards point B, taking an image every meter
    distance_moved = 0

    while distance_moved < distance_between:
        move_drone(-1, 0, 0)
        rospy.sleep(1)
        take_image()
        distance_moved += 1

    # Take an image at point B
    take_image()
    rospy.sleep(1)