#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# Define camera calibration parameters.


camera matrix = np.array([[446.632864, 0.0, 631.849391],
                          [0.0, 446.429350, 357.598039],
                          [0.0, 0.0, 1.0]])

distortion_coefficients = np.array([0.013419, -0.015527, 0.000078, -0.000308, 0.0])

def image_callback(msg):
    try:
        # Convert ROS messages to OpenCV images.
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")

        # undistort
        calibrated_image = cv2.undistort(cv_image, camera_matrix, distortion_coefficients, None)

        # Publish the calibrated image data to a new topic.
        calibrated_image_msg = bridge.cv2_to_imgmsg(calibrated_image, "bgr8")
        pub.publish(calibrated_image_msg)
    except Exception as e:
        rospy.logerr("Error processing image: %s", str(e))

if __name__ == '__main__':
    rospy.init_node('image_calibrator')

    # subscribe the camera topic
    rospy.Subscriber('/camera_0', Image, image_callback)

    # Create a publisher to broadcast the calibrated images to a new topic.
    pub = rospy.Publisher('/calibrated_camera_image', Image, queue_size=10)

    rospy.spin()