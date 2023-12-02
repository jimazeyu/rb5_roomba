#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import rospy
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray

class PatternDetector:

    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('pattern_detector', anonymous=True)
        self.image_sub = rospy.Subscriber("/calibrated_camera_image", Image, self.callback)
        self.pub = rospy.Publisher('detect_result', Int32MultiArray, queue_size=10)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # cv2.imwrite("result.jpg", cv_image)
            # print(cv_image.shape)
        except CvBridgeError as e:
            print(e)

        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Define range for red color
        lower_red_1 = np.array([0, 100, 100])
        upper_red_1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)

        lower_red_2 = np.array([160, 100, 100])
        upper_red_2 = np.array([179, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

        red_mask = mask1 + mask2

        # Define range for blue color
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

        # Find contours for red mask (detecting squares)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        approx = None

        for contour in red_contours:
            # Approximate contour to detect shape
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True) # Douglas-Peucker algorithm
            # print(approx)

        if approx is not None:
            # Check for square (4 vertices) and area > threshold
            if len(approx) == 4 and cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w)/h
                if 0.8 <= aspect_ratio <= 1.2:
                    # Check if there's a blue circle inside this square
                    print("Red Square: ", aspect_ratio, [x, y, w, h])
                    cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 0, 255), 5)
                    cv2.imwrite("/root/rb5_ws/src/rb5_ros/rb5_tracking/src/sq_result.jpg", cv_image)
                    roi = blue_mask[y:y+h, x:x+w]
                    roi_contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
                    fradius = 0
                    frcounter = None
                    for r_contour in roi_contours:
                        center, radius = cv2.minEnclosingCircle(r_contour)
                        if radius > fradius:
                            fcenter, fradius = center, radius
                            frcounter = r_contour
                    if frcounter is not None:
                        if cv2.contourArea(frcounter) > 800:  # Ensuring it's a significant circle and not noise
                            cv2.circle(cv_image, (int(x + fcenter[0]), int(y + fcenter[1])), int(fradius), (0, 255, 0), 2)
                            cv2.imwrite("/root/rb5_ws/src/rb5_ros/rb5_tracking/src/result.jpg", cv_image)
                            print("Detection succeed!", [x,y,w,h])
                            output_msg = Int32MultiArray()
                            output_msg.data = [x, y, w, h]
                            self.pub.publish(output_msg)
        else:
            print("Detecting......")

        # cv2.imshow("Detected Sign", cv_image)
        # cv2.imwrite("result.jpg", cv_image)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run(self):
        # Keep the node running
        rospy.spin()

if __name__ == '__main__':
    pd = PatternDetector()
    try:
        pd.run()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()