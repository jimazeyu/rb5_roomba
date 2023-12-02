#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import apriltag
import cv2
import numpy as np
import time
from mpi_control import MegaPiController


class Landmark:
    """Class for landmarks."""
    
    def __init__(self, x, y, theta, tag_code):
        self.x = x
        self.y = y
        self.theta = theta
        self.tag_code = tag_code  # Internal code


class Robot:
    """Class for the robot."""
    
    def __init__(self, x=0, y=0, theta=0):
        self.mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
        # Initialize robot state
        self.x = x
        self.y = y
        self.theta = theta
        self.vx = 0
        self.vy = 0
        self.omega = 0
        # Record the error from the previous moment
        self.prev_error_x = 0
        self.prev_error_y = 0

        # List of landmarks
        self.landmarks = []
        self.landmarks.append(Landmark(1.2, 0.35, 3.14, 0))
        self.landmarks.append(Landmark(0.5, 1.2, -3.14/2.0, 1))
        self.landmarks.append(Landmark(0.0, 1.0, 0.0, 2))
        self.landmark_now = 0
        # Time from the previous moment
        self.prev_time = time.time()

    def move_straight(self, vx, vy):
        """Move the robot straight."""
        self.omega = 0.0
        l = 0.1354
        r = 0.0305
        A = np.array([[1, -1, -1*l],[1, 1, l],[1, 1, -1*l],[1, -1, l]]) / r
        Omega = np.dot(A, np.array([vx, vy, self.omega])) * 100 / 14.1124
        print(Omega)
        self.mpi_ctrl.setFourMotors(-Omega[2], Omega[1], -1 * Omega[0], Omega[3])  # -bl, fr, -fl, br

    def rotate(self, omega):
        """Rotate the robot."""
        self.mpi_ctrl.setFourMotors(omega, omega, omega, omega)

    def stop(self):
        """Stop the robot."""
        self.mpi_ctrl.setFourMotors(0, 0, 0, 0)

    def detect_landmark(self, tag_id):
        """Detect landmark. If found, return (theta, distance). If not found, return False."""
        # Create CvBridge object
        bridge = CvBridge()
        
        # Get image
        msg = rospy.wait_for_message('/calibrated_camera_image', Image)

        # Convert ROS image message to OpenCV image
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Create Apriltag detector object
        detector = apriltag.Detector()
        
        # Detect Apriltag
        results = detector.detect(gray_image)

        landmark = None
        # If detected, calculate the relative pose
        for result in results:
            if result.tag_id == self.landmarks[tag_id].tag_code:
                landmark = result
        if landmark is None:
            return False
        else:
            camera_params = [451.32982, 450.96586, 631.24643, 349.08218]
            tag_pose = detector.detection_pose(landmark, camera_params)[0]
            # Convert pose to 2D relative pose
            x = tag_pose[0][3]
            y = tag_pose[1][3]
            z = tag_pose[2][3]
            roll = np.arctan2(tag_pose[2][1], tag_pose[2][2])
            pitch = np.arctan2(-tag_pose[2][0], np.sqrt(tag_pose[2][1]**2 + tag_pose[2][2]**2))
            yaw = np.arctan2(tag_pose[1][0], tag_pose[0][0])
            x_2d, y_2d = z*18.0/100.0, -x*18.0/100.0
            theta_2d = 3.14 + pitch
            # Limit theta_2d within the range [-pi, pi]
            if theta_2d > 3.14:
                theta_2d -= 2*3.14
            elif theta_2d < -3.14:
                theta_2d += 2*3.14
            print('x_2d: ', x_2d, 'y_2d: ', y_2d, 'theta_2d: ', theta_2d/3.14*180)
            return x_2d, y_2d, theta_2d

    def move_to(self, x, y, theta, tag_id):
        """Move the robot to a specific position."""
        self.landmark_now = tag_id
        while True:
            # Detect landmark
            try:
                detection = self.detect_landmark(tag_id)
            except Exception as e:
                detection = False
            # If not detected successfully, update the position using the current speed
            if detection == False:
                print('No landmark!')
                self.x += self.vx * (time.time() - self.prev_time)
                self.y += self.vy * (time.time() - self.prev_time)
                self.theta += self.omega * (time.time() - self.prev_time)
            else:
                print('Find landmark!')
                # Detected successfully, get relative pose with the landmark
                relative_x = detection[0]
                relative_y = detection[1]
                relative_theta = detection[2]
                # Calculate the absolute pose (using the absolute pose of the landmark)
                now_x = self.landmarks[self.landmark_now].x + relative_x * np.cos(self.landmarks[self.landmark_now].theta) - relative_y * np.sin(self.landmarks[self.landmark_now].theta)
                now_y = self.landmarks[self.landmark_now].y + relative_x * np.sin(self.landmarks[self.landmark_now].theta) + relative_y * np.cos(self.landmarks[self.landmark_now].theta)
                now_theta = self.landmarks[self.landmark_now].theta - relative_theta
                # Smooth the current frame with the previous frame
                self.x = self.x * 0.5 + now_x * 0.5
                self.y = self.y * 0.5 + now_y * 0.5
                self.theta = self.theta * 0.5 + now_theta * 0.5

            # Limit theta within the range [-pi, pi]
            if self.theta > 3.14:
                self.theta -= 2*3.14
            elif self.theta < -3.14:
                self.theta += 2*3.14                
            # Output the current pose
            print('x: ', self.x, 'y: ', self.y, 'theta: ', self.theta)
            with open('/root/rb5_ws/src/rb5_ros/rb5_tracking/src/robot_pose.txt', 'a') as file:
                # Write the pose to the file in a comma-separated format
                file.write("%s, %s, %s\n" % (self.x, self.y, self.theta))

            # If the error with the target point is within a certain range, break
            if abs(self.x - x) < 0.05 and abs(self.y - y) < 0.05 and abs(self.theta - theta) < 0.10:
                with open('/root/rb5_ws/src/rb5_ros/rb5_tracking/src/robot_error.txt', 'a') as file:
                    # Write the error to the file in a comma-separated format
                    file.write("%s, %s, %s\n" % (self.x, self.y, self.theta))
                print("Arrived!")
                robot.stop()
                break

            # Calculate the error in x, y, and theta
            error_x = x - self.x
            error_y = y - self.y
            error_distance = np.sqrt(error_x**2 + error_y**2)
            error_theta = theta - self.theta
            # Limit error_theta within the range [-pi, pi]
            if error_theta > 3.14:
                error_theta -= 2*3.14
            elif error_theta < -3.14:
                error_theta += 2*3.14
            print('error_x: ', error_x, 'error_y: ', error_y, 'error_theta: ', error_theta)

            # If error_theta is too large, prioritize correcting error_theta
            if abs(error_theta) > 0.20:
                kp = 0.6
                # Control rotation using a proportional controller
                print('rotate:', kp * error_theta * 300)
                # Limit the maximum angular speed to 40
                omega = min(40, abs(kp * error_theta * 300))
                omega = omega if error_theta > 0 else -omega

                robot.rotate(omega)
                # Update angular speed (requires calibration)
                self.omega = omega / 150.0
            else:
                kp = 0.4
                ki = 0.01
                kd = 0.01
                # Calculate the differential and integral terms
                error_x_diff = error_x - self.prev_error_x
                error_y_diff = error_y - self.prev_error_y
                # Only keep the error from the last two frames for integration (limiting)
                error_x_sum = error_x + self.prev_error_x
                error_y_sum = error_y + self.prev_error_y

                # Control straight movement using a PID controller
                print('move x:', kp * error_x + ki * error_x_sum + kd * error_x_diff)
                print('move y:', kp * error_y + ki * error_y_sum + kd * error_y_diff)
                vx = kp * error_x + ki * error_x_sum + kd * error_x_diff
                vy = kp * error_y + ki * error_y_sum + kd * error_y_diff

                # Limit the maximum linear speed to 0.2
                vx = min(0.2, abs(vx))
                vy = min(0.2, abs(vy))
                kp = 0.4
                # Limit the minimum linear speed to 0.135
                vx = max(0.135, abs(vx))
                vy = max(0.135, abs(vy))                
                vx = vx if error_x > 0 else -vx
                vy = vy if error_y > 0 else -vy

                # Convert the speed in the world coordinates to the robot's speed
                vxr = vx * np.cos(self.theta) + vy * np.sin(self.theta)
                vyr = - vx * np.sin(self.theta) + vy * np.cos(self.theta)                

                print('move_x:', vxr)
                print('move_y:', vyr)                
                robot.move_straight(vxr, vyr)
                # Update linear speed (requires calibration)
                robot.vx = vxr
                robot.vy = vyr


if __name__ == "__main__":
    try:
        if os.path.exists('/root/rb5_ws/src/rb5_ros/rb5_tracking/src/robot_pose.txt'):
            os.remove('/root/rb5_ws/src/rb5_ros/rb5_tracking/src/robot_pose.txt')

        if os.path.exists('/root/rb5_ws/src/rb5_ros/rb5_tracking/src/robot_error.txt'):
            os.remove('/root/rb5_ws/src/rb5_ros/rb5_tracking/src/robot_error.txt')

        rospy.init_node('navigation_node', anonymous=True)
        robot = Robot()
        time.sleep(5)
        robot.move_to(0.5, 0.0, 0, 0)
        time.sleep(3)
        robot.move_to(0.5, 1.0, 3.14/2.0, 1)
        time.sleep(0.1)
        robot.move_to(0.5, 1.0, 3.14, 2)
        time.sleep(3)
        robot.move_to(0.0, 0.0, 0.0, 0)
        robot.stop()
    except Exception as e:
        print(e)