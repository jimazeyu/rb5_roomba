#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import rospy
import pickle
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import apriltag
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse
import math
import time
from mpi_control import MegaPiController
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

# Random seed is used for result reproducibility.
np.random.seed(0)


class Robot:
    def __init__(self):
        self.xEstFull = np.array([0.3, 0.3, 0])
        self.PEstFull = np.diag([0.0, 0.0, 0.0])
        self.u = np.array([0.0, 0.0, 0.0])
        self.vx = 0
        self.vy = 0
        self.omega = 0
        self.prev_error_x = 0
        self.prev_error_y = 0

        # system noise
        self.R = np.diag([0.1**2, 0.1**2, np.deg2rad(5.0)**2])
        # observation noise
        self.Q = np.diag([0.1**2, np.deg2rad(5.0)**2])

        self.landmarksID = []

        self.mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
        self.l = 0.1354
        self.r = 0.0305
        self.A = np.array([[1, -1, -1*self.l],[1, 1, self.l],[1, 1, -1*self.l],[1, -1, self.l]]) / self.r
        self.num_step = 0

        self.prev_time = time.time()

        self.pub_x = rospy.Publisher('xEstFull', Float64MultiArray, queue_size=10)
        self.pub_P = rospy.Publisher('PEstFull', Float64MultiArray, queue_size=10)
        self.pub_lm = rospy.Publisher('Landmarks', Float64MultiArray, queue_size=10)

    def move_straight(self, vx, vy):
        omega = 0.0
        Omega = np.dot(self.A, np.array([vx, vy, omega])) * 100 / 14.1124
        # print(Omega)
        self.mpi_ctrl.setFourMotors(-Omega[2], Omega[1], -Omega[0], Omega[3]) # -bl, fr, -fl, br

    def rotate(self, omega):
        vx = 0
        vy = 0 
        Omega = np.dot(self.A, np.array([vx, vy, 1.2*omega])) * 100 / 14.1124
        self.mpi_ctrl.setFourMotors(-Omega[2], Omega[1], -Omega[0], Omega[3])

    def stop(self):
        self.mpi_ctrl.setFourMotors(0, 0, 0, 0)

    def observe(self):
        observations = []

        bridge = CvBridge()

        msg = rospy.wait_for_message('/camera_0', Image)

        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        detector = apriltag.Detector()

        results = detector.detect(gray_image)

        landmarks = []
        for result in results:
            landmarks.append(result)
        if len(landmarks) == 0:
            return []
        else:
            camera_params = [446.632864, 446.429350, 631.849391, 357.598039]
            for landmark in landmarks:
                tag_pose = detector.detection_pose(landmark, camera_params)[0]
                x = tag_pose[0][3]
                y = tag_pose[1][3]
                z = tag_pose[2][3]
                x_2d, y_2d = z*18.0/100.0, -x*18.0/100.0
                d = np.sqrt(x_2d**2 + y_2d**2)
                angle = np.arctan2(y_2d, x_2d)
                observations.append([d, angle, landmark.tag_id])
        return observations
    
    def move_to(self, x, y, theta_target, delay=10):
        """Move the robot to a specific position."""
        while True:
            self.stop()
            self.u = np.array([self.vx * (time.time()-self.prev_time), 
                               self.vy * (time.time()-self.prev_time), 
                               self.omega * (time.time()-self.prev_time)])
            
            # get observation  
            z = self.observe()   
            # If a new landmark is detected, add it to the list of landmarks.
            for iz in range(len(z)):
                landmark_id = int(z[iz][2])
                list_id = -1
                # If it is a new landmark, add it to the landmark history.
                for i, id in enumerate(self.landmarksID):
                    if id == landmark_id:
                        list_id = i
                        break
                if list_id < 0:
                    print("New Landmark observed, landmark_id = ", landmark_id)
                    self.landmarksID.append(landmark_id)
                    # Initialize the position of the landmark
                    x1 = self.xEstFull[0] + z[iz][0]*np.cos(self.xEstFull[2] + z[iz][1])
                    x2 = self.xEstFull[1] + z[iz][0]*np.sin(self.xEstFull[2] + z[iz][1])
                    # extend xEstFull
                    self.xEstFull = np.hstack((self.xEstFull, np.array([x1, x2])))
                    # extend PEstFull
                    PEstFullTemp = np.zeros((self.PEstFull.shape[0] + 2, self.PEstFull.shape[1] + 2))
                    PEstFullTemp[:self.PEstFull.shape[0], :self.PEstFull.shape[1]] = self.PEstFull
                    np.fill_diagonal(PEstFullTemp[-2:, -2:], 1e10)
                    self.PEstFull = PEstFullTemp

            # prediction robot pose
            Fx = np.hstack((np.eye(3), np.zeros((3, 2*len(self.landmarksID)))))
            theta = self.xEstFull[2]
            uR = np.array([self.u[0]*np.cos(theta) - self.u[1]*np.sin(theta),
                            self.u[0]*np.sin(theta) + self.u[1]*np.cos(theta),
                            self.u[2]])
            self.xEstFull = self.xEstFull + np.dot(Fx.T, uR)
    
            # update robot cov
            G = np.array([[0, 0, -self.u[0]*np.sin(theta) - self.u[1]*np.cos(theta)],
                            [0, 0, self.u[0]*np.cos(theta) - self.u[1]*np.sin(theta)],
                            [0, 0, 0]])
            G = np.dot(np.dot(Fx.T,G), Fx) + np.eye(Fx.shape[1])
            self.PEstFull = np.dot(np.dot(G.T, self.PEstFull), G) + np.dot(np.dot(Fx.T, self.R), Fx)
            
            # theta in -pi to pi
            self.xEstFull[2] = np.mod(self.xEstFull[2] + np.pi, 2*np.pi) - np.pi

            for iz in range(len(z)):
                # observed landmark ID
                landmark_id = int(z[iz][2])
                # index of the landmark in landmarkXEst
                list_id = -1
                for i, id in enumerate(self.landmarksID):
                    if id == landmark_id:
                        list_id = i
                        break
                # 1.δ = (δx δy)^T = (μj,x - μt,x, μj,y - μt,y)^T
                delta = np.array([self.xEstFull[2*list_id + 3] - self.xEstFull[0],
                                    self.xEstFull[2*list_id + 4] - self.xEstFull[1]])
                # 2.q = δ^T δ
                q = np.dot(delta.T, delta)
                # 3.z^i_t = (atan2(δy, δx) - μt,θ)^T
                z_hat = np.array([np.sqrt(q),
                                    np.arctan2(delta[1], delta[0]) - self.xEstFull[2]])
                # （very important）
                z_hat[1] = np.mod(z_hat[1] + np.pi, 2*np.pi) - np.pi
                # 4.Fx,j
                Fxj = np.zeros((5, 3 + 2*len(self.landmarksID)))
                Fxj[0:3, 0:3] = np.eye(3)
                Fxj[3:5, 2*list_id + 3:2*list_id + 5] = np.eye(2)
                # 5.H^i_t = 1/q [-√qδx, -√qδy, 0, √qδx, √qδy; δy, -δx, -q, -δy, δx] * Fx,j
                Hi = 1/q * np.array([[-np.sqrt(q)*delta[0], -np.sqrt(q)*delta[1], 0, np.sqrt(q)*delta[0], np.sqrt(q)*delta[1]],
                                        [delta[1], -delta[0], -q, -delta[1], delta[0]]])
                Hi = np.dot(Hi, Fxj)
                # 6.K^i_t = Σt H^i_t^T (H^i_t Σt H^i_t^T + Rt)^-1
                K = np.dot(np.dot(self.PEstFull, Hi.T), np.linalg.inv(np.dot(np.dot(Hi, self.PEstFull), Hi.T) + self.Q))
                # 7.μt = μt + K^i_t (z^i_t - z^i_t)
                self.xEstFull = self.xEstFull + np.dot(K, (z[iz][0:2] - z_hat))
                # 8.Σt = (I - K^i_t H^i_t) Σt
                self.PEstFull = np.dot((np.eye(self.PEstFull.shape[0]) - np.dot(K, Hi)), self.PEstFull)

            # Limit theta within the range [-pi, pi]
            if self.xEstFull[2] > 3.14:
                self.xEstFull[2] -= 2*3.14
            elif self.xEstFull[2] < -3.14:
                self.xEstFull[2] += 2*3.14    

            # print(np.array([self.xEstFull]).shape, self.PEstFull.shape, np.array(z).shape)            

            msg_x = np.array([self.xEstFull])
            xEstFull_msg = Float64MultiArray()
            xEstFull_msg.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
            xEstFull_msg.layout.dim[0].label = "rows"
            xEstFull_msg.layout.dim[1].label = "columns"
            xEstFull_msg.layout.dim[0].size = msg_x.shape[0]
            xEstFull_msg.layout.dim[1].size = msg_x.shape[1]

            xEstFull_msg.data = msg_x.flatten().tolist()
            self.pub_x.publish(xEstFull_msg)

            PEstFull_msg = Float64MultiArray()
            PEstFull_msg.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
            PEstFull_msg.layout.dim[0].label = "rows"
            PEstFull_msg.layout.dim[1].label = "columns"
            PEstFull_msg.layout.dim[0].size = self.PEstFull.shape[0]
            PEstFull_msg.layout.dim[1].size = self.PEstFull.shape[1]

            PEstFull_msg.data = self.PEstFull.flatten().tolist()
            self.pub_P.publish(PEstFull_msg)

            if len(z) > 0:
                lm_msg = Float64MultiArray()
                lm_msg.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
                lm_msg.layout.dim[0].label = "rows"
                lm_msg.layout.dim[1].label = "columns"
                lm_msg.layout.dim[0].size = np.array(z).shape[0]
                lm_msg.layout.dim[1].size = np.array(z).shape[1]

                lm_msg.data = np.array(z).flatten().tolist()
                self.pub_lm.publish(lm_msg)
                
            # Output the current pose
            print('x: ', self.xEstFull[0], 'y: ', self.xEstFull[1], 'theta: ', self.xEstFull[2])
            with open('/root/rb5_ws/src/rb5_ros/rb5_tracking/src/robot_pose_traj.txt', 'a') as file:
                # Write the pose to the file in a comma-separated format
                file.write("%s, %s, %s\n" % (self.xEstFull[0], self.xEstFull[1], self.xEstFull[2]))

            print(abs(self.xEstFull[0] - x), abs(self.xEstFull[1] - y), abs(self.xEstFull[2] - theta_target))
            # If the error with the target point is within a certain range, break
            if abs(self.xEstFull[0] - x) < 0.05 and abs(self.xEstFull[1] - y) < 0.05 and abs(self.xEstFull[2] - theta_target) < 0.10:
                with open('/root/rb5_ws/src/rb5_ros/rb5_tracking/src/robot_error_target.txt', 'a') as file:
                # Write the error to the file in a comma-separated format
                    file.write("%s, %s, %s\n" % (self.xEstFull[0], self.xEstFull[1], self.xEstFull[2]))
                print("Arrived!")
                self.stop()
                break
            
            if delay > 0:
                delay -= 1
                continue

            # Calculate the error in x, y, and theta
            error_x = x - self.xEstFull[0]
            error_y = y - self.xEstFull[1]
            error_distance = np.sqrt(error_x**2 + error_y**2)
            error_theta = theta_target - self.xEstFull[2]
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
                print('rotate:', kp * error_theta)
                # Limit the maximum angular speed to 40
                # omega = min(1.5, abs(kp * error_theta * 10))
                omega = 1.2
                omega = omega if error_theta > 0 else -omega
                self.rotate(omega)
                self.prev_time = time.time()
                time.sleep(0.2)
                # Update angular speed (requires calibration)
                self.omega = omega
            else:
                kp = 0.4
                # ki = 0.01
                # kd = 0.01
                # Calculate the differential and integral terms
                # error_x_diff = error_x - self.prev_error_x
                # error_y_diff = error_y - self.prev_error_y
                # Only keep the error from the last two frames for integration (limiting)
                # error_x_sum = error_x + .selfprev_error_x
                # error_y_sum = error_y + self.prev_error_y

                # Control straight movement using a PID controller
                # print('move x:', kp * error_x + ki * error_x_sum + kd * error_x_diff)
                # print('move y:', kp * error_y + ki * error_y_sum + kd * error_y_diff)
                print('move x:', kp * error_x)
                print('move y:', kp * error_y)
                # vx = kp * error_x + ki * error_x_sum + kd * error_x_diff
                # vy = kp * error_y + ki * error_y_sum + kd * error_y_diff
                vx = kp * error_x
                vy = kp * error_y

                # Limit the maximum linear speed to 0.2
                vx = min(0.2, abs(vx))
                vy = min(0.2, abs(vy))

                # Limit the minimum linear speed to 0.135
                vx = max(0.135, abs(vx))
                vy = max(0.135, abs(vy))                
                vx = vx if error_x > 0 else -vx
                vy = vy if error_y > 0 else -vy

                # Convert the speed in the world coordinates to the robot's speed
                vxr = vx * np.cos(self.xEstFull[2]) + vy * np.sin(self.xEstFull[2])
                vyr = - vx * np.sin(self.xEstFull[2]) + vy * np.cos(self.xEstFull[2])                

                print('move_x:', vxr)
                print('move_y:', vyr)                
                self.move_straight(vxr, vyr)
                self.prev_time = time.time()
                time.sleep(0.2)
                # Update linear speed (requires calibration)
                self.vx = vxr
                self.vy = vyr
                self.prev_error_x = error_x
                self.prev_error_y = error_y

if __name__ == "__main__":
    try:
        if os.path.exists('/root/rb5_ws/src/rb5_ros/rb5_tracking/src/robot_pose_traj.txt'):
            os.remove('/root/rb5_ws/src/rb5_ros/rb5_tracking/src/robot_pose_traj.txt')

        if os.path.exists('/root/rb5_ws/src/rb5_ros/rb5_tracking/src/robot_error_target.txt'):
            os.remove('/root/rb5_ws/src/rb5_ros/rb5_tracking/src/robot_error_target.txt')

        rospy.init_node('robot_control', anonymous=True)


        robot = Robot()
        explore_points = np.array([[0.3, 0.3, 0.0],
                                   [1.3, 0.3, math.pi/2],
                                   [1.3, 1.3, math.pi],
                                   [0.0, 1.3, -math.pi/2],
                                   [0.0, 0.0, 0.0]])
        
        for waypoint in explore_points:
            robot.move_to(waypoint[0], waypoint[1], waypoint[2])
            time.sleep(0.05)
        robot.stop()

        with open(r"path.pkl", "rb") as file:
            waypoints = pickle.load(file)

        for waypoint in waypoints:
            robot.move_to(waypoint[0], waypoint[1], waypoint[2])
            time.sleep(0.05)
        robot.stop()

    except Exception as e:
        print(e)