#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import rospy
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
from key_parser import get_key, save_terminal_settings, restore_terminal_settings

# Random seed is used for result reproducibility.
np.random.seed(0)

# 8 landmraks
# * *
#*   *
#*   *
# * *
LANDMARKS = 1.0 / 20.0 * 1.5 * np.array([[10.0, 5.0],
                        [5.0, 10.0],
                        [-5.0, 10.0],
                        [-10.0, 5.0],
                        [10.0, -5.0],
                        [5.0, -10.0],
                        [-5.0, -10.0],
                        [-10.0, -5.0]])


OBSERVATION_RANGE = 20.0  # 观测范围
OBSERVATION_ANGLE = np.deg2rad(60.0)  # 观测角度

# observe
# def observe(true_x, landmarks):
#     observations = []
#     for (i, landmark) in enumerate(landmarks):
#         # fov
#         if np.abs(true_x[2] - np.arctan2(landmark[1] - true_x[1], landmark[0] - true_x[0])) < OBSERVATION_ANGLE:
#             # calculate the distance and the angle
#             dx = landmark[0] - true_x[0]
#             dy = landmark[1] - true_x[1]
#             d = np.sqrt(dx**2 + dy**2) + np.random.randn()*Q[0, 0]
#             angle = np.arctan2(dy, dx) - true_x[2] + np.random.randn()*Q[1, 1]
#             # with the observation range
#             if d <= OBSERVATION_RANGE:
#                 observations.append([d, angle, i])
#     return observations

class Robot:
    def __init__(self, x=-5*(1/20), y=-5*(1/20), theta=0):
        self.mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
        self.l = 0.1354
        self.r = 0.0305
        self.A = np.array([[1, -1, -1*self.l],[1, 1, self.l],[1, 1, -1*self.l],[1, -1, self.l]]) / self.r
        self.num_step = 0
        self.x = x
        self.y = y
        self.theta = theta
        self.state = 0
        self.scale = 1 / 20
        self.target_point = np.array([[5.0 , -5.0, 0],
                                    [5.0 , -5.0, math.pi/2],
                                    [5.0, 5.0, math.pi/2],
                                    [5.0, 5.0, math.pi],
                                    [-5.0, 5.0, math.pi],
                                    [-5.0, 5.0, -math.pi/2],
                                    [-5.0, -5.0, -math.pi/2],
                                    [-5.0, -5.0, 0]])
        self.target_point[:,:2] = self.target_point[:,:2] * self.scale

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

    def step(self):
        
        if abs(self.theta - self.target_point[self.state][-1]) < 0.25 and \
            abs(self.x - self.target_point[self.state][0]) < 0.25 and \
                abs(self.y - self.target_point[self.state][1]) < 0.25:
            print("Arrived!")
            self.x, self.y, self.theta = self.target_point[self.state]
            self.state += 1
            if self.state == len(self.target_point):
                self.state = 0

        if abs(self.theta - self.target_point[self.state][-1]) >= 0.25:
            sleep_time = 0.2
            print("Rotate")
            omega = 1.2
            omega = -omega if self.theta - self.target_point[self.state][-1] > 0 else omega
            self.rotate(omega)
            time.sleep(sleep_time)
            self.stop()
            self.theta += 3.0 *omega * sleep_time
            self.theta = np.mod(self.theta + np.pi, 2*np.pi) - np.pi
            return np.array([0.0, 0.0, omega * sleep_time])
        else:
            sleep_time = 0.2
            print("Straight")
            vx = 0.15

            target_point_robot = np.dot(np.array([[np.cos(self.theta), np.sin(self.theta)], [-np.sin(self.theta), np.cos(self.theta)]]), self.target_point[:,:2].T).T

            vx = -vx if target_point_robot[self.state][0] < 0 else vx
            self.move_straight(vx, 0)
            time.sleep(sleep_time)
            self.x += 4.0 * vx * sleep_time * np.cos(self.theta)
            self.y += 4.0 * vx * sleep_time * np.sin(self.theta)
            return np.array([vx*sleep_time, 0.0, 0.0])
            

def observe():
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
    print(observations)
    return observations



def plot_covariance_ellipse(xEst, PEst, color='red'):
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    a = np.sqrt(eigval[bigind])
    b = np.sqrt(eigval[smallind])
    angle = np.arctan2(eigvec[bigind, 1], eigvec[bigind, 0])
    ell = Ellipse(xy=(xEst[0], xEst[1]),
                  width=a*1, height=b*1,
                  angle=np.rad2deg(angle),
                  color=color, fill=False)
    return ell



if __name__=="__main__":
    rospy.init_node('test_node', anonymous=True)
    robot = Robot()
    # Set the initial values of the true state and the estimated state.
    true_x = np.array([-5.0/20*1.5, -5.0/20*1.5, 0.0])
    xEstFull = np.array([-5.0/20*1.5, -5.0/20*1.5, 0.0])
    PEstFull = np.diag([0.0, 0.0, 0.0])

    # Set system noise
    R = np.diag([0.1**2, 0.1**2, np.deg2rad(20.0)**2])
    # Set observation noise
    Q = np.diag([0.1**2, np.deg2rad(5.0)**2])

    # Record the landmark ID of seen landmarks.
    landmarksID = []

    # Initialize figure.
    fig, ax = plt.subplots()
    plt.grid(True)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')

    # Draw placeholders for the true state and the estimated state.
    true_line, = ax.plot(true_x[0], true_x[1], 'bo', label='True Position')
    est_line, = ax.plot(xEstFull[0], xEstFull[1], 'ro', label='Estimated Position')

    # add legend
    ax.legend()

    # Initialize the robot covariance ellipses.
    cov_ellipse = None
    # Initialize the landmark covariance ellipses.
    cov_ellipse_landmarks = None
    # Initialize the true positions of landmarks
    true_landmarks = None
    # Initialize the true positions of landmarks which have been seen
    seen_landmarks = None
    # Initialize the true positions of landmarks which are being observed
    observed_landmarks = None

    delay = 10
    robot_pose_history = [[-5.0/20*1.5, -5.0/20*1.5]]
    robot_pose_dot, = ax.plot(robot_pose_history[0][0], robot_pose_history[0][1], 'ro', markersize=2)


    # for animation
    def update(frame):
        global true_x, xEstFull, PEstFull, cov_ellipse, true_landmarks, seen_landmarks, observed_landmarks, cov_ellipse_landmarks, robot_pose_history
        global robot
        global delay
        # u = None
        # if delay == 0:
        #     u = robot.step()
        # else:
        #     u = [0,0,0]
        #     delay -= 1

        settings = save_terminal_settings()
        key = get_key(settings, timeout=0.02)
        u = [0, 0, 0]

        # if delay <= 0 and abs(robot.theta - math.pi/2) > 0.2:
        #     omega = 1.2
        #     robot.rotate(omega)
        #     sleep_time=0.2
        #     time.sleep(sleep_time)
        #     robot.stop()
        #     u = [0, 0, omega * sleep_time]
        # delay-=1
        if key == 'w':
            vx = 0.35
            robot.move_straight(vx,0.0)
            sleep_time=0.2
            time.sleep(sleep_time)
            robot.stop()
            u = [sleep_time*vx, 0, 0]
        if key == 's':
            vx = -0.25
            robot.move_straight(vx,0.0)
            sleep_time=0.2
            time.sleep(sleep_time)
            robot.stop()
            u = [sleep_time*vx, 0, 0]
        elif key == 'a':
            omega = 1.2
            robot.rotate(omega)
            sleep_time=0.2
            time.sleep(sleep_time)
            robot.stop()
            u = [0, 0, omega * sleep_time]
        elif key == 'd':
            omega = -1.2
            robot.rotate(omega)
            sleep_time=0.2
            time.sleep(sleep_time)
            robot.stop()
            u = [0, 0, omega * sleep_time]
        elif key == 'q':
            robot.stop()

        z = observe()  
        print(z)
        # If a new landmark is detected, add it to the list of landmarks.
        for iz in range(len(z)):
            landmark_id = int(z[iz][2])
            list_id = -1
            # If it is a new landmark, add it to the landmark history.
            for id in landmarksID:
                if id == landmark_id:
                    list_id = id
                    break
            if list_id < 0:
                print("New Landmark observed, landmark_id = ", landmark_id)
                landmarksID.append(landmark_id)
                # Initialize the position of the landmark
                x1 = xEstFull[0] + z[iz][0]*np.cos(xEstFull[2] + z[iz][1])
                x2 = xEstFull[1] + z[iz][0]*np.sin(xEstFull[2] + z[iz][1])
                # extend xEstFull
                xEstFull = np.hstack((xEstFull, np.array([x1, x2])))
                # extend PEstFull
                PEstFullTemp = np.zeros((PEstFull.shape[0] + 2, PEstFull.shape[1] + 2))
                PEstFullTemp[:PEstFull.shape[0], :PEstFull.shape[1]] = PEstFull
                np.fill_diagonal(PEstFullTemp[-2:, -2:], 1e10)
                PEstFull = PEstFullTemp
                
    
        # prediction robot pose
        Fx = np.hstack((np.eye(3), np.zeros((3, 2*len(landmarksID)))))
        theta = xEstFull[2]
        uR = np.array([u[0]*np.cos(theta) - u[1]*np.sin(theta),
                        u[0]*np.sin(theta) + u[1]*np.cos(theta),
                        u[2]])
        xEstFull = xEstFull + np.dot(Fx.T, uR)
        
        # update robot cov
        G = np.array([[0, 0, -u[0]*np.sin(theta) - u[1]*np.cos(theta)],
                        [0, 0, u[0]*np.cos(theta) - u[1]*np.sin(theta)],
                        [0, 0, 0]])
        G = np.dot(np.dot(Fx.T, G), Fx) + np.eye(Fx.shape[1])
        PEstFull = np.dot(np.dot(G.T, PEstFull), G) + np.dot(np.dot(Fx.T, R), Fx)

        # theta in -pi to pi
        xEstFull[2] = np.mod(xEstFull[2] + np.pi, 2*np.pi) - np.pi


        # correction with observation
        for iz in range(len(z)):
            # observed landmark ID
            landmark_id = int(z[iz][2])
            # index of the landmark in landmarkXEst
            list_id = -1
            for i, id in enumerate(landmarksID):
                if id == landmark_id:
                    list_id = i
                    break
            # 1.δ = (δx δy)^T = (μj,x - μt,x, μj,y - μt,y)^T
            delta = np.array([xEstFull[2*list_id + 3] - xEstFull[0],
                                xEstFull[2*list_id + 4] - xEstFull[1]])
            # 2.q = δ^T δ
            q = np.dot(delta.T, delta)
            # 3.z^i_t = (atan2(δy, δx) - μt,θ)^T
            z_hat = np.array([np.sqrt(q),
                                np.arctan2(delta[1], delta[0]) - xEstFull[2]])
            # 角度修正（very important）
            z_hat[1] = np.mod(z_hat[1] + np.pi, 2*np.pi) - np.pi
            # 4.Fx,j
            Fxj = np.zeros((5, 3 + 2*len(landmarksID)))
            Fxj[0:3, 0:3] = np.eye(3)
            Fxj[3:5, 2*list_id + 3:2*list_id + 5] = np.eye(2)
            # 5.H^i_t = 1/q [-√qδx, -√qδy, 0, √qδx, √qδy; δy, -δx, -q, -δy, δx] * Fx,j
            Hi = 1/q * np.array([[-np.sqrt(q)*delta[0], -np.sqrt(q)*delta[1], 0, np.sqrt(q)*delta[0], np.sqrt(q)*delta[1]],
                                    [delta[1], -delta[0], -q, -delta[1], delta[0]]])
            Hi = np.dot(Hi, Fxj)
            # 6.K^i_t = Σt H^i_t^T (H^i_t Σt H^i_t^T + Rt)^-1
            K = np.dot(np.dot(PEstFull, Hi.T), np.linalg.inv(np.dot(np.dot(Hi, PEstFull), Hi.T) + Q))
            # 7.μt = μt + K^i_t (z^i_t - z^i_t)
            xEstFull = xEstFull + np.dot(K, (z[iz][0:2] - z_hat))
            # 8.Σt = (I - K^i_t H^i_t) Σt
            PEstFull = np.dot((np.eye(PEstFull.shape[0]) - np.dot(K, Hi)), PEstFull)
        
        robot.x = xEstFull[0]
        robot.y = xEstFull[1]
        robot.theta = xEstFull[2]


        # update figure
        est_line.set_data(xEstFull[0], xEstFull[1])

        if observed_landmarks:
            for landmark in observed_landmarks:
                landmark.remove()
        observed_landmarks = []
        for i in range(3, len(xEstFull), 2):
            lx = xEstFull[i]
            ly = xEstFull[i+1]
            observed_landmarks.append(ax.add_patch(plt.Circle((lx, ly), 0.05, color='r', fill=False)))

        if true_landmarks:
            for landmark in true_landmarks:
                landmark.remove()
        true_landmarks = []
        for landmark in LANDMARKS:
            true_landmarks.append(ax.add_patch(plt.Circle(landmark, 0.05, color='g', fill=True)))


        if cov_ellipse:
            cov_ellipse.remove()
        cov_ellipse = plot_covariance_ellipse(xEstFull[0:3], PEstFull[0:3, 0:3])
        ax.add_patch(cov_ellipse)
        if cov_ellipse_landmarks:
            for ellipse in cov_ellipse_landmarks:
                ellipse.remove()
        cov_ellipse_landmarks = []
        for i in range(3, len(xEstFull), 2):
            lx = xEstFull[i]
            ly = xEstFull[i+1]
            cov_ellipse_landmarks.append(plot_covariance_ellipse(xEstFull[i:i+2], PEstFull[i:i+2, i:i+2],'blue'))
            ax.add_patch(cov_ellipse_landmarks[-1])

        arrow_length = 1.0 
        est_arrow = ax.quiver(xEstFull[0], xEstFull[1], arrow_length*np.cos(xEstFull[2]), arrow_length*np.sin(xEstFull[2]), color='r', scale=15)
        # ax.add_patch(est_arrow)

        robot_pose_history.append([xEstFull[0], xEstFull[1]])
        robot_pose_history_np = np.array(robot_pose_history)
        robot_pose_dot.set_data(robot_pose_history_np[:, 0], robot_pose_history_np[:, 1])

        if os.path.exists('/root/rb5_ws/src/rb5_ros/rb5_tracking/src/xEstFull.txt'):
            os.remove('/root/rb5_ws/src/rb5_ros/rb5_tracking/src/xEstFull.txt')
        
        with open('/root/rb5_ws/src/rb5_ros/rb5_tracking/src/xEstFull.txt', 'a') as file:
            # Write the pose to the file in a comma-separated format
            for i in range(len(xEstFull)):
                file.write("%s, "%(xEstFull[i]))
            file.write("\n")
            # file.write("%s, %s, %s\n" % (xEstFull[0], xEstFull[1], xEstFull[2]))
        return [est_line, cov_ellipse] + observed_landmarks + [est_arrow] + cov_ellipse_landmarks + [robot_pose_dot]
        # return true_line, est_line, cov_ellipse, *true_landmarks, *seen_landmarks, *observed_landmarks, est_arrow, true_arrow, *cov_ellipse_landmarks

    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 200000, 0.1), blit=True, interval=10)

    ani.save('animation.mp4', writer='ffmpeg', fps=30)

    plt.show()