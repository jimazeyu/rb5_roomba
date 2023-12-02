#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import time
import matplotlib.pyplot as plt
import numpy as np
from mpi_control import MegaPiController
import rospy
from std_msgs.msg import String


# Robot Simulator (Omnidirectional)
class RobotSimulator:
    def __init__(self, x=0, y=0, theta=0, dt=0.05, 
                k_rho=2.0, k_alpha=1.0, k_beta=1.0, 
                v_max=0.30, omega_max=40 * math.pi / 180, 
                v_min=0.10, omega_min=0 * math.pi / 180,
                target_range_x=0.20, target_range_y=0.20, target_range_theta=0.3,
                l=0.1354, r=0.0305, calibration_parameters=100/14.1124):
        # time step
        self.dt = dt
        # pose
        self.x = x
        self.y = y
        self.theta = theta
        # velocity(world frame)
        self.vx = 0
        self.vy = 0
        self.omega = 0
        # control parameters
        # k_alpha equals to k_beta because of the omnidirection driver
        self.k_rho = k_rho
        self.k_alpha = k_alpha
        self.k_beta = k_beta
        # velocity limit
        self.v_max = v_max
        self.omega_max = omega_max
        self.v_min = v_min
        self.omega_min = omega_min
        # parameters of the robot
        self.l = l # l_x + l_y
        self.r = r # wheel radius
        self.A = np.array([[1, -1, -1*l],[1, 1, l],[1, 1, -1*l],[1, -1, l]]) / r # matrix for inverse kinematics
        self.calibration_parameters = calibration_parameters # parameters map real wheel velocities to input control signal
        # range parameters to demetermine if the robot arrives the target pose
        self.target_range_x = target_range_x
        self.target_range_y = target_range_y
        self.target_range_theta = target_range_theta


    def move(self, vx, vy, omega):
        dt = self.dt
        # Convert velocity to robot coordinate system
        v = np.array([vx, vy])
        R = np.array([[math.cos(self.theta), -math.sin(self.theta)],
                        [math.sin(self.theta), math.cos(self.theta)]])
        R = np.linalg.inv(R)
        v_robot = np.dot(R, v)

        # update velocities
        self.vx = v_robot[0]
        self.vy = v_robot[1]
        self.omega = omega

        # # update pose
        # self.x += vx * dt
        # self.y += vy * dt
        # self.theta += omega * dt

        # # Normalize theta to be within [-2π, 2π] when it exceeds 2π or goes below -2π
        # if self.theta > 2 * math.pi:
        #     self.theta -= 2 * math.pi
        # elif self.theta < -2 * math.pi:
        #     self.theta += 2 * math.pi

        # Use a smaller dt to simulate the integration of the real pose update
        division = 100
        for _ in range(division):
            # update position
            R = np.array([[math.cos(self.theta), -math.sin(self.theta)],
                            [math.sin(self.theta), math.cos(self.theta)]])
            v_robot_real = np.dot(R, v_robot)
            self.x += v_robot_real[0] * dt / division
            self.y += v_robot_real[1] * dt / division

            # uodate direction
            self.theta += self.omega * dt / division
            # Normalize theta to be within [-2π, 2π] when it exceeds 2π or goes below -2π
            if self.theta > 2 * math.pi:
                self.theta -= 2 * math.pi
            elif self.theta < -2 * math.pi:
                self.theta += 2 * math.pi


    def move_to(self, goal_x, goal_y, goal_theta):
        error_x = goal_x - self.x
        error_y = goal_y - self.y
        direction = math.atan2(error_y, error_x)

        # The angle between the current orientation and the straight line direction to the target position
        alpha = direction - self.theta
        # The angle between the straight line direction to the target position and the target orientation
        beta = goal_theta - direction
        # Distance to the target position
        rho = math.sqrt(error_x ** 2 + error_y ** 2)

        # Calculate control input
        v = self.k_rho * rho
        omega = self.k_alpha * alpha + self.k_beta * beta
        # Limit speed
        if abs(v) > self.v_max:
            v = self.v_max if v > 0 else -self.v_max
        elif abs(v) < self.v_min:
            v = self.v_min if v > 0 else -self.v_min
        if abs(omega) > self.omega_max:
            omega = self.omega_max if omega > 0 else -self.omega_max
        elif abs(omega) < self.omega_min:
            omega = self.omega_min if omega > 0 else -self.omega_min

        vx = v * math.cos(direction)
        vy = v * math.sin(direction)

        # move the robot
        self.move(vx, vy, omega)

    def get_pose(self):
        return self.x, self.y, self.theta

    def get_velocity(self):
        return self.vx, self.vy, self.omega

    def _visualize_simulation(self, ax, robot, goal_x, goal_y, goal_theta):
        # move the robot in simulation
        while True:
            # Predict the robot's pose after dt time and obtain the control input
            robot.move_to(goal_x, goal_y, goal_theta)
            x, y, theta = robot.get_pose()
            rospy.loginfo('x: %.2f, y: %.2f, theta: %.2f' % (x, y, theta))
            vx, vy, omega = robot.get_velocity()
            rospy.loginfo('vx: %.2f, vy: %.2f, omega: %.2f' % (vx, vy, omega))

            # plot the pose of the robot
            pose_arrow = ax.arrow(x, y, 0.1 * math.cos(theta), 0.1 * math.sin(theta), head_width=0.2, head_length=0.2, fc='r', ec='r')
            # plot the trajectory
            ax.plot(x, y, 'r.')

            plt.pause(0.001)
            pose_arrow.remove()

            # Arriving at the target position
            if abs(x - goal_x) < self.target_range_x and abs(y - goal_y) < self.target_range_y and abs(theta - goal_theta) < self.target_range_theta:
                time.sleep(1)
                break
            time.sleep(self.dt)

    def visualize_simulation(self, points_list=[[-1, 0, 0], [-1, 1, 1.57], [-2, 1, 0], [-2, 2, -1.57], [-1, 1, -0.78], [0, 0, 0]]):
        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        # Visualize the starting point and the target point
        ax.scatter(0, 0, marker='*', s=200, c='b')
        for point in points_list:
            goal_x, goal_y, goal_theta = point
            ax.scatter(goal_x, goal_y, marker='*', s=200, c='b')

        for point in points_list:
            goal_x, goal_y, goal_theta = point
            self._visualize_simulation(ax, robot, goal_x, goal_y, goal_theta)    


    def robot_follow_point(self, mpi_ctrl, goal_x, goal_y, goal_theta):
        # move the robot
        while True:
            # Predict the robot's pose after dt time and obtain the control input
            self.move_to(goal_x, goal_y, goal_theta)
            x, y, theta = self.get_pose()
            rospy.loginfo('x: %.2f, y: %.2f, theta: %.2f' % (x, y, theta))
            vx, vy, omega = self.get_velocity()
            rospy.loginfo('vx: %.2f, vy: %.2f, omega: %.2f' % (vx, vy, omega))

            Omega = np.dot(self.A, np.array([vx, vy, omega])) * self.calibration_parameters
            rospy.loginfo('Omega (for wheels): %.2f, %.2f, %.2f, %.2f' % (-Omega[2], Omega[1], -Omega[0], Omega[3]))

            mpi_ctrl.setFourMotors(-Omega[2], Omega[1], -Omega[0], Omega[3]) # -bl, fr, -fl, br

            # Determine whether the target pose has been reached
            if abs(x - goal_x) < self.target_range_x and abs(y - goal_y) < self.target_range_y and abs(theta - goal_theta) < self.target_range_theta:
                # stop robot
                # time.sleep(1)
                break
            time.sleep(self.dt)

def robot_main():
    rospy.init_node('mpi_openloop_controller')
    # Instantiate the robot
    robot = RobotSimulator()
    mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
    
    k = 0.8 # scale parameter
    points_list = np.array([[-0.5, 0, 0], [-0.5, 0.5, 1.57], [-1, 0.5, 0], [-1, 1, -1.57], [-0.5, 0.5, -0.78], [0, 0, 0]])
    # points_list = np.array([[-0.5, 0, 0], [-0.5, 0, 1.57], [-0.5, 0.5, 1.57], [-0.5, 0.5, 0], [-1, 0.5, 0], [-1, 0.5, -1.57], [-1, 1, -1.57], [-1, 1, -0.78], [-0.5, 0.5, -0.78], [-0.5, 0.5, 0], [0, 0, 0]])
    points_list[:,0:2] = points_list[:,0:2] * k

    # robot.visualize_simulation(self, points_list=points_list)
    while not rospy.is_shutdown():
        for point in points_list:
            goal_x, goal_y, goal_theta = point
            robot.robot_follow_point(mpi_ctrl, goal_x, goal_y, goal_theta)
            mpi_ctrl.carStop()
            time.sleep(1)


# main
if __name__ == '__main__':
    robot_main()