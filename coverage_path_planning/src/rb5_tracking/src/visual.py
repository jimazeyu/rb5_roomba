#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse
from std_msgs.msg import Float64MultiArray

def XEst_callback(msg):

    global xEstFull
    # Get the dimensions
    rows = msg.layout.dim[0].size
    cols = msg.layout.dim[1].size

    # Convert the data back to a numpy matrix
    xEstFull = np.array(msg.data).reshape((rows, cols))

def PEst_callback(msg):

    global PEstFull
    # Get the dimensions
    rows = msg.layout.dim[0].size
    cols = msg.layout.dim[1].size

    # Convert the data back to a numpy matrix
    PEstFull = np.array(msg.data).reshape((rows, cols))

def z_callback(msg):

    global z
    # Get the dimensions
    rows = msg.layout.dim[0].size
    cols = msg.layout.dim[1].size

    # Convert the data back to a numpy matrix
    z = np.array(msg.data).reshape((rows, cols))

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
                  width=a*10, height=b*10,
                  angle=np.rad2deg(angle),
                  color=color, fill=False)
    return ell



def update(frame):
    global cov_ellipse, true_landmarks, seen_landmarks, observed_landmarks, cov_ellipse_landmarks
    xEstFull = rospy.wait_for_message("xEstFull", Float64MultiArray)
    xEstFull = np.array(xEstFull.data).reshape((xEstFull.layout.dim[0].size, xEstFull.layout.dim[1].size))
    PEstFull = rospy.wait_for_message("PEstFull", Float64MultiArray)
    PEstFull = np.array(PEstFull.data).reshape((PEstFull.layout.dim[0].size, PEstFull.layout.dim[1].size))
    z = rospy.wait_for_message("Landmarks", Float64MultiArray)
    z = np.array(z.data).reshape((z.layout.dim[0].size, z.layout.dim[1].size))

    if xEstFull is not None:
        print(xEstFull)
        LANDMARKS = (1.5/20) * np.array([[10.0, 5.0],
                                            [5.0, 10.0],
                                            [-5.0, 10.0],
                                            [-10.0, 5.0],
                                            [10.0, -5.0],
                                            [5.0, -10.0],
                                            [-5.0, -10.0],
                                            [-10.0, -5.0]])

        
        est_line.set_data(xEstFull[0][0], xEstFull[0][1])
        
        if observed_landmarks:
            for landmark in observed_landmarks:
                landmark.remove()
        observed_landmarks = []
        for i in range(3, len(xEstFull[0]), 2):
            lx = xEstFull[0][i]
            ly = xEstFull[0][i+1]
            observed_landmarks.append(ax.add_patch(plt.Circle((lx, ly), 0.05, color='r', fill=False)))
        if true_landmarks:
            for landmark in true_landmarks:
                landmark.remove()
        true_landmarks = []
        for landmark in LANDMARKS:
            true_landmarks.append(ax.add_patch(plt.Circle(landmark, 0.05, color='g', fill=True)))
        if seen_landmarks:
            for landmark in seen_landmarks:
                landmark.remove()
        seen_landmarks = []
        for landmark in z:
            landmark_id = int(landmark[2])
            landmark_true = LANDMARKS[landmark_id]
            seen_landmarks.append(ax.add_patch(plt.Circle(landmark_true, 0.05, color='b', fill=True)))

        if cov_ellipse:
            cov_ellipse.remove()
        cov_ellipse = plot_covariance_ellipse(xEstFull[0][0:3], PEstFull[0:3, 0:3])
        ax.add_patch(cov_ellipse)
        if cov_ellipse_landmarks:
            for ellipse in cov_ellipse_landmarks:
                ellipse.remove()
        cov_ellipse_landmarks = []
        for i in range(3, len(xEstFull[0]), 2):
            lx = xEstFull[0][i]
            ly = xEstFull[0][i+1]
            cov_ellipse_landmarks.append(plot_covariance_ellipse(xEstFull[0][i:i+2], PEstFull[i:i+2, i:i+2],'blue'))
            ax.add_patch(cov_ellipse_landmarks[-1])

        arrow_length = 0.1
        est_arrow = ax.quiver(xEstFull[0][0], xEstFull[0][1], arrow_length*np.cos(xEstFull[0][2]), arrow_length*np.sin(xEstFull[0][2]), color='r', scale=15)
    
        return [est_line, cov_ellipse] + true_landmarks + seen_landmarks + observed_landmarks + [est_arrow] +  cov_ellipse_landmarks


if __name__ == "__main__":
    x0 = -5.0 * 1.5 / 20
    y0 = -5.0 * 1.5 / 20

    # xEstFull = None
    # PEstFull = None
    # z = None

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

    rospy.init_node("visual")

    # Initialize figure.
    fig, ax = plt.subplots()
    plt.grid(True)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')

    # Draw placeholders for the true state and the estimated state.
    est_line, = ax.plot(x0, y0, 'ro', label='Estimated Position')

    # add legend
    ax.legend()

    # rospy.Subscriber('xEstFull', Float64MultiArray, XEst_callback)
    # rospy.Subscriber('PEstFull', Float64MultiArray, PEst_callback)
    # rospy.Subscriber('Landmarks', Float64MultiArray, z_callback)

    # if xEstFull is not None:
    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 20, 0.1), blit=True, interval=10)

    plt.show()

    rospy.spin()