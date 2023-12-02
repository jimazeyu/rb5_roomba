import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse

# Random seed is used for result reproducibility.
np.random.seed(0)

# 8 landmraks
# * *
#*   *
#*   *
# * *
LANDMARKS = 1.2 * np.array([[10.0, 5.0],
                        [5.0, 10.0],
                        [-5.0, 10.0],
                        [-10.0, 5.0],
                        [10.0, -5.0],
                        [5.0, -10.0],
                        [-5.0, -10.0],
                        [-10.0, -5.0]])


OBSERVATION_RANGE = 20.0  # observation range
OBSERVATION_ANGLE = np.deg2rad(60.0)  # observation angle

# observe
def observe(true_x, landmarks):
    observations = []
    for (i, landmark) in enumerate(landmarks):
        # fov
        if np.abs(true_x[2] - np.arctan2(landmark[1] - true_x[1], landmark[0] - true_x[0])) < OBSERVATION_ANGLE:
            # calculate the distance and the angle
            dx = landmark[0] - true_x[0]
            dy = landmark[1] - true_x[1]
            d = np.sqrt(dx**2 + dy**2) + np.random.randn()*Q[0, 0]
            angle = np.arctan2(dy, dx) - true_x[2] + np.random.randn()*Q[1, 1]
            # with the observation range
            if d <= OBSERVATION_RANGE:
                observations.append([d, angle, i])
    return observations

# def observe():
#     observations = []

#     bridge = CvBridge()

#     msg = rospy.wait_for_message('/camera_0', Image)

#     cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
#     gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

#     detector = apriltag.Detector()

#     results = detector.detect(gray_image)

#     landmarks = []
#     for result in results:
#         landmarks.append(result)
#     if len(landmarks) == 0:
#         return False
#     else:
#         camera_params = [446.632864, 446.429350, 631.849391, 357.598039]
#         for landmark in landmarks:
#             tag_pose = detector.detection_pose(landmark, camera_params)[0]
#             x = tag_pose[0][3]
#             y = tag_pose[1][3]
#             z = tag_pose[2][3]
#             x_2d, y_2d = z*18.0/100.0, -x*18.0/100.0
#             d = np.sqrt(x_2d**2 + y_2d**2)
#             angle = np.arctan2(y_2d, x_2d)
#             observations.append([d, angle, landmark.tag_id])
#     return observations



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

# Set the initial values of the true state and the estimated state.
true_x = np.array([0.0, -5.0, 0.0])
xEstFull = np.array([0.0, -5.0, 0.0])
PEstFull = np.diag([0.0, 0.0, 0.0])

# Set system noise
R = np.diag([0.1**2, 0.1**2, np.deg2rad(5.0)**2])
# Set observation noise
Q = np.diag([0.1**2, np.deg2rad(5.0)**2])

# Record the landmark ID of seen landmarks.
landmarksID = []

# Initialize figure.
fig, ax = plt.subplots()
plt.grid(True)
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
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

# for animation
def update(frame):
    global true_x, xEstFull, PEstFull, cov_ellipse, true_landmarks, seen_landmarks, observed_landmarks, cov_ellipse_landmarks

    # Simulate control inputs (move 0.1m in the x-direction, rotate 1° in the theta direction)
    u = np.array([0.1, 0.00, np.deg2rad(1.0)])
    # simulation the true states
    true_x[0] += u[0]*np.cos(true_x[2]) - u[1]*np.sin(true_x[2]) + np.random.randn()*R[0, 0]
    true_x[1] += u[0]*np.sin(true_x[2]) + u[1]*np.cos(true_x[2]) + np.random.randn()*R[1, 1]
    true_x[2] += u[2] + np.random.randn()*R[2, 2]
    # theta in -pi to pi
    true_x[2] = np.mod(true_x[2] + np.pi, 2*np.pi) - np.pi

    # get observation  
    z = observe(true_x, LANDMARKS)   
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
    xEstFull = xEstFull + Fx.T @ uR
    
    # update robot cov
    G = np.array([[0, 0, -u[0]*np.sin(theta) - u[1]*np.cos(theta)],
                     [0, 0, u[0]*np.cos(theta) - u[1]*np.sin(theta)],
                     [0, 0, 0]])
    G = Fx.T @ G @ Fx + np.eye(Fx.shape[1])
    PEstFull = G.T @ PEstFull @ G + Fx.T @ R @ Fx

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
        q = delta.T @ delta
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
        Hi = Hi @ Fxj
        # 6.K^i_t = Σt H^i_t^T (H^i_t Σt H^i_t^T + Rt)^-1
        K = PEstFull @ Hi.T @ np.linalg.inv(Hi @ PEstFull @ Hi.T + Q)
        # 7.μt = μt + K^i_t (z^i_t - z^i_t)
        xEstFull = xEstFull + K @ (z[iz][0:2] - z_hat)
        # 8.Σt = (I - K^i_t H^i_t) Σt
        PEstFull = (np.eye(PEstFull.shape[0]) - K @ Hi) @ PEstFull
    
    # update figure
    true_line.set_data(true_x[0], true_x[1])
    est_line.set_data(xEstFull[0], xEstFull[1])

    if observed_landmarks:
        for landmark in observed_landmarks:
            landmark.remove()
    observed_landmarks = []
    for i in range(3, len(xEstFull), 2):
        lx = xEstFull[i]
        ly = xEstFull[i+1]
        observed_landmarks.append(ax.add_patch(plt.Circle((lx, ly), 0.5, color='r', fill=False)))

    if true_landmarks:
        for landmark in true_landmarks:
            landmark.remove()
    true_landmarks = []
    for landmark in LANDMARKS:
        true_landmarks.append(ax.add_patch(plt.Circle(landmark, 0.5, color='g', fill=True)))

    if seen_landmarks:
        for landmark in seen_landmarks:
            landmark.remove()
    seen_landmarks = []
    for landmark in z:
        landmark_id = int(landmark[2])
        landmark_true = LANDMARKS[landmark_id]
        seen_landmarks.append(ax.add_patch(plt.Circle(landmark_true, 0.5, color='b', fill=True)))

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
    true_arrow = ax.quiver(true_x[0], true_x[1], arrow_length*np.cos(true_x[2]), arrow_length*np.sin(true_x[2]), color='g', scale=15)
    
    return true_line, est_line, cov_ellipse, *true_landmarks, *seen_landmarks, *observed_landmarks, est_arrow, true_arrow, *cov_ellipse_landmarks


ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 20, 0.1), blit=True, interval=10)

plt.show()