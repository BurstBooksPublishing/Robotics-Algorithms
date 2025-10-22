import numpy as np
from scipy.linalg import null_space

def compute_jacobian(robot_links, joint_angles):
    n = len(joint_angles)
    J = np.zeros((6, n))
    T = np.eye(4)

    for i in range(n):
        z_axis = T[:3, 2]
        p_ee = T[:3, 3]
        p_joint = T[:3, 3]

        J[:3, i] = np.cross(z_axis, (p_ee - p_joint))
        J[3:, i] = z_axis

        c, s = np.cos(joint_angles[i]), np.sin(joint_angles[i])
        T_i = np.array([
            [c, -s, 0, robot_links[i]*c],
            [s,  c, 0, robot_links[i]*s],
            [0,  0, 1,               0],
            [0,  0, 0,               1]
        ])
        T = T @ T_i

    return J