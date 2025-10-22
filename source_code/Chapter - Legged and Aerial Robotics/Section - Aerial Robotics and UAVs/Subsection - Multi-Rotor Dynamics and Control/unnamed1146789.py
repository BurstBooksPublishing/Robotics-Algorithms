import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class MultiRotorDynamics:
    def __init__(self, mass=1.0, inertia=np.diag([0.01, 0.01, 0.02])):
        self.mass = mass
        self.inertia = inertia
        self.gravity = 9.81

    def dynamics(self, state, t, forces, torques):
        x, y, z, phi, theta, psi, u, v, w, p, q, r = state

        R = np.array([
            [np.cos(theta)*np.cos(psi),
             np.cos(theta)*np.sin(psi),
             -np.sin(theta)],
            [np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi),
             np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi),
             np.sin(phi)*np.cos(theta)],
            [np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi),
             np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi),
             np.cos(phi)*np.cos(theta)]
        ])

        # Linear acceleration
        accel = np.array(forces) / self.mass
        accel[2] -= self.gravity

        # Angular acceleration
        omega = np.array([p, q, r])
        tau = np.array(torques)
        omega_dot = np.linalg.inv(self.inertia) @ (tau - np.cross(omega, self.inertia @ omega))

        # Kinematic equations
        pos_dot = R @ np.array([u, v, w])
        angles_dot = np.array([
            p + q*np.sin(phi)*np.tan(theta) + r*np.cos(phi)*np.tan(theta),
            q*np.cos(phi) - r*np.sin(phi),
            q*np.sin(phi)/np.cos(theta) + r*np.cos(phi)/np.cos(theta)
        ])

        return np.concatenate((pos_dot, angles_dot, accel, omega_dot))

    def simulate(self, initial_state, forces, torques, t):
        return odeint(self.dynamics, initial_state, t, args=(forces, torques))

if __name__ == "__main__":
    uav = MultiRotorDynamics()
    initial_state = np.zeros(12)
    forces = [0, 0, uav.mass * uav.gravity]
    torques = [0.01, 0.01, 0.005]
    t = np.linspace(0, 5, 500)

    result = uav.simulate(initial_state, forces, torques, t)

    plt.figure()
    plt.plot(t, result[:, 0], label='x')
    plt.plot(t, result[:, 1], label='y')
    plt.plot(t, result[:, 2], label='z')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Multi-Rotor UAV Position over Time')
    plt.legend()
    plt.grid(True)
    plt.show()