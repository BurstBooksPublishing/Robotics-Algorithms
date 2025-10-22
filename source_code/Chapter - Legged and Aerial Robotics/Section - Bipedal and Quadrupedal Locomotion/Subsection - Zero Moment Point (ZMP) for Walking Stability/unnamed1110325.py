import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

class ZeroMomentPoint:
    def __init__(self, mass, gravity=9.81):
        self.mass = mass
        self.gravity = gravity

    def compute_zmp(self, com_pos, com_acc, foot_positions):
        x_zmp = com_pos[0] - (com_pos[2] * com_acc[0]) / (self.gravity + com_acc[2])
        y_zmp = com_pos[1] - (com_pos[2] * com_acc[1]) / (self.gravity + com_acc[2])
        zmp = np.array([x_zmp, y_zmp])
        support_polygon = np.array(foot_positions)[:, :2]
        hull = ConvexHull(support_polygon)
        is_stable = self.point_in_hull(zmp, support_polygon[hull.vertices])
        return zmp, is_stable

    def point_in_hull(self, point, hull_points):
        from matplotlib.path import Path
        return Path(hull_points).contains_point(point)

    def plot_zmp(self, zmp, foot_positions):
        support_polygon = np.array(foot_positions)[:, :2]
        hull = ConvexHull(support_polygon)
        hull_pts = support_polygon[hull.vertices]
        
        plt.figure()
        plt.plot(support_polygon[:, 0], support_polygon[:, 1], 'bo', label='Foot Positions')
        plt.fill(hull_pts[:, 0], hull_pts[:, 1], 'b', alpha=0.2, label='Support Polygon')
        plt.plot(zmp[0], zmp[1], 'rx', markersize=10, label='ZMP')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Zero Moment Point (ZMP) Stability')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

if __name__ == "__main__":
    mass = 50.0
    com_pos = np.array([0.5, 0.1, 0.8])
    com_acc = np.array([0.1, 0.05, 0.0])
    foot_positions = [
        [0.4, 0.0, 0.0],
        [0.6, 0.0, 0.0],
        [0.4, 0.2, 0.0],
        [0.6, 0.2, 0.0]
    ]

    zmp_calculator = ZeroMomentPoint(mass)
    zmp, stable = zmp_calculator.compute_zmp(com_pos, com_acc, foot_positions)
    print("ZMP:", zmp)
    print("Is stable:", stable)
    zmp_calculator.plot_zmp(zmp, foot_positions)