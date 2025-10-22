public class RobotKinematics {
    private double l1;
    private double l2;

    public RobotKinematics(double l1, double l2) {
        this.l1 = l1;
        this.l2 = l2;
    }

    public double[] forwardKinematics(double theta1, double theta2) {
        double x = l1 * Math.cos(theta1) + l2 * Math.cos(theta1 + theta2);
        double y = l1 * Math.sin(theta1) + l2 * Math.sin(theta1 + theta2);
        return new double[]{x, y};
    }

    public double[] inverseKinematics(double x, double y) throws IllegalArgumentException {
        double r = Math.sqrt(x*x + y*y);
        if (r > l1 + l2 || r < Math.abs(l1 - l2)) {
            throw new IllegalArgumentException("Target position out of reach");
        }

        double theta2 = Math.acos((x*x + y*y - l1*l1 - l2*l2) / (2 * l1 * l2));
        double alpha = Math.atan2(y, x);
        double beta = Math.acos((l1*l1 + r*r - l2*l2) / (2 * l1 * r));
        double theta1 = alpha - beta;

        return new double[]{theta1, theta2};
    }

    private double toDegrees(double radians) {
        return radians * (180 / Math.PI);
    }
}