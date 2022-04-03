from src.models import models
from src.plot import plot
import pandas as pd
import numpy as np


class runSOC(models, plot):
    def __init__(self):
        self.dt = 0.01
        self.N = 5000

        self.x0 = np.array([[0.0],                      # x position    [m]
                            [0.0],                      # y position    [m]
                            [0.0000001],                # yaw           [rad]
                            [0.0000001],                # velocity      [m/s]
                            [0.0000001]])               # yaw rate      [rad/s]

        self.p0 = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0]])

        self.zNoise = np.array([[1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0]])

        self.q = np.array([[1e-6, 0.0,       0.0,                0.0, 0.0],
                           [0.0, 1e-6,       0.0,                0.0, 0.0],
                           [0.0, 0.0,        1e-6,               0.0, 0.0],
                           [0.0, 0.0,        0.0,                1e-4, 0.0],
                           [0.0, 0.0,        0.0,                0.0, 1e-4]])

        self.hx = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0]])

        self.r = np.array([[0.015, 0.0, 0.0],
                           [0.0, 0.010, 0.0],
                           [0.0, 0.0, 0.01]])**2

        self.xEstCat = np.array(
            [self.x0[0, 0], self.x0[1, 0], self.x0[2, 0], self.x0[3, 0], self.x0[4, 0]])
        self.zCat = np.array([self.x0[0, 0], self.x0[1, 0], self.x0[2, 0]])
        self.velCat = np.array([self.x0[2, 0]])
        self.EstVelCat = np.array([self.x0[2, 0]])

    def flags(self):
        self.showFinal = 1
        self.showAnimation = 0
    
    def importData(self):
        self.data = pd.read('551_LA92.csv')

    def genMeasurement(self, i):
        x = float(self.data['XX'][i+1])
        y = float(self.data['YY'][i+1])
        vel = float(self.data['tv_velocity'][i+1])
        wZsens = float(self.data['yawRate'][i+1])
        gz = np.array([[x], [y], [wZsens]])
        # z = gz + z_noise @ np.random.randn(4, 1)
        return gz.astype(float), float(vel)

    def mainloop(self):
        for i in range(self.N):
            # x_true, p_true = extended_prediction(x_true, p_true)
            z, vel = self.genMeasurement(i)
            if i == (self.N - 1) and self.showFinal == 1:
                show_final_flag = 1
            else:
                show_final_flag = 0
            # x_true_cat = np.vstack((x_true_cat, np.transpose(x_true[0:2])))
            z_cat = np.vstack((z_cat, np.transpose(z[0:3])))
            vel_cat = np.vstack((vel_cat, vel))
            x_est_cat = np.vstack((x_est_cat, np.transpose(x_est[0:5])))
            est_vel_x = (x_est_cat[i+1, 0] - x_est_cat[i, 0])/self.dt
            est_vel_y = (x_est_cat[i+1, 1] - x_est_cat[i, 1])/self.dt
            est_vel = np.sqrt(est_vel_x**2 + est_vel_y**2)
            est_vel_cat = np.vstack([est_vel_cat, est_vel])
            self.postpross(i, x_est, p_est, x_est_cat, z,
                    z_cat, vel_cat, est_vel_cat, self.showAnimation, self.show_ellipse, show_final_flag)
            x_est, p_est = self.cubatureKalmanFilter(x_est, p_est, z)