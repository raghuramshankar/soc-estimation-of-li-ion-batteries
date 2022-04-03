import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import math
import numpy as np

class plot:
    def plotAnimation(self, i, xEstCat, z):
        if i == 0:
            plt.plot(z[0], z[1], 'g+')
            plt.plot(xEstCat[0], xEstCat[1], '.b')
        else:
            plt.plot(z[0], z[1], 'g+')
            plt.plot(xEstCat[0:, 0], xEstCat[0:, 1], 'b')
        plt.grid(True)
        plt.pause(0.001)

    def plotEllipse(self, xEst, p_est):
        phi = np.linspace(0, 2 * math.pi, 100)
        p_ellipse = np.array(
            [[p_est[0, 0], p_est[0, 1]], [p_est[1, 0], p_est[1, 1]]])
        x0 = 3 * sqrtm(p_ellipse)
        xy_1 = np.array([])
        xy_2 = np.array([])
        for i in range(100):
            arr = np.array([[math.sin(phi[i])], [math.cos(phi[i])]])
            arr = x0 @ arr
            xy_1 = np.hstack([xy_1, arr[0]])
            xy_2 = np.hstack([xy_2, arr[1]])
        plt.plot(xy_1 + xEst[0], xy_2 + xEst[1], 'r')
        plt.pause(0.00001)

    def plotFinal(self, xEstCat, zCat):
        fig = plt.figure()
        f = fig.add_subplot(111)
        # f.plot(x_true_cat[0:, 0], x_true_cat[0:, 1], 'r', label='True Position')
        f.plot(xEstCat[0:, 0], xEstCat[0:, 1],
               'b', label='Estimated Position')
        f.plot(zCat[0:, 0], zCat[0:, 1], '+g', label='Noisy Measurements')
        f.set_xlabel('x [m]')
        f.set_ylabel('y [m]')
        f.set_title('Cubature Kalman Filter - CTRV Model')
        f.legend(loc='upper right', shadow=True, fontsize='large')
        plt.grid(True)
        plt.show()

    def postpross(i, xEst, p_est, xEstCat, z, zCat, velCat, estVelCat, showAnimation, showEllipse, showFinalFlag):
        if self.showAnimation == 1:
            self.plotAnimation(i, xEstCat, z)
            if self.showEllipse == 1:
                self.plotEllipse(xEst[0:2], p_est)
        if self.showFinalFlag == 1:
            # plot_final_3(xEstCat, zCat, velCat, estVelCat, i)
            self.plotFinal(xEstCat, zCat)
