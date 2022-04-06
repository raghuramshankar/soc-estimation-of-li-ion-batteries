import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import math
import numpy as np

class plot:
    def plotAnimation(self, i):
        if i == 0:
            plt.plot(self.time[0], self.z, 'g+')
            plt.plot(self.time[0], self.xEstCat[1], '.b')
        else:
            plt.plot(self.time[0], self.z, 'g+')
            plt.plot(self.time[0:i+1], self.xEstCat[0:, 1], 'b')
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

    def plotFinal(self):
        fig = plt.figure()
        f = fig.add_subplot(111)
        # f.plot(x_true_cat[0:, 0], x_true_cat[0:, 1], 'r', label='True Position')
        f.plot(self.time[0:self.N+1], self.xEstCat[0:, 1]*100, 'b', label='Estimated SOC')
        # f.plot(zCat[0:, 0], zCat[0:, 1], '+g', label='Noisy Measurements')
        f.set_xlabel('Time [s]')
        f.set_ylabel('SOC [%]')
        f.set_title('Cubature Kalman Filter - SOC Estimation')
        f.legend(loc='upper right', shadow=True, fontsize='large')
        plt.grid(True)
        plt.show()

    def postpross(self, i, xEst, pEst, show_final_flag):
        if self.showAnimation == 1:
            self.plotAnimation(i)
            if self.showEllipse == 1:
                self.plotEllipse(xEst[0:2], p_est)
        if show_final_flag == 1:
            # plot_final_3(xEstCat, zCat, velCat, estVelCat, i)
            self.plotFinal()
