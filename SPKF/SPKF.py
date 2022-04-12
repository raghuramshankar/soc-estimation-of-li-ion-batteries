import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class linearKF():
    def __init__(self, N):
        '''number of iterations'''
        self.N = 200

        '''process noise covariance'''
        self.sigmaW = 1

        '''sensor noise covarciance'''
        self.sigmaV = 1

        '''plant definition matrices'''
        '''x_k = 1.x_k-1 + 1.u_k-1'''
        '''y_k = 1.x_k-1 + 0.u_k-1'''
        self.A = 1
        self.B = 1
        self.C = 1
        self.D = 0

        '''true system intial state'''
        self.xTrue = 100
        '''kalman filter initial estimate'''
        self.xHat = 0
        '''kalman filter inital covariance'''
        self.sigmaX = 0
        '''inital driving input'''
        self.u = 0

        '''reserve storage for variables'''
        self.xStore = np.zeros((np.size(self.xTrue), self.N+1))
        self.xStore[:, 0] = self.xTrue
        self.xHatStore = np.zeros((np.size(self.xHat), self.N))
        self.sigmaXStore = np.zeros((np.size(self.xHat)**2, self.N))

        self.flags()

    def flags(self):
        self.showFinal = 1
        self.showAnimation = 1
        self.showEllipse = 0

    def genInputMeasurement(self, k):
        self.u = 0.5 * np.random.randn(1) + np.cos(k/np.pi)
        try:
            w = np.transpose(np.linalg.cholesky(self.sigmaW)) * \
                np.random.randn(np.size(self.xTrue))
            v = self.sigmaV * np.random.randn(np.size(self.C * self.xTrue))
        except:
            w = self.sigmaW * np.random.randn(np.size(self.xTrue))
            v = self.sigmaV * np.random.randn(self.C * np.size(self.xTrue))
        self.yTrue = self.C * self.xTrue + self.D * self.u + v
        self.xTrue = self.A * self.xTrue + self.B * self.u + w

    def iterKF(self):
        for k in range(self.N):
            '''KF step 1a: state estimate time update'''
            self.xHat = self.A * self.xHat + self.B * self.u

            '''KF step 1b: error covariance time update'''
            self.sigmaX = self.A * self.sigmaX * \
                np.transpose(self.A) + self.sigmaW

            '''generate input and measurement'''
            self.genInputMeasurement(k)

            '''KF step 1c: estimate system output'''
            self.yHat = self.C * self.xHat + self.D * self.u

            '''KF step 2a: compute kalman gain'''
            sigmaY = self.C * self.sigmaX * np.transpose(self.C) + self.sigmaV
            L = self.sigmaX * np.transpose(self.C)/sigmaY

            '''KF step 2b: state estimate measurement update'''
            self.xHat = self.xHat + L * (self.yTrue - self.yHat)

            '''KF step 2c: error covariance measurement update'''
            self.sigmaX = self.sigmaX - L * sigmaY * np.transpose(L)

            '''store information'''
            self.xStore[:, k+1] = self.xTrue
            self.xHatStore[:, k] = self.xHat
            self.sigmaXStore[:, k] = self.sigmaX

    def postpross(self):
        fig = plt.figure()
        f = fig.add_subplot(111)
        f.plot(range(N), self.xStore[0, 1:], 'k+', label='True')
        f.plot(range(N), self.xHatStore[0, :], 'b', label='Estimate')
        f.plot(range(N), self.xHatStore[0, :] + np.sqrt(3) *
               self.sigmaXStore[0, :], 'g--', label='Upper bound')
        f.plot(range(N), self.xHatStore[0, :] - np.sqrt(3) *
               self.sigmaXStore[0, :], 'g--', label='Lower bound')
        f.set_xlabel('Iteration')
        f.set_ylabel('State')
        f.set_title('Linear kalman filter generic model')
        f.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    N = 200
    kalmanObj = linearKF(N)
    kalmanObj.iterKF()
    kalmanObj.postpross()
