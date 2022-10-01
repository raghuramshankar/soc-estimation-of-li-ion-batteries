import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class linearKF():
    def __init__(self, N):
        '''number of iterations'''
        self.N = N

        '''process noise covariance'''
        self.sigmaW = 0.001

        '''sensor noise covariance'''
        self.sigmaV = 0.001

        '''define cell parameters'''
        '''capacity [Ah]'''
        self.Q = 5
        '''equivalent series resistance [ohm]'''
        self.R0 = 0.1

        '''plant definition matrices'''
        '''z_k = z_k-1 - 1/(3600*Q)*I_k-1'''
        '''volt_k = 3.5 + 0.7*z_k-1 - R0*I_k-1'''
        self.A = 1
        self.B = -1/(3600 * self.Q)
        self.C = 0.7
        self.D = -self.R0

        '''true system intial state'''
        self.xTrue = 1.0
        '''kalman filter initial estimate'''
        self.xHat = 0
        '''kalman filter inital covariance'''
        self.sigmaX = -1
        '''inital driving input'''
        self.u = 0

        '''reserve storage for variables'''
        self.xStore = np.zeros((np.size(self.xTrue), self.N+1))
        self.xStore[:, 0] = self.xTrue
        self.xHatStore = np.zeros((np.size(self.xHat), self.N))
        self.sigmaXStore = np.zeros((np.size(self.xHat)**2, self.N))

    def genInputMeasurement(self, k):
        self.u = self.Q + 0.001 * np.random.randn(1)
        # self.u = self.Q
        try:
            w = np.transpose(np.linalg.cholesky(self.sigmaW)) * \
                np.random.randn(np.size(self.xTrue))
            v = np.transpose(np.linalg.cholesky(self.sigmaV)) * \
                np.random.randn(self.C * np.size(self.xTrue))
        except:
            w = self.sigmaW * np.random.randn(np.size(self.xTrue))
            v = self.sigmaV * np.random.randn(np.size(self.C * self.xTrue))
        self.yTrue = 3.5 + self.C * self.xTrue + self.D * self.u + w
        self.xTrue = self.A * self.xTrue + self.B * self.u + v

    def iterKF(self):
        for k in range(self.N):
            '''KF step 1a: state estimate time update'''
            self.xHat = self.A * self.xHat + self.B * self.u

            '''KF step 1b: error covariance time update'''
            self.sigmaX = self.A * self.sigmaX * \
                np.transpose(self.A) + self.sigmaW

            '''generate input and measurement'''
            self.genInputMeasurement(k)

            '''KF step 1c: estimate system output (after debias)'''
            self.yHat = self.C * self.xHat + self.D * self.u + 3.5

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
        f.plot(range(self.N), self.xStore[0, 1:], 'k--', label='True')
        f.plot(range(self.N), self.xHatStore[0, :], 'b', label='Estimate')
        f.plot(range(self.N), self.xHatStore[0, :] + np.sqrt(3) *
               self.sigmaXStore[0, :], 'g--', label='Upper bound')
        f.plot(range(self.N), self.xHatStore[0, :] - np.sqrt(3) *
               self.sigmaXStore[0, :], 'g--', label='Lower bound')
        f.set_xlabel('Iteration')
        f.set_ylabel('State of Charge')
        f.set_title('Linear kalman filter simple battery model')
        f.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    N = 3600
    kalmanObj = linearKF(N)
    kalmanObj.iterKF()
    kalmanObj.postpross()
