import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class extendedKF():
    def __init__(self, N):
        '''number of iterations'''
        self.N = N

        '''process noise covariance'''
        self.sigmaW = 1

        '''sensor noise covariance'''
        self.sigmaV = 1

        '''true system intial state'''
        self.xTrue = 2
        '''initial estimate'''
        self.xHat = 20
        '''inital covariance'''
        self.sigmaX = 1
        '''inital driving input'''
        self.u = 0

        '''reserve storage for variables'''
        self.xStore = np.zeros((np.size(self.xTrue), self.N+1))
        self.xStore[:, 0] = self.xTrue
        self.xHatStore = np.zeros((np.size(self.xHat), self.N))
        self.sigmaXStore = np.zeros((np.size(self.xHat)**2, self.N))

    def genInputMeasurement(self, k):
        try:
            w = np.transpose(np.linalg.cholesky(self.sigmaW)) * \
                np.random.randn(np.size(self.xTrue))
            v = self.sigmaV * np.random.randn(np.size(self.xTrue))
        except:
            w = self.sigmaW * np.random.randn(np.size(self.xTrue))
            v = self.sigmaV * np.random.randn(np.size(self.xTrue))

        '''y_k+1 = x_k^3 + v_k'''
        self.yTrue = self.xTrue**3 + v
        '''x_k+1 = sqrt(x_k + 5) + w_k'''
        self.xTrue = np.sqrt(self.xTrue + 5) + w

    def iterKF(self):
        for k in range(self.N):
            '''EKF step 0: compute Ahat, Bhat'''
            '''x_k+1 = sqrt(5 + x_k) + w_k'''
            '''Ahat = jac(f, x_k)'''
            '''Bhat = jac(f, w_k)'''
            Ahat = 0.5/np.sqrt(5 + self.xHat)
            Bhat = 1

            '''EKF step 1a: state estimate time update'''
            self.xHat = np.sqrt(self.xHat + 5)

            '''EKF step 1b: error covariance time update'''
            self.sigmaX = Ahat * self.sigmaX * \
                np.transpose(Ahat) + Bhat * self.sigmaW * np.transpose(Bhat)

            '''generate input and measurement'''
            self.genInputMeasurement(k)

            '''EKF step 1c: estimate system output'''
            '''Chat = jac(h, x_k)'''
            '''Dhat = jac(h, v_k)'''
            '''y_k+1 = x_k^3'''
            Chat = 3 * self.xHat**2
            Dhat = 1
            self.yHat = self.xHat**3

            '''EKF step 2a: compute kalman gain'''
            sigmaY = Chat * self.sigmaX * \
                np.transpose(Chat) + Dhat * \
                self.sigmaV * np.transpose(Dhat)
            L = self.sigmaX * np.transpose(Chat)/sigmaY

            '''EKF step 2b: state estimate measurement update'''
            self.xHat = self.xHat + L * (self.yTrue - self.yHat)

            '''EKF step 2c: error covariance measurement update'''
            self.sigmaX = self.sigmaX - L * sigmaY * np.transpose(L)

            '''make EKF robust'''
            '''make sure we do not get negative sqrt'''
            self.xHat = max(self.xHat, -5)
            '''make sure we keep covariance positive definite'''
            # D, S, V = np.linalg.svd(self.sigmaX)
            # HH = V * S * np.transpose(V)
            # self.sigmaX = (
            #     self.sigmaX + np.transpose(self.sigmaX) + HH + np.transpose(HH))/4

            '''store information'''
            self.xStore[:, k+1] = self.xTrue
            self.xHatStore[:, k] = self.xHat
            self.sigmaXStore[:, k] = self.sigmaX

    def postpross(self):
        fig = plt.figure()
        f = fig.add_subplot(111)
        f.plot(range(N), self.xStore[0, 1:], 'k-', label='True')
        f.plot(range(N), self.xHatStore[0, :], 'b', label='Estimate')
        f.plot(range(N), self.xHatStore[0, :] + np.sqrt(3) *
               self.sigmaXStore[0, :], 'g--', label='Upper bound')
        f.plot(range(N), self.xHatStore[0, :] - np.sqrt(3) *
               self.sigmaXStore[0, :], 'g--', label='Lower bound')
        f.set_xlabel('Iteration')
        f.set_ylabel('State')
        f.set_title('Extended kalman filter generic model')
        f.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    N = 200
    kalmanObj = extendedKF(N)
    kalmanObj.iterKF()
    kalmanObj.postpross()
