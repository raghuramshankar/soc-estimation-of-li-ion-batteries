import numpy as np
from scipy.linalg import sqrtm
import math


class models:
    def f(self, x):
        x[0] = x[0] + (x[3]/x[4]) * (np.sin(x[4] * dt + x[2]) - np.sin(x[2]))
        x[1] = x[1] + (x[3]/x[4]) * (- np.cos(x[4] * dt + x[2]) + np.cos(x[2]))
        x[2] = x[2] + x[4] * dt
        x[3] = x[3] + x[3] * dt
        x[4] = x[4]
        x.reshape((5, 1))
        return x.astype(float)

    def h(self, hx, x):
        x = hx @ x
        x.reshape((3, 1))
        return x

    def sigma(x, p):
        n = np.shape(x)[0]
        SP = np.zeros((n, 2*n))
        W = np.zeros((1, 2*n))
        for i in range(n):
            SD = sqrtm(p)
            SP[:, i] = (x + (math.sqrt(n) * SD[:, i]
                             ).reshape((n, 1))).flatten()
            SP[:, i+n] = (x - (math.sqrt(n) * SD[:, i]
                               ).reshape((n, 1))).flatten()
            W[:, i] = 1/(2*n)
            W[:, i+n] = W[:, i]
        return SP.astype(float), W.astype(float)

    def cubaturePrediction(self, xPred, pPred):
        n = np.shape(xPred)[0]
        [SP, W] = self.sigma(xPred, pPred)
        xPred = np.zeros((n, 1))
        pPred = q
        for i in range(2*n):
            xPred = xPred + (f(SP[:, i]).reshape((n, 1)) * W[0, i])
        for i in range(2*n):
            p_step = (f(SP[:, i]).reshape((n, 1)) - xPred)
            pPred = pPred + (p_step @ np.transpose(p_step) * W[0, i])
        return xPred.astype(float), pPred.astype(float)

    def cubatureUpdate(self, xPred, pPred, z):
        n = np.shape(xPred)[0]
        m = np.shape(z)[0]
        [SP, W] = self.sigma(xPred, pPred)
        y_k = np.zeros((m, 1))
        P_xy = np.zeros((n, m))
        s = r
        for i in range(2*n):
            y_k = y_k + (h(SP[:, i]).reshape((m, 1)) * W[0, i])
        for i in range(2*n):
            p_step = (h(SP[:, i]).reshape((m, 1)) - y_k)
            P_xy = P_xy + ((SP[:, i]).reshape((n, 1)) -
                           xPred) @ np.transpose(p_step) * W[0, i]
            s = s + p_step @ np.transpose(p_step) * W[0, i]
        xPred = xPred + P_xy @ np.linalg.pinv(s) @ (z - y_k)
        pPred = pPred - P_xy @ np.linalg.pinv(s) @ np.transpose(P_xy)
        return xPred, pPred

    def cubature_kalman_filter(xEst, pEst, z):
        xPred, pPred = self.cubaturePrediction(xEst, pEst)
        # return xPred.astype(float), pPred.astype(float)
        x_upd, p_upd = self.cubatureUpdate(xPred, pPred, z)
        return x_upd.astype(float), p_upd.astype(float)