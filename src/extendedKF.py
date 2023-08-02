# %%
import pandas as pd
import numpy as np
import os
import sys


class extendedKFBatt:
    def __init__(self, N):
        """add parent path"""
        self.parentPath = os.getcwd()

        """number of iterations"""
        self.N = N

        """initial estimates"""
        """states: soc, iRC1, iRC2"""
        self.xHat = np.array([np.array([0.0]), np.array([0.0]), np.array([0.0])])

        """initial covariance"""
        self.sigmaX = np.diag([1e-5, 1e-5, 1e-5])

        """process noise covariance"""
        socVar = 1e1
        iRC1Var = 1e2
        iRC2Var = 1e1
        self.sigmaW = np.diag([socVar, iRC1Var, iRC2Var])

        """sensor noise covariance"""
        self.sigmaV = np.array([np.array([1e-3])])

    def runKF(self, filename):
        """import dataframe"""
        self.importdf(filename)
        self.importmodel()

        """limit timesteps"""
        self.N = min(self.N, len(self.df))

        """reserve storage for variables"""
        self.socStore = np.zeros((self.N, 1)).flatten()
        self.socStore[0] = self.xHat[0]
        self.Irc1Store = np.zeros((self.N, 1)).flatten()
        self.Irc1Store[0] = self.xHat[1]
        self.Irc2Store = np.zeros((self.N, 1)).flatten()
        self.Irc2Store[0] = self.xHat[2]
        self.socSigmaStore = np.zeros((self.N, 1)).flatten()
        self.socSigmaStore[0] = self.sigmaX[0, 0]
        self.Irc1SigmaStore = np.zeros((self.N, 1)).flatten()
        self.Irc1SigmaStore[0] = self.sigmaX[1, 1]
        self.Irc2SigmaStore = np.zeros((self.N, 1)).flatten()
        self.Irc2SigmaStore[0] = self.sigmaX[2, 2]
        self.IStore = np.zeros((self.N, 1)).flatten()
        self.VStore = np.zeros((self.N, 1)).flatten()
        self.yHatStore = np.zeros((self.N, 1)).flatten()
        self.yHatKFStore = np.zeros((self.N, 1)).flatten()
        self.VOCVStore = np.zeros((self.N, 1)).flatten()
        self.innovationStore = np.zeros((self.N, 1)).flatten()
        self.dOCVSOCStore = np.zeros((self.N, 1)).flatten()
        self.errorStore = np.zeros((self.N, 1)).flatten()
        self.storeDF = pd.DataFrame(
            {
                "SOC": self.socStore,
                "Irc1": self.Irc1Store,
                "Irc2": self.Irc2Store,
                "socSigma": self.socSigmaStore,
                "Irc1Sigma": self.Irc1SigmaStore,
                "Irc2Sigma": self.Irc2SigmaStore,
                "I": self.IStore,
                "V": self.VStore,
                "yHat": self.yHatStore,
                "VOCV": self.VOCVStore,
                "Innovation": self.innovationStore,
                "dOCVSOC": self.dOCVSOCStore,
                "yHatKF": self.yHatKFStore,
            }
        )

        """iterate through KF"""
        self.iterKF()

    def importmodel(self):
        self.dfOCV = pd.read_csv(
            self.parentPath + "/data/OCV--25degC--549_C20DisCh.csv"
        )
        self.timeOCV = self.dfOCV["time"].to_numpy()
        self.voltOCV = self.dfOCV["OCV"].to_numpy()
        self.SOCOCV = self.dfOCV["SOC"].to_numpy()
        self.capacityOCV = self.dfOCV["disCapacity"].to_numpy()[0]
        self.dfCellParamsOpti = pd.read_csv(
            self.parentPath + "/data/CellParams--25degC--551_Mixed1.csv"
        )
        self.r0 = self.dfCellParamsOpti["r0"].to_numpy()
        self.r1 = self.dfCellParamsOpti["r1"].to_numpy()
        self.r2 = self.dfCellParamsOpti["r2"].to_numpy()
        self.c1 = self.dfCellParamsOpti["c1"].to_numpy()
        self.c2 = self.dfCellParamsOpti["c2"].to_numpy()
        self.RC1 = np.exp(-self.dt / (self.r1 * self.c1))
        self.RC2 = np.exp(-self.dt / (self.r2 * self.c2))

    def importdf(self, filename):
        self.df = pd.read_csv(
            self.parentPath + "/data/" + filename, skiprows=28, dtype=str
        )
        self.df = self.df.loc[:, ~self.df.columns.str.contains("^Unnamed")]
        self.df = self.df.drop(0)
        self.df = self.df.apply(pd.to_numeric, errors="ignore")
        self.progTime = [
            self.convertToSec(progTime) for progTime in self.df["Prog Time"]
        ]
        self.time = [progTime - self.progTime[0] for progTime in self.progTime]
        self.df["Time"] = [time for time in self.time]
        self.volt = np.asarray([voltage for voltage in self.df["Voltage"]])
        self.curr = np.asarray([-current for current in self.df["Current"]])
        self.disCap = np.asarray([capacity for capacity in self.df["Capacity"]])
        self.dt = np.mean(np.diff(self.time))
        self.eta = 1.0

    def convertToSec(self, progTime):
        [h, m, s] = map(float, progTime.split(":"))
        return h * 3600 + m * 60 + s

    def genInput(self, i):
        z = np.array([self.curr[i]])
        z = z + self.sigmaV @ np.random.randn(1, 1)
        return z.astype(float)

    def genMeasurement(self, i):
        z = np.array([self.volt[i]])
        z = z + self.sigmaV @ np.random.randn(1, 1)
        return z.astype(float)

    def gendOCVSOC(self, soc):
        dZ = np.mean(np.diff(self.SOCOCV))
        index = np.argmin(abs(self.SOCOCV - soc))
        if index != (len(self.SOCOCV) - 1):
            dOCVSOC_fwd = abs(self.voltOCV[index] - self.voltOCV[index + 1]) / dZ
        if index != 0:
            dOCVSOC_rvs = abs(self.voltOCV[index - 1] - self.voltOCV[index]) / dZ
        if index == (len(self.SOCOCV) - 1):
            dOCVSOC_fwd = dOCVSOC_rvs
        if index == 0:
            dOCVSOC_rvs = dOCVSOC_fwd
        dOCVSOC = -(dOCVSOC_fwd + dOCVSOC_rvs) / 2

        return dOCVSOC

    def iterKF(self):
        for k in range(self.N):
            """generate input and measurement"""
            I = self.genInput(k)
            V = self.genMeasurement(k)
            dOCVSOC = self.gendOCVSOC(self.xHat[0])
            OCV = self.voltOCV[np.argmin(abs(self.SOCOCV - self.xHat[0]))]

            """EKF step 0: compute Ahat, Bhat, Chat, Dhat"""
            Ahat = np.diag([1, self.RC1.item(0), self.RC2.item(0)])
            Bhat = np.array(
                [
                    np.array([-self.dt * self.eta / (3600 * self.capacityOCV)]),
                    1 - self.RC1,
                    1 - self.RC2,
                ]
            ).T
            Chat = np.array([np.array([dOCVSOC]), -self.r1, self.r2]).T
            Dhat = np.array([1])

            """EKF step 1: state estimate prediction update"""
            self.xHat[0] = self.xHat[0] - self.dt * I * self.eta / (
                3600 * self.capacityOCV
            )
            self.xHat[1] = self.xHat[1] * self.RC1 + (1 - self.RC1) * I
            self.xHat[2] = self.xHat[2] * self.RC2 + (1 - self.RC2) * I

            """EKF step 1b: state covariance prediction update"""
            self.sigmaX = np.dot(np.dot(Ahat, self.sigmaX), Ahat.T) + np.dot(
                np.dot(Bhat, self.sigmaW), Bhat.T
            )

            """EKF step 1c: predict system output"""
            self.yHat = (
                OCV - self.r1 * self.xHat[1] - self.r2 * self.xHat[2] - self.r0 * I
            )

            """EKF step 2a: compute kalman gain"""
            sigmaY = np.dot(np.dot(Chat, self.sigmaX), Chat.T) + np.dot(
                np.dot(Dhat, self.sigmaV), Dhat.T
            )
            L = np.dot(np.dot(self.sigmaX, Chat.T), np.linalg.inv(sigmaY))

            """EKF step 2b: state estimate measurement update"""
            innovation = V - self.yHat
            self.xHat = self.xHat + np.dot(L, innovation)

            """EKF step 2c: state covriance measurement update"""
            self.sigmaX = self.sigmaX - np.dot(np.dot(L, sigmaY), L.T)

            """make EKF robust"""
            """limit state values"""
            self.xHat[0] = np.clip(self.xHat[0], 0, 1)
            """make sure state covariance is positive definite"""
            D, S, Vsvd = np.linalg.svd(self.sigmaX)
            HH = Vsvd * S * np.transpose(Vsvd)
            self.sigmaX = (
                self.sigmaX + np.transpose(self.sigmaX) + HH + np.transpose(HH)
            ) / 4

            """get the final terminal voltage after KF update"""
            yHatKF = OCV - self.r1 * self.xHat[1] - self.r2 * self.xHat[2] - self.r0 * I

            """store in dataframe"""
            self.socStore[k] = self.xHat[0]
            self.Irc1Store[k] = self.xHat[1]
            self.Irc2Store[k] = self.xHat[2]
            self.socSigmaStore[k] = self.sigmaX[0, 0]
            self.Irc1SigmaStore[k] = self.sigmaX[1, 1]
            self.Irc2SigmaStore[k] = self.sigmaX[2, 2]
            self.IStore[k] = I
            self.VStore[k] = V
            self.yHatStore[k] = self.yHat
            self.VOCVStore[k] = OCV
            self.innovationStore[k] = innovation
            self.dOCVSOCStore[k] = dOCVSOC
            self.yHatKFStore[k] = yHatKF

        self.storeDF = pd.DataFrame(
            {
                "SOC": self.socStore,
                "Irc1": self.Irc1Store,
                "Irc2": self.Irc2Store,
                "socSigma": self.socSigmaStore,
                "Irc1Sigma": self.Irc1SigmaStore,
                "Irc2Sigma": self.Irc2SigmaStore,
                "I": self.IStore,
                "V": self.VStore,
                "yHat": self.yHatStore,
                "VOCV": self.VOCVStore,
                "Innovation": self.innovationStore,
                "dOCVSOC": self.dOCVSOCStore,
                "yHatKF": self.yHatKFStore,
            }
        )

        """get true SOC"""
        self.storeDF["trueSOC"] = (
            self.df["Capacity"] / self.capacityOCV
            + self.SOCOCV[np.argmin(abs(self.voltOCV - self.df["Voltage"][1]))]
        )


if __name__ == "__main__":
    if "__ipython__":
        # add base folder path
        sys.path.append(os.path.dirname(os.path.realpath(os.getcwd())))
        N = np.inf
        filename = "551_LA92.csv"
        obj = extendedKFBatt(N)
        # change parent path to base folder
        obj.parentPath = os.path.dirname(os.getcwd())
        obj.runKF(filename)
    else:
        N = np.inf
        filename = "551_LA92.csv"
        obj = extendedKFBatt(N)
        obj.runKF(filename)

    print("Done")
