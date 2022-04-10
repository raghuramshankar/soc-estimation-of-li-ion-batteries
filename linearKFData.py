import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class linearKF():
    def __init__(self):
        models.__init__(self)
        self.N = 200

        '''inital state mean'''
        self.x0 = np.array([[0.0],                      # current    [A]
                            [1.0]])                     # SOC       [abs]

        '''inital state covariance'''
        self.p0 = np.array([[1e-3, 0.0],
                            [0.0, 1e-3]])

        '''generate measurement noise'''
        self.zNoise = np.array([[1e-3]])                # current   [A]

        '''process noise'''
        self.q = np.array([[1e-6, 0.0],                 # current    [A]
                           [0.0, 1e-6]])                # SOC       [abs]

        '''measurement model matrix'''
        self.hx = np.array([[1.0, 0.0]])                # current    [A]

        '''measurement noise'''
        self.r = np.array([[1e-3]])**2                  # current   [A]

        self.importdf()
        self.importmodel()
        self.flags()

    def flags(self):
        self.showFinal = 1
        self.showAnimation = 1
        self.showEllipse = 0

    def importmodel(self):
        self.dfOCV = pd.read_csv('data/OCV--25degC--549_C20DisCh.csv')
        self.timeOCV = self.dfOCV["time"].to_numpy()
        self.voltOCV = self.dfOCV["OCV"].to_numpy()
        self.SOCOCV = self.dfOCV["SOC"].to_numpy()
        self.capacityOCV = self.dfOCV["disCapacity"].to_numpy()[0]
        self.dfCellParamsOpti = pd.read_csv(
            'data/CellParams--25degC--551_Mixed1.csv')
        self.r0 = self.dfCellParamsOpti["r0"].to_numpy()
        self.r1 = self.dfCellParamsOpti["r1"].to_numpy()
        self.r2 = self.dfCellParamsOpti["r2"].to_numpy()
        self.c1 = self.dfCellParamsOpti["c1"].to_numpy()
        self.c2 = self.dfCellParamsOpti["c2"].to_numpy()
        self.RC1 = np.exp(-self.dt/(self.r1 * self.c1))
        self.RC2 = np.exp(-self.dt/(self.r2 * self.c2))

    def importdf(self):
        self.df = pd.read_csv('data/551_Mixed1.csv', skiprows=28, dtype=str)
        self.df = self.df.loc[:, ~self.df.columns.str.contains("^Unnamed")]
        self.df = self.df.drop(0)
        self.df = self.df.apply(pd.to_numeric, errors="ignore")

        self.progTime = [self.convertToSec(progTime)
                         for progTime in self.df["Prog Time"]]
        self.time = [progTime - self.progTime[0] for progTime in self.progTime]
        self.df["Time"] = [time for time in self.time]

        self.volt = np.asarray([voltage for voltage in self.df["Voltage"]])
        self.curr = np.asarray([-current for current in self.df["Current"]])
        self.disCap = np.asarray(
            [capacity for capacity in self.df["Capacity"]])
        self.dt = np.mean(np.diff(self.time))
        self.eta = 1.0

    def convertToSec(self, progTime):
        [h, m, s] = map(float, progTime.split(":"))
        return h * 3600 + m * 60 + s

    