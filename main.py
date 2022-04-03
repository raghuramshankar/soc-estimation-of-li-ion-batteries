from src.runSOC import runSOC

def main():
    runSOCObj = runSOC()
    runSOCObj.cubatureKalmanFilter(runSOCObj.xEst, runSOCObj.pEst, runSOCObj.z)

main()