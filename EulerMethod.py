
import numpy as np

def HestonEuler(epsion, kappa, rho, T, theta, V0, S0,  timeStep):
    numSim = int(T / timeStep)
    
    normal_V = np.random.normal(size = numSim)
    normal_S = rho *  normal_V + np.sqrt(1 - rho ** 2) * np.random.normal(size = numSim)

    output_V = np.zeros(numSim + 1)
    outPut_S = np.zeros(numSim + 1) 
    output_V[0] = V0
    outPut_S[0] = S0

    for i in range(numSim):
        last_V_max = np.maximum(output_V[i], 0)
        next_V = output_V[i] + kappa * (theta - last_V_max) * timeStep + epsion * np.sqrt(last_V_max) * normal_V[i] * np.sqrt(timeStep)
        output_V[i + 1] = next_V

        next_S = outPut_S[i] - 0.5 * last_V_max * timeStep + np.sqrt(last_V_max) * normal_S[i] * np.sqrt(timeStep)
        outPut_S[i+1] = next_S
    
    return (output_V , outPut_S)
