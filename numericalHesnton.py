
from datetime import time
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
import TGMethodGrid 

class numerticalHenston:
    def __init__(self, kappa, theta, epsilon, rho):
        ## kappa, theta, epsilon: Henston model parameters
        self._kappa = kappa
        self._theta = theta
        self._epsilon = epsilon
        self._rho = rho
    def conditional_mean(self, Vt, t, T):
        ## conditional mean of V(T) | V(t).
        ##  T, t expressed in year
        result = self._theta + (Vt - self._theta) * np.exp(-1 * self._kappa * (T - t))
        return result
    
    def conditional_var(self, Vt, t, T):
        ## conditional var of V(T) | V(t).
        ##  T, t expressed in year
        result = 1 / self._kappa * Vt * np.power(self._epsilon, 2) * np.exp(-1 * self._kappa * (T - t)) * ( 1- np.exp(-1 * self._kappa * (T - t))) \
            + 1 / (2 * self._kappa) * self._theta * np.power(self._epsilon, 2) *  ( 1- np.exp(-1 * self._kappa * (T - t))) ** 2
        return result

class EulerHeston(numerticalHenston):
    def __init__(self, kappa, theta, epsilon, rho):
        super().__init__(kappa, theta, epsilon, rho)
    def simulateAssetPath(self, T, V0, S0, steps, numSim):
        ## T: Time to matirity
        ## V0: Start variance value
        ## S0: Start price value e.g. 4900
        ## steps: number of steps to simulated
        ## numSim: number of simulation. 
        timeStep = T / steps
        result_asset = np.zeros((numSim, steps + 1))
        result_var = np.zeros((numSim, steps + 1))
        result_asset[:,0] = np.ones(numSim) * S0
        result_var[:,0] = np.ones(numSim) * V0
        for i in range(steps):
            ## Simulate next variance value 
            last_V_max = np.maximum(result_var[:,i], 0)
            normal_v = np.random.normal(size = numSim)
            next_V = result_var[:,i] + self._kappa * (self._theta - last_V_max) * timeStep + \
                self._epsilon * np.sqrt(last_V_max ) * normal_v * np.sqrt(timeStep)
            result_var[:,i+1] = next_V 

            normal_S = self._rho * normal_v + np.sqrt(1 - self._rho ** 2) * np.random.normal(size = numSim)
            next_S = result_asset[:,i] - 0.5 * last_V_max * timeStep + np.sqrt(last_V_max) * normal_S * np.sqrt(timeStep)
            result_asset[:,i+1] = next_S
        return (result_asset, result_var)


class TGHeston(numerticalHenston):
    def __init__(self, kappa, theta, epsilon, rho, grid , discret_scheme = "Central"):
        super().__init__(kappa, theta, epsilon, rho)
        ## Create a grid
        self._gridpt = grid[:,0]
        self._gridvalue = grid[:,1]
        ## Create intepolator
        self.f_mu = interp1d(self._gridpt, TGMethodGrid.TG_mean(self._gridvalue), kind = "cubic")
        self.f_sigma = interp1d(self._gridpt, TGMethodGrid.TG_var(self._gridvalue, self._gridpt), kind = "cubic")
        self._phi_start = np.min(self._gridpt)
        self._discrete_scheme = discret_scheme
    def getMuAndSigmaFromPhi(self, Phi):
        result_mu = np.zeros(Phi.shape[0]) ## placeholder
        result_sigma = np.zeros(Phi.shape[0])
        
        noNeedForMomentMatching = np.where(Phi < self._phi_start)
        NeedForMomentMatching = np.where(Phi >= self._phi_start)
        result_mu[noNeedForMomentMatching] = 1
        result_sigma[noNeedForMomentMatching] = 1

        result_mu[NeedForMomentMatching] = self.f_mu(Phi[NeedForMomentMatching])
        result_sigma[NeedForMomentMatching] = self.f_sigma(Phi[NeedForMomentMatching])
        return (result_mu, result_sigma)

    def simulateNextVarValue(self, Vt, t, T):
        ## find m and s
        _m = self.conditional_mean(Vt,t,T)
        _s_2 = self.conditional_var(Vt,t,T)
        phi = _s_2 / np.power(_m, 2)
        ## get g_mu and f_sigma from cache
        (f_mu, f_sigma) = self.getMuAndSigmaFromPhi(phi)
        ## compute mu and sigma
        mu = f_mu * _m
        sigma = f_sigma * np.sqrt(_s_2)
        ## sample sd normal r.v
        standNorm = np.random.normal(size = Vt.shape[0])
        return np.maximum(mu + sigma * standNorm, 0)
    
    def simulateNextAssetValue(self, v_last, v_next, as_last, t, T):
        time_delta = T - t
        r1 = 0.5
        r2 = 0.5
        if (self._discrete_scheme == "Euler"):
            r1 = 1
            r2 = 0
        ## Step1: initialize K_i, i = 0,...,4
        K0 = - (self._rho * self._kappa * self._theta) / self._epsilon * time_delta
        K1 = r1 * time_delta * (self._kappa * self._rho / self._epsilon - 1 / 2) - self._rho / self._epsilon
        K2 = r2 * time_delta * (self._kappa * self._rho / self._epsilon - 1 / 2) + self._rho / self._epsilon
        K3 = r1 * time_delta * (1 - self._rho ** 2) 
        K4 = r2 * time_delta * (1 - self._rho ** 2) 

        ## Step2: Generate N(0,1) r.vs
        normal = np.random.normal(size=as_last.shape[0])

        ## Step 3: Calulate ln(X(t + \delta))
        result = as_last + K0 + K1 * v_last + K2 * v_next + np.sqrt(K3 * v_last + K4 * v_next) * normal

        return result
    def callOptionPriceFromCMC(self, T, V0, S0, steps, numSim, K, rt, scheme = "Central"):
        ## T: Time to matirity
        ## V0: Start volatility value
        ## SO: ln(X(0))
        ## steps: number of steps to simulated
        ## numSim: number of simulation. 
        ## K: Strike K 
        ## rt: risk-free rate
        ## Follow HVRTCMC page 7 in Stat 906 uwaterloo. 
        r1, r2 = 1/2
        if (scheme == "Euler"):
            r1 = 1
            r2 = 0
        timeStep = T / steps
        result_var = np.zeros((numSim, steps + 1))
        ## initialize V0
        result_var[:,0] = np.ones(numSim) * V0
        sum1 = np.zeros(numSim)   #\int_0^T \sigma_u dWu
        for i in range(steps):
            ## Simulate V(t + \delta) first
            t = i * timeStep ## Current time
            tp1 = i * timeStep + timeStep ## Next time
            next_var = self.simulateNextVarValue(result_var[:,i], t, tp1)
            result_var[:,i+1] = next_var
        sum0 = timeStep * (r1 * np.sum(result_var[:,0 : steps], axis = 1) + \
            r2 * np.sum(result_var[:,1 : steps + 1], axis = 1)) #\int_0^T \sqrt{\sigma_u} du
        sum1 = 1 / self._epsilon * (result_var[:,-1] - result_var[:,0]) - self._kappa * self._theta * timeStep + \
            self._kappa * sum0
        v2 = (1 - self._rho ** 2) / T * sum0
        eta = np.exp(-0.5 * (self._rho ** 2) * sum0 + self._rho * sum1)
        ## For BS call option formula 
        d1 = (S0 + np.log(eta / K * np.exp(-1 * rt * T)) + 0.5 * v2 * T) / (np.sqrt(v2 * T))
        d2 = d1 - np.sqrt(v2 * T)
        ## BS Formula
        CallPrice = np.exp(S0) * eta * stats.norm.cdf(d1) - K * np.exp(-1 * rt * T) * stats.norm.cdf(d2)
        return CallPrice

    def sampleVariancePath(self, T, V0, steps, numSim):
        ## T: Time to matirity
        ## V0: Start variance value
        ## steps: number of steps to simulated
        ## numSim: number of simulation. 
        timeStep = T / steps
        result = np.zeros((numSim, steps + 1))
        result[:,0] = np.ones(numSim) * V0 
        for i in range(steps):
            ## find m and s
            t = i * timeStep ## Current time
            tp1 = i * timeStep + timeStep ## Next time
            result[:,i + 1] = self.simulateNextVarValue(result[:,i], t, tp1)
        return result
    
    def simulateAssetPath(self, T, V0, S0, steps, numSim):
        ## T: Time to matirity
        ## V0: Start volatility value
        ## SO: ln(X(0))
        ## steps: number of steps to simulated
        ## numSim: number of simulation. 
        ## Follows the BK scheme to simulate 
        timeStep = T / steps
        result_asset = np.zeros((numSim, steps + 1))
        result_var = np.zeros((numSim, steps + 1))
        ## initialize V0, S0
        result_asset[:,0] = np.ones(numSim) * S0
        result_var[:,0] = np.ones(numSim) * V0
        for i in range(steps):
            ## Simulate V(t + \delta) first
            t = i * timeStep ## Current time
            tp1 = i * timeStep + timeStep ## Next time
            next_var = self.simulateNextVarValue(result_var[:,i], t, tp1)
            result_var[:,i+1] = next_var
            ## Get next asset price 
            next_as = self.simulateNextAssetValue(result_var[:,i], result_var[:,i+1], result_asset[:, i], t, tp1)
            result_asset[:,i+1] = next_as
        return (result_asset, result_var)


class QEHeston(numerticalHenston):
    def __init__(self, kappa, theta, epsilon, rho, switchingRule,discret_scheme = "Central"):
        super().__init__(kappa, theta, epsilon, rho)
        self._switchingCoef = switchingRule
        self._discrete_scheme = discret_scheme
    def quadratic_normal_param(self, m, _s_2):
        ## Assume phi = m^2 / _s_2 <= 2
        phi = _s_2 / np.power(m,2)
        _b_2 = 2 / phi - 1 + np.sqrt(2 / phi) * np.sqrt(2 / phi - 1)
        a = m / (1 + _b_2)
        return (a, np.sqrt(_b_2))
    
    def tail_dist_param(self, m, _s_2):
        ## Assume phi >= 1 
        phi = _s_2 / np.power(m,2)
        p = (phi - 1) / (phi + 1)
        beta = (1 - p) / m 
        return (p,beta)

    def tail_dist_inversion(self, p, beta, x):
        ## Assume x \in [0, 1]
        ## p, beta, x: ndarray 
        result = np.zeros(x.shape[0])
        nonZeroDensity = np.where(x > p)
        result[nonZeroDensity]  = 1 / beta[nonZeroDensity] * np.log( (1 - p[nonZeroDensity]) / (1 - x[nonZeroDensity]) )
        return result 

    def simulateNextVarValue(self, Vt,t,T):
        ## Get V(t + \delta) according to the QE scheme
        ## t: current time; T: next time, both in years
        ## numSim: Total number of simulations

        ## compute var and mean
        _m = self.conditional_mean(Vt,t,T)
        _s_2 = self.conditional_var(Vt,t,T)
        phi = _s_2 / np.power(_m,2)

        result_array = np.zeros(Vt.shape[0]) # placeholder 
        quadratic_scheme = np.where(phi <= self._switchingCoef) 
        tail_dist_scheme = np.where(phi > self._switchingCoef) 

        ## Prepare Quadratic_Scheme: 
        (a, b) = self.quadratic_normal_param(_m[quadratic_scheme],_s_2[quadratic_scheme])
        normal = np.random.normal(size = a.shape[0])
        result_array[quadratic_scheme] = a * np.power(b + normal, 2)

        ## Prepare Tail_Scheme Part 
        (p, beta) = self.tail_dist_param(_m[tail_dist_scheme],_s_2[tail_dist_scheme])
        uniform = np.random.rand(p.shape[0])
        result_array[tail_dist_scheme] = self.tail_dist_inversion(p, beta, uniform)

        return result_array

    def simulateNextAssetValue(self, v_last, v_next, as_last, t, T):
        time_delta = T - t
        r1 = 0.5
        r2 = 0.5
        if (self._discrete_scheme == "Euler"):
            r1 = 1
            r2 = 0
        ## Step1: initialize K_i, i = 0,...,4
        K0 = - (self._rho * self._kappa * self._theta) / self._epsilon * time_delta
        K1 = r1 * time_delta * (self._kappa * self._rho / self._epsilon - 1 / 2) - self._rho / self._epsilon
        K2 = r2 * time_delta * (self._kappa * self._rho / self._epsilon - 1 / 2) + self._rho / self._epsilon
        K3 = r1 * time_delta * (1 - self._rho ** 2) 
        K4 = r2 * time_delta * (1 - self._rho ** 2) 

        ## Step2: Generate N(0,1) r.vs
        normal = np.random.normal(size=as_last.shape[0])

        ## Step 3: Calulate ln(X(t + \delta))
        result = as_last + K0 + K1 * v_last + K2 * v_next + np.sqrt(K3 * v_last + K4 * v_next) * normal

        return result
    def simulateVariancePath(self, T, V0, steps, numSim):
        ## T: Time to matirity
        ## V0: Start volatility value
        ## steps: number of steps to simulated
        ## numSim: number of simulation. 
        timeStep = T / steps
        result = np.zeros((numSim, steps + 1))
        result[:,0] = np.ones(numSim) * V0 
        for i in range(steps):
            t = i * timeStep # Current time
            tp1 = (i + 1) * timeStep # Next time
            nextValue = self.simulateNextVarValue(result[:,i], t, tp1)
            result[:,i + 1] = nextValue
        return result
    
    def simulateAssetPath(self, T, V0, S0, steps, numSim):
        ## T: Time to matirity
        ## V0: Start volatility value
        ## SO: ln(X(0))
        ## steps: number of steps to simulated
        ## numSim: number of simulation. 
        ## Follows the BK scheme to simulate 
        timeStep = T / steps
        result_asset = np.zeros((numSim, steps + 1))
        result_var = np.zeros((numSim, steps + 1))
        ## initialize V0, S0
        result_asset[:,0] = np.ones(numSim) * S0
        result_var[:,0] = np.ones(numSim) * V0
        for i in range(steps):
            ## Simulate V(t + \delta) first
            t = i * timeStep ## Current time
            tp1 = i * timeStep + timeStep ## Next time
            next_var = self.simulateNextVarValue(result_var[:,i], t, tp1)
            result_var[:,i+1] = next_var
            ## Get next asset price 
            next_as = self.simulateNextAssetValue(result_var[:,i], result_var[:,i+1], result_asset[:, i], t, tp1)
            result_asset[:,i+1] = next_as
        return (result_asset, result_var)
    
    def callOptionPriceFromCMC(self, T, V0, S0, steps, numSim, K, rt, scheme = "Central"):
        ## T: Time to matirity
        ## V0: Start volatility value
        ## SO: ln(X(0))
        ## steps: number of steps to simulated
        ## numSim: number of simulation. 
        ## K: Strike K 
        ## rt: risk-free rate
        ## Follow HVRTCMC page 7 in Stat 906 uwaterloo. 
        r1 = 1/2
        r2 = 1/2
        if (scheme == "Euler"):
            r1 = 1
            r2 = 0
        timeStep = T / steps
        result_var = np.zeros((numSim, steps + 1))
        ## initialize V0
        result_var[:,0] = np.ones(numSim) * V0
        for i in range(steps):
            ## Simulate V(t + \delta) first
            t = i * timeStep ## Current time
            tp1 = i * timeStep + timeStep ## Next time
            next_var = self.simulateNextVarValue(result_var[:,i], t, tp1)
            result_var[:,i+1] = next_var
        sum0 = timeStep * (r1 * np.sum(result_var[:,0 : steps], axis = 1) + \
            r2 * np.sum(result_var[:,1 : steps + 1], axis = 1)) #\int_0^T \sigma_u du
        sum1 = 1 / self._epsilon * (result_var[:,-1] - result_var[:,0] - self._kappa * self._theta * T + \
            self._kappa * sum0) #\int_0^T \sqrt{\sigma_u} dWu
        v2 = (1 - self._rho ** 2) / T * sum0
        eta = np.exp(-0.5 * (self._rho ** 2) * sum0 + self._rho * sum1)
        ## For BS call option formula 
        d1 = (S0 + np.log(eta / K * np.exp(-1 * rt * T)) + 0.5 * v2 * T) / (np.sqrt(v2 * T))
        d2 = d1 - np.sqrt(v2 * T)
        ## BS Formula
        CallPrice = np.exp(S0) * eta * stats.norm.cdf(d1) - K * np.exp(-1 * rt * T) * stats.norm.cdf(d2)
        return CallPrice
"""
## For test 
epsion = 0.4
kappa = 0.5
theta = 0.04
T = 2
timeStep = 100
rho = 0.3
V0 = 0.04
S0 = 4900
grid = np.genfromtxt('/Users/zhiwang/Desktop/STAT906/Code/ResearchPaper/grid.csv')
myHeston = TGHeston(kappa,theta,epsion,rho,grid)
swtiching_rule = 1.5
myHentonQE = QEHeston(kappa, theta, epsion,rho, 1.5)
my_result = myHentonQE.simulateAssetPath(T,V0,S0,timeStep,100)


## Test 2
epsion = 0.4
kappa = 0.5
theta = 0.04
T = 2
timeStep = 1000
rho = -0.3
V0 = 0.04
S0 = 4900
grid = np.genfromtxt('/Users/zhiwang/Desktop/STAT906/Code/ResearchPaper/grid.csv')
swtiching_rule = 1.5
myHentonQE = QEHeston(kappa, theta, epsion,rho, 1.5)
CMC = myHentonQE.callOptionPriceFromCMC(T,V0, np.log(S0), timeStep, 100, S0,0)
print(np.mean(CMC))
"""
