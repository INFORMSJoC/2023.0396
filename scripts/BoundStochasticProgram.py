# MIT License

# Copyright (c) 2024 Xian Yu and Siqian Shen

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
Main function
'''

import numpy as np
from numpy import newaxis
import pandas as pd
import gurobipy as gb
import scipy.stats as stats
import time
import pickle
from haversine import haversine, haversine_vector, Unit
from msppy.msp import MSLP, MSIP
from msppy.solver import SDDP, SDDiP
from msppy.solver import Extensive
from msppy.evaluation import Evaluation

class BoundStochasticProgram():
    def __init__(self, numStages, numFacilities, numCustomers, numBranches, std, outScenarios, lambda_risk = 0.9, alpha_risk = 0.95):
        self.numStages = numStages
        self.numFacilities = numFacilities
        self.numCustomers = numCustomers
        self.numBranches = numBranches 
        self.std = std
        self.lambda_risk = lambda_risk
        self.alpha_risk = alpha_risk
        self.outScenarios = outScenarios

    def generateDataFromDaskinNonSAA(self, filename_facilities, filename_customers):
        # generate data from mark daskin and Lawrence paper: Reliability Models for Facility Location: The Expected Failure Cost Case
        df_fac = pd.read_excel(filename_facilities, 
                                header = 0
                                )
        df_cus = pd.read_excel(filename_customers, 
                                header = 0
                                )
        self.numFacilities = len(df_fac.index)
        self.numCustomers = len(df_cus.index)
        # self.maintenance_cost = df_fac['fixed cost'].to_numpy()
        self.maintenance_cost = np.ones(self.numFacilities) * 100 #maintenance cost for one charging station per year
        loc_fac = df_fac[['lat', 'real lon']].to_numpy()
        loc_cus = df_cus[['lat', 'real lon']].to_numpy()

        travel_distance = haversine_vector(loc_cus, loc_fac, Unit.MILES, comb=True) # pairwise distance in MILES
        self.transportation_cost = travel_distance * 0.00001 # see reference paper
        self.capacity = np.ones(self.numFacilities) * 6 * 30 * 12 # level 2 charger can charge a PHEV from empty in 2 hours


        self.demandMean = np.zeros([self.numStages, self.numCustomers])
        self.demandMean[0, :] = df_cus['demand'][:self.numCustomers].to_numpy() * 1e4 * 0.06 * 10 * 12 # the raw data is the population of each state in 1990, assuming 2% will order once per month
        for t in range(1, self.numStages):
            self.demandMean[t, :] = self.demandMean[0, :] * (1 + 0 * t)

        self.numScenarios = self.numBranches ** (self.numStages - 1)   
        self.demandNode = np.zeros([self.numStages, self.numCustomers, self.numScenarios]) 
        self.demandPath = np.zeros([self.numStages, self.numCustomers, self.numScenarios]) 
        np.random.seed(2)
        for t in range(self.numStages):
            for j in range(self.numCustomers):
                if t > 0:
                    lower, upper = 0, 1e10
                    mu, sigma = self.demandMean[t, j], self.demandMean[t, j] * (self.std + 0 * t)
                    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc = mu, scale = sigma)
                    self.demandNode[t, j, :] = (X.rvs(self.numScenarios)) 


        for s in range(self.numScenarios):
            for j in range(self.numCustomers):
                for tt in range(self.numStages):
                    if tt == 0:
                        self.demandPath[tt, j, s] = self.demandMean[tt, j]
                    else:
                        self.demandPath[tt, j, s] = self.demandNode[tt, j, int(np.floor(s / self.numBranches**(self.numStages - tt - 1)))]
        

        return self

    def generateSyntheticDataNew(self):

        self.capacity = 1e3 * np.ones(self.numFacilities) 
        self.maintenance_cost = 60000 * np.ones(self.numFacilities) #rental fee per year: 10000 square feet times 0.5 rental fee per month times 12 months = 60000 at default; set to 1e5 if want to get twostage=multistage  so that the bound is tight
        # np.random.seed(1)
        
        self.posFacilities = np.zeros([self.numFacilities, 2])
        self.posCustomers = np.zeros([self.numCustomers, 2])
        self.posFacilities[:, 0] = np.random.permutation(100)[0 : self.numFacilities]
        self.posFacilities[:, 1] = np.random.permutation(100)[0 : self.numFacilities]
        self.posCustomers[:, 0] = np.random.permutation(100)[:self.numCustomers]
        self.posCustomers[:, 1] = np.random.permutation(100)[:self.numCustomers]
        

        self.transportation_cost = np.zeros([self.numFacilities, self.numCustomers])
        for i in range(self.numFacilities):
            for j in range(self.numCustomers):
                self.transportation_cost[i, j] = abs(self.posFacilities[i, 0] - self.posCustomers[j, 0]) + abs(self.posFacilities[i, 1] - self.posCustomers[j, 1]) # miles between each pair
                self.transportation_cost[i, j] = self.transportation_cost[i, j] * 0.575 / 100

    def generateScenariosSAA(self):
        # SAA type scenario tree
        # for comparing two-stage and multistage, so demand scenarios must be forming a multistage scenario tree
        self.demandMean = np.zeros([self.numStages, self.numCustomers])
        self.demandNode = np.zeros([self.numStages, self.numCustomers, self.numBranches]) 

        np.random.seed(1)
        for t in range(self.numStages):
            for j in range(self.numCustomers):
                #self.demandMean[j] = self.demand_raw[int(np.where(self.indice == j)[0])]/1e5
                self.demandMean[t, j] = np.random.randint(1000 * (2 * t + 1), 5000 * (2 * t + 1))
                if t > 0:
                    lower, upper = 0, 2.5e10
                    mu, sigma = self.demandMean[t, j], self.demandMean[t, j] * self.std
                    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc = mu, scale = sigma)
                    self.demandNode[t, j, :] = (X.rvs(self.numBranches))  


        self.numScenarios = self.numBranches ** (self.numStages - 1)
        self.demandPath = np.zeros([self.numStages, self.numCustomers, self.numScenarios]) #represent the demand at each path

        
        for s in range(self.numScenarios):
            s = s + 1
            index = np.zeros(self.numStages - 1)
            numerator = s
            denominator = self.numBranches ** (self.numStages - 2)
            index[0] = np.ceil(numerator / denominator)
            for t in range(1, self.numStages - 1):
                numerator = numerator - denominator * (index[t - 1] - 1)
                denominator = self.numBranches ** (self.numStages - t - 2)
                index[t] = np.ceil(numerator / denominator)

            for j in range(self.numCustomers):
                for tt in range(self.numStages):
                    if tt == 0:
                        self.demandPath[tt, j, s - 1] = self.demandMean[tt, j]
                    else:
                        self.demandPath[tt, j, s - 1] = self.demandNode[tt, j, int(index[tt - 1]) - 1]
        
        return self

    def generateScenariosNonSAA(self):
        # non-SAA type scenario tree
        self.numScenarios = self.numBranches ** (self.numStages - 1)   
        self.demandMean = np.zeros([self.numStages, self.numCustomers])
        self.demandNode = np.zeros([self.numStages, self.numCustomers, self.numScenarios]) 

     
        #np.random.seed(0)
        for t in range(self.numStages):
            for j in range(self.numCustomers):
                self.demandMean[t, j] = np.random.randint(1000 * (2 * t + 1), 5000 * (2 * t + 1))
                if t > 0:
                    lower, upper = 0, 2.5e10
                    mu, sigma = self.demandMean[t, j], self.demandMean[t, j] * self.std
                    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc = mu, scale = sigma)
                    self.demandNode[t, j, :] = (X.rvs(self.numScenarios))  #np.ceil


        self.demandPath = np.zeros([self.numStages, self.numCustomers, self.numScenarios]) #represent the demand at each path

        
        for s in range(self.numScenarios):
            for j in range(self.numCustomers):
                for tt in range(self.numStages):
                    if tt == 0:
                        self.demandPath[tt, j, s] = self.demandMean[tt, j]
                    else:
                        self.demandPath[tt, j, s] = self.demandNode[tt, j, int(np.floor(s / self.numBranches**(self.numStages - tt - 1)))]
        
        return self

    def solveRiskAverseTwoStageModel2(self):

        st1 = time.time()
        m = gb.Model()
        xVar, yVar = {}, {}
        for t in range(self.numStages):
            for i in range(self.numFacilities):
                for s in range(self.numScenarios):
                    xVar[t, i, s] = m.addVar(lb = 0, vtype = gb.GRB.INTEGER)
                    for j in range(self.numCustomers):
                        yVar[t, i, j, s] = m.addVar(lb = 0, vtype = gb.GRB.CONTINUOUS)

        uVar, etaVar = {}, {}
        # add constraints
        for s in range(self.numScenarios):
            for t in range(self.numStages):
                if t < self.numStages - 1:
                    uVar[t, s] = m.addVar(lb = 0)
                    etaVar[t, s] = m.addVar()
                for i in range(self.numFacilities):
                    m.addConstr(sum(yVar[t, i, j, s] for j in range(self.numCustomers)) <= self.capacity[i] * sum(xVar[tt, i, s] for tt in range(t + 1)))
                for j in range(self.numCustomers):              
                    m.addConstr(sum(yVar[t, i, j, s] for i in range(self.numFacilities)) == (self.demandPath[t, j, s]))

        for t in range(self.numStages):
            for s in range(self.numScenarios):
                for i in range(self.numFacilities):
                    m.addConstr(xVar[t, i, s] == xVar[t, i, 0])  #additional constraint for two-stage

        # nonanticipativity constraints
        for t in range(self.numStages - 1):
            for s in range(self.numScenarios):
                f = self.numBranches ** (self.numStages - t - 1)
                flag = np.floor(s / f)
                m.addConstr(etaVar[t, s] == etaVar[t, int(flag * f)])
                for i in range(self.numFacilities):
                    m.addConstr(xVar[t, i, s] == xVar[t, i, int(flag * f)])
                    for j in range(self.numCustomers):
                        m.addConstr(yVar[t, i, j, s] == yVar[t, i, j, int(flag * f)])

                if t < self.numStages - 2:
                    f = self.numBranches ** (self.numStages - t - 2)
                    flag = np.floor(s / f)
                    m.addConstr(uVar[t, s] == uVar[t, int(flag * f)])

                m.addConstr(uVar[t, s] + etaVar[t, s] >= sum(sum(xVar[tt, i, s] for tt in range(t + 2)) * self.maintenance_cost[i] for i in range(self.numFacilities))
                            + sum(yVar[t + 1, i, j, s] * self.transportation_cost[i, j] for i in range(self.numFacilities) for j in range(self.numCustomers)))

        m.setObjective(float(1 / self.numScenarios) * sum(
                        sum(xVar[0, i, s] * self.maintenance_cost[i] for i in range(self.numFacilities)) + sum(yVar[0, i, j, s] * self.transportation_cost[i, j] for i in range(self.numFacilities) for j in range(self.numCustomers))
                        + self.lambda_risk * sum(etaVar[t, s] for t in range(self.numStages - 1))
                        + self.lambda_risk / (1 - self.alpha_risk) * sum(uVar[t, s] for t in range(self.numStages - 1))
                        + (1 - self.lambda_risk) * sum(sum(xVar[tt, i, s] for tt in range(t + 1)) * self.maintenance_cost[i] for t in range(1, self.numStages) for i in range(self.numFacilities))
                        + (1 - self.lambda_risk) * sum(yVar[t, i, j, s] * self.transportation_cost[i, j] for t in range(1, self.numStages) for i in range(self.numFacilities) for j in range(self.numCustomers))
                        for s in range(self.numScenarios)))
        
        m.setParam("TimeLimit", 3600)
        m.setParam("MIPGap", 0.005)
        m.setParam("OutputFlag", 0)
        m.update()
        st2 = time.time()
        m.optimize()
        st3 = time.time()

        results = {}
        results['optimal objective'] = m.objval
        results['construction time'] = st2 - st1
        results['optimization time'] = st3 - st2
        results['MIP gap'] = m.MIPGap
        xVarValues = np.zeros([self.numStages, self.numFacilities, self.numScenarios])
        yVarValues = np.zeros([self.numStages, self.numFacilities, self.numCustomers, self.numScenarios])
        uVarValues = np.zeros([self.numStages - 1, self.numScenarios])
        etaVarValues = np.zeros([self.numStages - 1, self.numScenarios])

        for t in range(self.numStages):
            for i in range(self.numFacilities):
                for j in range(self.numCustomers):
                    for s in range(self.numScenarios):
                        xVarValues[t, i, s] = xVar[t, i, s].x
                        yVarValues[t, i, j, s] = yVar[t, i, j, s].x
                        if t < self.numStages - 1:
                            uVarValues[t, s] = uVar[t, s].x
                            etaVarValues[t, s] = etaVar[t, s].x

        #round x to nearest integer as they are integers
        xVarValues = np.round(xVarValues)
        yVarValues = np.round(yVarValues, 7)
        yVarValues[yVarValues < 0] = 0

        delta = np.zeros([self.numStages, self.numFacilities, self.numScenarios])
        x_MS = np.zeros([self.numStages, self.numFacilities, self.numScenarios]) 
        x_TS = np.zeros([self.numStages, self.numFacilities, self.numScenarios]) 
        v_ms = np.zeros([self.numFacilities])
        v_ts = np.zeros([self.numFacilities])
        for t in range(self.numStages):
            for s in range(self.numScenarios):
                for i in range(self.numFacilities):
                    delta[t, i, s] = np.round(sum(yVarValues[t, i, j, s] for j in range(self.numCustomers)) / self.capacity[i], 9) # at default: 9

        # delta[delta > 1] = 1
        delta[delta < 0] = 0

        for t in range(self.numStages):
            for s in range(self.numScenarios):
                for i in range(self.numFacilities):       
                    if t == 0:
                        x_MS[t, i, s] = np.ceil(delta[t, i, s])
                        x_TS[t, i, s] = np.ceil(delta[t, i, s])
                    else:
                        x_MS[t, i, s] = max(np.ceil(delta[tt, i, s]) for tt in range(t + 1)) - max(np.ceil(delta[tt, i, s]) for tt in range(t))
                        x_TS[t, i, s] = max(np.ceil(max(delta[tt, i, :])) for tt in range(t + 1)) - max(np.ceil(max(delta[tt, i, :])) for tt in range(t))



        eta_MS = np.zeros([self.numStages - 1, self.numScenarios])
        eta_TS = np.zeros([self.numStages - 1, self.numScenarios])
        for t in range(1, self.numStages):
            for s in range(self.numScenarios):
                eta_MS[t - 1, s] = max(sum(sum(x_MS[tt, i, ss] for tt in range(t + 1)) * self.maintenance_cost[i] for i in range(self.numFacilities))
                            + sum(yVarValues[t, i, j, ss] * self.transportation_cost[i, j] for i in range(self.numFacilities) for j in range(self.numCustomers))
                            - uVarValues[t - 1, ss] 
                            for ss in range(int(self.numBranches**(self.numStages - t) * np.floor(s/(self.numBranches**(self.numStages - t)))), int(self.numBranches**(self.numStages - t) * (1 + np.floor(s/(self.numBranches**(self.numStages - t))))))) 
                eta_TS[t - 1, s] = max(sum(sum(x_TS[tt, i, ss] for tt in range(t + 1)) * self.maintenance_cost[i] for i in range(self.numFacilities))
                            + sum(yVarValues[t, i, j, ss] * self.transportation_cost[i, j] for i in range(self.numFacilities) for j in range(self.numCustomers))
                            - uVarValues[t - 1, ss] 
                            for ss in range(int(self.numBranches**(self.numStages - t) * np.floor(s/(self.numBranches**(self.numStages - t)))), int(self.numBranches**(self.numStages - t) * (1 + np.floor(s/(self.numBranches**(self.numStages - t)))))))

        new_bound = float(1 / self.numScenarios) * sum((1 - self.lambda_risk) * sum(self.maintenance_cost[i] * (max(np.ceil(max(delta[tt, i, :])) for tt in range(t + 1)) - max(np.ceil(delta[tt, i, s]) for tt in range(t + 1)))
            for t in range(1, self.numStages) for i in range(self.numFacilities)) + self.lambda_risk * sum(eta_TS[t, s] - eta_MS[t, s] for t in range(self.numStages - 1)) 
            for s in range(self.numScenarios))

            
        results['xValues'] = xVarValues
        results['yValues'] = yVarValues
        results['uValues'] = uVarValues
        results['etaValues'] = etaVarValues
        results['rental cost mean'] = sum(sum(xVarValues[tt, i, s] for tt in range(t + 1)) * self.maintenance_cost[i] * float(1 / self.numScenarios) 
                                                for t in range(self.numStages) for i in range(self.numFacilities) for s in range(self.numScenarios))
        results['transportation cost mean'] = sum(yVarValues[t, i, j, s] * self.transportation_cost[i, j] * float(1 / self.numScenarios)
                                                    for i in range(self.numFacilities) for j in range(self.numCustomers) for t in range(self.numStages) for s in range(self.numScenarios))

        results['cost profile'] = np.zeros([self.numStages + 1])
        results['cost mean'] = np.zeros([self.numStages + 1])
        for t in range(self.numStages):
            results['cost profile'][t] = np.percentile(sum(sum(xVarValues[tt, i, :] for tt in range(t + 1)) * self.maintenance_cost[i]
                                                 for i in range(self.numFacilities))+sum(yVarValues[t, i, j, :] * self.transportation_cost[i, j]
                                                    for i in range(self.numFacilities) for j in range(self.numCustomers)), 95)
            results['cost mean'][t] = np.mean(sum(sum(xVarValues[tt, i, :] for tt in range(t + 1)) * self.maintenance_cost[i]
                                                 for i in range(self.numFacilities))+sum(yVarValues[t, i, j, :] * self.transportation_cost[i, j]
                                                    for i in range(self.numFacilities) for j in range(self.numCustomers)))
        
        results['cost profile'][self.numStages] = np.percentile(sum(sum(xVarValues[tt, i, :] for tt in range(t + 1)) * self.maintenance_cost[i]
                                                 for i in range(self.numFacilities) for t in range(self.numStages)) + sum(yVarValues[t, i, j, :] * self.transportation_cost[i, j]
                                                    for i in range(self.numFacilities) for j in range(self.numCustomers) for t in range(self.numStages)), 95)
        results['cost mean'][self.numStages] = np.mean(sum(sum(xVarValues[tt, i, :] for tt in range(t + 1)) * self.maintenance_cost[i]
                                                 for i in range(self.numFacilities) for t in range(self.numStages))+ sum(yVarValues[t, i, j, :] * self.transportation_cost[i, j]
                                                    for i in range(self.numFacilities) for j in range(self.numCustomers) for t in range(self.numStages)))
        
        results['total cost profile'] = np.percentile(sum(sum(xVarValues[tt, i, :] for tt in range(t + 1)) * self.maintenance_cost[i]
                                                 for i in range(self.numFacilities) for t in range(self.numStages)) + sum(yVarValues[t, i, j, :] * self.transportation_cost[i, j]
                                                    for i in range(self.numFacilities) for j in range(self.numCustomers) for t in range(self.numStages)), 95)
    
        results['new bound'] = new_bound

        return results

    def solveRiskAverseMultistageModel2(self):
        
        st1 = time.time()
        m = gb.Model()
        xVar, yVar = {}, {}
        for t in range(self.numStages):
            for i in range(self.numFacilities):
                for s in range(self.numScenarios):
                    xVar[t, i, s] = m.addVar(lb = 0, vtype = gb.GRB.INTEGER)
                    for j in range(self.numCustomers):
                        yVar[t, i, j, s] = m.addVar(lb = 0, vtype = gb.GRB.CONTINUOUS)

        uVar, etaVar = {}, {}
        # add constraints
        for s in range(self.numScenarios):
            for t in range(self.numStages):
                if t < self.numStages - 1:
                    uVar[t, s] = m.addVar(lb = 0)
                    etaVar[t, s] = m.addVar()
                for i in range(self.numFacilities):
                    m.addConstr(sum(yVar[t, i, j, s] for j in range(self.numCustomers)) <= self.capacity[i] * sum(xVar[tt, i, s] for tt in range(t + 1)))
                for j in range(self.numCustomers):              
                    m.addConstr(sum(yVar[t, i, j, s] for i in range(self.numFacilities)) == (self.demandPath[t, j, s]))

        # nonanticipativity constraints
        for t in range(self.numStages - 1):
            for s in range(self.numScenarios):
                f = self.numBranches ** (self.numStages - t - 1)
                flag = np.floor(s / f)
                m.addConstr(etaVar[t, s] == etaVar[t, int(flag * f)])
                for i in range(self.numFacilities):
                    m.addConstr(xVar[t, i, s] == xVar[t, i, int(flag * f)])
                    for j in range(self.numCustomers):
                        m.addConstr(yVar[t, i, j, s] == yVar[t, i, j, int(flag * f)])

                if t < self.numStages - 2:
                    f = self.numBranches ** (self.numStages - t - 2)
                    flag = np.floor(s / f)
                    m.addConstr(uVar[t, s] == uVar[t, int(flag * f)])

                m.addConstr(uVar[t, s] + etaVar[t, s] >= sum(sum(xVar[tt, i, s] for tt in range(t + 2)) * self.maintenance_cost[i] for i in range(self.numFacilities))
                            + sum(yVar[t + 1, i, j, s] * self.transportation_cost[i, j] for i in range(self.numFacilities) for j in range(self.numCustomers)))

        m.setObjective(float(1 / self.numScenarios) * sum(
                        sum(xVar[0, i, s] * self.maintenance_cost[i] for i in range(self.numFacilities)) + sum(yVar[0, i, j, s] * self.transportation_cost[i, j] for i in range(self.numFacilities) for j in range(self.numCustomers))
                        + self.lambda_risk * sum(etaVar[t, s] for t in range(self.numStages - 1))
                        + self.lambda_risk / (1 - self.alpha_risk) * sum(uVar[t, s] for t in range(self.numStages - 1))
                        + (1 - self.lambda_risk) * sum(sum(xVar[tt, i, s] for tt in range(t + 1)) * self.maintenance_cost[i] for t in range(1, self.numStages) for i in range(self.numFacilities))
                        + (1 - self.lambda_risk) * sum(yVar[t, i, j, s] * self.transportation_cost[i, j] for t in range(1, self.numStages) for i in range(self.numFacilities) for j in range(self.numCustomers))
                        for s in range(self.numScenarios)))

        
        m.setParam('TimeLimit', 3600)
        m.setParam('OutputFlag', 0)
        m.setParam("MIPGap", 0.005)
        # m.update()
        st2 = time.time()
        m.optimize()
        st3 = time.time()

        results = {}
        results['optimal objective'] = m.objval
        results['construction time'] = st2 - st1
        results['optimization time'] = st3 - st2
        results['MIP gap'] = m.MIPGap
        xVarValues = np.zeros([self.numStages, self.numFacilities, self.numScenarios])
        yVarValues = np.zeros([self.numStages, self.numFacilities, self.numCustomers, self.numScenarios])
        uVarValues = np.zeros([self.numStages - 1, self.numScenarios])
        etaVarValues = np.zeros([self.numStages - 1, self.numScenarios])

        for t in range(self.numStages):
            for i in range(self.numFacilities):
                for s in range(self.numScenarios):
                    xVarValues[t, i, s] = xVar[t, i, s].x
                    for j in range(self.numCustomers):
                        yVarValues[t, i, j, s] = yVar[t, i, j, s].x
                        if t < self.numStages - 1:
                            uVarValues[t, s] = uVar[t, s].x
                            etaVarValues[t, s] = etaVar[t, s].x

            
        results['xValues'] = xVarValues
        results['yValues'] = yVarValues
        results['uValues'] = uVarValues
        results['etaValues'] = etaVarValues

        results['rental cost mean'] = sum(sum(xVarValues[tt, i, s] for tt in range(t + 1)) * self.maintenance_cost[i] * float(1 / self.numScenarios) 
                                                for t in range(self.numStages) for i in range(self.numFacilities) for s in range(self.numScenarios))
        results['transportation cost mean'] = sum(yVarValues[t, i, j, s] * self.transportation_cost[i, j] * float(1 / self.numScenarios)
                                                    for i in range(self.numFacilities) for j in range(self.numCustomers) for t in range(self.numStages) for s in range(self.numScenarios))

        results['cost profile'] = np.zeros([self.numStages + 1])
        results['cost mean'] = np.zeros([self.numStages + 1])
        for t in range(self.numStages):
            results['cost profile'][t] = np.percentile(sum(sum(xVarValues[tt, i, :] for tt in range(t + 1)) * self.maintenance_cost[i]
                                                 for i in range(self.numFacilities))+sum(yVarValues[t, i, j, :] * self.transportation_cost[i, j]
                                                    for i in range(self.numFacilities) for j in range(self.numCustomers)), 95)
            results['cost mean'][t] = np.mean(sum(sum(xVarValues[tt, i, :] for tt in range(t + 1)) * self.maintenance_cost[i]
                                                 for i in range(self.numFacilities))+sum(yVarValues[t, i, j, :] * self.transportation_cost[i, j]
                                                    for i in range(self.numFacilities) for j in range(self.numCustomers)))
        
        results['cost profile'][self.numStages] = np.percentile(sum(sum(xVarValues[tt, i, :] for tt in range(t + 1)) * self.maintenance_cost[i]
                                                 for i in range(self.numFacilities) for t in range(self.numStages)) + sum(yVarValues[t, i, j, :] * self.transportation_cost[i, j]
                                                    for i in range(self.numFacilities) for j in range(self.numCustomers) for t in range(self.numStages)), 95)
        results['cost mean'][self.numStages] = np.mean(sum(sum(xVarValues[tt, i, :] for tt in range(t + 1)) * self.maintenance_cost[i]
                                                 for i in range(self.numFacilities) for t in range(self.numStages)) + sum(yVarValues[t, i, j, :] * self.transportation_cost[i, j]
                                                    for i in range(self.numFacilities) for j in range(self.numCustomers) for t in range(self.numStages)))
        
        results['total cost profile'] = np.percentile(sum(sum(xVarValues[tt, i, :] for tt in range(t + 1)) * self.maintenance_cost[i]
                                                 for i in range(self.numFacilities) for t in range(self.numStages)) + sum(yVarValues[t, i, j, :] * self.transportation_cost[i, j]
                                                    for i in range(self.numFacilities) for j in range(self.numCustomers) for t in range(self.numStages)), 95)
    

        return results

    def solveLPRiskAverseMultistageModel2(self):
        
        st1 = time.time()
        m = gb.Model()
        xVar, yVar = {}, {}
        for t in range(self.numStages):
            for i in range(self.numFacilities):
                for s in range(self.numScenarios):
                    xVar[t, i, s] = m.addVar(lb = 0, vtype = gb.GRB.CONTINUOUS)
                    for j in range(self.numCustomers):
                        yVar[t, i, j, s] = m.addVar(lb = 0, vtype = gb.GRB.CONTINUOUS)

        uVar, etaVar = {}, {}
        # add constraints
        for s in range(self.numScenarios):
            for t in range(self.numStages):
                if t < self.numStages - 1:
                    uVar[t, s] = m.addVar(lb = 0)
                    etaVar[t, s] = m.addVar()
                for i in range(self.numFacilities):
                    m.addConstr(sum(yVar[t, i, j, s] for j in range(self.numCustomers)) <= self.capacity[i] * sum(xVar[tt, i, s] for tt in range(t + 1)))
                for j in range(self.numCustomers):              
                    m.addConstr(sum(yVar[t, i, j, s] for i in range(self.numFacilities)) == (self.demandPath[t, j, s]))

        # nonanticipativity constraints
        for t in range(self.numStages - 1):
            for s in range(self.numScenarios):
                f = self.numBranches ** (self.numStages - t - 1)
                flag = np.floor(s / f)
                m.addConstr(etaVar[t, s] == etaVar[t, int(flag * f)])
                for i in range(self.numFacilities):
                    m.addConstr(xVar[t, i, s] == xVar[t, i, int(flag * f)])
                    for j in range(self.numCustomers):
                        m.addConstr(yVar[t, i, j, s] == yVar[t, i, j, int(flag * f)])

                if t < self.numStages - 2:
                    f = self.numBranches ** (self.numStages - t - 2)
                    flag = np.floor(s / f)
                    m.addConstr(uVar[t, s] == uVar[t, int(flag * f)])

                m.addConstr(uVar[t, s] + etaVar[t, s] >= sum(sum(xVar[tt, i, s] for tt in range(t + 2)) * self.maintenance_cost[i] for i in range(self.numFacilities))
                            + sum(yVar[t + 1, i, j, s] * self.transportation_cost[i, j] for i in range(self.numFacilities) for j in range(self.numCustomers)))

        m.setObjective(float(1 / self.numScenarios) * sum(
                        sum(xVar[0, i, s] * self.maintenance_cost[i] for i in range(self.numFacilities)) + sum(yVar[0, i, j, s] * self.transportation_cost[i, j] for i in range(self.numFacilities) for j in range(self.numCustomers))
                        + self.lambda_risk * sum(etaVar[t, s] for t in range(self.numStages - 1))
                        + self.lambda_risk / (1 - self.alpha_risk) * sum(uVar[t, s] for t in range(self.numStages - 1))
                        + (1 - self.lambda_risk) * sum(sum(xVar[tt, i, s] for tt in range(t + 1)) * self.maintenance_cost[i] for t in range(1, self.numStages) for i in range(self.numFacilities))
                        + (1 - self.lambda_risk) * sum(yVar[t, i, j, s] * self.transportation_cost[i, j] for t in range(1, self.numStages) for i in range(self.numFacilities) for j in range(self.numCustomers))
                        for s in range(self.numScenarios)))

        
        m.setParam("TimeLimit", 1800)
        m.setParam("OutputFlag", 0)
        m.setParam("MIPGap", 0.005)
        m.update()
        st2 = time.time()
        m.optimize()
        st3 = time.time()

        results = {}
        results['optimal objective'] = m.objval
        results['construction time'] = st2 - st1
        results['optimization time'] = st3 - st2
        xVarValues = np.zeros([self.numStages, self.numFacilities, self.numScenarios])
        yVarValues = np.zeros([self.numStages, self.numFacilities, self.numCustomers, self.numScenarios])
        uVarValues = np.zeros([self.numStages - 1, self.numScenarios])
        etaVarValues = np.zeros([self.numStages - 1, self.numScenarios])

        for t in range(self.numStages):
            for i in range(self.numFacilities):
                for s in range(self.numScenarios):
                    xVarValues[t, i, s] = xVar[t, i, s].x
                    for j in range(self.numCustomers):
                        yVarValues[t, i, j, s] = yVar[t, i, j, s].x
                        if t < self.numStages - 1:
                            uVarValues[t, s] = uVar[t, s].x
                            etaVarValues[t, s] = etaVar[t, s].x

        delta = np.zeros([self.numStages, self.numFacilities, self.numScenarios])
        x_MS = np.zeros([self.numStages, self.numFacilities, self.numScenarios]) 
        x_TS = np.zeros([self.numStages, self.numFacilities, self.numScenarios]) 
        v_ms = np.zeros([self.numFacilities])
        v_ts = np.zeros([self.numFacilities])
        for t in range(self.numStages):
            for s in range(self.numScenarios):
                for i in range(self.numFacilities):
                    delta[t, i, s] = np.round(sum(yVarValues[t, i, j, s] for j in range(self.numCustomers)) / self.capacity[i], 9)

        # delta[delta > 1] = 1
        delta[delta < 0] = 0

        for t in range(self.numStages):
            for s in range(self.numScenarios):
                for i in range(self.numFacilities):       
                    if t == 0:
                        x_MS[t, i, s] = delta[t, i, s]
                        x_TS[t, i, s] = np.ceil(delta[t, i, s])
                    else:
                        x_MS[t, i, s] = max(delta[tt, i, s] for tt in range(t + 1)) - max(delta[tt, i, s] for tt in range(t))
                        x_TS[t, i, s] = max(np.ceil(max(delta[tt, i, :])) for tt in range(t + 1)) - max(np.ceil(max(delta[tt, i, :])) for tt in range(t))



        eta_MS = np.zeros([self.numStages - 1, self.numScenarios])
        eta_TS = np.zeros([self.numStages - 1, self.numScenarios])
        for t in range(1, self.numStages):
            for s in range(self.numScenarios):
                eta_MS[t - 1, s] = max(sum(sum(x_MS[tt, i, ss] for tt in range(t + 1)) * self.maintenance_cost[i] for i in range(self.numFacilities))
                            + sum(yVarValues[t, i, j, ss] * self.transportation_cost[i, j] for i in range(self.numFacilities) for j in range(self.numCustomers))
                            - uVarValues[t - 1, ss] 
                            for ss in range(int(self.numBranches**(self.numStages - t) * np.floor(s/(self.numBranches**(self.numStages - t)))), int(self.numBranches**(self.numStages - t) * (1 + np.floor(s/(self.numBranches**(self.numStages - t))))))) 
                eta_TS[t - 1, s] = max(sum(sum(x_TS[tt, i, ss] for tt in range(t + 1)) * self.maintenance_cost[i] for i in range(self.numFacilities))
                            + sum(yVarValues[t, i, j, ss] * self.transportation_cost[i, j] for i in range(self.numFacilities) for j in range(self.numCustomers))
                            - uVarValues[t - 1, ss] 
                            for ss in range(int(self.numBranches**(self.numStages - t) * np.floor(s/(self.numBranches**(self.numStages - t)))), int(self.numBranches**(self.numStages - t) * (1 + np.floor(s/(self.numBranches**(self.numStages - t)))))))

        upper_bound = sum(self.maintenance_cost[i] * (x_TS[0, i, 0] - x_MS[0, i, 0]) for i in range(self.numFacilities)) + float(1 / self.numScenarios) * sum((1 - self.lambda_risk) * sum(self.maintenance_cost[i] * (max(np.ceil(max(delta[tt, i, :])) for tt in range(t + 1)) - max(delta[tt, i, s] for tt in range(t + 1)))
            for t in range(1, self.numStages) for i in range(self.numFacilities)) + self.lambda_risk * sum(eta_TS[t, s] - eta_MS[t, s] for t in range(self.numStages - 1)) 
            for s in range(self.numScenarios))
            
        results['xValues'] = xVarValues
        results['yValues'] = yVarValues
        results['uValues'] = uVarValues
        results['etaValues'] = etaVarValues
        results['upper bound'] = upper_bound

        results['total rental cost'] = sum(sum(xVarValues[tt, i, s] for tt in range(t + 1)) * self.maintenance_cost[i] * float(1 / self.numScenarios) 
                                                for t in range(self.numStages) for i in range(self.numFacilities) for s in range(self.numScenarios))
        results['total transportation cost'] = sum(yVarValues[t, i, j, s] * self.transportation_cost[i, j] * float(1 / self.numScenarios)
                                                    for i in range(self.numFacilities) for j in range(self.numCustomers) for t in range(self.numStages) for s in range(self.numScenarios))


        return results

    def solveRiskAverseMultistageModel2WithKnownYU(self, y, u):
        # with known y and u

        results = {}

        delta = np.zeros([self.numStages, self.numFacilities, self.numScenarios])
        x_MS = np.zeros([self.numStages, self.numFacilities, self.numScenarios]) 

        for t in range(self.numStages):
            for s in range(self.numScenarios):
                for i in range(self.numFacilities):
                    delta[t, i, s] = max(np.round(sum(y[t, i, j, s] for j in range(self.numCustomers)) / self.capacity[i], 9), 0)
                    if t == 0:
                        x_MS[t, i, s] = np.ceil(delta[t, i, s])
                    else:
                        x_MS[t, i, s] = max(np.ceil(delta[tt, i, s]) for tt in range(t + 1)) - max(np.ceil(delta[tt, i, s]) for tt in range(t))



        eta_hat = np.zeros([self.numStages - 1, self.numScenarios])
        for t in range(1, self.numStages):
            for s in range(self.numScenarios):
                eta_hat[t - 1, s] = max(sum(sum(x_MS[tt, i, ss] for tt in range(t + 1)) * self.maintenance_cost[i] for i in range(self.numFacilities))
                            + sum(y[t, i, j, ss] * self.transportation_cost[i, j] for i in range(self.numFacilities) for j in range(self.numCustomers))
                            - u[t - 1, ss] 
                            for ss in range(int(self.numBranches**(self.numStages - t) * np.floor(s/(self.numBranches**(self.numStages - t)))), int(self.numBranches**(self.numStages - t) * (1 + np.floor(s/(self.numBranches**(self.numStages - t)))))))
        

        Q_M = float(1 / self.numScenarios) * sum(self.lambda_risk * sum(eta_hat[t - 1, s]
                            for t in range(1, self.numStages)) + sum(x_MS[0, i, s] * self.maintenance_cost[i] for i in range(self.numFacilities))
                            + (1 - self.lambda_risk) * sum(sum(x_MS[tt, i, s] for tt in range(t + 1)) * self.maintenance_cost[i] for t in range(1, self.numStages) for i in range(self.numFacilities))
                            for s in range(self.numScenarios))

            
        results['xValues'] = x_MS
        results['etaValues'] = eta_hat
        results['optimal objective'] = Q_M

        results['total rental cost'] = sum(sum(x_MS[tt, i, s] for tt in range(t + 1)) * self.maintenance_cost[i] * float(1 / self.numScenarios) 
                                                for t in range(self.numStages) for i in range(self.numFacilities) for s in range(self.numScenarios))


        return results

    def solveRiskAverseMultistageModel2WithKnownXEta(self, x, eta):
        
        st1 = time.time()
        m = gb.Model()
        yVar = {}
        for t in range(self.numStages):
            for i in range(self.numFacilities):
                for s in range(self.numScenarios):
                    for j in range(self.numCustomers):
                        yVar[t, i, j, s] = m.addVar(lb = 0, vtype = gb.GRB.CONTINUOUS)

        uVar = {}
        # add constraints
        for s in range(self.numScenarios):
            for t in range(self.numStages):
                if t < self.numStages - 1:
                    uVar[t, s] = m.addVar(lb = 0)
                for i in range(self.numFacilities):
                    m.addConstr(sum(yVar[t, i, j, s] for j in range(self.numCustomers)) <= self.capacity[i] * sum(x[tt, i, s] for tt in range(t + 1)))
                for j in range(self.numCustomers):              
                    m.addConstr(sum(yVar[t, i, j, s] for i in range(self.numFacilities)) == (self.demandPath[t, j, s]))

        # nonanticipativity constraints
        for t in range(self.numStages - 1):
            for s in range(self.numScenarios):
                f = self.numBranches ** (self.numStages - t - 1)
                flag = np.floor(s / f)
                for i in range(self.numFacilities):
                    for j in range(self.numCustomers):
                        m.addConstr(yVar[t, i, j, s] == yVar[t, i, j, int(flag * f)])

                if t < self.numStages - 2:
                    f = self.numBranches ** (self.numStages - t - 2)
                    flag = np.floor(s / f)
                    m.addConstr(uVar[t, s] == uVar[t, int(flag * f)])

                m.addConstr(uVar[t, s] + eta[t, s] >= sum(sum(x[tt, i, s] for tt in range(t + 2)) * self.maintenance_cost[i] for i in range(self.numFacilities))
                            + sum(yVar[t + 1, i, j, s] * self.transportation_cost[i, j] for i in range(self.numFacilities) for j in range(self.numCustomers)))

        m.setObjective(float(1 / self.numScenarios) * sum(
                        sum(yVar[0, i, j, s] * self.transportation_cost[i, j] for i in range(self.numFacilities) for j in range(self.numCustomers))
                        + self.lambda_risk / (1 - self.alpha_risk) * sum(uVar[t, s] for t in range(self.numStages - 1))
                        + (1 - self.lambda_risk) * sum(yVar[t, i, j, s] * self.transportation_cost[i, j] for t in range(1, self.numStages) for i in range(self.numFacilities) for j in range(self.numCustomers))
                        for s in range(self.numScenarios)))

        
        m.setParam("TimeLimit", 1800)
        m.setParam("OutputFlag", 0)
        m.setParam("MIPGap", 0.005)
        m.update()
        st2 = time.time()
        m.optimize()
        st3 = time.time()

        results = {}
        results['optimal objective'] = m.objval
        results['construction time'] = st2 - st1
        results['optimization time'] = st3 - st2
        yVarValues = np.zeros([self.numStages, self.numFacilities, self.numCustomers, self.numScenarios])
        uVarValues = np.zeros([self.numStages - 1, self.numScenarios])

        for t in range(self.numStages):
            for i in range(self.numFacilities):
                for s in range(self.numScenarios):
                    for j in range(self.numCustomers):
                        yVarValues[t, i, j, s] = yVar[t, i, j, s].x
                        if t < self.numStages - 1:
                            uVarValues[t, s] = uVar[t, s].x

            
        results['yValues'] = yVarValues
        results['uValues'] = uVarValues

        results['total transportation cost'] = sum(yVarValues[t, i, j, s] * self.transportation_cost[i, j] * float(1 / self.numScenarios)
                                                    for i in range(self.numFacilities) for j in range(self.numCustomers) for t in range(self.numStages) for s in range(self.numScenarios))


        return results

    def approximateRiskAverseMultistageModel2(self):

        st = time.time()
        results = self.solveLPRiskAverseMultistageModel2()
        ed1 = time.time()
        objective = []
        rental_cost = []
        transportation_cost = []
        results_y = results
        xValues = [results['xValues']]
        yValues = [results['yValues']]
        st2 = time.time()
        results_x = self.solveRiskAverseMultistageModel2WithKnownYU(results_y['yValues'], results_y['uValues'])
        ed2 = time.time()
        xValues.append(results_x['xValues'])
        st3 = time.time()
        results_y = self.solveRiskAverseMultistageModel2WithKnownXEta(results_x['xValues'], results_x['etaValues'])
        ed3 = time.time()
        yValues.append(results_y['yValues'])
        rental_cost.append(results_x['total rental cost'])
        transportation_cost.append(results_y['total transportation cost'])
        obj = (float(1 / self.numScenarios) * sum(
                        sum(results_x['xValues'][0, i, s] * self.maintenance_cost[i] for i in range(self.numFacilities)) + sum(results_y['yValues'][0, i, j, s] * self.transportation_cost[i, j] for i in range(self.numFacilities) for j in range(self.numCustomers))
                        + self.lambda_risk * sum(results_x['etaValues'][t, s] for t in range(self.numStages - 1))
                        + self.lambda_risk / (1 - self.alpha_risk) * sum(results_y['uValues'][t, s] for t in range(self.numStages - 1))
                        + (1 - self.lambda_risk) * sum(sum(results_x['xValues'][tt, i, s] for tt in range(t + 1)) * self.maintenance_cost[i] for t in range(1, self.numStages) for i in range(self.numFacilities))
                        + (1 - self.lambda_risk) * sum(results_y['yValues'][t, i, j, s] * self.transportation_cost[i, j] for t in range(1, self.numStages) for i in range(self.numFacilities) for j in range(self.numCustomers))
                        for s in range(self.numScenarios)))
        objective.append(obj)

        while (np.max(abs(xValues[-1] - xValues[-2])) > 1e-1 or np.max(abs(yValues[-1] - yValues[-2])) > 1e-1):
            st4 = time.time()
            results_x = self.solveRiskAverseMultistageModel2WithKnownYU(results_y['yValues'], results_y['uValues'])
            ed4 = time.time()
            xValues.append(results_x['xValues'])
            st5 = time.time()
            results_y = self.solveRiskAverseMultistageModel2WithKnownXEta(results_x['xValues'], results_x['etaValues'])
            ed5 = time.time()
            yValues.append(results_y['yValues'])
            rental_cost.append(results_x['total rental cost'])
            transportation_cost.append(results_y['total transportation cost'])
            obj = (float(1 / self.numScenarios) * sum(
                        sum(results_x['xValues'][0, i, s] * self.maintenance_cost[i] for i in range(self.numFacilities)) + sum(results_y['yValues'][0, i, j, s] * self.transportation_cost[i, j] for i in range(self.numFacilities) for j in range(self.numCustomers))
                        + self.lambda_risk * sum(results_x['etaValues'][t, s] for t in range(self.numStages - 1))
                        + self.lambda_risk / (1 - self.alpha_risk) * sum(results_y['uValues'][t, s] for t in range(self.numStages - 1))
                        + (1 - self.lambda_risk) * sum(sum(results_x['xValues'][tt, i, s] for tt in range(t + 1)) * self.maintenance_cost[i] for t in range(1, self.numStages) for i in range(self.numFacilities))
                        + (1 - self.lambda_risk) * sum(results_y['yValues'][t, i, j, s] * self.transportation_cost[i, j] for t in range(1, self.numStages) for i in range(self.numFacilities) for j in range(self.numCustomers))
                        for s in range(self.numScenarios)))
            objective.append(obj)
            

        ed = time.time()
        approx_time = ed - st
        return objective, rental_cost, transportation_cost, approx_time, xValues, yValues

    def solveRiskAverseMultistageModel_1SDDiP(self):

        st = time.time()
        model = MSIP(T = self.numStages, sense = 1)
        for t in range(self.numStages):
            m = model[t]
            if t == 0:
                ifopen_now, ifopen_past = m.addStateVars(self.numFacilities, name = "invested", obj = self.maintenance_cost, vtype = 'I')
                eta_now, eta_past = m.addStateVar(name = "eta", obj = self.lambda_risk, vtype = 'C')
                transportation = m.addVars(self.numFacilities, self.numCustomers, name = "transportation", lb = 0, obj = self.transportation_cost, vtype = 'C')            
            elif t == self.numStages - 1:
                ifopen_now, ifopen_past = m.addStateVars(self.numFacilities, name = "invested", obj = (1 - self.lambda_risk) * self.maintenance_cost, vtype = 'I')
                eta_now, eta_past = m.addStateVar(name = "eta", vtype = 'C')
                u = m.addVar(name = "u", lb = 0, obj = self.lambda_risk / (1 - self.alpha_risk), vtype = 'C')
                transportation = m.addVars(self.numFacilities, self.numCustomers, name = "transportation", lb = 0, obj = (1 - self.lambda_risk) * self.transportation_cost, vtype = 'C')            
            else:
                ifopen_now, ifopen_past = m.addStateVars(self.numFacilities, name = "invested", obj = (1 - self.lambda_risk) * self.maintenance_cost, vtype = 'I')
                eta_now, eta_past = m.addStateVar(name = "eta", obj = self.lambda_risk, vtype = 'C')
                u = m.addVar(name = "u", lb = 0, obj = self.lambda_risk / (1 - self.alpha_risk), vtype = 'C')
                transportation = m.addVars(self.numFacilities, self.numCustomers, name = "transportation", lb = 0, obj = (1 - self.lambda_risk) * self.transportation_cost, vtype = 'C')            

            # add constraints in each stage
            for i in range(self.numFacilities):
                m.addConstr(sum(transportation[i, j] for j in range(self.numCustomers)) <= self.capacity[i] * ifopen_now[i])
                m.addConstr(ifopen_now[i] >= ifopen_past[i])

            if t >= 1:
                for j in range(self.numCustomers):
                    m.addConstr(sum(transportation[i, j] for i in range(self.numFacilities)) == 0, uncertainty = {'rhs': self.demandNode[t, j, :]})
                m.addConstr(u + eta_past >= sum(ifopen_now[i] for i in range(self.numFacilities)) * self.maintenance_cost[0]
                        + sum(transportation[i, j] * self.transportation_cost[i, j] for i in range(self.numFacilities) for j in range(self.numCustomers)))
            else:
                for j in range(self.numCustomers):
                    m.addConstr(sum(transportation[i, j] for i in range(self.numFacilities)) == self.demandMean[t, j])

        # extensive = Extensive(model)
        # extensive.solve()
        # extensive.first_stage_solution
        model_sddip = SDDiP(model)
        max_iter = 10
        # model_sddip.solve(cuts=['B','SB','LG'], pattern={'cycle':[3,3,4]}, max_iterations = max_iter, logToConsole = 1)
        model_sddip.solve(cuts=['B','LG'], 
                          pattern={'cycle':[6,4]},
                          max_iterations = max_iter, 
                        #   level_max_stable_iterations=10, 
                          level_max_iterations=100, 
                        #   level_max_time=10, 
                          level_mip_gap=0.01, 
                          level_tol=0.01, 
                          logToConsole = 1)
        end = time.time()
        results = {}
        results['bound'] = model_sddip.db[-1]
        results['first stage solution'] = model_sddip.first_stage_solution
        results['time'] = end - st

        return results


