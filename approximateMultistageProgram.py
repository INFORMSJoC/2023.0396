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
Approximate multistage stochastic program using the proposed approximation algorithm
compare it with SDDiP
'''

import numpy as np
from numpy import newaxis
import pandas as pd
import gurobipy as gb
import scipy.stats as stats
import time
import pickle
from haversine import haversine, haversine_vector, Unit
from scripts.BoundStochasticProgram import *

if __name__ == "__main__":

    numStages = 3
    numFacilities = 5
    numCustomers = 10
    numBranches = 2
    lambda_risk = 0.5
    alpha_risk = 0.95
    sigma = 0.8
    outScenarios = 1

    for lambda_risk in [0.5]:
        print("stages: ", numStages)
        Test = BoundStochasticProgram(numStages, numFacilities, numCustomers, numBranches, sigma, outScenarios, lambda_risk = lambda_risk, alpha_risk = alpha_risk)

        iteration = 10
        results_two_risk2 = np.zeros(iteration)
        results_multistage_risk2 = np.zeros(iteration)
        results_riskaverse_sddip_bound = np.zeros(iteration)

        two_time_list = []
        multistage_time_list = []
        sddip_time_list = []
        two_MIP_gap = []
        multistage_MIP_gap = []
        multi_approx_obj = []
        multi_approx_time = []

        
        for iter in range(iteration):

            Test.generateSyntheticDataNew()
            Test.generateScenariosNonSAA()

            multi_obj, rental_cost, transportation_cost, multi_time, multi_xValues_approx, multi_yValues_approx = Test.approximateRiskAverseMultistageModel2()
            multi_approx_obj.append(multi_obj)
            multi_approx_time.append(multi_time)

            results_multistage_risk_2 = Test.solveRiskAverseMultistageModel2()
            results_multistage_risk2[iter] = results_multistage_risk_2['optimal objective']
            multistage_MIP_gap.append(results_multistage_risk_2['MIP gap'])
            multistage_time_list.append(results_multistage_risk_2['optimization time'] + results_multistage_risk_2['construction time'])

            
            results_two_risk_2 = Test.solveRiskAverseTwoStageModel2()
            results_two_risk2[iter] = results_two_risk_2['optimal objective']
            two_MIP_gap.append(results_two_risk_2['MIP gap'])
            two_time_list.append(results_two_risk_2['optimization time'] + results_two_risk_2['construction time'])

            results_riskaverse_sddip = Test.solveRiskAverseMultistageModel_1SDDiP()
            results_riskaverse_sddip_bound[iter] = results_riskaverse_sddip['bound']
            sddip_time_list.append(results_riskaverse_sddip['time'])

            print("iteration " + str(iter) + " risk averse multistage obj: ", results_multistage_risk_2['optimal objective'])
            print("iteration " + str(iter) + " risk averse multistage time: ", results_multistage_risk_2['optimization time'] + results_multistage_risk_2['construction time'])

            print("iteration " + str(iter) + " approximate risk averse multistage obj: ", multi_obj[-1])
            print("iteration " + str(iter) + " approximate risk averse multistage time: ", multi_time)

            print("iteration " + str(iter) + " risk averse two-stage obj: ", results_two_risk_2['optimal objective'])
            print("iteration " + str(iter) + " risk averse two-stage time: ", results_two_risk_2['optimization time'] + results_two_risk_2['construction time'])
            
            print("iteration " + str(iter) + " SDDiP bound: ", results_riskaverse_sddip['bound'])
            print("iteration " + str(iter) + " SDDiP time: ", results_riskaverse_sddip['time'])




        results = {}
        results['sddip bound'] = results_riskaverse_sddip_bound
        results['sddip time'] = sddip_time_list

        results['multistage approx obj'] = multi_approx_obj
        results['multistage approx time'] = multi_approx_time

        
        results['multistage obj'] = results_multistage_risk2
        results['multistage time'] = multistage_time_list

        results['two-stage obj'] = results_two_risk2
        results['two-stage time'] = two_time_list

        output = open('results/result_{}_{}_{}_{}_{}_{}_{}_{}_NonSAA.pkl'.format(Test.numStages, Test.numFacilities, Test.numCustomers, Test.numBranches, sigma, lambda_risk, alpha_risk, iteration), 'wb')
        pickle.dump(results, output)
        output.close()

