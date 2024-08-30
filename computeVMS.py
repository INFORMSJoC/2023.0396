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
Compute the VMS, relative VMS, and its lower and upper bounds, using synthetic dataset
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
        rgap_risk2 = np.zeros(iteration)
        gap_risk2 = np.zeros(iteration)

        lower_bound_risk = np.zeros(iteration)
        upper_bound_risk = np.zeros(iteration)

        two_time_list = []
        multistage_time_list = []
        two_MIP_gap = []
        multistage_MIP_gap = []
        two_xValues, two_yValues,multistage_xValues, multistage_yValues = [],[],[],[]
        
        for iter in range(iteration):

            Test.generateSyntheticDataNew()
            Test.generateScenariosNonSAA()

            results_two_risk_2 = Test.solveRiskAverseTwoStageModel2()
            results_two_risk2[iter] = results_two_risk_2['optimal objective']
            two_MIP_gap.append(results_two_risk_2['MIP gap'])
            two_xValues.append(results_two_risk_2['xValues'])
            two_yValues.append(results_two_risk_2['yValues'])
            two_time_list.append(results_two_risk_2['optimization time'] + results_two_risk_2['construction time'])
            lower_bound_risk[iter] = results_two_risk_2['new bound']
            
            results_multistage_risk_2 = Test.solveRiskAverseMultistageModel2()
            results_multistage_risk2[iter] = results_multistage_risk_2['optimal objective']
            multistage_MIP_gap.append(results_multistage_risk_2['MIP gap'])
            multistage_xValues.append(results_multistage_risk_2['xValues'])
            multistage_yValues.append(results_multistage_risk_2['yValues'])
            multistage_time_list.append(results_multistage_risk_2['optimization time'] + results_multistage_risk_2['construction time'])

            results_multistage_risk2_LP = Test.solveLPRiskAverseMultistageModel2()
            upper_bound_risk[iter] = results_multistage_risk2_LP['upper bound']

            gap_risk2[iter] = (results_two_risk_2['optimal objective'] - results_multistage_risk_2['optimal objective'])
            rgap_risk2[iter] = (results_two_risk_2['optimal objective'] - results_multistage_risk_2['optimal objective']) / results_two_risk_2['optimal objective']
            
            
            print("iteration " + str(iter) + " VMS: ", results_two_risk_2['optimal objective'] - results_multistage_risk_2['optimal objective'])     
            print("iteration " + str(iter) + " relative VMS: ", rgap_risk2[iter])

            print("iteration " + str(iter) + " lower bound to the VMS: ", lower_bound_risk[iter])    
            print("iteration " + str(iter) + " lower bound to the relative VMS: ", lower_bound_risk[iter] / results_two_risk_2['optimal objective'])  

            print("iteration " + str(iter) + " upper bound to the VMS: ", upper_bound_risk[iter])  
            print("iteration " + str(iter) + " upper bound to the relative VMS: ", upper_bound_risk[iter] / results_two_risk_2['optimal objective'])  


        results = {}
        results['multistage xValues'] = multistage_xValues
        results['multistage yValues'] = multistage_yValues
        results['two-stage xValues'] = two_xValues
        results['two-stage yValues'] = two_yValues
        results['multistage MIP gap'] = multistage_MIP_gap
        results['two-stage MIP gap'] = two_MIP_gap
        
        results['risk averse two-stage list'] = results_two_risk2
        results['risk averse multistage list'] = results_multistage_risk2

        results['two-stage opt time'] = two_time_list
        results['multistage opt time'] = multistage_time_list

        results['gaps'] = gap_risk2
        results['relative gaps'] = rgap_risk2
        results['lower bounds'] = lower_bound_risk
        results['upper bounds'] = upper_bound_risk

        results['gap in bounds'] = gap_risk2 - lower_bound_risk

        results['95 percentiles in gap of bounds'] = np.percentile(results['gap in bounds'], 95)
        results['90 percentiles in gap of bounds'] = np.percentile(results['gap in bounds'], 90)
        results['75 percentiles in gap of bounds'] = np.percentile(results['gap in bounds'], 75)
        results['50 percentiles in gap of bounds'] = np.percentile(results['gap in bounds'], 50)
        results['mean in gap of bounds'] = np.mean(results['gap in bounds'])

        results['95 percentiles in relative gaps'] = np.percentile(results['relative gaps'], 95)
        results['90 percentiles in relative gaps'] = np.percentile(results['relative gaps'], 90)
        results['75 percentiles in relative gaps'] = np.percentile(results['relative gaps'], 75)
        results['50 percentiles in relative gaps'] = np.percentile(results['relative gaps'], 50)
        results['mean in relative gaps'] = np.mean(results['relative gaps'])




