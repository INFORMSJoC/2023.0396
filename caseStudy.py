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
Compute the optimal solutions and costs of two-stage and multistage models using real-world dataset from 
Daskin MS (2011) Network and Discrete Location: Models, Algorithms, and Applications
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

    numStages = 5
    numFacilities = 49
    numCustomers = 88
    numBranches = 2
    lambda_risk = 0.5
    alpha_risk = 0.95
    sigma = 0.8
    outScenarios = 1

    for lambda_risk in [0.5]:
        print("stages: ", numStages)
        Test = BoundStochasticProgram(numStages, numFacilities, numCustomers, numBranches, sigma, outScenarios, lambda_risk = lambda_risk, alpha_risk = alpha_risk)


        two_time_list = []
        multistage_time_list = []
        two_MIP_gap = []
        multistage_MIP_gap = []
        multi_approx_obj = []
        multi_approx_time = []
        two_xValues, two_yValues,multistage_xValues, multistage_yValues = [],[],[],[]
        multistage_xValues_approx, multistage_yValues_approx = [],[]
    

        Test.generateDataFromDaskinNonSAA('data/data49UFLP.xls', 'data/data88UFLP.xls')

        results_two_risk_2 = Test.solveRiskAverseTwoStageModel2()
        results_two_risk2 = results_two_risk_2['optimal objective']
        two_MIP_gap.append(results_two_risk_2['MIP gap'])
        two_xValues.append(results_two_risk_2['xValues'])
        two_yValues.append(results_two_risk_2['yValues'])
        two_time_list.append(results_two_risk_2['optimization time'] + results_two_risk_2['construction time'])
        lower_bound_risk = results_two_risk_2['new bound']
        
        results_multistage_risk_2 = Test.solveRiskAverseMultistageModel2()
        results_multistage_risk2 = results_multistage_risk_2['optimal objective']
        multistage_MIP_gap.append(results_multistage_risk_2['MIP gap'])
        multistage_xValues.append(results_multistage_risk_2['xValues'])
        multistage_yValues.append(results_multistage_risk_2['yValues'])
        multistage_time_list.append(results_multistage_risk_2['optimization time'] + results_multistage_risk_2['construction time'])

        

        results_multistage_risk2_LP = Test.solveLPRiskAverseMultistageModel2()
        upper_bound_risk = results_multistage_risk2_LP['upper bound']

        multi_obj, rental_cost, transportation_cost, multi_time, multi_xValues_approx, multi_yValues_approx = Test.approximateRiskAverseMultistageModel2()
        multi_approx_obj.append(multi_obj)
        multi_approx_time.append(multi_time)
        multistage_xValues_approx.append(multi_xValues_approx)
        multistage_yValues_approx.append(multi_yValues_approx)


        gap_risk2 = (results_two_risk_2['optimal objective'] - results_multistage_risk_2['optimal objective'])
        rgap_risk2 = (results_two_risk_2['optimal objective'] - results_multistage_risk_2['optimal objective']) / results_two_risk_2['optimal objective']
        
        
        print(" VMS: ", results_two_risk_2['optimal objective'] - results_multistage_risk_2['optimal objective'])     
        print(" relative VMS: ", rgap_risk2)
        print("multistage model maintenance cost mean: ", results_multistage_risk_2['rental cost mean'])
        print("multistage model transportation cost mean: ", results_multistage_risk_2['transportation cost mean'])
        print("two-stage model maintenance cost mean: ", results_two_risk_2['rental cost mean'])
        print("two-stage model transportation cost mean: ", results_two_risk_2['transportation cost mean'])
        print("multistage approx maintenance cost mean: ", rental_cost)
        print("multistage approx transportation cost mean: ", transportation_cost)


        results = {}
        results['multistage xValues'] = multistage_xValues
        results['multistage yValues'] = multistage_yValues
        results['two-stage xValues'] = two_xValues
        results['two-stage yValues'] = two_yValues
        results['multistage MIP gap'] = multistage_MIP_gap
        results['two-stage MIP gap'] = two_MIP_gap

        results['multistage approx obj'] = multi_approx_obj
        results['multistage xValues approx'] = multistage_xValues_approx
        results['multistage yValues approx'] = multistage_yValues_approx
        results['multistage approx rental cost'] = rental_cost
        results['multitsage approx trans cost'] = transportation_cost
        
        results['risk averse two-stage list'] = results_two_risk2
        results['risk averse multistage list'] = results_multistage_risk2

        results['risk averse multistage rental cost mean'] = results_multistage_risk_2['rental cost mean']
        results['risk averse multistage trans cost mean'] = results_multistage_risk_2['transportation cost mean']
        results['risk averse two-stage rental cost mean'] = results_two_risk_2['rental cost mean']
        results['risk averse two-stage trans cost mean'] = results_two_risk_2['transportation cost mean']


        results['two-stage opt time'] = two_time_list
        results['multistage opt time'] = multistage_time_list

        results['gaps'] = gap_risk2
        results['relative gaps'] = rgap_risk2
        results['lower bounds'] = lower_bound_risk
        results['upper bounds'] = upper_bound_risk

        results['gap in bounds'] = gap_risk2 - lower_bound_risk

