from matplotlib import pyplot as plt
import numpy as np
import math
import numpy as np
import pandas as pd
import time
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns

sys.path.append("/Users/arielhannum/Documents/GitHub/gropt/python")
import gropt
from helper_utils import *


def calc_gropt(gropt_params,target_b,start_TE, tol, iterations=50, initial_learning_rate = 0.1,\
    max_learning_rate = 0.1,momentum = 0.8, ep = 1):
    gropt_params['TE'] = start_TE
    params = gropt_params.copy()
    prev_gradient = 0

    te_track = []
    b_track = []
    
    for i in range(iterations):
        G_gropt, debug = gropt.gropt(gropt_params)
        gropt_b = get_bval(G_gropt, gropt_params)

        te_track.append(gropt_params['TE'])
        b_track.append(gropt_b)

        print('\tIteration {}: Gropt TE: {:.2f} ms, Gropt Bval: {:.2f}'.format(i, gropt_params['TE'], gropt_b))
        if abs(gropt_b - target_b) <= tol * target_b:
            print('\tWithin tolerance, break')
            break

        if gropt_b > target_b:
            epsilon = ep
            TE_perturbed = gropt_params['TE'] + epsilon
            params['TE'] = TE_perturbed
            G_perturbed, debug = gropt.gropt(params)
            bval_perturbed = get_bval(G_perturbed, params)
            gradient = (bval_perturbed - gropt_b) / epsilon
            # Use adaptive learning rate with a maximum limit
            learning_rate = initial_learning_rate / (1 + i)
            learning_rate = min(learning_rate, max_learning_rate)  # Clip learning rate

            update = momentum * prev_gradient + (1 - momentum) * gradient
            gropt_params['TE'] -= learning_rate * update
        elif gropt_b < target_b: 
            epsilon = ep
            TE_perturbed = gropt_params['TE'] + epsilon
            params['TE'] = TE_perturbed
            G_perturbed, debug = gropt.gropt(params)
            bval_perturbed = get_bval(G_perturbed, params)
            gradient = (bval_perturbed - gropt_b) / epsilon
            # Use adaptive learning rate with a maximum limit
            learning_rate = initial_learning_rate / (1 + i)
            learning_rate = min(learning_rate, max_learning_rate)  # Clip learning rate

            update = momentum * prev_gradient + (1 - momentum) * gradient
            gropt_params['TE'] += learning_rate * update


        prev_gradient = update

    return gropt_params, G_gropt, te_track, b_track

def calc_gropt_binary(gropt_params, target_b, start_TE, tol, iterations=50):
    # Initial range for binary search
    TE_lower = start_TE /2  # Adjust the initial range as needed
    TE_upper = start_TE 

    te_track = []
    b_track = []

    for i in range(iterations):
        # Binary search for the TE value
        gropt_params['TE'] = (TE_lower + TE_upper) / 2

        G_gropt, debug = gropt.gropt(gropt_params)
        #bval_params = gropt_params.copy()
        #bval_params['dt'] = gropt_params['dt_out']
        gropt_b = get_bval(G_gropt, gropt_params)

        te_track.append(gropt_params['TE'])
        b_track.append(gropt_b)

        print('\tIteration {}: Gropt TE: {:.5f} ms, Gropt Bval: {:.2f}'.format(i, gropt_params['TE'], gropt_b))

        # If within margin of error break
        if abs(gropt_b - target_b) <= tol * target_b:
            break
        
        # If TE is not changing, break out of iterations
        if len(te_track) > 3:
            if te_track[-2] - te_track[-1] <= 0.001:
                if b_track[i-1] > target_b:
                    gropt_params['TE'] = te_track[i-1]
                break

        if gropt_b < target_b:
            # Update the lower bound of the TE range
            TE_lower = gropt_params['TE']
        else:
            # Update the upper bound of the TE range
            TE_upper = gropt_params['TE']

    return gropt_params, G_gropt, te_track, b_track   


def calc_grop_hybrid(gropt_params, target_b, start_TE, target, iterations=50,):
    # Initial range for binary search
    binary_params, G_gropt, te_track, b_track = calc_gropt_binary(gropt_params.copy(), target_b, start_TE, target,iterations)

    count = 0
    if b_track[-1] < target_b:
        for ii in range(len(b_track)):
            if b_track[ii] > target_b: 
                start_TE = te_track[ii]
                count +=1


    out_params, G_gropt, te_track, b_track = calc_gropt(binary_params.copy(),target_b,binary_params['TE'], target, iterations,  initial_learning_rate = 0.05,\
        max_learning_rate = 0.1,momentum = 0.8, ep = 0.5)

    return out_params, G_gropt, te_track, b_track   


