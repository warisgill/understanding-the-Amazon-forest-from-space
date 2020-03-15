from scipy.optimize import fmin_l_bfgs_b, basinhopping
from sklearn.metrics import fbeta_score, accuracy_score
from functools import partial
import numpy as np

def fbeta(true_label, prediction):
    return fbeta_score(true_label, prediction, beta=2, average='samples')


def get_optimal_threshhold(true_label, prediction, iterations = 100):

    best_threshholds = [0.2]*17    
    for t in range(17):
        best_fbeta = 0
        temp_threshhold = [0.2]*17
        for i in range(iterations):
            temp_value = i / float(iterations)
            temp_threshhold[t] = temp_value
            temp_fbeta = fbeta(true_label, prediction > temp_threshhold)
            print(temp_fbeta)
            if  temp_fbeta>best_fbeta:
                best_fbeta = temp_fbeta
                best_threshholds[t] = temp_value
    return best_threshholds


def bounds(**kwargs):
    x = kwargs["x_new"]
    tmax = bool(np.all(x <= 1))
    tmin = bool(np.all(x >= 0)) 
    return tmax and tmin

def f_neg(threshold, predictions, true_labels):
    return - fbeta_score(true_labels, predictions > threshold, beta=2, average='samples')

def best_f2_score(true_labels, predictions):
    
    temp_threshold = [0.20] * 17
    
    minimizer_kwargs = {"method": "L-BFGS-B", "bounds":[(0.,1.)] * 17, "options":{"eps": 0.05}}

    fg = partial(f_neg, true_labels = true_labels, predictions = predictions)
    
    opt_output = basinhopping(fg, temp_threshold,
                            stepsize = 0.1,
                            minimizer_kwargs=minimizer_kwargs,
                            niter=10,
                            accept_test=bounds)
    
    return opt_output.x