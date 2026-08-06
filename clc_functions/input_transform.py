### This script contains the transformations to be performed on the inputs 
### (and potentially on the output) to the classical and quantum NNs for
### predicting cloud cover.
### The transformations on the specific humidity ('hus' or 'qv'), 
### specific cloud water content ('clw' or 'qc'), specific cloud ice ('cli' or 'qi')
### and cloud cover ('clc') are based on fits to the cumulative distribution
### function empirically extracted from the training dataset. The transformations
### of the other features are then achieved via min-max scaling.
###
### Note that these transformations are computed AFTER the data has been balanced and 
### the zero-cloud-cover samples that are easy to predict (e.g., zero condensate)
### have been removed.


# Importing necessary packages
import numpy as np
import copy
from scipy.stats import gamma



### function for fitting empirical CDF of cloud cover:
### y = arcsin(2*(((exp(b*x)-1)/(exp(b)-1))^c - 0.5))/pi + 0.5
### inverse:
###
### np.pi * (y - 0.5) = np.arcsin(fx)
### np.sin(np.pi*(y-0.5)) = fx
### np.sin(np.pi*(y-0.5)) = 2.0 * (vx**c - 0.5)
### 0.5 * np.sin(np.pi*(y-0.5)) + 0.5 = vx**c
### (0.5 * np.sin(np.pi*(y-0.5)) + 0.5)**(1.0/c) = vx
### if np.abs(b)<0.001:
###     (0.5 * np.sin(np.pi*(y-0.5)) + 0.5)**(1.0/c) = xp
###     (0.5 * np.sin(np.pi*(y-0.5)) + 0.5)**(1.0/c) = x**a
###     ((0.5 * np.sin(np.pi*(y-0.5)) + 0.5)**(1.0/c))**(1.0/a) = x
### else:
###     (0.5 * np.sin(np.pi*(y-0.5)) + 0.5)**(1.0/c) = (np.exp(b * xp) - 1.0)/(np.exp(b) - 1.0)
###     (np.exp(b)-1.0) * (0.5 * np.sin(np.pi*(y-0.5)) + 0.5)**(1.0/c) + 1.0 = np.exp(b * xp)
###     np.log((np.exp(b)-1.0) * (0.5 * np.sin(np.pi*(y-0.5)) + 0.5)**(1.0/c) + 1.0) = b * xp
###     np.log((np.exp(b)-1.0) * (0.5 * np.sin(np.pi*(y-0.5)) + 0.5)**(1.0/c) + 1.0) / b = xp
###     np.log((np.exp(b)-1.0) * (0.5 * np.sin(np.pi*(y-0.5)) + 0.5)**(1.0/c) + 1.0) / b = x**a
###     (np.log((np.exp(b)-1.0) * (0.5 * np.sin(np.pi*(y-0.5)) + 0.5)**(1.0/c) + 1.0) / b)**(1.0/a) = x
###
def ecdf_fit_function_clc(x, a, b, c):
    xp = x**a
    if np.abs(b)<0.001:
        vx = xp
    else:
        vx = (np.exp(b * xp) - 1.0)/(np.exp(b) - 1.0)
    fx = 2.0 * (vx**c - 0.5)
    y = np.arcsin(fx)/np.pi + 0.5
    return y

def inverse_transform_function_clc(y, a, b, c):
    vx = (0.5 * np.sin(np.pi*(y - 0.5)) + 0.5)**(1.0/c)
    if np.abs(b)<0.001:
        x = vx**(1.0/a)
    else:
        gx = (np.exp(b) - 1.0) * vx
        xp = np.log(gx + 1.0) / b
        x = xp**(1.0/a)
    return x

### Function for transforming output (cloud cover)
def output_transform(output, bounds):
    a = 1.29407913
    b = -3.20011015
    c = 0.70308237
    transf_output = ecdf_fit_function_clc(np.squeeze(output), a, b, c)
    return transf_output

### Function for back-transforming output (cloud cover)
def inverse_transform_clc(transf_output):
    a = 1.29407913
    b = -3.20011015
    c = 0.70308237
    orig_output = inverse_transform_function_clc(transf_output, a, b, c)
    return orig_output




### function for fitting empirical CDF of specific water content
def ecdf_fit_function_water(x, a, b, s):
    xp = (x / s)**a
    y = gamma.cdf(xp, b, loc=0.0, scale=1.0)
    return y



### function for log transform of the inputs
### of the form 'log(1 + (e-1) * (x/xmax)^b)'
def log_transform_input(x, b, min_x, max_x):
    y0 = np.log(1.0 + (np.exp(1.0) - 1.0) * (min_x / max_x) ** b)
    y = (np.log(1.0 + (np.exp(1.0) - 1.0) * (x / max_x) ** b) - y0) / (1.0 - y0)
    return y



### Function for transforming all inputs
### 'inputs' is a (no_data, no_features) array
### 'feature_keys' the list of features according to which the colums in 'inputs' are ordered
### 'bounds' is a list of [min, max] pairs in which the inputs are rescaled
def inputs_transform(inputs, feature_keys, bounds):
    transf_inputs = copy.deepcopy(inputs)
    for key in feature_keys:
        ind = np.where(np.array([k==key for k in feature_keys]))[0][0]
        inputX = inputs[:, ind]
        boundsX = bounds[ind]
        if (key=='hus' or key=='qv'):
            b = 0.25; minX = 1.0e-07; maxX = 0.025
            outputX = log_transform_input(np.squeeze(inputX), b, minX, maxX)
            minX = boundsX[0]
            maxX = boundsX[1]
            outputX = (maxX - minX) * np.squeeze(outputX) + minX
            transf_inputs[:, ind] = outputX
        if (key=='clw' or key=='qc'):
            b = 0.25; minX = 0.0; maxX = 0.00145
            outputX = log_transform_input(np.squeeze(inputX), b, minX, maxX)
            minX = boundsX[0]
            maxX = boundsX[1]
            outputX = (maxX - minX) * np.squeeze(outputX) + minX
            transf_inputs[:, ind] = outputX
        if (key=='cli' or key=='qi'):
            b = 0.25; minX = 0.0; maxX = 0.00055
            outputX = log_transform_input(np.squeeze(inputX), b, minX, maxX)
            minX = boundsX[0]
            maxX = boundsX[1]
            outputX = (maxX - minX) * np.squeeze(outputX) + minX
            transf_inputs[:, ind] = outputX
        if (key=='clt' or key=='qt'):
            b = 0.25; minX = 1.0e-08; maxX = 0.00150
            outputX = log_transform_input(np.squeeze(inputX), b, minX, maxX)
            minX = boundsX[0]
            maxX = boundsX[1]
            outputX = (maxX - minX) * np.squeeze(outputX) + minX
            transf_inputs[:, ind] = outputX
        if (key=='ta' or key=='temp'):
            outputX = (np.squeeze(inputX) - 180.0)/(305.0 - 180.0)
            minX = boundsX[0]
            maxX = boundsX[1]
            outputX = (maxX - minX) * np.squeeze(outputX) + minX
            transf_inputs[:, ind] = outputX
        if (key=='pa' or key=='pres'):
            outputX = (np.squeeze(inputX) - 4000.0)/(105000.0 - 4000.0)
            minX = boundsX[0]
            maxX = boundsX[1]
            outputX = (maxX - minX) * np.squeeze(outputX) + minX
            transf_inputs[:, ind] = outputX
        if (key=='ua' or key=='u'):
            outputX = (np.squeeze(inputX) + 60.0)/(110.0 + 60.0)
            minX = boundsX[0]
            maxX = boundsX[1]
            outputX = (maxX - minX) * np.squeeze(outputX) + minX
            transf_inputs[:, ind] = outputX
        if (key=='va' or key=='v'):
            outputX = (np.squeeze(inputX) + 70.0)/(70.0 + 70.0)
            minX = boundsX[0]
            maxX = boundsX[1]
            outputX = (maxX - minX) * np.squeeze(outputX) + minX
            transf_inputs[:, ind] = outputX
        if (key=='hwind' or key=='hw'):
            b = 0.5; minX = 0.0015; maxX = 115.0
            outputX = log_transform_input(np.squeeze(inputX), b, minX, maxX)
            minX = boundsX[0]
            maxX = boundsX[1]
            outputX = (maxX - minX) * np.squeeze(outputX) + minX
            transf_inputs[:, ind] = outputX
        if (key=='zg'):
            outputX = (np.squeeze(inputX) - 0.0)/(20000.0 - 0.0)
            minX = boundsX[0]
            maxX = boundsX[1]
            outputX = (maxX - minX) * np.squeeze(outputX) + minX
            transf_inputs[:, ind] = outputX
        if (key=='coriolis'):
            Omega = 7.2921 * 10.0**(-5.0)
            outputX = (np.arcsin(0.5 * np.squeeze(inputX) / Omega) + np.pi/2.0)/np.pi
            minX = boundsX[0]
            maxX = boundsX[1]
            outputX = (maxX - minX) * np.squeeze(outputX) + minX
            transf_inputs[:, ind] = outputX
        if (key=='fr_land'):
            outputX = np.squeeze(inputX)
            minX = boundsX[0]
            maxX = boundsX[1]
            outputX = (maxX - minX) * np.squeeze(outputX) + minX
            transf_inputs[:, ind] = outputX
    return transf_inputs





### Function for transforming all inputs (approx. quantile transform)
### 'inputs' is a (no_data, no_features) array
### 'feature_keys' the list of features according to which the colums in 'inputs' are ordered
### 'bounds' is a list of [min, max] pairs in which the inputs are rescaled
def inputs_transform_quantile(inputs, feature_keys, bounds):
    transf_inputs = copy.deepcopy(inputs)
    for key in feature_keys:
        ind = np.where(np.array([k==key for k in feature_keys]))[0][0]
        inputX = inputs[:, ind]
        boundsX = bounds[ind]
        if (key=='hus' or key=='qv'):
            outputX = ecdf_fit_function_water(np.squeeze(inputX), 10.0, 0.05, 0.018)
            minX = boundsX[0]
            maxX = boundsX[1]
            outputX = (maxX - minX) * np.squeeze(outputX) + minX
            transf_inputs[:, ind] = outputX
        if (key=='clw' or key=='qc'):
            outputX = ecdf_fit_function_water(np.squeeze(inputX), 0.95, 0.25, 0.0001)
            minX = boundsX[0]
            maxX = boundsX[1]
            outputX = (maxX - minX) * np.squeeze(outputX) + minX
            transf_inputs[:, ind] = outputX
        if (key=='cli' or key=='qi'):
            outputX = ecdf_fit_function_water(np.squeeze(inputX), 0.5, 1.0, 0.000003)
            minX = boundsX[0]
            maxX = boundsX[1]
            outputX = (maxX - minX) * np.squeeze(outputX) + minX
            transf_inputs[:, ind] = outputX
        if (key=='clt' or key=='qt'):
            outputX = ecdf_fit_function_water(np.squeeze(inputX), 0.6, 0.6, 0.00005)
            minX = boundsX[0]
            maxX = boundsX[1]
            outputX = (maxX - minX) * np.squeeze(outputX) + minX
            transf_inputs[:, ind] = outputX
        if (key=='ta' or key=='temp'):
            outputX = (np.squeeze(inputX) - 180.0)/(305.0 - 180.0)
            minX = boundsX[0]
            maxX = boundsX[1]
            outputX = (maxX - minX) * np.squeeze(outputX) + minX
            transf_inputs[:, ind] = outputX
        if (key=='pa' or key=='pres'):
            outputX = (np.squeeze(inputX) - 4000.0)/(105000.0 - 4000.0)
            minX = boundsX[0]
            maxX = boundsX[1]
            outputX = (maxX - minX) * np.squeeze(outputX) + minX
            transf_inputs[:, ind] = outputX
        if (key=='ua' or key=='u'):
            outputX = (np.squeeze(inputX) + 60.0)/(110.0 + 60.0)
            minX = boundsX[0]
            maxX = boundsX[1]
            outputX = (maxX - minX) * np.squeeze(outputX) + minX
            transf_inputs[:, ind] = outputX
        if (key=='va' or key=='v'):
            outputX = (np.squeeze(inputX) + 70.0)/(70.0 + 70.0)
            minX = boundsX[0]
            maxX = boundsX[1]
            outputX = (maxX - minX) * np.squeeze(outputX) + minX
            transf_inputs[:, ind] = outputX
        if (key=='hwind' or key=='hw'):
            outputX = ecdf_fit_function_water(np.squeeze(inputX), 0.9, 2.4, 6.5)
            minX = boundsX[0]
            maxX = boundsX[1]
            outputX = (maxX - minX) * np.squeeze(outputX) + minX
            transf_inputs[:, ind] = outputX
        if (key=='zg'):
            outputX = (np.squeeze(inputX) - 0.0)/(20000.0 - 0.0)
            minX = boundsX[0]
            maxX = boundsX[1]
            outputX = (maxX - minX) * np.squeeze(outputX) + minX
            transf_inputs[:, ind] = outputX
        if (key=='coriolis'):
            Omega = 7.2921 * 10.0**(-5.0)
            outputX = (np.arcsin(0.5 * np.squeeze(inputX) / Omega) + np.pi/2.0)/np.pi
            minX = boundsX[0]
            maxX = boundsX[1]
            outputX = (maxX - minX) * np.squeeze(outputX) + minX
            transf_inputs[:, ind] = outputX
        if (key=='fr_land'):
            outputX = np.squeeze(inputX)
            minX = boundsX[0]
            maxX = boundsX[1]
            outputX = (maxX - minX) * np.squeeze(outputX) + minX
            transf_inputs[:, ind] = outputX
    return transf_inputs
