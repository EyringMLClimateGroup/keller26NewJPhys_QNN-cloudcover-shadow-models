import os
import copy
import gc
import re
import numpy as np
from sklearn.utils import shuffle
from input_transform import inputs_transform, output_transform, inverse_transform_clc

### Path to data
path_data_base = '/work/bd1179/b309245/data_for_QML4CLC/'
data_subfolder = 'coarsegrained_R2B5_27levs'





########################################################################
### ------------ Function for loading train and val data ----------- ###
########################################################################

def load_train_val_data(no_training_data, no_validation_data, dataset, features_kept,
                        transform_input=True, bound_input=[0.0, 2.0*np.pi],
                        transform_output=False, bound_output=[0.0, 1.0]):
    
    ### Select path for correspondinf dataset
    path = path_data_base + dataset + '/' + data_subfolder + '/'
    path_data = path + 'train_and_test_sets_cellbased/'
    order_of_vars = np.load(path_data + 'order_of_variables_files.npy')
    
    inds = np.where(np.array([k!='clc' for k in order_of_vars]))[0]
    all_features = copy.deepcopy(order_of_vars[inds])

    ### Get training data
    listallfiles = [f for f in os.listdir(path_data) if f.startswith('training_data_')]
    no_data_files = len(listallfiles)
    no_data_per_file = int(np.ceil((no_training_data + no_validation_data)/no_data_files)) + 100

    loaded_data = []
    for filename in listallfiles:
        data_file = np.load(path_data + filename)
    
        ### Remove samples very low condensate
        clw = data_file[:, (order_of_vars=='clw')]
        cli = data_file[:, (order_of_vars=='cli')]
        clt = clw + cli
        IIIlowclt = (clt <= 1.0e-08)
        III = np.squeeze(np.logical_not(IIIlowclt))
        data_file = data_file[III, :]
        no_kept_data = data_file.shape[0]
        if no_kept_data>=no_data_per_file:
            data_file = shuffle(data_file, n_samples=no_data_per_file)
        
        loaded_data.append(data_file)
    loaded_data = np.vstack(loaded_data)

    loaded_data = shuffle(loaded_data, n_samples=(no_training_data + no_validation_data))
    train_data_0 = copy.deepcopy(loaded_data[0:no_training_data, :])
    val_data_0 = copy.deepcopy(loaded_data[no_training_data:(no_training_data + no_validation_data), :])
    del loaded_data
    gc.collect()

    ### Separate inputs and outputs
    ind = np.where(np.array([k=='clc' for k in order_of_vars]))[0][0]
    train_output = copy.deepcopy(train_data_0[:, ind])
    inds = np.where(np.array([k!='clc' for k in order_of_vars]))[0]
    train_input_0 = copy.deepcopy(train_data_0[:, inds])
    del train_data_0
    gc.collect()

    ind = np.where(np.array([k=='clc' for k in order_of_vars]))[0][0]
    val_output = copy.deepcopy(val_data_0[:, ind])
    inds = np.where(np.array([k!='clc' for k in order_of_vars]))[0]
    val_input_0 = copy.deepcopy(val_data_0[:, inds])
    del val_data_0
    gc.collect()

    ### Add total horizontal wind magnitude 'hwind' to the inputs (condensing 'ua' and 'va')
    all_features = np.append(all_features, 'hwind')
    ### ... in training data
    ind = np.where(np.array([k=='ua' for k in all_features]))[0][0]
    ua = train_input_0[:, ind]
    ind = np.where(np.array([k=='va' for k in all_features]))[0][0]
    va = train_input_0[:, ind]
    hw = np.expand_dims(np.squeeze(np.sqrt(ua**2.0 + va**2.0)), axis=1)
    train_input_0 = np.hstack((train_input_0, hw))
    ### ... in validation data
    ind = np.where(np.array([k=='ua' for k in all_features]))[0][0]
    ua = val_input_0[:, ind]
    ind = np.where(np.array([k=='va' for k in all_features]))[0][0]
    va = val_input_0[:, ind]
    hw = np.expand_dims(np.squeeze(np.sqrt(ua**2.0 + va**2.0)), axis=1)
    val_input_0 = np.hstack((val_input_0, hw))

    ### Add total condensate fraction 'clt' to the inputs (condensing 'clw' and 'cli')
    all_features = np.append(all_features, 'clt')
    ### ... in training data
    ind = np.where(np.array([k=='clw' for k in all_features]))[0][0]
    clw = train_input_0[:, ind]
    ind = np.where(np.array([k=='cli' for k in all_features]))[0][0]
    cli = train_input_0[:, ind]
    clt = np.expand_dims(np.squeeze(clw + cli), axis=1)
    train_input_0 = np.hstack((train_input_0, clt))
    ### ... in validation data
    ind = np.where(np.array([k=='clw' for k in all_features]))[0][0]
    clw = val_input_0[:, ind]
    ind = np.where(np.array([k=='cli' for k in all_features]))[0][0]
    cli = val_input_0[:, ind]
    clt = np.expand_dims(np.squeeze(clw + cli), axis=1)
    val_input_0 = np.hstack((val_input_0, clt))

    ### Select features kept
    inds_kept = []
    for var in features_kept:
        ind = np.where(np.array([k==var for k in all_features]))[0][0]
        inds_kept.append(ind)
    train_input = copy.deepcopy(train_input_0[:, inds_kept])
    val_input = copy.deepcopy(val_input_0[:, inds_kept])

    if transform_input:
        ### Transform input features:
        ### The bounds on transformed inputs are set by default between 0 and 2*pi,
        ### since rotation gates in PL are defined as exp(i * 0.5*theta * sigma)
        bounds = [bound_input for _ in all_features]
        train_input = inputs_transform(train_input, features_kept, bounds)
        val_input = inputs_transform(val_input, features_kept, bounds)

    if transform_output:
        ### Transform outputs
        bounds = bound_output
        train_output = output_transform(train_output, bounds)
        val_output = output_transform(val_output, bounds)
    
    del train_input_0, val_input_0
    gc.collect()
    
    return train_input, train_output, val_input, val_output








########################################################################
### --------------- Function for loading testing data -------------- ###
########################################################################

def load_test_data(no_testing_data, dataset, features_kept,
                   transform_input=True, bound_input=[0.0, 2.0*np.pi],
                   transform_output=False, bound_output=[0.0, 1.0]):
    
    ### Select path for correspondinf dataset
    path = path_data_base + dataset + '/' + data_subfolder + '/'
    path_data = path + 'train_and_test_sets_cellbased/'
    order_of_vars = np.load(path_data + 'order_of_variables_files.npy')
    
    inds = np.where(np.array([k!='clc' for k in order_of_vars]))[0]
    all_features = copy.deepcopy(order_of_vars[inds])

    ### Get testing data
    listallfiles = [f for f in os.listdir(path_data) if f.startswith('testing_data_')]
    no_data_files = len(listallfiles)
    no_data_per_file = int(np.ceil(no_testing_data/no_data_files)) + 100
    
    loaded_data = []
    for filename in listallfiles:
        data_file = np.load(path_data + filename)
    
        ### Remove samples very low condensate
        clw = data_file[:, (order_of_vars=='clw')]
        cli = data_file[:, (order_of_vars=='cli')]
        clt = clw + cli
        IIIlowclt = (clt <= 1.0e-08)
        III = np.squeeze(np.logical_not(IIIlowclt))
        data_file = data_file[III, :]
        no_kept_data = data_file.shape[0]
        if no_kept_data>=no_data_per_file:
            data_file = shuffle(data_file, n_samples=no_data_per_file)
        
        loaded_data.append(data_file)
    loaded_data = np.vstack(loaded_data)
    
    loaded_data = shuffle(loaded_data, n_samples=no_testing_data)
    test_data_0 = copy.deepcopy(loaded_data)
    del loaded_data
    gc.collect()

    ### Separate inputs and outputs
    ind = np.where(np.array([k=='clc' for k in order_of_vars]))[0][0]
    test_output = copy.deepcopy(test_data_0[:, ind])
    inds = np.where(np.array([k!='clc' for k in order_of_vars]))[0]
    test_input_0 = copy.deepcopy(test_data_0[:, inds])
    del test_data_0
    gc.collect()

    ### Add total horizontal wind magnitude 'hwind' to the inputs (condensing 'ua' and 'va')
    all_features = np.append(all_features, 'hwind')
    ### ... in testing data
    ind = np.where(np.array([k=='ua' for k in all_features]))[0][0]
    ua = test_input_0[:, ind]
    ind = np.where(np.array([k=='va' for k in all_features]))[0][0]
    va = test_input_0[:, ind]
    hw = np.expand_dims(np.squeeze(np.sqrt(ua**2.0 + va**2.0)), axis=1)
    test_input_0 = np.hstack((test_input_0, hw))

    ### Add total condensate fraction 'clt' to the inputs (condensing 'clw' and 'cli')
    all_features = np.append(all_features, 'clt')
    ### ... in testing data
    ind = np.where(np.array([k=='clw' for k in all_features]))[0][0]
    clw = test_input_0[:, ind]
    ind = np.where(np.array([k=='cli' for k in all_features]))[0][0]
    cli = test_input_0[:, ind]
    clt = np.expand_dims(np.squeeze(clw + cli), axis=1)
    test_input_0 = np.hstack((test_input_0, clt))

    ### Select features kept
    inds_kept = []
    for var in features_kept:
        ind = np.where(np.array([k==var for k in all_features]))[0][0]
        inds_kept.append(ind)
    test_input = copy.deepcopy(test_input_0[:, inds_kept])

    if transform_input:
        ### Transform input features:
        ### The bounds on transformed inputs are set by default between 0 and 2*pi,
        ### since rotation gates in PL are defined as exp(i * 0.5*theta * sigma)
        bounds = [bound_input for _ in all_features]
        test_input = inputs_transform(test_input, features_kept, bounds)

    if transform_output:
        ### Transform outputs
        bounds = bound_output
        test_output = output_transform(test_output, bounds)
    
    del test_input_0
    gc.collect()
    
    return test_input, test_output










########################################################################
### ------------- Function for loading all testing data ------------ ###
### ------------ i.e. without filters on hus, clt and zg ----------- ###
########################################################################

def load_test_data_nofilter(no_testing_data, dataset, features_kept):
    
    ### Select path for correspondinf dataset
    path = path_data_base + dataset + '/' + data_subfolder + '/'
    path_data = path + 'train_and_test_sets_cellbased/'
    order_of_vars = np.load(path_data + 'order_of_variables_files.npy')
    
    inds = np.where(np.array([k!='clc' for k in order_of_vars]))[0]
    all_features = copy.deepcopy(order_of_vars[inds])

    ### Get testing data
    listallfiles = [f for f in os.listdir(path_data) if f.startswith('testing_data_')]
    no_data_files = len(listallfiles)
    no_data_per_file = int(np.ceil(no_testing_data/no_data_files)) + 100
    
    loaded_data = []
    for filename in listallfiles:
        data_file = np.load(path_data + filename)
        no_kept_data = data_file.shape[0]
        if no_kept_data>=no_data_per_file:
            data_file = shuffle(data_file, n_samples=no_data_per_file)
        
        loaded_data.append(data_file)
    loaded_data = np.vstack(loaded_data)
    
    loaded_data = shuffle(loaded_data, n_samples=no_testing_data)
    test_data_0 = copy.deepcopy(loaded_data)
    del loaded_data
    gc.collect()

    ### Separate inputs and outputs
    ind = np.where(np.array([k=='clc' for k in order_of_vars]))[0][0]
    test_output = copy.deepcopy(test_data_0[:, ind])
    inds = np.where(np.array([k!='clc' for k in order_of_vars]))[0]
    test_input_0 = copy.deepcopy(test_data_0[:, inds])
    del test_data_0
    gc.collect()

    ### Add total horizontal wind magnitude 'hwind' to the inputs (condensing 'ua' and 'va')
    all_features = np.append(all_features, 'hwind')
    ### ... in testing data
    ind = np.where(np.array([k=='ua' for k in all_features]))[0][0]
    ua = test_input_0[:, ind]
    ind = np.where(np.array([k=='va' for k in all_features]))[0][0]
    va = test_input_0[:, ind]
    hw = np.expand_dims(np.squeeze(np.sqrt(ua**2.0 + va**2.0)), axis=1)
    test_input_0 = np.hstack((test_input_0, hw))

    ### Add total condensate fraction 'clt' to the inputs (condensing 'clw' and 'cli')
    all_features = np.append(all_features, 'clt')
    ### ... in testing data
    ind = np.where(np.array([k=='clw' for k in all_features]))[0][0]
    clw = test_input_0[:, ind]
    ind = np.where(np.array([k=='cli' for k in all_features]))[0][0]
    cli = test_input_0[:, ind]
    clt = np.expand_dims(np.squeeze(clw + cli), axis=1)
    test_input_0 = np.hstack((test_input_0, clt))

    ### Select features kept
    inds_kept = []
    for var in features_kept:
        ind = np.where(np.array([k==var for k in all_features]))[0][0]
        inds_kept.append(ind)
    test_input = copy.deepcopy(test_input_0[:, inds_kept])
    
    del test_input_0
    gc.collect()
    
    return test_input, test_output









########################################################################
### --------------- Function for loading data for FIM -------------- ###
########################################################################

def load_input_data(no_data, dataset, features_kept,
                    transform_input=True, bound_input=[0.0, 2.0*np.pi],
                    transform_output=False, bound_output=[0.0, 1.0]):
    
    ### Select path for correspondinf dataset
    path = path_data_base + dataset + '/' + data_subfolder + '/'
    path_data = path + 'train_and_test_sets_cellbased/'
    order_of_vars = np.load(path_data + 'order_of_variables_files.npy')
    
    inds = np.where(np.array([k!='clc' for k in order_of_vars]))[0]
    all_features = copy.deepcopy(order_of_vars[inds])

    ### Get training data
    listallfiles = [f for f in os.listdir(path_data) if f.startswith('training_data_')]
    no_data_files = len(listallfiles)

    loaded_data = []
    for filename in listallfiles:
        data_file = np.load(path_data + filename)
    
        ### Remove samples very low condensate
        clw = data_file[:, (order_of_vars=='clw')]
        cli = data_file[:, (order_of_vars=='cli')]
        clt = clw + cli
        IIIlowclt = (clt <= 1.0e-08)
        III = np.squeeze(np.logical_not(IIIlowclt))
        data_file = data_file[III, :]
        no_kept_data = data_file.shape[0]
        data_file = shuffle(data_file)
        
        loaded_data.append(data_file)
    loaded_data = np.vstack(loaded_data)

    loaded_data = shuffle(loaded_data, n_samples=no_data)

    ### Separate inputs and outputs
    ind = np.where(np.array([k=='clc' for k in order_of_vars]))[0][0]
    output_vals = copy.deepcopy(loaded_data[:, ind])
    inds = np.where(np.array([k!='clc' for k in order_of_vars]))[0]
    input_vals_0 = copy.deepcopy(loaded_data[:, inds])
    del loaded_data
    gc.collect()

    ### Add total horizontal wind magnitude 'hwind' to the inputs (condensing 'ua' and 'va')
    all_features = np.append(all_features, 'hwind')
    ### ... in training data
    ind = np.where(np.array([k=='ua' for k in all_features]))[0][0]
    ua = input_vals_0[:, ind]
    ind = np.where(np.array([k=='va' for k in all_features]))[0][0]
    va = input_vals_0[:, ind]
    hw = np.expand_dims(np.squeeze(np.sqrt(ua**2.0 + va**2.0)), axis=1)
    input_vals_0 = np.hstack((input_vals_0, hw))

    ### Add total condensate fraction 'clt' to the inputs (condensing 'clw' and 'cli')
    all_features = np.append(all_features, 'clt')
    ### ... in training data
    ind = np.where(np.array([k=='clw' for k in all_features]))[0][0]
    clw = input_vals_0[:, ind]
    ind = np.where(np.array([k=='cli' for k in all_features]))[0][0]
    cli = input_vals_0[:, ind]
    clt = np.expand_dims(np.squeeze(clw + cli), axis=1)
    input_vals_0 = np.hstack((input_vals_0, clt))

    ### Select features kept
    inds_kept = []
    for var in features_kept:
        ind = np.where(np.array([k==var for k in all_features]))[0][0]
        inds_kept.append(ind)
    input_vals = copy.deepcopy(input_vals_0[:, inds_kept])

    if transform_input:
        ### Transform input features:
        ### The bounds on transformed inputs are set by default between 0 and 2*pi,
        ### since rotation gates in PL are defined as exp(i * 0.5*theta * sigma)
        bounds = [bound_input for _ in all_features]
        input_vals = inputs_transform(input_vals, features_kept, bounds)

    if transform_output:
        ### Transform outputs
        bounds = bound_output
        output_vals = output_transform(output_vals, bounds)
    
    del input_vals_0
    gc.collect()
    
    return input_vals, output_vals
