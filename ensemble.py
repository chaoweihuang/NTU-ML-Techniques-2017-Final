import os
import sys

import numpy as np
import xgboost as xgb
import scipy.sparse

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import log_loss, f1_score
from sklearn.externals import joblib
from scipy.sparse import csr_matrix

from utility import *


### Parameters ###

train_pred_dir = 'pred/train'
test_pred_dir = 'pred/test'
output_dir = 'pred/'
features = ['adonehot', 'userlast', 'usernext', 'isnouser']

##################

testX, testads, testtimes, testusers = load_test_data_sparse(
    return_users=True)

testX = get_feature(features, testX, None, None,
                    testads, testtimes, testusers, verbose=True)

file_list = sorted(os.listdir(train_pred_dir))

shape_sum = 0

for day in range(8):
    X, y, ads, times, users = load_train_data_by_day_sparse(
        day, "full", return_users=True)

    half = get_half_index(day)
    X = X[half:]
    y = y[half:]
    ads = ads[half:]
    times = times[half:]
    users = users[half:]

    preds = []
    test_preds = []
    for pred_file in file_list:
        if not '%d.model' % day in pred_file:
            continue

        pred = np.load(os.path.join(train_pred_dir, pred_file))
        
        pred = pred[shape_sum:shape_sum+y.shape[0]]
        loss = log_loss(y, pred)
        print('%s: %f' % (pred_file, loss))
        preds.append(pred.reshape((-1, 1)))
        
        test_pred = np.load(os.path.join(test_pred_dir, pred_file))
        test_preds.append(test_pred.reshape((-1, 1)))

    preds = np.hstack(preds)
    test_preds = np.hstack(test_preds)
   
    X = get_feature(features, X, y, None,
                    ads, times, users, verbose=True)
    X = scipy.sparse.hstack((X, preds), format='csr')
    
    split = int(X.shape[0] * 0.8)
    trainX, trainy = X[:split], y[:split]
    validX, validy = X[split:], y[split:]

    print('training shape', trainX.shape)
    print('validating shape', validX.shape)
    print(preds.shape)

    validate(validy, preds[split:].mean(axis=1))

    dtrain = xgb.DMatrix(trainX, trainy)
    dvalid = xgb.DMatrix(validX, validy)

    max_depth = 3
    min_child_weight = 13

    print('max_depth', max_depth)
    print('min_child_weight', min_child_weight)

    params = {'max_depth': max_depth,
              'min_child_weight': min_child_weight,
              'eta': 0.2,
              'eval_metric': 'logloss',
              'objective': 'binary:logistic',
              'silent': 1}

    evallist = [(dtrain, 'train'), (dvalid, 'eval')]
    model = xgb.train(params, dtrain, 100, 
                      evallist, early_stopping_rounds=10)

    validate(validy, model.predict(dvalid))

    ############# test ##############

    tX = scipy.sparse.hstack((testX, test_preds), format='csr')
    dtest = xgb.DMatrix(tX)

    pred = model.predict(dtest)
    write_submit_file(
        os.path.join(output_dir, 'output_ensemble_%d.txt' % day), 
        pred)
    
    shape_sum = shape_sum + y.shape[0]
