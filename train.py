import os
import sys

import argparse
import numpy as np
import xgboost as xgb
import scipy.sparse

from fastFM import als
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import log_loss, f1_score
from sklearn.externals import joblib
from scipy.sparse import csr_matrix

from utility import *


def train_by_type(model_name, _type, trainX, trainy, validX, validy):
    if _type == 'lr':
        model = LogisticRegression(solver='sag',
                                   verbose=2,
                                   n_jobs=24,
                                   max_iter=1000,
                                   tol=1e-6)
        model.fit(trainX, trainy)

        joblib.dump(model, os.path.join('models', model_name))

        valid_pred = model.predict_proba(validX)[:, 1]
        validate(validy, valid_pred)

    if _type == 'fm':
        model = als.FMRegression(n_iter=300,
                                 init_stdev=0.001,
                                 rank=2,
                                 l2_reg_w=0.5,
                                 l2_reg_V=0.5)
        model.fit(trainX, trainy)

        joblib.dump(model, os.path.join('models', model_name))

        valid_pred = model.predict(validX)
        validate(validy, valid_pred)

    if _type == 'xgb':
        dtrain = xgb.DMatrix(trainX, trainy)
        dvalid = xgb.DMatrix(validX, validy)

        max_depth = 30
        eta = 0.2
        min_child_weight = 35
        subsample = 0.9
        colsample_bytree = 0.9
        lamb = 1.0
        alpha = 0.0

        params = {
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'eta': eta,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'lambda': lamb,
            'alpha': alpha,
            'silent': 1,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }

        evallist = [(dtrain, 'train'), (dvalid, 'eval')]

        model = xgb.train(params, dtrain, 400,
                          evallist, early_stopping_rounds=30)
        model.save_model(os.path.join('models', model_name))

        valid_pred = model.predict(dvalid)
        validate(validy, valid_pred)

    return model


def train(model_name, _type, features, valid_day):

    print('validate on day %d' % valid_day)

    X, y, ads, times, users, train_idx, valid_idx = load_train_data_by_days_sparse(
        range(15), only_half=False, valid_day=valid_day)

    print('number of data: ', X.shape[0])

    no_user_idx = (users == get_no_user())
    user_idx = np.logical_not(no_user_idx)

    X = get_feature(features, X, y, train_idx,
                    ads, times, users, verbose=True)

    validX, validy = X[valid_idx], y[valid_idx]
    trainX, trainy = X[train_idx], y[train_idx]

    print('training data shape: ', trainX.shape)
    print('validation data shape: ', validX.shape)

    model = train_by_type(model_name, _type, 
                          trainX, trainy, 
                          validX, validy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        help='model name',
                        default='model')
    parser.add_argument('--type', 
                        help='model type, [xgb|lr|fm]',
                        default='xgb')
    parser.add_argument('--features', nargs='*')
    parser.add_argument('--valid_day', type=int, default=7)

    args = parser.parse_args()

    train(args.model_name, args.type,
          args.features, args.valid_day)
