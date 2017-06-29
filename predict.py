import os
import sys

import argparse
import numpy as np
import xgboost as xgb
import scipy.sparse

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import log_loss, f1_score
from sklearn.externals import joblib
from scipy.sparse import csr_matrix

from utility import *


### Parameters ###

train_pred_dir = 'pred/train'
test_pred_dir = 'pred/test'

##################

def predict_by_type(model, _type, X):
    if _type == 'fm':
        pred = model.predict(X)
        pred[pred < 1e-4] = 1e-4
    elif _type == 'lr':
        pred = model.predict_proba(X)[:, 1]
    elif _type == 'xgb':
        pred = model.predict(xgb.DMatrix(X))
    else:
        return None

    return pred


def predict(model_path, _type, features):

    model_name = os.path.basename(model_path)

    if _type == 'xgb':
        model = xgb.Booster()
        model.load_model(model_path)
    elif _type in ['lr', 'fm']:
        model = joblib.load(model_path)

    print('getting data')

    X, y, ads, times, users, _, _ = load_train_data_by_days_sparse(
        range(8), only_half=True)

    print('getting features for training set')
    X = get_feature(features, X, y, None,
                    ads, times, users,
                    verbose=True)

    pred = predict_by_type(model, _type, X)
    np.save(train_pred_dir + model_name + '.npy', pred)

    print('getting features for testing set')
    testX, testads, testtime, testuser = load_test_data_sparse(
        return_users=True)

    testX = get_feature(features, testX, None, None,
                        testads, testtime, testuser,
                        verbose=True)

    pred = predict_by_type(model, _type, testX)
    np.save(test_pred_dir + model_name + '.npy', pred)

    print('done')


def predict_all(prefix, _type, features):
    print('getting data')

    X, y, ads, times, users, _, _ = load_train_data_by_days_sparse(
        range(8), only_half=True)

    print('getting features for training set')
    X = get_feature(features, X, y, None,
                    ads, times, users,
                    verbose=True)

    print('getting features for testing set')
    testX, testads, testtime, testuser = load_test_data_sparse(
        return_users=True)

    testX = get_feature(features, testX, None, None,
                        testads, testtime, testuser,
                        verbose=True)

    for day in range(7):
        model_path = '%s_%d.model' % (prefix, day)
        model_name = os.path.basename(model_path)

        if not model_name in os.listdir('models'):
            continue

        print(model_name)

        if _type == 'xgb':
            model = xgb.Booster()
            model.load_model(model_path)
        elif _type in ['lr', 'fm']:
            model = joblib.load(model_path)

        pred = predict_by_type(model, _type, X)
        np.save(train_pred_dir + model_name + '.npy', pred)

        pred = predict_by_type(model, _type, testX)
        np.save(test_pred_dir + model_name + '.npy', pred)

        print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='path to model')
    parser.add_argument(
        '--type', help='model type, [xgb|lr|fm]', default='xgb')
    parser.add_argument('--features', nargs='*')
    parser.add_argument('--all', action='store_true')

    args = parser.parse_args()

    if not args.all:
        predict(args.model, args.type, args.features)
    else:
        predict_all(args.model, args.type, args.features)
