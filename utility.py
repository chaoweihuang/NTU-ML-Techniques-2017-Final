import os
import sys

import numpy as np
import scipy.sparse

from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score, log_loss
from sklearn.externals import joblib


MAX_FEAT = 136
data_dir = "./data/release1/"

def parse_line(line):
    l = line.strip().split()
    user_feat_start = 3 if l[2] == "|user" else 4
    
    timestamp = int(l[0])
    ad_id = l[1]
    click = int(l[2]) if user_feat_start == 4 else -1
    indices = [int(i)-1 for i in l[user_feat_start:]]
    
    return timestamp, ad_id, click, indices
    
def to_user_feat(indices):
    user_feat = np.zeros(MAX_FEAT)
    user_feat[indices] = 1
    
    return user_feat

def get_train_data_by_day_sparse(day, _type="full", return_ad=False):
    if not return_ad:
        X, y = get_train_data_by_day(day, _type, return_ad)
        return csr_matrix(X), y
    else:
        X, y, ads = get_train_data_by_day(day, _type, return_ad)
        return csr_matrix(X), y, ads

def get_train_data_by_day(day, _type="full", return_ad=False):
    X, y, ad_ids = [], [], []
    
    for line in open(os.path.join(data_dir, "train_%s_%d" % (_type, day))):
        timestamp, ad_id, click, indices = parse_line(line)
            
        X.append(to_user_feat(indices))
        y.append(click)
        ad_ids.append(ad_id)
 
    if not return_ad:
        return np.asarray(X), np.asarray(y)
    else:
        return np.asarray(X), np.asarray(y), ad_ids

def get_train_data():
    X, y = [], []
    
    for day in range(8):
        day_X, day_y = get_train_data_by_day(day)
        X = np.concatenate((X, day_X))
        y = np.concatenate((y, day_y))
     
    return X, y
    
def get_train_data_sparse():
    y = []
    
    row = 0
    col_inds = []
    row_inds = []
    for day in range(8):
        print('processing day %d' % day)
        path = os.path.join(data_dir, "train_full_%d" % day)
        for line in open(path):
            timestamp, ad_id, click, indices = parse_line(line)
            col_inds.append(indices)
            row_inds.append(np.full_like(indices, row))
            
            y.append(click)
            row += 1
    
    col_inds = np.concatenate(col_inds)
    row_inds = np.concatenate(row_inds)
    X = csr_matrix((np.ones_like(col_inds), (row_inds, col_inds)),
                   shape=(row, MAX_FEAT))
    
    return X, np.asarray(y)

def get_test_data():
    X = []
    
    for day in range(8, 15):
        for line in open(os.path.join(data_dir, "test_half_%d" % day)):
            timestamp, ad_id, click, indices = parse_line(line)
            
            X.append(to_user_feat(indices))
     
    return np.asarray(X)
    
def get_test_data_sparse(return_ad=False):
    row = 0
    col_inds = []
    row_inds = []
    ad_ids = []

    for day in range(8, 15):
        print('processing day %d' % day)
        for line in open(os.path.join(data_dir, "test_half_%d" % day)):
            timestamp, ad_id, click, indices = parse_line(line)
            
            col_inds.append(indices)
            row_inds.append(np.full_like(indices, row))
            ad_ids.append(ad_id)
            
            row += 1

    col_inds = np.concatenate(col_inds)
    row_inds = np.concatenate(row_inds)
    X = csr_matrix((np.ones_like(col_inds), (row_inds, col_inds)),
                   shape=(row, MAX_FEAT))
    
    if not return_ad:
        return X
    else:
        return X, ad_ids

def load_train_data_by_day_sparse(day, _type="full", return_users=False):
    X = scipy.sparse.load_npz('data/days/train_%s_%d_sparse.npz' % (_type, day))
    y = np.load('data/days/train_%s_%d_y.npy' % (_type, day))
    ads = np.load('data/days/train_%s_%d_ads.npy' % (_type, day))

    if not return_users:
        return X, y, ads
    else:
        time = np.load('data/days/train_%s_%d_time.npy' % (_type, day))
        users = np.load('data/days/train_%s_%d_users.npy' % (_type, day))
        return X, y, ads, time, users

def load_train_data_sparse():
    X = scipy.sparse.load_npz('data/X_sparse.npz')
    y = np.load('data/y.npy')

    return X, y

def load_train_data_by_days_sparse(days, only_half=False, valid_day=7):
    X = np.array([]).reshape((0, 136))
    y = np.array([]).reshape((0, ))
    times = []
    ads = []
    users = []

    for day in days:
        if day >= 8:
            X_, y_, ads_, times_, users_ = load_train_data_by_day_sparse(day, "half", return_users=True)
        elif day < 8:
            X_, y_, ads_, times_, users_ = load_train_data_by_day_sparse(day, "full", return_users=True)

            half = get_half_index(day)

            if day == valid_day:
                if only_half:
                    valid_split = list(range(X.shape[0], X.shape[0]+X_.shape[0]-half))
                else:
                    valid_split = list(range(X.shape[0]+half, X.shape[0]+X_.shape[0]))

            if only_half:
                X_ = X_[half:]
                y_ = y_[half:]
                ads_ = ads_[half:]
                times_ = times_[half:]
                users_ = users_[half:]
            
        X = scipy.sparse.vstack((X, X_), format='csr')
        y = np.concatenate((y, y_), axis=0)
        times = np.concatenate((times, times_), axis=0)
        ads = np.concatenate((ads, ads_), axis=0)
        users = np.concatenate((users, users_), axis=0)

    valid_idx = np.zeros(X.shape[0]).astype(bool)
    valid_idx[valid_split] = True
    train_idx = np.logical_not(valid_idx)

    valid_idx = np.where(valid_idx)[0]
    train_idx = np.where(train_idx)[0]

    return X, y, times, ads, users, train_idx, valid_idx

def load_test_data_sparse(return_users=False):
    X = scipy.sparse.load_npz('data/testX_sparse.npz')
    ads = np.load('data/test_ad_id.npy')
    
    if not return_users:
        return X, ads
    else:
        time = np.load('data/test_time.npy')
        users = np.load('data/test_users.npy')
        return X, ads, time, users

def get_ad_prob(ads, path='data/ad_prob_all.npy', silent=False, ad_prob=None):
    if ad_prob is None:
        ad_prob = np.load(path).item()

    ads_feat = []
    hits = []
    for ad in ads:
        feat = [ad_prob.get(ad, 0.035)]
        ads_feat.append(feat)
        hits.append(ad in ad_prob)
    
    if not silent:
        print('ad hit rate: %f' % np.mean(hits))

    return np.array(ads_feat), np.array(hits)

def get_user_prob(X, users=None, path='data/user_prob_all.npy', 
                    silent=False, user_prob=None):
    """
    Args
    
    X: sparse matrix of user features
    """
    if user_prob is None:
        user_prob = np.load(path).item()

    if users is None:
        users = []
        for i in range(X.shape[0]):
            X_ = X[i].toarray().astype(float).reshape(-1)
            h = hash(tuple(map(float, X_)))
            users.append(h)

    users_feat = []
    hits = []
    for h in users:
        feat = [user_prob.get(h, 0.035)]
        users_feat.append(feat)
        hits.append(h in user_prob)

    if not silent:
        print('user hit rate: %f' % np.mean(hits))

    return np.array(users_feat), np.array(hits)

def get_last_time_diff(times, ids, max_time=60, count=1):
    last_time = {}
    time_diff = []
    for time, id in zip(times, ids):
        if not id in last_time:
            last_time[id] = np.zeros(count)
        
        time_diff.append(np.clip(time-last_time[id], 0, max_time))
        last_time[id] = np.concatenate(([time], last_time[id][:-1]))

    return np.array(time_diff)

def get_next_time_diff(times, ids, max_time=60, count=1):
    next_time = {}
    time_diff = []
    for time, id in zip(reversed(times), reversed(ids)):
        if not id in next_time:
            next_time[id] = np.full(count, 1e10)

        time_diff.append(np.clip(next_time[id]-time, 0, max_time))
        next_time[id] = np.concatenate(([time], next_time[id][:-1]))

    return np.array(list(reversed(time_diff)))

def get_user_index(X):
    user_idx = {}
    for i in range(X.shape[0]):
        X_ = X[i].toarray().astype(float).reshape(-1)
        h = hash(tuple(map(float, X_)))
        if not h in user_idx:
            user_idx[h] = []
        user_idx[h].append(i)

    return user_idx

def cal_user_prob(X, y):
    user_idx = get_user_index(X)

    user_prob = {}
    for user in user_idx:
        user_prob[user] = (y[user_idx[user]].sum() + 0.35) / (y[user_idx[user]].shape[0] + 10)

    return user_prob

def get_ad_index(ads):
    ad_idx = {}
    index = np.arange(ads.shape[0])
    for ad in np.unique(ads):
        ad_idx[ad] = index[ads==ad]

    return ad_idx

def cal_ad_prob(ads, y):
    ad_idx = get_ad_index(ads)

    ad_prob = {}
    for ad in ad_idx:
        ad_prob[ad] = (y[ad_idx[ad]].sum() + 0.35) / (y[ad_idx[ad]].shape[0] + 10)

    return ad_prob

def get_half_index(day):
    indices = [576151, 763555, 764373, 437791, 546113, 764263, 631559, 584053]
    return indices[day]

def get_no_user():
    vec = np.zeros(136)
    vec[0] = 1
    no_user = hash(tuple(map(float, vec)))

    return no_user

def get_feature(features, X, y, train_idx=None, 
                ads=None, times=None, users=None,
                verbose=False):
    if features is None:
        return X

    X_mean = X.mean(axis=1)

    feats = []

    if 'adonehot' in features:
        if ads is None:
            print('adonehot in feature_list but ads are not provided')
            return X

        if verbose:
            print('getting adonehot feature')

        lb = joblib.load('models/lb_global_all.pkl')
        feat = lb.transform(ads)
        X = scipy.sparse.hstack((X, feat), format='csr')

    if 'adprob' in features:
        if ads is None:
            print('adprob in feature_list but ads are not provided')
            return X

        if verbose:
            print('getting adprob feature')

        ads_prob = cal_ad_prob(ads[train_idx], y[train_idx])
        feat, _ = get_ad_prob(ads, ad_prob=ads_prob)
        feats.append(feat)

    if 'userprob' in features:
        if verbose:
            print('getting userprob feature')

        user_prob = cal_user_prob(X[train_idx], y[train_idx])
        feat, _ = get_user_prob(X, users=users, user_prob=user_prob)
        feats.append(feat)

    if 'userlast' in features:
        if verbose:
            print('getting userlast feature')

        feat = get_last_time_diff(times, users, 60) / 60.0
        feats.append(feat)
    
    if 'userlast2' in features:
        if verbose:
            print('getting userlast2 feature')

        feat = get_last_time_diff(times, users, 60, 2) / 60.0
        feats.append(feat)

    if 'usernext' in features:
        if verbose:
            print('getting usernext feature')

        feat = get_next_time_diff(times, users, 60) / 60.0
        feats.append(feat)
    
    if 'usernext2' in features:
        if verbose:
            print('getting usernext2 feature')

        feat = get_next_time_diff(times, users, 60, 2) / 60.0
        feats.append(feat)
    
    if 'usernext3' in features:
        if verbose:
            print('getting usernext3 feature')

        feat = get_next_time_diff(times, users, 60, 3) / 60.0
        feats.append(feat)

    if 'usermean' in features:
        if verbose:
            print('getting usermean feature')

        feat = X_mean
        feats.append(feat)

    if 'isnouser' in features:
        if verbose:
            print('getting isnouser feature')

        feat = (users == get_no_user()).astype(int).reshape(-1, 1)
        feats.append(feat)
    
    if 'weekday' in features:
        if verbose:
            print('getting weekday feature')

        feat = np.zeros((X.shape[0], 7)).astype(int)
        weekdays = ((((times-258890)%(86400*7))/86400)%7).astype(int)
        feat[np.arange(times.shape[0]), weekdays] = 1
        feats.append(feat)

    if 'timeinday' in features:
        if verbose:
            print('getting timeinday feature')

        feat = (((times-258890) % 86400) / 86400).reshape(-1, 1)
        feats.append(feat)

    if verbose:
        print('stacking features')

    X = scipy.sparse.hstack((X, np.hstack(feats)), format='csr')

    return X

def write_submit_file(filename, answers):
    with open(filename, 'w') as fw:
        for ans in answers:
            fw.write('{:.5f}'.format(ans) + '\n')
        fw.flush()

def validate(y_true, y_pred): 
    y_pred[y_pred<0] = 0.0
    y_pred[y_pred>1] = 1.0
    y_pred = np.nan_to_num(y_pred)
    print('========================================================')
    print('mean of pred: ', y_pred.mean())
    print('percentage of zeros in pred: ', np.sum(y_pred==0.0) / y_pred.shape[0])
    print('logloss: %f' % log_loss(y_true, y_pred))
    print(('logloss of all 0.033 on validation set: %f' % 
            log_loss(y_true, np.full_like(y_pred, 0.033))))

    for thres in [1e-5, 1e-4, 1e-3, 1e-2]:
        for val in [1e-2, 1e-3]:
            y_pred_nonzero = np.copy(y_pred)
            y_pred_nonzero[y_pred<=thres] = val
            print(('\tlogloss after %f -> %f: %f' % 
                    (thres, val, log_loss(y_true, y_pred_nonzero))))
    
    for threshold in [0.03, 0.035, 0.04, 0.0425, 0.045, 0.0475, 0.05, 0.0525, 0.055, 0.0575, 0.06, 0.0625, 0.065, 0.0675, 0.07, 0.0725, 0.075, 0.0775, 0.08, 0.085, 0.09, 0.095, 0.1]:
        print('\tf1 score with threshold %f: %f' % 
                (threshold, f1_score(y_true, (y_pred>threshold).astype(int))))
    
    print('========================================================')

def logloss(y, pred):
    loss = 0.0
    if y.shape[0] != pred.shape[0]:
        return -1.0
    
    N = y.shape[0]

    for yi, predi in zip(y, pred):
        loss += yi*_log(predi) + (1-yi)*_log(1-predi)

    return -1/N * loss

def _log(x):
    if x < 1e-15:
        return np.log(1e-15)
    return np.log(x)
