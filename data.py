import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utility import *

ys = []
users = []
times = []
ads = []

# load training data
for day in range(15):
    if day >= 8:
        _, y_, ads_, times_, users_ = load_train_data_by_day_sparse(day, "half", return_users=True)
    elif day < 8:
        _, y_, ads_, times_, users_ = load_train_data_by_day_sparse(day, "full", return_users=True)

    ys.append(y_)
    users.append(users_)
    times.append(times_)
    ads.append(ads_)


# load testing data
_, tads, ttimes, tusers = load_test_data_sparse(return_users=True)


### ad exploration ###
train_ads = set(np.concatenate(ads))
test_ads = set(tads)

print('unique Ad ID in training set:', len(train_ads))
print('unique Ad ID in testing set:', len(test_ads))
print('Ad ID only in testing set:', len(test_ads - train_ads))
######################


### user exploration ###
train_users = set(np.concatenate(users))
test_users = set(tusers)

print('unique users in training set:', len(train_users))
print('unique users in testing set:', len(test_users))
print('users only in testing set:', len(test_users - train_users))

for time, user in zip(times[:8], users[:8]):
    time = time - time[0]

    rates = []
    for hour in range(24):
        idx = np.logical_and(time >= hour*3600, time < (hour+1)*3600)
        rate = (user[idx] == get_no_user()).mean()
        rates.append(rate)

    print(rates)
    plt.plot(rates)

plt.xlabel('hour', fontsize=15)
plt.ylabel('no user %', fontsize=15)
plt.savefig('pictures/nouser_rate.png')
plt.clf()
########################


### click exploration ###
rates = []
for y in ys[:8]:
    rate = y.mean()
    rates.append(rate)

print(rates)
plt.plot(rates)
plt.xlabel('day', fontsize=15)
plt.ylabel('click rate', fontsize=15)
plt.savefig('pictures/click_rate_by_day.png')
plt.clf()


for y, time in zip(ys[:8], times[:8]):
    time = time - time[0]

    rates = []
    for hour in range(24):
        idx = np.logical_and(time >= hour*3600, time < (hour+1)*3600)
        rate = y[idx].mean()
        rates.append(rate)

    print(rates)
    plt.plot(rates)

plt.xlabel('hour', fontsize=15)
plt.ylabel('click rate', fontsize=15)
plt.savefig('pictures/click_rate_by_hour.png')
plt.clf()
#########################


### other exploration ###
for time in times[:8]:
    time = time - time[0]

    rates = []
    for hour in range(24):
        idx = np.logical_and(time >= hour*3600, time < (hour+1)*3600)
        rate = idx.sum()
        rates.append(rate)

    print(rates)
    plt.plot(rates)

plt.xlabel('hour', fontsize=15)
plt.ylabel('# of examples', fontsize=15)
plt.savefig('pictures/examples_by_hour.png')
plt.clf()
#########################
