# Machine Learning Techniques Final Project
This repository is for the final project of *Machine Learning Techniques* course in *National Taiwan University* (spring 2017), instructed by *Prof. Hsuan-Tien Lin*.

We are team *God Dianyo*, and we ranked first on the private leaderboard in both tracks.


Task Description
------
The task is to predict whether an advertisement will be clicked by a user given an (Ad id, user) pair. There are two tracks with same dataset/target variable but different evaluation metrics.
* Track 1 requires us to predict the click through rate, which is a real number in [0, 1]. The evaluation metrics is log loss.
* Track 2 is a binary classification problem that requires us to predict whether or not the advertisement will be clicked by a user. The evaluation metrics is f1 score.

Our Methods
------

#### Features
Since the features have been preprocessed, it's hard to craft new features by heuristics. We decided to use the basic features plus some handcrafted features borrowed from the article [*Social network and click-through prediction with factorization machines*](http://www.algo.uni-konstanz.de/members/rendle/pdf/Rendle2012-KDDCup.pdf) by S.Rendle in KDDCUP 2012.

* Advertisement ID: We encode each unique Ad-ID into vectors using one-hot encoding.
* user features: We use the 136-dimension user features in the dataset.
* user_last: The time difference between the last impression and this impression for the user.
* user_next: The time difference between this impression and the next impression for the user.

As a result, there are about 760 dimensions for each example.

#### Models
We use logistic regression as a baseline model. It's a very simple model and it optimizes log loss directly. For the other experiments we use factorization machines(FM) [1] and XGBoost [2]. The best result we achieved was trained on  XGBoost.

#### Ensemble
We tried two different ensemble methods, averaging and 2-level ensemble. While averaging gives great results, 2-level ensemble needs careful tuning and validation.
* Averaging: Simply take average of all models' predictions. This is a very consistent method for ensemble. The result is usually better than every single model.
* 2-level ensemble: We take the predictions of different models, combine with original features. Then we train another classifier/regressor use these new features to obtain the final prediction. This method produces slightly worse result than averaging. But our best result uses averaging over several 2-level models, which, in my opinion, is very lucky :) .

#### Results
|  | public score | public rank | private score | private rank |
| :---: | :---: | :---: | :---: | :---: |
| Track 1 | 0.144169 | 2 | 0.144764 | 1 |
| Track 2 | 0.148968 | 1 | 0.152438 | 1 |


How To Run
------

#### Prerequisites
    numpy (0.12.1)
    scipy (0.19)
    scikit-learn (0.18.1)
    fastFM (0.2.9)
    xgboost (0.6)

All packages above are required but the version number is only for your reference. If you use different version of packages it may not run correctly or may produce unexpected results.

#### Get Data
First of all, you need to get the preprocessed data(not the original dataset, we dump the data into sparse matrix and numpy array to load faster).

    bash download.sh

This script will download the compressed data file then uncompress it.

#### Baseline
You can train some models and ensemble them as a baseline by

    bash baseline.sh

This takes about 15 hours on a 24-core machine. The submission files will be placed in `pred` directory. The resulting public score is 0.144516 in track 1 and 0.148063 in track 2.

#### Train
We can use `train.py` to train models. Note that this program use all training data, while we've done several experiments on different data split.

    python3 train.py [-h]
            [--model_name MODEL_NAME]
            [--type TYPE]
            [--features [FEATURES [FEATURES ...]]]
            [--valid_day VALID_DAY]

* **model_name**: The name of this model
* **type**: The type of this model, supports "xgb", "lr", "fm".
* **features**: The features you want to use, supports "adonehot", "userlast", "userlast2", "usernext", "usernext2", "usernext3", "userprob", "adprob", "isnouser", "weekday", "timeinday"
* **valid_day**: The day that is used for validation

    **Sample usage:**

        python3 train.py
            --model_name sample_5.model
            --type xgb
            --features adonehot userlast usernext
            --valid_day 5

#### Predict
We can use `predict.py` to obtain predictions of saved models.

    python3 predict.py [-h]
            [--type TYPE]
            [--features [FEATURES [FEATURES ...]]]
            [--all]
            model

* **all**: If you want to predict all models named `{prefix}_{valid_day}.model`, use this option and specify only the prefix as model path.
* **model**: The path to saved model.

    **Sample usage:**

        python3 predict.py
            models/sample_5.model
            --type xgb
            --features adonehot userlast usernext

#### Ensemble
We can use `ensemble.py` to ensemble the predictions which is predicted by `predict.py`

    python3 ensemble.py [-h] method

* **method**: The ensemble method, can only be either "average" or "2level"

Note that in order to use 2-level ensemble method, you need to name your model `{prefix}_{valid_day}.model`

References
------
[1] S. Rendle. Factorization machines with libFM. ACM Trans. Intell. Syst. Technol., 3(3):57:1–57:22, May 2012.
[2] Tianqi Chen and Carlos Guestrin. XGBoost: A Scalable Tree Boosting System. In 22nd SIGKDD Conference on Knowledge Discovery and Data Mining, 2016
