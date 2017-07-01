#! /bin/bash

for i in {0..7}
do
    python3 train.py --model_name baseline_${i}.model --type xgb --features adonehot userlast usernext --valid_day $i
done

python3 predict.py models/baseline --type xgb --features adonehot userlast usernext --all
python3 ensemble.py average
