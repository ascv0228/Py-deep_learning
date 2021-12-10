import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

toy = pd.read_csv('./toy_train.csv')
parameters_1 = {
    'max_depth' : range(2, 8),  #range(2, 8)
    'learning_rate' : [x*0.05 for x in range(2, 20)],  #
    'min_child_weight': [2, 5, 9, 10, 20, 30, 50],  #range(2, 10, 2),
    'gamma': range(5),
    'max_delta_step' : [0, 0.2, 0.4, 0.6, 1, 2, 5, 10],
    'subsample': [x*0.05 for x in range(10, 21)],
    'colsample_bytree' : [0.6, 0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.001, 0.01, 0.25, 0.5, 0.75, 1], #[0, 0.25, 0.5, 0.75, 1],
    'reg_lambda': [0.6, 0.8, 1, 5, 10],
    'scale_pos_weight': [0.6, 0.8, 1], #[0.2, 0.4, 0.6, 0.8, 1]
    'booster': ['gbtree'],
    'objective' : ['reg:squarederror']
}

parameters_2 = {
    'max_depth' : range(2, 8),  #range(2, 8)
    'learning_rate' : [x*0.05 for x in range(1, 20)],  #
    'min_child_weight': [2, 5, 10, 20, 30, 50, 80, 100, 150, 200, 250],  #range(2, 10, 2),
    'gamma': range(5),
    'max_delta_step' : [0, 0.2, 0.4, 0.6, 1, 2, 5, 10],
    'subsample': [x*0.05 for x in range(5, 21)],
    'colsample_bytree' : [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.001, 0.01, 0.25, 0.5, 0.75, 1, 2, 3], #[0, 0.25, 0.5, 0.75, 1],
    'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1, 5, 10, 20, 50, 100],
    'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1], #[0.2, 0.4, 0.6, 0.8, 1]
    'booster': ['gbtree'],
    'objective' : ['reg:squarederror' ]
}

def dict_random_choice(D):
    import random
    output = dict()
    for i in D:
        output[i] = random.choice(D[i])
    return output

state = 'clean_get_score'
print('\n', state)
if state == 'train':
    x = toy[['x', 'y', 'a', 'b', 'c', 'd', 'e']]
    y = toy['class'].map({'A': 0, 'B': 1})
    print(toy.loc[1])
    train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.333, random_state=0)

    param_to_val_loss = list()
    for i in range(50):
        param = dict_random_choice(parameters_1)
        res = {}
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dtest = xgb.DMatrix(valid_x, label=valid_y)
        watchlist = [(dtrain, 'train'), (dtest, 'val')]
        bst = xgb.train(param, dtrain, num_boost_round=200, evals=watchlist, verbose_eval=False, evals_result=res, early_stopping_rounds=30)
        param_to_val_loss.append((res['val']['rmse'][-1], param))

        ptrain = bst.predict(dtrain, output_margin=True)
        ptest = bst.predict(dtest, output_margin=True)
        dtrain.set_base_margin(ptrain)

    for i in sorted(param_to_val_loss, key=lambda param_to_val_loss: param_to_val_loss[0])[:5]:
        print(i)

if state == 'get_score':
    x = toy[['x', 'y', 'a', 'b', 'c', 'd', 'e']]
    y = toy['class'].map({'A': 0, 'B': 1})

    train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.333, random_state=0)
    # {'max_depth': 6, 'learning_rate': 0.9500000000000001, 'min_child_weight': 2, 'gamma': 4, 'max_delta_step': 5, 'subsample': 0.75, 'colsample_bytree': 0.9, 'reg_alpha': 0.001, 'reg_lambda': 5, 'scale_pos_weight': 1, 'booster': 'gbtree', 'objective': 'reg:squarederror'}")
    # param = {'max_depth': 4, 'learning_rate': 0.35, 'min_child_weight': 2, 'max_delta_step': 5, 'subsample': 0.85, 'colsample_bytree': 0.9, 'reg_alpha': 0.25, 'reg_lambda': 1, 'scale_pos_weight': 0.8}
    param = {'max_depth': 3,    'learning_rate': 0.7,    'min_child_weight': 9, 'max_delta_step': 10,
             'subsample': 0.95, 'colsample_bytree': 0.9, 'reg_alpha': 0.5, 'reg_lambda': 0.6,
             'scale_pos_weight': 0.6}
    res = {}
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(valid_x, label=valid_y)
    watchlist = [(dtrain, 'train'), (dtest, 'val')]
    bst = xgb.train(param, dtrain, num_boost_round=200, evals=watchlist, verbose_eval=False, evals_result=res,
                    early_stopping_rounds=30)
    print('\nparam : ', param)
    print('loss :', res['val']['rmse'][-1], '\n')
    for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
        print('%s: ' % importance_type, bst.get_score(importance_type=importance_type))

if state =='clean_train':
    x = toy[['x', 'y']]
    y = toy['class'].map({'A': 0, 'B': 1})
    # x['y*'] = x['y']
    # x['x*'] = x['x']
    print(x.columns)

    param_to_val_loss = list()
    for i in range(500):
        param = dict_random_choice(parameters_2)
        res = {}
        train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.333, random_state=0)
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dtest = xgb.DMatrix(valid_x, label=valid_y)
        watchlist = [(dtrain, 'train'), (dtest, 'val')]
        bst = xgb.train(param, dtrain, num_boost_round=200, evals=watchlist, verbose_eval=False, evals_result=res,
                        early_stopping_rounds=30)
        param_to_val_loss.append((res['val']['rmse'][-1], param))

        ptrain = bst.predict(dtrain, output_margin=True)
        ptest = bst.predict(dtest, output_margin=True)
        dtrain.set_base_margin(ptrain)

    for i in sorted(param_to_val_loss, key=lambda param_to_val_loss: param_to_val_loss[0])[:5]:
        print(i)

if state == 'clean_get_score':
    x = toy[['x', 'y']]
    y = toy['class'].map({'A': 0, 'B': 1})
    x['y*'] = x['y']
    x['x*'] = x['x']
    print("x.loc[0]\n", x.loc[0])
    print("x.loc[1000]\n", x.loc[1000])
    train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.333, random_state=0)
    param = {'max_depth': 6, 'learning_rate': 0.35000000000000003, 'min_child_weight': 10, 'gamma': 0, 'max_delta_step': 2, 'subsample': 0.30000000000000004, 'colsample_bytree': 0.7, 'reg_alpha': 0.5, 'reg_lambda': 0.8, 'scale_pos_weight': 0.8, 'booster': 'gbtree', 'objective': 'reg:squarederror'}


    res = {}
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(valid_x, label=valid_y)
    watchlist = [(dtrain, 'train'), (dtest, 'val')]
    bst = xgb.train(param, dtrain, num_boost_round=200, evals=watchlist, verbose_eval=False, evals_result=res,
                    early_stopping_rounds=30)
    print('\nparam : ', param)
    print('loss :', res['val']['rmse'][-1], '\n')
    for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
        print('%s: ' % importance_type, bst.get_score(importance_type=importance_type))
