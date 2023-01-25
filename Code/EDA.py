#%% Import modules
import os, shap, pickle, joblib
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score

#%% Functions
def get_result(test, predict_test, model_name):
    mse = mean_squared_error(test.iloc[:, -1], predict_test)
    mae = mean_absolute_error(test.iloc[:, -1], predict_test)
    r2 = r2_score(test.iloc[:, -1], predict_test)

    print('Mean squared error: ', mse)
    print('Mean absolute error: ', mae)
    print('R2 score: ', r2)

    final_result = pd.concat([test.iloc[:, 0].reset_index(drop=True), test.iloc[:, -1].reset_index(drop=True), pd.DataFrame(predict_test)], axis=1)
    final_result.columns = ['id', 'label', 'predict']
    sns.regplot(x='label', y='predict', data=final_result)
    plt.title('{} - R2score:{:.4f}, MSE:{:.4}, MAE:{:.4}'.format(model_name, r2, mse, mae))
    plt.savefig(os.path.join(save_dir, '{} plot.svg'.format(model_name)))
    plt.clf()

    final_result.to_csv(os.path.join(save_dir, '{} raw results.csv'.format(model_name)), index=False)
    return mse, mae, r2

#%% Settings
EX = 'Static-hd'

#%% Directories
root = os.getcwd()
data_dir = os.path.join(root, 'Data')
save_dir = os.path.join(root, 'Result', EX)
os.makedirs(save_dir, exist_ok=True)

#%% Load data
static_vars = pd.read_csv(os.path.join(data_dir, 'static_variable_daywise_ga.csv')).dropna()
date_vars = pd.read_csv(os.path.join(data_dir, 'date_series_variable.csv')).dropna()

var_names = static_vars.columns.values.tolist()


#%% Convert categorical vars to one-hot vectors
"""
var_to_cate   = ['sex'] # categorical variables to make one-hot vector
new_var_names = []
data = []

for i, name in enumerate(var_names): # make new data with categorical values
    if name not in var_to_cate:
        new_var_names.append(name)
        data.append(csv[name])
    else:
        temp_data = csv[name]
        category  = temp_data.unique().tolist()
        for cate in category:
            new_var_names.append('_'.join([name, str(cate)]))
        one_hot = pd.get_dummies(temp_data)
        data.append(one_hot)
data=pd.concat(data, axis=1)
data.columns=new_var_names
"""


#%% Set results
results = []


#%% Split data into train/test
# x,y = data.iloc[:,1:-2], data.iloc[:,-1]
train_, test_ = train_test_split(static_vars, test_size=0.33, random_state=1000)



train_ = pd.merge(train_, date_vars, on='pt_id').dropna()
test_ = pd.merge(test_, date_vars, on='pt_id').dropna()

train_ = train_.drop(columns=['los_x'])
test_ = test_.drop(columns=['los_x'])


train_['los_y'] = train_['los_y']-train_['hd']
test_['los_y'] = test_['los_y']-test_['hd']

train = train_
test = test_

#%% lgb regressor

params = {'learning_rate': 0.0001,
          'max_depth': 6,
          'boosting': 'gbdt',
          'objective': 'regression',
          'metric': 'mse',
          'is_training_metric': True,
          'num_leaves': 64,
          'feature_fraction': 0.9,
          'bagging_fraction': 0.7,
          'bagging_freq': 5,
          'seed':2018}

train_ds = lgb.Dataset(train.iloc[:,1:-2], label = train.iloc[:,-1])
test_ds = lgb.Dataset(test.iloc[:,1:-2], label = test.iloc[:,-1])
model = lgb.train(params, train_ds, 1000, test_ds, verbose_eval=100)
predict_train = model.predict(train.iloc[:,1:-2])
predict_test = model.predict(test.iloc[:,1:-2])
result = get_result(test, predict_test, 'lgb_regressor')
results.append(list(result))

#%% xgb regressor
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
model.fit(train.iloc[:,1:-1], train.iloc[:,-1])

predict_train = model.predict(train.iloc[:,1:-1])
predict_test = model.predict(test.iloc[:,1:-1])
result = get_result(test, predict_test, 'xgb_regressor')
results.append(list(result))

xgb.plot_importance(model)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'xgb_regressor feature importance.svg'))
plt.clf()


#%% linear regressor
model = LinearRegression()
model.fit(train.iloc[:,1:-2], train.iloc[:,-1])

predict_train = model.predict(train.iloc[:,1:-2])
predict_test = model.predict(test.iloc[:,1:-2])
result = get_result(test, predict_test, 'linear_regressor')
results.append(list(result))

#%% SVM regressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
model.fit(train.iloc[:,1:-2], train.iloc[:,-1])

predict_train = model.predict(train.iloc[:,1:-2])
predict_test = model.predict(test.iloc[:,1:-2])
result = get_result(test, predict_test, 'SVM_regressor')
results.append(list(result))

#%% RF regressor
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(max_depth=2, random_state=0)
model.fit(train.iloc[:,1:-2], train.iloc[:,-1])

predict_train = model.predict(train.iloc[:,1:-2])
predict_test = model.predict(test.iloc[:,1:-2])
result = get_result(test, predict_test, 'RF_regressor')
results.append(list(result))

#%% MLP regressor
from sklearn.neural_network import MLPRegressor
model = MLPRegressor(random_state=1, max_iter=500)
model.fit(train.iloc[:,1:-2], train.iloc[:,-1])

predict_train = model.predict(train.iloc[:,1:-2])
predict_test = model.predict(test.iloc[:,1:-2])
result = get_result(test, predict_test, 'MLP_regressor')
results.append(list(result))




results_df = pd.DataFrame(np.concatenate([results], axis=1), columns = ['MSE','MAE','R2'])
results_df.insert(0,'MODEL', pd.Series(['LGB','XGB','LR','SVM','RF','MLP']))
results_df.to_csv(os.path.join(root, 'Result', 'Results_'+EX+'.csv'), index=False)