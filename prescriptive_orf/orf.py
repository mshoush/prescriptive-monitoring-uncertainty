import pandas as pd

pd.set_option('display.max_columns', None)
import itertools
import econml
import warnings
warnings.filterwarnings('ignore')

# Main imports
from econml.ortho_forest import DMLOrthoForest, DROrthoForest
#from econml.causal_tree import CausalTree
from econml.sklearn_extensions.linear_model import WeightedLassoCVWrapper, WeightedLasso, WeightedLassoCV

from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

import EncoderFactory
from DatasetManager import DatasetManager

import pandas as pd
import numpy as np
from sklearn.pipeline import FeatureUnion

import time
import os
from sys import argv
import pickle


from plotly.graph_objs.volume.caps import X

import EncoderFactory
from DatasetManager import DatasetManager

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion

import time
import os
import sys
from sys import argv
import pickle

from catboost import Pool, CatBoostRegressor, CatBoostClassifier

print("Read input...")
dataset_name = argv[1]  # prepared_bpic2017
#optimal_params_filename = argv[2]  # params_dir
results_dir = argv[2]  # results_dir

#en_size = int(argv[4]) # size of the ensemble
#print(f"Ensemble size is: {en_size}")

calibrate = False
split_type = "temporal"
oversample = False
calibration_method = "beta"

train_ratio = 0.8
val_ratio = 0.2

# create results directory
if not os.path.exists(os.path.join(results_dir)):
    os.makedirs(os.path.join(results_dir))

print('Preparing data...')
start = time.time()

# read the data
dataset_manager = DatasetManager(dataset_name)
data = dataset_manager.read_dataset()


min_prefix_length = 1
max_prefix_length = int(np.ceil(data.groupby(dataset_manager.case_id_col).size().quantile(0.9)))

cls_encoder_args = {'case_id_col': dataset_manager.case_id_col,
                    'static_cat_cols': dataset_manager.static_cat_cols,
                    'static_num_cols': dataset_manager.static_num_cols,
                    'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                    'dynamic_num_cols': dataset_manager.dynamic_num_cols,
                    'fillna': True}

# split into training and test
if split_type == "temporal":
    train, test = dataset_manager.split_data_strict(data, train_ratio, split=split_type)
else:
    train, test = dataset_manager.split_data(data, train_ratio, split=split_type)

train, val = dataset_manager.split_val(train, val_ratio)

# generate data where each prefix is a separate instance
dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)
dt_train_prefixes.to_pickle(os.path.join(results_dir, "dt_train_prefixes_t.pkl"))

dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)
dt_test_prefixes.to_pickle(os.path.join(results_dir, "dt_test_prefixes_t.pkl"))

dt_val_prefixes = dataset_manager.generate_prefix_data(val, min_prefix_length, max_prefix_length)
dt_val_prefixes.to_pickle(os.path.join(results_dir, "dt_val_prefixes_t.pkl"))

# encode all prefixes
feature_combiner = FeatureUnion(
    [(method, EncoderFactory.get_encoder(method,"orf", **cls_encoder_args)) for method in ["static", "agg"]])
print("Start encoding...")

# train
X_train = feature_combiner.fit_transform(dt_train_prefixes)
y_train = dataset_manager.get_label_numeric(dt_train_prefixes)
t_train = dataset_manager.get_treatment_numeric(dt_train_prefixes)
train_data = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train), pd.DataFrame(t_train)], axis=1)  # pd.DataFrame(X_val)])
train_data.to_pickle(os.path.join(results_dir, "train_data_t.pkl"))
del X_train
del y_train

# test
X_test = feature_combiner.fit_transform(dt_test_prefixes)
y_test = dataset_manager.get_label_numeric(dt_test_prefixes)
t_test = dataset_manager.get_treatment_numeric(dt_test_prefixes)
test_data = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test), pd.DataFrame(t_test)], axis=1)  # pd.DataFrame(X_val)])
test_data.to_pickle(os.path.join(results_dir, "test_data_t.pkl"))

del X_test
del y_test

# Valid
X_val = feature_combiner.fit_transform(dt_val_prefixes)
y_val = dataset_manager.get_label_numeric(dt_val_prefixes)
t_val = dataset_manager.get_treatment_numeric(dt_val_prefixes)
valid_data = pd.concat([pd.DataFrame(X_val), pd.DataFrame(y_val), pd.DataFrame(t_val)], axis=1)  # pd.DataFrame(X_val)])
valid_data.to_pickle(os.path.join(results_dir, "valid_data_t.pkl"))
del X_val
del y_val

print("Read encoded data...")
df_agg = pd.read_csv('dt_transformed_agg.csv', sep=';')
df_static = pd.read_csv('dt_transformed_static.csv', sep=';')

static_agg_df = pd.concat([df_static, df_agg], axis=1)
cat_feat_idx = np.where(static_agg_df.dtypes != float)[0]

#  rename columns
train_data.columns = list(static_agg_df.columns) + ["Outcome"] + ["Treatment"]
test_data.columns = list(static_agg_df.columns) + ["Outcome"] + ["Treatment"]
valid_data.columns = list(static_agg_df.columns) + ["Outcome"] + ["Treatment"]

y_train = train_data['Outcome']
X_train = train_data.drop(['Outcome', ], axis=1)

y_valid = valid_data['Outcome']
X_valid = valid_data.drop(['Outcome'], axis=1)

y_test = test_data['Outcome']
X_test = test_data.drop(['Outcome'], axis=1)

# create results directory
if not os.path.exists(os.path.join(results_dir)):
    os.makedirs(os.path.join(results_dir))

print("\n=============================Start ORF======================================\n")

case_id_col = dataset_manager.case_id_col
activity_col = dataset_manager.activity_col
resource_col = dataset_manager.resource_col
timestamp_col = dataset_manager.timestamp_col

treatment = 'Treatment'
outcome = 'Outcome' # outcome: 1 or zero

train = train_data
test = test_data
valid = valid_data

features_train = train.drop([outcome, treatment], axis=1)

Y = train[outcome].to_numpy()
T = train[treatment].to_numpy()

scaler = StandardScaler()

W = scaler.fit_transform(features_train.to_numpy())

X = scaler.fit_transform(features_train.to_numpy())

N_trees = [200]
Min_leaf_size = [20]
Max_depth = [30]
Subsample_ratio = [0.04]
Lambda_reg = [0.01]

print("\n=========================Start ORF Test==========================\n")
features_test = test.drop([outcome, treatment], axis=1)
X_test = scaler.fit_transform(features_test.to_numpy())

for i in itertools.product(N_trees, Min_leaf_size, Max_depth, Subsample_ratio, Lambda_reg):
    print(i)
    n_trees = i[0]
    min_leaf_size = i[1]
    max_depth = i[2]
    subsample_ratio = i[3]
    lambda_reg = i[4]
    est = DMLOrthoForest(n_jobs=-1, backend='threading',
                         n_trees=n_trees, min_leaf_size=min_leaf_size, max_depth=max_depth,
                         subsample_ratio=subsample_ratio, discrete_treatment=True,
                         model_T=LogisticRegression(C=1 / (X.shape[0] * lambda_reg), penalty='l1', solver='saga'),
                         model_Y=Lasso(alpha=lambda_reg),
                         model_T_final=LogisticRegression(C=1 / (X.shape[0] * lambda_reg), penalty='l1', solver='saga'),
                         model_Y_final=WeightedLasso(alpha=lambda_reg),
                         random_state=106
                         )
    print("Start fitting...")
    ortho_model = est.fit(Y, T, X=X, W=W)
    # save the model to disk
    print("Save model")
    filename = 'ortho_model.sav'
    pickle.dump(ortho_model, open(filename, 'wb'))
    print("Get CATE ...")
    treatment_effects = est.const_marginal_effect(X_test)
    df_results = test
    df_results['Treatment Effects'] = treatment_effects
    df_results.to_pickle(os.path.join(results_dir, "df_results_orf_test_%s.pkl" % dataset_name))
    df_results.to_csv(os.path.join(results_dir, "df_results_orf_test.csv"), sep=";", index=False)


    # Calculate default (90%) confidence intervals for the default treatment points T0=0 and T1=1
    print("Get te_lower, te_upper")
    te_lower, te_upper = est.const_marginal_effect_interval(X_test)
    df_results['te_lower'] = te_lower
    df_results['te_upper'] = te_upper
    df_results['Interval Length'] = df_results['te_upper'] - df_results['te_lower']
    df_results.to_pickle(os.path.join(results_dir, "df_results_orf_test_%s.pkl" % dataset_name))
    df_results.to_csv(os.path.join(results_dir, "df_results_orf_test.csv"), sep=";", index=False)

    # some time later...

    # # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(X_test, Y_test)
    # print(result)




