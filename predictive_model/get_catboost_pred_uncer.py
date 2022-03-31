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
optimal_params_filename = argv[2]  # params_dir
results_dir = argv[3]  # results_dir

en_size = int(argv[4]) # size of the ensemble
print(f"Ensemble size is: {en_size}")

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
dt_train_prefixes.to_pickle(os.path.join(results_dir, "dt_train_prefixes.pkl"))

dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)
dt_test_prefixes.to_pickle(os.path.join(results_dir, "dt_test_prefixes.pkl"))

dt_val_prefixes = dataset_manager.generate_prefix_data(val, min_prefix_length, max_prefix_length)
dt_val_prefixes.to_pickle(os.path.join(results_dir, "dt_val_prefixes.pkl"))

# encode all prefixes
feature_combiner = FeatureUnion(
    [(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in ["static", "agg"]])
print("Start encoding...")

# train
X_train = feature_combiner.fit_transform(dt_train_prefixes)
y_train = dataset_manager.get_label_numeric(dt_train_prefixes)
train_data = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train), ], axis=1)  # pd.DataFrame(X_val)])
train_data.to_pickle(os.path.join(results_dir, "train_data.pkl"))

del X_train
del y_train

# test
X_test = feature_combiner.fit_transform(dt_test_prefixes)
y_test = dataset_manager.get_label_numeric(dt_test_prefixes)
test_data = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1)  # pd.DataFrame(X_val)])
test_data.to_pickle(os.path.join(results_dir, "test_data.pkl"))

del X_test
del y_test

# Valid
X_val = feature_combiner.fit_transform(dt_val_prefixes)
y_val = dataset_manager.get_label_numeric(dt_val_prefixes)
valid_data = pd.concat([pd.DataFrame(X_val), pd.DataFrame(y_val)], axis=1)  # pd.DataFrame(X_val)])
valid_data.to_pickle(os.path.join(results_dir, "valid_data.pkl"))
del X_val
del y_val

print("Read encoded data...")
df_agg = pd.read_csv('dt_transformed_agg.csv', sep=';')
df_static = pd.read_csv('dt_transformed_static.csv', sep=';')

static_agg_df = pd.concat([df_static, df_agg], axis=1)
cat_feat_idx = np.where(static_agg_df.dtypes != float)[0]

#  rename columns
train_data.columns = list(static_agg_df.columns) + ["Outcome"]  # + ["Treatment"]
test_data.columns = list(static_agg_df.columns) + ["Outcome"]  # + ["Treatment"]
valid_data.columns = list(static_agg_df.columns) + ["Outcome"]  # + ["Treatment"]

y_train = train_data['Outcome']
X_train = train_data.drop(['Outcome', ], axis=1)

y_valid = valid_data['Outcome']
X_valid = valid_data.drop(['Outcome'], axis=1)

y_test = test_data['Outcome']
X_test = test_data.drop(['Outcome'], axis=1)

# create results directory
if not os.path.exists(os.path.join(results_dir)):
    os.makedirs(os.path.join(results_dir))

# train the model with pre-tuned parameters
with open(optimal_params_filename, "rb") as fin:
    best_params = pickle.load(fin)

print("Create modle...")
print(f"Cat_feat_idx: {cat_feat_idx}")


# Ensemble of CatBoost
class Ensemble(object):

    def __init__(self, esize=10, iterations=1000, lr=0.1, random_strength=0, border_count=128, depth=6, seed=100, best_param=None):

        self.seed = seed
        self.esize = esize
        self.depth = depth
        self.iterations = iterations
        self.lr = lr  # from tunning
        self.random_strength = random_strength
        self.border_count = border_count
        self.best_param = best_param
        self.ensemble = []
        for e in range(self.esize):
            model = CatBoostClassifier(iterations=self.iterations,
                                       depth=self.depth,
                                       border_count=self.border_count,
                                       random_strength=self.random_strength,
                                       loss_function='Logloss',  # -ve likelihood
                                       verbose=False,
                                       bootstrap_type='Bernoulli',
                                       posterior_sampling=True,
                                       eval_metric='AUC',
                                       use_best_model=True,
                                       langevin=True,
                                       learning_rate=self.best_param['learning_rate'],
                                       subsample=self.best_param['subsample'],
                                       random_seed=self.seed + e)
            self.ensemble.append(model)

    def fit(self, X_train, y_train, cat_feat_idx, eval_set=None):
        for m in self.ensemble:
            print(f"\nFitting model...")
            m.fit(X_train, y=y_train, cat_features=cat_feat_idx,  eval_set=(X_valid, y_valid))
            print("best iter ", m.get_best_iteration())
            print("best score ", m.get_best_score())

    def predict_proba(self, x):
        probs = []
        for m in self.ensemble:
            prob = m.predict_proba(x)
            probs.append(prob)
        probs = np.stack(probs)
        return probs

    def predict(self, x):
        preds = []
        for m in self.ensemble:
            pred = m.predict(x)
            preds.append(pred)
        preds = np.stack(preds)
        return preds


# eoe: Total uncer: entropy of the avg predictions
def entropy_of_expected(probs, epsilon=1e-10):
    mean_probs = np.mean(probs, axis=0)
    log_probs = -np.log(mean_probs + epsilon)
    return np.sum(mean_probs * log_probs, axis=1)


# Data uncer: avg(entropy of indviduals)
def expected_entropy(probs, epsilon=1e-10):
    log_probs = -np.log(probs + epsilon)

    return np.mean(np.sum(probs * log_probs, axis=2), axis=0)


# Knowledge uncer
def mutual_information(probs, epsilon):
    eoe = entropy_of_expected(probs, epsilon)
    exe = expected_entropy(probs, epsilon)
    return eoe - exe  # knowldge_ucer = total_uncer - data_uncer


def ensemble_uncertainties(probs, epsilon=1e-10):
    #print(f"Probs: {np.max(probs)}")
    print(f"Ensemble size: {len(probs)}\n")
    mean_probs = np.mean(probs, axis=0)  # avg ensamble prediction
    conf = np.max(mean_probs, axis=1)  # max avg ensamble prediction: predicted class

    eoe = entropy_of_expected(probs, epsilon)
    exe = expected_entropy(probs, epsilon)
    mutual_info = eoe - exe

    uncertainty = {'confidence': conf,
                   'entropy_of_expected': eoe,  # Total uncer: entropy of the avg predictions
                   'expected_entropy': exe,  # Data uncer: avg(entropy of indviduals)
                   'mutual_information': mutual_info,  # Knowledge uncer
                   }
    print(f"total_uncer: {eoe}")
    print(f"len total_uncer: {len(eoe)}\n")

    print(f"Data_uncer: {exe}")
    print(f"len Data_uncer: {len(exe)}\n")

    print(f"Knowldge_uncer: {mutual_info}")
    print(f"len Knowldge_uncer: {len(mutual_info)}\n")

    return uncertainty


# cls = CatBoostClassifier(iterations=100,
#                          depth=6,
#                          border_count=128,
#                          random_strength=0,
#                          loss_function='Logloss',
#                          verbose=False,
#                          bootstrap_type='Bernoulli',
#                          posterior_sampling=True,
#                          eval_metric='Accuracy',
#                          use_best_model=True,
#                          langevin=True,
#                          learning_rate=best_params['learning_rate'],
#                          subsample=best_params['subsample'],
#                          random_seed=20)

#cls.fit(X_train, y_train, cat_features=cat_feat_idx, eval_set=(X_valid, y_valid))

ens = Ensemble(esize=en_size, iterations=1000, lr=0.1, depth=6, seed=2, random_strength = 100, best_param=best_params)
ens.fit(X_train, y_train, cat_feat_idx, eval_set=(X_valid, y_valid))

probs_train = ens.predict_proba(X_train)
probs_test = ens.predict_proba(X_test)
probs_valid = ens.predict_proba(X_valid)

preds_train_e = ens.predict(X_train)
preds_test_e = ens.predict(X_test)
preds_valid_e = ens.predict(X_valid)


probs_train_mean = np.mean(ens.predict_proba(X_train), axis=0)
probs_test_mean = np.mean(ens.predict_proba(X_test), axis=0)
probs_valid_mean = np.mean(ens.predict_proba(X_valid), axis=0)

uncerts_train = ensemble_uncertainties(probs_train)
uncerts_test = ensemble_uncertainties(probs_test)
uncerts_valid = ensemble_uncertainties(probs_valid)


print("Predict train...")
preds_train_prob_1 = probs_train_mean[:, 1]
preds_train_prob_0 = probs_train_mean[:, 0]
preds_train = np.array(pd.DataFrame(preds_train_e).mode().iloc[0].astype(int))

#np.array(pd.DataFrame(preds).mode().iloc[0].astype(int))

print("Predict test...")
preds_test_prob_1 = probs_test_mean[:, 1]
preds_test_prob_0 = probs_test_mean[:, 0]
preds_test = np.array(pd.DataFrame(preds_test_e).mode().iloc[0].astype(int))

print("Predict valid")
preds_valid_prob_1 = probs_valid_mean[:, 1]
preds_valid_prob_0 = probs_valid_mean[:, 0]
preds_valid = np.array(pd.DataFrame(preds_valid_e).mode().iloc[0].astype(int))

print("Save results")
# write train set predictions
dt_preds_train = pd.DataFrame({"predicted_proba_0": preds_train_prob_0,
                               "predicted_proba_1": preds_train_prob_1,
                               "predicted": preds_train,
                               "actual": y_train,
                               "total_uncer": uncerts_train['entropy_of_expected'],
                              "data_uncer": uncerts_train['expected_entropy'],
                                "knowledge_uncer": uncerts_train["mutual_information"],
                                "confidence": uncerts_train["confidence"] })
dt_preds_train.to_pickle(os.path.join(results_dir, "preds_train_%s.pkl" % dataset_name))

# write test set predictions
dt_preds_test = pd.DataFrame({"predicted_proba_0": preds_test_prob_0,
                              "predicted_proba_1": preds_test_prob_1,
                              "predicted": preds_test,
                              "actual": y_test,
                               "total_uncer": uncerts_test['entropy_of_expected'],
                              "data_uncer": uncerts_test['expected_entropy'],
                                "knowledge_uncer": uncerts_test["mutual_information"],
                                "confidence": uncerts_test["confidence"] })
dt_preds_test.to_pickle(os.path.join(results_dir, "preds_test_%s.pkl" % dataset_name))

# write valid set predictions
dt_preds_valid = pd.DataFrame({"predicted_proba_0": preds_valid_prob_0,
                               "predicted_proba_1": preds_valid_prob_1,
                               "predicted": preds_valid,
                               "actual": y_valid,
                               "total_uncer": uncerts_valid['entropy_of_expected'],
                              "data_uncer": uncerts_valid['expected_entropy'],
                                "knowledge_uncer": uncerts_valid["mutual_information"],
                                "confidence": uncerts_valid["confidence"]})
dt_preds_valid.to_pickle(os.path.join(results_dir, "preds_valid_%s.pkl" % dataset_name))

print("write train-val set predictions CSV")
dt_preds_train.to_csv(os.path.join(results_dir, "preds_train_%s.csv" % dataset_name), sep=";", index=False)
dt_preds_valid.to_csv(os.path.join(results_dir, "preds_val_%s.csv" % dataset_name), sep=";", index=False)
dt_preds_test.to_csv(os.path.join(results_dir, "preds_test_%s.csv" % dataset_name), sep=";", index=False)
