#!/usr/bin/env python
# coding: utf-8

# # 1. prepare predicitve and causal results

# In[ ]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sys import argv
pd.set_option('display.max_columns', None)
import pickle
from scipy.spatial.distance import cdist

from tqdm import tqdm
import os



def read_pickle_data(file):
    with open(file, "rb") as fh:
        data = pickle.load(fh)
    
    return data


df_test_prefix = read_pickle_data("./testing_phase_filtering_ranking_resource/data/dt_test_prefixes_t.pkl")
preds_test = read_pickle_data("./testing_phase_filtering_ranking_resource/data/preds_test_bpic2017_accepted.pkl")
preds_test

# Predictive part
preds_test["prefix_nr"]= list(df_test_prefix.groupby("Case ID").first()["prefix_nr"])
preds_test["case_id"]= list(df_test_prefix.groupby("Case ID").first()["orig_case_id"])
preds_test["activity"]= list(df_test_prefix.groupby("Case ID").last()["Activity"])
preds_test['time:timestamp'] = list(df_test_prefix.groupby("Case ID").last()["time:timestamp"])
preds_test = preds_test.sort_values(by=['time:timestamp']).reset_index(drop=True)
preds_test

# ORF part
df_results_test_orf = read_pickle_data("./testing_phase_filtering_ranking_resource/data/orf_test.pkl")
df_results_test_orf.rename(columns={'Treatment Effects':'CATE'}, inplace=True)
df_results_test_orf


# Combine predictive and causal models
orf_test = pd.concat([df_results_test_orf, preds_test], axis=1)
orf_test.to_pickle("orf_test.pkl")
orf_test


res = orf_test[['predicted_proba_0', 'predicted_proba_1', 'predicted', 'actual',
       'total_uncer', 'data_uncer', 'knowledge_uncer', 'confidence',
       'prefix_nr', 'case_id', 'activity', 'time:timestamp', 'CATE']]

del orf_test
del df_results_test_orf
del preds_test
del df_test_prefix


res = res.loc[:,~res.columns.duplicated()]
res.head()


# # 2. Filter and Ranking

# In[ ]:


# function to get scores, e.g., probs, uncert, etc at the next step or prefix

# pre_df: holds information about current prefix + 1

def estimate_future_scores(pre_df, cur_df, prefix_nr, res, c_t0, c_t1):
    if pre_df.empty:
        #print('\n ======================== DataFrame is empty! No previous similar cases ======================\n')
        return None
    else:    
        pre_df['prefix_nr'] =  pre_df.prefix_nr.astype(int)
        pre_df = pre_df.loc[(pre_df['prefix_nr'] == int(prefix_nr) +1)].drop_duplicates()


        pre_df_drop = pre_df.drop(['case_id', 'activity', 'time:timestamp', ], axis=1)
        cur_df_drop = cur_df.drop(['case_id', 'activity', 'time:timestamp', ], axis=1)


        #cols_pre_df_drop = pre_df_drop.columns
        pre_df_drop[pre_df_drop.columns] = pre_df_drop[pre_df_drop.columns].apply(pd.to_numeric, errors='coerce')
        cur_df_drop[cur_df_drop.columns] = cur_df_drop[cur_df_drop.columns].apply(pd.to_numeric, errors='coerce')

        # get similarties between curret and next step
        ary = cdist(pre_df_drop, cur_df_drop, metric='euclidean')          
        flat_list = [item for sublist in ary.tolist() for item in sublist]

        k = pre_df_drop.shape[0] # number of similar prefixes, pre_df_drop.shape[0] 

        # get the frequency of each similar row
        mydict = dict(pre_df['activity'].value_counts())    
        pre_df["frequency"] = pre_df["activity"].apply(lambda x: mydict.get(x))

        # weights
        w = [sum(x) for x in zip(flat_list, pre_df.frequency.to_list())]

        # estimated CATE
        cate_values = res.iloc[pre_df_drop[ary==flat_list[:k]].index.tolist()].CATE
        e_cate = np.ma.average(cate_values.to_list(), weights=w)

        # estimated uncer
        uncer_values = res.iloc[pre_df_drop[ary==flat_list[:k]].index.tolist()].total_uncer
        e_uncer = np.ma.average(uncer_values.to_list(), weights=w)

        # estimated proba
        proba_values = res.iloc[pre_df_drop[ary==flat_list[:k]].index.tolist()].predicted_proba_1
        e_proba = np.ma.average(proba_values.to_list(), weights=w)

        # estimated confidence
        conf_values = res.iloc[pre_df_drop[ary==flat_list[:k]].index.tolist()].confidence
        e_conf = np.ma.average(conf_values.to_list(), weights=w)    

        #print(f"pfx: {set(pre_df_drop.prefix_nr)}")
        prefix_nr = list(pre_df_drop.prefix_nr)[0]

        # estimated gain
        e_gain = (e_proba * c_t0) - ((np.subtract(e_proba, e_cate)) * c_t0 + c_t1)

        return e_gain, e_uncer, e_proba, e_cate, prefix_nr, e_conf    
    


# In[ ]:


# Filter cases

activites = {}


def filter_cases(row, condition, cost_t0, cost_t1, proba_thre):
    
    predicted_proba_1 = row[1]    
    total_uncer = row[4]    
    confidence = row[7]
    prefix_nr = row[8]
    case_id = row[9]   
    CATE = row[-1]
    Activity = row[-3]
    conf = row[-6]
    
    try:
        activites[case_id].append(Activity)
    except:
        activites[case_id] = [Activity]
    
    # CATA
    if condition=="cate":
        if CATE > 0:
            return row
        else:
            return None
        
    # Proba
    elif condition=="proba":
        if predicted_proba_1 > proba_thre:
            return row
        else:
            return None
        
    # Intervention constraint
    elif condition=="ic":
        if 'O_Create Offer' in activites[case_id]:
            return row
        else:
            return None
        
    # uncertainty
    elif condition=="uncer":
        return row
#     else:
#         return None
    
    
    #proba_cate
    elif condition=="proba_cate":
        if predicted_proba_1 > proba_thre and CATE > 0:
            return row
        else:
            return None
    
    #proba_ic
    elif condition=="proba_ic":
        if predicted_proba_1 > proba_thre and 'O_Create Offer' in activites[case_id]:
            return row
        else:
            return None   
    
    
    #proba_uncer
    elif condition=="proba_uncer": # uncer
        if predicted_proba_1 > proba_thre:
            return row
        else:
            return None
    
    # cate_ic
    elif condition=="cate_ic":
        if CATE > 0 and 'O_Create Offer' in activites[case_id]:
            return row
        else:
            return None
    
    
    # cate_uncer
    elif condition=="cate_uncer":
        if CATE > 0:
            return row
        else:
            return None
        
        # ic_uncer
    elif condition=="ic_uncer":
        if 'O_Create Offer' in activites[case_id]:
            return row
        else:
            return None
        
    # proba_cate_uncer
    elif condition=="proba_cate_uncer": # uncer
        if predicted_proba_1 > proba_thre and CATE > 0:
            return row
        else:
            return None
    
    #proba_cate_ic
    elif condition=="proba_cate_ic": # uncer
        if predicted_proba_1 > proba_thre and CATE > 0 and 'O_Create Offer' in activites[case_id]:
            return row
        else:
            return None
        
    # proba_ic_uncer
    elif condition=="proba_ic_uncer": # uncer
        if predicted_proba_1 > proba_thre  and 'O_Create Offer' in activites[case_id]:
            return row
        else:
            return None
    
        
    # cate_ic_uncer
    if condition=="cate_ic_uncer": 
        if CATE > 0 and 'O_Create Offer' in activites[case_id]:
            return row
        else:
            return None

    # proba_cate_ic_uncer
    elif condition=="proba_cate_ic_uncer":
        if predicted_proba_1 > proba_thre  and CATE > 0 and 'O_Create Offer' in activites[case_id]:
            return row
        else:
            return None
    else:
        print("No valid condition")
    
    
    
    
    
        
def check_uncer(e_df, c_df):
    e_gain, e_uncer, e_proba, e_cate, e_prefix_nr, e_conf =     estimate_future_scores(e_df, c_df, prefix_nr, res, c_t0=20, c_t1=1)
    
    c_ucer = c_df.total_uncer
    
    delta_uncer = c_ucer - e_uncer

    
    return delta_uncer
    

def apply_row_filter_cases(row,condition):
    row = filter_cases(row, condition, cost_t0, cost_t1, proba_thre)
    return row


# # 3. Resource allocator

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
from sys import argv
import numpy as np
import random
import pickle
import os
import sys



# Resource allocator
import time
import threading
import random


def allocateRes(s_res, dist): # Allocate
    t = threading.Thread(target=block_and_release_res, args=(s_res, dist))
    t.start()    
    print("Apply treatment")
    print(f"resource number: {s_res} blocked")
    print(f"strat timer for resource number: {s_res}")
        
    
def block_and_release_res(s_res,dist): # timer
#     t_dists = ["normal", "fixed", "exp"]
    t_dist=dist
    if t_dist =="normal":
        treatment_duration = int(random.uniform(1, 60))
    elif t_dist=="fixed":
        treatment_duration = 30#int(np.random.exponential(60, size=1))
    else:
        treatment_duration = int(np.random.exponential(60, size=1))    
    
    
    #treatment_duration = int(random.uniform(1, 60))
    time.sleep(treatment_duration)
    print(f"Treatment duration is: {treatment_duration}, for resource number: {s_res}")
    print(f"Release res: {s_res}")
    nr_res.append(s_res)
    print("Do more stuff here")
    print("")


# # 4. Run Exps

# In[ ]:


resources = 10 # argv[3]
cost_t0 = 20
cost_t1 = 1
t_dist="fixed" # argv[5]
condition="proba_cate_uncer"
proba_thre = 0.5
uncer_thre = -0.025
uncer_condition = True

# resources = int(argv[1])#10  # argv[3]
# cost_t0 = int(argv[2])#20
# cost_t1 = int(argv[3])#1
# t_dist = argv[4]#"fixed", "normal"
# condition = argv[5]#"proba", "ic", "uncer", "cate"
# proba_thre = float(argv[6])#0.5
# uncer_thre = float(argv[7])




adjgain_uncer = {} # {"adjgain": uncer}

nr_res = list(range(1, resources+1, 1))

list_adj_gains = []
list_uncer = []
import os

# t_dist = dist
if t_dist == "normal":
    folder = "results_normal_dur/"
elif t_dist == "fixed":
    folder = "results_fixed_dur/"
else:
    folder = "results_exp_pos_dur/"

results_dir = "results_test/" + folder + condition +"_"+ str(resources) +"/"
if not os.path.exists(os.path.join(results_dir)):
    os.makedirs(os.path.join(results_dir))



e_df = pd.DataFrame(columns=res.columns) # prefious prefixes hold rows at next step: prefix+1
temp_df = pd.DataFrame(columns=res.columns)

df_gains_with_res_2_1 = pd.DataFrame()


visitied_cases = {}
treated_cases = {}
waited_cases = {}
candidate_cases = {}

e_proba = {}
e_uncer = {}
e_cate = {}
activites = {}
row_c =0

gain_dicts = {}

#i=0

print(f"\n==================, exp with: {condition}, dist: {t_dist}, and resources: {resources} ============\n")

# for uncertainty
def get_max_gain(gain_dicts, adjgain_uncer):
    max_gain = gain_dicts[max(gain_dicts, key=gain_dicts.get)]
    print(f"get_max_gain: {max_gain}")
    if adjgain_uncer[max_gain] < uncer_thre:
        max_gain=max_gain
        return max_gain
    else:
        del gain_dicts[max(gain_dicts, key=gain_dicts.get)]
        get_max_gain(gain_dicts, adjgain_uncer)

for row in res.values:
    case_id = row[9]
    if case_id not in treated_cases.keys():
        row_f = pd.DataFrame([list(row)], columns=res.columns).        apply(lambda roww: apply_row_filter_cases(roww, condition), axis=1)
        if row_f.values.any()  is not None:
            c_df = row_f
            predicted_proba_1 = list(row_f.values)[0][1]    
            total_uncer = list(row_f.values)[0][4]    
            confidence = list(row_f.values)[0][7]
            prefix_nr = list(row_f.values)[0][8]
            case_id = list(row_f.values)[0][9]   
            CATE = list(row_f.values)[0][-1]
            Activity = list(row_f.values)[0][-3]
            conf = list(row_f.values)[0][-6]
            try:
                activites[case_id].append(Activity)
            except:
                activites[case_id] = [Activity]
            gain = (predicted_proba_1 * cost_t0) - (np.subtract(predicted_proba_1, CATE)) * cost_t0 + cost_t1

            try:                
                e_gain, e_uncer, e_proba, e_cate, e_prefix_nr, e_conf = estimate_future_scores(e_df, c_df, prefix_nr, res, c_t0=20, c_t1=1)

                opp_cost = e_gain - gain
                #print(f"oppertuninty cost: {opp_cost}")

                adjusted_gain = gain - opp_cost
                #print(f"Adjusted gain: {adjusted_gain}\n")
                
                delta_uncer = check_uncer(e_df, c_df,)[0] #                
                #print(f"delta_uncer: {delta_uncer}")

                
                candidate_cases[case_id] = [case_id, adjusted_gain, prefix_nr, Activity, opp_cost, gain, e_gain, delta_uncer]#
                gain_dicts[case_id] = candidate_cases[case_id][1]
                
                # get key for max value
                max(gain_dicts, key=gain_dicts.get)

                # get max value
                gain_dicts[max(gain_dicts, key=gain_dicts.get)]


                adjgain_uncer[adjusted_gain] = delta_uncer

                if gain_dicts: 
                    if condition=="uncer" or condition=="proba_uncer" or condition=="cate_uncer"                    or condition=="ic_uncer" or condition=="proba_cate_uncer" or condition=="proba_ic_uncer"                    or condition=="cate_ic_uncer" or condition=="proba_cate_ic_uncer":
                        max_gain = get_max_gain(gain_dicts, adjgain_uncer)
                    else:
                        max_gain = gain_dicts[max(gain_dicts, key=gain_dicts.get)]
                        
                    print(f"max_gain: {max_gain}")
                    print(f"nr_res: {nr_res}")

                        
                    if max_gain > 0 and nr_res:                        
                        #print(f"candidate_cases:\n {len(candidate_cases.keys())}")
                        #print(f"Case is...: {candidate_cases[max(gain_dicts, key=gain_dicts.get)]}")  # Prints george

                        df_gains_with_res_2_1 = df_gains_with_res_2_1.append(pd.Series(candidate_cases[max(gain_dicts, key=gain_dicts.get)]), ignore_index=True)
                        selceted_res = nr_res[0]
                        nr_res.remove(nr_res[0])
                        print(f"allocate res: {selceted_res}")
                        treated_cases[candidate_cases[max(gain_dicts, key=gain_dicts.get)][0]] = candidate_cases[max(gain_dicts, key=gain_dicts.get)]
                        print("Treat now")
                        print(f"\nCase is treated: {candidate_cases[max(gain_dicts, key=gain_dicts.get)][0]}, with gain: {treated_cases[candidate_cases[max(gain_dicts, key=gain_dicts.get)][0]]}\n") 
                        allocateRes(selceted_res, t_dist)
                        print("allocateRes func has returned")
                        print("")   
                        del candidate_cases[max(gain_dicts, key=gain_dicts.get)]
                        del gain_dicts[max(gain_dicts, key=gain_dicts.get)] 
                    else:
                        print(f"max_gain: {max_gain}, nr_res: {nr_res}")
                else:
                    print(f"No availble gains: {gain_dicts}")

            except:
                temp_df = temp_df.append(pd.Series(row, index=res.columns), ignore_index = True)
                e_df = pd.concat([e_df, temp_df], axis=0).drop_duplicates()

        else:
            temp_df = temp_df.append(pd.Series(row, index=res.columns), ignore_index = True)
            e_df = pd.concat([e_df, temp_df], axis=0).drop_duplicates()

    else:
        print("\n=== case is treated before, Done==\n")
        if len(treated_cases.keys())== len(set(res['case_id'])):
            break
            
df_gains_with_res_2_1.to_csv(results_dir+'df_gains_with_res_2_'+str(resources)+'.csv', index=False, sep=';')
pd.DataFrame(treated_cases.items()).to_csv(results_dir+'treated_cases_'+str(resources)+'.csv', index=False, sep=';')


# In[ ]:


get_ipython().system('pwd')


# In[ ]:





# In[ ]:





# In[ ]:




