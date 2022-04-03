#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
from sys import argv
import numpy as np
import random
import pickle
import os

import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.ticker as mtick


tqdm.pandas()
pd.set_option('display.max_columns', None)



import sklearn as sk
from sklearn.metrics import auc

import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import pickle


get_ipython().run_line_magic('matplotlib', 'inline')


#total_nu_cases= 3231


# In[2]:


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

df_test_prefix = read_pickle_data("./../results_from_vm_1-1-2022/dt_test_prefixes_t.pkl")
preds_test = read_pickle_data("./../results_from_vm_1-1-2022/preds_test_bpic2017_accepted.pkl")
preds_test

# Predictive part
preds_test["prefix_nr"]= list(df_test_prefix.groupby("Case ID").first()["prefix_nr"])
preds_test["case_id"]= list(df_test_prefix.groupby("Case ID").first()["orig_case_id"])
preds_test["activity"]= list(df_test_prefix.groupby("Case ID").last()["Activity"])
preds_test['time:timestamp'] = list(df_test_prefix.groupby("Case ID").last()["time:timestamp"])
preds_test = preds_test.sort_values(by=['time:timestamp']).reset_index(drop=True)
preds_test

# ORF part
df_results_test_orf = read_pickle_data("/home/mshoush/old_data/ut_cs_phd/phd/code/gitHub/predictive_and_prescriptive/orf_test.pkl")
df_results_test_orf.rename(columns={'Treatment Effects':'CATE'}, inplace=True)

df_results_test_orf


# In[6]:


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


# In[3]:


total_nu_cases= len(set(res['case_id']))
del res


# In[4]:



dfs_fixed = []
dfs_normal = []
dfs_exp = []


distribution = ["fixed", "normal", "exp"]
conditions = ["uncerb_proba_cate_0.75", "proba_cate", "proba_cate_uncer", "aproba_cate"]



for dist in distribution:
    dist_folder = dist
    print(dist)
    for c in conditions:  
        print(c)
        for f in range(1,11):
            try: 
                df = pd.read_csv("./results_test/results_test_baseuncer_thre/"+dist_folder + "/"+c+"_"+str(f)+                      "/"+c +"_" + str(f)+"_df_gains_with_res_2_"+str(f)+".csv", sep=';').iloc[: , :7]
            except:
                try:
                    df = pd.read_csv("./results_test/results_test_baseuncer_thre/"+dist_folder + "/"+c+"_"+str(f)+                      "/df_gains_with_res_2_"+str(f)+".csv", sep=';').iloc[: , :7]
                except:
                    df = pd.read_csv("./results_test/results_test_baseuncer_thre/"+dist_folder + "/"+c+"_"+str(f)+                      "/" +"proba_cate_"+ str(f)+"_df_gains_with_res_2_"+str(f)+".csv", sep=';').iloc[: , :7]
       
            df.name = str(f)+"0%"
            try:
                df.columns = ['case_id', 'adj_gain', 'prefix_nr', 'Activity',]# 'opp_cost', 'c_gain', 'e_gain',] #'delta_uncer']
            except:
                df.columns = ['case_id', 'adj_gain', 'prefix_nr', 'Activity', 'opp_cost', 'c_gain', 'e_gain',] #'delta_uncer']                
                df = df[["case_id", "c_gain", 'prefix_nr', 'Activity']]                
                df.columns = ['case_id', 'adj_gain', 'prefix_nr', 'Activity',]
                df.name = str(f)+"0%"

                
            df['condition'] = c
            if dist=="fixed":
                dfs_fixed.append(df)
            elif dist=="normal":
                dfs_normal.append(df)
            else:
                dfs_exp.append(df)
                


# In[5]:


def get_dicts(dfs):
   
    mydict = {"uncerb_proba_cate_0.75":[], "proba_cate":[],
             "proba_cate_uncer":[], "aproba_cate":[]} # "uncer":[]
    
    for df in dfs:        
        print(df.name)        
        print(sum(df['adj_gain']))
        print(set(df.condition))
        mydict[list(set(df.condition))[0]].append([df.name, list(set(df.condition))[0],                                                                         sum(df['adj_gain']), (np.round((len(set(list(df['case_id']))) * 100) / total_nu_cases, 2))])
    return mydict


# In[6]:



for dist in distribution:
    print(dist)
    if dist=="fixed":
        mydict_fixed = get_dicts(dfs_fixed)
    elif dist=="normal":
        mydict_normal = get_dicts(dfs_normal)
    else:
        mydict_exp = get_dicts(dfs_exp)


# In[7]:


def run_plot(names, gains, percent_cases, typee, dist):
    plt.figure()
    names, gains, percent_cases, condition = names, gains, percent_cases, typee
    
    df = pd.DataFrame({"names":names,
                       "gains": gains, 
                       "treated cases": percent_cases,
                     "condition": condition,
                      "dist":dist})

    ax = df.plot(x="names", y="gains", legend=False,  marker="o")
    ax.spines['left'].set_color('#105b7e')
    ax.tick_params(axis='y', color='#105b7e', labelcolor='#105b7e')
    ax.set_ylabel('Total gain', color="#105b7e")
    ax.set_xlabel('% of availble resources',)
    #plt.yscale("log",  base=2)

    ax2 = ax.twinx()
    df.plot(x="names", y="treated cases", ax=ax2, legend=False, color="#c6511a",  marker="o")
    ax2.set_ylabel('% of treated cases', color="#c6511a")
    ax2.spines['right'].set_color('#c6511a')
    ax2.tick_params(axis='y', color='#c6511a', labelcolor='#c6511a')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    ax.legend(bbox_to_anchor=(0.25, 1))
    ax2.legend(bbox_to_anchor=(0.37, 0.9))
    ax.grid(True)
    plt.tight_layout()
    plt.title(list(set(condition))[0] + "_" + dist)

    #plt.title("Impact of available resources on the gain and treated cases")
    plt.savefig(list(set(condition))[0]+".png")
    plt.show()
    return df


#df =  run_plot(names, gains, percent_cases, typee="CATE _ Fixed")


# In[8]:


res_dicts = [mydict_fixed, mydict_normal, mydict_exp]

df_res = []
i=0
for res_dict in res_dicts:
    
    for key, value in res_dict.items():    
        print(key)
        #print(value)
        #break
        gains = pd.DataFrame(value, columns=['names', 'condition', 'gain', 'number_cases']).gain
        names = pd.DataFrame(value, columns=['names', 'condition', 'gain', 'number_cases']).names
        percent_cases = pd.DataFrame(value, columns=['names', 'condition', 'gain', 'number_cases']).number_cases
        condition = pd.DataFrame(value, columns=['names', 'condition', 'gain', 'number_cases']).condition


        df =run_plot(names, gains, percent_cases, typee= condition, dist = distribution[i])

        df_res.append(df)
        names, gains, percent_cases =[],[],[]
    i+=1



# In[9]:


i =0
k=0
conditions = ["uncerb_proba_cate_0.75", "proba_cate", "proba_cate_uncer", "aproba_cate"]

for i in range(0,len(df_res),4):
    print(distribution[k])

    df_075 = df_res[i]
    df_075 = df_075.rename(columns={"names": "x_75", "gains": "avgProba_CATE_tUncer", "treated cases": 't_uncer_75'})
    
    df_proba_cate = df_res[i+1]
    df_proba_cate = df_proba_cate.rename(columns={"names": "x_proba_cate", "gains": "avgProba_CATE", "treated cases": 't_proba_cate'})
    
    df_proba_cate_uncer = df_res[i+2]
    df_proba_cate_uncer = df_proba_cate_uncer.rename(columns={"names": "x_proba_cate_uncer", "gains": "avgProba_CATE_oppCost_dUncer", "treated cases": 't_proba_cate_uncer'})

    df_aproba_cate = df_res[i+3]
    df_aproba_cate = df_aproba_cate.rename(columns={"names": "x_proba_cate_approach", "gains": "avgProba_CATE_oppCost", "treated cases": 't_proba_cate_approach'})




    df_final = pd.concat([df_075, df_proba_cate, df_proba_cate_uncer,
                         df_aproba_cate], axis=1)
    
    
    import matplotlib.pyplot as plt
    plt.figure()
    colors = ['#2874a6','#148f77',   "#b03a2e", "#839192", "#2e4053", "#884ea0",
             "#641e16","#d4ac0d", "#0b5345", "#82e0aa", "#e59866",]
    colors2 = ['#2874a6','#148f77',   "#b03a2e", "#839192", "#2e4053", "#884ea0" ,
               "#641e16","#d4ac0d", "#0b5345", "#82e0aa", "#e59866",]


    df = df_final


    ax = df.plot(x="x_75", y=["avgProba_CATE", "avgProba_CATE_tUncer",],color=colors,
                 legend=False,  marker="o")
    ax.tick_params(axis='y', )#color='#105b7e', labelcolor='#105b7e')
    ax.set_ylabel('Total Gain',)# color="#105b7e")
    ax.set_xlabel('% of availble resources',)

    ax.legend(bbox_to_anchor=(0.9, 1.2),
                  fancybox=True, shadow=True,  ncol=2,title="Total Gain",  prop={'size': 9},)

    def millions(x, pos):
        'The two args are the value and tick position'
        return '$%1.1fK' % (x*1e-3)

    ax.grid(True)
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(millions))
 

    plt.savefig("RQ1.pdf",  bbox_inches='tight')
    plt.show()
    k+=1
    break


# In[10]:


i =0
k=0
for i in range(0,len(df_res),4):
    print(distribution[k])
    
    df_025 = df_res[i]
    df_025 = df_025.rename(columns={"names": "x_25", "gains": "uncer_0.25", "treated cases": 't_uncer_25'})
    
    df_05 = df_res[i+1]
    df_05 = df_05.rename(columns={"names": "x_5", "gains": "uncer_0.5", "treated cases": 't_uncer_5'})

    df_075 = df_res[i+2]
    df_075 = df_075.rename(columns={"names": "x_75", "gains": "uncer_0.75", "treated cases": 't_uncer_75'})

    df_proba_cate = df_res[i+3]
    df_proba_cate = df_proba_cate.rename(columns={"names": "x_proba_cate", "gains": "proba_cate", "treated cases": 't_proba_cate'})


    df_final = pd.concat([df_025, df_05, df_075, df_proba_cate], axis=1)
    
    
    df_final


    import matplotlib.pyplot as plt
    plt.figure()
    colors = ['#2874a6','#148f77',   "#b03a2e", "#839192", "#2e4053", "#884ea0",
             "#641e16","#d4ac0d", "#0b5345", "#82e0aa", "#e59866",]
    colors2 = ['#2874a6','#148f77',   "#b03a2e", "#839192", "#2e4053", "#884ea0" ,
               "#641e16","#d4ac0d", "#0b5345", "#82e0aa", "#e59866",]


    df = df_final

    ax = df.plot(x="x_25", y=[ "uncer_0.25", "uncer_0.5", "uncer_0.75", 
                               "proba_cate"],color=colors,
                 legend=False,  marker="o")
    ax.tick_params(axis='y', )#color='#105b7e', labelcolor='#105b7e')
    ax.set_ylabel('Total Gain',)# color="#105b7e")
    ax.set_xlabel('% of availble resources',)
    
    if i==4:
        ax.legend(bbox_to_anchor=(0.75, 1.4),
                  fancybox=True, shadow=True,  ncol=2,title="Total Gain",  prop={'size': 9},)
    else:
        pass
    
    def millions(x, pos):
        'The two args are the value and tick position'
        return '$%1.1fK' % (x*1e-3)

    ax.grid(True)
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(millions))
    plt.title(distribution[k].upper())

    plt.savefig("RQ306.pdf",  bbox_inches='tight')
    plt.show()
    k+=1


# In[11]:


i =0
k=0
for i in range(0,len(df_res),4):
    print(distribution[k])
    
    df_025 = df_res[i]
    df_025 = df_025.rename(columns={"names": "x_25", "gains": "gains_uncer_25", "treated cases": 'uncer_0.25'})
    

    df_05 = df_res[i+1]
    df_05 = df_05.rename(columns={"names": "x_5", "gains": "gains_uncer_5", "treated cases": 'uncer_0.5'})
                         
    df_075 = df_res[i+2]
    df_075 = df_075.rename(columns={"names": "x_75", "gains": "gains_uncer_75", "treated cases": 'uncer_0.75'})
    
    df_proba_cate = df_res[i+3]
    df_proba_cate = df_proba_cate.rename(columns={"names": "x_proba_cate", "gains": "gains_proba_cate", "treated cases": 'proba_cate'})


    df_final = pd.concat([df_025, df_05, df_075,  df_proba_cate], axis=1)
    

    import matplotlib.pyplot as plt
    plt.figure()
    colors = ['#2874a6','#148f77',   "#b03a2e", "#839192", "#2e4053", "#884ea0",
             "#641e16","#d4ac0d", "#0b5345", "#82e0aa", "#e59866",]
    colors2 = ['#2874a6','#148f77',   "#b03a2e", "#839192", "#2e4053", "#884ea0" ,
               "#641e16","#d4ac0d", "#0b5345", "#82e0aa", "#e59866",]



    df = df_final
    ax = df.plot(x="x_25", y=[ "uncer_0.25", "uncer_0.5", "uncer_0.75", 
                               "proba_cate"],color=colors,
                 legend=False,  marker="o")
    ax.tick_params(axis='y', )
    
    ax.set_xlabel('% of availble resources',)
    ax.yaxis.set_ticks_position("right")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylabel('% of treated cases')# color="#105b7e")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    if i==4:
        ax.legend(bbox_to_anchor=(0.9, 1.4),
                  fancybox=True, shadow=True,  ncol=3,title="Treated cases",  prop={'size': 9},)
    else:
        pass
    ax.grid(True)
    plt.title(distribution[k].upper())

    plt.savefig("RQ306.pdf",  bbox_inches='tight')
    plt.show()
    k+=1


# In[12]:


i =0
k=0
for i in range(0,len(df_res),4):
    print(distribution[k])
    
    df_025 = df_res[i]
    df_025 = df_025.rename(columns={"names": "x_25", "gains": "gains_uncer_25", "treated cases": 'uncer_0.25'})
    df_025["ratio_25"] = np.round(df_025['gains_uncer_25']/ (df_025['uncer_0.25'] * total_nu_cases), 2)
    

    df_05 = df_res[i+1]
    df_05 = df_05.rename(columns={"names": "x_5", "gains": "gains_uncer_5", "treated cases": 'uncer_0.5'})
    df_05["ratio_05"] = np.round(df_05['gains_uncer_5']/ (df_05['uncer_0.5'] * total_nu_cases), 2)
                         
    df_075 = df_res[i+2]
    df_075 = df_075.rename(columns={"names": "x_75", "gains": "gains_uncer_75", "treated cases": 'uncer_0.75'})
    df_075["ratio_75"] = np.round(df_075['gains_uncer_75']/ (df_075['uncer_0.75'] * total_nu_cases), 2)
    
    df_proba_cate = df_res[i+3]
    df_proba_cate = df_proba_cate.rename(columns={"names": "x_proba_cate", "gains": "gains_proba_cate", "treated cases": 'proba_cate'})
    df_proba_cate["ratio_proba_cate"] = np.round(df_proba_cate['gains_proba_cate']/ (df_proba_cate['proba_cate'] * total_nu_cases), 2)


    df_final = pd.concat([df_025, df_05, df_075,  df_proba_cate], axis=1)
    
    import matplotlib.pyplot as plt
    plt.figure()
    colors = ['#2874a6','#148f77',   "#b03a2e", "#839192", "#2e4053", "#884ea0",
             "#641e16","#d4ac0d", "#0b5345", "#82e0aa", "#e59866",]
    colors2 = ['#2874a6','#148f77',   "#b03a2e", "#839192", "#2e4053", "#884ea0" ,
               "#641e16","#d4ac0d", "#0b5345", "#82e0aa", "#e59866",]



    df = df_final
    ax = df.plot(x="x_25", y=[ "ratio_25", "ratio_05", "ratio_75", 
                               "ratio_proba_cate"],color=colors,
                 legend=False,  marker="o")
    ax.tick_params(axis='y', )#color='#105b7e', labelcolor='#105b7e')
    
    ax.set_xlabel('% of availble resources',)
    ax.set_ylabel('total gain / #treated cases')
    

    if i==4:
        ax.legend(bbox_to_anchor=(0.9, 1.4),
                  fancybox=True, shadow=True,  ncol=3,title="Treated cases",  prop={'size': 9},)
    else:
        pass

    ax.grid(True)
    plt.title(distribution[k].upper())

    plt.savefig("RQ306.pdf",  bbox_inches='tight')
    plt.show()
    k+=1


# In[ ]:




