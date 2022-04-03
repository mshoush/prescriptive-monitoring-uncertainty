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


total_nu_cases= 6283


# In[2]:



dfs_fixed = []
dfs_normal = []
dfs_exp = []


distribution = ["fixed", "normal", "exp"]


conditions = ["aproba_cate", "proba_cate_uncer", "proba_cate_uncer05", "proba_cate_uncer01",
             "proba_cate_uncer015", "proba_cate_uncer02", "proba_cate_uncer025", "proba_cate_uncer03"]


for dist in distribution:
    dist_folder = dist
    print(dist)
    for c in conditions:  
        print(c)        
        for f in range(1,11):
            df = pd.read_csv("./results_test/results_from_vm/"+dist_folder + "/"+c+"_"+str(f)+"/"                             +c +"_" + str(f)+"_df_gains_with_res_2_"+str(f)+".csv", sep=';').iloc[: , :7]

                
            df.name = str(f)+"0%"
            df.columns = ['case_id', 'adj_gain', 'prefix_nr', 'Activity', 'opp_cost', 'c_gain', 'e_gain',] #'delta_uncer']
            df['condition'] = c
            if dist=="fixed":
                dfs_fixed.append(df)
            elif dist=="normal":
                dfs_normal.append(df)
            else:
                dfs_exp.append(df)
                


# In[3]:


get_ipython().system('pwd')


# In[6]:



def get_dicts(dfs):

    mydict = {"aproba_cate":[], "proba_cate_uncer":[], "proba_cate_uncer05":[],"proba_cate_uncer01":[],
              "proba_cate_uncer015":[], "proba_cate_uncer02":[], "proba_cate_uncer025":[], "proba_cate_uncer03":[]
             }
    
    for df in dfs:        
        print(df.name)
        print(sum(df['c_gain']))
        print(set(df.condition))
        mydict[list(set(df.condition))[0]].append([df.name, list(set(df.condition))[0],                                                                         sum(df['c_gain']), (np.round((len(set(list(df['case_id']))) * 100) / total_nu_cases, 2))])
    return mydict


# In[7]:



for dist in distribution:
    print(dist)
    if dist=="fixed":
        mydict_fixed = get_dicts(dfs_fixed)
    elif dist=="normal":
        mydict_normal = get_dicts(dfs_normal)
    else:
        mydict_exp = get_dicts(dfs_exp)


# In[8]:


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

    plt.savefig(list(set(condition))[0]+".png")
    plt.show()
    return df



# In[9]:


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



# In[10]:


i =0
k=0
for i in range(0,len(df_res),8):
    print(distribution[k])
    
    df_proba_cate_fixed = df_res[i]
    df_proba_cate_fixed = df_proba_cate_fixed.rename(columns={"names": "x_proba_cate", "gains": "proba_cate", "treated cases": 't_proba_cate'})
    
    df_proba_cate_uncer_fixed = df_res[i+1]
    df_proba_cate_uncer_fixed = df_proba_cate_uncer_fixed.rename(columns={"names": "x_proba_cate_uncer", "gains": "proba_cate_uncer", "treated cases": 't_proba_cate_uncer'})
    
    df_proba_cate_uncer05_fixed = df_res[i+2]
    df_proba_cate_uncer05_fixed = df_proba_cate_uncer05_fixed.rename(columns={"names": "x_proba_cate_uncer05", "gains": "proba_cate_uncer_0.05", "treated cases": 't_proba_cate_uncer_0.05'})
    
    df_proba_cate_uncer01_fixed = df_res[i+3]
    df_proba_cate_uncer01_fixed = df_proba_cate_uncer01_fixed.rename(columns={"names": "x_proba_cate_uncer01", "gains": "proba_cate_uncer_0.1", "treated cases": 't_proba_cate_uncer_0.1'})
    
    df_proba_cate_uncer015_fixed = df_res[i+4]
    df_proba_cate_uncer015_fixed = df_proba_cate_uncer015_fixed.rename(columns={"names": "x_proba_cate_uncer015", "gains": "proba_cate_uncer_0.15", "treated cases": 't_proba_cate_uncer_0.15'})
    
    df_proba_cate_uncer02_fixed = df_res[i+5]
    df_proba_cate_uncer02_fixed = df_proba_cate_uncer02_fixed.rename(columns={"names": "x_proba_cate_uncer02", "gains": "proba_cate_uncer_0.2", "treated cases": 't_proba_cate_uncer_0.2'})
    
    df_proba_cate_uncer025_fixed = df_res[i+6]
    df_proba_cate_uncer025_fixed = df_proba_cate_uncer025_fixed.rename(columns={"names": "x_proba_cate_uncer025", "gains": "proba_cate_uncer_0.25", "treated cases": 't_proba_cate_uncer_0.25'})
    
    df_proba_cate_uncer03_fixed = df_res[i+7]
    df_proba_cate_uncer03_fixed = df_proba_cate_uncer03_fixed.rename(columns={"names": "x_proba_cate_uncer03", "gains": "proba_cate_uncer_0.3", "treated cases": 't_proba_cate_uncer_0.3'})


    df_final = pd.concat([df_proba_cate_fixed, df_proba_cate_uncer_fixed,
                         df_proba_cate_uncer05_fixed, df_proba_cate_uncer01_fixed,
                         df_proba_cate_uncer015_fixed,df_proba_cate_uncer02_fixed, df_proba_cate_uncer025_fixed,
                         df_proba_cate_uncer03_fixed], axis=1)
   
    import matplotlib.pyplot as plt
    plt.figure()
    colors = ['#2874a6','#148f77',   "#b03a2e", "#839192", "#2e4053", "#884ea0",
             "#641e16","#d4ac0d", "#0b5345", "#82e0aa", "#e59866",]
    colors2 = ['#2874a6','#148f77',   "#b03a2e", "#839192", "#2e4053", "#884ea0" ,
               "#641e16","#d4ac0d", "#0b5345", "#82e0aa", "#e59866",]


    df = df_final

    ax = df.plot(x="x_proba_cate", y=["proba_cate", "proba_cate_uncer","proba_cate_uncer_0.05",
                                     "proba_cate_uncer_0.1", "proba_cate_uncer_0.15",
                                      "proba_cate_uncer_0.2", "proba_cate_uncer_0.25",
                                     "proba_cate_uncer_0.3"],color=colors,
                 legend=False,  marker="o")
    ax.tick_params(axis='y', )#color='#105b7e', labelcolor='#105b7e')
    ax.set_ylabel('Total adjusted gain',)# color="#105b7e")
    ax.set_xlabel('% of availble resources',)

    if i==8:
        ax.legend(bbox_to_anchor=(0.95, 1.5),
                  fancybox=True, shadow=True,  ncol=2,title="Total gian",  prop={'size': 9},)
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
for i in range(0,len(df_res),8):
    print(distribution[k])
    
    df_proba_cate_fixed = df_res[i]
    df_proba_cate_fixed = df_proba_cate_fixed.rename(columns={"names": "x_proba_cate", "gains": "g_proba_cate", "treated cases": 'proba_cate'})
    
    df_proba_cate_uncer_fixed = df_res[i+1]
    df_proba_cate_uncer_fixed = df_proba_cate_uncer_fixed.rename(columns={"names": "x_proba_cate_uncer", "gains": "g_proba_cate_uncer", "treated cases": 'proba_cate_uncer'})
    
    df_proba_cate_uncer05_fixed = df_res[i+2]
    df_proba_cate_uncer05_fixed = df_proba_cate_uncer05_fixed.rename(columns={"names": "x_proba_cate_uncer05", "gains": "g_proba_cate_uncer_0.05", "treated cases": 'proba_cate_uncer_0.05'})
    
    df_proba_cate_uncer01_fixed = df_res[i+3]
    df_proba_cate_uncer01_fixed = df_proba_cate_uncer01_fixed.rename(columns={"names": "x_proba_cate_uncer01", "gains": "g_proba_cate_uncer_0.1", "treated cases": 'proba_cate_uncer_0.1'})
    
    df_proba_cate_uncer015_fixed = df_res[i+4]
    df_proba_cate_uncer015_fixed = df_proba_cate_uncer015_fixed.rename(columns={"names": "x_proba_cate_uncer015", "gains": "g_proba_cate_uncer_0.15", "treated cases": 'proba_cate_uncer_0.15'})
    
    df_proba_cate_uncer02_fixed = df_res[i+5]
    df_proba_cate_uncer02_fixed = df_proba_cate_uncer02_fixed.rename(columns={"names": "x_proba_cate_uncer02", "gains": "g_proba_cate_uncer_0.2", "treated cases": 'proba_cate_uncer_0.2'})
    
    df_proba_cate_uncer025_fixed = df_res[i+6]
    df_proba_cate_uncer025_fixed = df_proba_cate_uncer025_fixed.rename(columns={"names": "x_proba_cate_uncer025", "gains": "g_proba_cate_uncer_0.25", "treated cases": 'proba_cate_uncer_0.25'})
    
    df_proba_cate_uncer03_fixed = df_res[i+7]
    df_proba_cate_uncer03_fixed = df_proba_cate_uncer03_fixed.rename(columns={"names": "x_proba_cate_uncer03", "gains": "g_proba_cate_uncer_0.3", "treated cases": 'proba_cate_uncer_0.3'})


    df_final = pd.concat([df_proba_cate_fixed, df_proba_cate_uncer_fixed,
                         df_proba_cate_uncer05_fixed, df_proba_cate_uncer01_fixed,
                         df_proba_cate_uncer015_fixed,df_proba_cate_uncer02_fixed, df_proba_cate_uncer025_fixed,
                         df_proba_cate_uncer03_fixed], axis=1)
   
    import matplotlib.pyplot as plt
    plt.figure()
    
    colors = ['#2874a6','#148f77',   "#b03a2e", "#839192", "#2e4053", "#884ea0",
             "#641e16","#d4ac0d", "#0b5345", "#82e0aa", "#e59866",]
    colors2 = ['#2874a6','#148f77',   "#b03a2e", "#839192", "#2e4053", "#884ea0" ,
               "#641e16","#d4ac0d", "#0b5345", "#82e0aa", "#e59866",]


    df = df_final

    ax = df.plot(x="x_proba_cate", y=["proba_cate", "proba_cate_uncer","proba_cate_uncer_0.05",
                                     "proba_cate_uncer_0.1", "proba_cate_uncer_0.15",
                                      "proba_cate_uncer_0.2", "proba_cate_uncer_0.25",
                                     "proba_cate_uncer_0.3"],color=colors,
                 legend=False,  marker="o")
    
    ax.tick_params(axis='y', )#color='#105b7e', labelcolor='#105b7e')
    
    ax.set_xlabel('% of availble resources',)
    ax.yaxis.set_ticks_position("right")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylabel('% of treated cases')# color="#105b7e")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    if i==8:
        ax.legend(bbox_to_anchor=(0.95, 1.5),
                  fancybox=True, shadow=True,  ncol=2,title="Treated cases",  prop={'size': 9},)
    else:
        pass
    ax.grid(True)
    plt.title(distribution[k].upper())

    plt.savefig("RQ306.pdf",  bbox_inches='tight')
    plt.show()
    k+=1


# In[ ]:





# In[ ]:




