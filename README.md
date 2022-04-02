# When to intervene? Prescriptive Process Monitoring Under Uncertainty and Resource Constraints

This project contains supplementary material for the article "When to intervene? Prescriptive Process Monitoring Under Uncertainty and Resource Constraints" by [Mahmoud Shoush](https://scholar.google.com/citations?user=Jw4rBlkAAAAJ&hl=en), and [Marlon Dumas](https://kodu.ut.ee/~dumas/). We propose a prescriptive process monitoring method to learn when to intervene in order to maximize payoff. 


The proposed method consists of two main phases, training and testing. The training phase train an ensemble and causal models to estimate the prediction scores and uncertainty. While the testing phase filters ongoing cases that are likely to end negatively into candidates, then rank them to select the most profitable one. Furthermore, determine when to trigger an intervention for the chosen case to maximize gain—considering ongoing cases’ current and future states scores, uncertainty estimation, and resource availability.



# Dataset: 
Dataset can be found in the "prepare_data" folder or on the following link.
* [BPIC2017, ie., a loan application process.]( https://owncloud.ut.ee/owncloud/index.php/s/rqk7wNinSzqLMRm)



# Reproduce results:
To reproduce esults, please run the following:

* First you need to install the environment using

                                     conda create -n <environment-name> --file requirements.txt

* Next, please execute the following notebook to run all experiments with default parameters. 

                                     run_exps.ipynb
                                     
        
