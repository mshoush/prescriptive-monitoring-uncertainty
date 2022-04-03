# When to intervene? Prescriptive Process Monitoring Under Uncertainty and Resource Constraints

This project contains supplementary material for the article "When to intervene? Prescriptive Process Monitoring Under Uncertainty and Resource Constraints" by [Mahmoud Shoush](https://scholar.google.com/citations?user=Jw4rBlkAAAAJ&hl=en) and [Marlon Dumas](https://kodu.ut.ee/~dumas/). We propose a prescriptive process monitoring method to learn when to intervene in order to maximize payoff. 


The proposed method consists of two main phases, training, and testing—the training phase train an ensemble and causal models to estimate the prediction scores and uncertainty. While the testing phase filters ongoing cases that are likely to end negatively in candidates, it then ranks them to select the most profitable one. Furthermore, determine when to trigger an intervention for the chosen case to maximize gain—considering ongoing cases' current and future states scores, uncertainty estimation, and resource availability.



# Dataset: 
Dataset can be found in the "prepare_data" folder or on the following link.
* [BPIC2017, i.e., a loan application process.]( https://owncloud.ut.ee/owncloud/index.php/s/rqk7wNinSzqLMRm)



# Reproduce results:
To reproduce the results, please run the following:

* First, you need to install the environment using

                                     conda create -n <environment-name> --file requirements.txt

* Next, execute the following notebook to run all experiments with default parameters. 

                                     run_exps.ipynb
                                     
                                                           

* Then, collect results from the previous step after experimenting with other parameters mentioned in the paper on page nr 12, i.e., Table 2, and execute scripts in the plots folder to obtain results. 

                                     run_exps.ipynb
