# SCOPE
This repository provides the code for the paper *"SCOPE: Sequential Causal Optimization of Process Interventions"*. 

# Experiments
Below are the main results of the paper, and additional results. For each experiment we used 10 different random seeds.

## Main
### SimBank
![Varying training size (SCOPE & SEP: S-learner, XGBoost)](src/utils/results_tools/figures/training_sizes.pdf)
![Varying learner types (10K; SCOPE & SEP: XGBoost)](src/utils/results_tools/figures/learners.pdf)
![Varying base model types (10K; SCOPE & SEP: S-learner)](src/utils/results_tools/figures/base_models.pdf)
### SimBPIC17
![Varying numbers of decision points (10K; SCOPE & SEP: S-learner, XGBoost)](src/utils/results_tools/figures/numbers_of_decision_points.pdf)

## Additional experiment SimBPIC17
### SimBPIC17
![Varying numbers of decision points for T-learner (10K; SCOPE & SEP: XGBoost)](src/utils/results_tools/figures/numbers_of_decision_points_T_xgb.pdf)
![Varying numbers of decision points for Random Forest (10K; SCOPE & SEP: S-learner)](src/utils/results_tools/figures/numbers_of_decision_points_S_rf.pdf)
These additional experiments on SimBPIC17 confirm the paper's finding that SCOPE performs better regardless of the learner or base model used, mirroring the results previously shown on SimBank.

# Code
The structure of the code is as follows:
```
SCOPE/
|_ SimBPIC17/                       # The full SimBPIC17 semi-synthetic simulator
|_ SimBank/                         # The full SimBank simulator
|_ config/                          # Contains all ranges of each parameter in the hyperparameter search
|_ data/                            # Data generated for all experiments (from SimBank & SimBPIC17)
|_ res/                             # Results of the experiments
|_ scripts/
    |_ main.py                      # Main script to run all experiments: generate data, preprocess data, tune, train & eval
|_ src/        
    |_ methods/                           
        |_ kmeans_q/                    # Code for any causal computations at decision points for the KMeans_Q method (placeholder for this method, since it does not require computations)
        |_ scope/                       # Code for any causal computations at decision points for the SCOPE method (e.g., calculating q-values, value function, causal effects...)
        |_ separate/                    # Code for any causal computations at decision points for the SCOPE method (e.g., calculating causal effects...)
        |_ method_main.py               # The overall file (called in scripts/main.py) used for training and tuning of each method
    |_ utils/                           
        |_ model_tools/                 # Code for any causal computations at decision points for the KMeans_Q method (placeholder for this method, since it does not require computations)
            |_ model_eval.py                # Evaluate a model
            |_ model_functions.py           # Classes for each possible model instanece (e.g., XGBoost, LSTM...)
            |_ model_training.py            # Train a model
        |_ prep_tools/                  # Code for all different preprocessing used (aggregation encoding, tensor encoding, preprocessing for Branchi et al. (REF), etc.)
        |_ results_tools/               # Code for generating plots

```

## Installation.
The ```requirements.txt``` file provides the necessary packages for SCOPE and all experiments.
All code was written for ```python 3.11.5```.

## Experiments of the paper
Download the SimBank and SimBPIC17 data for the experiments from ... 

Put the data in the ```data/``` folder. 

Now, the results from the paper can be reproduced by setting the ```path``` variable in the config/config.py file to your directory and running the appropriate script. For example:

```
python scripts/main.py \
        --config config/configs_methods/config_dtr.json \
        --delta 0.95 \
        --methods dtr-S-reg-R \
        --train_size 10000 \
```

Download the results of the experiments from ... 