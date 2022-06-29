# Overview

The code in this repo accompanies the paper 'BayesGrad: A neural network explainability framework to assist diagnosis of coronary artery disease from PET myocardial perfusion imaging'. 




# Repo Structure

This repository contains the following directories:

#### codebase
- Contains functions that can be reused across the code

#### masks
- Contains csv files with vessel segmentation masks for each of the LAD,
  RCA, and LCX separately as well as combined. The masks for individual vessels
  were originally created in flowquant based on a 20 ring scanner, and they
  were combined in Python.

#### notebooks
- Contains the results notebook where results figures are created

#### output
- Contains various output generated from running code including:
  - saved saliency maps generated across 20 model retrainings
  - saved models for detection and for localization (used as baseline in topK
predictions) over 20 model retrainings
  - results and artifacts of top K prediciton experiment
  - results and artifacts related to performance comparison between deterministic and Bayesian models

#### scripts
- Contains 4 scripts:
    - train_VGGDrop_norm_abn.py
      - Trains the detection model and the supervised localization model that
is used as a baseline comparison in topK prediction
    - make_saliency_map_generator_arg_file.py
      - A script to create the arg file necessary to run the SLURM job array
that generates test set saliency maps for all 20 model retrainings
    - saliency_map_generator.py
      - Generates and saves attention maps to be used in later analysis
    - evaluate_performance.py
      - Evaluates the Bayesian models and deterministic baselines in terms of
AUC, accuracy, and ECE

- The directory SLURM_scripts contains SLURM launch scripts to run the scripts
described above with the appropriate parameters.
    - run_train_VGGDrop_norm_abn_ja.sh starts a SLURM job array that runs 20 
model retrainings of the detection model with the appropriate hyperparameters
selected during hyperparameter tuning.
    - run_train_VGGDrop_norm_abn_loc_ja.sh starts a SLURM job array that runs
20 model retrainings of the supervised localization model that is used as a
baseline for comparison in topK predict with the appropriate hyperparameters
selected during hyperarameter tuning.
    - run_saliency_map_generator.sh starts a SLURM job array that generates
and saves saliency maps using the BayesGrad Var method for the test set for
all 20 models trained in the detection problem. This relies on being passed the
appropriate arguments for the job array which are stored in
/scripts/arg_files/saliency_map_generator_20_trials_args.txt.
    - run_evaluate_performance.sh launches the SLURM job to evaulate all 20 Bayesian
models and their deterministic counterparts in the detection task.

#### Singularity
- xnn4rad.def
  - Singularity def file used to build xnn4rad.sif Singularity environment








# Recreate results

### Step 1: Environment

- Create singularity environment from Singularity/xnn4rad.def

### Step 2: Data

- Data must be prepared and split in the format discussed in the following paper and its accompanying code: Machine and deep learning models for accurate diagnosis of coronary artery disease with myocardial blood flow PET imaging

### Step 3: Scripts

The following SLURM launch scripts are located in scripts/SLURM_scripts

#### 3.1 Run run_train_VGGDrop_norm_abn_ja.sh
- Run run_train_VGGDrop_norm_abn_ja.sh to launch the job array that trains 20 instances of the
detection model trained to predicts only normal/abnormal, and from which attention maps will
later be generated. Optimal hyperparameters identified in hyperparameter tuning are included
in this script.

#### 3.2 Run run_train_VGGDrop_norm_abn_loc_ja.sh
- Run run_train_VGGDrop_norm_abn_loc_ja.sh to launch the job array that trains 20 instances of the 
localization model which is used as a baseline comparison method for localization predictions
derived from attention maps in top K predict. Optimal hyperparameters identified in
hyperparameter tuning are included in this script.

#### 3.3 In a slurm script, run make_saliency_map_generator_arg_file.py
- Within a SLURM launch script, run make_saliency_map_generator_arg_file.py in order to generate the argument file needed to 
launch saliency_map_generator.py in a job array which will generate test set saliency maps 
for all 20 trained detection model instances.

#### 3.4 Run run_saliency_map_generator.sh
- Run run_saliency_map_generator.sh in order to launch the job array which will 
generate test set saliency maps for all 20 trained detection model instances.

#### 3.5 Run run_evaluate_performance.sh
- Run run_evaluate performance.sh in order to evaluate detection performance for all
20 instances of the Bayesian models and deterministic baselines in
terms of AUC, accuracy, and ECE.

### Step 4: Results figures

- Run through the steps in notebooks/Results.ipynb in order to regenerate results figures