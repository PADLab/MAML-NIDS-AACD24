# README.md

This repository contains the code for the MAML (Model-Agnostic Meta-Learning) implementation for the ACI (Army Cybersecurity Institute) dataset.

## Introduction

The MAML algorithm is a meta-learning approach that aims to repair and improve a backbone model by learning from a set of tasks. Tasks are defined as a new attack type where the model has to adapt to the new task with a few gradient steps. The MAML algorithm is used to train a model on a variety of tasks and generalize well to new tasks without a degradation in performance.

## Files

The repository includes the following files:

- `maml_aci.py`: The main script that performs the MAML training and evaluation on the ACI dataset.
- `utils.py`: Utility functions used in the MAML implementation, including data preprocessing, model definition, training, and result visualization.
- `statics.py`: Static variables and configurations used in the MAML implementation, such as hyperparameters, file paths, and device selection.
- `requirements.txt`: The list of required Python packages and their versions.

## Dataset

The ACI IoT dataset can be found and accessed at the following links: [ACI-IoT-2023-Kaggle](https://www.kaggle.com/datasets/emilynack/aci-iot-network-traffic-dataset-2023), [ACI-IoT-2023-IEEE](https://ieee-dataport.org/documents/aci-iot-network-traffic-dataset-2023).

The dataset contains network traffic data from an IoT environment, including various attack types such as DDoS, Reckon, and Spoofing. The goal is to classify the network traffic data into malicious or benign.

## Usage

To run the MAML implementation on the ACI dataset, follow these steps:

1. Install the required packages listed in `requirements.txt` by running the command `pip install -r requirements.txt`.
2. Make sure you have the ACI dataset file (`ACI-IoT-2023.csv`) in the same directory as the script, or specify the correct file path in the `statics.py` file.
3. Open the `maml_aci.py` script and modify the hyperparameters and file paths if needed.
4. Run the `maml_aci.py` script using Python: `python maml_aci.py`.

The script will perform the MAML training and evaluation on the ACI dataset, generating the necessary output files and visualizations in the specified output directory.

## Results

The results of the MAML training and evaluation on the ACI dataset will be saved in the output directory (`./Runs`). The results include:

- Confusion matrices for each attack type.
- Plots of the training and validation losses.
