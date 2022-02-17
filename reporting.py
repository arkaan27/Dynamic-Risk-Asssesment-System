"""
Name: reporting.py

Summary:
Module for reporting the machine learning model statistics to the app.py/ Flask api

Author: Arkaan Quanunga
Date: 17/02/2022

Functions:

- name: score_model
- input: data_path [str] The dataset path for the test data
- return: Saves the confusion matrix in output_model_path as "confusionmatrix.png"

"""

import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from diagnostics import model_predictions
import json
import os

###############Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config["test_data_path"])
output_model_path = os.path.join(config["output_model_path"])
test_dataset_path = os.path.join(test_data_path, "testdata.csv")


# Function for reporting
def score_model(data_path):
    """
    Plotting confusion matrix and saving the output in output_model_path
    :param data_path: [str] The test dataset path to be analysed for confusion matrix
    :return: Saves figure of confusion matrix in output_model_path as confusionmatrix.png
    """
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace
    df = pd.read_csv(data_path)

    predictions = model_predictions(data_path)
    y_test = df["exited"]
    cfm = metrics.confusion_matrix(y_test, predictions)

    # This code is taken from:  https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in cfm.flatten()]
    group_percentages = [
        "{0:.2%}".format(value) for value in cfm.flatten() / np.sum(cfm)
    ]
    labels = [
        f"{v1}\n{v2}\n{v3}"
        for v1, v2, v3 in zip(group_names, group_counts, group_percentages)
    ]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cfm, annot=labels, fmt="", cmap="Blues")

    plt.savefig(os.path.join(output_model_path, "confusionmatrix.png"))


if __name__ == '__main__':
    score_model(test_dataset_path)
