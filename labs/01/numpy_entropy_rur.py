#!/usr/bin/env python3
import numpy as np
from collections import defaultdict

if __name__ == "__main__":
    # Load data distribution, each data point on a line
    dataDict = defaultdict(int)
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            datum = line.rstrip("\n")
            dataDict[datum] += 1

    labels = list(dataDict.keys())
    
    # Create a NumPy array containing the data distribution
    # counts
    data_distro = np.array(list(dataDict.values()))
    # counts -> probs
    data_distro = data_distro/np.sum(data_distro)

    # Load model distribution, each line `word \t probability`, creating
    # a NumPy array containing the model distribution
    modelDict = defaultdict(int)
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            datum, prob = line.split('\t')
            modelDict[datum] = float(prob)
    model_distro = np.array([modelDict[l] for l in labels])

    # Compute and print entropy H(data distribution)
    data_self_info = np.log(1/data_distro)
    data_entropy = np.sum(data_distro * data_self_info)

    # Compute and print cross-entropy H(data distribution, model distribution)
    model_self_info = np.log(1/model_distro)
    cross_entropy = np.sum(data_distro * model_self_info)

    # and KL-divergence D_KL(data distribution, model_distribution)
    kl = cross_entropy - data_entropy

    for number in data_entropy, cross_entropy, kl:
        print("{:.2f}".format(number))

