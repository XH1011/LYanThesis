import pickle
import scipy.io as sio

import numpy as np
from scipy.io import loadmat
import random
from scipy.io import savemat
from sklearn.model_selection import train_test_split

import hdf5storage
import tensorflow as tf
# from data_provider import *
import matplotlib.pyplot as plt
# import nextbatch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#指定在第0块GPU上跑
import math
import scipy.io as scio
import pandas as pd
import numpy as np
import pickle


#latent_dim 128
# file_name='../[Step1]diffusion-encoded/DCAE_cwru/results/en_test.pkl'
file_name='../[Step1]diffusion-encoded/DCAE_chopper/results/en_test.pkl'
with open(file_name, "rb") as f:
     data = pickle.load(f)
data = np.squeeze(data)
data = data.T
data = np.float32(data)
labels=[]
for i in range(1,6):
    labels.extend([i]*300)
labels = np.array(labels)
labels = labels.reshape(-1,1)
# sio.savemat('data/ours/res101_cwru.mat', {'features': data, 'labels':labels})
sio.savemat('data/ours/res101_chopper.mat', {'features': data, 'labels':labels})
