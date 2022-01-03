import numpy as np
from matplotlib import pyplot as plt
import random
from bloodtype import BloodType
from tqdm import tqdm
import pickle

import ipdb


if __name__ == '__main__':
    # model = BloodType.load('model.pickle')
    random.seed(1234)
    # model = BloodType(10000)
    # model.load('1st_model.pickle')
    with open('1st_model.pickle', 'rb') as f:
        model = pickle.load(f)
    # model = BloodType.load('1st_model.pickle')
    ipdb.set_trace()
    model.plot_sex()
