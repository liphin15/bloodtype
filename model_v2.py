
import numpy as np
from matplotlib import pyplot as plt
import random
from bloodtype_v2 import BloodType

from tqdm import tqdm
import pickle

import ipdb

random.seed(1234)
model = BloodType(10000, timesteptype='m')

# iterations
for i in tqdm(range(1000)):
    model.step()
model.printState()

model.plotSize()
model.plotSex(ratio=True)
model.plotPhenotype(ratio=True)
