
import numpy as np
from matplotlib import pyplot as plt
import random
from bloodtype import BloodType

from tqdm import tqdm
import pickle

import ipdb

random.seed(1234)
model = BloodType(1000, timesteptype='m', birthRate=0.1, deathRate=0.1)

# iterations

for i in tqdm(range(500)):
    model.step(rf_mutation='-', mutations=10)
model.printState()


for i in tqdm(range(5000)):
    model.step()
model.printState()


# model.plotSize()
# model.plotSize()
# model.plotSex(ratio=True)
# model.plotBtPt()
model.plotBtPt(showRf=True)
ipdb.set_trace()
