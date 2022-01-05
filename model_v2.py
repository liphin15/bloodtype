
import numpy as np
from matplotlib import pyplot as plt
import random
from bloodtype_v2 import BloodType

from tqdm import tqdm
import pickle

import ipdb

random.seed(1234)
model = BloodType(1000, timesteptype='m')

# iterations
for i in tqdm(range(100)):
    model.step()
model.printState()

for i in tqdm(range(10)):
    model.step(bt_mutation='A', mutations=10)
    model.step(bt_mutation='B', mutations=10)
model.printState()

for i in tqdm(range(1000)):
    model.step()
model.printState()

model.setFitness(fitness = {
        'O': 10,
        'A': 1,
        'B': 20,
        'AB': 20
    })
model.setDeathRate(value=0.2)

for i in tqdm(range(200)):
    model.step()
model.printState()

model.setFitness(fitness = {
        'O': 1,
        'A': 1,
        'B': 1,
        'AB': 1
    })
model.setDeathRate(value=0.01)


for i in tqdm(range(1000)):
    model.step()
    if i % 50 ==0:
        model.printState()
model.printState()

# model.plotSize()
model.plotSize()
# model.plotSex(ratio=True)
model.plotBtPt()
ipdb.set_trace()
