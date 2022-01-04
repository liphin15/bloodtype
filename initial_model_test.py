import numpy as np
from matplotlib import pyplot as plt
import random
from bloodtype import BloodType
from tqdm import tqdm
import pickle

import ipdb


if __name__ == '__main__':
    random.seed(1234)
    model = BloodType(1000, '1st_model.pickle')

    print(np.unique([p.sex for p in model.persons], return_counts=True)[1])

    # iterations
    for i in tqdm(range(100)):
        model.step()
    model.printState()

    # generate childs with mutations
    for i in range(10):
        model.genMutatedOffspring(genotypeMutation='A')
        model.genMutatedOffspring(genotypeMutation='B')

    model.printState()

    # iterations
    for i in tqdm(range(1000000)):
        model.step()
    model.printState()

    # model.save()

    # model.plotSex()
    # model.plotSex(ratio=True)
    model.plotPhenotype()
    model.plotPhenotype(ratio=True)
    ipdb.set_trace()
