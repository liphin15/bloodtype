import random
from bloodtype import BloodType

from tqdm import tqdm, trange

import ipdb

random.seed(1234)
model = BloodType(10000,
                  deathRate=0.01,
                  birthRate=0.1,
                  timesteptype='w')

# iterations
t = trange(1000)
for i in t:
    t.set_description("Population Size {}".format(model.populationsize))
    t.refresh()
    model.step()
model.printState()

t = trange(1000)
for i in t:
    t.set_description("Population Size {}".format(model.populationsize))
    t.refresh()
    model.step(bt_mutation='A', mutations=10)
    model.step(bt_mutation='B', mutations=10)

model.printState()

t = trange(100000)
for i in t:
    t.set_description("Population Size {}".format(model.populationsize))
    t.refresh()
    model.step()
model.printState()

# model.setFitness(fitness={
#         'O': 100,
#         'A': 1,
#         'B': 5,
#         'AB': 20
#     })
# model.setDeathRate(value=0.8)
#
# for i in tqdm(range(36)):
#     model.step()
# model.printState()

# model.setFitness(fitness={
#         'O': 1,
#         'A': 1,
#         'B': 1,
#         'AB': 1
#     })
# model.setDeathRate(value=0.01)


# t = trange(1000)
# for i in t:
#     t.set_description("Population Size {}".format(model.populationsize))
#     t.refresh()
#     model.step()
# model.printState()


# model.plotSize()
model.plotSize(save=True)
# model.plotSex(ratio=True)
model.plotBtPt(save=True)
model.plotBtPt(showRf=True, save=True)
model.plotAgeGroups(ratio=False, save=True)
# ipdb.set_trace()
