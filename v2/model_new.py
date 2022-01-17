import random
from bloodtype import BloodType

from tqdm import tqdm, trange

import ipdb

random.seed(1234)
model = BloodType(10000,
                  deathRate=0.01,
                  birthRate=0.1,
                  timesteptype='m')

# model.step()
# ipdb.set_trace()
# model.step(bt_mutation='A', mutations=10)
# model.step(bt_mutation='A', mutations=10)
# ipdb.set_trace()


# iterations
t = trange(1000)
for i in t:
    t.set_description("Population Size {}".format(model.populationsize))
    t.refresh()
    model.step()
model.print_state()

# t = trange(1000)
# for i in t:
#     t.set_description("Population Size {}".format(model.populationsize))
#     t.refresh()
#     model.step(bt_mutation='A', mutations=10)
#     model.step(bt_mutation='B', mutations=10)
#
# model.print_state()

# t = trange(1000000)
# for i in t:
#     t.set_description("Population Size {}".format(model.populationsize))
#     t.refresh()
#     model.step()
# model.print_state()

# model.setFitness(fitness={
#         'O': 100,
#         'A': 1,
#         'B': 5,
#         'AB': 20
#     })
# model.set_death_rate(value=0.8)
#
# for i in tqdm(range(36)):
#     model.step()
# model.printState()
#
# model.setFitness(fitness={
#         'O': 1,
#         'A': 1,
#         'B': 1,
#         'AB': 1
#     })
# model.setDeathRate(value=0.01)
#
#
# t = trange(1000)
# for i in t:
#     t.set_description("Population Size {}".format(model.populationsize))
#     t.refresh()
#     model.step()
# model.printState()


# model.plotSize()
model.plot_size(save=True)
model.plot_sex(ratio=True, save=True)
model.plot_bt_pt(save=True)
model.plot_bt_pt(showRf=True, save=True)
model.plot_age_groups(ratio=False, save=True)
# ipdb.set_trace()
