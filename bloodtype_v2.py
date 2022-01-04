import numpy as np
from matplotlib import pyplot as plt
import random
from person import Person
import math
import pickle
from concurrent import futures

import ipdb

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


class BloodType:
    states = []
    population = []
    populationsize = 0
    phenotypes = ['O', 'A', 'B', 'AB']
    phenotypecolors = {
        'O': default_colors[0],
        'A': default_colors[1],
        'B': default_colors[2],
        'AB': default_colors[3]
    }
    timestep = 7 / 365
    timesteps = 0
    timesteptype = 'w'
    timesteptypes = {'d': 1 / 365, 'w': 7 / 365, 'm': 1 / 12, 'y': 1}
    dieRate = 0.01

    def __init__(self,
                 startsize,
                 filename='model.pkl'):
        plt.ion()
        self.fig, self.ax = plt.subplots(nrows=1, ncols=2)

        self.populationsize = startsize
        self.population = [Person() for i in range(self.populationsize)]
        self.filename = filename

    def getDieRate(self):
        return self.dieRate * self.timestep

    def getBirthRate(self):
        pass

    def step(self, size=1):

        self.timesteps += self.timestep

        # update plots
        self.updateSize()

    def removePerson(self, i_person):
        self.population.pop(i_person)
        self.populationsize -= 1

    def generateOffspring(self):
        pass

    def logState(self, step_size):
        phenotypeFromPopulation = [p.phenotype for p in self.population]
        phenoptypeCounts = [phenotypeFromPopulation.count(
            p) for p in self.phenotypes]
        sexs = np.unique([i.sex for i in self.population],
                         return_counts=True)[1]
        currentstate = [self.timesteps,
                        self.populationsize,
                        sexs,
                        phenoptypeCounts]

        self.states.append(currentstate)

    def updateSize(self):
        x = np.arange(0, self.timesteps)
        y = [state[2] for state in self.states]
        import ipdb
        ipdb.set_trace()
        self.ax[0].fill_between(x, y, np.zeros(len(y)))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plotPhenotype(self, ratio=False):
        x = np.cumsum([state[0] for state in self.states])
        y = np.cumsum([state[3] for state in self.states], axis=1)
        if ratio:
            y = (y.T / y[:, -1]).T

        fig, ax = plt.subplots()
        for i, p in enumerate(self.phenotypes):
            # ax.plot(X, Y[:, i], color=self.phenotypeColors[p], alpha=1.00, label=p, linewidth=0.1)
            ax.fill_between(x,
                            y[:, i],
                            np.zeros(y[:, 0].shape) if i == 0 else y[:, i - 1],
                            color=self.phenotypeColors[p],
                            alpha=1,
                            label=p
                            )
        plt.legend()
        plt.show()

    def plotSex(self, ratio=False):
        x = np.cumsum([entry[0] for entry in self.states])
        y = np.cumsum([entry[2] for entry in self.states], axis=1)
        if ratio:
            y = (y.T / y[:, -1]).T

        fig, ax = plt.subplots()
        ax.fill_between(x, y[:, 0], np.zeros(y[:, 0].shape),
                        color=default_colors[1], alpha=1, label='f')
        ax.fill_between(x, y[:, 1], y[:, 0],
                        color=default_colors[0], alpha=1, label='m')
        plt.legend()
        plt.show()

    def printState(self):
        print("step: {:8d}".format(np.sum([s[0] for s in self.states])),
              self.states[-1][1:]
              )

    def save(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self, f)
        # pickle.dump(self.__dict__, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            tmp_dict = pickle.load(f)

        self.__dict__.update(tmp_dict)
