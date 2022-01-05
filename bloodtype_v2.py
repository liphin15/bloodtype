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
    phenotypeColors = {
        'O': default_colors[0],
        'A': default_colors[1],
        'B': default_colors[2],
        'AB': default_colors[3]
    }
    timestep = 7 / 365
    timesteps = 0
    timesteptype = 'w'
    timesteptypes = {'d': 1 / 365, 'w': 7 / 365, 'm': 1 / 12, 'y': 1}
    deathRate = 0.01
    deaths = 0
    birthRate = 0.013
    births = 0
    fitness = {
        'O': 1,
        'A': 1,
        'B': 1,
        'AB': 1
    }
    filename='model.pkl'

    def __init__(self,
                 startsize,
                 timesteptype = 'w',
                 deathRate = 0.01,
                 birthRate = 0.013,
                 filename='model.pkl'):
        self.deathRate = deathRate
        self.birthRate = birthRate

        self.setTimesteps(timesteptype)

        # plt.ion()
        # self.fig, self.ax = plt.subplots(nrows=1, ncols=2)

        self.populationsize = startsize
        self.population = [Person() for i in range(self.populationsize)]
        self.filename = filename

    def setTimesteps(self, timesteptype):
        self.timesteptype = timesteptype
        self.timestep = self.timesteptypes[self.timesteptype]

    def getDeathRate(self):
        return self.deathRate * self.timestep

    def getBirthRate(self):
        return self.birthRate * self.timestep

    def step(self, size=1):

        self.killPersons()
        self.generateOffsprings()
        self.timesteps += self.timestep

        self.logState()
        # update plots
        # self.updateSize()

    def getDeathlist(self):
        cdf = np.cumsum([self.fitness[p.bt_pt] for p in self.population])
        cdf = cdf/cdf[-1]
        self.deaths = round(self.populationsize*self.getDeathRate())
        
        deathlist = [None]*self.deaths
        for i in range(self.deaths):
            while True:
                pos = np.sum(cdf - random.random() < 0)
                if pos not in deathlist:
                    deathlist[i] = pos
                    break
        return deathlist

    def killPersons(self):
        deathlist = self.getDeathlist().sort(reverse=True)
        if deathlist is not None:
            for i in deathlist:
                self.removePerson(i)

    def removePerson(self, i_person):
        self.population.pop(i_person)
        self.populationsize -= 1

    def generateOffsprings(self):
        self.births = round(self.populationsize*self.getBirthRate())
        for i in range(self.births):
            child = self.generateOffspring()
            self.population.append(child)
            self.populationsize += 1

    def generateOffspring(self):
        i = random.randint(0, self.populationsize - 1)
        sexs = np.array([p.sex for p in self.population])
        j = np.random.choice(np.arange(self.populationsize)[sexs != self.population[i].sex],1)[0]
        child = self.population[i].genOffspring(self.population[j])
        return child

    def logState(self):
        phenotypeFromPopulation = [p.bt_pt for p in self.population]
        phenoptypeCounts = [phenotypeFromPopulation.count(
            p) for p in self.phenotypes]
        sexs = np.unique([i.sex for i in self.population],
                         return_counts=True)[1]
        currentstate = [self.timesteps,
                        self.populationsize,
                        self.deaths,
                        self.births,
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

    def plotSize(self):
        x = np.arange(0, self.timesteps, step=self.timestep)
        y = np.array([state[1] for state in self.states])
        d = np.array([state[2] for state in self.states])
        b = np.array([state[3] for state in self.states])
        
        # import ipdb; ipdb.set_trace()
        fig, ax = plt.subplots()
        ax.fill_between(x, y, 0, color=default_colors[0], alpha=1, label='populationsize')
        ax.fill_between(x, y, y-b, color=default_colors[1], alpha=1, label='births')
        ax.fill_between(x, y+d, y, color=default_colors[2], alpha=1, label='deaths')
        plt.legend()
        plt.show()

    def plotPhenotype(self, ratio=False):
        x = np.cumsum([state[0] for state in self.states])
        y = np.cumsum([state[5] for state in self.states], axis=1)
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
        x = np.cumsum([state[0] for state in self.states])
        y = np.cumsum([state[4] for state in self.states], axis=1)
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
        print("step: {:3.2f}".format(np.sum([s[0] for s in self.states])),
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
