import numpy as np
from matplotlib import pyplot as plt
import random
from person import Person
import pickle
import os

import ipdb

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


class BloodType:
    states = []
    population = []
    populationsize = 0
    bt_pt = ['O', 'A', 'B', 'AB']
    rf_pt = ['+', '-']
    bt_pt_colors = {
        'O': default_colors[0],
        'A': default_colors[1],
        'B': default_colors[2],
        'AB': default_colors[3]
    }
    btrf_pt = ['O+', 'A+', 'B+', 'AB+', 'O-', 'A-', 'B-', 'AB-']
    btrf_pt_colors = {
        'O+': default_colors[0],
        'A+': default_colors[1],
        'B+': default_colors[2],
        'AB+': default_colors[3],
        'O-': default_colors[4],
        'A-': default_colors[5],
        'B-': default_colors[6],
        'AB-': default_colors[7],
    }
    timestep = 7 / 365
    timesteptype = 'w'
    timesteps = 0
    timesteptypes = {'d': 1 / 365, 'w': 7 / 365, 'm': 1 / 12, 'y': 1}
    deathRate = 0.01
    deaths = 0
    birthRate = 0.011
    births = 0
    fitness = {
        'O': 1,
        'A': 1,
        'B': 1,
        'AB': 1
    }
    filename = 'model.pkl'
    plots_dir = './plots/'

    def __init__(self,
                 startsize,
                 timesteptype='w',
                 deathRate=0.01,
                 birthRate=0.013,
                 filename='model.pkl'):
        self.states = []
        self.population = []

        self.timestep = 7 / 365
        self.timesteptype = 'w'
        self.timesteps = 0
        self.deaths = 0
        self.births = 0

        self.deathRate = deathRate
        self.birthRate = birthRate

        self.setTimesteps(timesteptype)

        # plt.ion()
        # self.fig, self.ax = plt.subplots(nrows=1, ncols=2)

        self.populationsize = startsize
        self.population = [Person() for i in range(self.populationsize)]
        self.filename = filename

    def setFitness(self, fitness):
        self.fitness = fitness

    def setTimesteps(self, timesteptype):
        self.timesteptype = timesteptype
        self.timestep = self.timesteptypes[self.timesteptype]

    def setDeathRate(self, value):
        self.deathRate = value

    def getDeathRate(self):
        return self.deathRate * self.timestep

    def setBirthRate(self, value):
        self.birthRate = value

    def getBirthRate(self):
        return self.birthRate * self.timestep

    def step(self, bt_mutation=None, rf_mutation=None, mutations=0):

        self.killPersons()
        self.generateOffsprings(bt_mutation=bt_mutation,
                                rf_mutation=rf_mutation,
                                mutations=mutations)
        self.timesteps += self.timestep

        self.logState()
        # update plots
        # self.updateSize()

    def getDeathlist(self):
        cdf = np.cumsum([self.fitness[p.bt_pt] for p in self.population])
        cdf = cdf / cdf[-1]
        self.deaths = round(self.populationsize * self.getDeathRate())

        deathlist = [None] * self.deaths
        for i in range(self.deaths):
            while True:
                pos = np.sum(cdf - random.random() < 0)
                if pos not in deathlist:
                    deathlist[i] = pos
                    break
        return deathlist

    def killPersons(self):
        deathlist = self.getDeathlist()
        deathlist.sort(reverse=True)
        if deathlist is not None:
            for i in deathlist:
                self.removePerson(i)

    def removePerson(self, i):
        self.population.pop(i)
        self.populationsize = self.populationsize - 1

    def generateOffsprings(self,
                           bt_mutation=None,
                           rf_mutation=None,
                           mutations=0):
        self.births = int(np.round(self.populationsize * self.getBirthRate()))
        if bt_mutation not in self.bt_pt and \
                rf_mutation not in self.rf_pt:
            mutations = 0

        mutations = self.births if self.births < mutations else mutations
        # print(self.births, mutations)

        for i in range(mutations):
            i, j = self.choseParents()
            child = self.population[i].genMutatedOffspring(
                self.population[j],
                bt_gtMutation=bt_mutation,
                rf_gtMutation=rf_mutation)
            self.population.append(child)
            self.populationsize = self.populationsize + 1

        for i in range(self.births - mutations):
            i, j = self.choseParents()
            child = self.population[i].genOffspring(self.population[j])
            self.population.append(child)
            self.populationsize = self.populationsize + 1

    def generateOffspring(self):
        i = random.randint(0, self.populationsize - 1)
        sexs = np.array([p.sex for p in self.population])
        j = np.random.choice(np.arange(self.populationsize)[
                             sexs != self.population[i].sex], 1)[0]
        child = self.population[i].genOffspring(self.population[j])
        return child

    def choseParents(self):
        i = random.randint(0, self.populationsize - 1)
        sexs = np.array([p.sex for p in self.population])
        j = np.random.choice(np.arange(self.populationsize)[
                             sexs != self.population[i].sex], 1)[0]
        return (i, j)

    def logState(self):
        bt_ptFromPopulation = [p.bt_pt for p in self.population]
        bt_ptCounts = [bt_ptFromPopulation.count(
            p) for p in self.bt_pt]

        btrf_ptFromPopulation = [p.bt_pt + p.rf_pt for p in self.population]
        btrf_ptCounts = [btrf_ptFromPopulation.count(
            p) for p in self.btrf_pt]
        sexs = np.unique([i.sex for i in self.population],
                         return_counts=True)[1]
        currentstate = [self.timesteps,
                        len(self.population),  # self.populationsize,
                        self.deaths,
                        self.births,
                        sexs,
                        bt_ptCounts,
                        btrf_ptCounts,
                        ]

        self.states.append(currentstate)

    def checkDirectory(self, dir):
        if not os.path.isdir(dir):
            os.makedirs(dir)

    def updateSize(self):
        x = np.arange(0, self.timesteps)
        y = [state[2] for state in self.states]
        self.ax[0].fill_between(x, y, np.zeros(len(y)))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plotSize(self, cumulativ=False, save=False):
        # x = np.arange(0, self.timesteps, step=self.timestep)
        x = np.array([state[0] for state in self.states])
        y = np.array([state[1] for state in self.states])
        d = np.array([state[2] for state in self.states])
        b = np.array([state[3] for state in self.states])

        # import ipdb; ipdb.set_trace()
        fig, ax = plt.subplots()
        ax.fill_between(x,
                        y,
                        0,
                        color=default_colors[0],
                        alpha=1,
                        label='initial size' if cumulativ else 'populationsize')
        ax.fill_between(x,
                        y,
                        y - b,
                        color=default_colors[1],
                        alpha=1,
                        label='cumulative births' if cumulativ else 'total births')
        ax.fill_between(x,
                        y + d,
                        y,
                        color=default_colors[2],
                        alpha=1,
                        label='cumulative deaths' if cumulativ else 'total deaths')
        plt.legend()
        if save:
            self.checkDirectory(self.plots_dir)
            plt.savefig(self.plots_dir+"populationsize.png")
        else:
            plt.show()

    def plotBtPt(self, showRf=False, ratio=False, save=False):
        # x = np.cumsum([state[0] for state in self.states])
        x = np.array([state[0] for state in self.states])
        if showRf:
            y = np.cumsum([state[6] for state in self.states], axis=1)
        else:
            y = np.cumsum([state[5] for state in self.states], axis=1)
        if ratio:
            y = (y.T / y[:, -1]).T

        fig, ax = plt.subplots()
        if showRf:
            for i, p in enumerate(self.btrf_pt):
                # ax.plot(X, Y[:, i], color=self.bt_pt_colors[p], alpha=1.00, label=p, linewidth=0.1)
                ax.fill_between(x,
                                y[:, i],
                                np.zeros(
                                    y[:, 0].shape) if i == 0 else y[:, i - 1],
                                color=self.btrf_pt_colors[p],
                                alpha=1,
                                label=p
                                )
        else:
            for i, p in enumerate(self.bt_pt):
                # ax.plot(X, Y[:, i], color=self.bt_pt_colors[p], alpha=1.00, label=p, linewidth=0.1)
                ax.fill_between(x,
                                y[:, i],
                                np.zeros(
                                    y[:, 0].shape) if i == 0 else y[:, i - 1],
                                color=self.bt_pt_colors[p],
                                alpha=1,
                                label=p
                                )
        plt.legend()
        if save:
            self.checkDirectory(self.plots_dir)
            plt.savefig(self.plots_dir
                        + "bloodtype{}{}.png".format('_rf' if showRf else '',
                                                     '_ratio' if ratio else ''))
        else:
            plt.show()

    def plotSex(self, ratio=False, save=False):
        # x = np.cumsum([state[0] for state in self.states])
        x = np.array([state[0] for state in self.states])
        y = np.cumsum([state[4] for state in self.states], axis=1)
        if ratio:
            y = (y.T / y[:, -1]).T

        fig, ax = plt.subplots()
        ax.fill_between(x, y[:, 0], np.zeros(y[:, 0].shape),
                        color=default_colors[1], alpha=1, label='f')
        ax.fill_between(x, y[:, 1], y[:, 0],
                        color=default_colors[0], alpha=1, label='m')
        plt.legend()
        if save:
            self.checkDirectory(self.plots_dir)
            plt.savefig(self.plots_dir
                        + "sex_distribution{}.png".format('_ratio' if ratio else ''))
        else:
            plt.show()

    def printState(self):
        print("step: {:3.2f}".format(self.states[-1][0]),
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
