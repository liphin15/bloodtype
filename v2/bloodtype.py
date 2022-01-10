import numpy as np
from matplotlib import pyplot as plt
import random
from person import Person
import pickle
import os
import numba as nb

import ipdb

# default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

default_colors = ['#1f77b4',
                  '#ff7f0e',
                  '#2ca02c',
                  '#d62728',
                  '#9467bd',
                  '#8c564b',
                  '#e377c2',
                  '#7f7f7f',
                  '#bcbd22',
                  '#17becf',
                  '#1a55FF']


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
    filename_age_penalty = 'age_penalty.csv'
    filename_birth_distribution = 'birth_distribution.csv'

    def __init__(self,
                 startsize,
                 timesteptype='w',
                 deathRate=0.01,
                 birthRate=0.014,
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
        self.loadAgePenalty()
        self.loadBirthDistribution()

        # plt.ion()
        # self.fig, self.ax = plt.subplots(nrows=1, ncols=2)

        self.populationsize = startsize
        self.population = [Person(age=20.0)
                           for i in range(self.populationsize)]
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

    def loadAgePenalty(self):
        self.age_penalty = np.genfromtxt(self.filename_age_penalty,
                                         delimiter=',',
                                         skip_header=1)
        prop = (self.age_penalty[:, 2]
                / np.sum(self.age_penalty[:, 2])).reshape(-1, 1)
        self.age_penalty = np.hstack([self.age_penalty, prop])

    def loadBirthDistribution(self):
        self.birth_distribution = np.genfromtxt(self.filename_birth_distribution,
                                                delimiter=',',
                                                skip_header=1)
        prop = (self.birth_distribution[:, 2]
                / np.sum(self.birth_distribution[:, 2])).reshape(-1, 1)
        cdf = np.cumsum(prop).reshape(-1, 1)
        self.birth_distribution = np.hstack([self.birth_distribution,
                                             prop,
                                             cdf])

    def step(self, bt_mutation=None, rf_mutation=None, mutations=0):
        self.timesteps += self.timestep
        # Parallel(n_jobs=5)(p.updateAge(self.timestep) for p in self.population)

        for p in self.population:
            p.updateAge(self.timestep)

        self.killPersons()
        self.generateOffsprings(bt_mutation=bt_mutation,
                                rf_mutation=rf_mutation,
                                mutations=mutations)

        self.logState()
        # update plots
        # self.updateSize()

    # @nb.jit
    def getDeathlist(self):
        prop_bt = np.array([self.fitness[p.bt_pt] for p in self.population])

        prop_age = np.array([self.age_penalty[((self.age_penalty[:, 0] - p.age) <= 0).sum() - 1, 3]
                             for p in self.population])

        prop = prop_bt * prop_age
        # prop = prop / np.sum(prop)

        self.deaths = round(self.populationsize * self.getDeathRate())

        # @nb.jit(nopython=True, parallel=True)
        def indexList(deaths, prop):
            indexList = np.empty((deaths), dtype=np.int32)
            # for i in nb.prange(deaths):
            for i in range(deaths):
                cdf = np.cumsum(prop)
                cdf = cdf / cdf[-1]
                indexList[i] = np.sum(cdf - random.random() < 0)
                prop[indexList[i]] = 0
            return np.flip(np.sort(indexList))
        return indexList(self.deaths, prop)

    # def getDeathlist(self):
    #     prop_bt = np.array([self.fitness[p.bt_pt] for p in self.population])
    #
    #     prop_age = np.array([self.age_penalty[((self.age_penalty[:, 0] - p.age) <= 0).sum() - 1, 3]
    #                          for p in self.population])
    #
    #     prop = prop_bt * prop_age
    #     # prop = prop / np.sum(prop)
    #
    #     cdf = np.cumsum(prop)
    #     cdf = cdf / cdf[-1]
    #     self.deaths = round(self.populationsize * self.getDeathRate())
    #
    #     @nb.jit(nopython=True, parallel=True)
    #     def indexList(deaths, cdf):
    #         indexList = np.empty((deaths), dtype=np.int32)
    #         for i in nb.prange(deaths):
    #             while True:
    #                 pos = np.sum(cdf - random.random() < 0)
    #                 if pos not in indexList:
    #                     indexList[i] = pos
    #                     break
    #         return np.flip(np.sort(indexList))
    #     return indexList(self.deaths, cdf)

    def killPersons(self):
        deathlist = self.getDeathlist()
        # deathlist.sort(reverse=True)
        # if deathlist is not None:
        if deathlist.shape != (0,):
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

        parents = self.choseParents(self.births)

        for i in range(mutations):
            i, j = parents[i]
            if (i, j) == (-1, -1):
                break
            child = self.population[i].genMutatedOffspring(
                self.population[j],
                bt_gtMutation=bt_mutation,
                rf_gtMutation=rf_mutation)
            self.population.append(child)
            self.populationsize = self.populationsize + 1

        for i in range(self.births - mutations):
            i, j = parents[mutations+i]
            if (i, j) == (-1, -1):
                break
            child = self.population[i].genOffspring(self.population[j])
            self.population.append(child)
            self.populationsize = self.populationsize + 1

    def choseParents(self, size=1):
        sexs = np.array([p.sex for p in self.population])

        prop_i = np.array([self.birth_distribution[(
            (self.birth_distribution[:, 0] - p.age) <= 0).sum() - 1, 3] for p in self.population])
        if (prop_i == 0).any():
            return np.full((size,2), -1)
        # print(prop_i, prop_i.max())

        # @nb.jit(nopython=True, parallel=True)
        # @nb.njit
        def indexList(sexs, prop_i, size=1):
            ij = np.empty((size,2), dtype=np.int32)
            # for i in nb.prange(size):
            for i in range(size):
                cdf_i = np.cumsum(prop_i)
                cdf_i = cdf_i / cdf_i[-1]
                ij[i,0] = np.sum(cdf_i - random.random() < 0)

                prop_j = prop_i * (sexs != sexs[i])
                cdf_j = np.cumsum(prop_j)
                cdf_j = cdf_j / cdf_j[-1]
                ij[i,1]  = np.sum(cdf_j - random.random() < 0)
            return ij

        # ipdb.set_trace()
        # j = np.random.choice(np.arange(self.populationsize)[
        #                      sex_age[:, 0] != self.population[i].sex], 1)[0]
        return indexList(sexs, prop_i, size)

    # def choseParent(self):
    #     sexs = np.array([p.sex for p in self.population])
    #
    #     prop_i = np.array([self.birth_distribution[(
    #         (self.birth_distribution[:, 0] - p.age) <= 0).sum() - 1, 3] for p in self.population])
    #     if (prop_i == 0).any():
    #         return np.full((size,2), -1)
    #     # print(prop_i, prop_i.max())
    #
    #     cdf_i = np.cumsum(prop_i)
    #     cdf_i = cdf_i / cdf_i[-1]
    #     i = np.sum(cdf_i - random.random() < 0)
    #
    #     prop_j = prop_i * (sexs != sexs[i])
    #     cdf_j = np.cumsum(prop_j)
    #     cdf_j = cdf_j / cdf_j[-1]
    #     j = np.sum(cdf_j - random.random() < 0)
    #
    #     # ipdb.set_trace()
    #     # j = np.random.choice(np.arange(self.populationsize)[
    #     #                      sex_age[:, 0] != self.population[i].sex], 1)[0]
    #     return (i,j)

    def logState(self):
        bt_ptFromPopulation = [p.bt_pt for p in self.population]
        bt_ptCounts = [bt_ptFromPopulation.count(
            p) for p in self.bt_pt]

        btrf_ptFromPopulation = [p.bt_pt + p.rf_pt for p in self.population]
        btrf_ptCounts = [btrf_ptFromPopulation.count(
            p) for p in self.btrf_pt]
        sexs = np.unique([p.sex for p in self.population],
                         return_counts=True)[1]
        ages = [self.age_penalty[(
            (self.age_penalty[:, 0] - p.age) <= 0).sum() - 1, 0] for p in self.population]
        age_groups = [ages.count(ag) for ag in self.age_penalty[:, 0]]

        currentstate = [self.timesteps,
                        len(self.population),  # self.populationsize,
                        self.deaths,
                        self.births,
                        sexs,
                        bt_ptCounts,
                        btrf_ptCounts,
                        age_groups,
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
            plt.savefig(self.plots_dir + "populationsize.png")
            plt.close()
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
            plt.close()
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
            plt.close()
        else:
            plt.show()

    def plotAgeGroups(self, ratio=False, save=False):
        # x = np.cumsum([state[0] for state in self.states])
        x = np.array([state[0] for state in self.states])
        y = np.cumsum([state[7] for state in self.states], axis=1)
        if ratio:
            y = (y.T / y[:, -1]).T
        # ipdb.set_trace()

        fig, ax = plt.subplots()

        for i in reversed(range(y.shape[1])):
            ax.fill_between(x,
                            y[:, i],
                            np.zeros(y[:, 0].shape) if i == 0 else y[:, i - 1],
                            color=default_colors[i],
                            alpha=1,
                            label='{}-{}'.format(int(self.age_penalty[i, 0]),
                                                 int(self.age_penalty[i, 1])))
        plt.legend()
        if save:
            self.checkDirectory(self.plots_dir)
            plt.savefig(self.plots_dir
                        + "age_distribution{}.png".format('_ratio' if ratio else ''))
            plt.close()
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
        self.__dict__.update(tmp_dict)
