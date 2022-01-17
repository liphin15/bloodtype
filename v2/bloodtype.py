import numpy as np
from matplotlib import pyplot as plt
import random
import pickle
import os
import numba as nb
import math
import pandas as pd

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

# df_pos = {
#     'age': 0,
#     'sex': 1,
#     'bt_gt': 2,
#     'rf_gt': 3,
#     'fitness': 4,
#     'offspring_prob': 5,
# }

df_cols = ['age',
           'sex',
           'bt_gt',
           'bt_pt',
           'rf_gt',
           'rf_pt',
           'fitness',
           'offspring_prob']


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
    time = 0
    timesteptypes = {'d': 1 / 365, 'w': 7 / 365, 'm': 1 / 12, 'y': 1}
    deathRate = 0.01
    deaths = 0
    birthRate = 0.011
    births = 0
    fitness = pd.Series({'O': 1, 'A': 1, 'B': 1, 'AB': 1})
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
        self.time = 0
        self.deaths = 0
        self.births = 0

        self.deathRate = deathRate
        self.birthRate = birthRate

        self.set_timesteps(timesteptype)
        self.load_age_penalty()
        self.load_birth_distribution()

        # plt.ion()
        # self.fig, self.ax = plt.subplots(nrows=1, ncols=2)

        self.populationsize = startsize
        self.population = self.gen_population(age='rand',
                                              size=self.populationsize)
        self.filename = filename

    def gen_population(self,
                       age=0.0,
                       sex=None,
                       bt_gt='OO',
                       rf_gt='++',
                       offspring_prob=None,
                       size=1
                       ):
        if age == 'rand':
            data = [self.gen_person(random.random()*50, sex, bt_gt, rf_gt)
                    for i in range(size)]
        else:
            data = [self.gen_person(age, sex, bt_gt, rf_gt)
                    for i in range(size)]
        df = pd.DataFrame(data=data,
                          columns=df_cols)
        return df

    def gen_person(self,
                   age=0.0,
                   sex=None,
                   bt_gt='OO',
                   rf_gt='++',
                   ):

        if sex not in ['f', 'm']:
            sex = 'f' if random.random() < 0.5 else 'm'

        bt_pt = self.get_bt_pt(bt_gt)
        rf_pt = self.get_rf_pt(rf_gt)

        fitness_factor = self.get_fitness(bt_pt)
        offspring_prob = self.get_offspring_prob(bt_pt)
        # ipdb.set_trace()

        return [age, sex, bt_gt, bt_pt, rf_gt, rf_pt, fitness_factor, offspring_prob]

    def get_fitness(self, bt_pt):
        return self.fitness[bt_pt]

    @staticmethod
    def get_offspring_prob(bt_gt, value=None):
        if value is not None:
            return value
        else:
            return 1

    @staticmethod
    def get_bt_pt(bt_gt):
        if bt_gt == 'OO':
            return 'O'
        elif bt_gt == 'AA' or bt_gt == 'AO' or bt_gt == 'OA':
            return 'A'
        elif bt_gt == 'BB' or bt_gt == 'BO' or bt_gt == 'OB':
            return 'B'
        elif bt_gt == 'AB' or bt_gt == 'BA':
            return 'AB'
        else:
            return None

    @staticmethod
    def get_rf_pt(rf_gt):
        if rf_gt == '++' or rf_gt == '-+' or rf_gt == '+-':
            return '+'
        else:
            return '-'

    def setFitness(self, fitness):
        self.fitness = fitness

    def set_timesteps(self, timesteptype):
        self.timesteptype = timesteptype
        self.timestep = self.timesteptypes[self.timesteptype]

    def set_death_rate(self, value):
        self.deathRate = value

    def get_death_rate(self):
        return self.deathRate * self.timestep

    def set_birth_rate(self, value):
        self.birthRate = value

    def get_birth_rate(self):
        return self.birthRate * self.timestep

    def load_age_penalty(self):
        self.age_penalty = np.genfromtxt(self.filename_age_penalty,
                                         delimiter=',',
                                         skip_header=1)
        prop = (self.age_penalty[:, 2]
                / np.sum(self.age_penalty[:, 2])).reshape(-1, 1)
        self.age_penalty = np.hstack([self.age_penalty, prop])

    def load_birth_distribution(self):
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
        # update time
        self.time += self.timestep

        # update age of population
        self.population['age'] += self.timestep

        self.kill_persons()
        self.generate_offsprings(bt_mutation=bt_mutation,
                                 rf_mutation=rf_mutation,
                                 mutations=mutations)

        self.log_state()
        # update plots
        # self.updateSize()

    # @nb.jit
    def get_deathlist(self):
        # prop_bt = np.array([self.fitness[bt]
        #                    for bt in self.population['bt_pt']])
        prop_bt = self.fitness[self.population['bt_pt']]

        prop_age = np.array([self.age_penalty[((self.age_penalty[:, 0] - age) <= 0).sum() - 1, 3]
                             for age in self.population['age']])

        prop = np.array(prop_bt * prop_age)
        # prop = prop / np.sum(prop)

        self.deaths = self.populationsize * self.get_death_rate()
        if random.random() < self.deaths-int(self.deaths):
            self.deaths = math.ceil(self.deaths)
        else:
            self.deaths = round(self.deaths)

        # @nb.jit(nopython=True, parallel=True)
        def indexlist(deaths, prop):
            indexList = np.empty((deaths), dtype=np.int32)
            for i in nb.prange(deaths):
                # for i in range(deaths):
                cdf = np.cumsum(prop)
                cdf = cdf / cdf[-1]
                indexList[i] = np.sum(cdf - random.random() < 0)
                prop[indexList[i]] = 0
            return np.flip(np.sort(indexList))
        return self.population.index[indexlist(self.deaths, prop)]

    def kill_persons(self):
        deathlist = self.get_deathlist()
        # deathlist.sort(reverse=True)
        # if deathlist is not None:
        if deathlist.shape != (0,):
            for i in deathlist:
                self.remove_person(i)

    def remove_person(self, index):
        self.population = self.population.drop(index)
        self.populationsize = self.populationsize - 1

    def add_person(self, person):
        self.population = self.population.append(person, ignore_index=True)
        self.populationsize = self.populationsize + 1

    def generate_offsprings(self,
                            bt_mutation=None,
                            rf_mutation=None,
                            mutations=0):
        self.births = self.populationsize * self.get_birth_rate()
        if random.random() < self.births-int(self.births):
            self.births = math.ceil(self.births)
        else:
            self.births = round(self.births)

        if bt_mutation not in self.bt_pt and \
                rf_mutation not in self.rf_pt:
            mutations = 0

        # reduce number of mutations to the proper size of births
        mutations = self.births if self.births < mutations else mutations

        # print(self.births, mutations)

        parents = self.chose_parents(self.births)

        for i in range(mutations):
            i, j = parents[i]
            if (i, j) == (-1, -1):
                break
            child = self.gen_mutated_offspring(i, j, bt_mutation, rf_mutation)
            self.add_person(child)

        for i in range(self.births - mutations):
            i, j = parents[mutations + i]
            if (i, j) == (-1, -1):
                break
            child = self.gen_offspring(i, j)
            self.add_person(child)

    def gen_offspring(self, pid1, pid2):
        child_bt_gt = self.population['bt_gt'][pid1][random.getrandbits(1)] + \
            self.population['bt_gt'][pid2][random.getrandbits(1)]
        child_rf_gt = self.population['rf_gt'][pid1][random.getrandbits(1)] + \
            self.population['rf_gt'][pid2][random.getrandbits(1)]

        child = self.gen_person(age=0.0,
                                sex=None,
                                bt_gt=child_bt_gt,
                                rf_gt=child_rf_gt,
                                )
        return pd.DataFrame(data=[child], columns=df_cols)

    def gen_mutated_offspring(self,
                              pid1,
                              pid2,
                              bt_gtMutation=None,
                              rf_gtMutation=None):

        child = self.gen_offspring(pid1, pid2)

        # ipdb.set_trace()
        if bt_gtMutation is not None:
            child['bt_gt'] = self.mutate_bt_gt(
                child['bt_gt'][0], bt_gtMutation)
            child['bt_pt'] = self.get_bt_pt(child['bt_gt'][0])
        if rf_gtMutation is not None:
            child['rf_gt'] = self.mutate_rf_gt(
                child['rf_gt'][0], rf_gtMutation)
            child['rf_pt'] = self.get_rf_pt(child['rf_gt'][0])

        return child

    @staticmethod
    def mutate_bt_gt(bt_gt, mutateTo):
        return BloodType.mutate_gt(bt_gt, mutateTo)

    @staticmethod
    def mutate_rf_gt(rf_gt, mutateTo):
        return BloodType.mutate_gt(rf_gt, mutateTo)

    @staticmethod
    def mutate_gt(gt, mutateTo):
        if bool(random.getrandbits(1)):
            return gt[0] + mutateTo
        else:
            return mutateTo + gt[1]

    def chose_parents(self, size=1):
        sexs = np.array(self.population['sex'])

        prop_i = np.array([self.birth_distribution[(
            (self.birth_distribution[:, 0] - age) <= 0).sum() - 1, 3] for age in self.population['age']])
        if (prop_i == 0).any():
            return np.full((size, 2), -1)
        # print(prop_i, prop_i.max())

        # @nb.njit
        # @nb.jit(nopython=True, parallel=True)
        def indexlist(sexs, prop_i, size=1):
            ij = np.empty((size, 2), dtype=np.int64)
            cdf_i = np.cumsum(prop_i)
            cdf_i = cdf_i / cdf_i[-1]
            for i in nb.prange(size):
                ij[i, 0] = np.sum(cdf_i - random.random() < 0)

                prop_j = prop_i * (sexs != sexs[i])
                cdf_j = np.cumsum(prop_j)
                cdf_j = cdf_j / cdf_j[-1]
                ij[i, 1] = np.sum(cdf_j - random.random() < 0)
            return ij

        # @nb.guvectorize([(nb.float64[:], nb.int64[:])], '(n)->(n)')
        # def parent1_id(cdf, pid):
        #     for i in nb.range(pid.shape[0]):
        #         pid[i] = np.sum(cdf - random.random() < 0)
        #
        # cdf_i = np.cumsum(prop_i)
        # i_pid = np.empty((size), dtype=np.int64)
        # parent1_id(cdf_i, i_pid)

        # @nb.guvectorize([(nb.float64[:], nb.int64[:])], '()->()', nopython=True)
        # def parent1_id(cdf, pid):
        #     for i in range(pid.shape[0]):
        #         pid[i] = np.sum(cdf - random.random() < 0)
        #
        # cdf_i = np.cumsum(prop_i)
        # i_pid = np.empty((size), dtype=np.int64)
        # parent1_id(cdf_i, i_pid)
        #
        # ipdb.set_trace()
        # j = np.random.choice(np.arange(self.populationsize)[
        #                      sex_age[:, 0] != self.population[i].sex], 1)[0]

        # try:
        index_list = indexlist(sexs, prop_i, size)
        # except Exception:
        #     ipdb.set_trace()

        return [[self.population.index[i], self.population.index[j]]
                for i, j in index_list]

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

    def log_state(self):
        bt_ptFromPopulation = self.population['bt_pt']
        bt_ptCounts = [bt_ptFromPopulation.str.count(p).sum()
                       for p in self.bt_pt]

        btrf_ptFromPopulation = self.population['bt_pt'] + \
            self.population['rf_pt']
        btrf_ptCounts = [btrf_ptFromPopulation.str.count(p).sum()
                         for p in self.btrf_pt]
        sexs = np.unique(self.population['sex'], return_counts=True)[1]
        ages = [self.age_penalty[(
            (self.age_penalty[:, 0] - age) <= 0).sum() - 1, 0] for age in self.population['age']]
        age_groups = [ages.count(ag) for ag in self.age_penalty[:, 0]]

        currentstate = [self.time,
                        len(self.population),  # self.populationsize,
                        self.deaths,
                        self.births,
                        sexs,
                        bt_ptCounts,
                        btrf_ptCounts,
                        age_groups,
                        ]

        self.states.append(currentstate)

    def check_directory(self, dir):
        if not os.path.isdir(dir):
            os.makedirs(dir)

    def updateSize(self):
        x = np.arange(0, self.time)
        y = [state[2] for state in self.states]
        self.ax[0].fill_between(x, y, np.zeros(len(y)))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot_size(self, cumulativ=False, save=False):
        # x = np.arange(0, self.time, step=self.timestep)
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
        plt.xlabel("years")
        plt.ylabel("")
        plt.tight_layout()
        if save:
            self.check_directory(self.plots_dir)
            plt.savefig(self.plots_dir + "populationsize.png")
            plt.close()
        else:
            plt.show()

    def plot_bt_pt(self, showRf=False, ratio=False, save=False):
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
        plt.xlabel("years")
        plt.ylabel("")
        plt.tight_layout()
        if save:
            self.check_directory(self.plots_dir)
            plt.savefig(self.plots_dir
                        + "bloodtype{}{}.png".format('_rf' if showRf else '',
                                                     '_ratio' if ratio else ''))
            plt.close()
        else:
            plt.show()

    def plot_sex(self, ratio=False, save=False):
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
        plt.xlabel("years")
        plt.ylabel("")
        plt.tight_layout()
        if save:
            self.check_directory(self.plots_dir)
            plt.savefig(self.plots_dir
                        + "sex_distribution{}.png".format('_ratio' if ratio else ''))
            plt.close()
        else:
            plt.show()

    def plot_age_groups(self, ratio=False, save=False):
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
        plt.xlabel("years")
        plt.ylabel("")
        plt.tight_layout()
        if save:
            self.check_directory(self.plots_dir)
            plt.savefig(self.plots_dir
                        + "age_distribution{}.png".format('_ratio' if ratio else ''))
            plt.close()
        else:
            plt.show()

    def print_state(self):
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
        self.__dict__.update(tmp_dict)
        self.__dict__.update(tmp_dict)
        self.__dict__.update(tmp_dict)
