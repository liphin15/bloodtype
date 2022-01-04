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
    state = []
    phenotypes = ['O', 'A', 'B', 'AB']
    phenotypeColors = {
        'O': default_colors[0],
        'A': default_colors[1],
        'B': default_colors[2],
        'AB': default_colors[3]
    }

    def __init__(self, start_pop, filename='model.pkl'):
        self.start_pop = start_pop
        self.persons = [Person() for i in range(self.start_pop)]
        self.filename = filename

    def step(self, size=1):
        if type(size) is not int:
            size = math.ceil(self.start_pop / 100)

        for i in range(size):
            i_person = random.randint(0, len(self.persons) - 1)
            child = self.checkPerson(i_person)  # return possible child
            if child is not None:
                self.persons.append(child)

    def checkPerson(self, i_person):

        if self.persons[i_person].possibleOffspring():
            sexs = np.array([p.sex for p in self.persons])
            if ('f' not in sexs) or ('m' not in sexs):
                print('No possible mating partner. ({})'.format(np.unique(sexs)))
                return None
            j_person = np.random.choice(np.arange(len(self.persons))[
                                        sexs != self.persons[i_person].sex], 1)[0]
            # j_person = random.randint(0, len(self.persons)-1)
            if self.persons[j_person].possibleOffspring():
                return self.persons[i_person].genOffspring(self.persons[j_person])

        if self.persons[i_person].dies():
            self.persons.pop(i_person)
            if len(self.persons) == 0:
                exit()
            return None

        return None

    def genMutatedOffspring(self,
                            genotypeMutation=None,
                            rfGenotypeMutation=None):

        i_person = random.randint(0, len(self.persons) - 1)
        sexs = np.array([p.sex for p in self.persons])
        j_person = np.random.choice(np.arange(len(self.persons))[
                                    sexs != self.persons[i_person].sex], 1)[0]
        child = self.persons[i_person].genMutatedOffspring(
            self.persons[j_person], genotypeMutation, rfGenotypeMutation)

        self.logState(1)

        self.persons.append(child)

    def logState(self, step_size):
        phenotypeFromPersons = [p.phenotype for p in self.persons]
        phenoptypeCounts = [phenotypeFromPersons.count(
            p) for p in self.phenotypes]
        self.state.append([
            1,
            np.unique([i.sex for i in self.persons],
                      return_counts=True)[1],
            phenoptypeCounts,
        ])

    def plotPhenotype(self, ratio=False):
        x = np.cumsum([entry[0] for entry in self.state])
        y = np.cumsum([entry[2] for entry in self.state], axis=1)
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
        x = np.cumsum([entry[0] for entry in self.state])
        y = np.cumsum([entry[1] for entry in self.state], axis=1)
        if ratio:
            y = (y.T / y[:, -1]).T

        fig, ax = plt.subplots()
        # ax.plot(X, Y[:,0], color=default_colors[1], alpha=1, label='f')
        # ax.plot(X, Y[:,1], color=default_colors[0], alpha=1, label='f+m')
        ax.fill_between(x, y[:, 0], np.zeros(y[:, 0].shape),
                        color=default_colors[1], alpha=1, label='f')
        ax.fill_between(x, y[:, 1], y[:, 0],
                        color=default_colors[0], alpha=1, label='f+m')
        plt.legend()
        plt.show()

    def printState(self):
        print("step: {:8d}".format(np.sum([s[0] for s in self.state])),
              self.state[-1][1:]
              )

    def save(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self, f)
        # pickle.dump(self.__dict__, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            tmp_dict = pickle.load(f)

        self.__dict__.update(tmp_dict)

    # @classmethod
    # def load(cls, filename):
    #     with open(filename, 'rb') as f:
    #         return pickle.load(f)
