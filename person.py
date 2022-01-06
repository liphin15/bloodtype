
from random import random, getrandbits


class Person:

    def __init__(self,
                 bt_gt=('O', 'O'),
                 rf_gt=('+', '+'),
                 sex=None,
                 fitness=None,
                 offspringProb=0.055
                 ):

        self.bt_gt = bt_gt
        self.rf_gt = rf_gt

        if sex in ['f', 'm']:
            self.sex = sex
        else:
            self.sex = 'f' if random() < 0.5 else 'm'

        self.setbt_pt()
        self.setRfbt_pt()

        self.setFitness(fitness)  # dependent on bt_pt
        self.setOffspringProb(offspringProb)

    def genOffspring(self, otherPerson):
        child_bt_gt = (self.bt_gt[getrandbits(1)],
                       otherPerson.bt_gt[getrandbits(1)])
        child_rf_gt = (self.rf_gt[getrandbits(
            1)], otherPerson.rf_gt[getrandbits(1)])

        child = Person(bt_gt=child_bt_gt, rf_gt=child_rf_gt)

        return child

    def genMutatedOffspring(self,
                            otherPerson,
                            bt_gtMutation=None,
                            rf_gtMutation=None):

        child = self.genOffspring(otherPerson=otherPerson)

        if bt_gtMutation is not None:
            child.mutateBtGenome(mutateTo=bt_gtMutation)
        if rf_gtMutation is not None:
            child.mutateRfGenome(mutateTo=rf_gtMutation)

        return child

    def mutateBtGenome(self, mutateTo):
        self.bt_gt = (self.bt_gt[0], mutateTo) if bool(
            getrandbits(1)) else (mutateTo, self.bt_gt[1])
        self.setbt_pt()

    def mutateRfGenome(self, mutateTo):
        self.rf_gt = (self.rf_gt[0], mutateTo) if bool(
            getrandbits(1)) else (mutateTo, self.rf_gt[1])
        self.setRfbt_pt()

    def dies(self):
        return True if random() < self.fitness else False

    def setFitness(self, value=None):
        fitness = 0.002
        if value is not None:
            self.fitness = value
        else:
            if self.bt_pt == 'O':
                self.fitness = fitness
            elif self.bt_pt == 'A':
                self.fitness = fitness / 2
            elif self.bt_pt == 'B':
                self.fitness = fitness * 2
            elif self.bt_pt == 'AB':
                self.fitness = fitness / 4

    def setOffspringProb(self, value=None):
        if value is not None:
            self.offspringProb = value
        else:
            pass  # implement bt_pt dependend die probability

    def possibleOffspring(self):
        return True if random() < self.offspringProb else False

    def setbt_pt(self):
        if self.bt_gt == ('O', 'O'):
            self.bt_pt = 'O'
        elif self.bt_gt == ('A', 'A') or \
                self.bt_gt == ('A', 'O') or \
                self.bt_gt == ('O', 'A'):
            self.bt_pt = 'A'
        elif self.bt_gt == ('B', 'B') or \
                self.bt_gt == ('B', 'O') or \
                self.bt_gt == ('O', 'B'):
            self.bt_pt = 'B'
        elif self.bt_gt == ('A', 'B') or \
                self.bt_gt == ('B', 'A'):
            self.bt_pt = 'AB'
        else:
            self.bt_pt = None

    def setRfbt_pt(self):
        if self.rf_gt == ('+', '+') or \
           self.rf_gt == ('-', '+') or \
           self.rf_gt == ('+', '-'):
            self.rf_pt = '+'
        else:  # self.rf_gt == ('-', '-'):
            self.rf_pt = '-'
