from random import random, getrandbits


class Population:

    size = 0
    population = []
    sex = []
    gt = []
    pt = []
    rhesusGt = []
    rhesusPt = []

    def __init__(self,
                 genotype=('O', 'O'),
                 rf_genotype=('+', '+'),
                 sex=None,
                 fitness=None,
                 offspringProb=0.055
                 ):

        self.genotype = genotype
        self.rf_genotype = rf_genotype

        if sex in ['f', 'm']:
            self.sex = sex
        else:
            self.sex = 'f' if random() < 0.5 else 'm'

        self.setPhenotype()
        self.setRfPhenotype()

        self.setFitness(fitness)  # dependent on phenotype
        self.setOffspringProb(offspringProb)

    def genOffspring(self, otherPerson):
        child_genotype = (self.genotype[getrandbits(
            1)], otherPerson.genotype[getrandbits(1)])
        child_rf_genotype = (self.rf_genotype[getrandbits(
            1)], otherPerson.rf_genotype[getrandbits(1)])

        child = Person(genotype=child_genotype, rf_genotype=child_rf_genotype)

        return child

    def genMutatedOffspring(self,
                            otherPerson,
                            genotypeMutation=None,
                            rfGenotypeMutation=None):

        child = self.genOffspring(otherPerson=otherPerson)

        if genotypeMutation is not None:
            child.mutateGenome(mutateTo=genotypeMutation)
        if rfGenotypeMutation is not None:
            child.mutateRfGenome(mutateTo=rfGenotypeMutation)

        return child

    def mutateGenome(self, mutateTo):
        self.genotype = (self.genotype[0], mutateTo) if bool(
            getrandbits(1)) else (mutateTo, self.genotype[1])
        self.setPhenotype()

    def mutateRfGenome(self, mutateTo):
        self.rf_genotype = (self.rf_genotype[0], mutateTo) if bool(
            getrandbits(1)) else (mutateTo, self.rf_genotype[1])
        self.setRfPhenotype()

    def dies(self):
        return True if random() < self.fitness else False

    def setFitness(self, value=None):
        fitness = 0.002
        if value is not None:
            self.fitness = value
        else:
            if self.phenotype == 'O':
                self.fitness = fitness
            elif self.phenotype == 'A':
                self.fitness = fitness / 2
            elif self.phenotype == 'B':
                self.fitness = fitness * 2
            elif self.phenotype == 'AB':
                self.fitness = fitness / 4

    def setOffspringProb(self, value=None):
        if value is not None:
            self.offspringProb = value
        else:
            pass  # implement phenotype dependend die probability

    def possibleOffspring(self):
        return True if random() < self.offspringProb else False

    def setPhenotype(self):
        if self.genotype == ('O', 'O'):
            self.phenotype = 'O'
        elif self.genotype == ('A', 'A') or \
                self.genotype == ('A', 'O') or \
                self.genotype == ('O', 'A'):
            self.phenotype = 'A'
        elif self.genotype == ('B', 'B') or \
                self.genotype == ('B', 'O') or \
                self.genotype == ('O', 'B'):
            self.phenotype = 'B'
        elif self.genotype == ('A', 'B') or \
                self.genotype == ('B', 'A'):
            self.phenotype = 'AB'

    def setRfPhenotype(self):
        if self.rf_genotype == ('+', '+') or \
           self.rf_genotype == ('-', '+') or \
           self.rf_genotype == ('+', '-'):
            self.rf_phenotype = '+'
        else:  # self.rf_genotype == ('-', '-'):
            self.rf_phenotype = '-'
