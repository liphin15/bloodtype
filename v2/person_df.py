
from random import random, getrandbits
import pandas as pd
import ipdb

df_pos = {
    'age': 0,
    'sex': 1,
    'bt_gt': 2,
    'rf_gt': 3,
    'fitness': 4,
    'offspring_prob': 5,
}


def gen_dataframe(age=0.0,
                  sex=None,
                  bt_gt='OO',
                  rf_gt='++',
                  fitness=None,
                  offspring_prob=None,
                  size=1
                  ):
    data = [gen_person(age, sex, bt_gt, rf_gt, fitness, offspring_prob)
            for i in range(size)]
    df = pd.DataFrame(data=data,
                      columns=['age',
                               'sex',
                               'bt_gt',
                               'bt_pt',
                               'rf_gt',
                               'rf_pt',
                               'fitness',
                               'offspring_prob'])
    return df


def gen_person(age=0.0,
               sex=None,
               bt_gt='OO',
               rf_gt='++',
               fitness=None,
               offspring_prob=None
               ):

    if sex not in ['f', 'm']:
        sex = 'f' if random() < 0.5 else 'm'

    bt_pt = get_bt_pt(bt_gt)
    rf_pt = get_rf_pt(rf_gt)

    fitness_factor = get_fitness(bt_pt, fitness)  # dependent on bt_pt
    offspring_prob = get_offspring_prob(bt_pt, offspring_prob)
    # ipdb.set_trace()

    return [age, sex, bt_gt, bt_pt, rf_gt, rf_pt, fitness_factor, offspring_prob]


def gen_offspring(parent1, parent2):
    child_bt_gt = parent1[df_pos['age']][getrandbits(1)] + \
        parent2[df_pos['age']][getrandbits(1)]
    child_rf_gt = parent1[df_pos['age']][getrandbits(1)] + \
        parent2[df_pos['age']][getrandbits(1)]

    child = gen_person(age=0.0,
                       sex=None,
                       bt_gt=child_bt_gt,
                       rf_gt=child_rf_gt,
                       fitness=None,
                       offspring_prob=None
                       )

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


def mutate_bt_gt(bt_gt, mutateTo):
    return mutate_gt(bt_gt, mutateTo)


def mutate_rf_gt(rf_gt, mutateTo):
    return mutate_gt(rf_gt, mutateTo)


def mutate_gt(gt, mutateTo):
    gt[random.getrandbits(1)] = mutateTo
    return gt


#
#
# def dies(self):
#     return True if random() < self.fitness else False
#
#
# def updateAge(self, time):
#     self.age = self.age + time


def get_fitness(bt_pt, value=None):
    if value is not None:
        return value
    else:
        fitness = 1
        if bt_pt == 'O':
            return fitness
        elif bt_pt == 'A':
            return fitness
        elif bt_pt == 'B':
            return fitness
        elif bt_pt == 'AB':
            return fitness


def get_offspring_prob(bt_gt, value=None):
    if value is not None:
        return value
    else:
        return 1


def possible_offspring(offspring_prob):
    return True if random() < offspring_prob else False


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


def get_rf_pt(rf_gt):
    if rf_gt == '++' or rf_gt == '-+' or rf_gt == '+-':
        return '+'
    else:
        return '-'
