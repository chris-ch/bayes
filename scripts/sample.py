from random import random, choice

import numpy
from scipy import stats
import pandas
from matplotlib import pyplot


class UnitSegmentDistribution(object):

    def __init__(self, range_min, range_max, scaling=1.):
        self._verified_hypothesis = 0
        self._failed_hypothesis = 0
        self._range_min = min(range_min, range_max)
        self._range_max = max(range_min, range_max)
        self._scaling = scaling

    def compute(self, value):
        assert self._range_max >= value >= self._range_min
        span = self._range_max - self._range_min
        scaled_value = (value + self._range_min) / span
        count_runs = self._verified_hypothesis + self._failed_hypothesis
        return self._scaling * stats.binom.pmf(self._verified_hypothesis, count_runs, scaled_value)

    def add_failed_hypothesis(self):
        self._failed_hypothesis += 1

    def add_verified_hypothesis(self):
        self._verified_hypothesis += 1


class DiscreteExperience(object):

    def __init__(self, outcomes, hypothesis_func):
        """
        
        :param outcomes: list of possible outcomes
        :param hypothesis_func: function accepting an outcome and returning true when hypothesis is verified
        """
        self._hypothesis_func = hypothesis_func
        self._outcomes = outcomes
        self._confidence_level = UnitSegmentDistribution(0., 1., scaling=1.E3)

    def new_outcome(self, value):
        assert value in self._outcomes

        if self._hypothesis_func(value):
            self._confidence_level.add_verified_hypothesis()

        else:
            self._confidence_level.add_failed_hypothesis()

    def discrete(self, count):
        outcome_estimates = numpy.linspace(0., 1., count)
        df = pandas.DataFrame({'estimates': outcome_estimates})
        df['confidence_level'] = df.apply(lambda row: self._confidence_level.compute(row['estimates']), axis=1)
        normalization_factor = df['confidence_level'].sum()
        df['confidence_level'] = numpy.divide(df['confidence_level'], normalization_factor)
        return df.set_index('estimates')

    @property
    def confidence_level(self):
        return self._confidence_level


def test_coin():
    experience = DiscreteExperience(['H', 'T'], lambda x: x == 'H')
    experience.new_outcome('H')
    experience.new_outcome('H')
    experience.new_outcome('H')
    experience.new_outcome('T')
    experience.new_outcome('T')
    experience.new_outcome('T')
    experience.new_outcome('T')
    experience.new_outcome('T')
    experience.new_outcome('T')
    experience.discrete(1000.).plot()
    pyplot.show()


def test_dice():
    experience = DiscreteExperience(['1', '2', '3', '4', '5', '6'], lambda x: x in ('1', ))
    for count in range(100000):
        if random() >= 0.6:
            experience.new_outcome('1')

        else:
            experience.new_outcome(choice(['2', '3', '4', '5', '6']))

    experience.discrete(1000.).plot()
    pyplot.show()


if __name__ == '__main__':
    test_dice()
