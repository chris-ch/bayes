from random import random, choice
from datetime import date

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
    experience = DiscreteExperience(['1', '2', '3', '4', '5', '6'], lambda x: x in ('2', '3', '1'))
    for count in range(100000):
        if random() >= 0.6:
            experience.new_outcome('1')

        else:
            experience.new_outcome(choice(['2', '3', '4', '5', '6']))

    experience.discrete(1000.).plot()
    pyplot.show()


def test_stock_returns():
    prices_data = [
        ('2017-05-25', 128.47),
        ('2017-05-24', 127.48),
        ('2017-05-23', 127.97),
        ('2017-05-22', 131.14),
        ('2017-05-19', 128.69),
        ('2017-05-18', 128.47),
        ('2017-05-17', 128.69),
        ('2017-05-16', 133.25),
        ('2017-05-15', 135.16),
        ('2017-05-12', 134.68),
        ('2017-05-11', 134.39),
        ('2017-05-10', 134.06),
        ('2017-05-09', 137.45),
        ('2017-05-08', 138.43),
        ('2017-05-05', 141.07),
        ('2017-05-04', 139.39),
        ('2017-05-03', 138.37),
        ('2017-05-02', 139.79),
        ('2017-05-01', 138.27)
    ]
    prices = {'date': [date(int(field_date[:4]),int(field_date[5:7]),int(field_date[-2:])) for field_date, field_value in prices_data],
             'price': [float(field_price) for field_date, field_price in prices_data]
             }
    df = pandas.DataFrame(prices)
    print(df)


def test_pattern():
    pattern = """
    . . . . . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . . . . . .
    . . . . . . . O . . . . . . . . . . .
    . . . . . . . . . . . . . . X . X . .
    . . . . O . . . O . . . O . . . . . .
    . . . . . . O . . . O . . O . X . X .
    . . . . . . . . O . . . O . . . . . .
    . . . O . . O . . . X . . . . X . . .
    . . . . O . . . X . . . X . . . X . .
    . . . . . . . . . . X . X . X . . . .
    . . . O . O . . X . . . . . . . . . .
    . . . . O . . X . . . X . . X . . . .
    . . . O . . . . . . . . . . . . . . .
    . . . . O . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . . . . . .
    """
    rows = [''.join([symbol for symbol in row if symbol != ' ']) for row in pattern.split('\n') if len(row) > 0]
    coordinates_type_x = list()
    coordinates_type_o = list()
    coordinates_all = list()
    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(row):
            coordinates = (row_index, column_index)
            coordinates_all.append(coordinates)
            if column == 'O':
                coordinates_type_o.append(coordinates)

            elif column == 'X':
                coordinates_type_x.append(coordinates)

    print(coordinates_type_x)
    print(coordinates_type_o)
    experience = DiscreteExperience(coordinates_all, lambda coord: coord in coordinates_type_x)
    for coordinates in coordinates_type_x:
        experience.new_outcome(coordinates)

    experience.discrete(1000.).plot()
    pyplot.show()


if __name__ == '__main__':
    test_pattern()
