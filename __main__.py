from functools import lru_cache
from collections import namedtuple

import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas

ValueTuple = namedtuple("ValueTuple", "value value_shifted")

pandas.set_option('display.max_rows', None)

#Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomNumberGenerator(object):
    def __init__(self, seed=0, a=5, c=1, m=8, steps_to_shift=0):
        logger.info(
            "Instantiating RandomNumberGenerator with seed = {}, p = {}, q = {}, m = {}, steps_to_shift = {}.".format(
                seed, a, c, m, steps_to_shift))
        #Setting parameters
        self.seed = seed
        self.a = a
        self.c = c
        self.m = m
        self.steps_to_shift = steps_to_shift
        self.max_value = m - 1 >> steps_to_shift

        #
        self.period = self.__calculate_period__()
        self.series = self.__calculate_series__()
        self.counts = self.__calculate_counts__()
        logger.info("Finished instantiating RandomNumberGenerator")

    @lru_cache(maxsize=None)
    def x_n(self, n):
        if n == 0:
            return ValueTuple(self.seed, self.seed >> self.steps_to_shift)
        else:
            previous, previous_shifted = self.x_n(n=(n - 1))
            number = ((self.a * previous) + self.c) % self.m
            number_shifted = number >> self.steps_to_shift
            return ValueTuple(number, number_shifted)

    def graph_series(self):
        sns.set()
        sns.lmplot('n', 'xn', data=self.series, fit_reg=False, scatter_kws={"s": 5})
        plt.xlabel(r'Index $n$')
        plt.ylabel(r'Value $x_n$')
        plt.title(r'Number series $x_n = {} \cdot x_{{n-1}} + {}$ (mod {})'.format(self.a, self.c, self.m))
        plt.show()

    def graph_counts(self):
        sns.set()
        fig = sns.lmplot('xn', 'count', data=self.counts, fit_reg=False, scatter_kws={"s": 5})
        axes = fig.axes
        max_count = self.counts['count'].max()
        logger.debug("Max count is {}".format(max_count))
        axes[0, 0].set_ylim(0, max_count * 2)
        plt.show()

    def __calculate_counts__(self):
        value_counts = self.series['xn'].value_counts()
        dataframe = pandas.DataFrame(data=value_counts.reset_index(), index=np.arange(0, self.max_value + 1))
        dataframe.columns = ['xn', 'count']
        dataframe = dataframe.sort_values(by='xn').reset_index(drop=True)
        logger.info("\n{}".format(dataframe))
        return dataframe

    def __calculate_series__(self):
        series = pandas.DataFrame(index=np.arange(0, self.period), columns=('n', 'xn'))
        logger.info("Generating values for entire series")
        for i in range(0, self.period):
            logger.debug("Getting the {}:th value".format(i))
            value, value_shifted = self.x_n(i)
            series.at[i, 'n'] = i
            series.at[i, 'xn'] = value_shifted
        logger.info("Values ({}) generated for series".format(len(series)))
        logger.debug("Series: {}".format(series))
        return series

    def __calculate_period__(self):
        logger.info("Calculating period of {}".format(self))
        series = []
        i = 0
        while (True):
            value, value_shifted = self.x_n(n=i)
            if (value) not in series:
                series.append(value)
                i += 1
                logger.debug("{} not found, period at least {}".format(value, len(series)))
                if (i % 1000) == 0:
                    logger.info("Period at least {}".format(len(series)))
            else:
                logger.debug(
                    "{} already in series at index {}, period is {}".format(value, series.index(value), len(series)))
                break
        period = len(series)
        return period


def main():
    #rng = RandomNumberGenerator(seed=0, a=5, c=1, m=8, steps_to_shift=0)
    rng = RandomNumberGenerator(seed=0, a=25173, c=13849, m=2 ** 16, steps_to_shift=7)
    for i in range(1, 20):
        value, value_shifted = rng.x_n(i)
        print("{}; ".format(value_shifted), end="")
    print("")
    print("Period: {}".format(rng.period))
    rng.graph_counts()
    rng.graph_series()


if __name__ == '__main__':
    main()
