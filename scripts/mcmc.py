import numpy
from matplotlib import pyplot
from scipy import optimize
from pymc3 import Model, Normal, HalfNormal
from pymc3 import find_MAP
from pymc3 import Slice
from pymc3 import sample
from pymc3 import traceplot


def init():
    # Initialize random number generator
    numpy.random.seed(123)

    # True parameter values
    alpha, sigma = 1, 1
    beta = [1, 2.5]

    # Size of dataset
    size = 100

    # Predictor variable
    x1 = numpy.random.randn(size)
    x2 = numpy.random.randn(size) * 0.2

    # Simulate outcome variable
    y = alpha + beta[0] * x1 + beta[1] * x2 + numpy.random.randn(size) * sigma
    return y, x1, x2


def show(y, x1, x2):
    fig, axes = pyplot.subplots(1, 2, sharex=True, figsize=(10, 4))
    axes[0].scatter(x1, y)
    axes[1].scatter(x2, y)
    axes[0].set_ylabel('Y')
    axes[0].set_xlabel('X1')
    axes[1].set_xlabel('X2')


def main():
    y, x1, x2 = init()
    show(y, x1, x2)
    basic_model = Model()

    with basic_model:

        # Priors for unknown model parameters
        alpha = Normal('alpha', mu=0, sd=10)
        beta = Normal('beta', mu=0, sd=10, shape=2)
        sigma = HalfNormal('sigma', sd=1)

        # Expected value of outcome
        mu = alpha + beta[0]*x1 + beta[1]*x2

        # Likelihood (sampling distribution) of observations
        y_obs = Normal('Y_obs', mu=mu, sd=sigma, observed=y)

        # draw 5000 posterior samples
        trace = sample(1000)
        traceplot(trace)

if __name__ == '__main__':
    main()
    pyplot.show()
