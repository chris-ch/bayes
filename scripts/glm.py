import numpy
from matplotlib import pyplot
import pymc3


def main():
    size = 200
    true_intercept = 1
    true_slope = 2

    x = numpy.linspace(0, 1, size)
    # y = a + b*x
    true_regression_line = true_intercept + true_slope * x
    # add noise
    y = true_regression_line + numpy.random.normal(scale=.5, size=size)

    fig = pyplot.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1, xlabel='x', ylabel='y', title='Generated data and underlying model')
    ax.plot(x, y, 'x', label='sampled data')
    ax.plot(x, true_regression_line, label='true regression line', lw=2.)
    pyplot.legend(loc=0)

    data = {'x': x, 'y': y}
    with pymc3.Model() as model: # model specifications in PyMC3 are wrapped in a with-statement
        # Define priors
        sigma = pymc3.HalfCauchy(beta=10, testval=1., name='sigma')
        intercept = pymc3.Normal(name='Intercept', mu=0, sd=20)
        x_coeff = pymc3.Normal(name='x', mu=0, sd=20)

        # Define likelihood
        likelihood = pymc3.Normal(name='y', mu=intercept + x_coeff * x, sd=sigma, observed=y)

        # Inference!
        start = pymc3.find_MAP() # Find starting value by optimization
        step = pymc3.NUTS(scaling=start) # Instantiate MCMC sampling algorithm
        # draw 2000 posterior samples using NUTS sampling
        trace = pymc3.sample(2000, step, start=start, progressbar=False)
        pymc3.traceplot(trace)

if __name__ == '__main__':
    main()
    pyplot.show()