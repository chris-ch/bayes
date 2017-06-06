import numpy
from matplotlib import pyplot
import pymc3
from statsmodels.api import OLS, add_constant


def generate_data_multi():
    size = 50
    sigma = 0.5
    t = numpy.linspace(0, 20, size)
    x = numpy.column_stack((t, numpy.sin(t), (t - 5) ** 2, numpy.ones(size)))
    beta = [0.5, 0.5, -0.02, 5.]
    y_true = numpy.dot(x, beta)
    y = y_true + sigma * numpy.random.normal(size=size)
    return x, y, y_true, 3


def generate_data():
    size = 200
    sigma = 0.5
    true_intercept = 1
    beta = 2
    t = numpy.linspace(0, 1, size)
    y_true = true_intercept + beta * t
    y = y_true + numpy.random.normal(scale=sigma, size=size)
    return t, y, y_true, 1


def main():
    x, y, true_regression_line, dimension = generate_data_multi()

    fig = pyplot.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1, xlabel='x', ylabel='y', title='Generated data and underlying model')
    ax.plot(x, y, 'x', label='sampled data')
    ax.plot(x, true_regression_line, label='true regression line', lw=2.)
    pyplot.legend(loc=0)

    pyplot.show()

    #
    # Frequentist approach
    #
    model = OLS(y, add_constant(x))
    results = model.fit()
    print(results.summary())

    #
    # Bayesian approach
    #
    with pymc3.Model() as model: # model specifications in PyMC3 are wrapped in a with-statement
        # Define priors
        sigma = pymc3.HalfCauchy(beta=10, testval=1., name='sigma', shape=dimension)
        y_0 = pymc3.Normal(name='alpha', mu=0, sd=20, shape=dimension)
        betas = pymc3.Normal(name='beta', mu=0, sd=20, shape=dimension)

        # Define likelihood
        likelihood = pymc3.Normal(name='y', mu=y_0 + betas * x, sd=sigma, observed=y)

        # Inference!
        start = pymc3.find_MAP() # Find starting value by optimization
        step = pymc3.NUTS(scaling=start) # Instantiate MCMC sampling algorithm
        # draw 2000 posterior samples using NUTS sampling
        trace = pymc3.sample(2000, step, start=start, progressbar=False)
        pymc3.traceplot(trace)

if __name__ == '__main__':
    main()
    pyplot.show()