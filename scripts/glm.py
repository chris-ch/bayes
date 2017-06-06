import numpy
from matplotlib import pyplot
import pymc3
from statsmodels.api import OLS, add_constant
from statsmodels.sandbox.regression.predstd import wls_prediction_std


def generate_data_multi():
    size = 200
    sigma = 0.5
    t = numpy.linspace(0, 1, size)
    t_scale = 20.
    x = numpy.column_stack((t * t_scale, numpy.sin(t * t_scale), (t * t_scale - 5) ** 2, numpy.ones(size)))
    beta = [0.5, 0.5, -0.02, 5.]
    y_true = numpy.dot(x, beta)
    y = y_true + numpy.random.normal(scale=sigma, size=size)
    return t, x, y, y_true, 3


def generate_data():
    size = 200
    sigma = 0.5
    t = numpy.linspace(0, 1, size)
    t_scale = 1.
    x = numpy.column_stack((t * t_scale, numpy.ones(size)))
    beta = [2, 1.]
    y_true = numpy.dot(x, beta)
    y = y_true + numpy.random.normal(scale=sigma, size=size)
    return t, x, y, y_true, 1


def main():
    numpy.random.seed(1)
    x, X, y, y_true, dimension = generate_data()

    #fig = pyplot.figure(figsize=(7, 7))
    #ax = fig.add_subplot(1, 1, 1, xlabel='x', ylabel='y', title='Generated data and underlying model')
    #ax.plot(x, y, 'x', label='sampled data')
    #ax.plot(x, y_true, label='true regression line', lw=2.)
    #pyplot.legend(loc=0)


    #
    # Frequentist approach
    #
    model = OLS(y, X)
    results = model.fit()
    print(results.summary())

    prstd, iv_l, iv_u = wls_prediction_std(results)
    fig, ax = pyplot.subplots(figsize=(8, 6))
    ax.plot(x, y, 'o', label="data")
    ax.plot(x, y_true, 'b-', label="True")
    ax.plot(x, results.fittedvalues, 'r--.', label="OLS")
    ax.plot(x, iv_u, 'r--')
    ax.plot(x, iv_l, 'r--')
    ax.legend(loc='best')
    pyplot.show()

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