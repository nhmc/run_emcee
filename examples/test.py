# if you get an error when running emcee in parallel, you often get an
# unhelpful error messgae, making debugging very difficult. To check
# everything is working before you start running emcee, use thsi test
# file.

from model import \
     ln_likelihood, P, x, ydata, ysigma, get_initial_positions, plot_model

p0 = get_initial_positions(1)
plot_model(p0)
