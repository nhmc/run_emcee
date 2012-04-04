"""
This model must define the following objects:

- a dictionary P with keys. The value of every key is a tuple with the
same length (the number of model parameters)

    name  : parameter names
    min   : minimum allowed parameter values
    max   : maximum allowed parameter values
    guess : parameter values to use to generate initial walker positions

- array of values x

- array of data values ydata

- array of data one sigma errors ysigma

- a ymodel(x, par) function that generates the model of the data given
  an array of parameter values

- a ln_likelihood(x, ydata, ysigma) function

"""
from __future__ import division
from astro.absorb import calc_iontau
from astro.pyvpfit import readatom
from astro.utilities import adict
import numpy as np

P = adict()

# these are what we'll find the posterior for
P.names = 'z1', 'logN1', 'b1', 'z2', 'logN2', 'b2'
P.guess = 2.5, 14, 20, 2.5005, 13.5, 25

# these are the upper and lower limits on the flat prior ranges. these
# will be used to generate guess positions for the sampler. If you get
# these wrong, the sampler can get confused (poor acceptance fractions
# or walkers getting lost in very low likelihood regions of parameter
# space). It will take some trial and error and inspection of the
# burn-in parameter distribution to find input ranges that give
# plausible results.

P.min = 2.4997, 12.5, 5, 2.500, 12.5, 5
P.max = 2.5003, 16.5, 70, 2.501, 15.5, 70

# function to generate the model at the x values from parameters
atom = readatom()
trans = atom['HI']

def ymodel(wa, *par):
    tau = np.zeros_like(wa)
    for i in xrange(len(par)//3):
        z,logN,b = par[3*i:3*i+3]
        tau += calc_iontau(wa, trans, z+1, logN, b)
    return np.exp(-tau)

############################################################
# Generate the data x, ydata, ysigma (in a real
# problem these would usually all be given)
############################################################

def make_data():
    # for generating the wavelength scale
    vrange = 500.
    dv = 3.

    # S/N per pixel for fake data
    snr = 15.

    # wavelength array (all velocities in km/s)
    wa0 = trans[0].wa
    c = 3e5
    zmean = np.mean(P.guess[::3])
    x = wa0 * (1 + zmean) * np.arange(1. - vrange/c, 1. + vrange/c, dv/c)
    ysigma = 1. / snr * np.ones(len(x))
    np.random.seed(99)
    noise = np.random.randn(len(x)) / snr
    ydata = ymodel(x, *P.guess) + noise
    return x, ydata, ysigma


x, ydata, ysigma = make_data()

# how do we generate the likelihood?

# for n data points, with each point having a probability p_i of taking
# the observed value (given some model) then
#
# likelihood = p0 * p1 * p2 * ... pn
#
# We want to maximise the likelihood.
#
# Assuming data values y_i and model values f_i at each x_i, and
# gaussian 1 sigma errors s_i on the y_i (and negligible error in the
# x_i values):
#
# = exp(-0.5*((y0-f0)/s0)**2) * exp(-0.5*((y1-f1)/s1)**2) * ...
#
# take natural logarithm so we can add rather than multiply,
# andhandily remove the exponents.
#
# ln(likelihood) = -0.5 * ( [(y0-f0)/s0]**2 + [(y1-f1)/s1]**2 + ... +
# [(yn-fn)/sn]**2 )
#
# simplify the notation by assuming Y, F and S are vectors
#
#  = -0.5 * np.sum( ((Y-F)/S**2 )
#
#  = -0.5 * np.dot(resid, resid)
#
#  where resid = (Y-F)/S, introducing vectors to represent each set of
#  points.

def ln_likelihood(pars, x, y, ysigma):
    # if we are outside the allowable parameter ranges, return 0
    # likelihood.
    for i,p in enumerate(pars):
        if not (P.min[i] < p < P.max[i]):
            return -np.inf

    resid = (y - ymodel(x, *pars)) / ysigma
    return -0.5 * np.dot(resid, resid)
