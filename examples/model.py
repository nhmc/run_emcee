"""
This module must define the following objects:

- a dictionary P with keys. The value of every key is a tuple with the
same length (the number of model parameters)

    name  : parameter names
    min   : minimum allowed parameter values
    max   : maximum allowed parameter values


- a model(*par) function that generates the model of the data given
  an array of parameter values

- a ln_likelihood(par) function

- a get_initial_positions(nwalkers) function that generates an array
  of shape (nwalkers, npar) with parameter values for the initial
  walker positions.

- a plot_model(par) function that plots a model fit to the data given a
  set of parameter values, in the same order as they are listed in P.

"""
from __future__ import division
from barak.absorb import calc_iontau, readatom
from barak.utilities import adict
from barak.io import writetxt
import pylab as pl
import numpy as np

P = adict()

# these are what we'll find the posterior for
P.names = 'z1', 'logN1', 'b1', 'z2', 'logN2', 'b2'

# these are the upper and lower limits on the flat prior ranges. these
# will be used to generate guess positions for the sampler. If you get
# these wrong, the sampler can get confused (poor acceptance fractions
# or walkers getting lost in very low likelihood regions of parameter
# space). It will take some trial and error and inspection of the
# burn-in parameter distribution to find input ranges that give
# plausible results.

P.min = 2.4997, 12.5, 5, 2.500, 12.5, 5
P.max = 2.5003, 16.5, 70, 2.501, 15.5, 70

Npar = len(P.names)

# function to generate the model at each data point from parameters

atom = readatom()
trans = atom['HI']
def model(*par):
    tau = np.zeros_like(X)
    for i in xrange(len(par)//3):
        z,logN,b = par[3*i:3*i+3]
        tau += calc_iontau(X, trans, z+1, logN, b)
    return np.exp(-tau)


def make_data(ptrue):
    """ Generate the data x, ydata, ysigma (in a real problem these
    would usually all be given)."""

    # for generating the wavelength scale
    vrange = 500.
    dv = 3.

    # S/N per pixel for fake data
    snr = 15.

    # wavelength array (all velocities in km/s)
    wa0 = trans[0].wa
    c = 3e5
    zmean = np.mean(ptrue[::3])
    x = wa0 * (1 + zmean) * np.arange(1. - vrange/c, 1. + vrange/c, dv/c)
    ysigma = 1. / snr * np.ones(len(x))
    np.random.seed(99)
    noise = np.random.randn(len(x)) / snr
    ydata = model(x, *ptrue) + noise
    return x, ydata, ysigma

P.true = 2.5, 14, 20, 2.5005, 13.5, 25
X, ydata, ysigma = make_data(P.true)

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
# and handily remove the exponents.
#
# ln(likelihood) = -0.5 * ( [(y0-f0)/s0]**2 + [(y1-f1)/s1]**2 + ... +
# [(yn-fn)/sn]**2 )
#
# (Note we've dropped the terms that don't depend on the parameters to be
# determined...)
#
# simplify the notation by assuming Y, F and S are vectors
#
#  = -0.5 * np.sum( ((Y-F)/S**2 )
#
#  = -0.5 * np.dot(resid, resid)
#
#  where resid = (Y-F)/S, introducing vectors to represent each set of
#  points.

def ln_likelihood(pars):
    # if we are outside the allowable parameter ranges, return 0
    # likelihood.
    for i,p in enumerate(pars):
        if not (P.min[i] < p < P.max[i]):
            return -np.inf

    resid = (ydata - model(*pars)) / ysigma
    return -0.5 * np.dot(resid, resid)


def get_initial_positions(nwalkers):
    # Get initial parameter positions (guesses!) for each walker

    # one possibility:
    
    # generate random values from a normal distribution with a 1
    # sigma width 5 times smaller than the limits for each parameter.
    #p0 = np.random.randn(nwalkers, Npar)
    #for i in range(Npar):
    #    p0[:, i] = P.true[i] + p0[:, i] * (P.max[i] - P.min[i]) / nsigma
    #    # clip so we are inside the parameter limits
    #    p0[:, i] = p0[:, i].clip(P.min[i], P.max[i])

    # another possibility:
    #
    # uniform distribution between parameter limits
    p0 = np.random.uniform(size=(nwalkers, Npar))
    for i in range(Npar):
        p0[:, i] = P.min[i] + p0[:, i] * (P.max[i] - P.min[i])

    return p0

def print_par(filename, par):
    """ Print the maximum likelihood parameters and their
    uncertainties.
    """
    rec = []
    for i in range(len(P.names)):
        p = P.ml[i]
        m1 = P.p1sig[i]
        m2 = P.p2sig[i]
        j1 = P.p1sig_joint[i]
        j2 = P.p2sig_joint[i]
        rec.append( (P.names[i], p,  p - j1[0], j1[1] - p,
                     p - j2[0], j2[1] - p, p - m1[0], m1[1] - p,
                     p - m2[0], m2[1] - p) )

    names = 'name,ml,j1l,j1u,j2l,j2u,m1l,m1u,m2l,m2u'
    rec = np.rec.fromrecords(rec, names=names)

    hd = """\
# name : parameter name
# ml   : maximum likelihood value
# j1l  : 1 sigma lower error (joint with all other parameters) 
# j1u  : 1 sigma upper error (joint)
# j2l  : 2 sigma lower error (joint) 
# j2u  : 2 sigma upper error (joint) 
# m1l  : 1 sigma lower error (marginalised over all other parameters)
# m1u  : 1 sigma upper error (marginalised)
# m2l  : 2 sigma lower error (marginalised) 
# m2u  : 2 sigma upper error (marginalised) 
"""
    #pdb.set_trace()
    writetxt('pars.txt', rec, header=hd, fmt_float='.4g', overwrite=1)

def plot_model(par):
    
    model = model(*par)
    pl.plot(X, ydata)
    pl.plot(X, model)
    pl.show()
