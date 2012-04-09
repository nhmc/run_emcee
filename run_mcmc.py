from __future__ import division
import sys, os
import time
from math import sqrt
import numpy as np
import emcee
from astro.io import loadobj, saveobj, parse_config

if not os.path.lexists('model.py'):
    print "The file 'model.py' must be in the current directory"
    sys.exit()
if '.' not in sys.path:
    sys.path.insert(0, '.')

from model import ln_likelihood, P, x, ydata, ysigma, get_initial_positions

# skip warnings when we add the -np.inf log likelihood value
np.seterr(invalid='ignore')

def save_samples(filename, sampler, pos, state):
    saveobj(filename, dict(
        chain=sampler.chain, accept=sampler.acceptance_fraction,
        lnprob=sampler.lnprobability, final_pos=pos, state=state), overwrite=1)

def run_burn_in(sampler, opt, p0):
    """ Run and save a set of burn-in iterations."""

    print 'Running burn-in with %i steps' % opt.Nburn
    pos, lnprob, state = sampler.run_mcmc(p0, opt.Nburn)

    print 'Saving results to samples_burn.sav'
    save_samples('samples_burn.sav', sampler, pos, state)

def run_mcmc(sampler, opt):

    print 'Reading initial state from sample_burn.sav'
    burn_in = loadobj('samples_burn.sav')
    sampler.reset()

    # Starting from the final position in the burn-in chain, sample for 1500
    # steps. (rstate0 is the state of the internal random number generator)
    print "Running MCMC with %i steps" % opt.Nmcmc
    pos, lnprob, state = sampler.run_mcmc(burn_in['final_pos'], opt.Nmcmc,
                                          rstate0=burn_in['state'])

    print 'Saving results to samples_mcmc.sav'
    save_samples('samples_mcmc.sav', sampler, pos, state)

def main(args=None):

    opt = parse_config('model.cfg')
    print '### Read parameters from model.cfg ###'

    print 'model parameters', P.names
    print 'minimum allowed values', P.min
    print 'maximum allowed values', P.max

    Npar = len(P.names)

    print opt.Nthreads, 'threads'
    print opt.Nwalkers, 'walkers'

    sampler = emcee.EnsembleSampler(
        opt.Nwalkers, Npar, ln_likelihood,
        args=[x, ydata, ysigma], threads=opt.Nthreads)

    if opt.Nburn > 0:
        t1 = time.time()
        p0 = get_initial_positions(opt.Nwalkers)
        run_burn_in(sampler, opt, p0)
        print '%.2g min elapsed' % ((time.time() - t1)/60.)

    if opt.Nmcmc > 0:
        t1 = time.time()
        run_mcmc(sampler, opt)
        print '%.2g min elapsed' % ((time.time() - t1)/60.)

    return sampler

if __name__ == 'main':
    main(sys.argv[1:])
