from __future__ import division
import sys, os
import time
from math import sqrt
import numpy as np
import emcee
from barak.io import loadobj, saveobj, parse_config

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

    iprint = opt.iprint
    # note the results are saved in the sampler object.
    for i,(pos, lnprob, state) in enumerate(
        sampler.sample(p0, iterations=opt.Nburn)):
        i += 1
        if not i % iprint:
            print i
    
    print 'Saving results to samples_burn.sav'
    save_samples('samples_burn.sav.gz', sampler, pos, state)

def run_mcmc(sampler, opt):

    print 'Reading initial state from sample_burn.sav'
    burn_in = loadobj('samples_burn.sav.gz')
    sampler.reset()

    # Starting from the final position in the burn-in chain, sample for 1500
    # steps. (rstate0 is the state of the internal random number generator)

    # note the results are saved in the sampler object.
    iprint = opt.iprint
    print "Running MCMC with %i steps" % opt.Nmcmc
    for i,(pos, lnprob, state) in enumerate(sampler.sample(
        burn_in['final_pos'], iterations=opt.Nmcmc, rstate0=burn_in['state'])):
        i += 1
        if not i % iprint:
            print i

    print 'Saving results to samples_mcmc.sav'
    save_samples('samples_mcmc.sav.gz', sampler, pos, state)

def main(args=None):

    path = os.path.abspath(__file__).rsplit('/', 1)[0]
    defaults = parse_config(path + '/default.cfg')
    opt = parse_config('model.cfg', defaults)
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
