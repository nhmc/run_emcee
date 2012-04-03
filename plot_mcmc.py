from __future__ import division
import sys, os
import pylab as pl
import numpy as np
from astro.io import loadobj, parse_config
from astro.utilities import autocorr, scoreatpercentile
from astro.plot import dhist, distplot, puttext
import numpy as np
import run_emcee

if not os.path.lexists('model.py'):
    print "The file 'model.py' must be in the current directory"
    sys.exit()

if '.' not in sys.path:
    sys.path.insert(0, '.')

from model import ymodel, P, x, ydata, ysigma

defaultpath = run_emcee.__path__[0] + '/'

pl.rc('font', size=9)

def get_fig_axes(nrows, ncols, npar):
    fig = pl.figure(figsize=(12., 12.*nrows/ncols))    
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.07, top=0.95)
    axes = [fig.add_subplot(nrows, ncols, i+1) for i in range(npar)]
    return fig, axes

def get_nrows_ncols(npar):
    nrows = max(int(np.sqrt(npar)), 1)
    ncols = nrows
    while npar > (nrows * ncols):
        ncols += 1

    return nrows, ncols

def find_min_interval(x, alpha):
    """ Determine the minimum interval containing a given probability.

    x is an array of parameter values (such as from an MCMC trace).

    alpha (0 -> 1) is the desired probability encompassed by the
    interval.

    Inspired by the pymc function of the same name.
    """

    x = np.sort(x)
    assert len(x) > 1

    # Initialize interval
    min_int = None, None

    # Number of elements in trace
    n = len(x)

    # Start at far left
    end0 = int(n*(1 - alpha))
    start, end = 0, end0

    # Initialize minimum width to large value
    min_width = np.inf

    for i in xrange(n - end0):
        hi, lo = x[end+i], x[start+i]
        width = hi - lo
        if width < min_width:
            min_width = width
            min_int = lo, hi

    return min_int


def plot_autocorr(chain, mean_accept):
    """ Plot the autocorrelation of all the parameters in the given
    chain.
    """
    nwalkers, nsamples, npar = chain.shape
    nrows, ncols = get_nrows_ncols(npar)
    fig,axes = get_fig_axes(nrows, ncols, npar)
    for i,ax in enumerate(axes):
        acor = [autocorr(chain[j,:,i], maxlag=200) for j in xrange(nwalkers)]
        distplot(np.transpose(acor), ax=ax)
        ax.axhline(0, color='r', lw=0.5)
        puttext(0.1, 0.1, P.names[i], ax, fontsize=16)

    fig.suptitle(
        'Autocorrelation for %i walkers with %i samples. (Mean acceptance fraction %.2f)' %
        (nwalkers, nsamples, mean_accept), fontsize=14)

    fig.savefig('fig/autocorr.jpg')


def plot_posteriors(chain, pbest):
    """ Plot the posterior distributions using the walker
    positions from the final sample.
    """
    c = chain[:,-1,:].squeeze()
    npar = c.shape[-1]
    nrows, ncols = get_nrows_ncols(npar)
    fig,axes = get_fig_axes(nrows, ncols, npar)

    for i,ax in enumerate(axes):
        j = i+1
        if j == npar:
            j = 0

        dhist(c[:, i], c[:, j], xbins=P.bins[i], ybins=P.bins[j],
              #contourbins=50,
              fmt='.', ms=1.5, c='0.5', chist='b', ax=ax, loc='left, bottom')

        #ax.plot(pbest[:,i], pbest[:,j], 'o', ms=10, mfc='none', mec='r')
        ax.plot(P.guess[i], P.guess[j], 'xr', ms=10, mew=2)

        puttext(0.95, 0.05, P.names[i], ax, fontsize=16, ha='right')
        puttext(0.05, 0.95, P.names[j], ax, fontsize=16, va='top')
        x0, x1 = scoreatpercentile(c[:,i], [5, 95])
        dx = x1 - x0
        ax.set_xlim(x0 - dx, x1 + dx)
        y0, y1 = scoreatpercentile(c[:,j], [5, 95])
        dy = y1 - y0
        ax.set_ylim(y0 - dy, y1 + dy)


    fig.savefig('fig/posterior_mcmc.jpg')

def plot_posteriors_burn(chain):
    """ Plot the posteriors of a burn-in sample
    """
    npar = chain.shape[-1]
    c = chain.reshape(-1, npar)
    
    nrows, ncols = get_nrows_ncols(chain.shape[-1])
    fig, axes = get_fig_axes(nrows, ncols, npar)

    for i,ax in enumerate(axes):
        j = i+1
        if j == npar:
            j = 0

        dhist(c[:, i], c[:, j], xbins=P.bins[i], ybins=P.bins[j],
              fmt='.', ms=1, c='0.5', chist='b', ax=ax, loc='left, bottom')

        # plot initial walker positions
        ax.plot(chain[:,0,i], chain[:,0,j], '.r', ms=4, label='p$_{initial}$')

        # and final positions
        ax.plot(chain[:,-1,i], chain[:,-1,j], '.y', ms=4, label='p$_{final}$')

        ax.plot(P.guess[i], P.guess[j], 'xr', ms=10, mew=2, label='guess')

        puttext(0.95, 0.05, P.names[i], ax, fontsize=16, ha='right')
        puttext(0.05, 0.95, P.names[j], ax, fontsize=16, va='top')
        x0, x1 = chain[:, 0, i].min(), chain[:, 0, i].max()
        dx = x1 - x0
        ax.set_xlim(x0 - 0.1*dx, x1 + 0.1*dx)
        y0, y1 = chain[:, 0, j].min(), chain[:, 0, j].max()
        dy = y1 - y0
        ax.set_ylim(y0 - 0.1*dy, y1 + 0.1*dy)

    fig.savefig('fig/posterior_burnin.jpg')


def main(args):

    try:
        print '### Reading parameters from emcee.cfg ###'
        opt = parse_config('emcee.cfg')
    except IOError:
        print '### Reading parameters from %sdefault.cfg ###' % defaultpath
        opt = parse_config(defaultpath + 'default.cfg')

    # bins for plotting posterior histograms
    P.bins = [np.linspace(lo, hi, opt.Nhistbins) for lo,hi in
              zip(P.min, P.max)]

    filename, = args
    samples = loadobj(filename)

    mean_accept =  samples['accept'].mean()
    print 'Mean acceptance fraction', mean_accept
    nwalkers, nsamples, npar = samples['chain'].shape

    if not os.path.lexists('fig/'):
        os.mkdir('fig')

    fig = pl.figure(figsize=(10, 5))
    pl.plot(x, ydata)
    pl.plot(x, ymodel(x, *P.guess), label='initial guess')
    pl.plot(x, ysigma)

    if filename != 'samples_burn.sav':
        # find the set of parameters that have the highest likelihood
        # in the chain.
        
        # the first 10 highest probability parameter sets from the
        # last sample

        pos = samples['final_pos']
        ibest = np.argsort(samples['lnprob'][:,-1], axis=None)
        pbest = pos[ibest[-10:]]
        P.best = pos[ibest[-1]]
        print 'pbest', P.best

        pl.plot(x, ymodel(x, *P.best), 'c', lw=2, label='maximum likelihood')

    pl.legend(frameon=0)
    fig.savefig('fig/model.jpg')

    print 'plotting posteriors'
    if filename == 'samples_burn.sav':
        plot_posteriors_burn(samples['chain'])
    else:
        plot_posteriors(samples['chain'], pbest)
        
    print 'plotting autocorrelation'
    plot_autocorr(samples['chain'], mean_accept)

    pl.show()

if __name__ == '__main__':
    main(sys.argv[1:])
