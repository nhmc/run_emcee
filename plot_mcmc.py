from __future__ import division
import sys, os, pdb
import pylab as pl
import numpy as np

scipy = True
try:
    from scipy.stats import gaussian_kde
    from scipy.spatial import Delaunay
    from scipy.optimize import minimize
except ImportError:
    scipy = False

from astro.io import loadobj, parse_config, writetxt
from astro.utilities import autocorr
from astro.plot import dhist, distplot, puttext


if not os.path.lexists('model.py'):
    print "The file 'model.py' must be in the current directory"
    sys.exit()
if '.' not in sys.path:
    sys.path.insert(0, '.')

from model import ymodel, P, x, ydata, ysigma, ln_likelihood

pl.rc('font', size=9)
pl.rc('legend', fontsize='large')

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

    assert len(x) > 1
    x = np.sort(x)

    # Initialize interval
    min_int = None, None

    # Number of elements in trace
    n = len(x)

    # Start at far left
    end0 = int(n*alpha)
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


def get_levels(pts, frac):
    """ Find the a fraction of highest-likelihood pts for an n-D
    distribution MCMC.

    pts is an array with shape (npts, npar)

    The likelihood function is estimated from the parameter samples
    using a kernel density estimation.

    return the indices of the points.
    """
    # Note 0.9973 is three sigma.
    
    # generate the kde estimation given the points
    kde = gaussian_kde(pts.T)
    vals_kde = kde(pts.T)
    isort = vals_kde.argsort()
    ind = isort[int((1-frac)*len(pts)):]

    return ind


def plot_autocorr(chain):
    """ Plot the autocorrelation of parameters in the chain.
    """
    nwalkers, nsamples, npar = chain.shape
    nrows, ncols = get_nrows_ncols(npar)
    fig,axes = get_fig_axes(nrows, ncols, npar)
    for i,ax in enumerate(axes):
        acor = [autocorr(chain[j,:,i], maxlag=150) for j in xrange(nwalkers)]
        distplot(np.transpose(acor), ax=ax)
        ax.axhline(0, color='r', lw=0.5)
        puttext(0.1, 0.1, P.names[i], ax, fontsize=16)

    return fig, axes


def plot_posteriors(chain, P):
    """ Plot the posterior distributions for a series of parameter
    samples. chain has shape (nsample, nparameters).
    """
    npar = chain.shape[-1]
    nrows, ncols = get_nrows_ncols(npar)
    fig,axes = get_fig_axes(nrows, ncols, npar)

    for i,ax in enumerate(axes):
        j = i+1
        if j == npar:
            j = 0

        #ax.plot(chain[:,0,i], chain[:,0,j], '.r', ms=4, label='p$_{initial}$')
        dhist(chain[:, i], chain[:, j], xbins=P.bins[i], ybins=P.bins[j],
              fmt='.', ms=1.5, c='0.5', chist='b', ax=ax, loc='left, bottom')

        #if contours:
        #    i1sig = get_levels(np.array([chain[:,i], chain[:,j]]).T,)
        #    #cont = ax.contour(*par, colors='k',linewidths=0.5)
        # for ind in P.ijoint_sig:
        #     x,y = chain[:,i][ind], chain[:,j][ind]
        #     delaunay = Delaunay(np.array([x, y]).T)
        #     for i0,i1 in delaunay.convex_hull:
        #         ax.plot([x[i0], x[i1]], [y[i0], y[i1]], 'k', lw=0.5)
        x,y = chain[:,i][P.ijoint_sig[1]], chain[:,j][P.ijoint_sig[1]]
        ax.plot(x,y,'g.', ms=3)
        x,y = chain[:,i][P.ijoint_sig[0]], chain[:,j][P.ijoint_sig[0]]
        ax.plot(x,y,'r.', ms=3)

        if hasattr(P, 'guess'):
            ax.plot(P.guess[i], P.guess[j], 'o', mfc='none', ms=10, mew=4,mec='k')
            ax.plot(P.guess[i], P.guess[j], 'o', mfc='none', ms=10, mew=2, mec='r')
        ax.plot(P.best[i], P.best[j], 'xk', ms=12, mew=4)
        ax.plot(P.best[i], P.best[j], 'xr', ms=10, mew=2)

        c = 'crimson'
        ax.axvline(P.p1sig[i][0], ymax=0.2, color=c, lw=0.5)
        ax.axvline(P.p1sig[i][1], ymax=0.2, color=c, lw=0.5)
        ax.axhline(P.p1sig[j][0], xmax=0.2, color=c, lw=0.5)
        ax.axhline(P.p1sig[j][1], xmax=0.2, color=c, lw=0.5)
        ax.axvline(P.median[i], ymax=0.2, color=c, lw=1.5)
        ax.axhline(P.median[j], xmax=0.2, color=c, lw=1.5)

        puttext(0.95, 0.05, P.names[i], ax, fontsize=16, ha='right')
        puttext(0.05, 0.95, P.names[j], ax, fontsize=16, va='top')
        x0, x1 = np.percentile(chain[:,i], [5, 95])
        dx = x1 - x0
        ax.set_xlim(x0 - dx, x1 + dx)
        y0, y1 = np.percentile(chain[:,j], [5, 95])
        dy = y1 - y0
        ax.set_ylim(y0 - dy, y1 + dy)

    return fig, axes


def plot_posteriors_burn(chain):
    """ Plot the posteriors of a burn-in sample
    """
    nwalkers, nsamples, npar = chain.shape
    c = chain.reshape(-1, npar)
    
    nrows, ncols = get_nrows_ncols(npar)
    fig, axes = get_fig_axes(nrows, ncols, npar)

    for i,ax in enumerate(axes):
        j = i+1
        if j == npar:
            j = 0

        ax.plot(c[:, i], c[:, j], '.', ms=1, color='0.5')

        # plot initial walker positions
        ax.plot(chain[:,0,i], chain[:,0,j], '.r', ms=4, label='p$_{initial}$')

        # and final positions
        ax.plot(chain[:,-1,i], chain[:,-1,j], '.y', ms=4, label='p$_{final}$')

        if hasattr(P, 'guess'):
            ax.plot(P.guess[i], P.guess[j], 'o', mfc='none', mec='k',ms=10, mew=4)
            ax.plot(P.guess[i], P.guess[j], 'o', mfc='none', mec='r',ms=10, mew=2, label='guess')

        puttext(0.95, 0.05, P.names[i], ax, fontsize=16, ha='right')
        puttext(0.05, 0.95, P.names[j], ax, fontsize=16, va='top')
        x0, x1 = chain[:, 0, i].min(), chain[:, 0, i].max()
        dx = x1 - x0
        ax.set_xlim(x0 - 0.1*dx, x1 + 0.1*dx)
        y0, y1 = chain[:, 0, j].min(), chain[:, 0, j].max()
        dy = y1 - y0
        ax.set_ylim(y0 - 0.1*dy, y1 + 0.1*dy)

    axes[0].legend()
    return fig, axes

def print_par(filename, P):
    """ Print the maximum likelihood parameters and their
    uncertainties.
    """
    rec = []
    for i in range(len(P.names)):
        p = P.best[i]
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
    writetxt(filename, rec, header=hd, fmt_float='.4g', overwrite=1)
    
def main(args):

    opt = parse_config('model.cfg')
    print '### Read parameters from model.cfg ###'

    # bins for plotting posterior histograms
    P.bins = [np.linspace(lo, hi, opt.Nhistbins) for lo,hi in zip(P.min, P.max)]

    filename, = args
    samples = loadobj(filename)

    mean_accept =  samples['accept'].mean()
    print 'Mean acceptance fraction', mean_accept
    nwalkers, nsamples, npar = samples['chain'].shape

    if not os.path.lexists('fig/'):
        os.mkdir('fig')

    if filename == 'samples_burn.sav':
        print 'Plotting burn-in sample posteriors'
        fig,axes = plot_posteriors_burn(samples['chain'])
        fig.suptitle('%i samples of %i walkers' % (
            nsamples, nwalkers), fontsize=14)
        fig.savefig('fig/posterior_burnin.' + opt.plotformat)

        print 'Plotting autocorrelation'
        fig, axes = plot_autocorr(samples['chain'])
        fig.suptitle('Autocorrelation for %i walkers with %i samples. '
                     '(Mean acceptance fraction %.2f)' %
                     (nwalkers, nsamples, mean_accept), fontsize=14)
        fig.savefig('fig/autocorr.' + opt.plotformat)

    else:
        # make a chain of independent samples
        Ns, Nt = opt.Nsamp, opt.Nthin
        assert Ns * Nt <= nsamples 
        chain = samples['chain'][:,0:Ns*Nt:Nt,:].reshape(-1, npar)
        lnprob = samples['lnprob'][:,0:Ns*Nt:Nt].ravel()
        isort = lnprob.argsort()
        levels = 0.6827, 0.9545
        P.ijoint_sig = [isort[int((1-l)*len(lnprob)):] for l in levels]

        P.p1sig = [find_min_interval(chain[:, i], 0.6827) for i in range(npar)]
        P.p2sig = [find_min_interval(chain[:, i], 0.9545) for i in range(npar)]

        # the joint 1 and 2 sigma regions, simulatenously estimating
        # all parameters.
        P.p1sig_joint = []
        P.p2sig_joint = []
        for i in range(npar):
            lo = chain[P.ijoint_sig[0], i].min()
            hi = chain[P.ijoint_sig[0], i].max() 
            P.p1sig_joint.append((lo, hi))
            lo = chain[P.ijoint_sig[1], i].min()
            hi = chain[P.ijoint_sig[1], i].max()
            P.p2sig_joint.append((lo, hi))
            
        P.median = np.median(chain, axis=0)

        # estimate maximum likelihood as the point in the chain with
        # the highest likelihood.
        i = samples['lnprob'].ravel().argmax()
        P.best = samples['chain'].reshape(-1, npar)[i]

        if opt.find_maximum_likelihood:
            if not scipy:
                raise ImportError('Scipy minimize not available')
            print 'Finding maximum likelihood parameter values'
            P.best = minimize(lambda *x: -ln_likelihood(*x),
                              P.best, args=(x, ydata, ysigma))

        print 'Plotting sample posteriors'
        fig, axes = plot_posteriors(chain, P)
        fig.suptitle('%i of %i samples, %i walkers, thinning %i' % (
            Ns, nsamples, nwalkers, Nt), fontsize=14)
        fig.savefig('fig/posterior_mcmc.' + opt.plotformat)

        print_par('parameters.txt', P)

    if opt.plotdata:
        fig = pl.figure(figsize=(10, 5))
        pl.plot(x, ydata)
        if hasattr(P, 'guess'):
            pl.plot(x, ymodel(x, P.guess), label='Initial guess')
        pl.plot(x, ysigma)

        if hasattr(P, 'best'):
            pl.plot(x, ymodel(x, P.best),'c',lw=2,label='Maximum likelihood')

        pl.legend(frameon=0)
        fig.savefig('fig/model.' + opt.plotformat)


    pl.show()

if __name__ == '__main__':
    main(sys.argv[1:])
