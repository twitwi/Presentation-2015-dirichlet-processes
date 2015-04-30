import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture


# Number of samples per component
n_samples = 600

mode = 'difficult'
seed = 1234

if mode == 'crazy': seed = 4242
if mode == 'difficult': seed = 424250
if mode == 'cool': seed = 11421142

def genIt(n_iter, fname=None, ext='.png', do_original=False, do_hard=False):

    # Generate random sample, two components
    np.random.seed(seed)

    W = np.random.random((3))
    W /= W.sum()
    MU = (np.random.random((3, 2))-0.5)*10
    three2by2Identity = np.repeat(np.array([[[1,0],[0,1]]]), 3, 0)
    C = three2by2Identity + 0.25 * (np.random.random((3, 2, 2)) - 0.5)
    C = C/1.25

    if mode == 'cool': C = C*1.5
    if mode == 'difficult': C[1, 1, 1] *= 2

    X = np.r_[
           np.dot(np.random.randn(n_samples/6, 2), C[0])+MU[0],
           np.dot(np.random.randn(n_samples/2, 2), C[1])+MU[1],
           np.dot(np.random.randn(n_samples/3, 2), C[2])+MU[2]]
    Ylabel = [0]*(n_samples/6) + [1]*(n_samples/2) + [2]*(n_samples/3)
    Y = np.zeros((X.shape[0], MU.shape[0]))
    Y[np.arange(Y.shape[0]), Ylabel] = 1


    colors = ['r', 'g', 'b']

    def fit(iters):
        # Fit a mixture of Gaussians with EM using five components
        gmm = mixture.GMM(n_components=3, covariance_type='full', n_iter=iters)
        gmm.fit(X)
        Y_ = gmm.predict(X)
        pYX = gmm.predict_proba(X)
        pYX = pYX / np.sum(pYX, axis=1, keepdims=True)
        pYXWinnerTakesAll = np.zeros(pYX.shape)
        pYXWinnerTakesAll[np.arange(pYX.shape[0]), pYX.argmax(axis=1)] = 1
        return gmm, pYX, pYXWinnerTakesAll

    def drawEllipses():
        for i, (mean, covar, color) in enumerate(zip(gmm.means_, gmm._get_covars(), colors)):
            v, w = linalg.eigh(covar)
            u = w[0] / linalg.norm(w[0])
            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color='k')
            #ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.25)
            splot.add_artist(ell)

    def saveim(suff):
        if fname is None:
            plt.show()
        else:
            plt.savefig(fname+suff+ext)
            print("Saved as "+fname+suff+ext)
        plt.clf()


    if do_original:
        splot = plt.subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], 3, edgecolor='none', facecolors=Y)
        saveim('complete')
        plt.scatter(X[:, 0], X[:, 1], 20, edgecolor='none', facecolors=Y)
        saveim('complete10')
        plt.scatter(X[:, 0], X[:, 1], 20, edgecolor='none', color='k')
        saveim('raw')

    gmm, pYX, pYXWinnerTakesAll = fit(n_iter)

    if do_hard:
        splot = plt.subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], 20, edgecolor='none', facecolors=pYXWinnerTakesAll)
        drawEllipses()
        saveim('hard')

    splot = plt.subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], 20, edgecolor='none', facecolors=pYX)
    drawEllipses()
    saveim('soft')


for i in range(0, 11) + range(12, 31, 2) + range(34, 111, 4):
    genIt(i, 'out/,,{:03}'.format(i), do_original=i==0)


# convert -delay 50 ,,*soft.png -loop -1 ../cool-soft.gif
