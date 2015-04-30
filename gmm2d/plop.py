import itertools
from docutils.nodes import thead

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture


# Number of samples per component
n_samples = 600
seed = 424246

def genIt(n_iter, fname=None):

    # Generate random sample, two components
    np.random.seed(seed)

    W = np.random.random((3))
    W /= W.sum()
    MU = (np.random.random((3, 2))-0.5)*10
    three2by2Identity = np.repeat(np.array([[[1,0],[0,1]]]), 3, 0)
    C = three2by2Identity + 0.25 * (np.random.random((3, 2, 2)) - 0.5)
    C = C/1.5

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
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.25)
            splot.add_artist(ell)


    splot = plt.subplot(2, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], 3, edgecolor='none', facecolors=Y)
    splot = plt.subplot(2, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], 10, edgecolor='none', facecolors=Y)

    gmm, pYX, pYXWinnerTakesAll = fit(n_iter)
    splot = plt.subplot(2, 2, 3)
    drawEllipses()
    plt.scatter(X[:, 0], X[:, 1], 10, edgecolor='none', facecolors=pYXWinnerTakesAll)
    splot = plt.subplot(2, 2, 4)
    drawEllipses()
    plt.scatter(X[:, 0], X[:, 1], 10, edgecolor='none', facecolors=pYX)

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)
        print("Saved as "+fname)
    plt.clf()

for i in range(0, 50):
    genIt(i, 'out/,,{:03}.png'.format(i))
