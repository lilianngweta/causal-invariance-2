
#############################################################################################################################
#   Code for calculating the Hilbert Schmidt Information Criterion (HSIC) with a Gaussian kernel. This is a pytorch version 
#   adapted by converting the numpy code from https://github.com/strumke/hsic_python/blob/master/hsic.py
#############################################################################################################################


"""
Hilbert Schmidt Information Criterion with a Gaussian kernel, based on the
following references
[1]: https://link.springer.com/chapter/10.1007/11564089_7
[2]: https://www.researchgate.net/publication/301818817_Kernel-based_Tests_for_Joint_Independence
"""
import torch

def centering(M):
    """
    Calculate the centering matrix
    """
    n = M.shape[0]
    unit = torch.ones([n, n])
    identity = torch.eye(n)
    H = identity - unit/n

    return torch.matmul(M, H)

def gaussian_grammat(x, sigma=None):
    """
    Calculate the Gram matrix of x using a Gaussian kernel.
    If the bandwidth sigma is None, it is estimated using the median heuristic:
    ||x_i - x_j||**2 = 2 sigma**2
    """
    try:
        x.shape[1]
    except IndexError:
        x = x.reshape(x.shape[0], 1)

    xxT = torch.matmul(x, x.T)
    xnorm = torch.diag(xxT) - xxT + (torch.diag(xxT) - xxT).T
    if sigma is None:
        mdist = torch.median(xnorm[xnorm!= 0])
        sigma = torch.sqrt(mdist*0.5)


   # --- If bandwidth is 0, add machine epsilon to it
    if sigma==0:
        eps = 7./3 - 4./3 - 1
        sigma += eps

    KX = - 0.5 * xnorm / sigma / sigma
#     torch.exp(KX, KX)
    torch.exp(KX)
    return KX


def HSIC(x, y):
    """
    Calculate the HSIC estimator for d=2, as in [1] eq (9)
    """
    n = x.shape[0]
    return torch.trace(torch.matmul(centering(gaussian_grammat(x)),centering(gaussian_grammat(y))))/n/n
