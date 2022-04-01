import jax
import jax.numpy as jnp

"""
Hessian eigenvalue metrics based on descriptions from Ghorbani et al 2019 https://arxiv.org/pdf/1901.10159.pdf
"""

def l1_energy(density,grids):
    """
    Integrate the positive and negative portions of the eigenspectrum
    Applies L1 norm to eigenspectrum density
    (since densities are positive, L1 norm of density is just the original density)
    Described in section 3 of the paper
    """
    bin_size = grids[1]-grids[0]
    pos_mask = grids>=0.
    neg_mask = grids<0.
    pos_energy = jnp.sum(bin_size*density[pos_mask])
    neg_energy = jnp.sum(bin_size*density[neg_mask])
    return pos_energy, neg_energy

def l2_energy(density,grids):
    """
    Integrate the positive and negative portions of the eigenspectrum
    Applies L2 norm to eigenspectrum density
    Described in section 3 of the paper
    """
    bin_size = grids[1]-grids[0]
    pos_mask = grids>=0.
    neg_mask = grids<0.
    pos_energy = jnp.sum(bin_size*density[pos_mask]**2)
    neg_energy = jnp.sum(bin_size*density[neg_mask]**2)
    return pos_energy, neg_energy

def eigenvalue_ratio(eig_vals,k):
    """
    Compute the ratio of the largest eigenvalue to the kth largest eigenvalue
    i.e. \frac{\lambda_1}{\lambda_k}
    See Figure 6 of the paper for an example
    """
    eig_vals_s = eig_vals.reshape(-1)[::-1] # largest to smallest
    return eig_vals_s[0]/eig_vals_s[k-1]

def topk_subspace_gradient_energy_ratio(jjvp,jac,eig_vals,k):
    """
    Compute the subspace spanned by the top k Jacobian covariance eigenvectors, project the gradient onto that subspace,
    then computes the ratio of the L2 norms of the projected gradient vs the unprojected gradient
    i.e. \frac{\| P \nabla_\theta L(\theta) \|^2_2} {\| \nabla_\theta L(\theta) \|^2_2}
    See Figure 11 of the paper for an example
    TODO: Not sure how to implement most efficiently
    """
    pass
    # let C be the covariance of the Jacobian
    # problem: can't solve C x = \lambda_k x exactly, requires inverting C
    # instead, use scalable root solver (like Broyden, from DEQ paper) to find roots of C x - \lambda_k x
    # citation: https://arxiv.org/pdf/1909.01377.pdf Equation 10 in Section 3.1.3
    # for i = 1 to k    
    #   solve C x = \lambda_k x approximately
    #   (C x can be computed with hvp)
    #   set eigenvector v_k = x^{\star}
    # assemble projection matrix P from eigenvectors { v_k }_k
    # compute P_jac =  P @ jac
    # return norm(P_jac) / norm(jac) 
