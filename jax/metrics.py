import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm

"""
Hessian eigenvalue metrics based on descriptions from Ghorbani et al 2019 https://arxiv.org/pdf/1901.10159.pdf
"""


def l1_energy(density,grids):
    """
    Integrate the positive and negative portions of the eigenspectrum
    Applies L1 norm to eigenspectrum density
    (since densities are positive, L1 norm of density is just the original density)
    Described in section 3 of the paper
    Note: if the matrix is PSD, binning might give lots of "negative" energy
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
    Note: if the matrix is PSD, binning might give lots of "negative" energy
    """
    bin_size = grids[1]-grids[0]
    pos_mask = grids>=0.
    neg_mask = grids<0.
    pos_energy = jnp.sum(bin_size*density[pos_mask]**2)
    neg_energy = jnp.sum(bin_size*density[neg_mask]**2)
    return pos_energy, neg_energy

def eig_val_ratio(eig_vals,k):
    """
    Compute the ratio of the largest eigenvalue to the kth largest eigenvalue
    i.e. \frac{\lambda_1}{\lambda_k}
    See Figure 6 of the paper for an example
    """
    # average over samples first
    eig_vals = jnp.mean(eig_vals,axis=0)
    # sort
    eig_vals = eig_vals[::-1] # largest to smallest
    return eig_vals[0]/eig_vals[k-1]

# def topk_subspace_gradient_energy_ratio(jjvp,jac,eig_vals,k):
#     """
#     Compute the subspace spanned by the top k Jacobian covariance eigenvectors, project the gradient onto that subspace,
#     then computes the ratio of the L2 norms of the projected gradient vs the unprojected gradient
#     i.e. \frac{\| P \nabla_\theta L(\theta) \|^2_2} {\| \nabla_\theta L(\theta) \|^2_2}
#     See Figure 11 of the paper for an example
#     TODO: Not sure how to implement most efficiently
#     """
#     pass
#     # let C be the covariance of the Jacobian
#     # problem: can't solve C x = \lambda_k x exactly, requires inverting C
#     # instead, use scalable root solver (like Broyden, from DEQ paper) to find roots of C x - \lambda_k x
#     # citation: https://arxiv.org/pdf/1909.01377.pdf Equation 10 in Section 3.1.3
#     # for i = 1 to k    
#     #   solve C x = \lambda_k x approximately
#     #   (C x can be computed with hvp)
#     #   set eigenvector v_k = x^{\star}
#     # assemble projection matrix P from eigenvectors { v_k }_k
#     # compute P_jac =  P @ jac
#     # return norm(P_jac) / norm(jac) 

def trace_eig_vals(eig_vals):

    # compute trace
    trace = jnp.sum(eig_vals,axis=1)
    # then average over samples
    trace = jnp.mean(trace,axis=0)
    return trace

def trace_density(eig_vals,density,grids):

    order = eig_vals.shape[1]
    bin_size = grids[1]-grids[0]
    # normalize based on order
    density = (order * density) / jnp.sum(density)
    trace = jnp.sum(bin_size*density*grids)
    return trace

def max_eig_val(eig_vals):

    # average over samples first
    eig_vals = jnp.mean(eig_vals,axis=0)
    # sort
    eig_vals = eig_vals[::-1] # largest to smallest
    return eig_vals[0]

def min_eig_val(eig_vals):

    # average over samples first
    eig_vals = jnp.mean(eig_vals,axis=0)
    return eig_vals[0]

# def proj(u,vs):
#     @jax.jit
#     def _proj(v):
#         num = jnp.dot(v,u)
#         denom = jnp.dot(v,v)
#         return v * num / denom
#     u_proj = jax.tree_util.tree_map(_proj,vs)
#     u_proj = jax.tree_util.tree_reduce(lambda x,y: x+y,u_proj)
#     # u_proj = u_proj / jnp.linalg.norm(u_proj)
#     return u_proj

def proj(u,V):
    # V is k x n, u is n
    num = V @ u.reshape(-1,1)
    denom = jnp.sum(V*V,axis=1).reshape(-1,1)
    U_proj = V * num / denom
    u_proj = jnp.sum(U_proj,axis=0)
    return u_proj

# def gradient_energy_ratio(jac,eig_vecs,lcz_vecs,k):

#     # compute full eigenvalue
#     num_draws = eig_vecs.shape[0]
#     ratios_1 = []
#     ratios_2 = []
#     ratios_3 = []
#     ratios_4 = []
#     ratios_5 = []
#     jac_energy = norm(jac)
#     # print(norm(jac))
#     # print(norm(lcz_vecs[0].T @ lcz_vecs[0] @ jac))
#     # print(norm((lcz_vecs[0].T @ lcz_vecs[0]) @ jac))
#     for i in range(num_draws):
#         jac_lcz = lcz_vecs[i] @ jac
#         eig_basis = eig_vecs[i].T
#         eig_basis_k = (eig_vecs[i,:,-k:]).T # k x n
#         jac_lcz_proj = proj(jac_lcz,eig_basis)
#         jac_lcz_proj_k = proj(jac_lcz,eig_basis_k)
#         jac_proj = lcz_vecs[i].T @ jac_lcz_proj
#         jac_proj_k = lcz_vecs[i].T @ jac_lcz_proj_k
#         jac_ = lcz_vecs[i].T @ jac_lcz
#         ratios_1.append(norm(jac_proj_k) / jac_energy)
#         ratios_2.append(norm(jac_lcz_proj_k) / norm(jac_lcz))
#         ratios_3.append(norm(jac_lcz_proj_k) / norm(jac_lcz_proj))
#         ratios_4.append(norm(jac_proj_k) / norm(jac_proj))
#         ratios_5.append(norm(jac_proj_k) / norm(jac_))
#     # average over random samples
#     mean_ratio_1 = jnp.mean(jnp.array(ratios_1))
#     mean_ratio_2 = jnp.mean(jnp.array(ratios_2))
#     mean_ratio_3 = jnp.mean(jnp.array(ratios_3))
#     mean_ratio_4 = jnp.mean(jnp.array(ratios_4))
#     mean_ratio_5 = jnp.mean(jnp.array(ratios_5))
#     print(mean_ratio_1,mean_ratio_2,mean_ratio_3,mean_ratio_4,mean_ratio_5)
#     return mean_ratio_2
    
def gradient_energy_ratio(jac,eig_vecs,lcz_vecs,k):

    # compute full eigenvalue
    num_draws = eig_vecs.shape[0]
    ratios = []
    for i in range(num_draws):
        jac_lcz = lcz_vecs[i] @ jac
        eig_basis_k = (eig_vecs[i,:,-k:]).T # k x n
        jac_lcz_proj_k = proj(jac_lcz,eig_basis_k)
        ratios.append(norm(jac_lcz_proj_k) / norm(jac_lcz))
    # average over random samples
    mean_ratio = jnp.mean(jnp.array(ratios))
    return mean_ratio

# def gradient_energy_ratio(jac,eig_vecs,lcz_vecs,k):

#     # compute full eigenvalue
#     num_draws = eig_vecs.shape[0]
#     ratios = []
#     jac_energy = norm(jac)
#     for i in range(num_draws):
#         jac_lcz = lcz_vecs[i] @ jac
#         eig_basis_k = (eig_vecs[i,:,-k:]).T # k x n
#         jac_lcz_proj_k = proj(jac_lcz,eig_basis_k)
#         jac_proj_k = lcz_vecs[i].T @ jac_lcz_proj_k
#         ratios.append(norm(jac_proj_k) / jac_energy)
#     # average over random samples
#     mean_ratio = jnp.mean(jnp.array(ratios))
#     return mean_ratio

def gradient_energy(jac):
    return norm(jac)

def trace_over_topk(eig_vals,k):
    assert k > 0, k
    # sort
    eig_vals = eig_vals[:,::-1] # largest to smallest
    # trace (sum over all)
    trace = jnp.sum(eig_vals,axis=1)
    # sum over k
    topk_eig_vals = jnp.sum(eig_vals[:,:k],axis=1)
    # average over samples
    return jnp.mean(trace / topk_eig_vals)