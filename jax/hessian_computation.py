# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code to perform Hessian vector products on neural networks.
"""

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as np
import jax.tree_util as tu


# TODO(gilmer): Should be possible to avoid backpropping through the
# ravel_pytree.
def full_hessian(loss, params):
  """Computes the full hessian matrix.

  Warning: The hessian is a [num_params, num_params] matrix, though this
  function is only intended for small use cases where this matrix can fit in
  memory. For large networks, one should work with the Hessian implicitely
  via hessian-vector products.

  Args:
    loss: function computing the loss with signature loss(params), and
      returns a scalar.
    params: pytree containing the parameters of the model, must be a valid input
      for loss.
  Returns:
    hessian_matrix: array of shape [num_params, num_params].
  """

  flat_params, unravel = ravel_pytree(params)

  def loss_flat(flat_params):
    params = unravel(flat_params)
    return loss(params)

  hessian_matrix = jax.hessian(loss_flat)(flat_params)
  return hessian_matrix


def full_jj(loss, params):

  flat_params, unravel = ravel_pytree(params)

  def loss_flat(flat_params):
    params = unravel(flat_params)
    return loss(params)

  j = jax.jacfwd(loss_flat)(flat_params).reshape(1,-1)
  jj_matrix = j.T @ j
  return jj_matrix


# TODO(gilmer, jamieas): consider other options for computing hvp (fwd_over_rev
# vs rev_over_fwd).
def hvp(loss, params, batch, v):
  """Computes the hessian vector product Hv.

  This implementation uses forward-over-reverse mode for computing the hvp.

  Args:
    loss: function computing the loss with signature
      loss(params, batch).
    params: pytree for the parameters of the model.
    batch:  A batch of data. Any format is fine as long as it is a valid input
      to loss(params, batch).
    v: pytree of the same structure as params.

  Returns:
    hvp: array of shape [num_params] equal to Hv where H is the hessian.
  """

  loss_fn = lambda params: loss(params, batch)
  return jax.jvp(jax.grad(loss_fn), [params], [v])[1]


def jjvp(loss, params, batch, v):
    # computes J^T @ J @ v, where J is 1xn
    loss_fn = lambda _params: loss(_params, batch)
    jvp = jax.jvp(loss_fn, [params], [v])[1]
    jjvp = jax.vjp(loss_fn, params)[1](jvp)[0]
    return jjvp

def jvp(loss, params, batch, v):
    # computes J @ v
    import pdb; pdb.set_trace()
    loss_fn = lambda _params: loss(_params, batch)
    jvp = jax.jvp(loss_fn, [params], [v])[1]
    return jvp

def ggnvp(wz_fn, zl_fn, params, batch, v):
    # compute GGN
    _wz_fn = lambda _params: wz_fn(_params, batch)
    wz, wz_jvp = jax.jvp(_wz_fn, [params], [v])
    zl_jvp, zl_hvp = jax.jvp(jax.grad(zl_fn), [wz], [wz_jvp])
    wz, zw_jvp_fn = jax.vjp(_wz_fn, params) 
    zw_zl_wz = zw_jvp_fn(zl_hvp)[0]
    return zw_zl_wz


def _tree_sum(tree_left, tree_right):
  """Computes tree_left + tree_right."""
  def f(x, y):
    return x + y
  return tu.tree_multimap(f, tree_left, tree_right)


def _tree_zeros_like(tree):
  def f(x):
    return np.zeros_like(x)
  return tu.tree_map(f, tree)


def get_jjvp_fn(loss, params, batches):

  flat_params, unravel = ravel_pytree(params)

  @jax.jit
  def jitted_jjvp(params, batch, v):
    return jjvp(loss, params, batch, v)

  def jjvp_fn(params, v):
    # The API of the function maps a 1d vector to a 1d vector. However for
    # efficiency we will perform all operations on the pytree representation
    # of params.
    v = unravel(v)  # convert v to the param tree structure
    jj_vp = _tree_zeros_like(params)
    # TODO(gilmer): Get rid of this for loop by using either vmap or lax.fori.
    count = 0
    for batch in batches():
      partial_vp = jitted_jjvp(params, batch, v)
      jj_vp = _tree_sum(jj_vp, partial_vp)
      count += 1
    if count == 0:
      raise ValueError("Provided generator did not yield any data.")
    jj_vp_flat, _ = ravel_pytree(jj_vp)
    jj_vp_flat /= count
    return jj_vp_flat

  return jjvp_fn, unravel, flat_params.shape[0]


def get_jvp_fn(loss, params, batches):

  # @jax.jit
  def jitted_jvp(params, batch, v):
    return jvp(loss, params, batch, v)

  def jvp_fn(params, v):
    # The API of the function maps a 1d vector to a 1d vector. However for
    # efficiency we will perform all operations on the pytree representation
    # of params.
    j_vp = _tree_zeros_like(v)
    # TODO(gilmer): Get rid of this for loop by using either vmap or lax.fori.
    count = 0
    for batch in batches():
      partial_vp = jitted_jvp(params, batch, v)
      j_vp = _tree_sum(j_vp, partial_vp)
      count += 1
    if count == 0:
      raise ValueError("Provided generator did not yield any data.")
    return j_vp

  return jvp_fn


def get_hvp_fn(loss, params, batches):
  """Generates a function mapping (params, v) -> Hv where H is the hessian.

  This function will batch the inputs and targets to be fed into loss. The
  hessian will be computed over all points xs, ys, potentially batching if
  needed. This function is intended to be used in cases where xs, ys are too
  large to run on a single pass. This function should not be jit compiled. If
  the computation is small enough to do all batches in memory then one can just
  do the following:

  @jit
  def jitted_hvp(params, v):
    return hvp(loss, params, all_data, v)

  Args:
    loss: scalar valued loss function with signature loss(params, batch).
      Assumes the loss computes a sum over all data points. If the loss computes
      the mean, results may be slightly off in cases where batch sizes are not
      uniform.
    params: params of the model, these will be flatten and concatentated into a
      single vector. Any pytree is valid
    batches: A generator yielding batches to be fed into loss. Must support the
      API "for b in batches(): ". batches() must yield a single epoch of data,
      it should also yield the same epoch of data everytime it is called.

  Returns:
    hvp: A function mapping (params, v) -> Hv. H is the Hessian of the loss
      with respect to the model parameters (it is a num_params by num_params
      matrix). v will be a flat vector of shape [num_params]. params will be
      the PyTree containing the model parameters (so calling ravel_pytree on
      parameters). The function signature is hvp(params, v).
    unravel: Maps v back to the form reprented as params.
    num_params: Total number of parameters in params (int).
  """

  flat_params, unravel = ravel_pytree(params)

  @jax.jit
  def jitted_hvp(params, batch, v):
    return hvp(loss, params, batch, v)

  def hvp_fn(params, v):
    # The API of the function maps a 1d vector to a 1d vector. However for
    # efficiency we will perform all operations on the pytree representation
    # of params.
    v = unravel(v)  # convert v to the param tree structure
    hessian_vp = _tree_zeros_like(params)
    # TODO(gilmer): Get rid of this for loop by using either vmap or lax.fori.
    count = 0
    for batch in batches():
      partial_vp = jitted_hvp(params, batch, v)
      hessian_vp = _tree_sum(hessian_vp, partial_vp)
      count += 1
    if count == 0:
      raise ValueError("Provided generator did not yield any data.")
    hessian_vp_flat, _ = ravel_pytree(hessian_vp)
    hessian_vp_flat /= count
    return hessian_vp_flat

  return hvp_fn, unravel, flat_params.shape[0]


def get_ggnvp_fn(wz_fn, zl_fn, params, batches):

  flat_params, unravel = ravel_pytree(params)

  @jax.jit
  def jitted_ggnvp(params, batch, v):
    return ggnvp(wz_fn, zl_fn, params, batch, v)

  def ggnvp_fn(params, v):
    # The API of the function maps a 1d vector to a 1d vector. However for
    # efficiency we will perform all operations on the pytree representation
    # of params.
    v = unravel(v)  # convert v to the param tree structure
    ggn_vp = _tree_zeros_like(params)
    # TODO(gilmer): Get rid of this for loop by using either vmap or lax.fori.
    count = 0
    for batch in batches():
      partial_vp = jitted_ggnvp(params, batch, v)
      ggn_vp = _tree_sum(ggn_vp, partial_vp)
      count += 1
    if count == 0:
      raise ValueError("Provided generator did not yield any data.")
    ggn_vp_flat, _ = ravel_pytree(ggn_vp)
    ggn_vp_flat /= count
    return ggn_vp_flat

  return ggnvp_fn, unravel, flat_params.shape[0]
