import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
import optax
import matplotlib.pyplot as plt
import copy

D = 10 #1000
N = 100
master_key = jax.random.PRNGKey(25412541)
init_keys = jax.random.split(master_key,num=10+1)
master_key = init_keys[0]
L = 100*jax.random.normal(init_keys[1],shape=(N,D)) # L is bigger
R = 0.01*jax.random.normal(init_keys[2],shape=(N,D))
M = 0.5*jax.random.normal(init_keys[3],shape=(N,N))
# make symmetric
M = M + M.T
params = {"L": L, "R": R}

def model_fn(L,R):
    return L @ R.T

def loss_fn(L,R):
    return jnp.linalg.norm(model_fn(L,R)-M)

def p_loss_fn(params):
    return loss_fn(params["L"],params["R"])

print(">>> initial")
print("Loss",p_loss_fn(params))
print("L_norm",jnp.linalg.norm(L))
print("R_norm",jnp.linalg.norm(R))
print("M_norm",jnp.linalg.norm(M))

T = 10000

alpha = 0.01
alpha_min = 0.00001
alpha_max = 1.0
# optimizer = optax.adam(learning_rate=alpha)
# optimizer = optax.sgd(learning_rate=alpha)
schedule_fn = optax.warmup_exponential_decay_schedule(
    init_value=alpha_min,
    peak_value=alpha_max,
    end_value=None,#10**((jnp.log10(alpha_min)+jnp.log10(alpha_max))/2),
    warmup_steps=T//3,
    transition_steps=1,
    decay_rate=0.99

)
optimizer = optax.chain(optax.scale(-alpha))
# optimizer = optax.chain(optax.scale_by_adam(eps=1e-4),optax.scale(-alpha))
# optimizer = optax.chain(optax.scale(-1.),optax.scale_by_schedule(schedule_fn))
opt_state = optimizer.init(params)

@jax.jit
def update_fn(t,params,opt_state):
    loss, grads = jax.value_and_grad(p_loss_fn)(params)
    updates, opt_state = optimizer.update(grads,opt_state,params)
    params = optax.apply_updates(params,updates)
    return loss, params, opt_state

ts, paramses, losses, L_norms, R_norms, LR_norms = [], [], [], [], [], []

for t in range(T):
    loss, params, opt_state = update_fn(t,params,opt_state)
    # print(">>>", t, loss, jnp.linalg.norm(params["L"]),jnp.linalg.norm(params["R"]))
    ts.append(t)
    paramses.append(copy.deepcopy(params))
    losses.append(loss)
    L_norms.append(jnp.linalg.norm(params["L"]))
    R_norms.append(jnp.linalg.norm(params["R"]))
    LR_norms.append(jnp.linalg.norm(params["L"]@params["R"].T))

print(">>> final")
print("Loss",losses[-1])
print("L_norm",L_norms[-1])
print("R_norm",R_norms[-1])
print("M_norm",jnp.linalg.norm(M))

# final_params = params
# final_loss = loss(params)
# print(">>> final", final_loss)
# print(jnp.linalg.norm(final_params["L"]),jnp.linalg.norm(final_params["R"]))

def plot(ts,**kwargs):
    num_plots = len(kwargs.keys())
    cm = plt.cm.get_cmap("viridis",num_plots)
    fig, axs = plt.subplots(nrows=1,ncols=num_plots)
    for ax_idx, (k,v) in enumerate(kwargs.items()):
        ax = axs[ax_idx]
        ax.plot(ts,v,label=k,color=cm(ax_idx))
    fig.legend()
    plt.show()
    plt.clf()

# plot(ts,Loss=losses,L_norm=L_norms,R_norm=R_norms,LR_norms=LR_norms)
# plot(ts,loss=losses)
# plot(ts,L_norm=L_norms)
# plot(ts,R_norm=R_norms)

# get the spectra

import lanczos
import density as density_lib
import hessian_computation
import time
from tqdm import tqdm

def compute_spectrum(params):


    def hvp_loss_fn(params,batch):
        return p_loss_fn(params)

    batches_list = [None]
    def batches_fn():
        for batch in batches_list:
            yield batch
    num_batches = len(batches_list)

    # Hessian-vector product function + Lanczos 
    order = 90
    num_samples = 10
    hvp, unravel, num_params = hessian_computation.get_hvp_fn(hvp_loss_fn, params, batches_fn)
    hvp_cl = lambda v: hvp(params, v) / num_batches # Match the API required by lanczos_alg

    print("num_params: {}".format(num_params))
    start = time.time()
    hvp_cl(jnp.ones(num_params)) # first call of a jitted function compiles it
    end = time.time()
    print("hvp compile time: {}".format(end-start))
    start = time.time()
    hvp_cl(2*jnp.ones(num_params)) # second+ call will be much faster
    end = time.time()
    print("hvp compute time: {}".format(end-start))

    # @jax.jit
    def get_tridiag_vecs(key):
      return lanczos.lanczos_alg(hvp_cl, num_params, order, rng_key=key)

    rng = jax.random.PRNGKey(420420)
    rngs = jax.random.split(rng,num=num_samples+1)
    rng = rngs[0]
    start = time.time()
    tridiags, vecses = [], []
    for i in tqdm(range(num_samples),desc="lanczos_samples"):
        tridiag, vecs = get_tridiag_vecs(rngs[i+1])
        tridiags.append(tridiag)
        vecses.append(vecs)
    end = time.time()
    print("Lanczos time: {}".format(end-start)) # this should be ~ num_samples * order * hvp compute time
    density, grids = density_lib.tridiag_to_density(tridiags, grid_len=10000, sigma_squared=1e-5)
    return density, grids

# @jax.jit
def compute_spectrum_exact(params,subset="LR"):

    # N = hessian.shape[0] // 2
    if subset == "L":
        # only differentiate wrt L
        @jax.jit
        def pl_loss_fn(_params):
            return loss_fn(_params["L"],params["R"])
        h_loss_fn = pl_loss_fn
    elif subset == "R":
        @jax.jit
        def pr_loss_fn(_params):
            return loss_fn(params["L"],_params["R"])
        h_loss_fn = pr_loss_fn
    else:
        assert subset == "LR"
        h_loss_fn = p_loss_fn
    hessian = hessian_computation.full_hessian(h_loss_fn,params)
    vals, vecs = jnp.linalg.eigh(hessian)
    density, grids = density_lib.eigv_to_density(vals.reshape(1,-1), grid_len=10000, sigma_squared=1e-5)
    out_d = {
        "vals": vals,
        "vecs": vecs,
        "density": density,
        "grids": grids
    }
    return out_d

def plot_density(grids, density, label=None):
    plt.semilogy(grids, density, label=label)
    # plt.ylim(1e-10, 1e2)
    plt.ylabel("Density")
    plt.xlabel("Eigenvalue")
#   plt.legend()
    plt.show()

# density, grids = compute_spectrum(paramses[0])
# plot_density(grids,density)

# compute analytic hessian
LR_H_d = compute_spectrum_exact(paramses[4],subset="LR")
L_H_d = compute_spectrum_exact(paramses[4],subset="L")
R_H_d = compute_spectrum_exact(paramses[4],subset="R")
LR_H_vals = LR_H_d["vals"][::-1]
L_H_vals = L_H_d["vals"][::-1]
R_H_vals = R_H_d["vals"][::-1]
print("LR",LR_H_vals[0],LR_H_vals[0]/LR_H_vals[9])
print("L",L_H_vals[0],L_H_vals[0]/L_H_vals[9])
print("R",R_H_vals[0],R_H_vals[0]/R_H_vals[9])

L_sigma = jnp.linalg.svd(paramses[4]["L"])[1]
R_sigma = jnp.linalg.svd(paramses[4]["R"])[1]
print(L_sigma[0],R_sigma[0],L_sigma[0]/R_sigma[0])

import pdb; pdb.set_trace()

# density, grids = compute_spectrum(paramses[1])
# plot_density(grids,density)
# density, grids = compute_spectrum(paramses[-1])
# plot_density(grids,density)