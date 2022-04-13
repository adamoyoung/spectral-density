import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.example_libraries import optimizers
import optax
import matplotlib.pyplot as plt
import copy
from pprint import pprint
from tqdm import tqdm
import seaborn as sns

import lanczos
import density as density_lib
import hessian_computation
import metrics
import time


def setup(train_mode="sgd",init="bad"):

    D = 10 #1000
    N = 100
    master_key = jax.random.PRNGKey(25412541)
    init_keys = jax.random.split(master_key,num=10+1)
    master_key = init_keys[0]
    if init == "bad":
        L = 100*jax.random.normal(init_keys[1],shape=(N,D)) # L is bigger
        R = 0.01*jax.random.normal(init_keys[2],shape=(N,D))
    else:
        assert init == "good"
        L = jax.random.normal(init_keys[1],shape=(N,D)) # L is bigger
        R = jax.random.normal(init_keys[2],shape=(N,D))
    M = 0.5*jax.random.normal(init_keys[3],shape=(N,N))
    # make symmetric
    M = M + M.T
    params = {"L": L, "R": R}


    def lr_model_fn(L,R):
        return L @ R.T

    def p_model_fn(params):
        return lr_model_fn(params["L"],params["R"])

    def m_loss_fn(M_hat):
        return jnp.linalg.norm(M_hat-M)

    def lr_loss_fn(L,R):
        return m_loss_fn(lr_model_fn(L,R))

    def p_loss_fn(params):
        # could also be m_loss_fn(p_model_fn(params))
        return lr_loss_fn(params["L"],params["R"])

    print(">>> Optimal")
    # compute analytic solution
    U,S,V = jnp.linalg.svd(M,hermitian=True)
    M_hat = U[:,:D] @ jnp.diag(S[:D]) @ V[:D,:]
    opt_loss = m_loss_fn(M_hat)
    opt_norm = jnp.linalg.norm(M_hat)
    print("Loss",opt_loss)
    print("M_norm",opt_norm)

    print(">>> initial")
    print("Loss",p_loss_fn(params))
    print("L_norm",jnp.linalg.norm(L))
    print("R_norm",jnp.linalg.norm(R))
    print("M_norm",jnp.linalg.norm(M))

    T = 10000

    if train_mode == "sgd":

        alpha = 0.01
        optimizer = optax.chain(optax.scale(-alpha))

    else:

        assert train_mode == "warmup"
        alpha_min = 0.00001
        alpha_max = 1.0
        schedule_fn = optax.warmup_exponential_decay_schedule(
            init_value=alpha_min,
            peak_value=alpha_max,
            end_value=None,#10**((jnp.log10(alpha_min)+jnp.log10(alpha_max))/2),
            warmup_steps=T//3,
            transition_steps=1,
            decay_rate=0.99

        )
        optimizer = optax.chain(optax.scale(-1.),optax.scale_by_schedule(schedule_fn))
    
    opt_state = optimizer.init(params)

    @jax.jit
    def update_fn(t,params,opt_state):
        loss, grads = jax.value_and_grad(p_loss_fn)(params)
        updates, opt_state = optimizer.update(grads,opt_state,params)
        params = optax.apply_updates(params,updates)
        return loss, params, opt_state

    ts, paramses, losses, L_norms, R_norms, LR_norms = [], [], [], [], [], []

    for t in range(T):
        if t > 0:
            loss, params, opt_state = update_fn(t,params,opt_state)
        else:
            # initial loss
            loss = p_loss_fn(params)
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

    result_d = {
        "ts": jnp.array(ts),
        "paramses": paramses,
        "losses": jnp.array(losses),
        "L_norms": jnp.array(L_norms),
        "R_norms": jnp.array(R_norms),
        "LR_norms": jnp.array(LR_norms),
        "opt_loss": opt_loss,
        "opt_norm": opt_norm,
        "p_loss_fn": p_loss_fn,
        "p_model_fn": p_model_fn,
        "m_loss_fn": m_loss_fn
    }

    return result_d

def plot_loss(result_ds,color_ds):
    num_plots = 4
    # cs = sns.color_palette(n_colors=num_plots)
    fig, axs = plt.subplots(nrows=1,ncols=num_plots,figsize=(15,4))
    for k,v in result_ds.items():
        ts = v["ts"]
        mask = (jnp.arange(len(ts)) % 25) == 0
        ts = ts[mask]
        axs[0].plot(ts,v["losses"][mask],label=k,color=color_ds[k])
        axs[0].set_title("Loss")
        axs[0].hlines(v["opt_loss"],ts[0],ts[-1],label=None,color="grey",linestyle="--")
        axs[1].plot(ts,v["L_norms"][mask],label=k,color=color_ds[k])
        axs[1].set_title("L Norm")
        axs[2].plot(ts,v["R_norms"][mask],label=k,color=color_ds[k])
        axs[2].set_title("R Norm")
        axs[3].plot(ts,v["LR_norms"][mask],label=k,color=color_ds[k])
        axs[3].set_title("LR Norm")
        axs[3].hlines(v["opt_norm"],ts[0],ts[-1],label=None,color="grey",linestyle="--")
    axs[3].legend(loc="upper right")
    for i in range(num_plots):
        axs[i].set_yscale("log")
    # fig.legend(loc=13)
    fig.tight_layout()
    # fig.subplots_adjust(right=12/15)
    fig.savefig("loss.png")
    plt.show()
    plt.clf()

def plot_metrics(result_ds,color_ds):

    num_plots = 4
    # cs = sns.color_palette(n_colors=num_plots)
    fig, axs = plt.subplots(nrows=1,ncols=num_plots,figsize=(15,4))
    for k,v in result_ds.items():
        ts = v["metric_ts"]
        axs[0].plot(ts,v["hvp_max_eig_val"],label=k,color=color_ds[k])
        axs[0].set_title("Hessian Eigv 1")
        axs[1].plot(ts,v["hvp_eig_val_ratio_top10"],label=k,color=color_ds[k])
        axs[1].set_title("Hessian Eigv 1 / Eigv 10")
        axs[2].plot(ts,v["ggnvp_trace_ratio_top1"],label=k,color=color_ds[k])
        axs[2].set_title("Gauss-Newton Trace / Eigv 1")
        axs[3].plot(ts,v["jjvp_grad_energy_ratio_top1"],label=k,color=color_ds[k])
        axs[3].set_title("Total / Projected Gradient Energy")
    axs[0].set_yscale("log")
    axs[3].legend(loc="upper right")
    # for i in range(num_plots):
    #     axs[i].set_yscale("log")
    fig.tight_layout()
    fig.savefig("metrics.png")
    plt.show()
    plt.clf()

def main():

    sgd_good_d = setup(train_mode="sgd",init="good")
    sgd_bad_d = setup(train_mode="sgd",init="bad")
    warmup_d = setup(train_mode="warmup",init="bad")

    result_ds = {
        "naive_good": sgd_good_d,
        "naive_bad": sgd_bad_d,
        "warmup_bad": warmup_d
    }

    colors = sns.color_palette(n_colors=3)
    color_ds = {
        "naive_good": colors[0],
        "naive_bad": colors[1],
        "warmup_bad": colors[2]
    }

    plot_loss(result_ds,color_ds)

    # result_keys = result_ds.keys()
    for k,v in result_ds.items():

        # v = result_ds[k]
        num_samples = 1
        step = 1000
        T = 10000

        mvp_keys = {
            "hvp": ["max_eig_val","eig_val_ratio_top10"],
            "ggnvp": ["trace_ratio_top1"],
            "jjvp": ["grad_energy_ratio_top1"]
        }
        ts = []
        metrics = {}
        for mvp_type,keys in mvp_keys.items():
            for key in keys:
                mvp_key = mvp_type+"_"+key
                metrics[mvp_key] = []
        for t in range(0,T,step):
            print(f">>> step {t}")
            for mvp_type in mvp_keys.keys():
                spec_d = analyze(
                    v["paramses"][t],
                    v["p_loss_fn"],
                    v["p_model_fn"],
                    v["m_loss_fn"],
                    mvp_type=mvp_type,
                    num_samples=num_samples,
                    get_jac=True
                )
                metric_d = compute_metrics(spec_d)
                for key in mvp_keys[mvp_type]:
                    metrics[mvp_type+"_"+key].append(metric_d[key])
            ts.append(t)
        for kk,vv in metrics.items():
            result_ds[k][kk] = vv
        result_ds[k]["metric_ts"] = ts

    plot_metrics(result_ds,color_ds)

    # spec_d = analyze(paramses[0],p_loss_fn,p_model_fn,m_loss_fn,mvp_type=mvp_type,num_samples=num_samples,get_jac=True)
    # metric_d = compute_metrics(spec_d)
    # pprint(metric_d)
    # plot_density(spec_d["grids"],spec_d["density"])

    # spec_d = analyze(paramses[100],p_loss_fn,p_model_fn,m_loss_fn,mvp_type=mvp_type,num_samples=num_samples,get_jac=True)
    # metric_d = compute_metrics(spec_d)
    # pprint(metric_d)
    # plot_density(spec_d["grids"],spec_d["density"])

    # # compute analytic hessian
    # LR_H_d = compute_spectrum_exact(paramses[4],subset="LR")
    # L_H_d = compute_spectrum_exact(paramses[4],subset="L")
    # R_H_d = compute_spectrum_exact(paramses[4],subset="R")
    # LR_H_vals = LR_H_d["vals"][::-1]
    # L_H_vals = L_H_d["vals"][::-1]
    # R_H_vals = R_H_d["vals"][::-1]
    # print("LR",LR_H_vals[0],LR_H_vals[0]/LR_H_vals[9])
    # print("L",L_H_vals[0],L_H_vals[0]/L_H_vals[9])
    # print("R",R_H_vals[0],R_H_vals[0]/R_H_vals[9])

    # L_sigma = jnp.linalg.svd(paramses[4]["L"])[1]
    # R_sigma = jnp.linalg.svd(paramses[4]["R"])[1]
    # print(L_sigma[0],R_sigma[0],L_sigma[0]/R_sigma[0])

    # import pdb; pdb.set_trace()

    # density, grids = compute_spectrum(paramses[1])
    # plot_density(grids,density)
    # density, grids = compute_spectrum(paramses[-1])
    # plot_density(grids,density)


# def plot(ts,**kwargs):
#     num_plots = len(kwargs.keys())
#     cm = plt.cm.get_cmap("viridis",num_plots)
#     fig, axs = plt.subplots(nrows=1,ncols=num_plots)
#     for ax_idx, (k,v) in enumerate(kwargs.items()):
#         ax = axs[ax_idx]
#         ax.plot(ts,v,label=k,color=cm(ax_idx))
#     fig.legend()
#     fig.tight_layout()
#     plt.show()
#     plt.clf()

# def plot_density(grids, density, label=None):
#     plt.semilogy(grids, density, label=label)
#     # plt.ylim(1e-10, 1e2)
#     plt.ylabel("Density")
#     plt.xlabel("Eigenvalue")
# #   plt.legend()
#     plt.show()

def analyze(params,p_loss_fn,p_model_fn,m_loss_fn,mvp_type="hvp",order=90,num_samples=10,get_jac=False):

    # data does not matter, just for interface compatibility
    batches_list = [None]
    def batches_fn():
        for batch in batches_list:
            yield batch
    num_batches = len(batches_list)

    if mvp_type == "hvp":

        def hvp_loss_fn(params,batch):
            return p_loss_fn(params)

        hvp, unravel, num_params = hessian_computation.get_hvp_fn(hvp_loss_fn, params, batches_fn)
        mvp_cl = lambda v: hvp(params, v) / num_batches

    elif mvp_type == "ggnvp":

        def ggnvp_wz_fn(params,batch):
            return p_model_fn(params)
        
        def ggnvp_zl_fn(params,batch):
            return m_loss_fn(params)

        ggnvp, unravel, num_params = hessian_computation.get_ggnvp_fn(ggnvp_wz_fn,ggnvp_zl_fn,params,batches_fn)
        mvp_cl = lambda v: ggnvp(params,v)
    
    elif mvp_type == "jjvp":

        def jjvp_loss_fn(params,batch):
            return p_loss_fn(params)

        jjvp, unravel, num_params = hessian_computation.get_jjvp_fn(jjvp_loss_fn, params, batches_fn)
        mvp_cl = lambda v: jjvp(params, v) / num_batches

    else:

        raise ValueError(f"invald mvp_type {mvp_type}")

    if get_jac:

        def jac_loss_fn(params,batch):
            return p_loss_fn(params)
        jac = hessian_computation.get_jac(jac_loss_fn, params, batches_fn)
        # print(jac.shape, jac.sum())

    else:

        jac = None

    out_d = compute_spectrum(mvp_cl,num_params,jac,order,num_samples)
    return out_d

def compute_spectrum(mvp_cl,num_params,jac,order,num_samples):

    print("num_params: {}".format(num_params))
    start = time.time()
    mvp_cl(jnp.ones(num_params)) # first call of a jitted function compiles it
    end = time.time()
    print("hvp compile time: {}".format(end-start))
    start = time.time()
    mvp_cl(2*jnp.ones(num_params)) # second+ call will be much faster
    end = time.time()
    print("hvp compute time: {}".format(end-start))

    # @jax.jit
    def get_tridiag_vecs(key):
      return lanczos.lanczos_alg(mvp_cl, num_params, order, rng_key=key)

    rng = jax.random.PRNGKey(420420)
    rngs = jax.random.split(rng,num=num_samples+1)
    rng = rngs[0]
    start = time.time()
    tridiags, lcz_vecs = [], []
    for i in tqdm(range(num_samples),desc="lanczos_samples"):
        tridiag, lcz_vec = get_tridiag_vecs(rngs[i+1])
        tridiags.append(tridiag)
        lcz_vecs.append(lcz_vec)
    end = time.time()
    print("Lanczos time: {}".format(end-start)) # this should be ~ num_samples * order * hvp compute time
    eig_vals, _, eig_vecs = density_lib.tridiag_to_eigv(tridiags, get_eig_vecs=True)
    density, grids = density_lib.tridiag_to_density(tridiags, grid_len=10000, sigma_squared=1e-5)
    out_d = {
        "eig_vals": eig_vals,
        "eig_vecs": eig_vecs,
        "lcz_vecs": jnp.stack(lcz_vecs,axis=0),
        "density": density,
        "grids": grids
    }
    if not (jac is None):
        out_d["jac"] = jac
    return out_d

# @jax.jit
def compute_spectrum_exact(params,subset="LR"):

    # N = hessian.shape[0] // 2
    if subset == "L":
        # only differentiate wrt L
        @jax.jit
        def pl_loss_fn(_params):
            return lr_loss_fn(_params["L"],params["R"])
        h_loss_fn = pl_loss_fn
    elif subset == "R":
        @jax.jit
        def pr_loss_fn(_params):
            return lr_loss_fn(params["L"],_params["R"])
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

def compute_metrics(spec_d):

    start = time.time()
    l1_energy_pos, l1_energy_neg = metrics.l1_energy(spec_d["density"],spec_d["grids"])
    l1_energy_log_ratio = jnp.log10(l1_energy_pos) - jnp.log10(l1_energy_neg)
    max_eig_val = metrics.max_eig_val(spec_d["eig_vals"])
    min_eig_val = metrics.min_eig_val(spec_d["eig_vals"])
    trace = metrics.trace_eig_vals(spec_d["eig_vals"])
    # trace_over_max = trace / max_eig_val
    metric_d = {
        "l1_energy_pos": l1_energy_pos,
        "l1_energy_neg": l1_energy_neg,
        "l1_energy_log_ratio": l1_energy_log_ratio,
        "max_eig_val": max_eig_val,
        "min_eig_val": min_eig_val,
        "trace": trace
    }
    for k in range(10):
        metric_d[f"trace_ratio_top{k+1}"] = metrics.trace_over_topk(spec_d["eig_vals"],k+1)
        metric_d[f"eig_val_ratio_top{k+1}"] = metrics.eig_val_ratio(spec_d["eig_vals"],k+1)
    if "jac" in spec_d:
        grad_energy = metrics.gradient_energy(spec_d["jac"])
        metric_d["gradient_energy"] = grad_energy
        for k in range(10):
            grad_energy_ratio_k = metrics.gradient_energy_ratio(spec_d["jac"],spec_d["eig_vecs"],spec_d["lcz_vecs"],k+1)
            metric_d[f"grad_energy_ratio_top{k+1}"] = grad_energy_ratio_k
    metric_d = {k:float(jnp.nan_to_num(v,nan=-1.)) for k,v in metric_d.items()}
    end = time.time()
    print("Metrics time: {}".format(end-start))
    return metric_d

if __name__ == "__main__":
    main()

# """
# So I was able to reproduce part of the toy example that Justin was referring to. 
# We're trying to find kxn matrices L,R such that |L@R.T - M| is small, where M is a symmetric nxn matrix (the target) 
# and | | is the Frobenius matrix ity(grids,density)(grids,density). The trick is to initialize L to have a large norm and R to have a small norm, 
# so the problem is poorly conditioned. Naive gradient descent will slowly reduce both L norm and R norm 
# (i.e. bring their values closer to 0), but if you use a learning rate warmup the L matrix will reduce in norm more quickly. 
# Initially this results in a huge increase in loss, but eventually the L norm is low so when you bring the learning rate 
# back down it can actually produce a better solution.
# """