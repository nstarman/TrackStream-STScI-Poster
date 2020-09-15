---
jupyter:
  jupytext:
    formats: ipynb,md,Rmd
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.2
  kernel_info:
    name: utilipy
  kernelspec:
    display_name: dev
    language: python
    name: dev
---

# Starkman STScI Poster - Sept 2020

```python inputHidden=false jupyter={"outputs_hidden": false} outputHidden=false
"""
    TITLE   : Stellar Stream Track Reconstruction, with Errors
    AUTHOR  : Nathaniel Starkman
    PROJECT : STSCI Poster Presentation
""";

__author__ = 'Nathaniel Starkman'
__version__ = 'Aug 27, 2020'
```

<span style='font-size:30px;font-weight:650'>
    About
</span>

This notebook contains the code for all figures in my STScI poster.


<br><br>

- - - 



## Prepare

<!-- #region inputHidden=false outputHidden=false -->
### Imports
<!-- #endregion -->

```python inputHidden=false jupyter={"outputs_hidden": false} outputHidden=false
# My custom library of utility functions
from utilipy import ipython
ipython.run_imports(base=True, astropy=True, matplotlib=True, plotly=True)
ipython.set_autoreload(2)


# PROJECT-SPECIFIC
# warning this will probably be changed to `trackstream`

import streamtrack as st
from streamtrack import examples
from streamtrack import preprocess as prep
from streamtrack import process
from streamtrack.utils import convert
```

<br><br>

- - - 



## Code

```python inputHidden=false jupyter={"outputs_hidden": false} outputHidden=false
orb_ord = examples.make_ordered_orbit_data()
orb_obs = examples.make_noisy_orbit_data()

orb_clr = np.linspace(0, 256, len(orb_ord))
```

```python
trace_orbit = go.Scatter3d(
    x=orb_ord.x, y=orb_ord.y, z=orb_ord.z,
    name="orbit", mode="lines", marker=dict(color='black')
)

trace_data = go.Scatter3d(
    x=orb_obs.x, y=orb_obs.y, z=orb_obs.z,
    name="observed", mode="markers",
    marker=dict(color=orb_clr,
                colorscale="viridis"
               )
)

layout = go.Layout(
    height=700,
    showlegend=False,
    scene=dict(
        xaxis_title='x [kpc]',
        yaxis_title='y [kpc]',
        zaxis_title='z [kpc]',
    ),
    scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
)

fig = go.Figure(
    data=[trace_orbit, trace_data],
    layout=layout
)
fig.write_image("solar_circle_stream.pdf")
fig.show()
```

```python
start_point, start_ind = prep.find_closest_point(
    orb_obs, np.array((-5.1, -2, 0.)),
)
```

```python
visit_order, som = prep.apply_SOM(
    orb_obs,
    learning_rate=1.5, sigma=15,
    iterations=int(5e3), random_seed=20, plot=False,
    reorder=start_ind,
    return_som=True
)
```

```python
fig, ax = plt.subplots(figsize=(5, 4))

rep = orb_obs.represent_as(coord.CartesianRepresentation)
data = rep._values.view("f8").reshape(-1, len(rep.components))

pts = ax.scatter(
    data[visit_order, 0],
    data[visit_order, 1],
    c=np.arange(0, len(data)),
    vmax=len(data),
    cmap="plasma_r",
    label="data",
)

ax.plot(data[visit_order][:, 0], data[visit_order][:, 1], c="gray")
ax.set_xlabel("x [kpc]", fontsize=16)
ax.set_ylabel("y [kpc]", fontsize=16)

cbar = plt.colorbar(pts, ax=ax)
cbar.ax.set_ylabel("SOM ordering", fontsize=14)

fig.tight_layout()
plt.savefig("SOM_ordering.pdf")
plt.show();
```

```python
orb_repdata, orb_trmat, orb_visit_orders, orb_start_point = prep.preprocess(
    orb_obs,
    start_point=np.array((-6, -6, 0.0)),
    iterations=int(3e3),
    learning_rate=1.5,
    sigma=15,
    plot=False,
    N_repeats=np.arange(0, 10, 1),
    _tqdm=True,
)

orb_repdata *= u.kpc

orb_best_order = prep.draw_most_probable_ordering(orb_trmat)
orb_arr = orb_repdata[orb_best_order].xyz.T.value
```

```python
orb_dts = process.make_dts(orb_arr, dt0=0.5, N=6, axis=1, plot=True)
orb_dt = orb_dts[0]

plt.gcf().set_size_inches(6.5, 5)
plt.title("")
plt.legend(loc="upper left")
plt.xlabel("SOM Index", fontsize=15)
plt.ylabel("Point-to-Point Distance [kpc]", fontsize=15)

plt.savefig("time_proxy.pdf")
plt.show();
```

```python
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag

orb_kf = KalmanFilter(dim_x=6, dim_z=3)
orb_kf.F = process.make_F(orb_dt)
orb_kf.Q = process.make_Q(dt=orb_dt, var=0.01, n_dims=3)
orb_kf.H = process.make_H()
orb_kf.R = process.make_R([0.05, 0.05, 0.003])[0]  # error in x, y, z

x = orb_arr[0]  # fist point
v = [0, 0, 0]
orb_kf.x = st.utils.intermix_arrays(x, v)  # intersperse velocity data

p = np.array([[0.0001, 0], [0, 1]])
orb_kf.P = block_diag(p, p, p)
```

```python
from collections import namedtuple
run_output = namedtuple("run_output", field_names=["Xs", "Ps", "Fs", "Qs"])
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import predict, update, rts_smoother

def make_F(dt):
    F_comp = np.array([[1, dt],
                       [0, 1]])
    F = block_diag(F_comp, F_comp, F_comp)
    return F
# /def

def make_Q(dt, dim=2, var=1, block_size=3):
    Q = Q_discrete_white_noise(dim=dim, dt=dt, var=var, block_size=block_size)
    return Q
# /def

def run_with_updatestep(data, dts, x0, P, R, H, u=0., B=0, alpha=1., *, qkw={}):
    x = x0

    Xs, Ps, Fs, Qs = [], [], [], []
    for i, z in enumerate(data):
        # F, Q
        F = make_F(dts[i])
        Q = make_Q(dts[i], **qkw)

        # predict
        x, P = predict(x, P=P, F=F, Q=Q, u=u, B=B, alpha=alpha)

        # update
        x, P, y, K, S, loglik = update(x, P=P, z=z, R=R, H=H, return_all=True)

        Xs.append(x)
        Ps.append(P)
        Fs.append(F)
        Qs.append(Q)

    Xs, Ps = np.array(Xs), np.array(Ps)
    Fs, Qs = np.array(Fs), np.array(Qs)

    # smooth
    sXs, sPs, sFs, sQs = rts_smoother(Xs, Ps, Fs, Qs)
    
    output = run_output(Xs, Ps, Fs, Qs)
    smooth = run_output(sXs, sPs, sFs, sQs)
    
    return output, smooth

orb_output, orb_smooth = run_with_updatestep(
    orb_arr, orb_dts, x0=orb_kf.x,
    P=orb_kf.P, R=orb_kf.R,
    H=orb_kf.H, u=0, B=0, alpha=1.,
    qkw=dict(dim=2, var=0.01, block_size=3)
#     qkw=dict(order=2, var=0.01, n_dims=3)
)
```

```python
# fig, axs = process.plot_path(
#     orb_repdata,
#     path = coord.CartesianRepresentation(x=orb_smooth.Xs[:,0]*u.kpc,
#                                          y=orb_smooth.Xs[:,2]*u.kpc,
#                                          z=orb_smooth.Xs[:,4]*u.kpc),
#     cov=orb_smooth.Ps,
#     true_path=orb_ord);


```

```python
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
from filterpy.stats import plot_covariance  # TODO replace

is_ordered = False
num_std=1
data = orb_repdata[orb_best_order]
true_path=orb_ord
cov_alpha=0.5
cov=orb_smooth.Ps
path = coord.CartesianRepresentation(x=orb_smooth.Xs[:,0]*u.kpc,
                                         y=orb_smooth.Xs[:,2]*u.kpc,
                                         z=orb_smooth.Xs[:,4]*u.kpc)

c = np.arange(0, len(data)),
ax.scatter(data.x, data.y, c=c, cmap="plasma_r", alpha=1)
if true_path is not None:
    ax.plot(true_path.x, true_path.y, c="gray", alpha=0.5)
if cov is not None:
    plt.sca(ax)
    for i, p in enumerate(cov):
        P = np.array([[p[0, 0], p[2, 0]], [p[0, 2], p[2, 2]]])
        mean = (path.x[i], path.y[i])
        plot_covariance(
            mean, cov=P, fc="tab:blue", ec='tab:blue', std=2, alpha=1
        )
    ax.set_aspect("auto")
ax.set_xlabel("x [kpc]", fontsize=16)
ax.set_ylabel("y [kpc]", fontsize=16)

plt.tight_layout()

cbar = plt.colorbar(scat, ax=ax)
cbar.ax.set_ylabel("SOM order", fontsize=13)

plt.savefig("kalman_path_xy.pdf")
plt.show();
```

```python
# plt.scatter(orb_smooth.Xs[:,0], orb_smooth.Xs[:,2])

data = coord.Galactocentric(orb_repdata[orb_best_order])

arclen = process.utils.p2p_distance(orb_smooth.Xs)

path = coord.CartesianRepresentation(x=orb_smooth.Xs[:,0]*u.kpc,
                                     y=orb_smooth.Xs[:,2]*u.kpc,
                                     z=orb_smooth.Xs[:,4]*u.kpc)
path = coord.Galactocentric(path)


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
plt.plot(arclen, np.zeros_like(arclen), c="k", label="stream path\nand uncertainty")

for i, p in enumerate(cov):
    P = np.array([[p[0, 0], p[2, 0]], [p[0, 2], p[2, 2]]])
    mean = (arclen[i], 0)
    plot_covariance(
        mean, cov=P, fc="tab:gray", ec='tab:gray', std=2, alpha=0.2
    )
ax.set_aspect("auto")

c = np.arange(0, len(data))
plt.scatter(arclen, data.separation_3d(path) * np.sign(data.data.norm() - path.data.norm()), c=c, cmap="plasma_r", label="data")

# plt.ylim((-1, 1))
plt.xlabel("Stream Arc-length [kpc]")
plt.ylabel("Distance from Path [kpc]")
plt.legend(fontsize=10)

cbar = plt.colorbar(scat, ax=ax)

plt.savefig("path_residual.pdf")
plt.show();
```

```python
# fig, axs = plt.subplots(1, 2, figsize=(12, 4))
# from filterpy.stats import plot_covariance  # TODO replace

# is_ordered = False
# num_std=1
# data = orb_repdata
# true_path=orb_ord
# cov_alpha=0.5
# cov=orb_smooth.Ps
# path = coord.CartesianRepresentation(x=orb_smooth.Xs[:,0]*u.kpc,
#                                          y=orb_smooth.Xs[:,2]*u.kpc,
#                                          z=orb_smooth.Xs[:,4]*u.kpc)

# c = np.arange(0, len(data.x))
# axs[0].scatter(data.x, data.y, c=c, cmap="plasma_r", alpha=1)
# if true_path is not None:
#     axs[0].plot(true_path.x, true_path.y, c="gray", alpha=0.5)
# if cov is not None:
#     plt.sca(axs[0])
#     for i, p in enumerate(cov):
#         P = np.array([[p[0, 0], p[2, 0]], [p[0, 2], p[2, 2]]])
#         mean = (path.x[i], path.y[i])
#         plot_covariance(
#             mean, cov=P, fc="tab:blue", ec='tab:blue', std=2, alpha=1
#         )
#     axs[0].set_aspect("auto")
# axs[0].set_xlabel("x [kpc]", fontsize=16)
# axs[0].set_ylabel("y [kpc]", fontsize=16)


# scat = axs[1].scatter(data.x, data.z, c=c, cmap="plasma_r")
# if true_path is not None:
#     axs[1].plot(true_path.x, true_path.z, c="gray", alpha=1)
# if cov is not None:
#     plt.sca(axs[1])
#     for i, p in enumerate(cov):
#         P = np.array([[p[0, 0], p[4, 0]], [p[0, 4], p[4, 4]]])
#         mean = (path.x[i], path.z[i])
#         plot_covariance(
#             mean, cov=P, fc="tab:blue", ec='tab:blue', std=num_std, alpha=1
#         )
#     axs[1].set_aspect("auto")
# axs[1].set_xlabel("x [kpc]", fontsize=16)
# axs[1].set_ylabel("z [kpc]", fontsize=16)
# # axs[1].yaxis.set_label_position("right")
# # axs[1].yaxis.tick_right()

# plt.tight_layout()

# cbar = plt.colorbar(scat, ax=axs)
# cbar.ax.set_ylabel("SOM order", fontsize=15)

# plt.savefig("kalman_path.pdf")
# plt.show();
```

<br><br>

- - - 

<span style='font-size:40px;font-weight:650'>
    END
</span>