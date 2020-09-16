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

The only issue with the code is the random state, which changes on each run.


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

# BUILT-IN

from collections import namedtuple


# THIRD PARTY

from scipy.linalg import block_diag
from filterpy.stats import plot_covariance  # a very nice plotter function


# PROJECT-SPECIFIC
# warning this will probably be changed to `trackstream`

import trackstream as st
from trackstream import examples
from trackstream import preprocess as prep
from trackstream import process
from trackstream.utils import convert
```

```python
run_output = namedtuple("run_output", field_names=["Xs", "Ps", "Fs", "Qs"])
```

<br><br>

- - - 



## Load Mock Stream

```python inputHidden=false jupyter={"outputs_hidden": false} outputHidden=false
# Load ordered mock stream
orb_ord = examples.make_ordered_orbit_data()
# Load noisy and shuffled stream
orb_obs = examples.make_noisy_orbit_data()

orb_clr = np.linspace(0, 256, len(orb_ord))  # color array
```

Plot mock stream.

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

<br><br>

- - - 



## Self-Organizing Maps


Find starting point for SOM

```python
start_point, start_ind = prep.find_closest_point(
    orb_obs, np.array((-5.1, -2, 0.)),
)
```

Apply a single SOM.

```python
visit_order, som = prep.apply_SOM(
    orb_obs,
    learning_rate=1.5, sigma=15,
    iterations=int(1e4), random_seed=20, plot=False,
    reorder=start_ind,
    return_som=True
)
```

Plot SOM ordered data.

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

Do this again, applying the full pre-processing pipeline.

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

orb_rep = coord.Galactocentric(orb_repdata[orb_best_order])
orb_arr = orb_rep.data.xyz.T.value

```

Plot Point-to-Point Distance.

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

<br><br>

- - - 



## Kalman Filter


Create initial Kalman state

```python
orb_kf = process.KalmanFilter()
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

Run Kalman Filter

```python
orb_output, orb_smooth = process.batch_predict_with_stepupdate(
    orb_arr, orb_dts, x0=orb_kf.x,
    P=orb_kf.P, R=orb_kf.R,
    H=orb_kf.H, u=0, B=0, alpha=1.,
#     qkw=dict(dim=2, var=0.01, block_size=3)
    qkw=dict(order=2, var=0.01, n_dims=3)
)

path = coord.Galactocentric(
    coord.CartesianRepresentation(
        x=orb_smooth.Xs[:,0]*u.kpc,
        y=orb_smooth.Xs[:,2]*u.kpc,
        z=orb_smooth.Xs[:,4]*u.kpc
    )
)

arclen = process.utils.p2p_distance(orb_smooth.Xs)
```

Plot Output

```python
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

scat = ax.scatter(orb_rep.x, orb_rep.y,
                  c=np.arange(0, len(orb_rep)), cmap="plasma_r", alpha=1)
ax.plot(orb_ord.x, orb_ord.y, c="gray", alpha=0.5)

for i, p in enumerate(orb_smooth.Ps):
    P = np.array([[p[0, 0], p[2, 0]],
                  [p[0, 2], p[2, 2]]])
    plot_covariance((path.x[i], path.y[i]), cov=P,
                    fc="tab:blue", ec='tab:blue', std=2, alpha=1)
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
fig, ax = plt.subplots(1, 1, figsize=(8, 4))

plt.plot(arclen, np.zeros_like(arclen), c="k", label="stream path\nand uncertainty")
for i, p in enumerate(orb_smooth.Ps):
    P = np.array([[p[0, 0], p[2, 0]],
                  [p[0, 2], p[2, 2]]])
    plot_covariance((arclen[i], 0), cov=P,
                    fc="tab:gray", ec='tab:gray', std=2, alpha=0.2)
ax.scatter(arclen, orb_rep.separation_3d(path) * np.sign(orb_rep.data.norm() - path.data.norm()),
           c=np.arange(0, len(orb_rep)), cmap="plasma_r", label="data")
plt.colorbar(scat, ax=ax)

ax.set_aspect("auto")
plt.xlabel("Stream Arc-length [kpc]")
plt.ylabel("Distance from Path [kpc]")
plt.legend(fontsize=10)

plt.savefig("path_residual.pdf")
plt.show();
```

<br><br>

- - - 

<span style='font-size:40px;font-weight:650'>
    END
</span>
