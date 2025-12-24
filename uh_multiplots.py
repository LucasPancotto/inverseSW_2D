import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter

from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from types import SimpleNamespace

import pySPEC as ps
# from pySPEC.time_marching import SWHD_1D, Adjoint_SWHD_1D
from utils import *

sns.set_style("white")
# palette = sns.color_palette("mako", as_cmap=True)
sns.set_palette(palette='Dark2')

plt.rcParams.update({"text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
 "text.latex.preamble": r"""
        \usepackage{amsmath}
        \usepackage{mathptmx}
        \usepackage{bm}
    """
})

##################################################
# particular_path = '/home/lpancotto/code/tesis/inverseSW_2D/2dgauss_double_hdata/alpha_test/'
particular_path = '/home/lpancotto/code/tesis/inverseSW_2D/2dcosk5_hu200_hdata/'
path = particular_path + 'time_frames'
figure_path = path +'/figures'
#############################################
dt = 2.5e-2
N = 1024
L = 2*np.pi/4
eta0 = 0.05
h0 = 1
c = 1
tL = L/c
domain = np.linspace(0,2*np.pi, N)/L

tidxs = [0,60,120,170]
pinn_times = [np.load(f'{path}/predicted{tt}.npy') for tt in tidxs]
ref_times = [np.load(f'{path}/ref{tt}.npy') for tt in tidxs]

plt.close('all')
f1,axs = plt.subplots(nrows = 2, ncols=4, figsize=(20,15)) # originally figsize=(15,10)
f1.canvas.draw()

axs1 = axs[0]
axs2 = axs[-1]

# Loop over all subplots and plot uu
for j, ax in enumerate(axs1):

    ref = ref_times[j]
    U = (ref[0] - ref[0].mean())/c
    # V = ref[1]/c
    # ETA = (ref[2]-h0)/eta0


    pinn = pinn_times[j]
    u = (pinn[0] - pinn[0].mean())/c
    # v = (pinn[1] - pinn[1].mean())/c
    # h = pinn[2]
    # eta = (h-h0)/eta0

    ax.yaxis.set_major_formatter(FuncFormatter(u_latex_sci_notation))
    ax.set_xlim(domain[0], domain[-1])  # <-- This removes x-axis margin

    ys = np.arange(0, 1024, 200)
    alphas = np.linspace(0.3, 1.0, len(ys))
    lws = np.linspace(0.8, 2.5, len(ys))

    # take cut depending on case
    if particular_path == '/home/lpancotto/code/tesis/inverseSW_2D/2dcosk5_hu200_hdata/':
        y0=512
    else:
        y0=348

    ax.plot(domain, U[:,y0], label = r'$Ground$ $truth$', color='black')
    ax.plot(domain, u[:,y0], label = r'$PINN$', color = 'red', linestyle = '--', alpha=0.7)

    ax.set_title(r'$t=$'+f'${tidxs[j]/240:.1f}$'+ r'$T$', fontsize=24, pad=10)

    # Remove ylabel and y-tick labels from subplots except the first
    if j != 0:
        ax.set_ylabel('')
        ax.tick_params(labelleft=False)
    ax.tick_params(labelbottom=False)


# Match y-axis limits across all subplots
y_min = min(ax.get_ylim()[0] for ax in axs1)
y_max = max(ax.get_ylim()[1] for ax in axs1)
for ax in axs1:
    ax.set_ylim(y_min, y_max)

axs1[0].legend(fontsize = 22, loc='lower right')
axs1[0].set_ylabel(r'$u/c$', fontsize = 28)



# Apply scientific notation formatting
for i, ax in enumerate(axs2):
    ref = ref_times[i]
    # U = ref[0]/c
    # V = ref[1]/c
    ETA = (ref[2]-h0)/eta0


    pinn = pinn_times[i]
    # u = (pinn[0] - pinn[0].mean())/c
    # v = (pinn[1] - pinn[1].mean())/c
    h = pinn[2]
    eta = (h-h0)/eta0

    # ax.yaxis.set_major_formatter(FuncFormatter(h_latex_sci_notation))
    ax.set_xlim(domain[0], domain[-1])  # <-- This removes x-axis margin

        # Add your own offset label in the same place
    #if i==0:
        #ax.yaxis.set_major_formatter(FuncFormatter(u_latex_sci_notation))

    ax.plot(domain, ETA[:,y0], label = r'$Ground$ $truth$', color='black')
    ax.plot(domain, eta[:,y0], label = r'$PINN$', color = 'red', linestyle='--', alpha =0.7)

# axs2[0].legend(loc='center right', fontsize = 30)
axs2[0].set_ylabel(r'$\eta/\eta_0$', fontsize = 28) # no ylabel, only legend

# Remove ylabel and y-tick labels from subplots except the first
for i, ax in enumerate(axs2):
    if i != 0:
        ax.set_ylabel('')
        ax.tick_params(labelleft=False)

# Match y-axis limits across all subplots
y_min = min(ax.get_ylim()[0] for ax in axs2)
y_max = max(ax.get_ylim()[1] for ax in axs2)
for ax in axs2:
    ax.set_ylim(y_min, y_max)
# labels = [r'$(a)$', r'$(b)$']
# labels = [r'$\text{(a)}$', r'$\text{(b)}$', r'$\text{(c)}$', r'$\text{(d)}$']
labels1 = [r'$\mathrm{(a)}$', r'$\mathrm{(b)}$', r'$\mathrm{(c)}$', r'$\mathrm{(d)}$']
labels2 = [r'$\mathrm{(e)}$', r'$\mathrm{(f)}$', r'$\mathrm{(g)}$', r'$\mathrm{(h)}$']

for i, ax in enumerate(axs1):  # assuming a 1D array of Axes
    ax.text(0.90, 0.95, labels1[i], transform=ax.transAxes,
            ha='center', va='top', fontsize = 28)
    if i==0:
        ax.tick_params(which='both', direction='in', bottom=False, left=True, labelsize= 28)

for i, ax in enumerate(axs2):  # assuming a 1D array of Axes
    ax.text(0.90, 0.95, labels2[i], transform=ax.transAxes,
            ha='center', va='top', fontsize = 28)
    ax.tick_params(which='both', direction='in', bottom=True, left=False, labelsize= 28)
    ax.set_xlabel('$x/L$', fontsize = 28)
    if i==0:
        ax.tick_params(which='both', direction='in', bottom=True, left=True, labelsize= 28)

plt.tight_layout()

plt.savefig(f'{figure_path}/uh_nx.pdf')
print('saving in : ', f'{figure_path}/uh_nx.pdf')
