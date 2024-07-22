import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from module.systems.response_runge import Response_HindmarshRose
import pandas as pd
from matplotlib import pyplot as plt


plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0


def response_3d_plot(k_1, k_2):
    response = Response_HindmarshRose()
    r_results = response.simulate(t_max=22000 + 1, dt=0.01, washout=20000, s=4.75, k_1=k_1, k_2=k_2)

    results = pd.DataFrame(r_results).T
    results = results.rename(columns={0: "x", 1: "y", 2: "z", 3: "phi"})

    # fig = plt.figure(figsize=(27, 8))
    fig = plt.figure(figsize=(13, 11))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_facecolor('white')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    x1 = results['x']
    y1 = results['y']
    z1 = results['phi']
    ax.plot(x1, y1, z1, color='b', linestyle='-', linewidth=3.0, label='Data Points')
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    ax.set_xlabel('$x_r$', fontsize=42, labelpad=20,rotation=7)
    ax.set_ylabel('$y_r$', fontsize=42, labelpad=20,rotation=7)
    ax.set_zlabel('$ϕ_r$', fontsize=42, labelpad=30,rotation=7)
    ax.xaxis.line.set_color("black")
    ax.yaxis.line.set_color("black")
    ax.zaxis.line.set_color("black")
    ax.xaxis.line.set_linewidth(1)
    ax.yaxis.line.set_linewidth(1)
    ax.zaxis.line.set_linewidth(1)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.set_alpha(1)
    ax.yaxis.pane.set_alpha(1)
    ax.zaxis.pane.set_alpha(1)
    ax.view_init(elev=20, azim=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='z', labelsize=30, pad=15)
    fig.text(0.49, 0.9, '(c)', fontsize=60, weight='bold')

    plt.tight_layout()
    plt.savefig('images/fig_4_c.eps', dpi=300, format='eps')
    plt.savefig('images/fig_4_c.png', dpi=300, format='png')
    plt.show()

    fig = plt.figure(figsize=(13, 11))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    x1 = results['x']
    y1 = results['z']
    z1 = results['phi']
    ax.plot(x1, y1, z1, color='b', linestyle='-', linewidth=3.0, label='Data Points')
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    ax.set_xlabel('$x_r$', fontsize=42, labelpad=20,rotation=7)
    ax.set_ylabel('$z_r$', fontsize=42, labelpad=20,rotation=7)
    ax.set_zlabel('$ϕ_r$', fontsize=42, labelpad=30,rotation=7)
    ax.xaxis.line.set_color("black")
    ax.yaxis.line.set_color("black")
    ax.zaxis.line.set_color("black")
    ax.xaxis.line.set_linewidth(1)
    ax.yaxis.line.set_linewidth(1)
    ax.zaxis.line.set_linewidth(1)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.set_alpha(1)
    ax.yaxis.pane.set_alpha(1)
    ax.zaxis.pane.set_alpha(1)
    ax.view_init(elev=20, azim=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='z', labelsize=30, pad=15)
    fig.text(0.49, 0.9, '(d)', fontsize=60, weight='bold')

    plt.tight_layout()
    plt.savefig('images/fig_4_d.eps', dpi=300, format='eps')
    plt.savefig('images/fig_4_d.png', dpi=300, format='png')
    plt.show()


