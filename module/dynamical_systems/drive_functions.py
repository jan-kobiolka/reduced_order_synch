from module.systems.drive_runge import Drive_HindmarshRose
from module.dynamical_systems.spike_classification_function import spike_classification
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap

plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0



def plot_timeseries(k_1, k_2, label,y_lim=0):
    """
    Visualises the time series as five individual plots over time
    """
    drive = Drive_HindmarshRose()
    results = drive.simulate(t_max=22000 + 1, dt=0.01, washout=21000, s=4.75, k_1=k_1, k_2=k_2)
    t = np.arange(int(21000), int(22000 + 1), 0.01)
    fig, ax = plt.subplots(5, sharex=True)
    fig.set_size_inches(12.5, 6.5)
    plt.figure(figsize=(10, 6))

    ax[0].plot(t, results[0], 'b', linewidth=1.5)
    ax[0].set_ylabel("$x$", fontsize=30)
    # ax[0].set_ylabel("$x_m$", fontsize=20)
    ax[0].axhline(y=0.75, linestyle="--", color="g", linewidth=2)
    ax[0].tick_params(axis='y', labelsize=18)
    ax[0].tick_params(axis='x', labelsize=18)
    ax[0].xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax[0].yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[0].set_ylim([-2, 2.2])

    ax[1].plot(t, results[1], 'b', linewidth=1.5)
    ax[1].set_ylabel("$y$", fontsize=30)
    # ax[1].set_ylabel("$y_m$", fontsize=20)
    ax[1].tick_params(axis='y', labelsize=18)
    ax[1].tick_params(axis='x', labelsize=18)
    ax[1].xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax[1].yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    y_min = results[1].min()
    y_max = results[1].max()
    ax[1].set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    ax[2].plot(t, results[2], 'b', linewidth=1.5)
    # ax[2].set_ylabel("$z_m$", fontsize=20)
    ax[2].set_ylabel("$z$", fontsize=30)
    ax[2].tick_params(axis='y', labelsize=18)
    ax[2].tick_params(axis='x', labelsize=18)
    ax[2].xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax[2].yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax[2].grid(True, which='both', linestyle='--', linewidth=0.5)
    y_min = results[2].min()
    y_max = results[2].max()
    ax[2].set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    ax[3].plot(t, results[3], 'b', linewidth=1.5)
    ax[3].set_ylabel("$w$", fontsize=30)
    # ax[3].set_ylabel("$w_m$", fontsize=20)
    ax[3].tick_params(axis='y', labelsize=18)
    ax[3].tick_params(axis='x', labelsize=18)
    ax[3].xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax[3].yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax[3].grid(True, which='both', linestyle='--', linewidth=0.5)
    y_min = results[3].min()
    y_max = results[3].max()
    ax[3].set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    ax[4].plot(t, results[4], 'b', linewidth=1.5)
    ax[4].set_ylabel("$ϕ$", fontsize=30)
    # ax[4].set_ylabel("$ϕ_m$", fontsize=20)
    ax[4].tick_params(axis='y', labelsize=18)
    ax[4].tick_params(axis='x', labelsize=18)
    ax[4].xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax[4].yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax[4].grid(True, which='both', linestyle='--', linewidth=0.5)
    y_min = results[4].min()
    y_max = results[4].max()
    ax[4].set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    if y_lim == "set":
        ax[1].set_ylim([-2, 2])
        ax[2].set_ylim([-6, 6])
        ax[3].set_ylim([-3, 3])
        ax[4].set_ylim([-3, 3])

    ax[-1].set_xlabel("t", fontsize=30)
    fig.align_ylabels(ax[:])
    plt.tight_layout()
    fig.text(0.49, 0.9, f'({label})', fontsize=30, weight='bold')
    fig.savefig(f"images/fig_1_{label}.eps", dpi=300, format='eps')
    fig.savefig(f'images/fig_1_{label}.png', dpi=300, format='png')
    plt.show()



def drive_inter_param(t_max: int, dt: float, washout: int,param_min: float, param_max: float, step: float, param: str,
                       compute: bool) -> pd.DataFrame():
    """
    Plots the inter spike interval of the 5-D Hindhmarsh rose neuron model as a specified parameter is varied.
    It uses the implementation from module.dynamics.drive_runge

    :param int t_max: Time frame
    :param float dt: Constant integration size
    :param int washout: Removal of transient time
    :param float param_min: min parameter size
    :param float param_max: max parameter size
    :param float step: specified step size
    :param str param: The parameter to be varied accepts ("s" and "gamma")
    :param bool compute: False: results in the previous data being loaded. True: results in actual computations
    :return:
    """
    if compute == False:
        df = pd.read_csv(f"""data/drive_isi_{param}.csv""")
        plot_drive_inter_param(t_max, dt, washout, df, param, param_min, param_max, label)
        return df

    if compute == True:
        from module.systems.drive_runge import Drive_HindmarshRose
        values = np.arange(param_min, param_max, step)
        df = pd.DataFrame(columns=[str(p) for p in values])
        for value in values:
            hr = Drive_HindmarshRose()
            if param == "s":
                timeseries = hr.simulate(t_max=t_max, dt=dt, washout=washout, s=value,k_1=0.7,k_2=0.5)
            elif param == "k_1":
                timeseries = hr.simulate(t_max=t_max, dt=dt, washout=washout, k_1=value,k_2 = 1.0, s=3.875)
            elif param == "k_2":
                timeseries = hr.simulate(t_max=t_max, dt=dt, washout=washout, k_2=value, k_1 = 1.0, s=3.875)
            elif param == "sigma":
                timeseries = hr.simulate(t_max=t_max, dt=dt, washout=washout, sigma=value, k_1=1.0, k_2=1.0,
                                         s=3.875)
            else:
                raise ValueError('param not implemented')
            timeseries = timeseries[0]
            timeseries = np.where((timeseries[:-1] <= 0.75) & (timeseries[1:] > 0.75), 1,
                                  0)  # counts a spike when it reaches 0.75 from below
            df[str(value)] = timeseries
        df.columns = [f"{float(col):.4g}" for col in df.columns]
        df.to_csv(f"data/drive_isi_{param}.csv", index=False)
        plot_drive_inter_param(t_max, dt, washout, df, param, param_min, param_max)
        return df


def plot_drive_inter_param(t_max, dt, washout, df, param, param_min, param_max):
    """
    Just used as a plotting function by another function (drive_inter_param)
    """
    t = np.arange(washout, t_max - dt, dt)
    df.reset_index(drop=True, inplace=True)
    df = df.set_index(t)
    column_indices = {column: df.index[df[column] == 1] for column in df.columns}
    df2 = pd.DataFrame.from_dict(column_indices, orient='index')
    df2 = df2.diff(axis=1)

    sns.stripplot(data=df2.T, size=2, color='.3', linewidth=0, jitter=False)
    tick_positions = np.linspace(0, df2.shape[0], 5)
    tick_labels = np.interp(tick_positions, [0, df2.shape[0]], [param_min, param_max])
    plt.xticks(tick_positions, tick_labels, fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlabel(f"{param}", fontsize=20)
    plt.ylabel("ISI", fontsize=20)

    plt.tight_layout()
    plt.show()


def drive_chaos_s_plot():
    column_names = ["s", 'lya_e_1', 'lya_e_2', 'lya_e_3', 'lya_e_4', 'lya_e_5']
    data = []
    with open("data/drive_lya_s.txt", 'r') as file:
        for line in file:
            values = line.strip().split()
            data.append(values)

    df_lya = pd.DataFrame(data, columns=column_names)
    df_lya = df_lya.astype(float)
    df_lya = df_lya.drop('s', axis=1)
    df_lya = df_lya.drop('lya_e_5', axis=1)
    df_lya = df_lya.drop('lya_e_4', axis=1)

    df_s = pd.read_csv("data/drive_isi_s.csv")
    t = np.arange(1000, 3000 - 0.01, 0.01)
    df_s.reset_index(drop=True, inplace=True)
    df_s = df_s.set_index(t)
    column_indices = {column: df_s.index[df_s[column] == 1] for column in df_s.columns}
    df_s = pd.DataFrame.from_dict(column_indices, orient='index')
    df_s = df_s.diff(axis=1)

    s_values = np.arange(3, 5 + 0.01, 0.01)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 14))

    sns.stripplot(data=df_s.T, size=4, color='.3', linewidth=0, jitter=False, ax=ax[0])
    tick_positions = np.linspace(0, df_s.shape[0], 9)
    tick_labels = np.interp(tick_positions, [0, df_s.shape[0]], [3, 5])
    ax[0].set_xticks(tick_positions)
    ax[0].set_xticklabels(tick_labels, fontsize=25)
    ax[0].tick_params(axis='x', labelsize=25)
    ax[0].tick_params(axis='y', labelsize=25)
    ax[0].set_xlabel("s", fontsize=32)
    ax[0].set_ylabel("ISI", fontsize=32)
    ax[0].tick_params(axis='x', pad=10)
    ax[0].set_ylim(0, 90)
    ax[0].get_xaxis().set_visible(False)


    labels = ['$LE_1$', '$LE_2$', '$LE_3$']

    for i, column in enumerate(df_lya.columns):
        line, = ax[1].plot(s_values, df_lya[column], label=labels[i],linewidth=4.0)  # Assign label to each line

    ax[1].set_xlabel('s', fontsize=32)
    ax[1].set_ylabel('Three Largest LEs',fontsize=32)
    ax[1].set_ylim([-0.20, 0.05])
    ax[1].tick_params(axis='x', labelsize=25)
    ax[1].tick_params(axis='y', labelsize=25)
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[1].legend(loc='lower right',fontsize=20)
    ax[1].tick_params(axis='x', pad=10)
    ax[1].set_ylim(-0.20, 0.05)
    ax[1].legend(loc='lower center',fontsize=25)


    fig.align_ylabels(ax[:])
    plt.subplots_adjust(left=0.18, right=0.95, top=0.92, bottom=0.09, hspace=0.1)
    fig.text(0.49, 0.95, '(a)', fontsize=45, weight='bold')
    #plt.tight_layout()
    fig.savefig('images/fig_2_a.eps', dpi=250, format='eps')
    fig.savefig('images/fig_2_a.png', dpi=300, format='png')
    plt.show()


def drive_chaos_k_1_plot():
    column_names = ["K_1", 'lya_e_1', 'lya_e_2', 'lya_e_3', 'lya_e_4', 'lya_e_5']
    data = []
    with open("data/drive_lya_K1.txt", 'r') as file:
        for line in file:
            values = line.strip().split()
            data.append(values)

    df_lya = pd.DataFrame(data, columns=column_names)
    df_lya = df_lya.astype(float)
    df_lya = df_lya.drop('K_1', axis=1)
    df_lya = df_lya.drop('lya_e_5', axis=1)
    df_lya = df_lya.drop('lya_e_4', axis=1)


    df_k_1 = pd.read_csv("data/drive_isi_k_1.csv")
    t = np.arange(1000, 3000 - 0.01, 0.01)
    df_k_1.reset_index(drop=True, inplace=True)
    df_k_1 = df_k_1.set_index(t)
    column_indices = {column: df_k_1.index[df_k_1[column] == 1] for column in df_k_1.columns}
    df_k_1 = pd.DataFrame.from_dict(column_indices, orient='index')
    df_k_1 = df_k_1.diff(axis=1)

    values = np.arange(0, 5 + 0.01, 0.01)



    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 14))

    sns.stripplot(data=df_k_1.T, size=4, color='.3', linewidth=0, jitter=False, ax=ax[0])
    tick_positions = np.linspace(0, df_k_1.shape[0], 6)
    tick_labels = np.interp(tick_positions, [0, df_k_1.shape[0]], [0, 5])
    ax[0].set_xticks(tick_positions)
    ax[0].set_xticklabels(tick_labels, fontsize=25)
    ax[0].tick_params(axis='x', labelsize=25)
    ax[0].tick_params(axis='y', labelsize=25)
    ax[0].set_xlabel("$k_1$", fontsize=32)
    ax[0].set_ylabel("ISI", fontsize=32)
    ax[0].tick_params(axis='x', pad=10)
    ax[0].set_ylim(0, 160)
    ax[0].get_xaxis().set_visible(False)

    # labels = ['$\Lambda_1$', '$\Lambda_2$', '$\Lambda_3$']
    labels = ['$LE_1$', '$LE_2$', '$LE_3$']

    for i, column in enumerate(df_lya.columns):
        line, = ax[1].plot(values, df_lya[column], label=labels[i],linewidth=4.0)  # Assign label to each line

    ax[1].set_xlabel('$k_1$', fontsize=32)
    ax[1].set_ylabel('Three Largest LEs',fontsize=32)
    ax[1].set_ylim([-0.20, 0.05])
    ax[1].tick_params(axis='x', labelsize=25)
    ax[1].tick_params(axis='y', labelsize=25)
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[1].legend(loc='lower right',fontsize=20)
    ax[1].tick_params(axis='x', pad=10)
    ax[1].set_ylim(-0.20, 0.05)
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[1].legend(loc='lower center',fontsize=25)

    fig.align_ylabels(ax[:])

    #plt.tight_layout()
    plt.subplots_adjust(left=0.18, right=0.95, top=0.92, bottom=0.09, hspace=0.1)
    fig.text(0.49, 0.95, '(c)', fontsize=45, weight='bold')
    fig.savefig('images/fig_2_c.eps', dpi=250, format='eps')
    fig.savefig('images/fig_2_c.png', dpi=300, format='png')
    plt.show()

def drive_chaos_k_2_plot():
    column_names = ["K_2", 'lya_e_1', 'lya_e_2', 'lya_e_3', 'lya_e_4', 'lya_e_5']
    data = []
    with open("data/drive_lya_K2.txt", 'r') as file:
        for line in file:
            values = line.strip().split()
            data.append(values)

    df_lya = pd.DataFrame(data, columns=column_names)
    df_lya = df_lya.astype(float)
    df_lya = df_lya.drop('K_2', axis=1)
    df_lya = df_lya.drop('lya_e_5', axis=1)
    df_lya = df_lya.drop('lya_e_4', axis=1)


    df_k_2 = pd.read_csv("data/drive_isi_k_2.csv")
    t = np.arange(1000, 3000 - 0.01, 0.01)
    df_k_2.reset_index(drop=True, inplace=True)
    df_k_2 = df_k_2.set_index(t)
    column_indices = {column: df_k_2.index[df_k_2[column] == 1] for column in df_k_2.columns}
    df_k_2 = pd.DataFrame.from_dict(column_indices, orient='index')
    df_k_2 = df_k_2.diff(axis=1)
    values = np.arange(0, 2 + 0.01, 0.01)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 14))

    sns.stripplot(data=df_k_2.T, size=4, color='.3', linewidth=0, jitter=False, ax=ax[0])

    ax[0].legend([], [], frameon=False)
    tick_positions = np.linspace(0, df_k_2.shape[0], 5)
    tick_labels = np.interp(tick_positions, [0, df_k_2.shape[0]], [0, 2])
    ax[0].set_xticks(tick_positions)
    ax[0].set_xticklabels(tick_labels, fontsize=25)
    ax[0].tick_params(axis='x', labelsize=25)
    ax[0].tick_params(axis='y', labelsize=25)
    ax[0].set_xlabel("$k_2$", fontsize=32)
    ax[0].set_ylabel("ISI", fontsize=32)
    ax[0].tick_params(axis='x', pad=10)
    ax[0].set_ylim(0, 150)
    ax[0].get_xaxis().set_visible(False)

    #labels = ['$\Lambda_1$', '$\Lambda_2$', '$\Lambda_3$']
    labels = ['$LE_1$', '$LE_2$', '$LE_3$']

    for i, column in enumerate(df_lya.columns):
        line, = ax[1].plot(values, df_lya[column], label=labels[i],linewidth=4.0)  # Assign label to each line

    ax[1].set_xlabel('$k_2$', fontsize=32)
    ax[1].set_ylabel('Three Largest LEs',fontsize=32)
    ax[1].set_ylim([-0.20, 0.05])
    ax[1].tick_params(axis='x', labelsize=25)
    ax[1].tick_params(axis='y', labelsize=25)
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[1].tick_params(axis='x', pad=10)
    ax[1].set_ylim(-0.30, 0.05)
    ax[1].locator_params(axis='x', nbins=5)
    ax[1].legend(loc='lower center',fontsize=25)

    fig.align_ylabels(ax[:])
    plt.subplots_adjust(left=0.18, right=0.95, top=0.92, bottom=0.09, hspace=0.1)
    fig.text(0.49, 0.95, '(d)', fontsize=45, weight='bold')
    #plt.tight_layout()
    fig.savefig('images/fig_2_d.eps', dpi=250, format='eps')
    fig.savefig('images/fig_2_d.png', dpi=300, format='png')
    plt.show()


def drive_chaos_sigma_plot():
    column_names = ["sigma", 'lya_e_1', 'lya_e_2', 'lya_e_3', 'lya_e_4', 'lya_e_5']
    data = []
    with open("data/drive_lya_sigma.txt", 'r') as file:
        for line in file:
            values = line.strip().split()
            data.append(values)

    df_lya = pd.DataFrame(data, columns=column_names)
    df_lya = df_lya.astype(float)
    df_lya = df_lya.drop('sigma', axis=1)
    df_lya = df_lya.drop('lya_e_5', axis=1)
    df_lya = df_lya.drop('lya_e_4', axis=1)

    df_s = pd.read_csv("data/drive_isi_sigma.csv")
    t = np.arange(1000, 3000 - 0.01, 0.01)
    df_s.reset_index(drop=True, inplace=True)
    df_s = df_s.set_index(t)
    column_indices = {column: df_s.index[df_s[column] == 1] for column in df_s.columns}
    df_s = pd.DataFrame.from_dict(column_indices, orient='index')
    df_s = df_s.diff(axis=1)

    gamma_values = np.arange(0, 0.125 + 0.0001, 0.0001)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 14))

    sns.stripplot(data=df_s.T, size=4, color='.3', linewidth=0, jitter=False, ax=ax[0])
    tick_positions = np.linspace(0, df_s.shape[0], 9)
    tick_labels = np.interp(tick_positions, [0, df_s.shape[0]], [3, 5])
    ax[0].set_xticks(tick_positions)
    ax[0].set_xticklabels(tick_labels, fontsize=25)
    ax[0].tick_params(axis='x', labelsize=25)
    ax[0].tick_params(axis='y', labelsize=25)
    ax[0].set_xlabel("σ", fontsize=32)
    ax[0].set_ylabel("ISI", fontsize=32)
    ax[0].tick_params(axis='x', pad=10)
    ax[0].set_ylim(0, 90)
    ax[0].get_xaxis().set_visible(False)


    labels = ['$LE_1$', '$LE_2$', '$LE_3$']

    for i, column in enumerate(df_lya.columns):
        line, = ax[1].plot(gamma_values, df_lya[column], label=labels[i],linewidth=4.0)  # Assign label to each line

    ax[1].set_xlabel('σ', fontsize=32)
    ax[1].set_ylabel('Three Largest LEs',fontsize=32)
    ax[1].set_ylim([-0.20, 0.05])
    ax[1].tick_params(axis='x', labelsize=25)
    ax[1].tick_params(axis='y', labelsize=25)
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[1].legend(loc='lower right',fontsize=20)
    ax[1].tick_params(axis='x', pad=10)
    ax[1].set_ylim(-0.0750, 0.05)
    ax[1].legend(loc='lower center',fontsize=25)


    fig.align_ylabels(ax[:])
    plt.subplots_adjust(left=0.20, right=0.95, top=0.92, bottom=0.09, hspace=0.1)
    fig.text(0.49, 0.95, '(b)', fontsize=45, weight='bold')
    #plt.tight_layout()
    fig.savefig('images/fig_2_b.eps', dpi=250, format='eps')
    fig.savefig('images/fig_2_b.png', dpi=300, format='png')
    plt.show()


def bifurcation():
    df_k_1_k_2 = pd.read_csv(f"data/drive_isi_k_1_k_2.csv", index_col=0)
    df_k_1_k_2 = df_k_1_k_2[::-1]
    df_k_1_k_2 = df_k_1_k_2.drop(df_k_1_k_2.index[-1])
    translation_table = {
        4: 'Supra-threshold \nbursting',
        3: 'Supra-threshold \nspiking',
        2: 'Sub-threshold \nbursting',
        1: 'Sub-threshold \nspiking',
        0: 'No activity'
    }

    column_names = ["k_1","k_2", 'lya_e_1', 'lya_e_2', 'lya_e_3', 'lya_e_4', 'lya_e_5']
    data = []
    with open("data/drive_lya_K1_K2.txt", 'r') as file:
        for line in file:
            values = line.strip().split()
            data.append(values)
    df_lya_max = pd.DataFrame(data, columns=column_names)
    df_lya_max = df_lya_max.astype(float)
    df_lya_max["max"] = df_lya_max[['lya_e_1', 'lya_e_2', 'lya_e_3', 'lya_e_4', 'lya_e_5']].max(axis=1)
    table_lya = pd.pivot_table(df_lya_max, values=['max'], index=['k_2'], columns=['k_1'],
                           aggfunc='max', fill_value=None)
    table_lya = table_lya.drop(table_lya.index[0])
    table_lya = table_lya.iloc[::-1]


    fig, ax = plt.subplots(2, 1, figsize=(14.5, 20))

    color_palette = sns.color_palette(n_colors=len(translation_table) - 1)
    color_palette = [(0, 0, 0)] + color_palette
    ax[0] = sns.heatmap(df_k_1_k_2, yticklabels=6, xticklabels=10, ax=ax[0], linewidths=0.0, rasterized=True,
                        cmap=color_palette)

    cbar = ax[0].collections[0].colorbar
    r = cbar.vmax - cbar.vmin
    cbar.set_ticks([cbar.vmin + 0.5 * r / (5) + r * i / (5) for i in range(5)])
    cbar.set_ticklabels(list(translation_table.values())[::-1])
    ax[0].set_xlabel('$k_1$', fontsize=40)
    ax[0].set_ylabel('$k_2$', fontsize=40)
    tick_positions = np.linspace(0, df_k_1_k_2.shape[0], 5)
    tick_labels = np.interp(tick_positions, [0, df_k_1_k_2.shape[0]], [2, 0])
    ax[0].set_yticks(tick_positions)
    ax[0].set_yticklabels(np.round(tick_labels, 1))
    tick_positions_x = np.linspace(0, df_k_1_k_2.shape[1], 6)
    tick_labels_x = np.interp(tick_positions_x, [0, df_k_1_k_2.shape[1]], [0, 5])
    ax[0].set_xticks(tick_positions_x)
    ax[0].set_xticklabels(np.round(tick_labels_x, 1))
    ax[0].tick_params(axis='x', labelsize=30)
    ax[0].tick_params(axis='y', labelsize=30)
    cbar = ax[0].collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)
    ax[0].get_xaxis().set_visible(False)

    sns.heatmap(table_lya,cmap="PiYG", yticklabels=6, xticklabels=10, ax=ax[1],linewidths=0.0,rasterized=True,center=0,vmin=-0.001,vmax=0.001)
    ax[1].set_xlabel('$k_1$', fontsize=40)
    ax[1].set_ylabel('$k_2$', fontsize=40)
    tick_positions = np.linspace(0, df_k_1_k_2.shape[0], 5)
    tick_labels = np.interp(tick_positions, [0, df_k_1_k_2.shape[0]], [2, 0])
    ax[1].set_yticks(tick_positions)
    ax[1].set_yticklabels(np.round(tick_labels, 1))
    tick_positions_x = np.linspace(0, df_k_1_k_2.shape[1], 6)
    tick_labels_x = np.interp(tick_positions_x, [0, df_k_1_k_2.shape[1]], [0, 5])
    ax[1].set_xticks(tick_positions_x)
    ax[1].set_xticklabels(np.round(tick_labels_x, 1))
    ax[1].tick_params(axis='x', labelsize=30)
    ax[1].tick_params(axis='y', labelsize=30)
    cbar = ax[1].collections[0].colorbar
    cbar.ax.tick_params(labelsize=25)
    cbar.set_label("$\Lambda_{max}$", fontsize=32, labelpad=8)

    plt.subplots_adjust(hspace=0.2)
    plt.tight_layout()
    fig.text(0.02, 0.92, '(a)', fontsize=50, weight='bold')
    fig.text(0.02, 0.45, '(b)', fontsize=50, weight='bold')
    plt.savefig('images/fig_3.eps', dpi=250, format='eps')
    plt.savefig('images/fig_3.png', dpi=400, format='png')
    plt.show()




def drive_3d_plot(k_1, k_2):

    drive = Drive_HindmarshRose()
    d_results = drive.simulate(t_max=22000 + 1, dt=0.01, washout=20000, s=4.75, k_1=k_1, k_2=k_2)
    d_results=np.array(d_results)
    d_results=np.delete(d_results,3,0)
    # plot_3d(d_results.tolist())

    results = pd.DataFrame(d_results).T
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
    ax.set_xlabel('x', fontsize=42, labelpad=20)
    ax.set_ylabel('y', fontsize=42, labelpad=20)
    ax.set_zlabel('ϕ', fontsize=42, labelpad=30)
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
    fig.text(0.49, 0.9, '(a)', fontsize=60, weight='bold')

    plt.tight_layout()
    plt.savefig('images/fig_4_a.eps', dpi=300, format='eps')
    plt.savefig('images/fig_4_a.png', dpi=300, format='png')
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
    ax.set_xlabel('x', fontsize=42, labelpad=20)
    ax.set_ylabel('z', fontsize=42, labelpad=20)
    ax.set_zlabel('ϕ', fontsize=42, labelpad=30)
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
    fig.text(0.49, 0.9, '(b)', fontsize=60, weight='bold')

    plt.tight_layout()
    plt.savefig('images/fig_4_b.eps', dpi=300, format='eps')
    plt.savefig('images/fig_4_b.png', dpi=300, format='png')
    plt.show()


def Lya_k_1_k_2():
    column_names = ["k_1", 'k_2', "max_lya"]
    df = pd.read_csv("data/Lya_k_1_k_2.txt", delim_whitespace=True, header=None,
                     names=column_names)
    table = pd.pivot_table(df, values='max_lya', index=['k_2'], columns=['k_1'])
    table = table.drop(table.index[0])
    table = table.iloc[::-1]

    new_color = "forestgreen"  # Specify the color for 0 values
    cmap = plt.get_cmap("Reds")
    cmap_list = [cmap(i) for i in range(cmap.N)]
    cmap_list[0] = plt.cm.colors.to_rgba(new_color)
    custom_lya = ListedColormap(cmap_list, name="custom_cmap")


    fig, ax = plt.subplots(figsize=(10, 7.5))
    sns.heatmap(table, cmap=custom_lya, yticklabels=5, xticklabels=10, ax=ax,linewidths=0.0,rasterized=True)
    ax.set_xlabel('$k_1$', fontsize=32)
    ax.set_ylabel('$k_2$', fontsize=32)
    tick_positions = np.linspace(0, table.shape[0], 5)
    tick_labels = np.interp(tick_positions, [0, table.shape[0]], [2, 0])
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(np.round(tick_labels, 1))
    tick_positions_x = np.linspace(0, table.shape[1], 6)
    tick_labels_x = np.interp(tick_positions_x, [0, table.shape[1]], [0, 5])
    ax.set_xticks(tick_positions_x)
    ax.set_xticklabels(np.round(tick_labels_x, 1))
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    cbar = ax.collections[0].colorbar
    cbar.set_label('$\dot{V}$', fontsize=34, labelpad=50)
    cbar.ax.tick_params(labelsize=25)
    plt.subplots_adjust(hspace=0.2)
    plt.tight_layout()

    plt.savefig('images/fig_8.eps', dpi=250, format='eps')
    plt.savefig('images/fig_8.png', dpi=400, format='png')
    plt.show()