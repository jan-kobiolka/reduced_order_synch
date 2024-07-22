from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from module.machine_learning.echo_state_network_function import EchoStateNetwork, EchoStateObserver
from skopt import gp_minimize
from skopt.space import Real
from matplotlib.ticker import MaxNLocator


def plot_train_pred_controlled(drive, response, train_time, pred_time, control_time, model,label):
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0.05
    time_step = train_time + pred_time + control_time
    t = np.linspace(20000, 20000 + ((time_step) * 0.2), int(time_step))
    t = np.round(t, 1)

    fig, ax = plt.subplots(4, sharex=True)
    fig.set_size_inches(10.5, 6.5)
    plt.figure(figsize=(10, 6))

    ax[0].plot(t, drive[0, :], 'blue', label='Train')
    ax[0].plot(t, response[0, :], 'red', label='Train')
    ax[0].axvline(x=20000 + (train_time) * 0.2, color='g', label='End of Warmup', linestyle="--")
    ax[0].axvline(x=20000 + (train_time + pred_time) * 0.2, color='purple', label='End of Warmup', linestyle="--")
    ax[0].set_ylabel("x", fontsize=26)
    ax[0].tick_params(axis='x', labelsize=16)
    ax[0].tick_params(axis='y', labelsize=16)
    y_min = min(drive[0, :].min(), response[0, :].min())
    y_max = max(drive[0, :].max(), response[0, :].max())
    ax[0].set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    ax[1].plot(t, drive[1, :], 'blue', label='Train')
    ax[1].plot(t, response[1, :], 'red', label='Train')
    ax[1].axvline(x=20000 + (train_time) * 0.2, color='g', label='End of Warmup', linestyle="--")
    ax[1].axvline(x=20000 + (train_time + pred_time) * 0.2, color='purple', label='End of Warmup', linestyle="--")
    ax[1].set_ylabel("y", fontsize=26)
    ax[1].tick_params(axis='x', labelsize=16)
    ax[1].tick_params(axis='y', labelsize=16)
    y_min = min(drive[1, :].min(), response[1, :].min())
    y_max = max(drive[1, :].max(), response[1, :].max())
    ax[1].set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    ax[2].plot(t, drive[2, :], 'blue', label='Train')
    ax[2].plot(t, response[2, :], 'red', label='Train')
    ax[2].axvline(x=20000 + (train_time) * 0.2, color='g', label='End of Warmup', linestyle="--")
    ax[2].axvline(x=20000 + (train_time + pred_time) * 0.2, color='purple', label='End of Warmup', linestyle="--")
    ax[2].set_ylabel("z", fontsize=26)
    ax[2].tick_params(axis='x', labelsize=16)
    ax[2].tick_params(axis='y', labelsize=16)
    y_min = min(drive[2, :].min(), response[2, :].min())
    y_max = max(drive[2, :].max(), response[2, :].max())
    ax[2].set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    ax[3].plot(t, drive[3, :], 'blue', label='Train')
    ax[3].plot(t, response[3, :], 'red', label='Train')
    ax[3].axvline(x=20000 + (train_time) * 0.2, color='g', label='End of Warmup', linestyle="--")
    ax[3].axvline(x=20000 + (train_time + pred_time) * 0.2, color='purple', label='End of Warmup', linestyle="--")
    ax[3].set_ylabel("ϕ", fontsize=26)
    ax[3].tick_params(axis='x', labelsize=16)
    ax[3].tick_params(axis='y', labelsize=16)
    y_min = min(drive[3, :].min(), response[3, :].min())
    y_max = max(drive[3, :].max(), response[3, :].max())
    ax[3].set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    ax[-1].set_xlabel("t", fontsize=26)
    ax[-1].xaxis.set_major_locator(MaxNLocator(nbins=10))
    fig.align_ylabels(ax[:])
    plt.tight_layout()
    fig.text(0.01, 0.93, f'{label}', fontsize=30, weight='bold')
    fig.savefig(f"images/{model}.eps", dpi=300, format='eps')
    fig.savefig(f'images/{model}.png', dpi=300, format='png')
    plt.show()


def plot_transition_controlled(true_system, pred_system, start_time, control_time, model, label):
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0.05

    t = np.linspace(20000+ 0.2 + (start_time * 0.2), 20000+0.2 + ((control_time + start_time) * 0.2), control_time)

    fig, ax = plt.subplots(5, sharex=True)
    fig.set_size_inches(10.5, 6.5)
    plt.figure(figsize=(10, 6))

    ax[0].plot(t, true_system[0, :], 'blue', label='Train')
    ax[0].plot(t, pred_system[0, :], 'red', label='Train')
    ax[0].set_ylabel("$x_m$", fontsize=26)
    ax[0].tick_params(axis='x', labelsize=16)
    ax[0].tick_params(axis='y', labelsize=16)
    y_min = min(true_system[0, :].min(), pred_system[0, :].min())
    y_max = max(true_system[0, :].max(), pred_system[0, :].max())
    ax[0].set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    ax[1].plot(t, true_system[1, :], 'blue', label='Train')
    ax[1].plot(t, pred_system[1, :], 'red', label='Train')
    ax[1].set_ylabel("$y_m$", fontsize=26)
    ax[1].tick_params(axis='x', labelsize=16)
    ax[1].tick_params(axis='y', labelsize=16)
    y_min = min(true_system[1, :].min(), pred_system[1, :].min())
    y_max = max(true_system[1, :].max(), pred_system[1, :].max())
    ax[1].set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    ax[2].plot(t, true_system[2, :], 'blue', label='Train')
    ax[2].plot(t, pred_system[2, :], 'red', label='Train')
    ax[2].set_ylabel("$z_m$", fontsize=26)
    ax[2].tick_params(axis='x', labelsize=16)
    ax[2].tick_params(axis='y', labelsize=16)
    y_min = min(true_system[2, :].min(), pred_system[2, :].min())
    y_max = max(true_system[2, :].max(), pred_system[2, :].max())
    ax[2].set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    ax[3].plot(t, true_system[3, :], 'blue', label='Train')
    ax[3].plot(t, pred_system[3, :], 'red', label='Train')
    ax[3].set_ylabel("$w_m$", fontsize=26)
    ax[3].tick_params(axis='x', labelsize=16)
    ax[3].tick_params(axis='y', labelsize=16)
    y_min = min(true_system[3, :].min(), pred_system[3, :].min())
    y_max = max(true_system[3, :].max(), pred_system[3, :].max())
    ax[3].set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    ax[4].plot(t, true_system[4, :], 'blue', label='Train')
    ax[4].plot(t, pred_system[4, :], 'red', label='Train')
    ax[4].set_ylabel("$ϕ_m$", fontsize=26)
    ax[4].tick_params(axis='x', labelsize=16)
    ax[4].tick_params(axis='y', labelsize=16)
    y_min = min(true_system[4, :].min(), pred_system[4, :].min())
    y_max = max(true_system[4, :].max(), pred_system[4, :].max())
    ax[4].set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    ax[-1].set_xlabel("t", fontsize=20)
    fig.align_ylabels(ax[:])
    plt.tight_layout()
    fig.savefig(f"images/{model}.eps", dpi=300, format='eps')
    fig.savefig(f'images/{model}.png', dpi=300, format='png')
    plt.show()


def plot_transition_errors(data_1, data_2, control_time, train_time, pred_time, model):
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0.05
    time_step = train_time + pred_time
    t = np.arange(20000 + 0.2+ (time_step * 0.2), 20000+ 0.2 + ((time_step+ control_time) * 0.2), 0.2)

    if data_1.shape[0] == 5:
        fig, ax = plt.subplots(5, sharex=True)
        fig.set_size_inches(10.5, 6.5)
        plt.figure(figsize=(10, 6))

        ax[0].plot(t, data_1[0, :] - data_2[0, :], 'r', label='Train')
        ax[0].axhline(y=0, color='g', linestyle="--")
        ax[0].set_ylabel("$e_x$", fontsize=26)
        ax[0].tick_params(axis='x', labelsize=16)
        ax[0].tick_params(axis='y', labelsize=16)
        y_max = max(np.abs(data_1[0, :] - data_2[0, :]))
        # y_max = max(np.abs(response[0, :]-drive[0,:]))
        ax[0].set_ylim(-y_max - 0.1 * y_max, y_max + 0.1 * y_max)
        ax[0].ticklabel_format(style='sci', scilimits=(-0, 0), axis='y')

        ax[1].plot(t, data_1[1, :] - data_2[1, :], 'r', label='Train')
        ax[1].axhline(y=0, color='g', linestyle="--")
        ax[1].set_ylabel("$e_y$", fontsize=26)
        ax[1].tick_params(axis='x', labelsize=16)
        ax[1].tick_params(axis='y', labelsize=16)
        y_max = max(np.abs(data_1[1, 200:] - data_2[1, 200:]))
        # y_max = max(np.abs(response[1,:]-drive[1, :]))
        ax[1].set_ylim(-y_max - 0.1 * y_max, y_max + 0.1 * y_max)
        ax[1].ticklabel_format(style='sci', scilimits=(-0, 0), axis='y')

        ax[2].plot(t, data_1[2, :] - data_2[2, :], 'r', label='Train')
        ax[2].axhline(y=0, color='g', linestyle="--")
        ax[2].set_ylabel("$e_z$", fontsize=26)
        ax[2].tick_params(axis='x', labelsize=16)
        ax[2].tick_params(axis='y', labelsize=16)
        y_max = max(np.abs(data_1[2, 200:] - data_2[2, 200:]))
        # y_max = max(np.abs(response[2, :] - drive[2, :]))
        ax[2].set_ylim(-y_max - 0.1 * y_max, y_max + 0.1 * y_max)
        ax[2].ticklabel_format(style='sci', scilimits=(-0, 0), axis='y')

        ax[3].plot(t, data_1[3, :] - data_2[3, :], 'r', label='Train')
        ax[3].axhline(y=0, color='g', linestyle="--")
        ax[3].set_ylabel("$e_w$", fontsize=26)
        ax[3].tick_params(axis='x', labelsize=16)
        ax[3].tick_params(axis='y', labelsize=16)
        y_max = max(np.abs(data_1[3, 200:] - data_2[3, 200:]))
        # y_max = max(np.abs(response[3,:]-drive[3, :]))
        ax[3].set_ylim(-y_max - 0.1 * y_max, y_max + 0.1 * y_max)
        ax[3].ticklabel_format(style='sci', scilimits=(-0, 0), axis='y')

        ax[4].plot(t, data_1[4, :] - data_2[4, :], 'r', label='Train')
        ax[4].axhline(y=0, color='g', linestyle="--")
        ax[4].set_ylabel("$e_ϕ$", fontsize=26)
        ax[4].tick_params(axis='x', labelsize=16)
        ax[4].tick_params(axis='y', labelsize=16)
        y_max = max(np.abs(data_1[4, 200:] - data_2[4, 200:]))
        # y_max = max(np.abs(response[3,:]-drive[3, :]))
        ax[4].set_ylim(-y_max - 0.1 * y_max, y_max + 0.1 * y_max)
        ax[4].ticklabel_format(style='sci', scilimits=(-0, 0), axis='y')

        ax[-1].set_xlabel("t", fontsize=26)
        fig.align_ylabels(ax[:])
        plt.tight_layout()
        fig.savefig(f"images/{model}.eps", dpi=300, format='eps')
        fig.savefig(f'images/{model}.png', dpi=300, format='png')
        plt.show()


def plot_errors(response, drive, control_time, train_time, pred_time, model, label):
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0.05

    time = train_time + pred_time

    t = np.linspace(20000 + (time * 0.2), 20000 + ((time + control_time) * 0.2), control_time)
    t = np.round(t, 1)

    fig, ax = plt.subplots(4, sharex=True)
    fig.set_size_inches(11, 6.5)
    plt.figure(figsize=(10, 6))

    ax[0].plot(t, response[0, :] - drive[0, :], 'r', label='Train')
    ax[0].axhline(y=0, color='g', linestyle="--")
    ax[0].set_ylabel("$e_x$", fontsize=26)
    ax[0].tick_params(axis='x', labelsize=16)
    ax[0].tick_params(axis='y', labelsize=16)
    y_max = max(np.abs(response[0, 200:] - drive[0, 200:]))
    # y_max = max(np.abs(response[0, :]-drive[0,:]))
    ax[0].set_ylim(-y_max - 0.1 * y_max, y_max + 0.1 * y_max)

    ax[1].plot(t, response[1, :] - drive[1, :], 'r', label='Train')
    ax[1].axhline(y=0, color='g', linestyle="--")
    ax[1].set_ylabel("$e_y$", fontsize=26)
    ax[1].tick_params(axis='x', labelsize=16)
    ax[1].tick_params(axis='y', labelsize=16)
    y_max = max(np.abs(response[1, 200:] - drive[1, 200:]))
    # y_max = max(np.abs(response[1,:]-drive[1, :]))
    ax[1].set_ylim(-y_max - 0.1 * y_max, y_max + 0.1 * y_max)

    ax[2].plot(t, response[2, :] - drive[2, :], 'r', label='Train')
    ax[2].axhline(y=0, color='g', linestyle="--")
    ax[2].set_ylabel("$e_z$", fontsize=26)
    ax[2].tick_params(axis='x', labelsize=16)
    ax[2].tick_params(axis='y', labelsize=16)
    y_max = max(np.abs(response[2, 200:] - drive[2, 200:]))
    # y_max = max(np.abs(response[2, :] - drive[2, :]))
    ax[2].set_ylim(-y_max - 0.1 * y_max, y_max + 0.1 * y_max)

    ax[3].plot(t, response[3, :] - drive[3, :], 'r', label='Train')
    ax[3].axhline(y=0, color='g', linestyle="--")
    ax[3].set_ylabel("$e_ϕ$", fontsize=26)
    ax[3].tick_params(axis='x', labelsize=16)
    ax[3].tick_params(axis='y', labelsize=16)
    y_max = max(np.abs(response[3, 200:] - drive[3, 200:]))
    # y_max = max(np.abs(response[3,:]-drive[3, :]))
    ax[3].set_ylim(-y_max - 0.1 * y_max, y_max + 0.1 * y_max)

    ax[-1].set_xlabel("t", fontsize=26)
    fig.align_ylabels(ax[:])
    plt.tight_layout()
    fig.text(0.01, 0.93, f'{label}', fontsize=30, weight='bold')
    fig.savefig(f"images/{model}.eps", dpi=300, format='eps')
    fig.savefig(f'images/{model}.png', dpi=300, format='png')
    plt.show()


def plot_pred_errors(data_test, data_pred, train_time, pred_time, model, label,data_train=None):
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0.05
    time_step = train_time + pred_time
    t = np.linspace(20000, 20000 + ((time_step) * 0.2), time_step)
    t = np.round(t, 1)

    if data_train is not None:
        data = np.hstack([data_train, data_pred])
    else:
        data = data_pred

    if data_test.shape[0] == 4:
        fig, ax = plt.subplots(4, sharex=True)
        fig.set_size_inches(10.5, 6.5)
        variables = ["$e_x$", "$e_y$", "$e_z$", "$e_ϕ$"]

    elif data_test.shape[0] == 5:
        fig, ax = plt.subplots(5, sharex=True)
        fig.set_size_inches(10.5, 6.5)
        variables = ["$e_x$", "$e_y$", "$e_z$", "$e_w$", "$e_ϕ$"]

    for i, variable in enumerate(variables):
        y_max = max(np.abs(data_test[i, 200:] - data[i, 200:]))
        y_min = - y_max

        ax[i].plot(t, data_test[i,] - data[i,], 'blue', label='Target')
        ax[i].axvline(x=20000 + (train_time) * 0.2, color='g', label='End of Warmup', linestyle="--")
        ax[i].set_ylabel(variable, fontsize=26)
        ax[i].tick_params(axis='x', labelsize=16)
        ax[i].tick_params(axis='y', labelsize=16)
        ax[i].set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    ax[-1].set_xlabel("t", fontsize=26)
    fig.align_ylabels(ax[:])
    fig.text(0.01, 0.95, f'{label}', fontsize=30, weight='bold')
    plt.tight_layout()
    plt.savefig(f"images/{model}.eps", dpi=300, format='eps')
    plt.savefig(f'images/{model}.png', dpi=300, format='png')
    plt.show()


def plot_pred(data_test, data_pred, train_time, pred_time, model, label,data_train=None):
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0.05
    time_step = train_time + pred_time
    t = np.linspace(20000, 20000 + ((time_step) * 0.2), time_step)
    t = np.round(t, 1)

    if data_train is not None:
        data = np.hstack([data_train, data_pred])
    else:
        data = data_pred

    if data_test.shape[0] == 1:
        fig, ax = plt.subplots(1, sharex=True)
        fig.set_size_inches(10.5, 6.5)
        ax.plot(t, data_test[0, :time_step], 'blue', label='Target')
        ax.plot(t, data[0, :], 'red', label='Predicted')
        ax.axvline(x=20000 + (train_time) * 0.2, color='g', label='End of Warmup', linestyle="--")
        ax.set_ylabel("$x_m$", fontsize=26)
        ax.set_xlabel("t", fontsize=26)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        plt.tight_layout()
        plt.savefig(f"images/{model}.eps", dpi=300, format='eps')
        plt.savefig(f'images/{model}.png', dpi=300, format='png')
        plt.show()
        return None

    if data_test.shape[0] == 4:
        fig, ax = plt.subplots(4, sharex=True)
        fig.set_size_inches(10.5, 6.5)
        variables = ["$x_r$", "$y_r$", "$z_r$", "$ϕ_r$"]

    elif data_test.shape[0] == 5:
        fig, ax = plt.subplots(5, sharex=True)
        fig.set_size_inches(10.5, 6.5)
        variables = ["$x$", "$y$", "$z$", "$w$", "$ϕ$"]

    for i, variable in enumerate(variables):
        y_min = min(data_test[i, 100:time_step].min(), data[i, 100:].min())
        y_max = max(data_test[i, 100:time_step].max(), data[i, 100:].max())

        ax[i].plot(t, data_test[i,], 'blue', label='Target')
        ax[i].plot(t, data[i, :], 'red', label='Predicted')
        ax[i].axvline(x=20000 + (train_time) * 0.2, color='g', label='End of Warmup', linestyle="--")
        ax[i].set_ylabel(variable, fontsize=26)
        ax[i].tick_params(axis='x', labelsize=16)
        ax[i].tick_params(axis='y', labelsize=16)
        ax[i].set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    ax[-1].set_xlabel("t", fontsize=26)
    fig.align_ylabels(ax[:])
    fig.text(0.01, 0.95, f'{label}', fontsize=30, weight='bold')
    plt.tight_layout()
    plt.savefig(f"images/{model}.eps", dpi=300, format='eps')
    plt.savefig(f'images/{model}.png', dpi=300, format='png')
    plt.show()


def rmse(target: np.array, predicted: np.array) -> float:
    """
    Computes the NRMSE for the prediction time
    :param target:
    :param predicted:
    :param pred_time:
    :return:
    """
    return np.sqrt(mean_squared_error(y_true=target, y_pred=predicted))
