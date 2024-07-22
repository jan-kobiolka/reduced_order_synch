import numpy as np
from matplotlib import pyplot as plt
from numba import jit

plt.rcParams['axes.xmargin'] = 0
from matplotlib.ticker import MaxNLocator


# plt.rcParams['axes.ymargin'] = 0.01

@jit(nopython=True, parallel=False)
def response_system(x_r, y_r, z_r, phi_r, a, b, c, d, alpha, beta, I, theta, s, x_0, k_1, k_2):
    dx_dt_r = (a * x_r ** 2) - (b * x_r ** 3) + (y_r) - z_r - k_1 * (
            alpha + 3 * beta * phi_r ** 2) * x_r + I
    dy_dt_r = c - d * x_r ** 2 - y_r
    dz_dt_r = theta * (s * (x_r - x_0) - z_r)
    dphi_dt_r = x_r - k_2 * phi_r
    return np.array([dx_dt_r, dy_dt_r, dz_dt_r, dphi_dt_r])


class Response_HindmarshRose:
    """
    A class to simulate and visualise the behavior of the 4-D Hindmarsh-Rose neuron model.
    """

    def __init__(self):
        # Set within script
        self.t_max = None
        self.dt = None
        self.results = None

    def simulate(self, t_max: int, dt: float, washout: int, a=3, b=1, alpha=0.1, beta=0.02, I=3.1, c=1, d=5, theta=0.006,
                 s=4.75, x_0=-1.56, k_1=0.7, k_2=0.5) -> np.array:
        """
        Generates the time series of the 4-D Hindmarsh-Rose neuron model using fourth order rungnee kutta method.
        Speed up is generated via numba.

        :param int t_max: Until when the time series should be generated
        :param float dt: Constant integration step size
        :param int washout: How much of the transient time should be removed
        :param optional a: Model param
        :param optional b: Model param
        :param optional alpha: Model param
        :param optional beta: Model param
        :param optional I: Model param
        :param optional c: Model param
        :param optional d: Model param
        :param optional r: Model param
        :param optional s: Model param
        :param optional x_0: Model param
        :param optional k_1: Model param
        :param optional k_2: Model param
        :return:
        """
        self.t_max = t_max
        self.dt = dt
        self.washout = washout
        self.s = s
        results = self.compute(t_max=t_max, dt=dt, a=a, b=b, c=c, d=d, alpha=alpha, beta=beta, I=I, theta=theta, s=s, x_0=x_0,
                               k_1=k_1, k_2=k_2)
        washout = int(washout / dt)
        self.results = [arr[washout:] for arr in results]
        return self.results

    @staticmethod
    @jit(nopython=True, parallel=False)
    def compute(t_max, dt, a, b, c, d, alpha, beta, I, theta, s, x_0, k_1, k_2):
        integration_step = int(t_max / dt)
        x_r, y_r, z_r, phi_r = [np.zeros(integration_step) for _ in range(4)]

        x_r[0], y_r[0], z_r[0], phi_r[0] = 0.1, 0.2, 0.3, 0.1  # Initial values for the systems param
        for t in range(1, integration_step):
            # Uses fourth order rungnee kutta
            k1_r = response_system(x_r=x_r[t - 1], y_r=y_r[t - 1], z_r=z_r[t - 1], phi_r=phi_r[t - 1],
                                   a=a, b=b, c=c, d=d, alpha=alpha, beta=beta, I=I,
                                   theta=theta, s=s, x_0=x_0, k_1=k_1, k_2=k_2)

            k2_r = response_system(x_r=x_r[t - 1] + 0.5 * dt * k1_r[0],
                                   y_r=y_r[t - 1] + 0.5 * dt * k1_r[1],
                                   z_r=z_r[t - 1] + 0.5 * dt * k1_r[2],
                                   phi_r=phi_r[t - 1] + 0.5 * dt * k1_r[3],
                                   a=a, b=b, c=c, d=d, alpha=alpha, beta=beta, I=I,
                                   theta=theta, s=s, x_0=x_0, k_1=k_1, k_2=k_2)

            k3_r = response_system(x_r=x_r[t - 1] + 0.5 * dt * k2_r[0],
                                   y_r=y_r[t - 1] + 0.5 * dt * k2_r[1],
                                   z_r=z_r[t - 1] + 0.5 * dt * k2_r[2],
                                   phi_r=phi_r[t - 1] + 0.5 * dt * k2_r[3],
                                   a=a, b=b, c=c, d=d, alpha=alpha, beta=beta, I=I,
                                   theta=theta, s=s, x_0=x_0, k_1=k_1, k_2=k_2)

            k4_r = response_system(x_r=x_r[t - 1] + dt * k3_r[0],
                                   y_r=y_r[t - 1] + dt * k3_r[1],
                                   z_r=z_r[t - 1] + dt * k3_r[2],
                                   phi_r=phi_r[t - 1] + dt * k3_r[3],
                                   a=a, b=b, c=c, d=d, alpha=alpha, beta=beta, I=I,
                                   theta=theta, s=s, x_0=x_0, k_1=k_1, k_2=k_2)

            x_r[t] = x_r[t - 1] + ((k1_r[0] + 2 * k2_r[0] + 2 * k3_r[0] + k4_r[0]) * dt) / 6.0
            y_r[t] = y_r[t - 1] + ((k1_r[1] + 2 * k2_r[1] + 2 * k3_r[1] + k4_r[1]) * dt) / 6.0
            z_r[t] = z_r[t - 1] + ((k1_r[2] + 2 * k2_r[2] + 2 * k3_r[2] + k4_r[2]) * dt) / 6.0
            phi_r[t] = phi_r[t - 1] + ((k1_r[3] + 2 * k2_r[3] + 2 * k3_r[3] + k4_r[3]) * dt) / 6.0
        return [x_r, y_r, z_r, phi_r]

    def plot_timeseries(self):
        """
        Visualises the time series as four individual plots over time
        """
        t = np.arange(int(self.washout), int(self.t_max), self.dt)
        fig, ax = plt.subplots(4, sharex=True)
        fig.set_size_inches(10.5, 6.5)
        plt.figure(figsize=(10, 6))

        ax[0].plot(t, self.results[0], 'b')
        ax[0].set_ylabel("x", fontsize=20)
        # ax[0].set_ylabel("$x_s$", fontsize = 20)
        ax[0].tick_params(axis='y', labelsize=14)
        ax[0].tick_params(axis='x', labelsize=14)
        ax[0].xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax[0].yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
        y_min = self.results[0].min()
        y_max = self.results[0].max()
        ax[0].set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

        ax[1].plot(t, self.results[1], 'b')
        ax[1].set_ylabel("y", fontsize=20)
        # ax[1].set_ylabel("$y_s$", fontsize = 20)
        ax[1].tick_params(axis='y', labelsize=14)
        ax[1].tick_params(axis='x', labelsize=14)
        ax[1].xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax[1].yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
        y_min = self.results[1].min()
        y_max = self.results[1].max()
        ax[1].set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

        ax[2].plot(t, self.results[2], 'b')
        ax[2].set_ylabel("z", fontsize=20)
        # ax[2].set_ylabel("$z_s$", fontsize = 20)
        ax[2].tick_params(axis='y', labelsize=14)
        ax[2].tick_params(axis='x', labelsize=14)
        ax[2].xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax[2].yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax[2].grid(True, which='both', linestyle='--', linewidth=0.5)
        y_min = self.results[2].min()
        y_max = self.results[2].max()
        ax[2].set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

        ax[3].plot(t, self.results[3], 'b')
        ax[3].set_ylabel("ϕ", fontsize=20)
        # ax[3].set_ylabel("$ϕ_s$", fontsize = 20)
        ax[3].tick_params(axis='y', labelsize=14)
        ax[3].tick_params(axis='x', labelsize=14)
        ax[3].xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax[3].yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax[3].grid(True, which='both', linestyle='--', linewidth=0.5)
        y_min = self.results[3].min()
        y_max = self.results[3].max()
        ax[3].set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

        ax[-1].set_xlabel("t", fontsize=20)
        fig.align_ylabels(ax[:])
        plt.tight_layout()
        plt.show()