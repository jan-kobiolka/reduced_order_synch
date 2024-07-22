import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from numba import jit
from matplotlib.ticker import MaxNLocator

plt.rcParams['axes.xmargin'] = 0


@jit(nopython=True, parallel=False)
def drive_system(x, y, z, w, phi, a, b, c, d, alpha, beta, I, sigma, theta, s, x_0, y_0, mu,
                 gamma, delta, k_1, k_2):
    dx_dt = (a * x ** 2) - (b * x ** 3) + y - z - k_1 * (alpha + 3 * beta * phi ** 2) * x + I
    dy_dt = c - d * x ** 2 - y - sigma * w
    dz_dt = theta * (s * (x - x_0) - z)
    dw_dt = mu * (gamma * (y - y_0) - delta * w)
    dphi_dt = x - k_2 * phi
    return np.array([dx_dt, dy_dt, dz_dt, dw_dt, dphi_dt])


class Drive_HindmarshRose:
    """
    A class to simulate and visualise the behavior of the 5-D Hindmarsh-Rose neuron model.
    """

    def __init__(self):
        self.t_max = None
        self.dt = None
        self.results = None
        self.washout = None

    def simulate(self, t_max: int, dt: float, washout: int, a=3, b=1, alpha=0.1, beta=0.02, I=3.1, c=1, d=5,
                 sigma=0.0278,
                 theta=0.006, s=4.75, x_0=-1.56, y_0=-1.619, mu=0.0009, gamma=3, delta=0.9573, k_1=0.7,
                 k_2=0.5) -> np.array:
        """
        Generates the time series of the 5-D Hindmarsh-Rose neuron model using fourth order rungnee kutta method.
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
        :param optional sigma: Model param
        :param optional theta: Model param
        :param optional s: Model param
        :param optional x_0: Model param
        :param optional y_0: Model param
        :param optional mu: Model param
        :param optional gamma: Model param
        :param optional delta: Model param
        :param optional k_1: Model param
        :param optional k_2: Model param
        :return
        """
        self.t_max = t_max
        self.dt = dt
        self.washout = washout
        results = self.compute(t_max=t_max, dt=dt, a=a, b=b, c=c, d=d, alpha=alpha, beta=beta, I=I, sigma=sigma,
                               theta=theta, s=s, x_0=x_0, y_0=y_0, mu=mu, gamma=gamma, delta=delta, k_1=k_1, k_2=k_2)
        washout = int(washout / dt)
        self.results = [arr[washout:] for arr in results]
        return self.results

    @staticmethod
    @jit(nopython=True, parallel=False)
    def compute(t_max, dt, a, b, c, d, alpha, beta, I, sigma, theta, s, x_0, y_0, mu, gamma, delta, k_1, k_2):
        integration_step = int(t_max / dt)
        x, y, z, w, phi = [np.zeros(integration_step) for _ in range(5)]
        x[0], y[0], z[0], w[0], phi[0] = 0.1, 0.2, 0.3, 0.1, 0.2

        for t in range(1, integration_step):
            # Uses fourth order rungnee kutta
            k1 = drive_system(x=x[t - 1], y=y[t - 1], z=z[t - 1], w=w[t - 1], phi=phi[t - 1],
                              a=a, b=b, c=c, d=d, alpha=alpha, beta=beta, I=I, sigma=sigma,
                              theta=theta, s=s, x_0=x_0, y_0=y_0, mu=mu, gamma=gamma, delta=delta, k_1=k_1, k_2=k_2)

            k2 = drive_system(x=x[t - 1] + 0.5 * dt * k1[0],
                              y=y[t - 1] + 0.5 * dt * k1[1],
                              z=z[t - 1] + 0.5 * dt * k1[2],
                              w=w[t - 1] + 0.5 * dt * k1[3],
                              phi=phi[t - 1] + 0.5 * dt * k1[4],
                              a=a, b=b, c=c, d=d, alpha=alpha, beta=beta, I=I, sigma=sigma,
                              theta=theta, s=s, x_0=x_0, y_0=y_0, mu=mu, gamma=gamma, delta=delta, k_1=k_1, k_2=k_2)

            k3 = drive_system(x=x[t - 1] + 0.5 * dt * k2[0],
                              y=y[t - 1] + 0.5 * dt * k2[1],
                              z=z[t - 1] + 0.5 * dt * k2[2],
                              w=w[t - 1] + 0.5 * dt * k2[3],
                              phi=phi[t - 1] + 0.5 * dt * k2[4],
                              a=a, b=b, c=c, d=d, alpha=alpha, beta=beta, I=I, sigma=sigma,
                              theta=theta, s=s, x_0=x_0, y_0=y_0, mu=mu, gamma=gamma, delta=delta, k_1=k_1, k_2=k_2)

            k4 = drive_system(x=x[t - 1] + dt * k3[0],
                              y=y[t - 1] + dt * k3[1],
                              z=z[t - 1] + dt * k3[2],
                              w=w[t - 1] + dt * k3[3],
                              phi=phi[t - 1] + dt * k3[4],
                              a=a, b=b, c=c, d=d, alpha=alpha, beta=beta, I=I, sigma=sigma,
                              theta=theta, s=s, x_0=x_0, y_0=y_0, mu=mu, gamma=gamma, delta=delta, k_1=k_1, k_2=k_2)

            x[t] = x[t - 1] + ((k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) * dt) / 6.0
            y[t] = y[t - 1] + ((k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) * dt) / 6.0
            z[t] = z[t - 1] + ((k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) * dt) / 6.0
            w[t] = w[t - 1] + ((k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) * dt) / 6.0
            phi[t] = phi[t - 1] + ((k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4]) * dt) / 6.0
        return [x, y, z, w, phi]

    def plot_timeseries(self):
        """
        Visualises the time series as five individual plots over time
        """
        t = np.arange(int(self.washout), int(self.t_max), self.dt)
        fig, ax = plt.subplots(5, sharex=True)
        fig.set_size_inches(10.5, 6.5)
        plt.figure(figsize=(10, 6))

        ax[0].plot(t, self.results[0], 'b')
        ax[0].set_ylabel("$x$", fontsize=20)
        # ax[0].set_ylabel("$x_m$", fontsize=20)
        ax[0].tick_params(axis='y', labelsize=14)
        ax[0].tick_params(axis='x', labelsize=14)
        ax[0].xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax[0].yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
        y_min = self.results[0].min()
        y_max = self.results[0].max()
        ax[0].set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

        ax[1].plot(t, self.results[1], 'b')
        ax[1].set_ylabel("$y$", fontsize=20)
        # ax[1].set_ylabel("$y_m$", fontsize=20)
        ax[1].tick_params(axis='y', labelsize=14)
        ax[1].tick_params(axis='x', labelsize=14)
        ax[1].xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax[1].yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
        y_min = self.results[1].min()
        y_max = self.results[1].max()
        ax[1].set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

        ax[2].plot(t, self.results[2], 'b')
        # ax[2].set_ylabel("$z_m$", fontsize=20)
        ax[2].set_ylabel("$z$", fontsize=20)
        ax[2].tick_params(axis='y', labelsize=14)
        ax[2].tick_params(axis='x', labelsize=14)
        ax[2].xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax[2].yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax[2].grid(True, which='both', linestyle='--', linewidth=0.5)
        y_min = self.results[2].min()
        y_max = self.results[2].max()
        ax[2].set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

        ax[3].plot(t, self.results[3], 'b')
        ax[3].set_ylabel("$w$", fontsize=20)
        # ax[3].set_ylabel("$w_m$", fontsize=20)
        ax[3].tick_params(axis='y', labelsize=14)
        ax[3].tick_params(axis='x', labelsize=14)
        ax[3].xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax[3].yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax[3].grid(True, which='both', linestyle='--', linewidth=0.5)
        y_min = self.results[3].min()
        y_max = self.results[3].max()
        ax[3].set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

        ax[4].plot(t, self.results[4], 'b')
        ax[4].set_ylabel("$ϕ$", fontsize=20)
        # ax[4].set_ylabel("$ϕ_m$", fontsize=20)
        ax[4].tick_params(axis='y', labelsize=14)
        ax[4].tick_params(axis='x', labelsize=14)
        ax[4].xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax[4].yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax[4].grid(True, which='both', linestyle='--', linewidth=0.5)
        y_min = self.results[4].min()
        y_max = self.results[4].max()
        ax[4].set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

        ax[-1].set_xlabel("t", fontsize=20)
        fig.align_ylabels(ax[:])
        plt.tight_layout()
        plt.show()



