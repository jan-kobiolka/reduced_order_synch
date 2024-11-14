import numpy as np
from matplotlib import pyplot as plt
from numba import jit
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, MaxNLocator

plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
@jit(nopython=True, parallel=False)
def drive_system(x, y, z, w, phi):
    a = 3
    b = 1
    alpha = 0.1
    beta = 0.02
    I = 3.1
    c = 1
    d = 5
    sigma = 0.0278
    theta = 0.006
    s = 4.75
    x_0 = -1.56
    y_0 = -1.619
    mu = 0.0009
    gamma = 3
    delta = 0.9573
    k_1 = 0.94
    k_2 = 0.51
    dx_dt = (a * x ** 2) - (b * x ** 3) + y - z - k_1 * (alpha + 3 * beta * phi ** 2) * x + I
    dy_dt = c - d * x ** 2 - y - sigma * w
    dz_dt = theta * (s * (x - x_0) - z)
    dw_dt = mu * (gamma * (y - y_0) - delta * w)
    dphi_dt = x - k_2 * phi
    return np.array([dx_dt, dy_dt, dz_dt, dw_dt, dphi_dt])


@jit(nopython=True, parallel=False)
def response_system(x_r, y_r, z_r, phi_r, a_t, b_t, d_t, theta_t, g_1, e_1, g_2, e_2, g_3, e_3, g_4, e_4):
    alpha = 0.1
    beta = 0.02
    I = 3.1
    c = 1
    s = 4.75
    x_0 = -1.56
    k_1 = 0.94
    k_2 = 0.51
    dx_dt_r = (a_t * x_r ** 2) - (b_t * x_r ** 3) + (y_r) - z_r - k_1 * (
            alpha + 3 * beta * phi_r ** 2) * x_r + I - g_1 * e_1
    dy_dt_r = c - d_t * x_r ** 2 - y_r - g_2 * e_2
    dz_dt_r = theta_t * (s * (x_r - x_0) - z_r) - g_3 * e_3
    dphi_dt_r = x_r - k_2 * phi_r - g_4 * e_4
    return np.array([dx_dt_r, dy_dt_r, dz_dt_r, dphi_dt_r])


@jit(nopython=True, parallel=False)
def error_system(x_r, z_r, phi_r, a_t, b_t, d_t, theta_t, g_1, e_1, g_2, e_2, g_3, e_3, g_4, e_4, w):
    a = 3
    b = 1
    alpha = 0.1
    beta = 0.02
    sigma = 0.0278
    d = 5
    r = 0.006
    s = 4.75
    x_0 = -1.56
    k_1 = 0.94
    k_2 = 0.51
    e_1_dt = (2 * a * x_r - 3 * b * x_r ** 2 - k_1 * alpha - 3 * k_1 * beta * phi_r ** 2) * e_1 + e_2 - e_3 - 6 * k_1 \
             * beta * x_r * phi_r * e_4 + (a_t - a) * x_r ** 2 - (
                     b_t - b) * x_r ** 3 - g_1 * e_1
    e_2_dt = -2 * d * x_r * e_1 - e_2 - (d_t - d) * x_r ** 2 + sigma * w - g_2 * e_2
    e_3_dt = r * s * e_1 - r * e_3 + (theta_t - r) * (s * (x_r - x_0) - z_r) - g_3 * e_3
    e_4_dt = e_1 - k_2 * e_4 - g_4 * e_4
    return np.array([e_1_dt, e_2_dt, e_3_dt, e_4_dt])


@jit(nopython=True, parallel=False)
def g_system(e_1, e_2, e_3, e_4):
    g_1_dt = e_1 ** 2
    g_2_dt = e_2 ** 2
    g_3_dt = e_3 ** 2
    g_4_dt = e_4 ** 2
    return np.array([g_1_dt, g_2_dt, g_3_dt, g_4_dt])


@jit(nopython=True, parallel=False)
def param_system(x_r, z_r, e_1, e_2, e_3):
    s = 4.75
    x_0 = -1.56
    a_t_dt = -x_r ** 2 * e_1
    b_t_dt = x_r ** 3 * e_1
    d_t_dt = x_r ** 2 * e_2
    theta_t_dt = -(s * (x_r - x_0) - z_r) * e_3
    return np.array([a_t_dt, b_t_dt, d_t_dt, theta_t_dt])


class Error_dynamics:
    def __init__(self):
        self.washout = None
        self.t_max = None
        self.dt = None

        # Data
        self.drive = None
        self.response = None

    def simulate(self, t_max, dt, washout) -> np.array:
        self.t_max = t_max
        self.dt = dt
        self.washout = washout

        drive, response, error, uncertainty, controllers = self.compute(t_max, dt)
        washout = int(washout / dt)
        self.drive = [arr[washout:] for arr in drive]
        self.response = [arr[washout:] for arr in response]
        self.error = [arr[washout:] for arr in error]
        self.controlers = [arr[washout:] for arr in controllers]
        self.uncertainty = [arr[washout:] for arr in uncertainty]
        return [self.drive, self.response, error, uncertainty, controllers]

    @staticmethod
    @jit(nopython=True, parallel=False)
    def compute(t_max, dt):
        integration_step = int(t_max / dt)
        # Model parameters

        x, y, z, w, phi = [np.zeros(integration_step) for _ in range(5)]
        x_r, y_r, z_r, phi_r = [np.zeros(integration_step) for _ in range(4)]
        e_1, e_2, e_3, e_4 = [np.zeros(integration_step) for _ in range(4)]
        g_1, g_2, g_3, g_4 = [np.zeros(integration_step) for _ in range(4)]
        a_t, b_t, d_t, theta_t = [np.zeros(integration_step) for _ in range(4)]

        # Initial values
        x[0], y[0], z[0], w[0], phi[0] = 1.0,0.5,1.3,-0.5,-1.2
        x_r[0], y_r[0], z_r[0], phi_r[0] = 1.1,-2.2,-0.6,0.5


        e_1[0], e_2[0], e_3[0], e_4[0] = 2, -2, 2, -2
        g_1[0], g_2[0], g_3[0], g_4[0] = 0.5, 0.5, 0.5, 0.5
        a_t[0], b_t[0], d_t[0], theta_t[0] = 0, 0, 0, 0

        for t in range(1, integration_step):

            k1 = drive_system(x[t - 1], y[t - 1], z[t - 1], w[t - 1], phi[t - 1])
            k1_r = response_system(x_r[t - 1], y_r[t - 1], z_r[t - 1], phi_r[t - 1],
                                a_t[t - 1], b_t[t - 1], d_t[t - 1], theta_t[t - 1],
                                g_1[t - 1], e_1[t - 1], g_2[t - 1], e_2[t - 1], g_3[t - 1], e_3[t - 1],
                                g_4[t - 1], e_4[t - 1])
            k1_e = error_system(x_r[t - 1], z_r[t - 1], phi_r[t - 1],
                                a_t[t - 1], b_t[t - 1], d_t[t - 1], theta_t[t - 1],
                                g_1[t - 1], e_1[t - 1], g_2[t - 1], e_2[t - 1], g_3[t - 1], e_3[t - 1],
                                g_4[t - 1], e_4[t - 1], w[t - 1])
            k1_g = g_system(e_1[t - 1], e_2[t - 1], e_3[t - 1], e_4[t - 1])
            k1_param = param_system(x_r[t - 1], z_r[t - 1], e_1[t - 1], e_2[t - 1], e_3[t - 1])

            k2 = drive_system(x[t - 1] + 0.5 * dt * k1[0],
                                 y[t - 1] + 0.5 * dt * k1[1],
                                 z[t - 1] + 0.5 * dt * k1[2],
                                 w[t - 1] + 0.5 * dt * k1[3],
                                 phi[t - 1] + 0.5 * dt * k1[3])

            k2_r = response_system(x_r[t - 1] + 0.5 * dt * k1_r[0],
                                y_r[t - 1] + 0.5 * dt * k1_r[1],
                                z_r[t - 1] + 0.5 * dt * k1_r[2],
                                phi_r[t - 1] + 0.5 * dt * k1_r[3],
                                a_t[t - 1] + 0.5 * dt * k1_param[0],
                                b_t[t - 1] + 0.5 * dt * k1_param[1],
                                d_t[t - 1] + 0.5 * dt * k1_param[2],
                                theta_t[t - 1] + 0.5 * dt * k1_param[3],
                                g_1[t - 1] + 0.5 * dt * k1_g[0],
                                e_1[t - 1] + 0.5 * dt * k1_e[0],
                                g_2[t - 1] + 0.5 * dt * k1_g[1],
                                e_2[t - 1] + 0.5 * dt * k1_e[1],
                                g_3[t - 1] + 0.5 * dt * k1_g[2],
                                e_3[t - 1] + 0.5 * dt * k1_e[2],
                                g_4[t - 1] + 0.5 * dt * k1_g[3],
                                e_4[t - 1] + 0.5 * dt * k1_e[3])

            k2_e = error_system(x_r[t - 1] + 0.5 * dt * k1_r[0],
                                z_r[t - 1] + 0.5 * dt * k1_r[2],
                                phi_r[t - 1] + 0.5 * dt * k1_r[3],
                                a_t[t - 1] + 0.5 * dt * k1_param[0],
                                b_t[t - 1] + 0.5 * dt * k1_param[1],
                                d_t[t - 1] + 0.5 * dt * k1_param[2],
                                theta_t[t - 1] + 0.5 * dt * k1_param[3],
                                g_1[t - 1] + 0.5 * dt * k1_g[0],
                                e_1[t - 1] + 0.5 * dt * k1_e[0],
                                g_2[t - 1] + 0.5 * dt * k1_g[1],
                                e_2[t - 1] + 0.5 * dt * k1_e[1],
                                g_3[t - 1] + 0.5 * dt * k1_g[2],
                                e_3[t - 1] + 0.5 * dt * k1_e[2],
                                g_4[t - 1] + 0.5 * dt * k1_g[3],
                                e_4[t - 1] + 0.5 * dt * k1_e[3],
                                w[t - 1] + 0.5 * dt * k1[3])

            k2_g = g_system(e_1[t - 1] + 0.5 * dt * k1_e[0],
                            e_2[t - 1] + 0.5 * dt * k1_e[1],
                            e_3[t - 1] + 0.5 * dt * k1_e[2],
                            e_3[t - 1] + 0.5 * dt * k1_e[3])

            k2_param = param_system(x_r[t - 1] + 0.5 * dt * k1_r[0],
                                    z_r[t - 1] + 0.5 * dt * k1_r[2],
                                    e_1[t - 1] + 0.5 * dt * k1_e[0],
                                    e_2[t - 1] + 0.5 * dt * k1_e[1],
                                    e_3[t - 1] + 0.5 * dt * k1_e[2])

            k3 = drive_system(x[t - 1] + 0.5 * dt * k2[0],
                                 y[t - 1] + 0.5 * dt * k2[1],
                                 z[t - 1] + 0.5 * dt * k2[2],
                                 w[t - 1] + 0.5 * dt * k2[3],
                                 phi[t - 1] + 0.5 * dt * k2[3])

            k3_r = response_system(x_r[t - 1] + 0.5 * dt * k2_r[0],
                                y_r[t - 1] + 0.5 * dt * k2_r[1],
                                z_r[t - 1] + 0.5 * dt * k2_r[2],
                                phi_r[t - 1] + 0.5 * dt * k2_r[3],
                                a_t[t - 1] + 0.5 * dt * k2_param[0],
                                b_t[t - 1] + 0.5 * dt * k2_param[1],
                                d_t[t - 1] + 0.5 * dt * k2_param[2],
                                theta_t[t - 1] + 0.5 * dt * k2_param[3],
                                g_1[t - 1] + 0.5 * dt * k2_g[0],
                                e_1[t - 1] + 0.5 * dt * k2_e[0],
                                g_2[t - 1] + 0.5 * dt * k2_g[1],
                                e_2[t - 1] + 0.5 * dt * k2_e[1],
                                g_3[t - 1] + 0.5 * dt * k2_g[2],
                                e_3[t - 1] + 0.5 * dt * k2_e[2],
                                g_4[t - 1] + 0.5 * dt * k2_g[3],
                                e_4[t - 1] + 0.5 * dt * k2_e[3])

            k3_e = error_system(x_r[t - 1] + 0.5 * dt * k2_r[0],
                                z_r[t - 1] + 0.5 * dt * k2_r[2],
                                phi_r[t - 1] + 0.5 * dt * k2_r[3],
                                a_t[t - 1] + 0.5 * dt * k2_param[0],
                                b_t[t - 1] + 0.5 * dt * k2_param[1],
                                d_t[t - 1] + 0.5 * dt * k2_param[2],
                                theta_t[t - 1] + 0.5 * dt * k2_param[3],
                                g_1[t - 1] + 0.5 * dt * k2_g[0],
                                e_1[t - 1] + 0.5 * dt * k2_e[0],
                                g_2[t - 1] + 0.5 * dt * k2_g[1],
                                e_2[t - 1] + 0.5 * dt * k2_e[1],
                                g_3[t - 1] + 0.5 * dt * k2_g[2],
                                e_3[t - 1] + 0.5 * dt * k2_e[2],
                                g_4[t - 1] + 0.5 * dt * k2_g[3],
                                e_4[t - 1] + 0.5 * dt * k2_e[3],
                                w[t - 1] + 0.5 * dt * k2[3])

            k3_g = g_system(e_1[t - 1] + 0.5 * dt * k2_e[0],
                            e_2[t - 1] + 0.5 * dt * k2_e[1],
                            e_3[t - 1] + 0.5 * dt * k2_e[2],
                            e_3[t - 1] + 0.5 * dt * k2_e[3])

            k3_param = param_system(x_r[t - 1] + 0.5 * dt * k2_r[0],
                                    z_r[t - 1] + 0.5 * dt * k2_r[2],
                                    e_1[t - 1] + 0.5 * dt * k2_e[0],
                                    e_2[t - 1] + 0.5 * dt * k2_e[1],
                                    e_3[t - 1] + 0.5 * dt * k2_e[2])

            k4 = drive_system(x[t - 1] + dt * k3[0],
                                 y[t - 1] + dt * k3[1],
                                 z[t - 1] + dt * k3[2],
                                 w[t - 1] + dt * k3[3],
                                 phi[t - 1] + dt * k3[3])

            k4_r = response_system(x_r[t - 1] + dt * k3_r[0],
                                y_r[t - 1] + dt * k3_r[1],
                                z_r[t - 1] + dt * k3_r[2],
                                phi_r[t - 1] + dt * k3_r[3],
                                a_t[t - 1] + dt * k3_param[0],
                                b_t[t - 1] + dt * k3_param[1],
                                d_t[t - 1] + dt * k3_param[2],
                                theta_t[t - 1] + dt * k3_param[3],
                                g_1[t - 1] + dt * k3_g[0],
                                e_1[t - 1] + dt * k3_e[0],
                                g_2[t - 1] + dt * k3_g[1],
                                e_2[t - 1] + dt * k3_e[1],
                                g_3[t - 1] + dt * k3_g[2],
                                e_3[t - 1] + dt * k3_e[2],
                                g_4[t - 1] + dt * k3_g[3],
                                e_4[t - 1] + dt * k3_e[3])

            k4_e = error_system(x_r[t - 1] + dt * k3_r[0],
                                z_r[t - 1] + dt * k3_r[2],
                                phi_r[t - 1] + dt * k3_r[3],
                                a_t[t - 1] + dt * k3_param[0],
                                b_t[t - 1] + dt * k3_param[1],
                                d_t[t - 1] + dt * k3_param[2],
                                theta_t[t - 1] + dt * k3_param[3],
                                g_1[t - 1] + dt * k3_g[0],
                                e_1[t - 1] + dt * k3_e[0],
                                g_2[t - 1] + dt * k3_g[1],
                                e_2[t - 1] + dt * k3_e[1],
                                g_3[t - 1] + dt * k3_g[2],
                                e_3[t - 1] + dt * k3_e[2],
                                g_4[t - 1] + dt * k3_g[3],
                                e_4[t - 1] + dt * k3_e[3],
                                w[t - 1] + dt * k3[3])

            k4_g = g_system(e_1[t - 1] + dt * k3_e[0],
                            e_2[t - 1] + dt * k3_e[1],
                            e_3[t - 1] + dt * k3_e[2],
                            e_3[t - 1] + dt * k3_e[3])

            k4_param = param_system(x_r[t - 1] + dt * k3_r[0],
                                    z_r[t - 1] + dt * k3_r[2],
                                    e_1[t - 1] + dt * k3_e[0],
                                    e_2[t - 1] + dt * k3_e[1],
                                    e_3[t - 1] + dt * k3_e[2])


            x[t] = x[t - 1] + ((k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) * dt) / 6.0
            y[t] = y[t - 1] + ((k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) * dt) / 6.0
            z[t] = z[t - 1] + ((k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) * dt) / 6.0
            w[t] = w[t - 1] + ((k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) * dt) / 6.0
            phi[t] = phi[t - 1] + ((k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4]) * dt) / 6.0

            x_r[t] = x_r[t - 1] + ((k1_r[0] + 2 * k2_r[0] + 2 * k3_r[0] + k4_r[0]) * dt) / 6.0
            y_r[t] = y_r[t - 1] + ((k1_r[1] + 2 * k2_r[1] + 2 * k3_r[1] + k4_r[1]) * dt) / 6.0
            z_r[t] = z_r[t - 1] + ((k1_r[2] + 2 * k2_r[2] + 2 * k3_r[2] + k4_r[2]) * dt) / 6.0
            phi_r[t] = phi_r[t - 1] + ((k1_r[3] + 2 * k2_r[3] + 2 * k3_r[3] + k4_r[3]) * dt) / 6.0

            e_1[t] = e_1[t - 1] + ((k1_e[0] + 2 * k2_e[0] + 2 * k3_e[0] + k4_e[0]) * dt) / 6.0
            e_2[t] = e_2[t - 1] + ((k1_e[1] + 2 * k2_e[1] + 2 * k3_e[1] + k4_e[1]) * dt) / 6.0
            e_3[t] = e_3[t - 1] + ((k1_e[2] + 2 * k2_e[2] + 2 * k3_e[2] + k4_e[2]) * dt) / 6.0
            e_4[t] = e_4[t - 1] + ((k1_e[3] + 2 * k2_e[3] + 2 * k3_e[3] + k4_e[3]) * dt) / 6.0

            a_t[t] = a_t[t - 1] + ((k1_param[0] + 2 * k2_param[0] + 2 * k3_param[0] + k4_param[0]) * dt) / 6.0
            b_t[t] = b_t[t - 1] + ((k1_param[1] + 2 * k2_param[1] + 2 * k3_param[1] + k4_param[1]) * dt) / 6.0
            d_t[t] = d_t[t - 1] + ((k1_param[2] + 2 * k2_param[2] + 2 * k3_param[2] + k4_param[2]) * dt) / 6.0
            theta_t[t] = theta_t[t - 1] + ((k1_param[3] + 2 * k2_param[3] + 2 * k3_param[3] + k4_param[3]) * dt) / 6.0

            g_1[t] = g_1[t - 1] + ((k1_g[0] + 2 * k2_g[0] + 2 * k3_g[0] + k4_g[0]) * dt) / 6.0
            g_2[t] = g_2[t - 1] + ((k1_g[1] + 2 * k2_g[1] + 2 * k3_g[1] + k4_g[1]) * dt) / 6.0
            g_3[t] = g_3[t - 1] + ((k1_g[2] + 2 * k2_g[2] + 2 * k3_g[2] + k4_g[2]) * dt) / 6.0
            g_4[t] = g_4[t - 1] + ((k1_g[3] + 2 * k2_g[3] + 2 * k3_g[3] + k4_g[3]) * dt) / 6.0

        return [x, y, z, w, phi], [x_r, y_r, z_r, phi_r], [e_1, e_2, e_3, e_4], [a_t, b_t, d_t, theta_t], [g_1,
                                                                                                                 g_2,
                                                                                                                 g_3,
                                                                                                                 g_4]



    def plot_timeseries_error(self):
        t = np.log(np.arange(int(self.washout+1), int(self.t_max+1), self.dt))
        fig, ax = plt.subplots(2, 2, sharex=True)
        fig.set_size_inches(10.5, 10.5)
        ax = ax.flatten()

        ax[0].plot(t, self.error[0], 'b', label='membrane potential', linewidth=2.0)
        ax[0].set_ylabel('$e_x$', fontsize=26)
        ax[0].set_ylim([-2.5, 2.5])
        ax[0].tick_params(axis='x', labelsize=22)
        ax[0].tick_params(axis='y', labelsize=22)
        ax[0].xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax[0].yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)

        ax[1].plot(t, self.error[1], 'b', label='membrane potential', linewidth=2.0)
        ax[1].set_ylabel('$e_y$', fontsize=26)
        ax[1].set_ylim([-6, 6])
        ax[1].tick_params(axis='x', labelsize=22)
        ax[1].tick_params(axis='y', labelsize=22)
        ax[1].xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax[1].yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)

        ax[2].plot(t, self.error[2], 'b', label='membrane potential', linewidth=2.0)
        ax[2].set_ylabel('$e_z$', fontsize=26)
        ax[2].set_ylim([-2.5, 2.5])
        ax[2].tick_params(axis='x', labelsize=22)
        ax[2].tick_params(axis='y', labelsize=22)
        ax[2].xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax[2].yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax[2].grid(True, which='both', linestyle='--', linewidth=0.5)

        ax[3].plot(t, self.error[3], 'b', label='membrane potential', linewidth=2.0)
        ax[3].set_ylabel('$e_ϕ$', fontsize=26)
        ax[3].set_ylim([-2.5, 2.5])
        ax[3].tick_params(axis='x', labelsize=22)
        ax[3].tick_params(axis='y', labelsize=22)
        ax[3].xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax[3].yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax[3].grid(True, which='both', linestyle='--', linewidth=0.5)

        ax[2].set_xlabel("t", fontsize=26)
        ax[3].set_xlabel("t", fontsize=26)

        fig.align_ylabels(ax[:])
        plt.tight_layout()
        fig.savefig('images/fig_5.eps', dpi=300, format='eps')
        fig.savefig('images/fig_5.png', dpi=300, format='png')
        plt.show()

    def plot_timeseries_uncertainty(self):
        t = np.log(np.arange(int(self.washout + 1), int(self.t_max + 1), self.dt))
        fig, ax = plt.subplots(2, 2, sharex=True)
        fig.set_size_inches(10.5, 10.5)
        ax = ax.flatten()

        ax[0].plot(t, self.uncertainty[0], 'b', linewidth=2.0)
        ax[0].set_ylabel(r'$\bar{a}(t)$', fontsize=26)
        ax[0].set_ylim([-1, 6])
        ax[0].tick_params(axis='x', labelsize=22)
        ax[0].tick_params(axis='y', labelsize=22)
        ax[0].xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax[0].yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)

        ax[1].plot(t, self.uncertainty[1], 'b', linewidth=2.0)
        ax[1].set_ylabel(r'$\bar{b}(t)$', fontsize=26)
        ax[1].set_ylim([-1, 2])
        ax[1].tick_params(axis='x', labelsize=22)
        ax[1].tick_params(axis='y', labelsize=22)
        ax[1].xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax[1].yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)

        ax[2].plot(t, self.uncertainty[2], 'b', linewidth=2.0)
        ax[2].set_ylabel(r'$\bar{d}(t)$', fontsize=26)
        ax[2].set_ylim([-2, 10])
        ax[2].tick_params(axis='x', labelsize=22)
        ax[2].tick_params(axis='y', labelsize=22)
        ax[2].xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax[2].yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax[2].grid(True, which='both', linestyle='--', linewidth=0.5)

        ax[3].plot(t, self.uncertainty[3], 'b', linewidth=2.0)
        ax[3].set_ylabel(r'$\bar{θ}(t)$', fontsize=26)
        ax[3].set_ylim([-2, 2])
        ax[3].tick_params(axis='x', labelsize=22)
        ax[3].tick_params(axis='y', labelsize=22)
        ax[3].xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax[3].yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax[3].grid(True, which='both', linestyle='--', linewidth=0.5)

        ax[2].set_xlabel("t", fontsize=26)
        ax[3].set_xlabel("t", fontsize=26)
        fig.align_ylabels(ax[:])
        plt.tight_layout()
        fig.savefig('images/fig_7.png', dpi=300, format='png')
        fig.savefig('images/fig_7.eps', dpi=300, format='eps')
        plt.show()



    def plot_timeseries_g(self):
        t = np.log(np.arange(int(self.washout+1), int(self.t_max+1), self.dt))
        fig, ax = plt.subplots(2, 2, sharex=True)
        fig.set_size_inches(10.5, 10.5)
        ax = ax.flatten()

        ax[0].plot(t, self.controlers[0], 'b', label='membrane potential', linewidth=2.0)
        ax[0].set_ylabel('$g_x$', fontsize=26)
        ax[0].set_ylim([0, 4])
        ax[0].tick_params(axis='x', labelsize=22)
        ax[0].tick_params(axis='y', labelsize=22)
        ax[0].xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax[0].yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)

        ax[1].plot(t, self.controlers[1], 'b', label='membrane potential', linewidth=2.0)
        ax[1].set_ylabel('$g_y$', fontsize=26)
        ax[1].set_ylim([0, 15])
        ax[1].tick_params(axis='x', labelsize=22)
        ax[1].tick_params(axis='y', labelsize=22)
        ax[1].xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax[1].yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)

        ax[2].plot(t, self.controlers[2], 'b', label='membrane potential', linewidth=2.0)
        ax[2].set_ylabel('$g_z$', fontsize=26)
        ax[2].set_ylim([0, 4])
        ax[2].tick_params(axis='x', labelsize=22)
        ax[2].tick_params(axis='y', labelsize=22)
        ax[2].xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax[2].yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax[2].grid(True, which='both', linestyle='--', linewidth=0.5)

        ax[3].plot(t, self.controlers[3], 'b', label='membrane potential', linewidth=2.0)
        ax[3].set_ylabel('$g_ϕ$', fontsize=26)
        ax[3].set_ylim([0, 4])
        ax[3].tick_params(axis='x', labelsize=22)
        ax[3].tick_params(axis='y', labelsize=22)
        ax[3].xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax[3].yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax[3].grid(True, which='both', linestyle='--', linewidth=0.5)

        ax[2].set_xlabel("t", fontsize=26)
        ax[3].set_xlabel("t", fontsize=26)

        fig.align_ylabels(ax[:])

        # fig.text(0.49, 0.95, '(b)', fontsize=30, weight='bold')

        plt.tight_layout()

        fig.savefig('images/fig_6.eps', dpi=300, format='eps')
        fig.savefig('images/fig_6.png', dpi=300, format='png')

        plt.show()

