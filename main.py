from module.dynamical_systems.drive_functions import *
from module.dynamical_systems.response_functions import *
from module.dynamical_systems.error_system_runge import *

from module.machine_learning.esn import visualise_esn
from module.machine_learning.esn_observer import visualise_ro
from module.machine_learning.online_control import visualise_online_controller
from module.machine_learning.predictive_control import visualise_pred_control

# figure 1
plot_timeseries(k_1=2.3, k_2=0.5, label="a")
plot_timeseries(k_1=0.1, k_2=0.1, label="b")
plot_timeseries(k_1=5, k_2=1.5, label="c")
plot_timeseries(k_1=0.08, k_2=0.4, label="d")
plot_timeseries(k_1=2.5, k_2=0.5, label="e", y_lim="set")

# fig 2 # Simulations which generate and store the necessary data
d_isi_s = drive_inter_param(t_max=4000, dt=0.01, washout=2000, param_min=3, param_max=5, step=0.01, param="s",
                            compute=True)
d_isi_sigma = drive_inter_param(t_max=4000, dt=0.01, washout=2000, param_min=0, param_max=0.125, step=0.0001,
                                param="sigma",
                                compute=True)
d_isi_k_1 = drive_inter_param(t_max=4000, dt=0.01, washout=2000, param_min=0, param_max=5, step=0.01, param="k_1",
                              compute=True)
d_isi_k_2 = drive_inter_param(t_max=4000, dt=0.01, washout=2000, param_min=0, param_max=2, step=0.01, param="k_2",
                              compute=True)

# Fig 2  Plot Images from the csv/txt files
drive_chaos_s_plot()
drive_chaos_sigma_plot()
drive_chaos_k_1_plot()
drive_chaos_k_2_plot()

# Fig 3
bifurcation()

# Fig 4
drive_3d_plot(k_1=0.08, k_2=0.4)
response_3d_plot(k_1=0.08, k_2=0.4)

# Fig 5
ED = Error_dynamics()
ED.simulate(23030, 0.001, 0)
ED.plot_timeseries_error()
ED.plot_timeseries_uncertainty()
ED.plot_timeseries_g()

# Fig 6
Lya_k_1_k_2()

# Fig 9
visualise_esn()

# Fig 11
visualise_ro()

# Fig 13
visualise_online_controller()

# Fig 14 & 16
visualise_pred_control()
