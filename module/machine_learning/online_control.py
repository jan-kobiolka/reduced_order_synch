from module.machine_learning.echo_state_network_function import EchoStateObserver
from module.machine_learning.echo_state_network_support_functions import plot_pred, plot_train_pred_controlled, \
    plot_errors, rmse
import numpy as np
from matplotlib import pyplot as plt
import pickle
import seaborn as sns
from skopt import gp_minimize
from skopt.space import Real
from sklearn.metrics import mean_absolute_error
from module.systems.drive_runge import Drive_HindmarshRose
from module.systems.response_runge import Response_HindmarshRose


def visualise_online_controller():
    drive = Drive_HindmarshRose()
    drive_data = drive.simulate(t_max=22000, dt=0.1, washout=20000, s=4.75, k_1=0.21, k_2=0.4)
    # drive.plot_timeseries()
    drive_data = np.array(drive_data)[:, ::2]

    response = Response_HindmarshRose()
    response_data = response.simulate(t_max=22000, dt=0.1, washout=20000, s=4.75, k_1=0.21, k_2=0.4)  # Bursts
    # response.plot_timeseries()
    response_data = np.array(response_data)[:, ::2]

    transient = 300
    train = 3001
    pred = 3000
    controlled = 3000

    drive_hyp = [0.44831637275430014, 0.09455267650625794, 246.54699776768345, 0.4251795804696316,
                 3.2301031248050223e-06]
    response_hyp = [1.8373420048933036, 0.33720229714680794, 142.6927804890221, 0.7260192492270072,
                    3.9481620022095794e-06]

    # Drive---------------------------------------------------------------------------------------------------------
    D_X_train = drive_data[1:, :train - 1]
    D_X_train_observed = drive_data[0, 1:train]
    D_Y_target = drive_data[1:, 1:train]
    D_X_eval_observed = drive_data[0, train:train + pred]

    drive_esn = EchoStateObserver()
    drive_esn.create_network(input_dim=5, neurons=int(drive_hyp[2]), rewiring_prob=drive_hyp[3],
                             w_res_spectral=drive_hyp[0], alpha=drive_hyp[1], seed=42)
    drive_train_output = drive_esn.train(input_data=D_X_train, observed_data=D_X_train_observed, target_data=D_Y_target,
                                         transient=transient, normalised=drive_hyp[4])
    rmse(target=drive_data[:, 1:train], predicted=drive_train_output)
    drive_pred_output = drive_esn.inference_all(observed_data=D_X_eval_observed)
    rmse(target=drive_data[:, train:train + pred], predicted=drive_pred_output)  # 0.0003
    # plot_pred(data_test=drive_subset[:, 1:train + pred], data_train=drive_train_output, data_pred=drive_pred_output,
    #          train_time=train-1, pred_time=pred, model="esn_observer_drive_pred")

    # Response --------------------------------------------------------------------------------------------------------
    R_X_train = response_data[1:, :train - 1]
    R_X_train_observed = response_data[0, 1:train]
    R_Y_target = response_data[1:, 1:train]
    R_X_eval_observed = response_data[0, train:train + pred]

    response_esn = EchoStateObserver()
    response_esn.create_network(input_dim=4, neurons=int(response_hyp[2]), rewiring_prob=response_hyp[3],
                                w_res_spectral=response_hyp[0],
                                alpha=response_hyp[1], seed=42)
    response_train_output = response_esn.train(input_data=R_X_train, observed_data=R_X_train_observed,
                                               target_data=R_Y_target,
                                               transient=transient, normalised=response_hyp[4])
    rmse(target=response_data[:, 1:train], predicted=response_train_output)
    response_pred_output = response_esn.inference_all(observed_data=R_X_eval_observed)
    rmse(target=response_data[:, train:train + pred], predicted=response_pred_output)
    # plot_pred(data_test=response_subset[:, :train + pred], data_train=response_train_output, data_pred=response_pred_output,
    #           train_time=train, pred_time=pred, model="esn_observer_online_response_pred")

    error_between = np.zeros((controlled + 1, 4))
    response_controlled = np.zeros((controlled + 1, 4))
    drive_controlled = np.zeros((controlled + 1, 5))
    response_controlled[0] = response_pred_output[:, -1]
    drive_observed = drive_data[0, train + pred:train + pred + controlled]
    for x in range(controlled):
        response_input = np.insert(response_controlled[x, 1:], 0, drive_observed[x])
        response_controlled[x + 1] = response_esn.control_one(input_data=response_input,
                                                              observed_data=drive_observed[x],
                                                              control_input=error_between[x, :],
                                                              error=error_between[x, 1:],
                                                              learning_rate=0.001, online=True)
        drive_controlled[x + 1] = drive_esn.inference_one(observed_data=drive_observed[x])
        x_out_drive_projected = np.delete(drive_controlled[x + 1], 3).reshape(-1, 1).T
        error_between[x + 1] = response_controlled[x + 1] - x_out_drive_projected

    error_between = error_between[1:, :]
    response_controlled = response_controlled[1:, :]
    drive_controlled = drive_controlled[1:, :]
    drive_projected = np.delete(drive_controlled, 3, 1)
    drive_true_projected = np.delete(drive_data, 3, 0)
    drive_all = np.hstack((drive_train_output, drive_pred_output, drive_controlled.T))
    drive_all_projected = np.delete(drive_all, 3, 0)

    # Both systems
    mean_absolute_error(response_controlled[1:], drive_projected[1:])  # 0.00870
    rmse(predicted=drive_projected[1:], target=response_controlled[1:])  # 0.0370

    plot_train_pred_controlled(response=np.hstack((response_train_output, response_pred_output, response_controlled.T)),
                               drive=drive_all_projected, train_time=train - 1, pred_time=pred, control_time=controlled,
                               model="fig_15_a", label="(a)")
    plot_errors(response=response_controlled.T, drive=drive_projected.T, train_time=train - 1, pred_time=pred,
                control_time=controlled, model="fig_15_b", label="(b)")
