from module.machine_learning.echo_state_network_function import EchoStateObserver, EchoStateNetwork
from module.machine_learning.echo_state_network_support_functions import plot_pred, plot_train_pred_controlled, \
    plot_errors, rmse, plot_transition_controlled, plot_transition_errors
import numpy as np
from module.systems.drive_runge import Drive_HindmarshRose
from module.systems.response_runge import Response_HindmarshRose
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
from skopt import gp_minimize
from skopt.space import Real


def visualise_pred_control():
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

    ## Can be uncommented to run HPO
    # D_tran_X_train = drive_data[0, :train].reshape(1, -1)
    # D_tran_Y_target = drive_data[0, 1:train + 1].reshape(1, -1)
    #
    # D_X_eval_observed = drive_data[0, train:train + pred]
    # D_X_eval_target = drive_data[0, train + 1:train + pred + 1]
    #
    # def x_pred_drive(param):
    #     drive_x_pred_esn = EchoStateNetwork()
    #     drive_x_pred_esn.create_network(input_dim=1, neurons=int(param[2]),
    #                                     rewiring_prob=param[3],
    #                                     w_res_spectral=param[0],
    #                                     alpha=param[1], seed=42)
    #     drive_pred_train_output = drive_x_pred_esn.train(input_data=D_tran_X_train, target_data=D_tran_Y_target,
    #                                                      transient=transient, normalised=param[4])
    #     rmse(target=D_tran_Y_target.reshape(1, -1), predicted=drive_pred_train_output)
    #
    #     drive_pred_pred_output = drive_x_pred_esn.one_step_pred(pred_time=pred,
    #                                                             input_signal=drive_data[0, train:train + pred].reshape(
    #                                                                 1,
    #                                                                 -1))
    #     return rmse(target=D_X_eval_target.reshape(1, -1), predicted=drive_pred_pred_output)
    #
    # search_space = list()
    # search_space.append(Real(0.1, 2, name='w_res_spectral'))
    # search_space.append(Real(0, 1, name='alpha'))
    # search_space.append(Real(50, 1000, name="size"))
    # search_space.append(Real(0.05, 1, name="rewiringprob"))
    # search_space.append(Real(1e-09, 1e-05, name="normalised"))
    #
    # results_drive_transition = gp_minimize(func=x_pred_drive, dimensions=search_space,
    #                                        n_calls=300, noise=1e-10)
    #
    # print('drive_transition: Lowest MSE: %.3f' % (results_drive_transition.fun))
    # print('drive_transition: Best Parameters: %s' % (results_drive_transition.x))

    drive_hyp = [0.44831637275430014, 0.09455267650625794, 246.54699776768345, 0.4251795804696316,
                 3.2301031248050223e-06]
    response_hyp = [1.8373420048933036, 0.33720229714680794, 142.6927804890221, 0.7260192492270072,
                    3.9481620022095794e-06]
    results_drive_transition = [1.9463897602408042, 0.9571325051415408, 971.8401323019777, 0.9877476753158976,
                                8.953940395059738e-06]

    # Drive--------------------------------------------------------------------------------------------------------------
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
    #           train_time=train - 1, pred_time=pred, model="esn_control_drive_infer")

    # response --------------------------------------------------------------------------------------------------------------
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
    rmse(target=response_data[:, train:train + pred], predicted=response_pred_output)  # 0.00032
    # plot_pred(data_test=response_subset[:, 1:train + pred], data_train=response_train_output, data_pred=response_pred_output,
    #           train_time=train - 1, pred_time=pred, model="esn_control_response_infer")

    # x_pred_drive --------------------------------------------------------------------------------------------------
    D_tran_X_train = drive_data[0, :train].reshape(1, -1)
    D_X_eval_observed = drive_data[0, train:train + pred]

    D_tran_Y_target = drive_data[0, 1:train + 1].reshape(1, -1)
    D_X_eval_target = drive_data[0, train + 1:train + pred + 1]

    drive_x_pred_esn = EchoStateNetwork()
    drive_x_pred_esn.create_network(input_dim=1, neurons=int(results_drive_transition[2]),
                                    rewiring_prob=results_drive_transition[3],
                                    w_res_spectral=results_drive_transition[0],
                                    alpha=results_drive_transition[1], seed=42)

    drive_pred_train_output = drive_x_pred_esn.train(input_data=D_tran_X_train, target_data=D_tran_Y_target,
                                                     transient=transient, normalised=results_drive_transition[4])
    rmse(target=D_tran_Y_target.reshape(1, -1), predicted=drive_pred_train_output)

    drive_pred_pred_output = drive_x_pred_esn.one_step_pred(pred_time=pred,
                                                            input_signal=D_X_eval_observed.reshape(1,
                                                                                                   -1))
    rmse(target=D_X_eval_target.reshape(1, -1), predicted=drive_pred_pred_output)
    # plot_pred(data_test=drive_subset[0, 1:train + pred + 1].reshape(1, -1), data_train=drive_pred_train_output,
    #           data_pred=drive_pred_pred_output,
    #           train_time=train, pred_time=pred, model="esn_control_x_pred")

    # control
    drive_pred = np.zeros((controlled + 1, 1))
    drive_controlled = np.zeros((controlled + 1, 5))
    transition_drive = np.zeros((controlled + 1, 4))
    response_controlled = np.zeros((controlled + 1, 4))
    response_controlled[0] = response_pred_output[:, -1]
    transition_response = np.zeros((controlled + 1, 3))
    error_between = np.zeros((controlled + 1, 4))
    controller = np.zeros((controlled + 1, 4))
    zeros = np.zeros((controlled + 1, 4))
    drive_observed = drive_data[0, train + pred:train + pred + controlled]
    # controlled=1000
    for x in range(controlled):
        # 1 step
        drive_controlled[x + 1] = drive_esn.inference_one(observed_data=drive_observed[x])
        response_in = np.insert(response_controlled[x, 1:], 0, drive_observed[x])
        response_controlled[x + 1] = response_esn.control_one(input_data=response_in, observed_data=drive_observed[x],
                                                              control_input=controller[x, :],
                                                              error=error_between[x, 1:],
                                                              learning_rate=0.001, online=True)

        x_out_drive_projected = np.delete(drive_controlled[x + 1], 3).reshape(-1, 1).T
        error_between[x + 1] = response_controlled[x + 1] - x_out_drive_projected

        drive_pred[x + 1] = drive_x_pred_esn.one_step_pred(input_signal=drive_observed[x].reshape(1, -1), pred_time=1)

        # Predict drive
        transition_drive[x + 1] = drive_esn.predict_one(input_data=drive_controlled[x + 1, 1:],
                                                        observed_data=drive_pred[x + 1], online=False)

        drive_transition_projected = np.delete(transition_drive[x + 1], 2).reshape(-1, 1).T
        drive_projected = np.delete(drive_controlled[x + 1, 1:], 2).flatten()

        def bound_array(arr):
            return [[val - 0.5, val + 0.5] for val in arr]

        bounds = bound_array(drive_projected)

        def loss_function(input):
            result = response_esn.predict_one(input_data=input, observed_data=drive_pred[x + 1],
                                              error=error_between[x + 1],
                                              learning_rate=0.001, online=True)
            return np.sum(np.square(result - drive_transition_projected))

        # Find predict best controllers
        result = minimize(loss_function, drive_projected, bounds=bounds, method="L-BFGS-B")

        transition_response[x + 1] = response_esn.predict_one(input_data=result.x, observed_data=drive_pred[x + 1],
                                                              error=error_between[x + 1], learning_rate=0.001,
                                                              online=True)

        controller_with_prev = np.insert(result.x, 0, drive_observed[x])
        controller[x + 1] = response_controlled[x + 1] - controller_with_prev

    controller = controller[1:, :]
    drive_pred = drive_pred[1:]
    error_between = error_between[1:, :]
    transition_response = transition_response[1:, :]
    transition_drive = transition_drive[1:, :]
    response_controlled = response_controlled[1:, :]
    drive_controlled = drive_controlled[1:, :]
    drive_projected = np.delete(drive_controlled, 3, 1)
    drive_subset_projected = np.delete(drive_data, 3, 0)
    transition_drive_projected = np.delete(transition_drive, 2, 1)
    drive_all = np.hstack((drive_train_output, drive_pred_output, drive_controlled.T))
    drive_all_projected = np.delete(drive_all, 3, 0)

    mean_absolute_error(response_controlled[1:], drive_projected[1:])  # 0.00235
    rmse(predicted=drive_projected[1:], target=response_controlled[1:])  # 0.0235

    # plot_transition_controlled(pred_system=np.hstack((drive_pred[:-1], transition_drive[:-1])).T,
    #                            true_system=drive_controlled[1:].T, start_time=train + pred - 1,
    #                            control_time=controlled - 1,
    #                            model="fig_14")
    plot_transition_errors(data_1=np.hstack((drive_pred[:-1], transition_drive[:-1])).T, data_2=drive_controlled[1:].T,
                           train_time=train, pred_time=pred,
                           control_time=controlled - 1, model="fig_16")

    plot_train_pred_controlled(response=np.hstack((response_train_output, response_pred_output, response_controlled.T)),
                               drive=drive_all_projected, train_time=train - 1, pred_time=pred, control_time=controlled,
                               model="fig_18_a", label="(a)")
    plot_errors(response=response_controlled.T, drive=drive_projected.T, train_time=train - 1, pred_time=pred,
                control_time=controlled, model="fig_18_b", label="(b)")
