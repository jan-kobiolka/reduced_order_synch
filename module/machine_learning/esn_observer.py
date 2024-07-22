from module.machine_learning.echo_state_network_function import EchoStateObserver
from module.machine_learning.echo_state_network_support_functions import plot_pred, plot_train_pred_controlled, \
    plot_errors, rmse, plot_pred_errors
import numpy as np
from matplotlib import pyplot as plt
import pickle
import seaborn as sns
from skopt import gp_minimize
from skopt.space import Real
from sklearn.metrics import mean_absolute_error
from module.systems.drive_runge import Drive_HindmarshRose
from module.systems.response_runge import Response_HindmarshRose


def visualise_ro():
    drive = Drive_HindmarshRose()
    drive_data = drive.simulate(t_max=22000, dt=0.1, washout=20000, s=4.75, k_1=0.21, k_2=0.4)
    drive_data = np.array(drive_data)[:, ::2]

    response = Response_HindmarshRose()
    response_data = response.simulate(t_max=22000, dt=0.1, washout=20000, s=4.75, k_1=0.21, k_2=0.4)  # Bursts
    response_data = np.array(response_data)[:, ::2]

    transient = 300
    train = 3001
    pred = 2000

    ## Can be uncommented to run HPO
    # D_X_train = drive_data[1:, :train - 1]
    # D_X_train_observed = drive_data[0, 1:train]
    # D_Y_target = drive_data[1:, 1:train]
    # D_X_eval_observed = drive_data[0, train:train + pred]
    #
    # R_X_train = response_data[1:, :train - 1]
    # R_X_train_observed = response_data[0, 1:train]
    # R_Y_target = response_data[1:, 1:train]
    # R_X_eval_observed = response_data[0, train:train + pred]
    #
    # def model_evaluate_drive(param):
    #     drive_esn = EchoStateObserver()
    #     drive_esn.create_network(input_dim=5, neurons=int(param[2]), rewiring_prob=param[3], w_res_spectral=param[0],
    #                               alpha=param[1], seed=42)
    #     drive_train_output = drive_esn.train(input_data=D_X_train, observed_data=D_X_train_observed,
    #                                            target_data=D_Y_target,
    #                                            transient=transient, normalised=param[4])
    #     rmse(target=drive_data[:, 1:train], predicted=drive_train_output)
    #     drive_pred_output = drive_esn.inference_all(observed_data=D_X_eval_observed)
    #     return rmse(target=drive_data[:, train:train + pred], predicted=drive_pred_output)
    #
    #
    # def model_evaluate_slave(param):
    #     slave_esn = EchoStateObserver()
    #     slave_esn.create_network(input_dim=4, neurons=int(param[2]), rewiring_prob=param[3], w_res_spectral=param[0],
    #                              alpha=param[1], seed=42)
    #     slave_train_output = slave_esn.train(input_data=R_X_train, observed_data=R_X_train_observed, target_data=R_Y_target,
    #                                          transient=transient, normalised=param[4])
    #     rmse(target=response_data[:, 1:train], predicted=slave_train_output)
    #     slave_pred_output = slave_esn.inference_all(observed_data=R_X_eval_observed)
    #     return rmse(target=response_data[:, train:train + pred], predicted=slave_pred_output)
    #
    #
    # search_space = list()
    # search_space.append(Real(0.1, 2, name='w_res_spectral'))
    # search_space.append(Real(0, 1, name='alpha'))
    # search_space.append(Real(100, 1000, name="size"))
    # search_space.append(Real(0.05, 1, name="rewiring_prob"))
    # search_space.append(Real(0.0000001, 0.00001, name="normalised"))
    #
    # result_drive = gp_minimize(func=model_evaluate_drive, dimensions=search_space,n_calls=300,noise=1e-10)
    # result_slave = gp_minimize(func=model_evaluate_slave, dimensions=search_space,n_calls=300,noise=1e-10)
    # #
    # print('Drive: Lowest MSE: %.3f' % (result_drive.fun))
    # print('Drive:Best Parameters: %s' % (result_drive.x))
    # #
    # print('Slave:Lowest MSE: %.3f' % (result_slave.fun))
    # print('Slave: Best Parameters: %s' % (result_slave.x))

    drive_hyp = [0.44831637275430014, 0.09455267650625794, 246.54699776768345, 0.4251795804696316,
                 3.2301031248050223e-06]
    response_hyp = [1.8373420048933036, 0.33720229714680794, 142.6927804890221, 0.7260192492270072,
                    3.9481620022095794e-06]

    pred = 3000
    D_X_train = drive_data[1:, :train - 1]
    D_X_train_observed = drive_data[0, 1:train]
    D_Y_target = drive_data[1:, 1:train]
    D_X_eval_observed = drive_data[0, train:train + pred]

    R_X_train = response_data[1:, :train - 1]
    R_X_train_observed = response_data[0, 1:train]
    R_Y_target = response_data[1:, 1:train]
    R_X_eval_observed = response_data[0, train:train + pred]

    # Drive ---------------------------------------------------------------------------------------------------------
    drive_esn = EchoStateObserver()
    drive_esn.create_network(input_dim=5, neurons=int(drive_hyp[2]), rewiring_prob=drive_hyp[3],
                             w_res_spectral=drive_hyp[0], alpha=drive_hyp[1], seed=42)
    drive_train_output = drive_esn.train(input_data=D_X_train, observed_data=D_X_train_observed, target_data=D_Y_target,
                                         transient=transient, normalised=drive_hyp[4])
    rmse(target=drive_data[:, 1:train], predicted=drive_train_output)
    drive_pred_output = drive_esn.inference_all(observed_data=D_X_eval_observed)
    rmse(target=drive_data[:, train:train + pred], predicted=drive_pred_output)  # 0.0003
    # plot_pred(data_test=drive_data[:, 1:train + pred], data_train=drive_train_output, data_pred=drive_pred_output,
    #           train_time=train-1, pred_time=pred, model="master")

    # Response --------------------------------------------------------------------------------------------------------
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

    plot_pred(data_test=drive_data[:, 1:train + pred], data_train=drive_train_output, data_pred=drive_pred_output,
              train_time=train - 1, pred_time=pred, model="fig_11_a", label="(a)")
    plot_pred_errors(data_test=drive_data[:, 1:train + pred], data_train=drive_train_output,
                     data_pred=drive_pred_output
                     , train_time=train - 1, pred_time=pred, model="fig_11_b", label="(b)")

    plot_pred(data_test=response_data[:, 1:train + pred], data_train=response_train_output,
              data_pred=response_pred_output,
              train_time=train - 1, pred_time=pred, model="fig_11_c", label="(c)")
    plot_pred_errors(data_test=response_data[:, 1:train + pred], data_train=response_train_output,
                     data_pred=response_pred_output
                     , train_time=train - 1, pred_time=pred, model="fig_11_d", label="(d)")
