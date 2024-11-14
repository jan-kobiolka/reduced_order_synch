from module.machine_learning.echo_state_network_function import EchoStateNetwork
import numpy as np
from module.systems.drive_runge import Drive_HindmarshRose
from module.systems.response_runge import Response_HindmarshRose
from module.machine_learning.echo_state_network_support_functions import plot_pred, plot_pred_errors, rmse
from matplotlib import pyplot as plt
import pickle
import seaborn as sns
from skopt import gp_minimize
from skopt.space import Real
from sklearn.metrics import mean_absolute_error

def visualise_esn():
    drive = Drive_HindmarshRose()
    drive_data = drive.simulate(t_max=22000, dt=0.1, washout=20000, s=4.75, k_1=0.21, k_2=0.4)
    drive_data = np.array(drive_data)[:, ::2]

    response = Response_HindmarshRose()
    response_data = response.simulate(t_max=22000, dt=0.1, washout=20000, s=4.75, k_1=0.21, k_2=0.4)  # Bursts
    response_data = np.array(response_data)[:, ::2]

    transient = 300
    train = 3000
    pred = 2000

    ## Can be uncommented to run HPO
    # def model_evaluate_drive(param):
    #     esn = EchoStateNetwork()
    #     esn.create_network(input_dim=5, neurons=int(param[2]), rewiring_prob=param[3], w_res_spectral=param[0],
    #                        alpha=param[1],
    #                        seed=42)
    #     esn.train(input_data=drive_data[:, :train], target_data=drive_data[:, 1:train + 1], transient=transient,
    #               normalised=param[4])
    #     drive_pred_output = esn.predict_all(pred_time=pred)
    #     return rmse(target=drive_data[:, train + 1:train + pred + 1], predicted=drive_pred_output)
    #
    #
    # def model_evaluate_response(param):
    #     esn = EchoStateNetwork()
    #     esn.create_network(input_dim=4, neurons=int(param[2]), rewiring_prob=param[3], w_res_spectral=param[0],
    #                        alpha=param[1],
    #                        seed=42)
    #     esn.train(input_data=response_data[:, :train], target_data=response_data[:, 1:train + 1], transient=transient,
    #               normalised=param[4])
    #     response_pred_output = esn.predict_all(pred_time=pred)
    #     return rmse(target=response_data[:, train + 1:train + pred + 1], predicted=response_pred_output)
    #
    #
    # search_space = list()
    # search_space.append(Real(0.5, 1.5, name='w_res_spectral'))
    # search_space.append(Real(0, 1, name='alpha'))
    # search_space.append(Real(100, 1000, name="size"))
    # search_space.append(Real(0.05, 1, name="rewiringprob"))
    # search_space.append(Real(0.0000001, 0.00001, name="normalised"))
    #
    # result_drive = gp_minimize(func=model_evaluate_drive, dimensions=search_space,n_calls=300,noise=1e-10)
    # result_response = gp_minimize(func=model_evaluate_response, dimensions=search_space,n_calls=300,noise=1e-10)
    # #
    # print('drive: Lowest MSE: %.3f' % (result_drive.fun))
    # print('drive: Best Parameters: %s' % (result_drive.x))
    # #
    # print('Response: Lowest MSE: %.3f' % (result_response.fun))
    # print('Response: Best Parameters: %s' % (result_response.x))

    pred = 3000

    drive_hyp = [0.8812920448130991, 0.06917834090553708, 314.52163602781684, 0.8417238367758675,
                 1.4383663603497001e-06]
    response_hyp = [0.9628587549203287, 0.5361144184298384, 386.5310180224124, 0.3977705580458187,
                    5.205269994548834e-06]

    # Drive
    esn_drive = EchoStateNetwork()
    esn_drive.create_network(input_dim=5, neurons=int(drive_hyp[2]), rewiring_prob=drive_hyp[3],
                             w_res_spectral=drive_hyp[0], alpha=drive_hyp[1],
                             seed=42)
    drive_train_output = esn_drive.train(input_data=drive_data[:, :train], target_data=drive_data[:, 1:train + 1],
                                         transient=transient,
                                         normalised=drive_hyp[4])
    rmse(target=drive_data[:, 1:train + 1], predicted=drive_train_output)
    drive_pred_output = esn_drive.predict_all(pred_time=pred)
    rmse(target=drive_data[:, train + 1:train + pred + 1], predicted=drive_pred_output)

    # Response
    esn_response = EchoStateNetwork()
    esn_response.create_network(input_dim=4, neurons=int(response_hyp[2]), rewiring_prob=response_hyp[3],
                                w_res_spectral=response_hyp[0],
                                alpha=response_hyp[1],
                                seed=42)
    response_train_output = esn_response.train(input_data=response_data[:, :train],
                                               target_data=response_data[:, 1:train + 1],
                                               transient=transient, normalised=response_hyp[4])
    rmse(target=response_data[:, 1:train + 1], predicted=response_train_output)
    response_pred_output = esn_response.predict_all(pred_time=pred)
    rmse(target=response_data[:, train + 1:train + pred + 1], predicted=response_pred_output)

    plot_pred(data_test=drive_data[:, 1:train + pred + 1], data_train=drive_train_output, data_pred=drive_pred_output
              , train_time=train, pred_time=pred, model="fig_11_a", label="(a)")
    plot_pred_errors(data_test=drive_data[:, 1:train + pred + 1], data_train=drive_train_output,
                     data_pred=drive_pred_output
                     , train_time=train, pred_time=pred, model="fig_11_b", label="(b)")

    # plot_pred(data_test=response_data[:, 1:train + pred + 1], data_train=response_train_output,
    #           data_pred=response_pred_output,
    #           train_time=train, pred_time=pred, model="fig_9/esn_response_pred", label="(a)")
    # plot_pred_errors(data_test=response_data[:, 1:train + pred + 1], data_train=response_train_output,
    #                  data_pred=response_pred_output,
    #                  train_time=train, pred_time=pred, model="fig_9/esn_response_errors", label="(b)")
