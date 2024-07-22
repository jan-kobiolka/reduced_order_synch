import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from numpy.random import RandomState

class EchoStateNetwork:
    def __init__(self):
        self.b = 0  # todo If we keep b as 0 we can remove it

        self.alpha = None
        self.r_t = None
        self.w_res = None
        self.w_in = None
        self.w_out = None
        self.r_t_warmup = None
        self.r_t_pred = None
        self.pred_time = 0

    def create_network(self, input_dim, neurons, rewiring_prob, w_res_spectral, alpha, seed):
        """
        Creates our network and initialises the W_in and W_res matrix
        :param input_dim: Defines the number of inputs
        :param neurons: Size of reservoir
        :param k: Amount of connections to nearest neighbors
        :param rewiring_prob:  rewiring probability
        :param w_res_spectral: Largest
        :param alpha:
        :return:
        """
        np.random.seed(seed)
        self.alpha = alpha
        w_res_upper_tri = np.triu(
            nx.to_numpy_array(nx.gnp_random_graph(neurons, rewiring_prob, seed=seed)))  # no self loops
        w_res_weights = np.random.uniform(low=-1, high=1, size=(w_res_upper_tri.shape))  # Link weights
        w_res = w_res_upper_tri * w_res_weights
        w_res = w_res + w_res.T
        eigenvalues = np.max(np.abs(np.linalg.eigvals(w_res)))
        self.w_res = w_res / (eigenvalues / w_res_spectral)  # Noralises by spectral radius
        self.w_in = np.random.uniform(low=-1, high=1, size=(neurons, input_dim))

    def train(self, input_data, target_data, transient, normalised =1e-07) -> np.array:
        """
        Takes an array and updates r_t (reservoir state), computes w_out and transforms the data back to its
        original timescale and returns our training time series.
        :param u_t:
        :return:
        """
        n = input_data.shape[1] +1
        r_t = np.zeros((n , self.w_res.shape[0]))
        for i in range(1, n):
            r_t[i] = (1 - self.alpha) * r_t[i - 1, :] + (
                    self.alpha * np.tanh((self.w_res @ r_t[i - 1, :]) + self.w_in @ input_data[:, i - 1] + self.b))
        r_t = r_t[1:]
        self.r_t_warmup = r_t
        r_t_opt = r_t[transient:, :]
        target_data = target_data[:, transient:]
        self.w_out = (target_data @ r_t_opt) @ np.linalg.inv(r_t_opt.T @ r_t_opt + (normalised * np.identity(len(self.w_res))))
        x_out = self.w_out @ r_t.T
        return x_out

    def predict_all(self, pred_time) -> np.array:
        """
        Takes the timeseries and t_warmup and then updates r_t until it reaches the end of t_warmup. Then, it starts
        predicting until the end of the timeseries is reached. We then transform the data back and return it.
        :param u_t: timeseries
        :param warmup: t_warmup
        :return:
        """
        self.pred_time = pred_time
        r_t_predict = np.zeros((pred_time + 1, len(self.w_res)))
        r_t_predict[0] = self.r_t_warmup[-1]  # Copy the state from r_t_warmup
        for i in range(1, pred_time + 1):
            x_out = self.w_out @ r_t_predict[i - 1, :]
            r_t_predict[i] = (1 - self.alpha) * r_t_predict[i - 1, :] + (
                    self.alpha * np.tanh((self.w_res @ r_t_predict[i - 1, :]) + self.w_in @ x_out + self.b))
        x_out = self.w_out @ r_t_predict[1:].T
        return x_out



    def one_step_pred(self, pred_time, input_signal):
        self.pred_time = pred_time
        r_t_real = np.zeros((self.pred_time + 1, len(self.w_res)))
        r_t_real[0] = self.r_t_warmup[-1]
        for i in range(1, pred_time + 1):
            r_t_real[i] = (1 - self.alpha) * r_t_real[i - 1, :] + (
                    self.alpha * np.tanh((self.w_res @ r_t_real[i - 1, :]) + self.w_in @ input_signal[:, i - 1] + self.b))
        self.r_t_warmup = np.vstack([self.r_t_warmup, r_t_real[1:]])
        return self.w_out @ r_t_real[1:].T

class EchoStateObserver:
    def __init__(self):
        self.b = 0
        self.scaler = StandardScaler()

        self.alpha = None
        self.r_t = None
        self.w_res = None
        self.w_in = None
        self.w_out = None
        self.r_t_warmup = None
        self.w_out_pred = None
        self.time = 1

    def create_network(self, input_dim, neurons, rewiring_prob, w_res_spectral, alpha, seed):
        """
        Creates our network and initialises the W_in and W_res matrix
        :param input_dim: Defines the number of inputs
        :param neurons: Size of reservoir
        :param k: Amount of connections to nearest neighbors
        :param rewiring_prob:  rewiring probability
        :param w_res_spectral: Largest
        :param alpha:
        :return:
        """
        np.random.seed(seed)
        self.alpha = alpha
        w_res_upper_tri = np.triu(
            nx.to_numpy_array(nx.gnp_random_graph(neurons, rewiring_prob, seed=seed)))  # no self loops
        w_res_weights = np.random.uniform(low=-1, high=1, size=(w_res_upper_tri.shape))  # Link weights
        w_res = w_res_upper_tri * w_res_weights
        w_res = w_res + w_res.T
        eigenvalues = np.max(np.abs(np.linalg.eigvals(w_res)))
        self.w_res = w_res / (eigenvalues / w_res_spectral)  # Noralises by spectral radius
        self.w_in = np.random.uniform(low=-1, high=1, size=(neurons, input_dim))

    def train(self, input_data, observed_data, target_data ,transient,normalised=1e-07) -> np.array:
        """
        Takes an array and updates r_t (reservoir state), computes w_out and transforms the data back to its
        original timescale and returns our training time series.
        :param u_t:
        :return:
        """
        n = input_data.shape[1]+1
        r_t = np.zeros((n, self.w_res.shape[0]))
        for i in range(1, n):
            u_t_update = np.insert(input_data[:,i-1], 0, observed_data[i - 1])
            r_t[i] = (1 - self.alpha) * r_t[i - 1, :] + (
                    self.alpha * np.tanh((self.w_res @ r_t[i - 1, :]) + self.w_in @ u_t_update.T + self.b)) #+ (noise * np.random.uniform(low=-1, high=1, size=(self.w_res.shape[0])))
        r_t = r_t[1:]
        self.r_t_warmup = r_t
        self.observed = observed_data
        r_t_opt = r_t[transient:, :]
        target_data = target_data[:, transient:]
        self.w_out = (target_data @ r_t_opt) @ np.linalg.inv(r_t_opt.T @ r_t_opt + (normalised * np.identity(len(self.w_res))))
        x_out = self.w_out @ r_t.T
        x_out = np.insert(x_out, 0, self.observed, axis=0)
        return x_out

    def inference_all(self, observed_data) -> np.array:
        """
        Takes the timeseries and t_warmup and then updates r_t until it reaches the end of t_warmup. Then, it starts
        predicting until the end of the timeseries is reached. We then transform the data back and return it.
        :param u_t: timeseries
        :param warmup: t_warmup
        :return:
        """
        pred_time = observed_data.shape[0]
        r_t_predict = np.zeros((pred_time + 1, len(self.w_res)))
        r_t_predict[0] = self.r_t_warmup[-1]  # Copy the state from r_t_warmup
        for i in range(1, pred_time + 1):
            x_out = self.w_out @ r_t_predict[i - 1, :]
            u_t = np.insert(x_out, 0, observed_data[i - 1])
            r_t_predict[i] = (1 - self.alpha) * r_t_predict[i - 1, :] + (
                    self.alpha * np.tanh((self.w_res @ r_t_predict[i - 1, :]) + self.w_in @ u_t + self.b))
        self.r_t_warmup = np.vstack([self.r_t_warmup, r_t_predict[1:]])
        self.observed = np.hstack([self.observed, observed_data])
        x_out = self.w_out @ r_t_predict[1:].T
        x_out = np.insert(x_out, 0, observed_data, axis=0)
        return x_out

    def inference_all_1(self, observed_data,prev_data,observed_dim) -> np.array:
        """
        Takes the timeseries and t_warmup and then updates r_t until it reaches the end of t_warmup. Then, it starts
        predicting until the end of the timeseries is reached. We then transform the data back and return it.
        :param u_t: timeseries
        :param warmup: t_warmup
        :return:
        """
        pred_time = observed_data.shape[0]
        r_t_predict = np.zeros((pred_time + 1, len(self.w_res)))
        r_t_predict[0] = self.r_t_warmup[-1]  # Copy the state from r_t_warmup
        for i in range(1, pred_time + 1):
            x_out = self.w_out @ r_t_predict[i - 1, :]
            u_t = np.insert(x_out, observed_dim, observed_data[i - 1])
            r_t_predict[i] = (1 - self.alpha) * r_t_predict[i - 1, :] + (
                    self.alpha * np.tanh((self.w_res @ r_t_predict[i - 1, :]) + self.w_in @ u_t + self.b))
        self.r_t_warmup = np.vstack([self.r_t_warmup, r_t_predict[1:]])
        self.observed = np.hstack([self.observed, observed_data])
        x_out = self.w_out @ r_t_predict[1:].T
        x_out = np.insert(x_out, observed_dim, observed_data, axis=0)
        return x_out

    def inference_one(self, observed_data): # todo done
        prev_rt = self.r_t_warmup[-1]
        x_out = self.w_out @ prev_rt
        u_t = np.insert(x_out, 0, observed_data)
        r_t_predict = (1 - self.alpha) * prev_rt + (
                self.alpha * np.tanh((self.w_res @ prev_rt) + self.w_in @ u_t.flatten() + self.b))
        self.r_t_warmup = np.vstack([self.r_t_warmup, r_t_predict])
        x_out = self.w_out @ r_t_predict.T
        return np.insert(x_out, 0, observed_data, axis=0).flatten()

    def control_one(self, input_data, observed_data, control_input, online,error=None, learning_rate=None):# todo done
        u_t = input_data - control_input
        prev_rt = self.r_t_warmup[-1]  # Copy the state from r_t_warmup
        r_t_predict = (1 - self.alpha) * prev_rt + (
                self.alpha * np.tanh((self.w_res @ prev_rt) + self.w_in @ u_t.flatten() + self.b))
        self.r_t_warmup = np.vstack([self.r_t_warmup, r_t_predict])
        if online:
            self.w_out = self.w_out - learning_rate * (error.reshape(-1, 1) @ prev_rt.reshape(-1, 1).T)
        x_out = self.w_out @ r_t_predict.T
        return np.insert(x_out, 0, observed_data, axis=0).flatten()

    def predict_one(self, input_data, observed_data,online,error=None, learning_rate=None):
        prev_rt = self.r_t_warmup[-1]
        u_t = np.insert(input_data, 0, observed_data)
        r_t_predict = (1 - self.alpha) * prev_rt + (
                self.alpha * np.tanh((self.w_res @ prev_rt) + self.w_in @ u_t + self.b))
        if online:
            self.w_out_pred = self.w_out - learning_rate * (error[1:].reshape(-1, 1) @ prev_rt.reshape(-1, 1).T)
        else:
            self.w_out_pred = self.w_out
        x_out = self.w_out_pred @ r_t_predict.T
        return x_out.flatten()

