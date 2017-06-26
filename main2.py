import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from nimblenet.activation_functions import sigmoid_function
from nimblenet.neuralnet import NeuralNet
from nimblenet.data_structures import Instance
from nimblenet.cost_functions import cross_entropy_cost
from nimblenet.learning_algorithms import *


def load_data(N, V, S, J):
    cantidades = sio.loadmat('cantidades.mat')
    cantidades = cantidades['cantidades']
    cantidades = cantidades[0]
    cantLatidos = cantidades[0]
    N_all = cantidades[0]
    V_all = cantidades[1]
    S_all = cantidades[2]
    J_all = cantidades[3]
    all_N = sio.loadmat('all_N.mat')
    all_N = all_N['all_N']
    all_V = sio.loadmat('all_V.mat')
    all_V = all_V['all_V']
    all_S = sio.loadmat('all_S.mat')
    all_S = all_S['all_S']
    all_J = sio.loadmat('all_J.mat')
    all_J = all_J['all_J']

    training_set_N = all_N[0:N]
    training_set_V = all_V[0:V]
    training_set_S = all_S[0:S]
    training_set_J = all_J[0:J]
    test_set_N = all_N[N:N_all]
    test_set_V = all_V[V:V_all]
    test_set_S = all_S[S:S_all]
    test_set_J = all_J[J:J_all]
    N_left = N_all - N
    V_left = V_all - V
    S_left = S_all - S
    J_left = J_all - J

    return N, V, S, J, N_left, V_left, S_left, J_left, training_set_N, training_set_V, training_set_S, training_set_J, test_set_N, test_set_V, test_set_S, test_set_J


[N, V, S, J, N_left, V_left, S_left, J_left, training_set_N, training_set_V, training_set_S, training_set_J, test_set_N,
 test_set_V, test_set_S, test_set_J] = load_data(10, 20, 20, 20)
tam = len(training_set_N[0])
settings = {"n_inputs": tam, "layers": [(10, sigmoid_function), (4, sigmoid_function)]}
network = NeuralNet(settings)
training_set = []
for i in range(0, N):
    training_set = np.append(training_set, Instance(training_set_N[i], [1, 0, 0, 0]))
for i in range(0, V):
    training_set = np.append(training_set, Instance(training_set_V[i], [0, 1, 0, 0]))
for i in range(0, S):
    training_set = np.append(training_set, Instance(training_set_S[i], [0, 0, 1, 0]))
for i in range(0, J):
    training_set = np.append(training_set, Instance(training_set_J[i], [0, 0, 0, 1]))

test_set = []
for i in range(0, N_left):
    test_set = np.append(test_set, Instance(test_set_N[i], [1, 0, 0, 0]))
for i in range(0, V_left):
    test_set = np.append(test_set, Instance(test_set_V[i], [0, 1, 0, 0]))
for i in range(0, S_left):
    test_set = np.append(test_set, Instance(test_set_S[i], [0, 0, 1, 0]))
for i in range(0, J_left):
    test_set = np.append(test_set, Instance(test_set_J[i], [0, 0, 0, 1]))

cost_function = cross_entropy_cost
# test_set = training_set
RMSprop(
    network,  # the network to train
    training_set,  # specify the training set
    test_set,  # specify the test set
    cost_function,  # specify the cost function to calculate error

    ERROR_LIMIT=1e-2,  # define an acceptable error limit
    # max_iterations         = 100,      # continues until the error limit is reach if this argument is skipped
)
