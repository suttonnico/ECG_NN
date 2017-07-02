import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from nimblenet.activation_functions import sigmoid_function
from nimblenet.neuralnet import NeuralNet
from nimblenet.data_structures import Instance
from nimblenet.cost_functions import cross_entropy_cost
from nimblenet.learning_algorithms import *
from nimblenet.cost_functions import sum_squared_error

def kill_beat(latidos, latido_a_eliminar,cant_org,tam):
    new_beats = np.zeros([cant_org-1,2*tam])
    for i in range(0,latido_a_eliminar):
        new_beats[i] = latidos[i]
    for i in range(latido_a_eliminar+1,cant_org):
        new_beats[i-1] = latidos[i]
    return new_beats

def load_data(N, V, S, J):
    cantidades = sio.loadmat('cantidades.mat')
    cantidades = cantidades['cantidades']
    cantidades = cantidades[0]
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
    tam = 125
    #all_cant = N_all+V_all+S_all+J_all
    #LatidosNorm = np.zeros([all_cant, 2 * tam])
    #LatidosNorm[0:N_all] = all_N
    #LatidosNorm[N_all:N_all+V_all] = all_V
    #LatidosNorm[N_all+ V_all:N_all + V_all+ S_all] = all_S
    #atidosNorm[N_all + V_all+ S_all:N_all + V_all + S_all+ J_all] = all_J
    training_set_N = np.zeros([N_all, 2 * tam])
    for i in range(0,N):
        j = int(np.floor(np.random.rand()*N_all))
        training_set_N[i] = all_N[j]
        all_N = kill_beat(all_N, j, N_all, 125)
        N_all -= 1
    training_set_V = np.zeros([V_all, 2 * tam])
    for i in range(0, V):
        j = int(np.floor(np.random.rand() * V_all))
        training_set_V[i] = all_V[j]
        all_V = kill_beat(all_V, j, V_all, 125)
        V_all -= 1
    training_set_S = np.zeros([S_all, 2 * tam])
    for i in range(0, S):
        j = int(np.floor(np.random.rand() * S_all))
        training_set_S[i] = all_S[j]
        all_S = kill_beat(all_S, j, S_all, 125)
        S_all -= 1
    training_set_J = np.zeros([J_all, 2 * tam])
    for i in range(0, J):
        j = int(np.floor(np.random.rand() * J_all))
        training_set_J[i] = all_J[j]
        all_J = kill_beat(all_J, j, J_all, 125)
        J_all -= 1

    test_set_N = all_N
    test_set_V = all_V
    test_set_S = all_S
    test_set_J = all_J
    N_left = N_all
    V_left = V_all
    S_left = S_all
    J_left = J_all

    return N, V, S, J, N_left, V_left, S_left, J_left, training_set_N, training_set_V, training_set_S, training_set_J, test_set_N, test_set_V, test_set_S, test_set_J


[N, V, S, J, N_left, V_left, S_left, J_left, training_set_N, training_set_V, training_set_S, training_set_J, test_set_N,
 test_set_V, test_set_S, test_set_J] = load_data(10 , 10, 10, 10)
tam = len(training_set_N[0])
#PCA
cantDim=16

C=np.cov(training_set_N[0])
(auval,auvec)=np.linalg.eigh(C)

settings = {"n_inputs": tam, "layers": [(36, sigmoid_function),(16, sigmoid_function), (4, sigmoid_function)]}
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


cost_function = cross_entropy_cost#sum_squared_error
test_set = training_set
RMSprop(
    network,  # the network to train
    training_set,  # specify the training set
    test_set,  # specify the test set
    cost_function,  # specify the cost function to calculate error

    ERROR_LIMIT=1e-3,  # define an acceptable error limit
    max_iterations         = 3000,      # continues until the error limit is reach if this argument is skipped

    input_layer_dropout=0.5,    # Dropout fraction of the input layer
    hidden_layer_dropout=0.5,   # Dropout fraction of in the hidden layer(s)
)
test_set = []
wrn = 0
for i in range(1, N_left+1):
    p = network.predict([Instance(test_set_N[i-1])])
    s = p == np.max(p)
    s = s[0]
    if s[0] == 1:
        wrn = 1.0*(wrn*(i-1)+1)/i
    else:
        wrn = 1.0*(wrn * (i - 1)) / i
print 'wrn',wrn
wrv = 0
for i in range(1, V_left+1):
    p = network.predict([Instance(test_set_V[i-1])])
    s = p == np.max(p)
    s = s[0]
    if s[1] == 1:
        wrv = 1.0*(wrv*(i-1)+1)/i
    else:
        wrv = 1.0*(wrv * (i - 1)) / i
print 'wrv', wrv
wrs = 0
for i in range(1, S_left+1):
    p = network.predict([Instance(test_set_S[i-1])])
    s = p == np.max(p)
    s = s[0]
    if s[2] == 1:
        wrs = 1.0*(wrs*(i-1)+1)/i
    else:
        wrs = 1.0*(wrs * (i - 1)) / i
print 'wrs', wrs
wrj = 0
for i in range(1, J_left+1):
    p = network.predict([Instance(test_set_J[i-1])])
    s = p == np.max(p)
    s = s[0]
    if s[3] == 1:
        wrj = 1.0*(wrj*(i-1)+1)/i
    else:
        wrj = 1.0*(wrj * (i - 1)) / i
print 'wrj', wrj
