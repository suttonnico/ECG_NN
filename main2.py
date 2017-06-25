import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from nimblenet.activation_functions import sigmoid_function
from nimblenet.neuralnet import NeuralNet


def load_data():
    cantidades = sio.loadmat('cantidades.mat')
    cantidades = cantidades['cantidades']
    cantidades = cantidades[0]
    cantLatidos = cantidades[0]
    N = cantidades[1]
    V = cantidades[2]
    S = cantidades[3]
    J = cantidades[4]

    latidosNorm = sio.loadmat('latidosNorm.mat')
    latidosNorm = latidosNorm['latidosNorm']
    training_set_N = sio.loadmat('training_set_N.mat')
    training_set_N = training_set_N['training_set_N']
    training_set_V = sio.loadmat('training_set_V.mat')
    training_set_V = training_set_V['training_set_V']
    training_set_S = sio.loadmat('training_set_S.mat')
    training_set_S = training_set_S['training_set_S']
    training_set_J = sio.loadmat('training_set_J.mat')
    training_set_J = training_set_J['training_set_J']

    anotaciones = sio.loadmat('anotaciones.mat')
    anotaciones = anotaciones['anotaciones']
    return cantLatidos, N, V, S, J, latidosNorm, training_set_N, training_set_V, training_set_S, training_set_J, anotaciones


[cantLatidos, N, V, S, J, latidosNorm, training_set_N, training_set_V, training_set_S, training_set_J,
 anotaciones] = load_data()
