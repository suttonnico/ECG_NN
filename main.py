import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from nimblenet.activation_functions import sigmoid_function
from nimblenet.neuralnet import NeuralNet
from nimblenet.data_structures import Instance
from nimblenet.cost_functions import cross_entropy_cost
from nimblenet.learning_algorithms import *
from nimblenet.cost_functions import sum_squared_error
from nimblenet.learning_algorithms import backpropagation
import wfdb
from scipy import signal


def PCA(lats4C, all_lats, cantDim, tam):
    # lats4C = all_lats
    lats4C = np.matrix.transpose(lats4C)
    C = np.cov(lats4C)
    (auval, auvec) = np.linalg.eigh(C)
    idx = auval.argsort()[::-1]
    auval = auval[idx]
    auvec = auvec[:, idx]
    auvec = np.matrix.transpose(auvec)
    all_lats = np.matrix.transpose(all_lats)
    u = np.dot(auvec, all_lats)  # transformo los latidos normalizados al espacio PCA
    # plt.figure()
    # plt.plot(np.matrix.transpose(u)[0], label='Componentes totales')
    # u[cantDim:2 * tam] = 0  # Hago cero las componentes mayores a cantDim
    # plt.plot(np.matrix.transpose(u)[0], label='Componentes reducidas')
    # plt.legend()
    # plt.show()
    #

    u = u[0:cantDim]
    u = np.matrix.transpose(u)

    # latidosRec = np.dot(np.matrix.transpose(auvec), u)
    # latidosRec = np.matrix.transpose(latidosRec)
    # plt.figure()
    # plt.plot(np.matrix.transpose(all_lats)[2010])
    # plt.plot(np.matrix.transpose(latidosRec)[2010])
    # plt.title('Latido original y reconstruido')
    # plt.show()
    # sum1= 0
    # for i in range(0,len(all_lats)):
    #    sum1 += (np.var(np.matrix.transpose(all_lats)[i]))
    # sum2 = 0
    # for i in range(0,len(all_lats)):
    #    sum2 += (np.var(np.matrix.transpose(latidosRec)[i]))
    # print sum1, sum2, 1-(sum1-sum2)/sum1
    # exit(123)
    return u  # latidosRec


def kill_beat(latidos, latido_a_eliminar, cant_org, tam):
    new_beats = np.zeros([cant_org - 1, tam])
    for i in range(0, latido_a_eliminar):
        new_beats[i] = latidos[i]
    for i in range(latido_a_eliminar + 1, cant_org):
        new_beats[i - 1] = latidos[i]
    return new_beats


def lp(f, sig):
    N = len(sig)
    ffts = np.fft.fft(sig)
    freqs = np.fft.fftfreq(N)
    for i in range(0, N):
        if abs(freqs[i]) > f:
            ffts[i] = 0
    sig = np.fft.ifft((ffts))
    return sig


def hp(f, sig):
    N = len(sig)
    ffts = np.fft.fft(sig)
    freqs = np.fft.fftfreq(N)
    for i in range(0, N):
        if abs(freqs[i]) < f:
            ffts[i] = 0
    sig = np.fft.ifft((ffts))
    return sig


def load_data(N, V, S, J, cantDim, tam):
    psignals2, fields2 = wfdb.srdsamp('data/14172', channels=[0])
    psignals4, fields4 = wfdb.srdsamp('data/14172', channels=[1])
    signals2 = np.zeros(len(psignals2))
    signals4 = np.zeros(len(psignals2))

    for i in range(0, len(signals2)):
        signals2[i] = psignals2[i, 0]
        signals4[i] = psignals4[i, 0]
    # HP FILTER
    fc = 250
    fp = 1.0

    # signals2 = np.abs(hp(1.0*fp/fc,signals2))
    #signals4 = np.abs(hp(1.0 * fp / fc, signals4))

    # LP FILTER
    fc = 250
    fp = 230.0
    # signals2 = np.abs(lp(1.0 * fp / fc, signals2))
    #signals4 = np.abs(lp(1.0 * fp / fc, signals4))

    annotation = wfdb.rdann('data/14172', 'atr')

    # Me quedo con los picos de cada ECG que es donde estan las anotaciones
    picos = annotation.annsamp

    # Transpongo las senales leidas para poder operar mas comodamente
    # signals = np.matrix.transpose(signals2)
    # signals3 = np.matrix.transpose(signals4)
    signals = signals2
    signals3 = signals4
    picos[-1] = 0  # descarto el ultimo pico por si el latido esta interrumpido
    # tam = 50 * 2  # Tamano del latido que tomo a la izquierda y a la derecha de la anotacion
    cantLatidos = sum(picos > tam)  # Cuento la cantidad de anotaciones en el segemento a analizar
    latidos = np.zeros([cantLatidos, 2 * tam])  # Reservo espacio para la matriz de datos cantLatidos x 2*tam
    latidosNorm = np.zeros([cantLatidos, 2 * tam])  # Reservo espacio para la matriz de datos normalizados
    numLat = 0  # Inicializo en cero para procesar cada uno de los picos
    count = 0
    for i in picos:
        count += 1
        if i > tam * 2:  # i>tam por si el primer latido esta trunco
            if numLat == 0:
                count_fin = count - 1
            latidos[numLat] += np.append(signals[i - tam / 2:i + tam / 2], np.zeros(tam))
            # latidos[numLat] += np.append(signals[i - tam :i ], np.zeros(tam))# tomo tam muestras a la izquerda y a la derecha de la anotacion
            latidos[numLat] += np.append(np.zeros(tam), signals3[i - tam / 2:i + tam / 2], )
            numLat += 1  # incremento el numero de latidos
    # latidos = np.matrix.transpose(latidos)
    # El set de latidos esta en latidos y el largo de cada latido es de tam muestras
    medias = np.zeros(cantLatidos)
    desv = np.zeros(cantLatidos)
    # latidosNorm = np.matrix.transpose(latidosNorm)
    for i in range(0, cantLatidos):
        medias[i] = np.mean(latidos[i])
        desv[i] = np.std(latidos[i])

    for idx in range(0, cantLatidos):
        latidosNorm[idx] = latidos[idx] - np.ones(tam * 2) * medias[idx]
        latidosNorm[idx] = latidosNorm[idx] / desv[idx]

    anotaciones = annotation.anntype
    anotaciones = anotaciones[count_fin:len(anotaciones) - 1]
    print anotaciones
    N_all = 10000
    V_all = 1000
    S_all = 300
    J_all = 140
    all_N = np.zeros([N_all, 2 * tam])
    last_j = 0
    for i in range(0, N_all):
        for j in range(last_j, cantLatidos):
            if anotaciones[j] == 'N':
                # print 'j',j,'i',i
                all_N[i] = latidosNorm[j]
                last_j = j + 1
                break
    all_V = np.zeros([V_all, 2 * tam])
    last_j = 0
    for i in range(0, V_all):
        for j in range(last_j, cantLatidos):
            if anotaciones[j] == 'V':
                all_V[i] = latidosNorm[j]
                last_j = j + 1
                break
    all_S = np.zeros([S_all, 2 * tam])
    last_j = 0
    for i in range(0, S_all):
        for j in range(last_j, cantLatidos):
            if anotaciones[j] == 'S':
                all_S[i] = latidosNorm[j]
                last_j = j + 1
                break
    all_J = np.zeros([J_all, 2 * tam])
    last_j = 0
    for i in range(0, J_all):
        for j in range(last_j, cantLatidos):
            if anotaciones[j] == 'J':
                all_J[i] = latidosNorm[j]
                last_j = j + 1
                break

    all_cant = N_all + V_all + S_all + J_all
    LatidosNorm = np.zeros([all_cant, 2 * tam])
    LatidosNorm[0:N_all] = all_N
    LatidosNorm[N_all:N_all + V_all] = all_V
    LatidosNorm[N_all + V_all:N_all + V_all + S_all] = all_S
    LatidosNorm[N_all + V_all + S_all:N_all + V_all + S_all + J_all] = all_J
    # cantDim = 40#2 * tam  # 110
    lats4pca = np.zeros([40, 2 * tam])
    lats4pca[0:10] = all_N[0:10]
    lats4pca[10:20] = all_V[0:10]
    lats4pca[20:30] = all_S[0:10]
    lats4pca[30:40] = all_J[0:10]
    for i in range(0, len(LatidosNorm)):
        fc = 250
        fp = 240.0
        latidosNorm[i] = np.abs(lp(1.0 * fp / fc, latidosNorm[i]))
    if cantDim > 0:
        LatidosNorm = PCA(lats4pca, LatidosNorm, cantDim, tam)
    else:
        cantDim = 2 * tam
    all_N = LatidosNorm[0:N_all]
    all_V = LatidosNorm[N_all:N_all + V_all]
    all_S = LatidosNorm[N_all + V_all:N_all + V_all + S_all]
    all_J = LatidosNorm[N_all + V_all + S_all:N_all + V_all + S_all + J_all]

    training_set_N = np.zeros([N_all, cantDim])
    for i in range(0, N):
        # j = N_all-i-1#int(np.floor(np.random.rand()*N_all))
        j = int(np.floor(np.random.rand() * N_all))
        training_set_N[i] = all_N[j]
        all_N = kill_beat(all_N, j, N_all, cantDim)
        N_all -= 1
    training_set_V = np.zeros([V_all, cantDim])
    for i in range(0, V):
        j = int(np.floor(np.random.rand() * V_all))
        training_set_V[i] = all_V[j]
        all_V = kill_beat(all_V, j, V_all, cantDim)
        V_all -= 1
    training_set_S = np.zeros([S_all, cantDim])
    for i in range(0, S):
        j = int(np.floor(np.random.rand() * S_all))
        training_set_S[i] = all_S[j]
        all_S = kill_beat(all_S, j, S_all, cantDim)
        S_all -= 1
    training_set_J = np.zeros([J_all, cantDim])
    for i in range(0, J):
        j = int(np.floor(np.random.rand() * J_all))
        training_set_J[i] = all_J[j]
        all_J = kill_beat(all_J, j, J_all, cantDim)
        J_all -= 1
    test_set_N = all_N
    test_set_V = all_V
    test_set_S = all_S
    test_set_J = all_J
    N_left = N_all
    V_left = V_all
    S_left = S_all
    J_left = J_all
    # PLOTEO
    plt.figure()
    plt.plot(all_N[0])
    plt.title('N')
    plt.figure()
    plt.plot(all_V[0])
    plt.title('V')
    plt.figure()
    plt.plot(all_S[0])
    plt.title('S')
    plt.figure()
    plt.plot(all_J[0])
    plt.title('J')
    # plt.show()

    return N, V, S, J, N_left, V_left, S_left, J_left, training_set_N, training_set_V, training_set_S, training_set_J, test_set_N, test_set_V, test_set_S, test_set_J


def train_net(N, V, S, J, tam, training_set_N, training_set_V, training_set_S, training_set_J, anwsers, error):
    training_set_NN = [Instance(training_set_N[i], [anwsers[0]]) for i in range(0, N)]
    training_set_VV = [Instance(training_set_V[i], [anwsers[1]]) for i in range(0, V)]
    training_set_SS = [Instance(training_set_S[i], [anwsers[2]]) for i in range(0, S)]
    training_set_JJ = [Instance(training_set_J[i], [anwsers[3]]) for i in range(0, J)]
    training_set = np.append(training_set_NN, training_set_VV)
    training_set = np.append(training_set, training_set_SS)
    training_set = np.append(training_set, training_set_JJ)
    test_set_NN = [Instance(test_set_N[i], [anwsers[0]]) for i in range(0, 100)]
    test_set_VV = [Instance(test_set_V[i], [anwsers[1]]) for i in range(0, 100)]
    test_set_SS = [Instance(test_set_S[i], [anwsers[2]]) for i in range(0, 100)]
    test_set_JJ = [Instance(test_set_J[i], [anwsers[3]]) for i in range(0, 100)]
    test_set = np.append(test_set_NN, test_set_VV)
    test_set = np.append(test_set, test_set_SS)
    test_set = np.append(test_set, test_set_JJ)
    settings = {"n_inputs": tam,
                "layers": [(int(np.ceil(tam / 2)), sigmoid_function), (int(np.ceil(tam / 4)), sigmoid_function),
                           (1, sigmoid_function)]}
    net = NeuralNet(settings)
    cost_function = sum_squared_error
    backpropagation(
        net,  # the network to train
        training_set,  # specify the training set
        test_set,  # specify the test set
        cost_function,  # specify the cost function to calculate error

        early_stop=4,
        ERROR_LIMIT=error,  # define an acceptable error limit
        max_iterations=2000,  # continues until the error limit is reach if this argument is skipped
        print_rate=1000,
        # input_layer_dropout=0.8,  # Dropout fraction of the input layer
        # hidden_layer_dropout=0.8,  # Dropout fraction of in the hidden layer(s)
    )
    return net


def train_4net(N, V, S, J, tam, training_set_N, training_set_V, training_set_S, training_set_J, anwsers):
    training_set_NN = [Instance(training_set_N[i], [1, 0, 0, 0]) for i in range(0, N)]
    training_set_VV = [Instance(training_set_V[i], [0, 1, 0, 0]) for i in range(0, V)]
    training_set_SS = [Instance(training_set_S[i], [0, 0, 1, 0]) for i in range(0, S)]
    training_set_JJ = [Instance(training_set_J[i], [0, 0, 0, 1]) for i in range(0, J)]
    training_set = np.append(training_set_NN, training_set_VV)
    training_set = np.append(training_set, training_set_SS)
    training_set = np.append(training_set, training_set_JJ)
    test_set_NN = [Instance(test_set_N[i], [1, 0, 0, 0]) for i in range(0, 100)]
    test_set_VV = [Instance(test_set_V[i], [0, 1, 0, 0]) for i in range(0, 100)]
    test_set_SS = [Instance(test_set_S[i], [0, 0, 1, 0]) for i in range(0, 100)]
    test_set_JJ = [Instance(test_set_J[i], [0, 0, 0, 1]) for i in range(0, 100)]
    test_set = np.append(test_set_NN, test_set_VV)
    test_set = np.append(test_set, test_set_SS)
    test_set = np.append(test_set, test_set_JJ)
    settings = {"n_inputs": tam,
                "layers": [(int(np.ceil(tam / 2)), sigmoid_function), (int(np.ceil(tam / 4)), sigmoid_function),
                           (4, sigmoid_function)]}
    net = NeuralNet(settings)
    cost_function = sum_squared_error
    backpropagation(
        net,  # the network to train
        training_set,  # specify the training set
        test_set,  # specify the test set
        cost_function,  # specify the cost function to calculate error

        early_stop=5,
        # ERROR_LIMIT=1e-3,  # define an acceptable error limit
        max_iterations=10000,  # continues until the error limit is reach if this argument is skipped
        print_rate=1000,
        # input_layer_dropout=0.8,  # Dropout fraction of the input layer
        # hidden_layer_dropout=0.8,  # Dropout fraction of in the hidden layer(s)
    )
    return net


def test_net(net, test_set, left, UpOrDown):
    wr = 0
    for i in range(1, left + 1):
        p = net.predict([Instance(test_set[i - 1])])
        s = p == np.max(p)
        s = s[0]
        if p >= 0.5:
            wr = 1.0 * (wr * (i - 1) + 1) / i
        else:
            wr = 1.0 * (wr * (i - 1)) / i
    if UpOrDown == 1:
        return wr
    else:
        return 1 - wr


def predict_1(N_net, V_net, S_net, J_net, test_case):
    np = N_net.predict([Instance(test_case)])[0]
    vp = V_net.predict([Instance(test_case)])[0]
    sp = S_net.predict([Instance(test_case)])[0]
    jp = J_net.predict([Instance(test_case)])[0]
    p = [np[0], vp[0], sp[0], jp[0]]
    th = max(p)
    s = p == th
    if s[0] == 1:
        return 1
    else:
        if s[1] == 1:
            return 2
        else:
            if s[2] == 1:
                return 3
            else:
                return 4


def test_p(N_net, V_net, S_net, J_net, test_set, left, anwser1, anwser2, anwser3, anwser4):
    wrn = 0
    wrv = 0
    wrs = 0
    wrj = 0
    for i in range(1, left + 1):
        p = predict_1(N_net, V_net, S_net, J_net, test_set[i - 1])
        if p == anwser1:
            wrn = 1.0 * (wrn * (i - 1) + 1) / i
        else:
            wrn = 1.0 * (wrn * (i - 1)) / i
        if p == anwser2:
            wrv = 1.0 * (wrv * (i - 1) + 1) / i
        else:
            wrv = 1.0 * (wrv * (i - 1)) / i
        if p == anwser3:
            wrs = 1.0 * (wrs * (i - 1) + 1) / i
        else:
            wrs = 1.0 * (wrs * (i - 1)) / i
        if p == anwser4:
            wrj = 1.0 * (wrj * (i - 1) + 1) / i
        else:
            wrj = 1.0 * (wrj * (i - 1)) / i

    return wrn, wrv, wrs, wrj


def test_p4(net, test_set, left, anwser):
    wr = 0
    for i in range(1, left + 1):
        p = net.predict([Instance(test_set[i - 1])])
        p = p == np.max(p)
        p = p[0]
        if p[0] == anwser[0] & p[1] == anwser[1] & p[2] == anwser[2] & p[3] == anwser[3]:
            wr = 1.0 * (wr * (i - 1) + 1) / i
        else:
            wr = 1.0 * (wr * (i - 1)) / i
    return wr


iters = 1
wn = 0
wv = 0
ws = 0
wj = 0

for it in range(0, iters):
    [N, V, S, J, N_left, V_left, S_left, J_left, training_set_N, training_set_V, training_set_S, training_set_J,
     test_set_N,
     test_set_V, test_set_S, test_set_J] = load_data(40, 40, 40, 40, 40, 50 * 2)
    tam = len(training_set_N[0])

    # network = train_4net(10, 10, 10, 10, tam, training_set_N, training_set_V, training_set_S, training_set_J,[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    # print 'N encontrados', test_p4(network, test_set_N, N_left, [1,0,0,0])
    # print 'V encontrados', test_p4(network, test_set_V, V_left, [0,1,0,0])
    # print 'S encontrados', test_p4(network, test_set_S, S_left, [0,0,1,0])
    # print 'J encontrados', test_p4(network, test_set_J, J_left, [0,0,0,1])


    N_network = train_net(50, 40, 40, 40, tam, training_set_N, training_set_V, training_set_S, training_set_J,
                          [1, 0, 0, 0], 1e-3)
    V_network = train_net(40, 40, 40, 40, tam, training_set_N, training_set_V, training_set_S, training_set_J,
                          [0, 1, 0, 0], 1e-3)
    S_network = train_net(40, 40, 40, 40, tam, training_set_N, training_set_V, training_set_S, training_set_J,
                          [0, 0, 1, 0], 1e-3)
    J_network = train_net(40, 40, 40, 40, tam, training_set_N, training_set_V, training_set_S, training_set_J,
                          [0, 0, 0, 1], 1e-3)

    print test_p(N_network, V_network, S_network, J_network, test_set_N, N_left, 1, 2, 3, 4)
    print test_p(N_network, V_network, S_network, J_network, test_set_V, V_left, 1, 2, 3, 4)
    print test_p(N_network, V_network, S_network, J_network, test_set_S, S_left, 1, 2, 3, 4)
    print test_p(N_network, V_network, S_network, J_network, test_set_J, J_left, 1, 2, 3, 4)


    # N_network = train_net(50, 40, 40, 40, tam, training_set_N, training_set_V, training_set_S, training_set_J,[1,0,0,0],1e-3)
    # V_network = train_net(40, 40, 40, 40, tam, training_set_N, training_set_V, training_set_S, training_set_J,[0,1,0,0],1e-3)
    # S_network = train_net(40, 40, 40, 40, tam, training_set_N, training_set_V, training_set_S, training_set_J,[0,0,1,0],1e-3)
    # J_network = train_net(40, 40, 40, 40, tam, training_set_N, training_set_V, training_set_S, training_set_J,[0,0,0,1],1e-3)


    # wn += test_p(N_network, V_network, S_network, J_network, test_set_N, N_left, 1)
    # wv += test_p(N_network, V_network, S_network, J_network, test_set_V, V_left, 2)
    # ws += test_p(N_network, V_network, S_network, J_network, test_set_S, S_left, 3)
print 'FINAL'
print tam
print 'N encontrados', wn / iters
print 'V encontrados', wv / iters
print 'S encontrados', ws / iters
print 'J encontrados', wj/iters
