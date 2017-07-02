import wfdb
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def kill_beat(latidos, anotaciones, latido_a_eliminar,cant_org,tam):
    new_beats = np.zeros([cant_org-1,2*tam])
    new_an = np.chararray(cant_org-1)
    for i in range(0,latido_a_eliminar):
        new_beats[i] = latidos[i]
        new_an[i] = anotaciones[i]
    for i in range(latido_a_eliminar+1,cant_org):
        new_beats[i-1] = latidos[i]
        new_an[i-1] = anotaciones[i]
    return new_beats, new_an

def crate_training_cases(latidos, anotaciones, char, cant_training, cant_org, tam):
    training_set = np.zeros([cant_training,2*tam])
    j_ant = 0
    for i in range(0,cant_training):
        print 1.0 * i / cant_training
        for j in range(j_ant, cant_org):
            if anotaciones[j] == char:
                training_set[i] = latidos[j]
                [latidos, anotaciones] = kill_beat(latidos,anotaciones,j,cant_org,tam)
                cant_org -= 1
                j_ant = j
                break

    return training_set,latidos,anotaciones

signals2, fields2= wfdb.srdsamp('data/14172', channels=[0, 1])
annotation = wfdb.rdann('data/14172', 'atr')
# signals2, fields2= wfdb.srdsamp('data/14046', channels=[0, 1])
# annotation = wfdb.rdann('data/14046', 'atr')

picos=annotation.annsamp
#Me quedo con los picos de cada ECG que es donde estan las anotaciones
picos=annotation.annsamp

#Transpongo las senales leidas para poder operar mas comodamente
signals=np.matrix.transpose(signals2)
picos[-1]=0 #descarto el ultimo pico por si el latido esta interrumpido
tam = 125  #Tamano del latido que tomo a la izquierda y a la derecha de la anotacion
cantLatidos=sum(picos>tam) #Cuento la cantidad de anotaciones en el segemento a analizar
latidos=np.zeros([cantLatidos,2*tam]) #Reservo espacio para la matriz de datos cantLatidos x 2*tam
latidosNorm=np.zeros([cantLatidos,2*tam])#Reservo espacio para la matriz de datos normalizados
numLat=0 #Inicializo en cero para procesar cada uno de los picos

for i in picos:
    if i>tam: # i>tam por si el primer latido esta trunco
        latidos[numLat]=signals[0][i-tam:i+tam] #tomo tam muestras a la izquerda y a la derecha de la anotacion
        numLat+=1 #incremento el numero de latidos
#El set de latidos esta en latidos y el largo de cada latido es de tam muestras

for i in range(0,cantLatidos):
    latidosNorm[i] = latidos[i] - np.mean(latidos[i])
    latidosNorm[i] /= np.std(latidosNorm[i])


anotaciones = annotation.anntype

#genero training set
N = 10  #cantidad de latidos normales en el training set
V = 20  # cantidad de latidos ventriculares en el training set
S = 20  # cantidad de latidos superventriculares prematuros en el training set
J = 20  # cantidad de latidos nodales prematuros en el training set
print anotaciones

# print 'N'
# [training_set_N,latidosNorm,anotaciones] = crate_training_cases(latidosNorm,anotaciones,'N',N,cantLatidos,tam)
# cantLatidos -= N
# print 'V'
# [training_set_V,latidosNorm,anotaciones] = crate_training_cases(latidosNorm,anotaciones,'V',V,cantLatidos,tam)
# cantLatidos -= V
# print 'S'
# [training_set_S,latidosNorm,anotaciones] = crate_training_cases(latidosNorm,anotaciones,'S',S,cantLatidos,tam)
# cantLatidos -= S
# print 'J'
# [training_set_J,latidosNorm,anotaciones] = crate_training_cases(latidosNorm,anotaciones,'J',J,cantLatidos,tam)
# cantLatidos -= J
V_left = 0
for i in range(0, cantLatidos):
    if anotaciones[i] == 'V':
        V_left += 1
[all_V, latidosNorm, anotaciones] = crate_training_cases(latidosNorm, anotaciones, 'V', V_left, cantLatidos, tam)
cantLatidos -= V_left
S_left = 0
for i in range(0, cantLatidos):
    if anotaciones[i] == 'S':
        S_left += 1
[all_S, latidosNorm, anotaciones] = crate_training_cases(latidosNorm, anotaciones, 'S', S_left, cantLatidos, tam)
cantLatidos -= S_left
J_left = 0
for i in range(0, cantLatidos):
    if anotaciones[i] == 'J':
        J_left += 1
[all_J, latidosNorm, anotaciones] = crate_training_cases(latidosNorm, anotaciones, 'J', J_left, cantLatidos, tam)
cantLatidos -= J_left
all_N = latidosNorm
N_left = cantLatidos
# sio.savemat('training_set_J.mat', {'training_set_J': training_set_J})
# sio.savemat('training_set_V.mat', {'training_set_V': training_set_V})
# sio.savemat('training_set_S.mat', {'training_set_S': training_set_S})
# sio.savemat('training_set_N.mat', {'training_set_N': training_set_N})
sio.savemat('all_J.mat', {'all_J': all_J})
sio.savemat('all_V.mat', {'all_V': all_V})
sio.savemat('all_S.mat', {'all_S': all_S})
sio.savemat('all_N.mat', {'all_N': all_N})
cantidades = [N_left, V_left, S_left, J_left]
sio.savemat('cantidades.mat', {'cantidades': cantidades})
