import wfdb
import numpy as np
import matplotlib.pyplot as plt


signals2, fields2= wfdb.srdsamp('data/14172', channels=[0, 1])
annotation = wfdb.rdann('data/14172', 'atr')
picos=annotation.annsamp
#Me quedo con los picos de cada ECG que es donde estan las anotaciones
picos=annotation.annsamp

#Transpongo las senales leidas para poder operar mas comodamente
signals=np.matrix.transpose(signals2)
picos[-1]=0 #descarto el ultimo pico por si el latido esta interrumpido
tam=125 #Tamano del latido que tomo a la izquierda y a la derecha de la anotacion
cantLatidos=sum(picos>tam) #Cuento la cantidad de anotaciones en el segemento a analizar
latidos=np.zeros([cantLatidos,2*tam]) #Reservo espacio para la matriz de datos cantLatidos x 2*tam
latidosNorm=np.zeros([2*tam,cantLatidos])#Reservo espacio para la matriz de datos normalizados
numLat=0 #Inicializo en cero para procesar cada uno de los picos
for i in picos:
    if i>tam: # i>tam por si el primer latido esta trunco
        latidos[numLat]=signals[0][i-tam:i+tam] #tomo tam muestras a la izquerda y a la derecha de la anotacion
        numLat+=1 #incremento el numero de latidos

#El set de latidos esta en latidos y el largo de cada latido es de tam muestras

latidos=np.matrix.transpose(latidos) #vuelvo a acondicionar los latidos para seguir operando comodamente

# A continuacion calculo la media y la varianza de cada elemento del vector de datos para poder normalizar cada dato
medias=np.zeros(2*tam)
desv=np.zeros(2*tam)
for i in range(0,2*tam):
    medias[i]=np.mean(latidos[i,:])
    desv[i]=np.std(latidos[i,:])

# Normalizo cada latido
for idx in range(0,2*tam):
    latidosNorm[idx]=latidos[idx]-np.ones(cantLatidos)*medias[idx]
    latidosNorm[idx]=latidosNorm[idx]/desv[idx]

#Grafico los latidos normalizados
plt.plot(latidosNorm[0])
plt.show()
