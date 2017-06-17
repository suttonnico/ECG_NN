import numpy as np

#def dw_update()

class mlp:
    #n debe ser un vector con el primero elemento la cantidad de entradas a la red, y las restantes la cantidad de
    #neuronas en cada capa, la idea es manejar backpropagation solo para una red de etapas multiples generica
    def __init__(self, n):
        self.n = n
        self.N = np.size(n)
        M = np.max(n)
        #el primero es el numero de entradas a la red por eso N-1
        self.W = np.random.rand(self.N-1,M,M)
        #W[capa,neurona
    def out(self, rn_in):
        for j in range(0,self.N-1):
            outl = np.zeros(self.n[j+1])
            for i in range(0, self.n[j+1]):
                outl[i] = np.tanh(np.dot(self.W[j, i, :], rn_in))
                rn_in = outl
        return rn_in

    def train(self,train_data):
        print 'not done yet'





    def update(self, total_error):
        self.W = self.W

TP = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
rn = mlp([2,2,1])

exit(100)
T=[0,1,1,0]
e = 1
lr = 0.5

#for i in range(0, 100):
#    for t in range(0, 4):

