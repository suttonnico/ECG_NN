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
        self.W = np.random.rand(self.N-1,M,M+1)     #+1 por bias
        #W[capa,neurona,entradas]
    def out(self, rn_in):
        for j in range(0,self.N-1):
            outl = np.zeros(self.n[j+1])
            for i in range(0, self.n[j+1]):
                rn_in = np.append([1], rn_in)
                outl[i] = self.F(np.dot(self.W[j, i, :], rn_in))
                rn_in = outl
        return rn_in


    def F(self, input):
        return np.tanh(input)


    def dF(self, input):
        return 1 - self.F(input) ** 2


    def train(self,train_input, train_output):
        training_cases = len(train_input[:, 1])
        for i in range(0,training_cases):
            O = self.out(train_input[i, :])
            e = O - train_output[i]
            total_error = self.sum_of_squares(e)
            self. update(train_input,train_output,total_error)
    def sum_of_squares(self,e):
        sum_c = 0
        for i in range(0,len(e)):
            sum_c += e[i] ** 2
        return sum_c

    def update(self, train_input,train_output,total_error):

        self.W = self.W



train_data = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
train_output = [0,1,1,0]
rn = mlp([2,2,1])
print rn.W[0,0,:]
rn.train(train_data, train_output)
exit(100)

e = 1
lr = 0.5

#for i in range(0, 100):
#    for t in range(0, 4):

