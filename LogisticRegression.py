import theano
import theano.tensor as T
from theano import function
import numpy as np
import matplotlib.pyplot as plt

class LR(object):
    x = T.dmatrix('x')
    y = T.lvector('y')
    def __init__(self, i_dim, max_train_step):
        self.train_step = max_train_step
        self.rng = np.random
        self.w = theano.shared(self.rng.randn(i_dim), name = 'w')
        self.b = theano.shared(0.0, name = 'b')
        self.logisticFunc = 1 / (1 + T.exp(-T.dot(self.x, self.w) - self.b))
        self.prediction = self.logisticFunc > 0.5
        self.cross_entropy = -self.y * T.log(self.logisticFunc) - (1 - self.y) * T.log(1 - self.logisticFunc)
        self.cost = self.cross_entropy.mean() + 0.01 * (self.w ** 2).sum()
        self.gw, self.gb = T.grad(self.cost, [self.w, self.b])
        self.trainFunc = function([self.x, self.y], [self.prediction, self.cost],
                                  updates = ((self.w, self.w - 0.1 * self.gw), (self.b, self.b - 0.1 * self.gb)))
        self.predict = function([self.x], self.prediction)

    def train(self, data):
        for i in range(self.train_step):
            print i
            prediction, cost = self.trainFunc(data[0], data[1])

    def predict(self, data):
        return self.predict(data)


if __name__ == '__main__':
    N = 400
    feats = 784
    D = [np.random.randn(N, feats), np.random.randint(0,2,N)]
    lr = LR(784, 10000)
    lr.train(D)
    print lr.predict(D[0])
    print D[1]
