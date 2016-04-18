import gzip
import pickle
import theano.tensor as T
import theano
import numpy as np
from theano import function

def load_data(dataSet = './data/mnist.pkl.gz'):
    with gzip.open(dataSet, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    def shared_dataSet(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.array(data_x, dtype=theano.config.floatX), borrow = borrow)
        shared_y = theano.shared(np.array(data_y, dtype=theano.config.floatX), borrow = borrow)
        return shared_x, shared_y

    train_set_x, train_set_y = shared_dataSet(train_set)
    valid_set_x, valid_set_y = shared_dataSet(valid_set)
    test_set_x, test_set_y = shared_dataSet(test_set)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

class softMaxClassifier(object):
    def __init__(self, train_x, train_y, validation_x, validation_y, max_iter_times, rate = 0.13):
        self.rng = np.random
        self.train_x = train_x
        self.train_y = train_y
        self.validation_x = validation_x
        self.validation_y = validation_y
        self.max_iter_times = max_iter_times
        self.x = T.matrix('input')
        self.y = T.lvector('output')
        self.w = theano.shared(self.rng.randn(train_x.shape[1], train_y.shape[0]), borrow = True)
        self.b = theano.shared(np.zeros(train_y.shape[0]), borrow = True)
        self.p_y_given_x = T.nnet.softmax(T.dot(self.x, self.w) + self.b)
        self.cost = self.negativeLogLikelihood(self.y) #+ 0.01 * T.sum()(self.w ** 2)#.sum()
        self.gw, self.gb = T.grad(self.cost, [self.w, self.b])
        updates = [(self.w, self.w - self.gw), (self.b, self.b - self.gb)]
        self.trainFunc = function([self.x, self.y], self.cost, updates = updates)
        self.zero_one_loss = T.sum(T.neq(T.argmax(self.x), self.y))
        self.zero_one = function([self.x, self.y], self.zero_one_loss)

    def negativeLogLikelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(0, y.shape[0]), y])


    def train(self, batch_size = 600):









