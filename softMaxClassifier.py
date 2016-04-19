# -*- coding:utf-8 -*-
import gzip
import cPickle as pickle
import theano.tensor as T
import theano
import timeit
import numpy as np
from theano import function
import matplotlib.pyplot as plt

def load_data(dataSet = '.\\data\\mnist.pkl.gz'):
    with gzip.open(dataSet, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)

    def shared_dataSet(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.array(data_x, dtype=theano.config.floatX), borrow = borrow)
        shared_y = theano.shared(np.array(data_y, dtype=theano.config.floatX), borrow = borrow)
        return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_dataSet(train_set)
    valid_set_x, valid_set_y = shared_dataSet(valid_set)
    test_set_x, test_set_y = shared_dataSet(test_set)
    rval = [train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y]
    return rval

class softMaxClassifier(object):
    x = T.matrix('input')
    y = T.ivector('output')
    def __init__(self, n_out, train_x, train_y, validation_x, validation_y, test_x = None, test_y = None,
                 max_iter_times = 1000, rate = 0.13):
        self.rng = np.random
        self.test_x = test_x
        self.test_y = test_y
        self.train_x = train_x
        self.train_y = train_y
        self.validation_x = validation_x
        self.validation_y = validation_y
        self.max_iter_times = max_iter_times
        self.n_in = train_x.get_value().shape[1]
        self.n_out = n_out
        print self.n_in, self.n_out
        self.w = theano.shared(np.zeros((self.n_in, self.n_out),dtype=theano.config.floatX), borrow = True)
        self.b = theano.shared(np.zeros((self.n_out, ), dtype=theano.config.floatX), borrow = True)
        self.p_y_given_x = T.nnet.softmax(T.dot(self.x, self.w) + self.b)
        self.y_predict = T.argmax(self.p_y_given_x, axis = 1)
        self.cost = self.negativeLogLikelihood(self.y) #+ 0.01 * (self.w ** 2).sum() #加正则项效果反而变差
        self.gw, self.gb = T.grad(self.cost, [self.w, self.b])
        self.updates = [(self.w, self.w - rate * self.gw), (self.b, self.b - rate * self.gb)]
        self.zero_one_loss = T.mean(T.neq(self.y_predict, self.y))
        self.predict = function([self.x], self.y_predict)

    def negativeLogLikelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(0, y.shape[0]), y])


    def train(self, batch_size = 500):
        n_train_batches = self.train_x.get_value(borrow=True).shape[0] // batch_size
        n_validation_batches = self.validation_x.get_value(borrow=True).shape[0] // batch_size
        index = T.lscalar('index')

        train_model = function([index], self.cost,
                               givens={self.x : self.train_x[index * batch_size: (index + 1) * batch_size],
                                        self.y : self.train_y[index * batch_size: (index + 1) * batch_size]},
                               updates = self.updates)

        validation_model = function([index], self.zero_one_loss,
                                givens={self.x : self.validation_x[index * batch_size: (index + 1) * batch_size],
                                         self.y : self.validation_y[index * batch_size: (index + 1) * batch_size]})

        test_model = function([], self.zero_one_loss,
                              givens={self.x: self.test_x,
                                      self.y: self.test_y})

        print "start training"

        best_validation_loss = np.inf
        patience = 5000
        patience_increase = 2

        improvement_threshold = 0.995
        validation_frequency = min(patience//2, n_train_batches)
        start_time = timeit.default_timer()
        done_looping = False
        epoch = 0
        error_list = []
        while(epoch < self.max_iter_times and not done_looping):
            epoch += 1
            for miniBatch_index in range(n_train_batches):
                miniBatch_avg_cost = train_model(miniBatch_index)
                iter = (epoch - 1) * n_train_batches + miniBatch_index

                if(iter + 1) % validation_frequency == 0:
                    validation_losses = [validation_model(i) for i in range(n_validation_batches)]
                    this_validation_loss = np.mean(validation_losses)
                    error_list.append(this_validation_loss)

                    print 'epoch %i, minibatch %i/%i, validation error %f %%' % ( epoch, miniBatch_index + 1,
                                                                                  n_train_batches, this_validation_loss * 100. )
                    if this_validation_loss < best_validation_loss:
                        if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                            patience = max(patience, iter * patience_increase)
                        best_validation_loss = this_validation_loss

                if patience <= iter:
                    done_looping = True
                    break

        if self.test_x and self.test_y:
            test_error = test_model()
            print 'test error: %f %%' % (test_error * 100, )

        plt.plot(range(len(error_list)), error_list, 'r-')
        plt.show()
        end_time = timeit.default_timer()


if __name__ == '__main__':
    train_set, train_label, validation_set, validation_label, test_set, test_label = load_data()
    sr = softMaxClassifier(10, train_set, train_label, validation_set, validation_label, test_set, test_label, rate=0.2)
    sr.train()
