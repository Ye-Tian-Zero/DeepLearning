# -*- coding:utf-8 -*-
import timeit
import numpy as np
import theano
import theano.tensor as T
from theano import function
from softMaxClassifier import load_data

class softMaxClassifier(object):
    def __init__(self, train_x, train_y, n_out, n_in):
        self.rng = np.random
        self.n_in = n_in
        self.n_out = n_out
        self.W = theano.shared(np.zeros((self.n_in, self.n_out),dtype=theano.config.floatX), borrow = True)
        self.b = theano.shared(np.zeros((self.n_out, ), dtype=theano.config.floatX), borrow = True)

        self.p_y_given_x = T.nnet.softmax(T.dot(train_x, self.W) + self.b)
        self.y_predict = T.argmax(self.p_y_given_x, axis = 1)

        self.errors = T.mean(T.neq(self.y_predict, train_y))

        self.params = [self.W, self.b]

    def negativeLogLikelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(0, y.shape[0]), y])

class HiddenLayer(object):
    def __init__(self, n_in, n_out, input, W = None, b = None, activation = T.tanh):
        self.rng = np.random
        self.input = input
        if W is None:
            W_values = np.array(self.rng.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                                                 high=np.sqrt(6. / (n_in + n_out)),
                                                 size=(n_in, n_out)),
                                dtype=theano.config.floatX)
            if activation == T.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(W_values, borrow=True, name='W')
        if b is None:
            b_values = np.array(np.zeros(n_out, dtype=theano.config.floatX))
            b = theano.shared(b_values, borrow=True, name='b')

        self.W = W
        self.b = b

        lin_output = self.b + T.dot(self.input, self.W)
        self.output = (lin_output if activation is None
                       else activation(lin_output))

        self.params = [self.W, self.b]

class MLP(object):
    x = T.matrix('x')
    y = T.ivector('y')
    def __init__(self, n_in, n_hidden, n_out,
                 train_x, train_y,
                 validation_x, validation_y,
                 test_x = None, test_y = None,
                 max_iter_steps = 1000, rate = 0.01):
        self.train_x, self.train_y, \
        self.validation_x, self.validation_y, \
        self.test_x, self.test_y = train_x, train_y, validation_x, validation_y, test_x, test_y

        self.hiddenLayer = HiddenLayer(n_in = n_in,
                                       n_out = n_hidden,
                                       input = self.x,
                                       activation = T.nnet.relu)

        self.outputLayer = softMaxClassifier(n_in = n_hidden,
                                             n_out = n_out,
                                             train_x = self.hiddenLayer.output,
                                             train_y = self.y)

        self.errors = self.outputLayer.errors
        self.params = self.hiddenLayer.params + self.outputLayer.params
        self.negativeLogLikelihood = self.outputLayer.negativeLogLikelihood
        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.outputLayer.W).sum()
        self.L2_sqr = T.sum(self.hiddenLayer.W ** 2) + T.sum(self.outputLayer.W ** 2)

        self.rate = rate
        self.steps = max_iter_steps

    def train(self, batch_size = 20, L1_reg = 0.000, L2_reg = 0.0001):
        n_train_batches = self.train_x.get_value(borrow=True).shape[0] // batch_size
        n_validation_batches = self.validation_x.get_value(borrow=True).shape[0] // batch_size
        n_test_batches= self.test_x.get_value(borrow=True).shape[0] // batch_size

        print 'building model'
        start_time = timeit.default_timer()

        index = T.lscalar()

        cost = self.negativeLogLikelihood(self.y) + L1_reg * self.L1 + L2_reg * self.L2_sqr

        if self.test_x and self.test_y:
            test_model = function([], self.errors,
                                  givens={
                                      self.x: self.test_x,
                                      self.y: self.test_y
                                  })

        validation_model = function([index], self.errors,
                                    givens={
                                        self.x: self.validation_x[batch_size * index: batch_size * (index + 1)],
                                        self.y: self.validation_y[batch_size * index: batch_size * (index + 1)]
                                    })



        gparams = [T.grad(cost, param) for param in self.params]

        updates = [(param, param - self.rate * gp)
                  for param, gp in zip(self.params, gparams)]

        train_model = function([index], cost,
                               givens={
                                   self.x: self.train_x[batch_size * index: batch_size * (index + 1)],
                                   self.y: self.train_y[batch_size * index: batch_size * (index + 1)]
                               }, updates = updates)

        print 'training'

        patience = 10000
        patience_increase = 2

        improvement_threshold = 0.995
        validation_frequency = min(n_train_batches, patience // 2)

        best_validation_loss = np.Inf
        test_score = 0
        epoch = 0
        done_looping = False
        while epoch < self.steps and not done_looping:
            epoch += + 1
            for miniBatch_index in range(n_train_batches):
                miniBatch_avg_cost = train_model(miniBatch_index)
                iter = (epoch - 1) * n_train_batches + miniBatch_index
                if (iter + 1) % validation_frequency == 0:
                    validation_losses = [validation_model(i) for i in range(n_validation_batches)]
                    this_validation_loss = np.mean(validation_losses)
                    print 'epoch %i, minibatch %i/%i, validation error: %f %%' % \
                          (epoch, miniBatch_index + 1, n_train_batches, this_validation_loss * 100)

                    if this_validation_loss < best_validation_loss:
                        if this_validation_loss < best_validation_loss * improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss

                        if self.test_x and self.test_y:
                            test_score = test_model()
                            print 'epoch %i, minibatch %i/%i, test error: %f %%' % \
                                  (epoch, miniBatch_index + 1, n_train_batches, test_score * 100)

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print 'Optimization complete, best validation score is %f %%' \
              ' with test performance %f %%' % (best_validation_loss * 100, test_score * 100)

        print 'time cost: %.2fm'% ((end_time - start_time) / 60.0, )


if __name__ == '__main__':
    train_set, train_label, validation_set, validation_label, test_set, test_label = load_data()
    mlp = MLP(28*28, 500, 10, train_set, train_label, validation_set, validation_label, test_set, test_label)
    mlp.train()
