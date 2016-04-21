from __future__ import division
from MLP import MLP
from theano import function
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
from softMaxClassifier import load_data
import numpy as np
import theano.tensor as T
import theano

class LeNetConvPoolLayer(object):
    def __init__(self, rng, input, layer_shape, input_shape, pool_size = (2,2)):
        '''
        :param rng: random number generator
        :param input: 4D tensor with shape of input_shape
        :param layer_shape: 4D matrix, out_put_num * input_num * kernel_rows * kernel_cols
        :param input_shape: 4D matrix, batch_size * input_num * image_rows * image_cols
        :param pool_size: pool_size
        :return: Nothing
        '''
        assert input_shape[1] == layer_shape[1]
        self.input = input

        fan_in = np.prod(layer_shape[1:])
        fan_out = (layer_shape[0] * np.prod(layer_shape[2:])) // np.prod(pool_size)

        W_bound = np.sqrt(6.0 / (fan_out + fan_in))

        self.W = theano.shared(np.array(rng.uniform(low = - W_bound, high= W_bound, size = layer_shape), dtype = theano.config.floatX),
                                borrow=True)

        self.b = theano.shared(np.zeros(shape = (layer_shape[0], ), dtype = theano.config.floatX), borrow = True)

        convolution_out = conv2d(input, self.W, filter_shape = layer_shape, input_shape = input_shape) #what will happen if I delete the last two parameters
        pool_out = downsample.max_pool_2d(convolution_out, pool_size, ignore_border = True)
        self.output = T.nnet.relu(pool_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]

def trainLeNet(train_x, train_y, validation_x, validation_y, test_x, test_y,
               convolution_layer_size = None, rate = 0.1, batch_size = 500, n_epochs = 200):
    rng = np.random.RandomState(274563533)
    x = T.matrix('x')
    y = T.ivector('y')
    layer_0_input = x.reshape((batch_size, 1, 28, 28))
    layer_0 = LeNetConvPoolLayer(rng, input = layer_0_input,
                                 layer_shape = (convolution_layer_size[0], 1, 5, 5),
                                 input_shape = (batch_size, 1, 28, 28),
                                 pool_size = (2,2))
    layer_1 = LeNetConvPoolLayer(rng, input = layer_0.output,
                                 layer_shape = (convolution_layer_size[1], convolution_layer_size[0], 5, 5),
                                 input_shape = (batch_size, convolution_layer_size[0], 12, 12),
                                 pool_size = (2,2))

    MLP_input = layer_1.output.flatten(2)
    layer_final = MLP(MLP_input, convolution_layer_size[1] * 4 * 4, 500, 10)

    cost = layer_final.negativeLogLikelihood(y)
    error = layer_final.errors(y)

    index = T.lscalar('index')

    validation_model = function([index], error,
                                givens={x: validation_x[index * batch_size : (index + 1) * batch_size],
                                        y: validation_y[index * batch_size : (index + 1) * batch_size]})

    test_model = function([index], error,
                          givens={x: test_x[index * batch_size : (index + 1) * batch_size],
                                  y: test_y[index * batch_size : (index + 1) * batch_size]})

    params = layer_final.params + layer_1.params + layer_0.params

    param_grad = T.grad(cost, params)
    updates = [(p, p - rate * pg) for p, pg in zip(params, param_grad)]

    train_model = function([index], cost,
                           givens={x:train_x[index * batch_size : (index + 1) * batch_size],
                                   y:train_y[index * batch_size : (index + 1) * batch_size]},
                           updates = updates)

    n_train_batches = train_x.get_value().shape[0] // batch_size
    n_test_batches = test_x.get_value().shape[0] // batch_size
    n_validation_batches = validation_x.get_value().shape[0] // batch_size

    epoch = 0
    best_validation_cost = np.Inf
    patience = 10000
    improvement_thread = 0.995
    patience_increase = 2
    validation_frequency = min(n_train_batches, patience / 2)
    loop_done = False
    while epoch <= n_epochs and not loop_done:
        epoch += 1
        for minibatch_index in range(n_train_batches):
            batch_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter

            if(iter + 1) % validation_frequency == 0:
                validation_losses = [validation_model(i) for i in range(n_validation_batches)]
                this_validation_loss = np.mean(validation_losses)
                print 'epoch %i, minibatch %i / %i, validation error %f %%' \
                      % (epoch, minibatch_index+1, n_train_batches, this_validation_loss * 100)
                if this_validation_loss < best_validation_cost:
                    if this_validation_loss < best_validation_cost * improvement_thread:
                        patience = max(patience, iter * patience_increase)
                    best_validation_cost = this_validation_loss
                    test_losses = [test_model(i) for i in range(n_test_batches)]#lkfanldnfaklfnklasnfklasnklfnalksdfnkl
                    test_loss = np.mean(test_losses)
                    print 'test error: %f %%'%(test_loss * 100)
            if patience <= iter:
                loop_done = True
                break





if __name__ == '__main__':
    train_set, train_label, validation_set, validation_label, test_set, test_label = load_data()
    trainLeNet(train_set, train_label, validation_set, validation_label, test_set, test_label, [20, 50])
