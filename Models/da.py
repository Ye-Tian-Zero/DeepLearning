import os
import theano
import theano.tensor as T
import numpy as np
import timeit
from PIL import Image
from theano import function
from softMaxClassifier import load_data
from theano.tensor.shared_randomstreams import RandomStreams
from theano import shared
from Plotlib import tile_raster_images

class DA(object):
    def __init__(self,
                 input,
                 np_rng,
                 theano_rng = None,
                 n_visible = 784,
                 n_hidden = 500,
                 W = None,
                 b_vis = None,
                 b_hid = None):

        if theano_rng is None:
            theano_rng = RandomStreams(np_rng.randint(2**30))

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if W is None:
            W = shared(np.array(np_rng.uniform(low = -4 * (6.0 / (n_hidden + n_visible)),
                                               high = 4 * (6.0 / (n_hidden+n_visible)),
                                               size=(n_visible, n_hidden)),
                                dtype=theano.config.floatX),
                       name = 'W',
                       borrow=True)

        if b_vis is None:
            b_vis = shared(np.zeros(n_visible, dtype=theano.config.floatX), borrow=True)
        if b_hid is None:
            b_hid = shared(np.zeros(n_hidden, dtype=theano.config.floatX), borrow=True)

        self.W = W
        self.W_prime = self.W.T
        self.b = b_hid
        self.b_prime = b_vis
        self.theano_rng = theano_rng

        if input is None:
            self.x = T.matrix('x')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def getHiddenValues(self, input):
        return T.nnet.relu(T.dot(input, self.W) + self.b)

    def getReconstructResult(self, hidden):
        return T.nnet.relu(T.dot(hidden, self.W_prime) + self.b_prime)

    def getCorruptionInput(self, input, corruption_level):
        return self.theano_rng.binomial(size = input.shape, n = 1,
                                        p = 1 - corruption_level,
                                        dtype = theano.config.floatX) * input

    def getCostUpdates(self, corruption_level, learning_rate = 0.1):
        corrupted_x = self.getCorruptionInput(self.x, corruption_level)

        hidden_output = self.getHiddenValues(corrupted_x)
        reconstruct_result = self.getReconstructResult(hidden_output)

        #L = -T.sum(self.x * T.log(reconstruct_result) + (1 - self.x) * T.log( 1 - reconstruct_result), axis = 1)
        L = T.sqrt(T.sum((self.x - reconstruct_result) ** 2, axis = 1))
        cost = T.mean(L)

        param_grads = T.grad(cost, self.params)

        updates = [(param, param - learning_rate * param_g) for param, param_g in zip(self.params, param_grads)]

        return (cost, updates)

def testDA(batch_size = 20, train_epochs = 200):
    train_set_x, train_set_y, \
    validation_set_x, validation_set_y,\
    test_set_x, test_set_y = load_data()

    n_train_batches = train_set_x.get_value().shape[0] // batch_size

    x = T.matrix('x')
    index = T.lscalar('index')
    da = DA(input = x,
            np_rng = np.random,
            n_visible = 28*28,
            n_hidden = 500
            )

    cost, updates = da.getCostUpdates(0.3)

    train_da = function(
        [index],
        cost,
        updates = updates,
        givens={
            x: train_set_x[index * batch_size : (index + 1) * batch_size]
        }
    )

    start_time = timeit.default_timer()

    ############
    # TRAINING #
    ############

    for epoch in range(train_epochs):
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))

        print('Training epoch %d, cost' % epoch, np.mean(c))

        image = Image.fromarray(tile_raster_images(
            X=da.W.get_value(borrow=True).T,
            img_shape=(28, 28), tile_shape=(10, 10),
            tile_spacing=(1, 1)))
        image.save('.\\DA_IMG_relu\\filters_corruption_30_epoch_%d.png' % epoch)

    end_time = timeit.default_timer()

    training_time = end_time - start_time

    print('The 30% corruption code for file' + os.path.split(__file__)[1] +
           'ran for %.2fm' % (training_time / 60.))

if __name__ == "__main__":
    testDA()