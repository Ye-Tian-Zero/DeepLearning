import numpy as np
import theano.tensor as T
import timeit
import cPickle as pickle
import theano
from theano.tensor.shared_randomstreams import RandomStreams
from MLP import HiddenLayer
from softMaxClassifier import  softMaxClassifier, load_data
from RBM import RBM

class DBN(object):
    def __init__(self, np_rng = None, theano_rng = None,
                 n_hidden_size = None, n_ins = 784, n_outs = 10):
        if np_rng is None:
            np_rng = np.random.RandomState(274573)

        if theano_rng is None:
            theano_rng = RandomStreams(np_rng.randint(2 ** 30))

        if n_hidden_size is None:
            n_hidden_size = [500, 500]

        self.sigmoid_layers = []
        self.RBM_layers = []
        self.params = []
        self.n_layers = len(n_hidden_size)

        assert self.n_layers > 0

        self.x = T.matrix('x')
        self.y = T.ivector('y')

        for i in range(self.n_layers):

            if i is 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[i - 1].output

            if i is 0:
                input_size = n_ins
            else:
                input_size = n_hidden_size[i - 1]

            self.sigmoid_layers.append(HiddenLayer(n_in = input_size,
                                                   n_out = n_hidden_size[i],
                                                   input = layer_input,
                                                   activation = T.nnet.sigmoid))

            self.RBM_layers.append(RBM(input = layer_input,
                                       n_visible = input_size,
                                       n_hidden = n_hidden_size[i],
                                       W = self.sigmoid_layers[i].W,
                                       h_bias= self.sigmoid_layers[i].b))

            self.params.extend(self.sigmoid_layers[i].params)

        self.softMax_layer = softMaxClassifier(input = self.sigmoid_layers[ - 1].output,
                                          n_in = n_hidden_size[ - 1],
                                          n_out = n_outs)

        self.params.extend(self.softMax_layer.params)
        self.fineTune_cost = self.softMax_layer.negativeLogLikelihood(self.y)
        self.error = self.softMax_layer.zero_one_loss(self.y)

    def buildFineTuneFunctions(self, train_data, batch_size = 500, learning_rate = 0.1):
        train_x, train_y, \
        validation_x, validation_y, \
        test_x, test_y = train_data

        n_test_batches = test_x.get_value().shape[0] // batch_size
        n_validation_batches = validation_x.get_value().shape[0] // batch_size

        index = T.lscalar('index')
        g_params = T.grad(self.fineTune_cost, self.params)
        updates = []
        for param, g_param in zip(self.params, g_params):
            updates.append((param, param - learning_rate * g_param))

        testFunction = theano.function([index], self.error,
                                       givens = {self.x:test_x[index * batch_size : (index + 1) * batch_size],
                                               self.y:test_y[index * batch_size : (index + 1) * batch_size]})

        validationFunction = theano.function([index], self.error,
                                             givens = {self.x:validation_x[index * batch_size : (index + 1) * batch_size],
                                                     self.y:validation_y[index * batch_size : (index + 1) * batch_size]})

        trainFunction = theano.function([index], self.fineTune_cost,
                                        givens = {self.x:train_x[index * batch_size : (index + 1) * batch_size],
                                                self.y:train_y[index * batch_size : (index + 1) * batch_size]},
                                        updates = updates)

        def testScore():
            return np.mean([testFunction(i) for i in range(n_test_batches)])

        def validationScore():
            return np.mean([validationFunction(i) for i in range(n_validation_batches)])

        return trainFunction, validationScore, testScore

    def preTrainFunction(self, train_x, learning_rate = 0.1, batch_size = 20, step = 15):
        index = T.lscalar('index')

        preTrain_lst = []
        for rbm in self.RBM_layers:
            persistent_chain = theano.shared(np.zeros(shape = (batch_size, rbm.n_hidden),
                                                      dtype = theano.config.floatX),
                                             borrow = True)
            cost, updates= rbm.getCostUpdate(learning_rate= learning_rate, persistent = persistent_chain,
                                             k = step)

            preTrain_func = theano.function([index], cost, updates = updates,
                                            givens = {self.x:train_x[index * batch_size : (index + 1) * batch_size]})

            preTrain_lst.append(preTrain_func)

        return preTrain_lst

def trainDBN(learning_rate = 0.1, preTrain_learning_rate = 0.01, batch_size = 10,
             preTrain_epochs = 100, step = 1, train_epochs = 1000, ):
    dbn = DBN(n_hidden_size=[1000,500,500])
    train_x, train_y, \
    validation_x, validation_y, \
    test_x, test_y = load_data()

    n_train_batches = train_x.get_value().shape[0] // batch_size

    train_data = [train_x, train_y, validation_x, validation_y, test_x, test_y]
    trainFunc, validationScore, testScore = dbn.buildFineTuneFunctions(batch_size = batch_size,
                                                                       learning_rate = learning_rate,
                                                                       train_data = train_data)
    preTrain_lst = dbn.preTrainFunction(train_x, learning_rate = preTrain_learning_rate,
                                        batch_size = batch_size, step = step)
    for (preTrain_index, preTrainFunc) in enumerate(preTrain_lst):
        for epoch in range(preTrain_epochs):
            mean_cost = []
            for batch_index in range(n_train_batches):
                mean_cost.append(preTrainFunc(batch_index))
            print "mean_cost in epoch %d : %f, pre-training level %d" % (epoch, np.mean(mean_cost), preTrain_index + 1)

    with open('dbnPreTrainParams.pkl', 'w') as params_pkl:
        pickle.dump(dbn.params, params_pkl)

    print "fineTuning..."

    patience = 10000
    patience_increase = 2

    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = np.Inf
    test_score = 0
    epoch = 0
    done_looping = False
    while epoch < train_epochs and not done_looping:
        epoch += 1
        for miniBatch_index in range(n_train_batches):
            miniBatch_avg_cost = trainFunc(miniBatch_index)
            iter = (epoch - 1) * n_train_batches + miniBatch_index
            if (iter + 1) % validation_frequency == 0:
                this_validation_loss = validationScore()
                print 'epoch %i, minibatch %i/%i, validation error: %f %%' %\
                      (epoch, miniBatch_index + 1, n_train_batches, this_validation_loss * 100)

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    best_validation_loss = this_validation_loss

                    test_score = testScore()
                    print 'epoch %i, minibatch %i/%i, test error: %f %%' %\
                           (epoch, miniBatch_index + 1, n_train_batches, test_score * 100)
            if patience < iter:
                done_looping = True
                break

    print 'Optimization complete, best validation score is %f %%' \
          ' with test performance %f %%' % (best_validation_loss * 100, test_score * 100)





if __name__ == "__main__":
    trainDBN()


