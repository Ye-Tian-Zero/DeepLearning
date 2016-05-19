import cPickle as pickle
import theano.tensor as T
import numpy as np
import theano

def load_data(fold_num, path = "../data/ATIS/"):
    file_name = path + "atis.fold%d.pkl" % fold_num
    with open(file_name, 'r') as pkl_file:
        train_set, valid_set, test_set, dic = pickle.load(pkl_file)
    return train_set, valid_set, test_set, dic

class RNN(object):
    def __init__(self, nh, nc, ne, de, cs):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size
        '''

        self.emb = theano.shared(np.random.uniform(-1, 1,
                                                   (ne + 1, de)).astype(theano.config.floatX))

        self.W_input = theano.shared(np.random.uniform(-1, 1,
                                                  (de * cs, nh)).astype(theano.config.flaotX))

        self.W_out = theano.shared(np.random.uniform(-1, 1,
                                                         (nh, nc)).astype(theano.config.floatX))

        self.W_hidden = theano.shared(np.random.uniform(-1, 1,
                                                        (nh, nh)).astype(theano.config.floatX))

        self.b_input = theano.shared(np.zeros(nh, dtype=theano.config.floatX))

        self.b_output = theano.shared(np.zeros(nc, dtype=theano.config.floatX))

        self.h0 = theano.shared(np.zeros(nh, dtype=theano.config.floatX))

        self.params=[self.emb, self.W_input, self.W_out,
                     self.W_hidden, self.b_input, self.b_output, self.h0]

        self.names=['emb', 'W_input', 'W_out',
                    'W_hidden', 'b_input', 'b_output', 'h0']

        idxs = T.imatrix()
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        y = T.iscalar('y')

        def recurrence(input, h):
            h_out = T.nnet.sigmoid(T.dot(input, self.W_input) + T.dot(h, self.W_hidden) + self.b_input)
            softMax_out = T.nnet.softmax(T.dot(h_out, self.W_out) + self.b_output)
            return h_out, softMax_out

        [hidden, output], _= theano.scan(recurrence, sequences = x,
                                       outputs_info=[self.h0, None], n_steps=x.shape[0]) #Is n_step necessary?

        p_y_given_x_last_word = output[-1, 0, :]
        p_y_given_x_sentence = output[:, 0, :]

        y_pred = T.argmax(p_y_given_x_sentence, axis = 1)

        lr = T.scalar('lr')
        nll = -T.log(p_y_given_x_last_word)[y]
        gradients = T.grad(nll, self.params)
        updates = [(param, param - lr * g) for param, g in zip(self.params, gradients)]

        self.classify = theano.function(inputs = [idxs], outputs = y_pred)
        self.train = theano.function(inputs = [idxs, y, lr], outputs = nll,
                                     updates = updates)
        self.normalize = theano.function(inputs = [], updates = [(self.emb, self.emb / T.sqrt((self.emb ** 2).sum(axis = 1)).dimshuffle(0, 'x'))])

        def save(self, folder):
            for param, name in zip(self.params, self.names):
                np.save(os.path.join(folder, name+'.npy'), param.get_value())

