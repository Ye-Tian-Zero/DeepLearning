from __future__ import division
from theano.tensor.shared_randomstreams import RandomStreams
from MLP import load_data
from Plotlib import tile_raster_images
import cPickle as Pickle
import PIL.Image as Image
import theano
import timeit
import theano.tensor as T
import numpy as np


class RBM(object):
    def __init__(self,
                 input = None,
                 n_visible = 784,
                 n_hidden = 500,
                 W = None,
                 h_bias = None,
                 v_bias = None,
                 numpy_rng = None,
                 theano_rng = None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng is None:
            numpy_rng = np.random.RandomState(274563533)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            initial_W = np.array(
                numpy_rng.uniform(
                    low = -4 * np.sqrt(6. / (n_visible + n_hidden)),
                    high = 4 * np.sqrt(6. / (n_visible + n_hidden)),
                    size = (n_visible, n_hidden)
                ),
                dtype = theano.config.floatX
            )
            W = theano.shared(initial_W, name = 'W', borrow = True)

        if h_bias is None:
            initial_h_bias = np.array(np.zeros(n_hidden), dtype = theano.config.floatX)
            h_bias = theano.shared(initial_h_bias, name = 'h_bias', borrow = True)

        if v_bias is None:
            initial_v_bias = np.array(np.zeros(n_visible), dtype = theano.config.floatX)
            v_bias = theano.shared(initial_v_bias, name = 'v_bias', borrow = True)

        self.input = input
        if not input:
            self.input = T.matrix('input')
        self.W = W
        self.h_bias = h_bias
        self.v_bias = v_bias
        self.theano_rng = theano_rng
        self.numpy_rng = numpy_rng
        self.params = [self.W, self.h_bias, self.v_bias]

    def propUp(self, vis):
        pre_sigmoid_activation = T.dot(vis, self.W) + self.h_bias
        return pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)

    def sample_h_given_v(self, v0_sample):
        pre_sigmoid_h1, h1_prob = self.propUp(v0_sample)
        h1_sample = self.theano_rng.binomial(size = h1_prob.shape,
                                             n = 1, p = h1_prob,
                                             dtype = theano.config.floatX)
        return pre_sigmoid_h1, h1_prob, h1_sample

    def propDown(self, hid):
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.v_bias
        return pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)

    def sample_v_given_h(self, h0_sample):
        pre_sigmoid_v1, v1_prob = self.propDown(h0_sample)
        v1_sample = self.theano_rng.binomial(size = v1_prob.shape,
                                         n =1, p = v1_prob,
                                         dtype = theano.config.floatX)
        return pre_sigmoid_v1, v1_prob, v1_sample

    def gibbsHVH(self, h0_sample):
        pre_sigmoid_v1, v1_prob, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_prob, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_prob, v1_sample,
                pre_sigmoid_h1, h1_prob, h1_sample]

    def gibbsVHV(self, v0_sample):
        pre_sigmoid_h1, h1_prob, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_prob, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_prob, h1_sample,
                pre_sigmoid_v1, v1_prob, v1_sample]

    def freeEnergy(self, v_in) :
        wx_b = T.dot(v_in, self.W) + self.h_bias
        bv = T.dot(v_in, self.v_bias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis = 1)
        return -bv - hidden_term

    def getCostUpdate(self, learning_rate = 0.1, persistent = None, k = 1): #one step Gibbs Sample
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)
        if persistent == None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        ([pre_sigmoid_nvs, nv_probs, nv_samples,
          pre_sigmoid_nhs, nh_probs, nh_samples], updates) = theano.scan(
            self.gibbsHVH,
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps = k,
            name = 'gibbsHVH'
        )

        chain_end = nv_samples[-1]

        cost = T.mean(self.freeEnergy(self.input)) - T.mean(self.freeEnergy(chain_end))

        g_params = T.grad(cost, self.params, consider_constant=[chain_end])

        for g_param, param in zip(g_params, self.params):
            updates[param] = param - g_param * T.cast(learning_rate, theano.config.floatX)

        if persistent:
            updates[persistent] = nh_samples[-1]
            monitor_cost = self.getPseudoLikelihoodCost(updates)
        else:
            monitor_cost = self.getReconstructionCost(pre_sigmoid_nvs[-1])

        return monitor_cost, updates

    def getPseudoLikelihoodCost(self, updates):
        bit_i_idx = theano.shared(0, name='bit_i_idx')

        xi = T.round(self.input)

        fe_xi = self.freeEnergy(xi)

        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        fe_xi_flip = self.freeEnergy(xi_flip)

        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))

    def getReconstructionCost(self, pre_sigmoid_v):
        return T.mean(T.sum(self.input * T.log(T.nnet.sigmoid(pre_sigmoid_v))+
                      (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_v)), axis=1))


def trainRBM(train_epochs = 15, batch_size = 20, learning_rate = 0.1, n_hidden = 500, n_samples = 10, n_chains = 20):
    train_set_x, train_set_y, \
    validation_set_x, validation_set_y, \
    test_set_x, test_set_y = load_data()
    n_train_batches = train_set_x.get_value().shape[0] // batch_size

    index = T.lscalar('index')
    x = T.matrix('x')
    rbm = RBM(x,n_hidden=n_hidden)

    persistent_chain = theano.shared(np.zeros(shape=(batch_size, n_hidden), dtype = theano.config.floatX), borrow = True)

    cost, updates = rbm.getCostUpdate(learning_rate = learning_rate, persistent = persistent_chain, k = 15)


    train_rbm = theano.function(
        [index],
        cost,
        updates = updates,
        givens={x: train_set_x[index * batch_size : (index + 1) * batch_size]},
        name = 'train_rbm'
    )

    start_time = timeit.default_timer()

    for epoch in range(train_epochs):
        mean_cost = []
        for mini_batch_index in range(n_train_batches):
            mean_cost += [train_rbm(mini_batch_index)]

        print 'Train epoch %d, mean cost: ' % epoch, np.mean(mean_cost)

        image = Image.fromarray(
            tile_raster_images(
                X=rbm.W.get_value(borrow=True).T,
                img_shape = (28, 28),
                tile_shape = (10, 10),
                tile_spacing = (1, 1)
            )
        )
        image.save('.\\RBM_IMG_W\\filterrs_at_epoch_%i.png' % epoch)

    with open("RBM_params.pkl", 'w') as param_file:
        Pickle.dump(rbm.params, param_file)

    end_time = timeit.default_timer()
    pretraining_time = (end_time - start_time)
    print ('Training took %f minutes', pretraining_time / 60.)

    #test_idx = np.random.randint(numb)
    persistent_vis_chain = theano.shared(
        np.random.rand(20, 784).astype(theano.config.floatX)
    )
    plot_every = 1000
    ([presig_hids, hid_mfs, hid_samples,
      presig_vis, vis_mfs, vis_samples],
     updates) = theano.scan(
        rbm.gibbsVHV,
        outputs_info=[None, None, None, None, None, persistent_vis_chain],
        n_steps=plot_every,
        name = 'gibbs_vhv'
    )
    updates[persistent_vis_chain] = vis_samples[-1]
    sample_fn = theano.function(
        [],
        [
            vis_mfs[-1],
            vis_samples[-1]
        ],
        updates = updates,
        name = 'sample_fn'
    )

    image_data = np.zeros(
        (29 * n_samples + 1, 29 * n_chains - 1),
        dtype = 'uint8'
    )

    for idx in range(n_samples):
        vis_mf, vis_sample = sample_fn()
        print '..plotting sample %d' % idx
        image_data[29*idx:29*idx+28,:]=tile_raster_images(
            X=vis_mf,
            img_shape=(28,28),
            tile_shape=(1,n_chains),
            tile_spacing=(1,1)
        )
    image = Image.fromarray(image_data)
    image.save(".\\RBM_IMG_W\\samples.png")


def testRBM(pickle_file, n_chains = 20, n_samples = 10):
    with open(pickle_file, 'r') as param_file:
        params = Pickle.load(param_file)

    rbm = RBM(W = theano.shared(params[0].get_value()),
              h_bias = theano.shared(params[1].get_value()),
              v_bias = theano.shared(params[2].get_value()))

    persistent_vis_chain = theano.shared(
        np.zeros(shape=(20, 784), dtype='float32')
    )
    plot_every = 5000
    ([presig_hids, hid_mfs, hid_samples,
      presig_vis, vis_mfs, vis_samples],
     updates) = theano.scan(
        rbm.gibbsVHV,
        outputs_info=[None, None, None, None, None, persistent_vis_chain],
        n_steps=plot_every,
        name = 'gibbs_vhv'
    )
    updates[persistent_vis_chain] = vis_samples[-1]
    sample_fn = theano.function(
        [],
        [
            vis_mfs[-1],
            vis_samples[-1]
        ],
        updates = updates,
        name = 'sample_fn'
    )

    image_data = np.zeros(
        (29 * n_samples + 1, 29 * n_chains - 1),
        dtype = 'uint8'
    )

    for idx in range(n_samples):
        vis_mf, vis_sample = sample_fn()
        print '..plotting sample %d' % idx
        image_data[29*idx:29*idx+28,:]=tile_raster_images(
            X=vis_mf,
            img_shape=(28,28),
            tile_shape=(1,n_chains),
            tile_spacing=(1,1)
        )
    image = Image.fromarray(image_data)
    image.save(".\\RBM_IMG_W\\samples_test.png")



if __name__ == "__main__":
    trainRBM(batch_size=10, learning_rate=0.1, train_epochs=20)
    testRBM(".\\RBM_params.pkl", n_samples=20)

