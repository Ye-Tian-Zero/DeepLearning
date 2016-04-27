from cv2 import *
from Models.softMaxClassifier import load_data
from Models.MLP import MLP
import numpy as np
import Models.LeNet as LN
import theano
import theano.tensor as T
import cPickle as pickle

class LeNet(object):
    x = T.matrix('x')
    def __init__(self, convolution_layer_size = None):
        rng = np.random.RandomState(274563533)
        with open('.\\Models\\LeNet_params.pkl') as serial:
            params = pickle.load(serial)
        layer_final_params = params[:4]
        layer_1_params = params[4:6]
        layer_0_params = params[6:8]
        layer_0_input = self.x.reshape((1, 1, 28, 28))
        self.layer_0 = LN.LeNetConvPoolLayer(rng, input = layer_0_input,
                                        layer_shape = (convolution_layer_size[0], 1, 5, 5),
                                        input_shape = (1, 1, 28, 28),
                                        pool_size = (2,2))

        self.layer_1 = LN.LeNetConvPoolLayer(rng, input = self.layer_0.output,
                                        layer_shape = (convolution_layer_size[1], convolution_layer_size[0], 5, 5),
                                        input_shape = (1, convolution_layer_size[0], 12, 12),
                                        pool_size = (2,2))

        MLP_input = self.layer_1.output.flatten(2)

        self.layer_final = MLP(MLP_input, convolution_layer_size[1] * 4 * 4, 500, 10)
        self.layer_0.W.set_value(layer_0_params[0].get_value(), borrow= True)
        self.layer_0.b.set_value(layer_0_params[1].get_value(), borrow= True)
        self.layer_1.W.set_value(layer_1_params[0].get_value(), borrow= True)
        self.layer_1.b.set_value(layer_1_params[1].get_value(), borrow= True)
        self.layer_final.hiddenLayer.W.set_value(layer_final_params[0].get_value(), borrow= True)
        self.layer_final.hiddenLayer.b.set_value(layer_final_params[1].get_value(), borrow= True)
        self.layer_final.outputLayer.W.set_value(layer_final_params[2].get_value(), borrow= True)
        self.layer_final.outputLayer.b.set_value(layer_final_params[3].get_value(), borrow= True)
        self.predict = theano.function([self.x], self.layer_final.predict)

       # self.get_feature = theano.function([self.x], self.layer_1.output)
        #()!@J#()!@J#(J#

if __name__ == '__main__':
    leNet = LeNet((20, 50))
    img = imread('.\\12345.jpg')
    img = 255 - img
    img_gray = cvtColor(img, COLOR_BGR2GRAY)
    img_gray = threshold(img_gray, 50, 255, THRESH_BINARY)[1]
    img_gray = img_gray.astype('float32') / 255.0
    split_line = []
    cut_status = False

    for col in range(img_gray.shape[1]):
        col_sum = img_gray[:, col].sum()
        if col_sum != 0 and cut_status is False:
            split_line.append(col)
            cut_status = True
        elif col_sum == 0 and cut_status is True:
            split_line.append(col)
            cut_status = False

    splited_image = [img_gray[:,i:j] for i, j in zip(split_line[::2], split_line[1::2])]

    for (i, s_img) in enumerate(splited_image):
        start = 0
        end = s_img.shape[0]
        row_sum = s_img.sum(axis = 1)
        while start != end:
            if row_sum[start] != 0:
                break
            start += 1

        while end != 0:
            end -= 1
            if row_sum[end] != 0:
                break

        splited_image[i] = s_img[start:end,:]

    for i, s_img in enumerate(splited_image):
        max_axis = np.maximum(*s_img.shape)
        new_img = np.zeros((max_axis, max_axis), dtype='float32')
        s_r = s_img.shape[0]
        s_c = s_img.shape[1]
        #print max_axis -s_c
        new_img[(max_axis - s_r) / 2 : s_img.shape[0] + (max_axis - s_r) / 2, (max_axis - s_c) / 2 : s_img.shape[1] + (max_axis - s_c) / 2] = s_img
        new_img = resize(new_img, (20,20), interpolation = INTER_AREA)
        result = np.zeros((28, 28), dtype = 'float32')
        result[4:24, 4:24] = new_img
        splited_image[i] = result.reshape(1, 28*28)

    result = 0
    for s_img in splited_image:
        result = result * 10 + leNet.predict(s_img)[0]

    print result


