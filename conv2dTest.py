import theano.tensor as T
import theano
from theano.tensor.nnet import conv2d
import pylab
from PIL import Image
import numpy as np

rng = np.random.RandomState(234553123)

input = T.tensor4(name = 'input')
w_shp = (3, 3, 7, 7)
w_bound = np.sqrt(3*7*7)
W = theano.shared(np.array(rng.uniform(low = 0, high = 1 / w_bound, size=w_shp), dtype = input.dtype))
b = theano.shared(np.array(rng.uniform(low = 0, high = 0.5, size = (3,)), dtype  = input.dtype), name = 'b')
conv_out = conv2d(input, W)
output = T.nnet.relu(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

f = theano.function([input], output)

img = Image.open('.\\data\\0.jpg')

img = np.array(img, dtype = 'float32') / 256

img_ = img.transpose(2, 0, 1).reshape(1, 3, img.shape[0], img.shape[1])

filtered_img = f(img_)

pylab.subplot(1, 4, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray()
# recall that the convOp output (filtered image) is actually a "minibatch",
# of size 1 here, so we take index 0 in the first dimension:
pylab.subplot(1, 4, 2); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
pylab.subplot(1, 4, 3); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])
pylab.subplot(1, 4, 4); pylab.axis('off'); pylab.imshow(filtered_img[0, 2, :, :])
pylab.show()