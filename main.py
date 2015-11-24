import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

caffe.set_mode_cpu()
net = caffe.Net('../start_deep/facenet.prototxt', caffe.TEST)

im_array = np.array(Image.open('samples/images/faces.jpg'))
im_array = np.array(Image.fromarray(im_array, 'RGB'))
im_array = rgb2gray(im_array)

#Image.fromarray(im, 'RGB').save('faces.pgm')
#im = np.array(Image.open('1_0__t0,0_r0_s1.pgm'))

im_array = im_array[0:36, 0:36]
im_input = im_array[np.newaxis, np.newaxis, :, :]
net.blobs['data'].reshape(*im_input.shape)
net.blobs['data'].data[...] = im_input

out = net.forward()

#predicted predicted class
print out['prob'].argmax()

