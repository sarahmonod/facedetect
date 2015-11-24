import numpy as np
from PIL import Image
import caffe

# converts from RGB to greyscale
def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def window(x, y, im):
    return im[x:x+WINDOW_SIZE, y:y+WINDOW_SIZE]

# parameters
WINDOW_SIZE = 36
DETECTION_THRESHOLD = 0.8
PROTOTXT = 'NET.prototxt'
CAFFE_MODEL = 'facenet_iter_200000.caffemodel'
INPUT_IMAGE = 'samples/images/faces.jpg'
DETECTED_OUTPUT_PATH = 'samples/detected/'

caffe.set_mode_cpu()
net = caffe.Net(PROTOTXT, CAFFE_MODEL, caffe.TEST)

im_array = np.array(Image.open(INPUT_IMAGE).convert('RGB'))
im_array = rgb2gray(im_array)

# TODO check if bounds are correct (getting execution error at the end...)
for x in range(0, im_array.size - WINDOW_SIZE):
    for y in range(0, im_array[0].size - WINDOW_SIZE):
        window_array = window(x, y, im_array)
        im_input = window_array[np.newaxis, np.newaxis, :, :]
        net.blobs['data'].reshape(*im_input.shape)
        net.blobs['data'].data[...] = im_input
        out = net.forward()

        if out['loss'][0][1] > DETECTION_THRESHOLD:
            print 'face detected at', x, ';', y, ' (', out['loss'][0][1], ')'
            Image.fromarray(window_array).convert('RGB').save(DETECTED_OUTPUT_PATH + 'face_%d_%d.pgm' % (x, y))






