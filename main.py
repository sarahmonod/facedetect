import numpy as np
from PIL import Image, ImageDraw
import caffe

# converts from RGB to greyscale
def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def window(x, y, im):
    return im[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]

# parameters
WINDOW_SIZE = 36
DETECTION_THRESHOLD = 0.9
PROTOTXT = 'NET.prototxt'
CAFFE_MODEL = 'facenet_iter_200000.caffemodel'
INPUT_IMAGE = 'samples/images/faces.jpg'
DETECTED_OUTPUT_PATH = 'samples/detected/'

caffe.set_mode_cpu()
net = caffe.Net(PROTOTXT, CAFFE_MODEL, caffe.TEST)

image = Image.open(INPUT_IMAGE).convert('RGB')
im_array = np.array(image)
im_array = rgb2gray(im_array)

for x in xrange(0, len(im_array[0]) - WINDOW_SIZE, 2):
    for y in xrange(0, len(im_array) - WINDOW_SIZE, 2):
        window_array = window(x, y, im_array)
        im_input = window_array[np.newaxis, np.newaxis, :, :]
        net.blobs['data'].reshape(*im_input.shape)
        net.blobs['data'].data[...] = im_input
        out = net.forward()

        if out['loss'][0][1] > DETECTION_THRESHOLD:
            print 'face detected at', x, ';', y, ' (', out['loss'][0][1], ')'
            draw = ImageDraw.Draw(image)
            draw.rectangle((x, y, x + WINDOW_SIZE, y + WINDOW_SIZE), fill=None, outline="red")
            Image.fromarray(window_array).convert('RGB').save(DETECTED_OUTPUT_PATH + 'face_%d_%d.pgm' % (x, y))



image.show()
