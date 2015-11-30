import numpy as np
from PIL import Image, ImageDraw
import caffe
import scipy.ndimage


# converts from RGB to greyscale
def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def window(x, y, im):
    return im[y:y + WINDOW_SIZE, x:x + WINDOW_SIZE]

def merge_windows(w1, w2):
    w1x1, w1y1, w1x2, w1y2 = w1
    w2x1, w2y1, w2x2, w2y2 = w2
    return min(w1x1, w2x1), min(w1y1, w2y1), max(w1x2, w2x2), max(w1y2, w2y2)

def windows_distance(w1,w2):
    [w1x1, w1y1, w1x2, w1y2] = w1
    [w2x1, w2y1, w2x2, w2y2] = w2
    w1center = np.array([(w1x1+w1x2) /2 ,  (w1y1+w1y2) /2])
    w2center = np.array([(w2x1+w2x2) /2 ,  (w2y1+w2y2) /2])
    return np.linalg.norm(w1center-w2center)


def merge_windows_list(wList, max_distance):
    merged = False
    merged_list = []

    for i, w1 in enumerate(detected):
        for j, w2 in enumerate(detected[i+1:]):
            if windows_distance(w1, w2) < max_distance:
                merged_list.append(merge_windows(w1, w2))
                print 'merged windows', w1, w2
                del detected[i]
                del detected[j]
                merged = True
                break

    if merged:
        return merge_windows_list(merged_list + wList, max_distance)
    else:
        return wList


# parameters
WINDOW_SIZE = 36
DETECTION_THRESHOLD = 0.8
PROTOTXT = 'NET.prototxt'
CAFFE_MODEL = 'facenet_iter_200000.caffemodel'
INPUT_IMAGE = 'samples/images/faces.jpg'
DETECTED_OUTPUT_PATH = 'samples/detected/'
ZOOM_RANGE = np.arange(0.1, 1.1, 0.1)

caffe.set_mode_cpu()
net = caffe.Net(PROTOTXT, CAFFE_MODEL, caffe.TEST)
detected = []

image = Image.open(INPUT_IMAGE).convert('RGB')
image2 = Image.open(INPUT_IMAGE).convert('RGB')
im_array_rgb = np.array(image)
for rzoom in ZOOM_RANGE:
    im_array = rgb2gray(scipy.misc.imresize(im_array_rgb, rzoom))
    print 'zoom : ', rzoom
    for x in xrange(0, len(im_array[0]) - WINDOW_SIZE, 2):
        for y in xrange(0, len(im_array) - WINDOW_SIZE, 2):
            window_array = window(x, y, im_array)
            im_input = window_array[np.newaxis, np.newaxis, :, :]
            net.blobs['data'].reshape(*im_input.shape)
            net.blobs['data'].data[...] = im_input
            out = net.forward()

            if out['loss'][0][1] > DETECTION_THRESHOLD:
                real_x = int(x * 1 / rzoom)
                real_y = int(y * 1 / rzoom)
                print 'face detected at', real_x, ';', real_y, ' (', out['loss'][0][1], ')'

                detected.append((real_x, real_y, real_x + int(WINDOW_SIZE * 1 / rzoom), real_y + int(WINDOW_SIZE * 1 / rzoom)))
                #Image.fromarray(window_array).convert('RGB').save(DETECTED_OUTPUT_PATH + 'face_%d_%d.pgm' % (real_x, real_y))

MAX_DISTANCE = 10
for w in detected:
    print w
    draw = ImageDraw.Draw(image)
    draw.rectangle(w, fill=None, outline="red")

# for w in merge_windows_list(detected, MAX_DISTANCE):
#     print w
#     draw = ImageDraw.Draw(image2)
#     draw.rectangle(w, fill=None, outline="red")

image.show()
