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


def merge_windows_list(window_list, max_distance):
    merged = False
    merged_list = []
    already_merged_list = []

    for i, w1 in enumerate(window_list):
        for j, w2 in enumerate(window_list[i+1:]):
            if w1 not in already_merged_list and w2 not in already_merged_list and windows_distance(w1, w2) < max_distance :
                merged_list.append(merge_windows(w1, w2))
                already_merged_list.append(w1)
                already_merged_list.append(w2)
                merged = True
                print 'merged windows', w1, w2

    if merged:
        not_merged_list = [item for item in window_list if item not in already_merged_list]
        return merge_windows_list(merged_list + not_merged_list, max_distance)
    else:
        return window_list


def show_detected(detected_windows, image, color):
    for w in detected_windows:
        draw = ImageDraw.Draw(image)
        draw.rectangle(w, fill=None, outline=color)
    image.show()

# parameters
WINDOW_SIZE = 36
DETECTION_THRESHOLD = 0.8
MAX_DISTANCE = 10
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

#show detection before merging windows
show_detected(detected, image, "red")
#show detection after merging windows
show_detected(merge_windows_list(detected, MAX_DISTANCE), image2, "blue")