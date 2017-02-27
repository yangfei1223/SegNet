import numpy as np
import argparse

caffe_root = '/home/SegNet/caffe-segnet/'  # Change this to the absolute directoy to SegNet Caffe
import sys

sys.path.insert(0, caffe_root + 'python')

import caffe

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--iter', type=int, required=True)
parser.add_argument('--prefix', type=str, required=True)
args = parser.parse_args()

caffe.set_mode_cpu()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

for i in range(0, args.iter):
    print  'frameID = %d\n' %(i)
    net.forward()

    image = net.blobs['data'].data
    label = net.blobs['label'].data
    predicted = net.blobs['prob'].data
    image = np.squeeze(image)
    output = np.squeeze(predicted)
    out_0 = output[0, :, :]
    out_1 = output[1, :, :]
    filename_0 = args.prefix + '_iter_%d_0.bin' % (i)
    filename_1 = args.prefix + '_iter_%d_1.bin' % (i)
    out_0.tofile(filename_0)
    out_1.tofile(filename_1)
    

print 'Success!'
