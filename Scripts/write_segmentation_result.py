import numpy as np
import os.path
import json
import scipy
import scipy.io as scio
import argparse
import math
import pylab
from sklearn.preprocessing import normalize
caffe_root = '/home/SegNet/caffe-segnet/' 			# Change this to the absolute directoy to SegNet Caffe
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--iter', type=int, required=True)
parser.add_argument('--prefix',type=str,required=True)
args = parser.parse_args()

caffe.set_mode_cpu()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)


for i in range(0, args.iter):

	net.forward()

	image = net.blobs['data'].data
	label = net.blobs['label'].data
	predicted = net.blobs['prob'].data
	image = np.squeeze(image)
	output = np.squeeze(predicted)
        mat_0 = np.mat(output[0,:,:])
        mat_1 = np.mat(output[1,:,:])
        filename0 = args.prefix + '_iter_%d_0.mat' %(i)
        filename1 = args.prefix + '_iter_%d_1.mat' %(i)
        scio.savemat(filename0,{'mat_0':mat_0})
        scio.savemat(filename1,{'mat_1':mat_1})

print 'Success!'

