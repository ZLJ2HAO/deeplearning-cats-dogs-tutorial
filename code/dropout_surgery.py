import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
from PIL import Image
import cv2
import statistics
import time
import math


drop_thresh = 0.5   #(0~1)


compression_net = caffe.Net('../caffe_models/caffe_model_1/caffenet_deploy_drop05.prototxt','../caffe_models/caffe_model_1/snapshot/alexnet_drop05_5000.caffemodel',caffe.TEST)
nocompression_net = caffe.Net('../caffe_models/caffe_model_1/caffenet_deploy_1.prototxt','../caffe_models/caffe_model_1/original_model/keeptrain_iter_5000.caffemodel',caffe.TEST)


weight1 = nocompression_net.params['fc6'][0].data[...]
weight2 = nocompression_net.params['fc7'][0].data[...]
[original_layer_len,n] = weight1.shape
determined_layer_len = int(original_layer_len * drop_thresh)



print('one weight matrix shape is {}'.format(weight1.shape))
print('another weight matrix shape is {}'.format(weight2.shape))
print('Original target layer length is {}'.format(original_layer_len))
print('Determined layer length is {}'.format(determined_layer_len))

kkk = np.dot(weight2,weight1)
print('Multiple two weight matrix, the result shape is {}'.format(kkk.shape))

u, s, vh = np.linalg.svd(kkk)
print()
print('u shape is {}'.format(u.shape))
print('s shape is {}'.format(s.shape))
print('vh shape is {}'.format(vh.shape))


hhh = 0

k = np.zeros((determined_layer_len,vh.shape[0]))
k[0:determined_layer_len,0:determined_layer_len] = np.diag(s)[hhh:determined_layer_len+hhh,hhh:determined_layer_len+hhh]

a = u[:,hhh:determined_layer_len+hhh]
b = np.dot(k,vh)

print(k.shape)
print(a.shape)
print(b.shape)

compression_net.params['fc6'][0].data[...] = b
compression_net.params['fc6'][1].data[...] = compression_net.params['fc6'][1].data[...] * 0
compression_net.params['fc7'][0].data[...] = a
compression_net.params['fc7'][1].data[...] = nocompression_net.params['fc7'][1].data[...]

compression_net.save('../caffe_models/caffe_model_1/new_model/mydrop05compression.caffemodel')



