import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
from PIL import Image
import cv2


compression_net = caffe.Net('../caffe_models/caffe_model_1/caffenet_deploy_SVD3000.prototxt','../caffe_models/caffe_model_1/snapshot/alexnet_SVD3000_5000.caffemodel',caffe.TEST)
nocompression_net = caffe.Net('../caffe_models/caffe_model_1/caffenet_deploy_1.prototxt','../caffe_models/caffe_model_1/snapshot/alexnet_origin_5000.caffemodel',caffe.TEST)


# params = ['conv1','conv2','conv3','conv4','conv5','fc6','fc7','fc8','fc9']
params = []
for k,v in nocompression_net.params.items():
    params.append(k)
    print(k,v[0].data.shape,v[1].data.shape)

for pr in params:
    if pr != 'fc6':
        compression_net.params[pr] = nocompression_net.params[pr]
        print('{} layer has been transfered'.format(pr))

add1 = compression_net.params['add1'][0].data[...]
print('add1 layer shape is {}\n '.format(add1.shape))
[add1_x,add1_y] = add1.shape

fc6 = compression_net.params['fc6'][0].data[...]
print('fc6 layer shape is {}\n '.format(fc6.shape))
[fc6_x,fc6_y] = fc6.shape


fc6_weight_matrix = nocompression_net.params['fc6'][0].data[...]
u, s, vh = np.linalg.svd(fc6_weight_matrix)
print('u shape is {}'.format(u.shape))
print('s shape is {}'.format(s.shape))
print('vh shape is {}'.format(vh.shape))


hhh = 0

k = np.zeros((add1_x,vh.shape[0]))
k[0:add1_x,0:add1_x] = np.diag(s)[hhh:add1_x+hhh,hhh:add1_x+hhh]

a = u[:,hhh:add1_x+hhh]
b = np.dot(k,vh)

print(k.shape)
print(a.shape)
print(b.shape)

compression_net.params['fc6'][0].data[...] = a
compression_net.params['fc6'][1].data[...] = nocompression_net.params['fc6'][1].data[...] * 0
compression_net.params['add1'][0].data[...] = b

compression_net.save('../caffe_models/caffe_model_1/new_model/mySVD3000compression.caffemodel')

# compression_net.params['add1'][0].data =











# fc6 = compression_net.params['fc6'][0].data[...].reshape([add1add1_length])
#
#
# print('fc6 layer shape is\n {}\n '.format(fc6.shape))
# k = nocompression_net.params['conv1']
# fc9=k
# print('fc6_biaos_matrix is\n {}\n shape is {}'.format(fc6_biaos_matrix,fc6_biaos_matrix.shape))


# print('the input data is {}'.format(net.blobs['data'].data[...].shape))
# print('the net is {}'.format(net))
