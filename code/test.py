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


# drop_net = caffe.Net('/home/roy/end-to-end-car-caffe/SVD_compression/test_for_layer/deploy5_5dropout.prototxt','/home/roy/end-to-end-car-caffe/SVD_compression/test_for_layer/model/5_5drop_iter_424.caffemodel',caffe.TEST)
# nodrop_net = caffe.Net('/home/roy/end-to-end-car-caffe/SVD_compression/test_for_layer/deploy5_5.prototxt','/home/roy/end-to-end-car-caffe/SVD_compression/test_for_layer/model/5_5_iter_781.caffemodel',caffe.TEST)


# params = []
# drop_weights_pool = []
# nodrop_weights_pool = []



# for k,v in nodrop_net.params.items():
#     params.append(k)
#     nodrop_weights_pool.append(v[0].data)

#     print(k,v[0].data,v[1].data)

# for k,v in drop_net.params.items():
#     params.append(k)
#     drop_weights_pool.append(v[0].data)

#     print(k,v[0].data,v[1].data)



print('sdfasf'+str(drop_thresh).replace(".","")+'.txt')


