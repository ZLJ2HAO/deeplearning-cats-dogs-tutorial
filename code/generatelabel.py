import os
import glob
import random
import numpy as np

import cv2

import caffe
from caffe.proto import caffe_pb2
import lmdb

test_file = open("../caffe_models/caffe_model_1/snapshot/testlabel.txt",'w')
test_data = [img for img in glob.glob("../input/train/*jpg")]
for in_idx, img_path in enumerate(test_data):
    if in_idx % 6 != 0:
        continue
    if 'cat' in img_path:
        label = 0
        name = img_path + ' ' + '0' + '\n'
        test_file.write(name)
    else:
        label = 1
        name = img_path + ' ' + '1' + '\n'
        test_file.write(name)
test_file.close()
