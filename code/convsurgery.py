import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
from PIL import Image
import cv2
import statistics
import time
import math
from numpy import *



# define a class for least weight pruning

class least_prune:
    def __init__(self,name,weights,bios):
        self.layer_name = name
        self.weight_matrix = weights
        self.bios_matrix = bios
        # self.prune_index = []
    
    def get_prune_index(self,ratio):

        prune_index = []

        weights = self.weight_matrix
        
        the_index = 0
        prune_time = int(weights.shape[0]*ratio)

        print('The prune iteration for {} is {}'.format(self.layer_name,prune_time))

        for kk in range(prune_time):
            min_weights = sys.maxint
            for map_num in range(weights.shape[0]):
                the_matrix = weights[map_num,:,:,:]
                the_sum = np.sum(np.absolute(the_matrix))
                if (the_sum<min_weights):
                    min_weights = the_sum
                    the_index = map_num
            prune_index.append(the_index)
            weights = np.delete(weights,(the_index),axis = 0)
        for kkk in range(prune_time):
            prune_index[kkk] += kkk
        return prune_index

    def least_weight_prune(self,ratio):
        weights = self.weight_matrix
        
        the_index = 0
        prune_time = int(weights.shape[0]*ratio)

        # print('The prune iteration for {} is {}'.format(self.layer_name,prune_time))

        for kk in range(prune_time):
            min_weights = sys.maxint
            for map_num in range(weights.shape[0]):
                the_matrix = weights[map_num,:,:,:]
                the_sum = np.sum(np.absolute(the_matrix))
                if (the_sum<min_weights):
                    min_weights = the_sum
                    the_index = map_num
            weights = np.delete(weights,(the_index),axis = 0)
        return weights

    def least_bios_prune(self,prune_index,ratio):
        bios = self.bios_matrix
        prune_time = int(self.weight_matrix.shape[0]*ratio)
        for kkk in range(prune_time):
            prune_index[kkk] -= kkk
        for index in prune_index:
            bios = np.delete(bios,(index),axis = 0)
        return bios

    def has_pool(self,result):
        can_do_flag = 0
        if (result==0 or self.weight_matrix.shape[1]==result):
            can_do_flag = 1
        return can_do_flag



def input_prune(weights,ratio):        
    the_index = 0
    prune_time = int(weights.shape[1]*ratio)

    # print('The prune iteration for {} is {}'.format(self.layer_name,prune_time))

    for kk in range(prune_time):
        min_weights = sys.maxint
        for map_num in range(weights.shape[1]):
            the_matrix = weights[:,map_num,:,:]
            the_sum = np.sum(np.absolute(the_matrix))
            if (the_sum<min_weights):
                min_weights = the_sum
                the_index = map_num
        weights = np.delete(weights,(the_index),axis = 1)
    return weights





# variable initialize here

drop_thresh = 0.5   #(0~1)

compression_net = caffe.Net('../caffe_models/caffe_model_1/caffenet_deploy_convdrop05.prototxt','../caffe_models/caffe_model_1/snapshot/convdrop05_iter_5000.caffemodel',caffe.TEST)
nocompression_net = caffe.Net('../caffe_models/caffe_model_1/caffenet_deploy_1.prototxt','../caffe_models/caffe_model_1/original_model/keeptrain_iter_5000.caffemodel',caffe.TEST)



# get out all the conv parameters for nocompression net

params = []
nocompression_weights_pool = []
nocompression_bios_pool = []

print("Here is the structure of no compression net: ")

for k,v in nocompression_net.params.items():
    params.append(k)
    nocompression_weights_pool.append(v[0].data)
    nocompression_bios_pool.append(v[1].data)
    print(k,v[0].data.shape,v[1].data.shape)

print("Here is the structure of compression net: ")

for k,v in compression_net.params.items():
    # params.append(k)
    # nocompression_weights_pool.append(v[0].data)
    # nocompression_bios_pool.append(v[1].data)
    print(k,v[0].data.shape,v[1].data.shape)


# doing the least_prune for all the convolutional layer

layer_name = list(nocompression_net._layer_names)


previous_prune_result = 0
previous_prune_index = []


for k in range(len(layer_name)):
    # print(layer_name[k],nocompression_net.layers[k].type)
    if (previous_prune_result == 0 and nocompression_net.layers[k].type == 'Convolution'):
            index = params.index(layer_name[k])
            previous_prune_index = least_prune(params[index],nocompression_weights_pool[index],nocompression_bios_pool[index]).get_prune_index(drop_thresh)
            compression_net.params[layer_name[k]][1].data[...] = least_prune(params[index],nocompression_weights_pool[index],nocompression_bios_pool[index]).least_bios_prune(previous_prune_index,drop_thresh)
            compression_net.params[layer_name[k]][0].data[...] = least_prune(params[index],nocompression_weights_pool[index],nocompression_bios_pool[index]).least_weight_prune(drop_thresh)
            previous_prune_result = len(compression_net.params[layer_name[k]][1].data[...])

            # print('The prune index is {}'.format(previous_prune_index))
            print('The prune result for {} is {}'.format(params[index],previous_prune_result))
    elif (previous_prune_result != 0 and nocompression_net.layers[k].type == 'Convolution'):
        index = params.index(layer_name[k])
        # if (least_prune(params[index],nocompression_weights_pool[index],nocompression_bios_pool[index]).has_pool(previous_prune_result)==1):

        a = input_prune(nocompression_weights_pool[index],drop_thresh)
        previous_prune_index = least_prune(params[index],a,nocompression_bios_pool[index]).get_prune_index(drop_thresh)
        compression_net.params[layer_name[k]][1].data[...] = least_prune(params[index],a,nocompression_bios_pool[index]).least_bios_prune(previous_prune_index,drop_thresh)
        compression_net.params[layer_name[k]][0].data[...] = least_prune(params[index],a,nocompression_bios_pool[index]).least_weight_prune(drop_thresh)
        previous_prune_result = len(compression_net.params[layer_name[k]][1].data[...])

        # print('The prune index is {}'.format(previous_prune_index))
        print('The prune result for {} is {}'.format(params[index],previous_prune_result))









