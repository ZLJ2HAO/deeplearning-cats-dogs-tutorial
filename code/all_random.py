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
import random


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
        prune_time = int(weights.shape[0]*(1-ratio))

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
        prune_time = int(weights.shape[0]*(1-ratio))

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
        prune_time = int(self.weight_matrix.shape[0]*(1-ratio))
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
    prune_time = int(weights.shape[1]*(1-ratio))

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




# def first_fc_prune(weights,ratio):
#     the_index = 0
#     prune_time = int(weights.shape[1]*ratio)

#     # print('The prune iteration for {} is {}'.format(self.layer_name,prune_time))

#     for kk in range(prune_time):
#         min_weights = sys.maxint
#         for map_num in range(weights.shape[1]):
#             the_matrix = weights[:,map_num]
#             the_sum = np.sum(np.absolute(the_matrix))
#             if (the_sum<min_weights):
#                 min_weights = the_sum
#                 the_index = map_num
#         weights = np.delete(weights,(the_index),axis = 1)
#         print(kk)
#     return weights

def first_fc_prune(weights,the_axis,ratio):
    the_index = 0
    prune_time = int(weights.shape[the_axis]*(1-ratio))

    # print('The prune iteration for {} is {}'.format(self.layer_name,prune_time))

    for kk in range(prune_time):
        the_index = random.randint(0,weights.shape[the_axis]-1)
        weights = np.delete(weights,the_index,axis = the_axis)
        
    return weights


class all_random(object):
    # def __init__(self):
    #     self.prune_index=[]
    #     self.weights = []
    #     self.bios = []
    prune_index=[]
    weights = []
    bios = []
    def first_fc_prune(self,weights,the_axis,ratio):
        the_index = 0
        prune_time = int(weights.shape[the_axis]*(1-ratio))
        haha = []
        # print('The prune iteration for {} is {}'.format(self.layer_name,prune_time))

        for kk in range(prune_time):
            the_index = random.randint(0,weights.shape[the_axis]-1)
            haha.append(the_index)
            weights = np.delete(weights,the_index,axis = the_axis)
            # print(kk)
            # print("{},{}".format(weights.shape[the_axis],the_index))
        self.prune_index = haha
        # print("{}??????????????????".format(self.prune_index))
        self.weights = weights
        
    def prune_bios(self,thebios,index,ratio):
        prune_time = int(len(thebios)*(1-ratio))
        # print("{}!!!!!!!!!!!".format(len(thebios)))
        # for kkk in range(prune_time):
        #     index[kkk] += kkk      
        for hhh in index:
            thebios = np.delete(thebios,(hhh),axis = 0)
        self.bios = thebios
        





class SVD_drop(object):
    w1 = []
    w2 = []
    def SVD_weights(self,weight1,weight2,ratio):
        [original_layer_len,n] = weight1.shape
        determined_layer_len = int(original_layer_len * ratio)
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

        self.w2 = u[:,hhh:determined_layer_len+hhh]
        self.w1 = np.dot(k,vh)
    # def SVD_bios(self,bios,ratio):
    #     determined_length = len(bios) * ratio
    #     return bios[:,0:determined_length] * 0

# variable initialize here




want_compress = 0.0625   #(0~1)

drop_thresh = sqrt(want_compress)

print('The compression ratio is {}'.format(want_compress))

nocompression_net = caffe.Net('../caffe_models/caffe_model_1/caffenet_deploy_1.prototxt','../caffe_models/caffe_model_1/original_model/solver_1_iter_5000.caffemodel',caffe.TEST)
compression_net = caffe.Net('../caffe_models/caffe_model_1/caffenet_deploy_adaptive_drop00625.prototxt','../caffe_models/caffe_model_1/snapshot/adaptivedrop00625_iter_5000.caffemodel',caffe.TEST)



# get out all the conv parameters for nocompression net

params = []
nocompression_weights_pool = []
nocompression_bios_pool = []

print("Here is the target layers of no compression net: ")

for k,v in nocompression_net.params.items():
    params.append(k)
    nocompression_weights_pool.append(v[0].data)
    nocompression_bios_pool.append(v[1].data)
    print("shape of nocompression_bios is {}".format(v[1].data.shape))
    print(k,v[0].data.shape,v[1].data.shape)

# print("shape of nocompression_bios is {}".format(v[1].data.shape))


print("Here is the target layers of compression net: ")

for k,v in compression_net.params.items():
    # params.append(k)
    # nocompression_weights_pool.append(v[0].data)
    # nocompression_bios_pool.append(v[1].data)
    print(k,v[0].data.shape,v[1].data.shape)


# doing the least_prune for all the convolutional layer

layer_name = list(nocompression_net._layer_names)



previous_prune_result = 0
# previous_prune_index = []
fc_count = 0
previous_fc_weight = []
previous_prune_index = []

for k in range(len(layer_name)):
    # print(layer_name[k],nocompression_net.layers[k].type)
    if (previous_prune_result == 0 and nocompression_net.layers[k].type == 'Convolution'):
            print("I am at first layer")
            h1 = all_random()
            index = params.index(layer_name[k])
            # previous_prune_index = least_prune(params[index],nocompression_weights_pool[index],nocompression_bios_pool[index]).get_prune_index(drop_thresh)
            h1.first_fc_prune(nocompression_weights_pool[index],0,drop_thresh)
            previous_prune_index = h1.prune_index
            h1.prune_bios(nocompression_bios_pool[index],previous_prune_index,drop_thresh)
            compression_net.params[layer_name[k]][1].data[...] = h1.bios
            compression_net.params[layer_name[k]][0].data[...] = h1.weights
            previous_prune_result = len(compression_net.params[layer_name[k]][1].data[...])

            # print('The prune index is {}'.format(previous_prune_index))
            print('The prune result for {} is {}'.format(params[index],previous_prune_result))
    elif (previous_prune_result != 0 and nocompression_net.layers[k].type == 'Convolution'):
        print("I am at following layer")
        index = params.index(layer_name[k])
        # if (least_prune(params[index],nocompression_weights_pool[index],nocompression_bios_pool[index]).has_pool(previous_prune_result)==1):
        h2 = all_random()
        
        a = first_fc_prune(nocompression_weights_pool[index],1,drop_thresh)
        # print("hhhhhhhhhhhhh")
        h2.first_fc_prune(a,0,drop_thresh)
        compression_net.params[layer_name[k]][0].data[...] = h2.weights
        previous_prune_index = h2.prune_index
        h2.prune_bios(nocompression_bios_pool[index],previous_prune_index,drop_thresh)
        compression_net.params[layer_name[k]][1].data[...] = h2.bios
        previous_prune_result = len(compression_net.params[layer_name[k]][1].data[...])

        # print('The prune index is {}'.format(previous_prune_index))
        print('The prune result for {} is {}'.format(params[index],previous_prune_result))
    elif (nocompression_net.layers[k].type == 'InnerProduct'):
        fc_count += 1

print('finish compress conv layer')


target1 = []
target2 = []
compression_fc_pool = []
temp_count = fc_count
compression_bios_pool = []

for k in range(len(layer_name)):
    if (nocompression_net.layers[k].type == 'InnerProduct'):
        if (temp_count<=1):
            
            index = params.index(layer_name[k])
            print("last fc layer index is {}".format(index))
            this_bios = nocompression_bios_pool[index]
            # h3 = all_random()
            b = first_fc_prune(nocompression_weights_pool[index],1,drop_thresh)  
            compression_fc_pool.append(b)
            compression_bios_pool.append(this_bios)
            print('put {} in the weights pool with shape {}'.format(layer_name[k],b.shape))
            print('put {} in the bios pool with length {}'.format(layer_name[k],len(this_bios)))       
            print('No need to continue compressing fc layer')
            break          
        else:
            index = params.index(layer_name[k])
            print("this fc layer index is {}".format(index))

            hehe = nocompression_bios_pool[index]
            h4 = all_random()
            h4.first_fc_prune(nocompression_weights_pool[index],0,drop_thresh)
            a = h4.weights
            b = first_fc_prune(a,1,drop_thresh)
            compression_fc_pool.append(b)
            print('put {} in the weights pool with shape {}'.format(layer_name[k],b.shape))
            for i in range(len(h4.prune_index)):
                hehe = np.delete(hehe,(h4.prune_index[i]),axis = 0)
            compression_bios_pool.append(hehe) 
            print('put {} in the bios pool with length {}'.format(layer_name[k],len(hehe)))   
         
        temp_count -= 1
                     

for k in range(len(layer_name)):
    if (nocompression_net.layers[k].type == 'InnerProduct'):
        if (compression_fc_pool == []):
            print('The compression fc pool is empty')
            break
        else:
            compression_net.params[layer_name[k]][0].data[...] = compression_fc_pool[0]
            compression_net.params[layer_name[k]][1].data[...] = compression_bios_pool[0]
            print('get {} from the weight pool'.format(layer_name[k]))
            print('get {} from the bios pool'.format(layer_name[k]))
            del compression_fc_pool[0]
            del compression_bios_pool[0]




compression_net.save('../caffe_models/caffe_model_1/adaptive_model/random'+str(want_compress).replace(".","")+'.caffemodel')
