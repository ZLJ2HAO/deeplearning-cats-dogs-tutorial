import os
# import time
import matplotlib.pyplot as plt
import numpy as np
import math

data1 = np.loadtxt('../caffe_models/caffe_model_1/snapshot/submission_model_1.csv', delimiter=',', dtype='str')
[row1, col1] = data1.shape
name1 = data1[1:row1, 0]
label1 = data1[1:row1, 1].astype('int')
print('row1 {},col1 {}'.format(row1-1, col1))


data2 = np.loadtxt('../caffe_models/caffe_model_1/snapshot/testlabel.txt', delimiter=' ', dtype='str')
[row2, col2] = data2.shape
name2 = data2[1:row2, 0]
label2 = data2[1:row2, 1].astype('int')
print('row2 {},col2 {}'.format(row2, col2))

print('prediction label1 max is {}'.format(max(label1)))
print('ground truth label2 max is {}'.format(max(label2)))

index = 0
n = 0
error = []
k = 0
# my_range = 4
# print('my_range is {}'.format(my_range))
# name = name1[0]
# for index in range(0,row2-1,1):
#     if name2[index].find(name)>=0:
#         print('result is {}'.format(name2[index].find(name)))
#         print('index is {}'.format(index))

# print('name1 is {}'.format(name1))
# print('name2 is {}'.format(name2))


for name in name2:
    # print('name is {}'.format(name))
    name = name.replace('../input/train/','')
    # name = name.replace('dog.','')
    # name = name.replace('cat.','')
    name = name.replace('.jpg','')
    for index in range(0,row1-1,1):
        if name == name1[index]:
            error.append(abs(label2[n] - label1[index]))
            if label2[n] - label1[index] == 0:
                k = k + 1
            # print('match')
    n = n + 1


# max_error = max(error)
mean_error = np.mean(error)
# print('max error is {}'.format(max_error))
print('mean error is {}'.format(mean_error))

# plt.hist(error, bins=np.arange(0, 1.2, 0.001))
# plt.title("error Histogram")
# plt.xlabel("error")
# plt.ylabel("picture number")
# plt.show()



# error = sum([item for item in error])
# error = math.sqrt(error)
# print('error is {}'.format(error))
print('match is {}'.format(k))
