set -e
cd /home/roy/caffe/build/tools

./caffe train --weights=/home/roy/deeplearning-cats-dogs-tutorial/caffe_models/caffe_model_1/adaptive_model/random025.caffemodel --solver=/home/roy/deeplearning-cats-dogs-tutorial/caffe_models/caffe_model_1/solver_1.prototxt
