import caffe
import numpy as np

net = caffe.Net('model.prototxt', caffe.TEST)

# Set all weights in 'fc' layer to 1
net.params['fc'][0].data[...] = 1

# Save the weights to a caffemodel file
net.save('model.caffemodel')
