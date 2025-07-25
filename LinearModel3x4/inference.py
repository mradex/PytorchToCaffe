import caffe
import numpy as np

# Set Caffe to CPU mode (or GPU if available)
caffe.set_mode_cpu()

# Load model
net = caffe.Net('model.prototxt', 'model.caffemodel', caffe.TEST)

# Prepare input data: a single 3-element vector
input_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

# Reshape input blob to batch size 1
net.blobs['data'].reshape(1, 3)
net.blobs['data'].data[...] = input_data

# Run forward pass
output = net.forward()

# Get the output
output_data = output['fc']
print("Output:", output_data)
