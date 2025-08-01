import caffe
import numpy as np

# Set Caffe to CPU mode (or GPU if available)
caffe.set_mode_cpu()

# Load model
net = caffe.Net('model.prototxt', 'model.caffemodel', caffe.TEST)

# Prepare input data: a single 3-element vector
#input_data = np.ones((1,1,2,3), dtype=np.float32)
#input_data = np.ones((1,4,2,3), dtype=np.float32)
#input_data = np.ones((1,16,2,3), dtype=np.float32)
input_data = np.ones((1,17,2,3), dtype=np.float32)

# Reshape input blob to batch size 1
net.blobs['data'].data[...] = input_data

# Run forward pass
output = net.forward()

# Get the output
output_data = output['fc']
print("Output:", output_data)
