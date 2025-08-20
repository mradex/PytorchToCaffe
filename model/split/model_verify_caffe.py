import caffe
import numpy as np

# Set to CPU or GPU
caffe.set_mode_cpu()
# caffe.set_mode_gpu(); caffe.set_device(0)

# Load model
net = caffe.Net('model.prototxt', 'model.caffemodel', caffe.TEST)
print(net.blobs.keys())

# Prepare input (same shape as during training)
#input_data = np.ones((1, 3, 4, 4), dtype=np.float32)
input_data = np.load("input.npy")

# Assign input to Caffe net
net.blobs['blob1'].data[...] = input_data

# Run forward pass
output = net.forward()
#
## Get output
output_blob = list(output.keys())[0]  # usually the first output
#output_blob = list(output.keys())[1]  # usually the first output
output_data = net.blobs[output_blob].data
#
print("Output shape:", output_data.shape)
print("Output values:", output_data)
