import sys
sys.path.insert(0,'.')
sys.path.insert(0,'..')
import torch
from torch.autograd import Variable
import pytorch_to_caffe
from convlayer import SimpleConvLayer


if __name__=='__main__':
    name='convlayer'
    #net=alexnet(True)
    net = SimpleConvLayer()
    net.load_state_dict(torch.load("convlayer.pth"))

    input=Variable(torch.ones([1,3,4,4]))
    pytorch_to_caffe.trans_net(net,input,name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))
