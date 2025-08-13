import sys
sys.path.insert(0,'.')
sys.path.insert(0,'../..')
import torch
from torch.autograd import Variable
import pytorch_to_caffe
from splitfclayer import SplitFCLayer


if __name__=='__main__':
    name='SplitFCLayer'
    model = SplitFCLayer()
    model.load_state_dict(torch.load("SplitFCLayer.pth"))

    input=Variable(torch.load("input.pth"))
    pytorch_to_caffe.trans_net(model,input,name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))
