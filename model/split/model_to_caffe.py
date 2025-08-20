import sys
sys.path.insert(0,'../..')
sys.path.insert(0,'.')
import torch
from torch.autograd import Variable
import pytorch_to_caffe
#from splitfclayer import SplitFCLayer
from model import SplitConcat


if __name__=='__main__':
    name='SplitConcat'
    model = SplitConcat()
    model.load_state_dict(torch.load("model.pth"))

    input=Variable(torch.load("input.pth"))
    pytorch_to_caffe.trans_net(model,input,name)
    pytorch_to_caffe.save_prototxt('model.prototxt')
    pytorch_to_caffe.save_caffemodel('model.caffemodel')
