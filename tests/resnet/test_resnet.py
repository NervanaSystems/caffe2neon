import os
import caffe
import pickle
import argparse
import numpy as np

from future.moves.urllib.request import Request, urlopen

import sys
sys.path.insert(0, '')
from decaffeinate import Decaffeinate

from neon.models.model import Model
from neon.util.modeldesc import ModelDescription
from neon.data import ArrayIterator
from neon.backends import gen_backend
from neon.data import Dataset

import caffe

# hijacking the code in Dataset to download the model def and weights
download = Dataset.fetch_dataset

model_path = os.path.dirname(__file__)

# get the weights, the weights need to be downloaded and may need to be
# covereted to use the newer caffe protobuf format, the old format uses
# the V1LayereParameter protobuf objects, this is not compatible with these tools

# download from the link @ https://github.com/KaimingHe/deep-residual-networks
weights_pb = os.path.join(model_path, 'ResNet-50-model.caffemodel')
# use slightly altered protxt files for testing googlnet
# not the ones found in caffe release
deploy_pb = os.path.join(model_path, 'ResNet-50-deploy.prototxt')
train_pb = os.path.join(model_path, 'train_val.prototxt')

# make the train_val prototxt
with open(deploy_pb, 'r') as fid:
    lines = fid.readlines()

dummy = """
layer {
    top: "data"
    name: "data"
    type: "DummyData"
    dummy_data_param: {
        shape: {
            dim: %d
            dim: %d
            dim: %d
            dim: %d
        }    
    }    
}
"""
dims = []
inds_pop = []
for ind in range(len(lines)):
    line = lines[ind]
    if line.find('input:') == 0:
        inds_pop.append(ind)
    elif line.find('input_dim') == 0:
        tmp_ = line.strip().split(':')[1]
        dims.append(int(tmp_))
        inds_pop.append(ind)

inds_pop.reverse()
for ind in inds_pop:
    lines.pop(ind)
lines.insert(inds_pop[-1], dummy % tuple(dims))

with open(train_pb, 'w') as fid:
    fid.write(''.join(lines))

for fn in [weights_pb, train_pb, deploy_pb]:
    if not os.path.exists(fn):
        # if the weights_pb file is missing it may need to be downloaded from
        # https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
        # or 
        # http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
        print fn
        raise IOError('Could not find file %s (weights need to be downloaded)' % fn)

# run the conversion from caffe to neon
decaf = Decaffeinate(train_pb, weights_pb)
pdict = decaf.convert_to_neon()
conv_file_path = os.path.join(model_path, 'neon_conv.pkl')
with open(conv_file_path, 'w') as fid:
    pickle.dump(pdict, fid)

pdict = {}
del pdict

# deserialize and run the model in neon
# generate a backend
be = gen_backend(backend='gpu', rng_seed=1, batch_size=16, device_id=1)

with open(conv_file_path, 'r') as fid:
    pdict_l = pickle.load(fid)

# for testing we need to switch dropout to keep=1.0
# get the dropout values set in the serialized file and reset them to
# have no dropout for comparison between caffe 
md = ModelDescription(pdict_l)
#md['model']['config']['layers'].pop(1)
md = dict(md)

# generate a fake input
IM_SIZE = (be.bsz, 3, 224, 224)
np.random.seed(1)
im = np.random.randint(-127, 129, IM_SIZE)
fake_labels = np.zeros((IM_SIZE[0], 10))

# need this iterator to initialize the model
train = ArrayIterator(im.reshape(IM_SIZE[0], -1).copy(), fake_labels, nclass=10, lshape=IM_SIZE[1:])

# deserialize the neon model
model = Model(md, weights_only=True)
model.initialize(train)

# make sure the deserialization is setting the compat mode correctly
assert be.compat_mode == 'caffe'

# generate a fake input
im_neon = be.array(im.reshape(IM_SIZE[0], -1).astype(np.float32).T.copy())
out_neon = model.fprop(im_neon, inference=True)

# run the same input through caffe model
# need to set
caffe.set_mode_gpu()
caffe.set_device(1)
net = caffe.Net(deploy_pb, weights_pb, caffe.TEST)


net.blobs['data'].reshape(*IM_SIZE)
net.blobs['data'].data[...] = im.copy()
#net.layers[1].blobs[1].data[:] = 0.0
net.forward()


outn= model.layers.layers[2].outputs.get().reshape((64, 112, 112, 16)).transpose((3,0,1,2))
outc=np.array(net.blobs['conv1'].data)

outn = model.layers.layers[5].outputs.get().reshape((256,56,56,16)).transpose((3,0,1,2))
outc = np.array(net.blobs['res2a'].data)


out_caffe = net.blobs['prob'].data
outn = out_neon.get().T
outc = np.array(out_caffe)
dd = outn - outc
mx_ind = np.argmax(np.abs(dd))
mx = np.max(np.abs(dd))
print 'max diff: %.4e, neon val: %.4e, caffe val: %.4e' % (mx,
                                                           outn.flatten()[mx_ind],
                                                           outc.flatten()[mx_ind])
