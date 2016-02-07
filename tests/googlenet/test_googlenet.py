import os
import caffe
import pickle
import argparse
import numpy as np

import sys
sys.path.insert(0, '')
from decaffeinate import Decaffeinate

from neon.models.model import Model
from neon.util.modeldesc import ModelDescription
from neon.data import ArrayIterator
from neon.backends import gen_backend

import caffe


model_path = os.path.dirname(__file__)

# get the weights, the weights need to be downloaded and may need to be
# covereted to use the newer caffe protobuf format, the old format uses
# the V1LayereParameter protobuf objects, this is not compatible with these tools
weights_pb = os.path.join(model_path, 'bvlc_googlenet.caffemodel')
# use slightly altered protxt files for testing googlnet
# not the ones found in caffe release
train_pb = os.path.join(model_path, 'train_val.prototxt')
deploy_pb = os.path.join(model_path, 'deploy.prototxt')
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
be = gen_backend(backend='gpu', rng_seed=1, batch_size=64)

with open(conv_file_path, 'r') as fid:
    pdict_l = pickle.load(fid)

# for testing we need to switch dropout to keep=1.0
# get the dropout values set in the serialized file and reset them to
# have no dropout for comparison between caffe 
md = ModelDescription(pdict_l)

drop_layers = {ky: -1 for ky in ["loss1/drop_fc", "loss2/drop_fc", "pool5/drop_7x7_s1"]}
for l in drop_layers:
    drop_layer = md.getlayer(l)
    drop_layers[l] = drop_layer['config']['keep']
    drop_layer['config']['keep'] = 1.0
md = dict(md)

# generate a fake input
IM_SIZE = (be.bsz, 3, 224, 224)
np.random.seed(1)
im = np.random.randint(-150, 150, IM_SIZE)
fake_labels = np.zeros((IM_SIZE[0], 10))

# need this iterator to initialize the model
train = ArrayIterator(im.reshape(IM_SIZE[0], -1).copy(), fake_labels, nclass=10, lshape=IM_SIZE[1:])

# deserialize the neon model
model = Model(md, train)

# make sure the deserialization is setting the compat mode correctly
assert be.compat_mode == 'caffe'

# generate a fake input
im_neon = be.array(im.reshape(IM_SIZE[0], -1).astype(np.float32).T.copy())
out_neon = model.fprop(im_neon)

# get the neon layer output order using the layer names
l_map = []
for l in model.layers.layers:
    loss_lay = l.layers[-1].name
    l_map.append(loss_lay.split('_Softmax')[0])


# run the same input through caffe model
# need to set
caffe.set_mode_gpu()
caffe.set_device(1)
net = caffe.Net(deploy_pb, weights_pb, caffe.TEST)

net.blobs['data'].reshape(*IM_SIZE)
net.blobs['data'].data[...] = im.copy()
net.forward()
output_names = []
# layer names in caffe are getting remapped?
for nm in l_map:
    nm_ = nm.split('/')[0]
    for nm_c in net.blobs.keys():
        if nm_c.find(nm_+'/loss') > -1:
            output_names.append(nm_c)
            break

out_caffe = [net.blobs[nm].data for nm in output_names]

# test the 3 outputs
for n, c in zip(out_neon, out_caffe):
    n = n.get().T
    c = np.array(c)
    dd = n - c
    mx_ind = np.argmax(np.abs(n-c))
    mx = np.max(np.abs(n-c))
    print 'max diff: %.4e, neon val: %.4e, caffe val: %.4e' % (mx,
                                                               n.flatten()[mx_ind],
                                                               c.flatten()[mx_ind])
    assert np.abs(n.flatten()[mx_ind]-c.flatten()[mx_ind]) == mx
