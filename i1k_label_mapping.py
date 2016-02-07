'''
This is a helper script to generate the mapping between the neon and caffe default
ILSVRC 2012 cateories.  Since the image categories as not ordered in the same way
a mapping between the categories needs to be generated and this mapping is then used
to permute the weights of the output linear layers in order to generate the proper
category for each image.

This process is only necessary for converting a caffe model to neon and using the neon
ILSVRC 2012 data batches.
'''

import re
import os
import zlib
import pickle
import tarfile
import argparse
import numpy as np

try:
    from neon.util.modeldesc import ModelDescription
except:
    raise ImportError("Could not import neon library, check paths or venv setup/activation")

parser = argparse.ArgumentParser()
parser.add_argument('model_file', help='Neon formatted model file')
parser.add_argument('--use_existing', action='store_true',
                    help='use a pre-existing mappign file "neon_caffe_label_map.pkl"')
parser.add_argument('--caffe_synset', default=None,
                    help='path to the dir with caffe ilsvrc12 synset_words.txt file')
parser.add_argument('--neon_mb', default=None,
                    help='path to the dir ILSVRC2012 devkit tar file')
parser.add_argument('--layers', nargs='*')

args = parser.parse_args()


assert os.path.exists(args.model_file)

if args.use_existing:
    assert os.path.exists('neon_caffe_label_map.pkl'), 'Could not find mapping file'
    with open('neon_caffe_label_map.pkl', 'r') as fid:
        lbl_map = pickle.load(fid)
else:
    # caffe data file are usually in <caffe root>/data/ilsvrc12/synset_words.txt
    assert os.path.isdir(args.caffe_synset), 'Can not find directory with caffe synset words'
    caffe_synset = os.path.join(args.caffe_synset, 'synsets.txt')
    assert os.path.exists(caffe_synset), 'Can not find caffe synset words'


    if args.neon_mb is None:
        print 'No ILSVRC2012 devkit dir given, will try to download gist with default mapping'
        # TODO put this up on gist so people can just get that
    else:
        devkit = os.path.join(args.neon_mb, 'ILSVRC2012_devkit_t12.tar.gz')
        if not os.path.exists(devkit):
            raise IOError(infile + " not found. Please ensure you have ImageNet downloaded."
                          "More info here: http://www.image-net.org/download-imageurls")
        synsetfile = 'ILSVRC2012_devkit_t12/data/meta.mat'
        with tarfile.open(devkit, "r:gz") as tf:
            meta_buff = tf.extractfile(synsetfile).read()
            decomp = zlib.decompressobj()
            neon_synsets = re.findall(re.compile('n\d+'), decomp.decompress(meta_buff[136:]))

    # remap all the classifiers
    #caffe_root = '/home/users/evren/repos/caffe/'
    #imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'

    # load up the caffe categories
    clabels = list(np.loadtxt(caffe_synset, str, delimiter='\t'))
    # needed when using synset_words
    # clabels = [ ' '.join(x.split(' ')[1:]) for x in clabels]

    lbl_map = []
    for lbl in neon_synsets[0:1000]:
        ind = clabels.index(lbl)
        lbl_map.append(ind)
    lbl_map = np.array(lbl_map)

    with open('neon_caffe_label_map.pkl', 'w') as fid:
        pickle.dump(lbl_map, fid)
    print 'Wrote mapping to neon_caffe_label_map.pkl'

print 'loading model file %s' % args.model_file
model = ModelDescription(args.model_file)

def find_output_layer(check_lay):
    layers = []
    for ind in range(len(check_lay)-1, -1, -1):
        if check_lay[ind]['type'].find('Linear') > -1:
            layers.append(check_lay[ind]['config']['name'])
            break
        if check_lay[ind]['type'].find('Bias') > -1:
            layers.append(check_lay[ind]['config']['name'])
            if ind > 0 and check_lay[ind-1]['type'].find('Linear') > -1:
                layers.append(check_lay[ind-1]['config']['name'])
            break
    print 'found following layers: '
    print layers
    return layers

layers = args.layers
if layers is None or len(layers) == 0:
    print 'No layers indicated - will try to guess'
    try:
        if model['model']['type'].find('Tree') > -1:
            layers = []
            for clayer in model['model']['config']['layers']:
                layers.extend(find_output_layer(clayer['config']['layers']))
        else:
            layers = find_output_layer(model['model']['config']['layers'])
    except:
        raise ValueError('Could not parse the layers which need to be updated '
                         'provide name(s) of the layer(s) explicitly on command line')


for layer in layers:
    l = model.getlayer(layer)
    if l is None:
        raise ValueError('Could not find layer %s in model file' % layer)
    if l['type'].find('Bias') > -1:
        l['params']['W'] = l['params']['W'][lbl_map].copy()
    elif l['type'].find('Linear') > -1:
        l['params']['W'] = l['params']['W'][lbl_map, :].copy()
    else:
        raise ValueError('Currently only bias and linear layers can be shuffled')
new_file = os.path.splitext(args.model_file)
new_file = new_file[0] + '_neon_shuffle' + new_file[1]
print 'saving to file: %s' % new_file
with open(new_file, 'w') as fid:
    pickle.dump(dict(model), fid)
