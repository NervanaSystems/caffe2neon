# decaffeinate
Tools to convert Caffe models to neon's serialization format

## Introduction

This repo contains tools to convert caffe models into a format compatible with the neon deep learning libraries.  The main
script, "decaffeinate.py" takes as input a caffe model definition file and the corresponding model weights file and returns
a neon serialized model file.  This output file can be used to instantiate the neon Model object, which will generate
a model in neon that should replicate the behavior of the Caffe model.



## Model conversion

To convert a model from Caffe to neon use the "decaffeinate.py" script.  This script requires the model_file, which is the 
text protobuf file that defines the Caffe model configuration and the param_file which is a binary protobuf file that
contains the model parameters and weights.  The decaffeinate script will generate a neon
model serialization file which can be used to instantiate a model in neon with the same topology and weights.

Example:
```
python decaffeinate.py train_val.prototxt  bvlc_googlenet.caffemodel
```

Once the conversion is done, the model can be loaded into neon with the following commands in python (note that neon
must be installed and, if using the virtual env install the virtual env must be active):

```
from neon.models.model import Model
from neon.util.persist import load_obj
from neon.backends import gen_backend
from neon.data import ImageLoader

# generate a backend
be = gen_backend(backend='gpu', compat_mode='caffe', batch_size=128)

# get the data iterator setup
data_dir = /path/to/I1K/macrobatches
# use scale_range = 0 to match Caffe style cropping
test = ImageLoader(repo_dir=data_dir, set_name='validation', inner_size=224, scale_range=0, do_transforms=False)

model_dict = load_obj(path_to_model_file)
model = Model(model_dict, dataset=test)

# for example, can now evaluate the TopK misclassification metric on the validation data set
from neon.transforms.cost import TopKMisclassification
model.eval(test, TopKMisclassification(k=5))
```

See the neon documentation for how to generate the imagenet macrobatches used above [link](http://neon.nervanasys.com/docs/latest/datasets.html#imagenet).

## Imagenet category ordering

There is also a helper script that adapts models to the default ILSVRC12 category mapping.  The ILSVRC12 data set used
by Caffe has the output categories remapped from the default ordering and the neon data iterators for this data set use
the default ordering.  SO models ported to neon from caffe will need to have the output layer units reordered to match
the default ILSVRC12 category ordering.  The "i1k_label_mapping.py" script takes the *neon* model file and the list of
output layers and reorders the weights of output linear and bias layers to match the neon category ordering.

Currently this script only works on linear layers and the associated bias layer, reordering the units from the Caffe
category order to the neon order.

To run the script for the first time, users will need to provide the path to the synsets.txt file in the Caffe
distribution, usually this would be the directory <caffe root>/data/ilsvrc12.  Also users will need to provide
the path to the ILSVCR2012 devkit tarfile.  After the first run, the script will generate a pickle file with
the categrory mapping between the two frameworks, and in future runs only the --use_existing option need to given and
the script will use the saved file to generate the mapping between categories.  Note that the script is expecting
the directory of the caffe synsets.txt and ILSVCR2012 devkit, not the file itself.

For example:
```
python i1k_label_mapping.py googlenet.prm  --caffe_synset ~/repos/caffe/data/ilsvrc12/ --neon_mb /mnt/data/I1K_raw/

# on subsequent runs:
python i1k_label_mapping.py googlenet.prm --use_existing

```

The script will try to infer the correct layers to alter, if this does not work users will have to enter all the
layers to alter by hand using the --layers parameter.  Note that in neon, bias layers are seperate layers, and both
the name of the linear layer and its corresponding bias layer will need to be specified.


## Versions

These tools require neon version 1.2.1.  The scripts have been developed using the Caffe repo with the git SHA
ca4e3428b.

## Disclaimer
Due to differences in supported layers and layer implementations between neon and Caffe, not all models can be
converted.  Here we try to alert the user whena  conversion may not be possible.  Also only a limited set of
model architectures have been tested, so these tools are very much a work in progress.  We appreciate any feedback
and, as this is an open source library, we appreciate outside contributions to expand these tools.
