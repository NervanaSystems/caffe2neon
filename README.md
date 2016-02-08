# decaffeinate

Tools to convert Caffe models to neon format

## Introduction

This repo contains tools to convert Caffe models into a format compatible with the 
[neon deep learning library](https://github.com/NervanaSystems/neon).  The main
script, "decaffeinate.py", takes as input a caffe model definition file and the corresponding
model weights file and returns a neon serialized model file.  This output file can be used to
instantiate the neon Model object, which will generate a model in neon that should replicate the 
behavior of the Caffe model.


## Model conversion

To convert a model from Caffe to neon use the
[decaffeinate.py](https://github.com/NervanaSystems/decaffeinate/blob/master/decaffeinate.py) script.
This script requires a text protobuf formatted Caffe model definition file and a binary protobuf model
parameter file.  The decaffeinate script will generate a neon compatible model file with the same model
configuration and weights.

Example:
```
python decaffeinate.py train_val.prototxt  bvlc_googlenet.caffemodel
```

Once the conversion is done, the model can be loaded into neon with the following python commands.
In order to instantiate a model a data iterator object is necessary and to run the code below neon
must be installed, importable and the virtual environment should be active if necessary.

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

See the neon documentation for how to generate the imagenet
[macrobatches](http://neon.nervanasys.com/docs/latest/datasets.html#imagenet) used above.

## Imagenet category ordering

The ILSVRC12 data set used by Caffe has the output categories remapped from the default ordering which is
inconsistent with the category mapping used by the neon dataset utilities.  This repo includes a helper script
to reorder output classifier layers from the Caffe mapping to the neon mapping.  This only applies to the
ILSVRC12 data set, but a similar remapping could be done for other applications.  The
[i1k_label_mapping.py](https://github.com/NervanaSystems/decaffeinate/blob/master/i1k_label_mapping.py) script
takes the *neon* model file and a list of output layers as input and reorders the weights of output linear and bias
layers to match the neon category ordering.  Currently this script only works on linear layers and the associated
bias layer.

When running the script for the first time, users will need to provide the path to the synsets.txt file in the Caffe
distribution, usually this would be the directory _<caffe root>/data/ilsvrc12_, and
the path to the ILSVCR2012 devkit tarfile.  After the first run, the script will generate a pickle file
(_neon_caffe_label_map.pkl_) with the categrory mapping between the two frameworks, and future runs can use the
_--use_existing_ option which will load the mapping from this saved file.

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
