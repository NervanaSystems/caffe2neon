# caffe2neon

Tools to convert Caffe models to neon format

## Introduction

This repo contains tools to convert Caffe models into a format compatible with the 
[neon deep learning library](https://github.com/NervanaSystems/neon).  The main
script, "decaffeinate.py", takes as input a caffe model definition file and the corresponding
model weights file and returns a neon serialized model file.  This output file can be used to
instantiate the neon Model object, which will generate a model in neon that should replicate the 
behavior of the Caffe model.


## Installation

First make sure you have neon installed and activate the neon virtualenv.  This tools requires
that some extra packages be installed into the virtualenv.  To do that, run the command:
```
pip install -r path/to/caffe2neon/requirements.txt
```

Usually neon is installed into a virtualenv and caffe will be installed systemwide, not in the virtualenv.
So it may be necessary to add caffe and caffe protobuf libraries to your PYTHONPATH environment variable:
```
export $PYTHONPATH=$PYTHONPATH:path/to/cafferepo/python:path/to/cafferepo//python/caffe/proto/
```

Also note that if caffe was built with a numpy version that is different than that used by neon you may have
issues importing caffe into python from the neon virtualenv.  You may need to install the same numpy version
installed by neon into the virtualenv and rebuild the caffe installation.

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
ILSVRC12 data set, but a similar remapping could be done for other data sets if needed.  The
[i1k_label_mapping.py](https://github.com/NervanaSystems/decaffeinate/blob/master/i1k_label_mapping.py) script
takes the *neon* model file and a list of output layers as input and reorders the weights of the specified
layers to match the neon category ordering.  If no layers are specified, then the script will attempt to
identify the output linear and bias layers.

When running the script for the first time, users will need to provide the path to the _synsets.txt_ file in the Caffe
distribution, usually this would be the directory _<caffe root>/data/ilsvrc12_, and
the path to the ILSVCR2012 devkit tarfile.  After the first run, the script will generate a pickle file
named _neon_caffe_label_map.pkl_ which will store the categrory mapping between the two frameworks and
future runs can obtain the mapping from this file instead of recreating it from scratch.
Subsequent execution of this script can use the _--use_existing_ option to pull the mapping from this pickle
file and will not require the ILSVRC files as options.

The pickle file with the mapping can also be downloaded from the following link: [neon_caffe_label_map.p]( https://s3-us-west-1.amazonaws.com/nervana-modelzoo/neon_caffe_label_map.p).  Place the file in the main repo directory.

For example:
```
#first run
python i1k_label_mapping.py googlenet.prm  --caffe_synset ~/repos/caffe/data/ilsvrc12/ --neon_mb /mnt/data/I1K_raw/

# on subsequent runs:
python i1k_label_mapping.py googlenet.prm --use_existing

```

The script will try to infer the correct layers to alter, if this does not work users will have to enter the name
of all the layers to alter by hand using the _--layers_ command line parameter.  Note that in neon, bias layers are
seperate layers, and both the name of the linear layer and its corresponding bias layer will need to be specified.


## Versions

These tools require the neon repo at the commit version tag [v1.4.0](https://github.com/NervanaSystems/neon/tree/v1.4.0) and were developed using the Caffe repo state at commit SHA
ca4e3428b.

## Disclaimer
Due to differences between neon and Caffe in which types of layers are supported and the implementation of those layers
not all models can be converted to neon format.  Here we try to alert the user when a conversion may not be possible,
but there may be incompatible cases that do not trigger an alert.  This is a work in progress and only a limited number
of model  architectures have been tested.  We appreciate any feedback and, as this is an open source library, we
appreciate outside contributions to improve and expand these tools.
