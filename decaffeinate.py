# !/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
#
# Convert Caffe model to neon format.
# 
# First the caffe prototxt file and model are loaded up and
# a graph of the layer configuration is generated.  Each node
# of the graph corresponds to a caffe layer.  The graph is
# converted to neon type layers - in the neon format some complex
# layer type, like inception layers, are incapsulated in a single
# node.  Finally, the neon formatted graph is serialized into a
# pickle file that can be loaded by neon.
# 
# To run this caffe must be on the python path
#

import os
import pickle
import argparse
import numpy as np
from copy import deepcopy
from google.protobuf import text_format
from google.protobuf.internal.containers import RepeatedScalarFieldContainer

try:
    import caffe
    from caffe.proto import caffe_pb2
except:
    raise ImportError('Must be able to import Caffe modules to use this module')


class GNode():
    # a node in the graph generated from the caffe model description
    #
    # Arguments:
    #   layer (caffe.proto.LayerParameter): caffe layer description object
    #
    def __init__(self, layer):
        self.layer = layer

        self.inp_names = list(layer.bottom)

        # upstream nodes
        self.us_nodes = []

        tops = list(layer.top)
        if len(tops) > 1:
            if layer.type != 'Data':
                raise ValueError('Currently only support multiple tops in data layer')
            else:
                # will only use the top named 'data'
                try:
                    datal = tops.index('data')
                except ValueError:
                    raise ValueError('Need to have data layer output named "data"')
            tops = [tops.pop(datal)]

        self.out_names = tops

        # downstream nodes
        self.ds_nodes = []

        # track inplace layer like Bias
        self.inplace_nodes = []

        self.name = layer.name
        self.ltype = layer.type

        # flag to speed up graph traversal
        self.touched = False

    def touch(self):
        self.touched = True

    def wipe(self):
        self.touched = False


# define wrapper to clean up tracking of visited nodes
def clean_traverse(func):
     def wrap(self, *args, **kwargs):
         self.wipe()
         retval = func(self, *args, **kwargs)
         self.wipe()
         return retval
     return wrap


class graph():
    #
    # class to generate network topology graph from
    # caffe model descritpion
    #
    # Arguments:
    #   layers (list of caffe_pb2.LayerParameter): layer params from
    #           caffe model definition
    #
    def __init__(self, layers):

        # try to find the top layer
        # currently this must be a Data layer with
        # a top named 'data'
        root_layer_ind = []
        layers = list(layers)
        for ind, l in enumerate(layers):
            if len(l.bottom) == 0:
                assert l.type.lower() in ['data', 'dummydata']  # may add more data layer types here
                root_layer_ind.append(ind)


        if len(root_layer_ind) != 1:
            raise ValueError('Found %d data layers, should have only 1' % len(root_layer_ind))
        root_layer_ind = root_layer_ind[0]
        # put data layer first
        root_layer = layers.pop(root_layer_ind)
        # generate the root layer GNode object
        self.root = GNode(root_layer)
        self.layers = layers

        #TODO ADD OTHER SUPPORTED LAYERS bartchnorm, tanh, other loss functions

        print 'building graph ...'
        self.build_graph()
        print 'done\n'

        print 'checking basic conversion support...'
        if self.check_support(self.root):
            print 'supported!\n'
        else:
            raise NotImplementedError('Can not convert to neon')

    def build_graph(self):
        #
        # constructs graph with model topology
        # uses the layers configuration from caffe
        # stored in self.layers
        #
        for ind, layer in enumerate(self.layers):
            new_node = GNode(layer)
            self.add_links(new_node, self.root)

    def add_links(self, new_node, tree_node):
        #
        # add new node to existing graph
        # descend down graph until the bottom of the new node
        # is found, add the new node above the found node
        #
        # recursively descends down graph
        #
        # Arguemnts:
        #   new_node (GNode): node to add to graph
        #   tree_node (GNode): node in current graph
        #

        # data source for new_node (i.e. bottom)
        inps = new_node.inp_names

        # outputs for the node in the tree (i.e. top)
        outs = tree_node.out_names

        #does the tree node supply to the new_node
        check_links = [inp in outs for inp in inps]
                            
        if any(check_links):
            for ind in range(len(check_links)):
                if check_links[ind]:
                    if inps == new_node.out_names:
                        # inplace op
                        if new_node not in tree_node.inplace_nodes:
                            tree_node.inplace_nodes.append(new_node)
                    else:
                        # add new_node to the downstream nodes of tree_node
                        if new_node not in tree_node.ds_nodes:
                            tree_node.ds_nodes.append(new_node)

                        # add tree_node to the upstream nodes of new_node
                        if tree_node not in new_node.us_nodes:
                            new_node.us_nodes.append(tree_node)

        for node in tree_node.ds_nodes:
            # keep descending down the graph
            self.add_links(new_node, node)

    def check_support(self, node):
        # layers supported by neon
        SUPPORTED_LAYERS = ['InnerProduct', 'Bias', 'Dropout', 'Convolution',
                            'Pooling', 'Data', 'ReLU', 'Concat','Softmax', 
                            'SoftmaxWithLoss', 'LRN', 'DummyData']

        if node.ltype not in SUPPORTED_LAYERS:
            print 'Found unsupported layer type %s [%s]' % (node.name, node.ltype)
            return False

        for node in node.ds_nodes:
            if not self.check_support(node):
                return False
        return True

    def check_merge_broadcast(self, node):
        #
        # inception like graph strucutres are encapsulated
        # in neon in MergeBroadcast container layers
        #
        # this function checks to see if graph nodes with
        # mutliple downstream nodes leads to a MergeBroadcast
        # like strucuture.  If found, this function returns the
        # merge node and the ds_nodes that termninate there
        #
        # Arguments:
        #   node (GNode): node to check
        #
        # Returns:
        #   tuple(GNode, list(int)) or None: if a mergebroadcast
        #       structure is detects then a tuple is returned with
        #       the top node where the merge occurs and a list of
        #       indicies of the node.ds_nodes that are part of the
        #       structure
        #

        fanout = len(node.ds_nodes)
        end_nodes = []
        for n in node.ds_nodes:
            end_nodes.append(self.find_merge_node(n, fanout, 1))

        if all([x is None for x in end_nodes]):
            return None

        branch_in_merge = []
        end_nodes_test = []
        for ind, n in enumerate(end_nodes):
            if n is not None:
                branch_in_merge.append(ind)
                end_nodes_test.append(n)
        if len(end_nodes_test) > 1 and all([x == end_nodes_test[0] for x in end_nodes_test]) and \
               end_nodes_test[0].ltype == 'Concat':
            # need at least 2 branches for a merge broadcast and all must end at the same node
            return (end_nodes_test[0], branch_in_merge)
        return None

    def find_merge_node(self, node, degree, depth):
        #
        # serach for a Concat ndoe
        #
        # currently this code is not supporting complex topologies in
        # the branches of a mergebroadcast like structure
        #
        # Arguments:
        #   node (GNode): current node being tested
        #   degree (int): fanout of the source node
        #   depth (int): depth into network from source node
        #
        # descend until a Concat node is hit
        if node.ltype == 'Concat':
            return node
        if len(node.ds_nodes) != 1:
            # do not currently support case with len(ds_nodes) > 1
            # merge broadcast structures can not have
            # branching in the seperate branches
            # also len == 0 mean this is a terminal node
            return None
        return self.find_merge_node(node.ds_nodes[0], degree, depth+1)

    def wipe(self, node=None):
        #
        # remove the touched tag used to speed up
        # recursive traverse through the model graph
        #
        if node == None:
            node = self.root
        node.wipe()
        for n in node.ds_nodes:
            if node.touched:
                self.wipe(node=n)

    @clean_traverse
    def get_names(self):
        #
        # helper function to get all node names
        #
        names = []
        names = self._get_names(self.root, names)
        return names

    def _get_names(self, node, names):
        names.append(node.name)
        node.touch()
        for n in node.ds_nodes:
            if not n.touched:
                names = self._get_names(n, names)
        return names

    @clean_traverse
    def find_node(self, name):
        #
        # helper function to find a
        # node in the graph by its name
        #
        node = self._find_node(self.root, name)
        if node is None:
            print 'Could not find node named %s' % name
        return node

    def _find_node(self, node, name):
        if node.name == name:
            return node

        for dsnode in node.ds_nodes:
            if not dsnode.touched:
                node = self._find_node(dsnode, name)
                if node is not None:
                    return node

    @clean_traverse
    def get_terminal_nodes(self):
        #
        # help function to get all the terminal nodes
        #
        tnodes = set()
        self._get_terminal_nodes(self.root, tnodes)
        return tnodes

    def _get_terminal_nodes(self, node, tnodes):
        if len(node.ds_nodes) == 0:
            tnodes.add(node)
            return

        for nextnode in node.ds_nodes:
            if not nextnode.touched:
                nextnode.touch()
                self._get_terminal_nodes(nextnode, tnodes)

    def get_desc(self, topnode):
        print '-'*10
        print topnode.name
        for node in topnode.ds_nodes:
            if not node.touched:
                print 'downstream: %s' % (node.name)
        for node in topnode.ds_nodes:
            self.get_desc(node)

    @clean_traverse
    def print_graph(self):
        self.get_desc(self.root)


class NeonNode():
    # 
    # container for neon node which are being initialized
    # from a caffe model
    #
    # Arguments:
    #   ltype (str): layer type
    #   name (str): layer name
    #   loss_layer (bool): True if this is a loss compjutation
    #                      loss/cost in neon are not layer objects
    #                      and need to be treated differently
    #
    # usually this class will not be instantiated directly but
    # rather through the various generator class methods
    # linked to the caffe layer type
    #

    def __init__(self, ltype, name=None, loss_layer=False):
        self.name = name
        self.ds_nodes = []
        self.ltype = ltype
        self.pdict = {}
        self.pdict['type'] = ltype
        self.pdict['config'] = {'name': name}
        self.loss_layer = loss_layer

    @classmethod
    def load_from_caffe_node(cls, node):
        #
        # helper function
        # takes a caffe node and calls the appropriate
        # generator function using the class name
        #
        return getattr(cls, node.ltype)(node)

    # classmethods below are generators which return
    # NeonNode class instances based on the specific layer
    # type
    #
    # the returned instance will contain the proper config
    # dict used by neon
    #
    # for layers with inplace computations (like Bias)
    # multiple NeonNode instances will be generated
    # since neon has seperate layers for these computations
    # also both the new node and the final node (bias node)
    # will be returned

    @classmethod
    def Data(cls, node):
        new_node = cls()
        new_node.name = node.name
        assert len(node.inplace_nodes) == 0
        return (new_node,)

    @classmethod
    def DummyData(cls, node):
        new_node = cls()
        new_node.name = node.name
        assert len(node.inplace_nodes) == 0
        return (new_node,)

    @classmethod
    def MergeBroadcast(cls, end_node, branch_heads, name='none'):
        newlayer = cls('neon.layers.container.MergeBroadcast', name = name + '_inception')
        newlayer.pdict['config']['merge'] = 'depth'

        newlayer.nhead = []
        for cnode in branch_heads:
            new_head = NeonNode.load_from_caffe_node(cnode)
            newlayer.nhead.append(new_head[0])
            nnode = new_head[-1]
            while (True):
                assert len(cnode.ds_nodes) == 1
                cnode = cnode.ds_nodes[0]
                if cnode == end_node:
                    break
                new_node = NeonNode.load_from_caffe_node(cnode)
                nnode.ds_nodes.append(new_node[0])
                nnode = new_node[-1]

        # since this is a container the pdict is more complicated
        # than for a single layer

        # add a merge broadcast container
        newlayer.pdict.update({'container': True})
        newlayer.pdict['config'].update({'layers': [], 'merge': 'depth'})
        for nnode in newlayer.nhead:
            # each branch is in a sequential container
            cont = {'type': 'neon.layers.container.Sequential',
                    'container': True,
                    'config': {'layers': []}}
            node = nnode
            while True:
                # currently not supporteding brancing inside
                # merbe broadcast branches
                # generate the new neon node (last node has 0 ds_nodes)
                assert len(node.ds_nodes) < 2

                cont['config']['layers'].append(node.pdict)
                if len(node.ds_nodes) == 0:
                    break
                node = node.ds_nodes[0]
    
            newlayer.pdict['config']['layers'].append(cont)

        return (newlayer,)

    @classmethod
    def Convolution(cls, node):
        newlayer = cls('neon.layers.layer.Convolution', name=node.layer.name)
        params = cls.parse_layer_params(node.layer.convolution_param, conv=True)
        newlayer.pdict['config'].update(params)
        newlayer.pdict['config']['init'] = {'type': 'neon.initializers.initializer.Constant',
                                            'config': {'val': 0.0}}

        dimc = np.array(node.layer.blobs[0].shape.dim)
        dimn = (np.prod(dimc[1:4]), dimc[0])

        # making it a list first seems to speed things up
        w = np.array(list(node.layer.blobs[0].data)).reshape(dimc).transpose((1,2,3,0))  # neon ordering
        w = w.reshape(dimn)
        newlayer.pdict['params'] = {'W': np.ascontiguousarray(w.astype(np.float32))}

        last_node = newlayer
        if  node.layer.convolution_param.bias_term:
            bias_node = cls.Bias(node.layer)
            newlayer.ds_nodes.append(bias_node[0])
            last_node = bias_node[-1]

        for ipnode in node.inplace_nodes:
            iplayer = NeonNode.load_from_caffe_node(ipnode)
            last_node.ds_nodes.append(iplayer[0])
            last_node = iplayer[-1]
        return (newlayer, last_node)

    @staticmethod
    def parse_layer_params(lparam, conv=False):
        #
        # helper function to parse the layer config
        # parameters for layers like pooling and convolution
        #

        # parse the convolution layer parameters
        fshape = NeonNode.parse_size(lparam, 'kernel')
        fshape = {'R': fshape[0], 'S': fshape[1]}
        if conv:
            assert hasattr(lparam, 'num_output') and lparam.num_output > 0
            fshape['K'] = lparam.num_output

        padding = NeonNode.parse_size(lparam, 'pad')
        if padding[0] == padding[1]:
            padding = padding[0]
        else:
            padding = {'pad_h': padding[0], 'pad_w': padding[1]}

        stride = NeonNode.parse_size(lparam, 'stride')
        if stride[0] == stride[1]:
            stride = stride[0]
        else:
            stride = {'pad_h': stride[0], 'pad_w': stride[1]}

        params = {}
        params['strides'] = stride
        params['padding'] = padding
        params['fshape'] = fshape
        return params

    @staticmethod
    def parse_size(lparam, param):
        return getattr(NeonNode, 'parse_'+param)(lparam)

    @staticmethod
    def parse_kernel(lparam):
        # pasre the kernel size from the caffe config parameters
        # easiest to do this case by case to cover the different caffe protobuf defs
        key = 'kernel_size'
        if hasattr(lparam, key):
            ks = getattr(lparam, key)
            if type(ks) is int and ks > 0:
                return [ks]*2
            
            if type(ks) is list or type(ks) is RepeatedScalarFieldContainer:
                ks = list(ks)
                if len(ks) > 3:
                    raise NotImplementedError()
                if len(ks) > 0 and all([x > 0 for x in ks]):
                    return [ks[0], ks[-1]]
                # ks is [] or has an element that is <= 0

        # try to parse from kernel_h and _w
        key = 'kernel'
        if not (hasattr(lparam, key+'_h') and hasattr(lparam, key+'_w')):
            raise ValueError('Can not parse kernel size')

        k_h = getattr(lparam, key+'_h')
        k_w = getattr(lparam, key+'_w')
        if k_h == 0 and k_w == 0:
            import ipdb; ipdb.set_trace()
            raise ValueError('Can not parse kernel size')
        return [k_h, k_w]

    @staticmethod
    def parse_pad(lparam):
        # pasre the padding size from the caffe config parameters
        # easiest to do this case by case to cover the different caffe protobuf defs
        key = 'pad'
        if hasattr(lparam, key):
            ps = getattr(lparam, key)
            if type(ps) is int:
                return [ps]*2
            
            if type(ps) is list or type(ps) is RepeatedScalarFieldContainer:
                ps = list(ps)
                if len(ps) > 3:
                    raise NotImplementedError()
                if len(ps) > 0:
                    return [ps[0], ps[-1]]

        # default to [0, 0]
        ps = [0, 0]

        try:
            k_h = getattr(lparam, key+'_h')
            ps[0] = k_h
            k_w = getattr(lparam, key+'_w')
            ps[1] = k_w
        except:
            pass
        return ps

    @staticmethod
    def parse_stride(lparam):
        # pasre the strides from the caffe config parameters
        # easiest to do this case by case to cover the different caffe protobuf defs
        key = 'stride'
        if hasattr(lparam, key):
            ss = getattr(lparam, key)
            if type(ss) is int:
                return [ss]*2
            
            if type(ss) is list or type(ss) is RepeatedScalarFieldContainer:
                ss = list(ss)
                if len(ss) > 3:
                    raise NotImplementedError()
                if len(ss) > 0 and all([x > 0 for x in ss]):
                    return [ss[0], ss[-1]]

        # default to [1, 1]
        ss = [1, 1]

        try:
            k_h = getattr(lparam, key+'_h')
            if k_h > 0:
                ss[0] = k_h
            k_w = getattr(lparam, key+'_w')
            if k_w > 0:
                ss[1] = k_w
        except:
            pass
        return ss

    @classmethod
    def Pooling(cls, node):
        newlayer = cls('neon.layers.layer.Pooling', name=node.name)

        op_ind = node.layer.pooling_param.pool

        max_ind = node.layer.pooling_param.MAX
        ave_ind = node.layer.pooling_param.AVE
        assert op_ind in [max_ind, ave_ind], 'Only MAX and AVE pooling supported'

        global_pool = False
        if op_ind == max_ind:
            op = 'max'
        else:
            op = 'avg'
            # check for global pooling
            global_pool = node.layer.pooling_param.global_pooling

        newlayer.pdict['config']['op'] = op

        if not global_pool:
            params = cls.parse_layer_params(node.layer.pooling_param)
            newlayer.pdict['config'].update(params)
        else:
            newlayer.pdict['config'].update({'fshape': 'all'})

        last_node = newlayer
        for ipnode in node.inplace_nodes:
            ipnode = NeonNode.load_from_caffe_node(ipnode)
            last_node.ds_nodes.append(ipnode[0])
            last_node = ipnode[-1]
        
        return (newlayer, last_node)

    @classmethod
    def Bias(cls, parent_layer):
        newlayer = cls('neon.layers.layer.Bias', name=parent_layer.name + '_bias')
        newlayer.pdict['params'] = {'W': np.array(parent_layer.blobs[1].data).copy().astype(np.float32)}
        newlayer.pdict['config']['init'] = {'type': 'neon.initializers.initializer.Constant',
                                            'config': {'val': 0.0}}
        return (newlayer,)
        
    @classmethod
    def InnerProduct(cls, node):
        newlayer = cls('neon.layers.layer.Linear', name=node.name)
        nout = int(node.layer.inner_product_param.num_output)
        newlayer.pdict['config']['nout'] = nout
        newlayer.pdict['config']['init'] = {'type': 'neon.initializers.initializer.Constant',
                                            'config': {'val': 0.0}}

        # blob to list to ndarray is faster then direct to nd array
        w = np.array(list(node.layer.blobs[0].data)).astype(np.float32)
        newlayer.pdict['params'] = {'W': np.copy(w.reshape(nout, -1))}

        last_node = newlayer
        if node.layer.inner_product_param.bias_term:
            bias_node = cls.Bias(node.layer)
            newlayer.ds_nodes.append(bias_node[0])
            last_node = bias_node[-1]

        for ipnode in node.inplace_nodes:
            ipnode = NeonNode.load_from_caffe_node(ipnode)
            last_node.ds_nodes.append(ipnode[0])
            last_node = ipnode[-1]

        return (newlayer, last_node)

    @classmethod
    def LRN(cls, node):
        newlayer = cls('neon.layers.layer.LRN', name = node.name)
        newlayer.pdict['config']['ascale'] = node.layer.lrn_param.alpha
        newlayer.pdict['config']['bpower'] = node.layer.lrn_param.beta
        newlayer.pdict['config']['depth'] = node.layer.lrn_param.local_size
        return (newlayer,)

    @classmethod
    def Dropout(cls, node):
        newlayer = cls('neon.layers.layer.Dropout', name = node.name)
        newlayer.pdict['config']['keep'] = 1.0 - node.layer.dropout_param.dropout_ratio
        return (newlayer,)

    @classmethod
    def activation(cls, act_type, node):
        # macro for activations
        act_type_short = act_type.split('.')[-1]
        newlayer = cls('neon.layers.layer.Activation', name = node.name + '_' + act_type_short)
        newlayer.pdict['config']['transform'] = {'type': act_type}
        return newlayer

    @classmethod
    def ReLU(cls, node):
        return (cls.activation('neon.transforms.activation.Rectlin', node),)

    @classmethod
    def Tanh(cls, node):
        return (cls.activation('neon.transforms.activation.Tanh', node),)

    @classmethod
    def Sigmoid(cls, node):
        return (cls.activation('neon.transforms.activation.Logistic', node),)

    @classmethod
    def Softmax(cls, node):
        return (cls.activation('neon.transforms.activation.Softmax', node),)

    @classmethod
    def SoftmaxWithLoss(cls, node):
        newlayer = cls.Softmax(node)[-1]
        newlayer.ds_nodes.append(cls.CrossEntropyMulti(node)[-1])
        last_layer = newlayer.ds_nodes[0]
        return (newlayer, last_layer)
    
    @classmethod
    def CrossEntropyMulti(cls, node):
        newlayer = cls('neon.layers.layer.GeneralizedCost', name=node.name, loss_layer=True)
        newlayer.pdict['config']['costfunc'] = {'type': 'neon.transforms.cost.CrossEntropyMulti'}
        return (newlayer,)

    @classmethod
    def EuclideanLoss(cls, node):
        # cost not handled here
        # TODO ADD COST
        #newlayer.loss = True
        return (None,)

    @classmethod
    def Accuracy(cls, node):
        # cost not handled here
        # TODO ADD METRIC
        return (None,)


class Decaffeinate():
    #
    # container class for converting caffe model to neon
    #
    # Arguments:
    #   model_file (str): path to caffe prototxt file with model config
    #   param_file (str): path to caffe binary prototxt file with model
    #                     weights
    #

    def __init__(self, model_file, param_file):
        assert os.path.exists(model_file), 'Could not find model file'
        assert os.path.exists(param_file), 'Could not find weights file'

        self.net = caffe_pb2.NetParameter()

        # load the model def prototxt
        with open(model_file, 'r') as fid:
            text_format.Merge(fid.read(), self.net)

        layers = self.net.layer
        if len(layers) == 0:
            raise NotImplementedError('Convert model def prototxt to use new caffe '
                                      'format (layer not layers) [%s]' % model_file)

        # remove layers used for testing only 
        for ind in range(len(layers)-1, -1, -1):
            l = layers[ind]
            if len(l.include) > 0:
                if not any([x.phase == caffe_pb2.TRAIN for x in l.include]):
                    del(layers[ind])

        with open(param_file, 'rb') as fid:
            net_w = caffe_pb2.NetParameter()
            net_w.ParseFromString(fid.read())

        # caffe2neon does not work with the old caffe protobuf format
        if len(net_w.layer) == 0:
            # have had success converting old V1 format to new format by training 
            # for 1 iteration and serializing
            raise NotImplementedError('caffemodel file is using the old V1LayerParameter format '
                                      'try to convert to newest format if possible')
        layer_params = net_w.layer

        # list out the layer names
        lnames = [l.name for l in layer_params]

        # loading the params this way avoids the extra layers 
        # in the caffemodel files (e.g. split) and avoids
        # getting the old style protobuf objects (net.layers instead of net.layer)

        # load up the learned parameters
        data_layers = []
        for l in layers:
            # check for a data layer
            if l.type.lower in ['data', 'dummydata']:
                data_layers.append(l.name)

            try:
                ind = lnames.index(l.name)
            except ValueError:
                print '%s from prototxt file '  % l.name + \
                      'not in layer parameter file'
                print 'continuing without loading any parameters...'
                continue

            l.blobs.extend(net_w.layer[ind].blobs)

        if len(data_layers) == 0:
            print 'Found no data layers in model file'
            print 'Checking for input parameters...'
            assert len(self.net.input_shape) > 0
            print 'Generating dummy data layer for input'
            print 'Assuming input data blob is named "data"'
            data_layer = caffe_pb2.LayerParameter()
            data_layer.name = 'data'
            data_layer.type = 'DummyData'
            data_layer.top.append('data')
            inp_shape = self.net.input_shape
            data_layer.dummy_data_param.shape.MergeFrom(inp_shape)
            newl = layers.add()
            newl.MergeFrom(data_layer)

        # generate the graph
        self.graph = graph(layers)

    def convert_to_neon(self):
        #
        # convert the caffe layers and their params stored
        # in the grpah to neon format
        #
        # Returns:
        #   dict: neon formatted dictionary object which can be 
        #         deserialized in neon to generate the model
        #

        # run down the graph and generate a neon "compatible" graph
        graph = self.graph
        data_node = graph.root
        assert data_node.ltype in ['Data', 'DummyData']
        assert 'data' in data_node.out_names, 'Data layer must have output named "data"'

        # generate neon graph
        self.neon_root = NeonNode('Data', name='root')
        self.neonize(data_node, self.neon_root)

        pdict = {}  # neon formatted model config dict
        self.branch_cnt = 0

        # check if this model is tree or sequential
        tnodes = graph.get_terminal_nodes()
        if len(tnodes) > 1:
            ISTREE = True
            pdict['model'] = {'type': 'neon.layers.container.Tree'}
            lwght = {}
            # assign the loss weights
            # note that the neon and caffe terminal nodes may 
            # be in different orders so need to do this by name
            for tnode in tnodes:
                try:
                    loss_weight = tnode.layer.loss_weight
                    assert len(loss_weight) < 2
                    lwght[tnode.name] = loss_weight[0]
                except:
                    warn = 'Could not properly parse loss_weight for '
                    warn += 'layer %s, using 1.0 by default'
                    print warn % tnode.name
                    lwght[tnode.name] = 1.0
            seq_layer = {'type': 'neon.layers.container.Sequential',
                         'config': {'layers': [], 'name': 'main' }}
            pdict['model']['config'] = {'layers': [seq_layer]}
        else:
            ISTREE = False
            pdict['model'] = {'type': 'neon.layers.container.Sequential'}
            pdict['model']['config'] = {'layers': []}

        # build up the top level of the dict
        pdict['model']['container'] = True
        pdict['model']['config']['name'] = 'main branch'
        pdict['model'] = self.descend_model(pdict['model'], self.neon_root)
        
        # add in the layer weights, map them by name
        weights = []
        if ISTREE:
            # will use a multicost layer for tree models
            cdict = {'type': 'neon.layers.container.Multicost'}
            cdict['config'] = {'costs': [], 'weights': []}
            for tnode in pdict['model']['config']['layers']:
                lnode = tnode['config']['layers'][-1]
                name = lnode['config']['name']
                assert name in lwght
                cdict['config']['weights'].append(lwght[name])

                if lnode['type'] == 'neon.layers.layer.GeneralizedCost':
                    # pop cost layers off the stack
                    tnode['config']['layers'].pop()
                    cdict['config']['costs'].append(lnode)
                else:
                    cdict['config']['costs'].append(None)

            cost_present = [tmpc is not None for tmpc in cdict['config']['costs']]
            if all(cost_present):
                pdict['cost'] = cdict
            else:
                assert not any(cost_present), 'Missing cost for some branches of tree'
        else:
            # sequential model uses a single cost function
            head = pdict['model']['config']['layers']
            lnode = head[-1]
            if lnode['type'] == 'neon.layers.layer.GeneralizedCost':
                # pop cost layers off the stack
                head.pop()
                cdict = lnode
                pdict['cost'] = cdict
            else:
                pdict['cost'] = None

        # set backend for caffe compatibility
        pdict['backend'] = {'compat_mode': 'caffe'}
        return pdict

    @staticmethod
    def add_node(cur_cont, node):
        if node.pdict['type'].lower() != 'data':
            cur_cont.append(node.pdict)
        return

    def descend_model(self, pdict, nnode):
        #
        # recursive function which traverses the neon
        # graph and generates the neon formatted config
        # dictionary
        #
        # Arguments:
        #   pdict (dict): neon config dictionary, will be
        #                 updated with nnode config
        #   nnode (NeonNode): current neon graph node
        #

        dsnodes = nnode.ds_nodes

        # short cut to current container head
        head = pdict['config']['layers']
        ISTREE = pdict['type'] == 'neon.layers.container.Tree'

        # short cut to current layer config
        # len(head) == 0 is the case for a sequential main container
        if ISTREE:
            cur_cont = head[-1]['config']['layers']
        else:
            cur_cont = head

        if len(dsnodes) > 1:
            # add a branch node
            self.branch_cnt += 1
            branch_node = {'type': 'neon.layers.layer.BranchNode', 
                           'config': {'name': 'branch_%d' % self.branch_cnt}}

            # will use this as the main branch
            self.add_node(cur_cont, nnode)

            # add the branch node
            cur_cont.append(branch_node)

            # first downstream node is the main branch
            pdict = self.descend_model(pdict, dsnodes.pop(0))

            # the other nodes will need to be added as new branches
            for node in dsnodes:
                # add a new container and the branch node
                new_cont = {'type': 'neon.layers.container.Sequential',
                            'container': True,
                            'config': {'layers': [branch_node]}}
                head.append(new_cont)

                # recurse down the structure
                pdict = self.descend_model(pdict, node)
        elif len(dsnodes) == 1:
            self.add_node(cur_cont, nnode)
            pdict = self.descend_model(pdict, dsnodes[0])
        else:
            self.add_node(cur_cont, nnode)
        return pdict

    def neonize(self, cnode, nnode):
        #
        # convert caffe graph to neon graph
        #
        # Arguments:
        #   cnode (GNode): caffe graph node
        #   nnode (NeonNode): neon graph node
        #
        # will generate a graph starting at self.neon_root

        allinds = range(len(cnode.ds_nodes))
        if len(cnode.ds_nodes) > 1:
            # check for inception/mergeBroadcast type layer
            mb_node = self.graph.check_merge_broadcast(cnode)
            if mb_node:
                end_node, mbinds = mb_node
                branches = [cnode.ds_nodes[ind] for ind in mbinds]

                mb_node = NeonNode.MergeBroadcast(end_node, branches, name=cnode.name)
                nnode.ds_nodes.append(mb_node[0])

                for ind in mbinds:
                    allinds.remove(ind)

                self.neonize(end_node, mb_node[-1])

        for ind in allinds:
            new_node = NeonNode.load_from_caffe_node(cnode.ds_nodes[ind])
            nnode.ds_nodes.append(new_node[0])
            new_node = new_node[-1]
            self.neonize(cnode.ds_nodes[ind], new_node)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', help='path to caffe model text prototxt file')
    parser.add_argument('param_file', help='path to caffe serialized model parameter files'
                                           '(binary prototxt/.caffemodel file)')
    parser.add_argument('--output_file', '-o', default=None,
                        help='output file (neon serialization format)')
    args = parser.parse_args()
    convert = Decaffeinate(args.model_file, args.param_file)
    pdict = convert.convert_to_neon()
    if args.output_file is None:
        out_file = os.path.splitext(args.param_file)[0] + '.prm'
    else:
        out_file = args.output_file
    print 'Saving neon model to %s' % out_file
    with open(out_file, 'w') as fid:
        pickle.dump(pdict, fid)
