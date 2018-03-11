import numpy as np
import tensorflow as tf

from rpn_msr.proposal_layer_tf import proposal_layer as proposal_layer_py
from rpn_msr.anchor_target_layer_tf import anchor_target_layer as anchor_target_layer_py
from rpn_msr.proposal_target_layer_tf import proposal_target_layer as proposal_target_layer_py
from spatial_transformer import transformer, batch_transformer

from fast_rcnn.config import cfg


DEFAULT_PADDING = 'SAME'
identity = np.array([[1., 0., 0.],
                     [0., 1., 0.]], dtype=np.float32)
identity = identity.flatten()
identity_theta = tf.Variable(initial_value=identity)

DEBUG = False

def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs)==0:
            raise RuntimeError('No input variables found for layer %s.'%name)
        elif len(self.inputs)==1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self
    return layer_decorated

class Network(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, saver, ignore_missing=False,
             load_lowlevel_only=False):
        #if data_path.endswith('.ckpt'):
        if (not data_path.endswith('.npy')):
            saver.restore(session, data_path)
        else: # load pre-grained vgg-net convnet model (for feature extraction)
            data_dict = np.load(data_path).item()
            skip_list = ['fc6', 'fc7']
            if (load_lowlevel_only):
                skip_list += ['conv5_1', 'conv5_2', 'conv5_3']
            for key in data_dict:
                if (key in skip_list):
                    print("Skipping %s" % key)
                    continue
                with tf.variable_scope(key, reuse=True):
                    for subkey in data_dict[key]:
                        try:
                            var = tf.get_variable(subkey)
                            session.run(var.assign(data_dict[key][subkey]))
                            print "assign pretrain model "+subkey+ " to "+key
                        except ValueError:
                            print "ignore "+key
                            if not ignore_missing:
                                raise


    def feed(self, *args):
        assert len(args)!=0
        self.inputs = []
        for layer in args:
            if isinstance(layer, basestring):
                try:
                    layer = self.layers[layer]
                    print layer
                except KeyError:
                    print self.layers.keys()
                    raise KeyError('Unknown layer name fed: %s'%layer)
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print self.layers.keys()
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t,_ in self.layers.items())+1
        return '%s_%d'%(prefix, id)

    def make_var(self, name, shape, initializer=None, trainable=True):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING, group=1, trainable=True):
        """
        k_h:    kernel height
        k_w:    kernel wideth
        c_o:    channel output
        s_h:    strides height
        s_w:    stirdes width
        """
        if (isinstance(input, tuple)):
            input = input[0] # spatial transformer output, only consider data

        self.validate_padding(padding)
        c_i = input.get_shape()[-1]  #channel input
        assert c_i%group==0
        assert c_o%group==0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i/group, c_o], init_weights, trainable)
            biases = self.make_var('biases', [c_o], init_biases, trainable)

            if group==1:
                conv = convolve(input, kernel)
            else:
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
                conv = tf.concat(3, output_groups)
            if relu:
                bias = tf.nn.bias_add(conv, biases)
                return tf.nn.relu(bias, name=scope.name)
            return tf.nn.bias_add(conv, biases, name=scope.name)

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def proposal_layer(self, input, _feat_stride, anchor_scales, anchor_ratios, cfg_key, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        return tf.reshape(tf.py_func(proposal_layer_py,[input[0],input[1],input[2], cfg_key, _feat_stride, anchor_scales, anchor_ratios], [tf.float32]),[-1,5],name =name)


    @layer
    def anchor_target_layer(self, input, _feat_stride, anchor_scales, anchor_ratios, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        with tf.variable_scope(name) as scope:

            rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights =\
             tf.py_func(anchor_target_layer_py,[input[0],input[1],input[2],input[3], #input[4][1],
             _feat_stride, anchor_scales, anchor_ratios],[tf.float32,tf.float32,tf.float32,tf.float32])

            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels,tf.int32), name = 'rpn_labels')
            rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name = 'rpn_bbox_targets')
            rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights , name = 'rpn_bbox_inside_weights')
            rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights , name = 'rpn_bbox_outside_weights')


            return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


    @layer
    def proposal_target_layer(self, input, classes, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        with tf.variable_scope(name) as scope:

            rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights =\
             tf.py_func(proposal_target_layer_py,
                        [input[0], input[1], classes],
                        [tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])

            rois = tf.reshape(rois,[-1,5] , name = 'rois')
            labels = tf.convert_to_tensor(tf.cast(labels,tf.int32), name = 'labels')
            bbox_targets = tf.convert_to_tensor(bbox_targets, name = 'bbox_targets')
            bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name = 'bbox_inside_weights')
            bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights, name = 'bbox_outside_weights')


            return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


    @layer
    def reshape_layer(self, input, d,name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob_reshape':
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),[input_shape[0],
                    int(d),tf.cast(tf.cast(input_shape[1],tf.float32)/tf.cast(d,tf.float32)*tf.cast(input_shape[3],tf.float32),tf.int32),input_shape[2]]),[0,2,3,1],name=name)
        else:
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),[input_shape[0],
                    int(d),tf.cast(tf.cast(input_shape[1],tf.float32)*(tf.cast(input_shape[3],tf.float32)/tf.cast(d,tf.float32)),tf.int32),input_shape[2]]),[0,2,3,1],name=name)

    @layer
    def feature_extrapolating(self, input, scales_base, num_scale_base, num_per_octave, name):
        return feature_extrapolating_op.feature_extrapolating(input,
                              scales_base,
                              num_scale_base,
                              num_per_octave,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(concat_dim=axis, values=inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True, trainable=True):
        if (DEBUG and 'fc6' == name):
            print('FC6 input shape {0}'.format(input.get_shape().as_list()))
        with tf.variable_scope(name) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(tf.transpose(input,[0,3,1,2]), [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            if name == 'bbox_pred':
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
                init_biases = tf.constant_initializer(0.0)
            else:
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
                init_biases = tf.constant_initializer(0.0)

            weights = self.make_var('weights', [dim, num_out], init_weights, trainable)
            biases = self.make_var('biases', [num_out], init_biases, trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def st_pool(self, input, pooled_height, pooled_width, spatial_scale, name, phase):
        """
        Spatial Transformer-based RoI pooling

        input shape:    blob shape: [cfg.TRAIN.BATCH_SIZE, 5], proposal shape: [cfg.TRAIN.BATCH_SIZE, 4]
        output shape:   [cfg.TRAIN.BATCH_SIZE, 7, 7, 512]
        """
        # only use the first input
        with tf.variable_scope(name) as scope:
            if isinstance(input[0], tuple):
                input[0] = input[0][0]

            if isinstance(input[1], tuple):
                input[1] = input[1][0]

            print input
            if ('TRAIN' == phase):
                num_prop = cfg.TRAIN.BATCH_SIZE
                img_size = cfg.TRAIN.SCALES[0]
                #angle = tf.random_uniform([], maxval=np.pi)
            elif ('TEST' == phase):
                num_prop = cfg.TEST.RPN_POST_NMS_TOP_N
                img_size = cfg.TEST.SCALES[0]
                #angle = 0.0
            else:
                raise Exception('unknown phase for st_pool')

            #s_ang = tf.sin(angle)
            #c_ang = tf.cos(angle)

            proposals = tf.reshape(input[1], [num_prop, 5])
            proposals = proposals * spatial_scale
            out_size = (pooled_height, pooled_width)
            Wp = np.floor(img_size * spatial_scale)
            Hp = np.floor(img_size * spatial_scale)
            W = tf.convert_to_tensor(Wp, dtype=tf.float32)
            H = tf.convert_to_tensor(Hp, dtype=tf.float32)

            old_shape = input[0].get_shape().as_list()

            conv5_3 = tf.reshape(input[0], [1, int(Wp), int(Hp), old_shape[-1]])  # shape = [1, 37, 37, 512]

            x1v = tf.slice(proposals, [0, 1], [num_prop, 1])
            #x1v = tf.Print(x1v, [x1v, "x1v value"])
            x2v = tf.slice(proposals, [0, 3], [num_prop, 1])
            y1v = tf.slice(proposals, [0, 2], [num_prop, 1])
            y2v = tf.slice(proposals, [0, 4], [num_prop, 1])
            #y2v = tf.Print(y2v, [y2v, "y2v value"])

            xc = tf.divide(tf.add(x1v, x2v), 2.0)
            yc = tf.divide(tf.add(y1v, y2v), 2.0)
            w = tf.subtract(x2v, x1v)
            h = tf.subtract(y2v, y1v)

            h_translate_p = tf.subtract(tf.subtract(tf.multiply(2.0, yc), H), 1.0)
            h_translate = tf.divide(h_translate_p, tf.subtract(H, 1.0))
            row2 = tf.concat([np.zeros([num_prop, 1]), tf.divide(h, H), h_translate], axis=1)
            #row2 = tf.concat([s_ang * tf.divide(w, W), tf.divide(h, H) * c_ang, h_translate], axis=1)

            w_translate_p = tf.subtract(tf.subtract(tf.multiply(2.0, xc), W), 1.0)
            w_translate = tf.divide(w_translate_p, tf.subtract(W, 1.0))
            row1 = tf.concat([tf.divide(w, W), np.zeros([num_prop, 1]), w_translate], axis=1)
            #row1 = tf.concat([tf.divide(w, W) * c_ang, -1 * s_ang * tf.divide(w, W), w_translate], axis=1)

            thetas = tf.stack([row1, row2], axis=1)
            thetas = tf.reshape(thetas, [1, num_prop, 6])

            return batch_transformer(conv5_3, thetas, out_size)

    @layer
    def spatial_transform(self, input, name, do_transform=False, num_hidden=20,
                          rotation_dim=1, keep_prob=0.7):
        """
        Based on https://github.com/daviddao/spatial-transformer-tensorflow/\
        blob/master/cluttered_mnist.py
        """
        if (not do_transform):
            return input, identity_theta
        with tf.variable_scope(name) as scope:
            if isinstance(input, tuple):
                input = input[0]
            input_shape = input.get_shape().as_list() # used for output shape
            w_shape_1 = [input_shape[1] * input_shape[2] * input_shape[3],
                         num_hidden]
            out_size = (input_shape[1], input_shape[2]) #remain the same size
            init_weights1 = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_weights2 = tf.truncated_normal_initializer(0.0, stddev=0.01)
            x = tf.reshape(input, [-1, w_shape_1[0]])
            W_fc_loc1 = self.make_var('loc_weights_1', w_shape_1,
                                      init_weights1)#tf.constant_initializer(0.0))
            b_fc_loc1 = self.make_var('loc_biases_1', [num_hidden],
                                      tf.constant_initializer(0.0))
            W_fc_loc2 = self.make_var('loc_weights_2', [num_hidden, rotation_dim],
                                      init_weights2)#tf.constant_initializer(0.0))
            #initial = np.array([[1, 0, 0], [0, 1, 0]]).astype('float32').flatten()
            #initial = np.array([[0.0]]).astype('float32').flatten()
            b_fc_loc2 = self.make_var('loc_biases_2', [rotation_dim],
                                      tf.constant_initializer(0.0))

            # Define the two layer localisation network
            h_fc_loc1 = tf.nn.relu_layer(x, W_fc_loc1, b_fc_loc1,
                                         name=scope.name + '_loc1')
            # h_fc_loc1 = tf.nn.tanh(tf.matmul(x, W_fc_loc1) + b_fc_loc1, name=scope.name + '_loc1')
            if (keep_prob < 1):
                h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob=keep_prob)
            else:
                h_fc_loc1_drop = h_fc_loc1

            # %% Second layer
            #h_fc_loc2 = tf.nn.relu_layer(h_fc_loc1_drop, W_fc_loc2,
                                         #b_fc_loc2, name=scope.name + '_loc2')
            # h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, W_fc_loc2) +
            #                              b_fc_loc2, name=scope.name + '_loc2')
            #alpha = tf.nn.relu_layer(tf.matmul(h_fc_loc1_drop, W_fc_loc2) +
            #                             b_fc_loc2, name=scope.name + '_loc2')
            alpha = tf.nn.xw_plus_b(h_fc_loc1_drop, W_fc_loc2, b_fc_loc2,
                                    name=scope.name + '_loc2')

    	    #print("h_fc_loc2 shape = {0}".format(h_fc_loc2.get_shape().as_list()))
    	    #h_fc_loc2 = tf.Print(h_fc_loc2, [h_fc_loc2, "h_fc_loc2 value"])
            #h_fc_loc2_max = tf.argmax(h_fc_loc2, 1)
            #h_fc_loc2_max = tf.nn.softmax(h_fc_loc2)
            # TODO but argmax is not differentiable! see
            #https://www.reddit.com/r/MachineLearning/comments/4e2get/argmax_differentiable/

            #print("h_fc_loc2_max.shape = {0}".format(h_fc_loc2_max.get_shape().as_list()))
            #h_fc_loc2_max = tf.reshape(h_fc_loc2_max, [1, 1])
            #h_fc_loc2_max = tf.to_float(h_fc_loc2_max)

            #alpha = tf.multiply(tf.reshape(h_fc_loc2, []), # convert to scalar
            #alpha = tf.multiply(h_fc_loc2_max, # convert to scalar
                                #tf.convert_to_tensor(2 * np.pi / rotation_dim, dtype=tf.float32))
            #alpha = tf.Print(alpha, [alpha, "alpha value"])
            #print("alpha shape = {0}".format(alpha.get_shape().as_list()))
            sin_alpha = tf.sin(alpha)
            #sin_alpha = tf.Print(sin_alpha, [sin_alpha, "sin_alpha value"])
            cos_alpha = tf.cos(alpha)
            #cos_alpha = tf.Print(cos_alpha, [cos_alpha, "cos_alpha value"])
            n_sin_alpha = tf.multiply(sin_alpha,
                                      tf.convert_to_tensor(-1.0, dtype=tf.float32))
	        #print("n_sin_alpha shape = {0}".format(n_sin_alpha.get_shape().as_list()))
            scale_tensor = tf.convert_to_tensor([[1.0]], dtype=tf.float32)
            scale_tensor = tf.divide(scale_tensor,
                                     tf.add(tf.abs(sin_alpha), tf.abs(cos_alpha)))
            sin_alpha = tf.multiply(scale_tensor, sin_alpha)
            cos_alpha = tf.multiply(scale_tensor, cos_alpha)
            n_sin_alpha = tf.multiply(scale_tensor, n_sin_alpha)

            zero_tensor = tf.convert_to_tensor([[0.0]], dtype=tf.float32)
            #print("scale_tensor shape = {0}".format(scale_tensor.get_shape().as_list()))
            row1 = tf.concat([cos_alpha, n_sin_alpha, zero_tensor], axis=1)
            row1 = tf.reshape(row1, [3])
            #print("row1 shape = {0}".format(row1.get_shape().as_list()))
            row2 = tf.concat([sin_alpha, cos_alpha, zero_tensor], axis=1)
            row2 = tf.reshape(row2, [3])
            #print("row2 shape = {0}".format(row2.get_shape().as_list()))
            theta = tf.stack([row1, row2], axis=0)
            theta_shape = theta.get_shape().as_list()
            #print("theta shape = {0}".format(theta_shape))
            assert(theta_shape[0] == 2 and theta_shape[1] == 3)
            h_trans = transformer(input, theta, out_size)
            #print("transformed shape", h_trans.get_shape().as_list())
            return h_trans, theta

    @layer
    def softmax(self, input, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob':
            return tf.reshape(tf.nn.softmax(tf.reshape(input,[-1,input_shape[3]])),[-1,input_shape[1],input_shape[2],input_shape[3]],name=name)
        else:
            return tf.nn.softmax(input,name=name)

    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)
