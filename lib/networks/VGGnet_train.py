import tensorflow as tf
from networks.network import Network

from fast_rcnn.config import cfg

n_classes = 7


class VGGnet_train(Network):
    def __init__(self, trainable=True, anchor_scales=[8, 16, 32],
                 feat_stride=[16, ], low_level_trainable=False,
                 anchor_ratios=[0.5, 1, 2], transform_img=False):
        self.inputs = []
        self._anchor_scales = anchor_scales
        self._feat_stride = feat_stride
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        #target_size = cfg.TRAIN.SCALES[0]
        #self.data = tf.placeholder(tf.float32, shape=[1, target_size, target_size, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict(
            {'data': self.data, 'im_info': self.im_info, 'gt_boxes': self.gt_boxes})
        self.trainable = trainable
        self.low_level_trainable = low_level_trainable
        self.anchor_ratio_size = len(anchor_ratios)
        self.anchor_ratios = anchor_ratios
        self.transform_img = transform_img
        self.setup()

        # create ops and placeholders for bbox normalization process
        with tf.variable_scope('bbox_pred', reuse=True):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")

            self.bbox_weights = tf.placeholder(
                weights.dtype, shape=weights.get_shape())
            self.bbox_biases = tf.placeholder(
                biases.dtype, shape=biases.get_shape())

            self.bbox_weights_assign = weights.assign(self.bbox_weights)
            self.bbox_bias_assign = biases.assign(self.bbox_biases)

    def setup(self):
        (self.feed('data')
         #.spatial_transform(name='spt_trans', do_transform=self.transform_img)
             .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=self.low_level_trainable)
             .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=self.low_level_trainable)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=self.low_level_trainable)
             .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=self.low_level_trainable)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 512, 1, 1, name='conv4_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .conv(3, 3, 512, 1, 1, name='conv5_3'))
        #========= RPN ============
        (self.feed('conv5_3')
             .conv(3, 3, 512, 1, 1, name='rpn_conv/3x3')
             .conv(1, 1, len(self._anchor_scales) * self.anchor_ratio_size * 2, 1,
                   1, padding='VALID', relu=False, name='rpn_cls_score'))

        (self.feed('rpn_cls_score', 'gt_boxes', 'im_info', 'data')  # , 'spt_trans')
             .anchor_target_layer(self._feat_stride, self._anchor_scales, self.anchor_ratios, name='rpn-data'))

        # Loss of rpn_cls & rpn_boxes

        # directly predict the "delta shift", thus t_x, t_y, t_w, t_h in the Eq(2)
        # of the original Faster-RCNN paper
        # it DOES NOT directly predict the four raw coordinates of bbox corresponding
        # to each anchor
        (self.feed('rpn_conv/3x3')
             .conv(1, 1, len(self._anchor_scales) * self.anchor_ratio_size * 4, 1,
                   1, padding='VALID', relu=False, name='rpn_bbox_pred'))

        #========= RoI Proposal ============
        (self.feed('rpn_cls_score')
             .reshape_layer(2, name='rpn_cls_score_reshape')
             .softmax(name='rpn_cls_prob'))

        (self.feed('rpn_cls_prob')
             .reshape_layer(len(self._anchor_scales) * self.anchor_ratio_size * 2,
                            name='rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
             .proposal_layer(self._feat_stride, self._anchor_scales, self.anchor_ratios, 'TRAIN', name='rpn_rois'))

        (self.feed('rpn_rois', 'gt_boxes')  # , 'spt_trans')
             .proposal_target_layer(n_classes, name='roi-data'))

        #========= RCNN ============
        (self.feed('conv5_3', 'roi-data')
         #.roi_pool(7, 7, 1.0/16, name='pool_5')
             .st_pool(7, 7, 1.0 / 16, name='pool_5', phase='TRAIN')
         # FC6 input shape [RPN_POST_NMS_TOP_N, 7, 7, 512]
             .fc(4096, name='fc6')
             .dropout(0.5, name='drop6')
             .fc(4096, name='fc7')
             .dropout(0.5, name='drop7')
             .fc(n_classes, relu=False, name='cls_score')
             .softmax(name='cls_prob'))

        (self.feed('drop7')
             .fc(n_classes * 4, relu=False, name='bbox_pred'))
