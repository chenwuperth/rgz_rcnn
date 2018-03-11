# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
#
# Modified by Chen Wu (chen.wu@icrar.org)
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

from fast_rcnn.config import cfg
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer
import numpy as np
import os
import tensorflow as tf
import sys
from tensorflow.python.client import timeline
import time


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, sess, saver, network, imdb, roidb, output_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        print 'Computing bounding-box regression targets...'
        if cfg.TRAIN.BBOX_REG:
            self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(
                roidb)
        print 'done'

        # For checkpoint
        self.saver = saver

    def snapshot(self, sess, iter_num):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED and \
                        cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            # save original values
            with tf.variable_scope('bbox_pred', reuse=True):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")

            orig_0 = weights.eval()
            orig_1 = biases.eval()

            # scale and shift with bbox reg unnormalization; then save snapshot
            weights_shape = weights.get_shape().as_list()
            sess.run(net.bbox_weights_assign, feed_dict={
                     net.bbox_weights: orig_0 * np.tile(self.bbox_stds, (weights_shape[0], 1))})
            sess.run(net.bbox_bias_assign, feed_dict={
                     net.bbox_biases: orig_1 * self.bbox_stds + self.bbox_means})

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        # remove iter_num in the file name, let tensorflow manage it via global_step
        modelname = (cfg.TRAIN.SNAPSHOT_PREFIX + infix)  # +
        #'_iter_num_{:d}'.format(iter_num+1) + '.ckpt)
        modelname = os.path.join(self.output_dir, modelname)

        snapshot_file = self.saver.save(sess, modelname,
                                        global_step=iter_num + 1)
        print 'Wrote snapshot to: {:s}'.format(snapshot_file)

        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED and cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            with tf.variable_scope('bbox_pred', reuse=True):
                # restore net to original state
                sess.run(net.bbox_weights_assign, feed_dict={
                         net.bbox_weights: orig_0})
                sess.run(net.bbox_bias_assign, feed_dict={
                         net.bbox_biases: orig_1})

    def _zerofy_non_class_bbox(self, bbox_pred, lbn):
        label, batch_size, num_classes = lbn
        M = batch_size # batch size
        C = num_classes # number of classes
        N = 4 # four coordinates per subject

        one_hot_tensor = tf.one_hot(label, num_classes)
        A2 = tf.reshape(one_hot_tensor, [C, M, 1])
        A2_tile = tf.tile(A2, [1, 1, N])
        A2_final = tf.reshape(A2_tile, [M, C * N])

        return tf.multiply(bbox_pred, A2_final)

    def _modified_smooth_l1(self, sigma, bbox_pred, bbox_targets,
                            bbox_inside_weights, bbox_outside_weights,
                            lbn=None):
        """
            ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        """
        if (lbn is not None):
            bbox_pred = self._zerofy_non_class_bbox(bbox_pred, lbn)

        sigma2 = sigma * sigma

        inside_mul = tf.multiply(
            bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

        smooth_l1_sign = tf.cast(
            tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(
            tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

        return outside_mul

    def train_model(self, sess, max_iters, start_iter=0):
        """Network training loop."""

        data_layer = get_data_layer(self.roidb, self.imdb.num_classes)

        # RPN
        # classification loss
        rpn_cls_score = tf.reshape(self.net.get_output(
            'rpn_cls_score_reshape'), [-1, 2])
        rpn_label = tf.reshape(self.net.get_output('rpn-data')[0], [-1])
        rpn_cls_score = tf.reshape(
            tf.gather(rpn_cls_score, tf.where(tf.not_equal(rpn_label, -1))), [-1, 2])
        rpn_label = tf.reshape(
            tf.gather(rpn_label, tf.where(tf.not_equal(rpn_label, -1))), [-1])
        rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=rpn_cls_score, labels=rpn_label))

        # bounding box regression L1 loss
        rpn_bbox_pred = self.net.get_output('rpn_bbox_pred')
        rpn_bbox_targets = tf.transpose(
            self.net.get_output('rpn-data')[1], [0, 2, 3, 1])
        rpn_bbox_inside_weights = tf.transpose(
            self.net.get_output('rpn-data')[2], [0, 2, 3, 1])
        rpn_bbox_outside_weights = tf.transpose(
            self.net.get_output('rpn-data')[3], [0, 2, 3, 1])

        rpn_smooth_l1 = self._modified_smooth_l1(
            3.0, rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)
        rpn_loss_box = tf.reduce_mean(tf.reduce_sum(
            rpn_smooth_l1, reduction_indices=[1, 2, 3]))

        # R-CNN
        # classification loss
        cls_score = self.net.get_output('cls_score')
        batch_size, num_classes = cls_score.get_shape().as_list()
        label = tf.reshape(self.net.get_output('roi-data')[1], [-1])
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

        # bounding box regression L1 loss
        bbox_pred = self.net.get_output('bbox_pred')
        bbox_targets = self.net.get_output('roi-data')[2]
        bbox_inside_weights = self.net.get_output('roi-data')[3]
        bbox_outside_weights = self.net.get_output('roi-data')[4]

        if (cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED):
            hl_sigma = 1.0
        else:
            hl_sigma = 3.0
        smooth_l1 = self._modified_smooth_l1(
            hl_sigma, bbox_pred, bbox_targets, bbox_inside_weights,
                    bbox_outside_weights, lbn=(label, batch_size, num_classes))
        loss_box = tf.reduce_mean(tf.reduce_sum(
            smooth_l1, reduction_indices=[1]))

        # final loss
        loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box

        # optimizer and learning rate
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step,
                                        cfg.TRAIN.STEPSIZE, 0.1, staircase=True)
        momentum = cfg.TRAIN.MOMENTUM
        train_op = tf.train.MomentumOptimizer(
            lr, momentum).minimize(loss, global_step=global_step)

        # iintialize variables
        sess.run(tf.global_variables_initializer())
        if self.pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            self.net.load(self.pretrained_model, sess, self.saver, True)

        last_snapshot_iter = -1
        timer = Timer()
        check_thres = max_iters * 0.8
        for iter in range(start_iter, start_iter + max_iters):
            # get one batch
            blobs = data_layer.forward()

            # Make one SGD update
            feed_dict = {self.net.data: blobs['data'],
                         self.net.im_info: blobs['im_info'],
                         self.net.keep_prob: 0.5,
                         self.net.gt_boxes: blobs['gt_boxes']}

            run_options = None
            run_metadata = None
            if cfg.TRAIN.DEBUG_TIMELINE:
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            timer.tic()

            rpn_loss_cls_value, rpn_loss_box_value, loss_cls_value, loss_box_value, lrate, _\
             = sess.run([rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, lr, train_op],
                        feed_dict=feed_dict,
                        options=run_options,
                        run_metadata=run_metadata)

            timer.toc()

            if cfg.TRAIN.DEBUG_TIMELINE:
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open(str(long(time.time() * 1000)) +
                                  '-train-timeline.ctf.json', 'w')
                trace_file.write(
                    trace.generate_chrome_trace_format(show_memory=False))
                trace_file.close()

            total_loss = rpn_loss_cls_value + rpn_loss_box_value + loss_cls_value + loss_box_value
            if (iter + 1) % (cfg.TRAIN.DISPLAY) == 0:
                print 'iter: %d / %d, total loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, loss_box: %.4f, lr: %f' %\
                    (iter + 1, max_iters + start_iter, total_loss,
                     rpn_loss_cls_value, rpn_loss_box_value, loss_cls_value, loss_box_value, lrate)
                print 'speed: {:.3f}s / iter.'.format(timer.average_time)

            # find out who is the culprit...
            if (total_loss > 0.2 and ((iter - start_iter) > check_thres)):
                # save them in a list, which can be used as a test set to
                # show what the model predicts and then compare that with gt
                # visually
                print("Culprit found %s, %.4f" % (blobs['img_id'], total_loss))

            if (iter + 1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    if cfg.TRAIN.HAS_RPN:
        rdl_roidb.prepare_roidb(imdb)
    else:
        rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb


def get_data_layer(roidb, num_classes):
    """return a data layer."""
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            layer = GtDataLayer(roidb)
        else:
            layer = RoIDataLayer(roidb, num_classes)
    else:
        layer = RoIDataLayer(roidb, num_classes)

    return layer


def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after)
    return filtered_roidb


def train_net(network, imdb, roidb, output_dir, pretrained_model=None,
              max_iters=40000, start_iter=0):
    """Train a Fast R-CNN network."""
    roidb = filter_roidb(roidb)
    saver = tf.train.Saver(max_to_keep=100)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sw = SolverWrapper(sess, saver, network, imdb, roidb,
                           output_dir, pretrained_model=pretrained_model)
        print 'Solving...'
        sw.train_model(sess, max_iters, start_iter=start_iter)
        print 'done solving'
