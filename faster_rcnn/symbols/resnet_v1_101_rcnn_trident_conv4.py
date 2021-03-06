# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Guodong Zhang, Bin Xiao
# --------------------------------------------------------

import cPickle
import mxnet as mx
from utils.symbol import Symbol
from operator_py.proposal import *
from operator_py.proposal_target_trident import *
from operator_py.box_annotator_ohem import *


class resnet_v1_101_rcnn_trident_conv4(Symbol):
    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.use_global_stats = True
        self.workspace = 512
        self.units = (3, 4, 23, 3) # use for 101
        self.filter_list = [256, 512, 1024, 2048]

    def get_resnet_v1_conv1(self, data):
        conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=64, pad=(3, 3), kernel=(7, 7), stride=(2, 2),
                                      no_bias=True)
        bn_conv1 = mx.symbol.BatchNorm(name='bn_conv1', data=conv1, use_global_stats=True, fix_gamma=False, eps=self.eps)
        scale_conv1 = bn_conv1
        conv1_relu = mx.symbol.Activation(name='conv1_relu', data=scale_conv1, act_type='relu')
        pool1 = mx.symbol.Pooling(name='pool1', data=conv1_relu, pooling_convention='full', pad=(0, 0), kernel=(3, 3),
                                  stride=(2, 2), pool_type='max')
        return pool1

    def get_resnet_v1_conv_var(self, stage, blocks):

        if blocks == 3:
            block_name = ['b' , 'c']
        else:
            block_name = ['b%d'%i for i in range(1, blocks)]
        branch_name = ['1', '2a', '2b', '2c']
        names = []

        for j in range(4):
            names.append('res%da_branch%s_weight'%(stage, branch_name[j]))
            names.append('bn%da_branch%s_moving_mean'%(stage, branch_name[j]))
            names.append('bn%da_branch%s_moving_var'%(stage, branch_name[j]))
            names.append('bn%da_branch%s_gamma'%(stage, branch_name[j]))
            names.append('bn%da_branch%s_beta'%(stage, branch_name[j]))

        for i in range(1, blocks):
            for j in range(1, 4):
                names.append('res%d%s_branch%s_weight'%(stage, block_name[i-1], branch_name[j]))
                names.append('bn%d%s_branch%s_moving_mean'%(stage, block_name[i-1], branch_name[j]))
                names.append('bn%d%s_branch%s_moving_var'%(stage, block_name[i-1], branch_name[j]))
                names.append('bn%d%s_branch%s_gamma'%(stage, block_name[i-1], branch_name[j]))
                names.append('bn%d%s_branch%s_beta'%(stage, block_name[i-1], branch_name[j]))

        var = {}
        for name in names:
            var[name] = mx.sym.Variable(name=name)
        return var

    def get_resnet_v1_conv(self, data, stage, blocks, num_filter, var,stride=2,dilate=1,trident=False,trident_blocks=0):

        if blocks == 3:
            block_name = ['b' , 'c']
        else:
            block_name = ['b%d'%i for i in range(1, blocks)]
        branch_name = ['1', '2a', '2b', '2c']

        conv1_shortcut = mx.sym.Convolution(name="res%da_branch1"%stage, data=data, num_filter=num_filter, pad=(0, 0), kernel=(1, 1), stride=(stride, stride), no_bias=True,weight=var["res%da_branch1_weight"%stage])
        bn1_shortcut = mx.sym.BatchNorm(name="bn%da_branch1"%stage,data=conv1_shortcut,use_global_stats=True,fix_gamma=False,eps=self.eps,moving_mean=var["bn%da_branch1_moving_mean"%stage],moving_var=var["bn%da_branch1_moving_var"%stage],gamma=var["bn%da_branch1_gamma"%stage],beta=var["bn%da_branch1_beta"%stage])

        conv2a = mx.sym.Convolution(name="res%da_branch2a"%stage, data=data, num_filter=num_filter/4, pad=(0, 0), kernel=(1, 1), stride=(stride, stride), no_bias=True,weight=var["res%da_branch2a_weight"%stage])
        bn2a = mx.sym.BatchNorm(name="bn%da_branch2a"%stage,data=conv2a,use_global_stats=True,fix_gamma=False,eps=self.eps,moving_mean=var["bn%da_branch2a_moving_mean"%stage],moving_var=var["bn%da_branch2a_moving_var"%stage],gamma=var["bn%da_branch2a_gamma"%stage],beta=var["bn%da_branch2a_beta"%stage])
        relu2a = mx.sym.Activation(name="res%da_branch2a_relu"%stage, data=bn2a, act_type='relu')
        conv2b = mx.sym.Convolution(name="res%da_branch2b"%stage, data=relu2a, num_filter=num_filter/4, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True,weight=var["res%da_branch2b_weight"%stage])
        bn2b = mx.sym.BatchNorm(name="bn%da_branch2b"%stage,data=conv2b,use_global_stats=True,fix_gamma=False,eps=self.eps,moving_mean=var["bn%da_branch2b_moving_mean"%stage],moving_var=var["bn%da_branch2b_moving_var"%stage],gamma=var["bn%da_branch2b_gamma"%stage],beta=var["bn%da_branch2b_beta"%stage])
        relu2b = mx.sym.Activation(name="res%da_branch2b_relu"%stage, data=bn2b, act_type='relu')
        conv2c = mx.sym.Convolution(name="res%da_branch2c"%stage, data=relu2b, num_filter=num_filter, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,weight=var["res%da_branch2c_weight"%stage])
        bn2c = mx.sym.BatchNorm(name="bn%da_branch2c"%stage,data=conv2c,use_global_stats=True,fix_gamma=False,eps=self.eps,moving_mean=var["bn%da_branch2c_moving_mean"%stage],moving_var=var["bn%da_branch2c_moving_var"%stage],gamma=var["bn%da_branch2c_gamma"%stage],beta=var["bn%da_branch2c_beta"%stage])
        x = mx.sym.broadcast_add(name="res%da"%stage, *[bn1_shortcut, bn2c])
        x = mx.sym.Activation(name="res%da_relu"%stage, data=x, act_type='relu')

        if not trident:
            for i in range(1, blocks):
                conv2a = mx.sym.Convolution(name="res%d%s_branch2a"%(stage, block_name[i-1]), data=x, num_filter=num_filter/4, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,weight=var["res%d%s_branch2a_weight"%(stage, block_name[i-1])])
                bn2a = mx.sym.BatchNorm(name="bn%d%s_branch2a"%(stage, block_name[i-1]),data=conv2a,use_global_stats=True,fix_gamma=False,eps=self.eps,moving_mean=var["bn%d%s_branch2a_moving_mean"%(stage, block_name[i-1])],moving_var=var["bn%d%s_branch2a_moving_var"%(stage, block_name[i-1])],gamma=var["bn%d%s_branch2a_gamma"%(stage, block_name[i-1])],beta=var["bn%d%s_branch2a_beta"%(stage, block_name[i-1])])
                relu2a = mx.sym.Activation(name="res%d%s_branch2a_relu"%(stage, block_name[i-1]), data=bn2a, act_type='relu')
                conv2b = mx.sym.Convolution(name="res%d%s_branch2b"%(stage, block_name[i-1]), data=relu2a, num_filter=num_filter/4, pad=(dilate, dilate), kernel=(3, 3), stride=(1, 1), dilate=(dilate, dilate), no_bias=True,weight=var["res%d%s_branch2b_weight"%(stage, block_name[i-1])])
                bn2b = mx.sym.BatchNorm(name="bn%d%s_branch2b"%(stage, block_name[i-1]),data=conv2b,use_global_stats=True,fix_gamma=False,eps=self.eps,moving_mean=var["bn%d%s_branch2b_moving_mean"%(stage, block_name[i-1])],moving_var=var["bn%d%s_branch2b_moving_var"%(stage, block_name[i-1])],gamma=var["bn%d%s_branch2b_gamma"%(stage, block_name[i-1])],beta=var["bn%d%s_branch2b_beta"%(stage, block_name[i-1])])
                relu2b = mx.sym.Activation(name="res%d%s_branch2b_relu"%(stage, block_name[i-1]), data=bn2b, act_type='relu')
                conv2c = mx.sym.Convolution(name="res%d%s_branch2c"%(stage, block_name[i-1]), data=relu2b, num_filter=num_filter, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,weight=var["res%d%s_branch2c_weight"%(stage, block_name[i-1])])
                bn2c = mx.sym.BatchNorm(name="bn%d%s_branch2c"%(stage, block_name[i-1]),data=conv2c,use_global_stats=True,fix_gamma=False,eps=self.eps,moving_mean=var["bn%d%s_branch2c_moving_mean"%(stage, block_name[i-1])],moving_var=var["bn%d%s_branch2c_moving_var"%(stage, block_name[i-1])],gamma=var["bn%d%s_branch2c_gamma"%(stage, block_name[i-1])],beta=var["bn%d%s_branch2c_beta"%(stage, block_name[i-1])])
                x = mx.sym.broadcast_add(name="res%d%s"%(stage, block_name[i-1]), *[x, bn2c])
                x = mx.sym.Activation(name="res%d%s_relu"%(stage, block_name[i-1]), data=x, act_type='relu')
            return x

        else:
            for i in range(1, blocks - trident_blocks + 1):
                conv2a = mx.sym.Convolution(name="res%d%s_branch2a"%(stage, block_name[i-1]), data=x, num_filter=num_filter/4, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,weight=var["res%d%s_branch2a_weight"%(stage, block_name[i-1])])
                bn2a = mx.sym.BatchNorm(name="bn%d%s_branch2a"%(stage, block_name[i-1]),data=conv2a,use_global_stats=True,fix_gamma=False,eps=self.eps,moving_mean=var["bn%d%s_branch2a_moving_mean"%(stage, block_name[i-1])],moving_var=var["bn%d%s_branch2a_moving_var"%(stage, block_name[i-1])],gamma=var["bn%d%s_branch2a_gamma"%(stage, block_name[i-1])],beta=var["bn%d%s_branch2a_beta"%(stage, block_name[i-1])])
                relu2a = mx.sym.Activation(name="res%d%s_branch2a_relu"%(stage, block_name[i-1]), data=bn2a, act_type='relu')
                conv2b = mx.sym.Convolution(name="res%d%s_branch2b"%(stage, block_name[i-1]), data=relu2a, num_filter=num_filter/4, pad=(dilate, dilate), kernel=(3, 3), stride=(1, 1), dilate=(dilate, dilate), no_bias=True,weight=var["res%d%s_branch2b_weight"%(stage, block_name[i-1])])
                bn2b = mx.sym.BatchNorm(name="bn%d%s_branch2b"%(stage, block_name[i-1]),data=conv2b,use_global_stats=True,fix_gamma=False,eps=self.eps,moving_mean=var["bn%d%s_branch2b_moving_mean"%(stage, block_name[i-1])],moving_var=var["bn%d%s_branch2b_moving_var"%(stage, block_name[i-1])],gamma=var["bn%d%s_branch2b_gamma"%(stage, block_name[i-1])],beta=var["bn%d%s_branch2b_beta"%(stage, block_name[i-1])])
                relu2b = mx.sym.Activation(name="res%d%s_branch2b_relu"%(stage, block_name[i-1]), data=bn2b, act_type='relu')
                conv2c = mx.sym.Convolution(name="res%d%s_branch2c"%(stage, block_name[i-1]), data=relu2b, num_filter=num_filter, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,weight=var["res%d%s_branch2c_weight"%(stage, block_name[i-1])])
                bn2c = mx.sym.BatchNorm(name="bn%d%s_branch2c"%(stage, block_name[i-1]),data=conv2c,use_global_stats=True,fix_gamma=False,eps=self.eps,moving_mean=var["bn%d%s_branch2c_moving_mean"%(stage, block_name[i-1])],moving_var=var["bn%d%s_branch2c_moving_var"%(stage, block_name[i-1])],gamma=var["bn%d%s_branch2c_gamma"%(stage, block_name[i-1])],beta=var["bn%d%s_branch2c_beta"%(stage, block_name[i-1])])
                x = mx.sym.broadcast_add(name="res%d%s"%(stage, block_name[i-1]), *[x, bn2c])
                x = mx.sym.Activation(name="res%d%s_relu"%(stage, block_name[i-1]), data=x, act_type='relu')

            group_x = [x, x, x]
            for i in range(blocks - trident_blocks + 1, blocks):
                tmp_x = []
                for j in range(3):
                    conv2a = mx.sym.Convolution(name="trident%d_res%db%d_branch2a"%(j+1, stage, i), data=group_x[j], num_filter=num_filter/4, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,weight=var["res%db%d_branch2a_weight"%(stage, i)])
                    bn2a = mx.sym.BatchNorm(name="trident%d_bn%db%d_branch2a"%(j+1, stage, i),data=conv2a,use_global_stats=True,fix_gamma=False,eps=self.eps,moving_mean=var["bn%db%d_branch2a_moving_mean"%(stage, i)],moving_var=var["bn%db%d_branch2a_moving_var"%(stage, i)],gamma=var["bn%db%d_branch2a_gamma"%(stage, i)],beta=var["bn%db%d_branch2a_beta"%(stage, i)])
                    relu2a = mx.sym.Activation(name="trident%d_res%db%d_branch2a_relu"%(j+1, stage, i), data=bn2a, act_type='relu')
                    conv2b = mx.sym.Convolution(name="trident%d_res%db%d_branch2b"%(j+1, stage, i), data=relu2a, num_filter=num_filter/4, pad=(j+1, j+1), kernel=(3, 3), stride=(1, 1), dilate=(j+1, j+1), no_bias=True,weight=var["res%db%d_branch2b_weight"%(stage, i)])
                    bn2a = mx.sym.BatchNorm(name="trident%d_bn%db%d_branch2b"%(j+1, stage, i),data=conv2b,use_global_stats=True,fix_gamma=False,eps=self.eps,moving_mean=var["bn%db%d_branch2b_moving_mean"%(stage, i)],moving_var=var["bn%db%d_branch2b_moving_var"%(stage, i)],gamma=var["bn%db%d_branch2b_gamma"%(stage, i)],beta=var["bn%db%d_branch2b_beta"%(stage, i)])
                    relu2b = mx.sym.Activation(name="trident%d_res%db%d_branch2b_relu"%(j+1, stage, i), data=bn2b, act_type='relu')
                    conv2c = mx.sym.Convolution(name="trident%d_res%db%d_branch2c"%(j+1, stage, i), data=relu2b, num_filter=num_filter, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True,weight=var["res%db%d_branch2c_weight"%(stage, i)])
                    bn2c = mx.sym.BatchNorm(name="trident%d_bn%db%d_branch2c"%(j+1, stage, i),data=conv2c,use_global_stats=True,fix_gamma=False,eps=self.eps,moving_mean=var["bn%db%d_branch2c_moving_mean"%(stage, i)],moving_var=var["bn%db%d_branch2c_moving_var"%(stage, i)],gamma=var["bn%db%d_branch2c_gamma"%(stage, i)],beta=var["bn%db%d_branch2c_beta"%(stage, i)])
                    x = mx.sym.broadcast_add(name="trident%d_res%db%d"%(j+1, stage, i), *[group_x[j], bn2c])
                    x = mx.sym.Activation(name="trident%d_res%db%d_relu"%(j+1, stage, i), data=x, act_type='relu')
                    tmp_x.append(x)
                group_x = tmp_x
            group_x = mx.sym.Concat(*group_x, dim=0)

            return group_x

    def get_rpn(self, conv_feat, num_anchors):
        rpn_conv = mx.sym.Convolution(
            data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
        rpn_relu = mx.sym.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

        return rpn_cls_score, rpn_bbox_pred



    def get_symbol(self, cfg, is_train=True):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS

        # input init
        if is_train:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")
            gt_boxes = mx.sym.Variable(name="gt_boxes")
            rpn_label = mx.sym.Variable(name='label')
            rpn_bbox_target = mx.sym.Variable(name='bbox_target')
            rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')

            valid_ranges = mx.sym.Variable(name="valid_ranges")
            valid_ranges = mx.sym.Reshape(valid_ranges, (-1, 2))

            im_info = mx.sym.Concat(im_info, im_info, im_info, dim=0)
            gt_boxes = mx.sym.Concat(gt_boxes, gt_boxes, gt_boxes, dim=0)
        else:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")



        # shared convolutional layers
        conv1 = self.get_resnet_v1_conv1(data)

        conv2_var = self.get_resnet_v1_conv_var(2, self.units[0])
        conv3_var = self.get_resnet_v1_conv_var(3, self.units[1])
        conv4_var = self.get_resnet_v1_conv_var(4, self.units[2])
        conv5_var = self.get_resnet_v1_conv_var(5, self.units[3])
        

        conv2 = self.get_resnet_v1_conv(conv1, 2, self.units[0], self.filter_list[0], conv2_var, stride=1)
        conv3 = self.get_resnet_v1_conv(conv2, 3, self.units[1], self.filter_list[1], conv3_var)
        conv4 = self.get_resnet_v1_conv(conv3, 4, self.units[2], self.filter_list[2], conv4_var, trident=True, trident_blocks=10)
        
        conv5 = self.get_resnet_v1_conv(conv4, 5, self.units[3], self.filter_list[3], conv5_var, dilate=2, stride=1)

        rpn_cls_score, rpn_bbox_pred = self.get_rpn(conv4, num_anchors)
        
        if is_train:
            rpn_cls_score_reshape = mx.sym.Reshape(
                data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")


            rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                                   normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob")

            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))
            rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)


            # ROI proposal
            rpn_cls_act = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
            rpn_cls_act_reshape = mx.sym.Reshape(
                data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')


            proposal = mx.sym.contrib.Proposal_v2(
                cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE, valid_ranges=valid_ranges, iou_loss=False, filter_scales=True)

            rois, label, bbox_target, bbox_weight = mx.sym.ProposalTarget_v2(
                                                                   rois=proposal,
                                                                   gt_boxes=gt_boxes,
                                                                   valid_ranges=valid_ranges,
                                                                   num_classes=num_reg_classes,
                                                                   class_agnostic=cfg.CLASS_AGNOSTIC,
                                                                   batch_images=cfg.TRAIN.BATCH_IMAGES * 3,
                                                                   proposal_without_gt=True,
                                                                   image_rois=cfg.TRAIN.BATCH_ROIS,
                                                                   fg_fraction=cfg.TRAIN.FG_FRACTION,
                                                                   fg_thresh=cfg.TRAIN.FG_THRESH,
                                                                   bg_thresh_hi=cfg.TRAIN.BG_THRESH_HI,
                                                                   bg_thresh_lo=cfg.TRAIN.BG_THRESH_LO,
                                                                   bbox_weight=tuple(cfg.TRAIN.BBOX_WEIGHTS),
                                                                   bbox_mean=cfg.TRAIN.BBOX_MEANS,
                                                                   bbox_std=cfg.TRAIN.BBOX_STDS,
                                                                   filter_scales=True,
                                                                   name="proposal_target"
                                                                   )

            label = mx.sym.Reshape(label, (-3, -2))
            bbox_target = mx.sym.Reshape(bbox_target, (-3, -2))
            bbox_weight = mx.sym.Reshape(bbox_weight, (-3, -2))

        else:
            # ROI Proposal
            rpn_cls_score_reshape = mx.sym.Reshape(
                data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
            rpn_cls_prob = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
            rpn_cls_prob_reshape = mx.sym.Reshape(
                data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
            if cfg.TEST.CXX_PROPOSAL:
                rois = mx.contrib.sym.Proposal(
                    cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                    ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
            else:
                rois = mx.sym.Custom(
                    cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                    scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)

        conv_new_1 = mx.sym.Convolution(data=conv5, kernel=(1, 1), num_filter=256, name="conv_new_1")
        conv_new_1_relu = mx.sym.Activation(data=conv_new_1, act_type='relu', name='conv_new_1_relu')

        roi_pool = mx.sym.contrib.ROIAlign_v2(name="roi_pool", data=conv_new_1_relu, rois=rois, pooled_size=(7, 7), spatial_scale=0.0625)
        roi_pool = mx.sym.Reshape(roi_pool, (-3, -2))

        # 2 fc
        fc_new_1 = mx.symbol.FullyConnected(name='fc_new_1', data=roi_pool, num_hidden=1024)
        fc_new_1_relu = mx.sym.Activation(data=fc_new_1, act_type='relu', name='fc_new_1_relu')

        fc_new_2 = mx.symbol.FullyConnected(name='fc_new_2', data=fc_new_1_relu, num_hidden=1024)
        fc_new_2_relu = mx.sym.Activation(data=fc_new_2, act_type='relu', name='fc_new_2_relu')

        # cls_score/bbox_pred
        cls_score = mx.symbol.FullyConnected(name='cls_score', data=fc_new_2_relu, num_hidden=num_classes)
        bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=fc_new_2_relu, num_hidden=num_reg_classes * 4)

        if is_train:
            if cfg.TRAIN.ENABLE_OHEM:
                labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                               num_reg_classes=num_reg_classes,
                                                               roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                               cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
                                                               bbox_targets=bbox_target, bbox_weights=bbox_weight)
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem,
                                                normalization='valid', use_ignore=True, ignore_label=-1)
                bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0,
                                                                  data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_,
                                            grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS_OHEM)
                rcnn_label = labels_ohem
            else:
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid')
                bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0,
                                                            data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS)
                rcnn_label = label


            # reshape output
            rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_loss_reshape')
            group = mx.sym.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)])
        else:
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_pred_reshape')
            group = mx.sym.Group([rois, cls_prob, bbox_pred])

        self.sym = group
        return group


    def init_weight_rcnn(self, cfg, arg_params, aux_params):
        arg_params['conv_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_new_1_weight'])
        arg_params['conv_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_new_1_bias'])
        arg_params['fc_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_1_weight'])
        arg_params['fc_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_1_bias'])
        arg_params['fc_new_2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_2_weight'])
        arg_params['fc_new_2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_2_bias'])
        arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['cls_score_weight'])
        arg_params['cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['cls_score_bias'])
        arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bbox_pred_weight'])
        arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['bbox_pred_bias'])

    def init_weight_rpn(self, cfg, arg_params, aux_params):
        arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_conv_3x3_weight'])
        arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_3x3_bias'])
        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01,
                                                              shape=self.arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01,
                                                              shape=self.arg_shape_dict['rpn_bbox_pred_weight'])
        arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred_bias'])

    def init_weight(self, cfg, arg_params, aux_params):
        self.init_weight_rpn(cfg, arg_params, aux_params)
        self.init_weight_rcnn(cfg, arg_params, aux_params)

