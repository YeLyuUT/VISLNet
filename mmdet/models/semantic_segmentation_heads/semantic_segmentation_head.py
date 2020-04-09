import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import build_loss

from ..registry import HEADS

@HEADS.register_module
class Semantic_Segmentation_Head(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 in_scales=[1. / 4, 1. / 8, 1. / 16, 1. / 32],
                 resize_to_level=0,
                 feat_channels=256,
                 use_refine=True,
                 use_softmax=False,
                 loss_cls=dict(
                     type='NLLLoss',
                     loss_weight=None)):
        assert len(in_channels)==len(in_scales), 'Number of input channels should be equal to number of input scales,\
                                                 but here are %d and %d'%(len(in_channels), len(in_scales))
        #TODO check below.
        self.loss_cls = build_loss(loss_cls)
        self.resize_to_level = resize_to_level
        self.in_channels = in_channels
        self.in_scales = in_scales
        self.use_refine = use_refine
        self.use_softmax = use_softmax

        # Add convolution for refinement if use_refine is True.
        if self.use_refine:
            self.refine = nn.ConvModule(
                sum(in_channels),
                feat_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
            n_feat_channels = feat_channels * len(in_channels)
        else:
            n_feat_channels = sum(in_channels)
        # output semantic segmentation head.
        self.pred_head = nn.Conv2D(n_feat_channels,num_classes,1,1,0)
        # semantic segmentation loss.
        assert loss_cls['type'] in ['NLLLoss'], 'Unknown loss type %s'%(loss_cls['type'])
        self.nllloss = nn.NLLLoss(weight=loss_cls['loss_weight'])


    def init_weights(self):
        pass

    def _resize_merge(self, feats):
        gather_size = feats[self.refine_level].size()[2:]
        resized_feats = []
        for i in range(len(feats)):
            gathered = F.interpolate(feats[i], size=gather_size, mode='bilinear')
            resized_feats.append(gathered)
        out_feats = torch.cat(resized_feats, dim=1)
        return out_feats

    def forward(self, feats):
        feats = self._resize_merge(feats)
        if self.use_refine:
            feats = self.refine(feats)
        feats = self.pred_head(feats)
        if self.use_softmax: #is True during inference.
            feats = F.softmax(feats, dim=1)
        else:
            feats = F.logsoftmax(feats, dim=1)


    def loss(self,
             preds,
             gt_labels):
        loss = self.nllloss(preds, gt_labels)
        acc = self.pixel_acc(preds, gt_labels)
        return loss, acc

    def pixel_acc(self, preds, gt_label):
        _, preds = torch.max(preds, dim=1)
        valid = (gt_label>=0).long()
        acc_sum = torch.sum(valid*(preds==gt_label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float()/(pixel_sum.float()+1e-10)
        return acc