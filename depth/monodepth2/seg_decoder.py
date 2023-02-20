# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from depth.monodepth2.layers import *
import seg.layers as seg_layers

from utils.utils import *


class DepthDecoder(nn.Module):
    def __init__(self, opt, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.opt = opt
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.convs = OrderedDict()
        self.seg_convs = OrderedDict()

        # Segmentation head
        # if self.opt.seg_model == self.opt.depth_model:
            # self.seg_convs = OrderedDict()
        self.seg_convs["seman_dispconv"] = nn.Sequential(
            seg_layers.DoubleConv(np.sum(self.num_ch_dec), 128),
            seg_layers.OutConv(128, self.opt.num_classes)
        )
        # self.seman_dispconv = nn.Sequential(
        #     seg_layers.DoubleConv(np.sum(self.num_ch_dec), 128),
        #     seg_layers.OutConv(128, self.opt.num_classes)
        # )

        # decoder
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        # for s in self.scales:
        #     # if self.opt.bins_regression:
        #     #     self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.opt.num_bins)
        #     #     # bin_width = (1./self.opt.min_depth - 1./self.opt.max_depth) / self.opt.num_bins
        #     #     # self.bins = bin_width * (torch.arange(self.opt.num_bins) + 0.5) + 1./self.opt.max_depth
        #     #     bin_width = (self.opt.max_depth - self.opt.min_depth) / self.opt.num_bins
        #     #     self.bins = bin_width * (torch.arange(self.opt.num_bins) + 0.5) + self.opt.min_depth
        #     #     self.bins = self.bins[None, :, None, None].cuda()
        #     # else:
        #     self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.sep_decoder = nn.ModuleList(list(self.convs.values()))
        # if self.opt.seg_model == self.opt.depth_model:
        self.seg_sep_decoder = nn.ModuleList(list(self.seg_convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, rst, data, nets):
        input_features = rst["features"]
        self.outputs = {}

        # decoder
        x = input_features[-1]
        if self.opt.seg_model == self.opt.depth_model:
            features = []
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            
            if self.opt.seg_model == self.opt.depth_model:
                features += [x]
            
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            if (self.opt.feature_loss or self.opt.optical_flow_model == "raft_github") and i == 0:
                self.outputs["seg_feature"] = x
            
            # if i in self.scales:
            #     # if self.opt.bins_regression:
            #     #     logit = F.softmax(self.convs[("dispconv", i)](x), dim=1)
            #     #     self.outputs[("depth", i)] = (logit * self.bins).sum(1, keepdims=True)
            #     # else:
            #     self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        # if self.opt.seg_model == self.opt.depth_model:
        features = torch.cat([torch_resize(feature, features[-1].size()[2:]) \
            for feature in features], dim=1)
        self.outputs["seman"] = self.seg_convs["seman_dispconv"](features)
        # self.outputs["seman"] = self.seman_dispconv(features)

        return self.outputs

    # def gcm_forward(self, rst, layer):
    #     input_features = rst['enc']

    #     if not hasattr(self, 'gcm_id'):
    #         self.gcm_id = 4

    #     # decoder
    #     for i in range(self.gcm_id, -1, -1):
    #         if ("disp_fea", i) in rst:
    #             assert self.gcm_id == i

    #             x = rst[("disp_fea", i)]
    #             self.x = x
    #             self.gcm_id = i-1
            
    #         else:
    #             if not hasattr(self, 'x'):
    #                 self.x = input_features[-1]

    #             x = self.convs[("upconv", i, 0)](self.x)
    #             x = [upsample(x)]
    #             if self.use_skips and i > 0:
    #                 x += [input_features[i - 1]]
    #             x = torch.cat(x, 1)
    #             x = self.convs[("upconv", i, 1)](x)
    #             self.x = x

    #             if i == layer:
    #                 rst[("disp_fea", i)] = x
    #                 self.gcm_id = i
    #                 break

    #         if i in self.scales:
    #             rst[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
            
    #         if i == layer:
    #             break

    #     if layer == -1: # Reset.
    #         del self.gcm_id
    #         del self.x

    #     return rst

    def gcm_forward(self, rst, inputs, nets, layers, key, get_graph):
        """
        graph_data: BxHxW or Bx1xHxW
        """
        input_features = rst["features"]
        self.outputs = {}

        # decoder
        graph_losses = []
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            
            if i in layers:
                x, seman, graph_loss, graph_coords = get_graph.refine_with_graph(x, 
                    ('seg-monodepth2s', 'l'+str(i), inputs, nets['graph'].models))
                # self.outputs[("seman", i)] = seman
                self.outputs[("seman", 0)] = seman
                self.outputs[("seman", 1)] = seman
                self.outputs[("seman", 2)] = seman
                self.outputs[("seman", 3)] = seman
                graph_losses.append(graph_loss)

            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        self.outputs["graph_loss"] = torch.stack(graph_losses).mean()

        return self.outputs, graph_coords