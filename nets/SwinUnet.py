# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging

import torch
import torch.nn as nn

from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    def __init__(self, img_size=512, num_classes=21843, zero_head=False, vis=False, pretrained=False, backbone=None):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.backbone = backbone
        self.swin_unet = SwinTransformerSys(img_size=img_size,
                                            patch_size=4,
                                            in_chans=3,
                                            num_classes=self.num_classes,
                                            embed_dim=96,
                                            depths=[2, 2, 6, 2],
                                            num_heads=[3, 6, 12, 24],
                                            window_size=8,
                                            mlp_ratio=4.,
                                            qkv_bias=True,
                                            qk_scale=None,
                                            drop_rate=0.,
                                            drop_path_rate=0.1,
                                            ape=False,
                                            patch_norm=True,
                                            use_checkpoint=False)

    def forward(self, x):
        logits = self.swin_unet(x)
        return logits

    def load_from(self, pretrained_path):
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" in pretrained_dict:
                pretrained_dict = pretrained_dict['model']
            model_dict = self.swin_unet.state_dict()
            full_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            for k, v in pretrained_dict.items():
                if k in model_dict:
                    if v.shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]
                    else:
                        continue
                else:
                    del full_dict[k]

            with open('full_dict.txt', 'w') as f:
                for k, v in sorted(full_dict.items()):
                    f.write(k + '\n')
            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            print(msg)
        else:
            print("none pretrain")